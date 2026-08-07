"""
Moteur de backtest événementiel, au jour le jour.

Règle d'exécution (uniforme, pas de look-ahead) : toute décision (nouveau
signal DCF publié, stop-loss/take-profit déclenché) est prise sur la base de
la clôture du jour J, et exécutée à l'OUVERTURE du jour de bourse suivant
J+1. Seule exception : un symbole dont les données de prix s'arrêtent
totalement (radiation non couverte par 03b/Stooq) est clôturé immédiatement
au dernier cours connu, faute d'une ouverture future à laquelle exécuter
l'ordre (voir _handle_stale_symbols).

Positions "gelées" : un symbole actuellement en portefeuille mais qui ne
fait plus partie du panier éligible de la stratégie (son écart est repassé
sous le seuil, ou il est sorti du S&P 500) N'EST PAS vendu -- conformément
au choix explicite de l'utilisateur (sortie uniquement par stop-loss/
take-profit). Il reste en portefeuille, à taille inchangée, jusqu'à ce
qu'un de ces deux déclencheurs le ferme. Le capital alloué aux nouvelles
positions/positions actives est donc le NAV diminué de la valeur des
positions gelées (voir _rebalance).

Stop-loss/take-profit : mesurés depuis l'ouverture de la THÈSE
(Position.stop_reference_price, figée à la première entrée), pas depuis le
prix de revient courant -- lequel continue d'être moyenné à chaque renfort
pour le P&L. Un renfort ne déplace donc jamais le seuil.

Coûts de transaction : commission + slippage fusionnés en un seul cost_bps
appliqué symétriquement à l'achat et à la vente (prix d'exécution effectif
= prix marché x (1 ± cost_bps/10000)), pour que le P&L par trade reflète le
coût réel d'un aller-retour sans bookkeeping séparé.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import config
from backtest import data_loader
from backtest.strategies.base import Strategy

logger = logging.getLogger("backtest.engine")

MIN_TRADE_DOLLAR = 1.0  # en dessous, un ordre de rebalancement est ignoré (évite le "churn" sur des écarts négligeables)

# Au-delà de cette part d'ordres d'achat rognés faute de cash, le
# sous-investissement n'est plus un accident isolé mais le régime normal du
# run : il est signalé en fin de backtest (cf. execution_diagnostics).
TRUNCATED_ORDERS_WARNING_PCT = 10.0


@dataclass
class Position:
    symbol: str
    shares: float
    entry_price: float  # prix d'exécution effectif, coût de transaction d'entrée déjà inclus
    entry_date: pd.Timestamp

    # Référence du stop-loss/take-profit, figée à la PREMIÈRE ouverture et
    # jamais recalculée -- à distinguer de entry_price, prix de revient
    # comptable qui continue d'être moyenné à chaque renfort (c'est ce
    # qu'attend le P&L des trades).
    #
    # Sans cette distinction, renforcer une position en baisse abaissait le
    # prix de revient, donc le seuil du stop avec lui : le stop ne se
    # déclenchait pratiquement jamais tant qu'on moyennait à la baisse,
    # exactement la situation où il devrait protéger. Même sémantique que
    # OptionPosition.stop_reference_premium côté options : le stop mesure la
    # perte depuis l'ouverture de la THÈSE.
    stop_reference_price: float = 0.0


@dataclass
class _PendingOrder:
    target_dollar: float
    reason: str
    queued_on: pd.Timestamp


class BacktestEngine:
    def __init__(
        self,
        price_panel: data_loader.PricePanel,
        signal_events: pd.DataFrame,
        universe_history: Optional[pd.DataFrame],
        fallback_universe_symbols: set[str],
        strategy: Strategy,
        initial_capital: float,
        cost_bps: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        max_positions: int,
        signal_max_age_days: int = config.BACKTEST_SIGNAL_MAX_AGE_DAYS,
        momentum_min_pct: Optional[float] = config.BACKTEST_MOMENTUM_MIN_PCT,
        material_events_8k: Optional[pd.DataFrame] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ):
        self.prices = price_panel
        self.last_valid_date = price_panel.last_valid_date
        # Les événements sont regroupés par date de publication une fois pour
        # toutes : les rechercher par masque booléen sur la table complète à
        # chacun des ~2500 jours de bourse d'un run coûtait plus cher que le
        # reste de la journée simulée.
        self.events_by_date = {
            published_date: group.to_dict("records")
            for published_date, group in signal_events.groupby("published_date", sort=False)
        }
        self.universe = data_loader.UniverseResolver(universe_history, fallback_universe_symbols)
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.cost_bps = cost_bps
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions
        self.signal_max_age_days = signal_max_age_days
        self.momentum_min_pct = momentum_min_pct
        self.material_events = data_loader.MaterialEventResolver(material_events_8k)

        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.known_signals: dict[str, dict] = {}
        self.pending_orders: dict[str, _PendingOrder] = {}

        self.trades: list[dict] = []
        self.equity_curve_rows: list[dict] = []
        self.positions_history_rows: list[dict] = []
        self.signals_history_rows: list[dict] = []

        # Diagnostics d'exécution (voir execution_diagnostics). _rebalance
        # alloue un budget égal au NAV diminué des positions GELÉES, mais ces
        # positions ne sont pas vendues : le cash réellement disponible est
        # souvent inférieur au budget alloué, et _execute_trade tronquait
        # l'achat en silence. Combiné à la règle "on ne sort jamais sur perte
        # de signal", le portefeuille dérive alors vers un buy-and-hold de
        # positions périmées sans que rien ne le signale.
        self.buy_orders_count = 0
        self.truncated_orders_count = 0

        calendar = price_panel.close.index
        if start_date is not None:
            calendar = calendar[calendar >= start_date]
        if end_date is not None:
            calendar = calendar[calendar <= end_date]
        if len(calendar) == 0:
            raise ValueError("Aucun jour de bourse dans la plage demandée : vérifie start_date/end_date et les données de prix.")
        self.calendar = calendar

        if universe_history is None:
            logger.warning(
                "Pas d'historique d'univers (lance 01b_historique_univers_sp500.py) : "
                "l'univers ACTUEL du S&P 500 est appliqué à toutes les dates passées -- "
                "biais de survivance connu, résultats optimistes."
            )

    # ------------------------------------------------------------------ #
    # Boucle principale
    # ------------------------------------------------------------------ #
    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        for today in self.calendar:
            self._execute_pending_orders(today)
            exited_today = self._handle_stale_symbols(today)
            exited_today |= self._check_stop_loss_take_profit(today)

            todays_events = self.events_by_date.get(today)
            if todays_events:
                self._update_known_signals(todays_events, today)
                self._rebalance(today, exclude=exited_today)

            self._mark_to_market(today)
            self._record_positions(today)

        equity_curve = pd.DataFrame(self.equity_curve_rows)
        positions_history = pd.DataFrame(self.positions_history_rows)
        trades = pd.DataFrame(self.trades)
        signals_history = pd.DataFrame(self.signals_history_rows)
        return equity_curve, positions_history, trades, signals_history

    # ------------------------------------------------------------------ #
    # Exécution des ordres décidés la veille (clôture J-1 -> ouverture J)
    # ------------------------------------------------------------------ #
    def _execute_pending_orders(self, today: pd.Timestamp) -> None:
        if not self.pending_orders:
            return

        still_pending: dict[str, _PendingOrder] = {}
        # Ventes d'abord, achats ensuite : le produit d'une sortie finance les
        # achats du même jour (cf. _rebalance, qui raisonne en NAV). Sans cet
        # ordre, la garde de cash de _execute_trade tronquerait des achats
        # pourtant couverts, selon le seul ordre d'insertion du dictionnaire.
        ordered = sorted(self.pending_orders.items(), key=lambda kv: kv[1].target_dollar > 0)
        for symbol, order in ordered:
            price = self.prices.open_at(symbol, today)
            if price is None:
                if (today - order.queued_on).days > data_loader.FORWARD_FILL_MAX_DAYS:
                    logger.warning(
                        "Ordre en attente pour %s abandonné (%s) : aucune ouverture "
                        "disponible depuis plus de %d jours -- trou de couverture.",
                        symbol, order.reason, data_loader.FORWARD_FILL_MAX_DAYS,
                    )
                    continue
                still_pending[symbol] = order
                continue
            self._fill_order(symbol, order.target_dollar, price, today, order.reason)
        self.pending_orders = still_pending

    def _fill_order(self, symbol: str, target_dollar: float, price: float, today: pd.Timestamp, reason: str) -> None:
        pos = self.positions.get(symbol)
        current_shares = pos.shares if pos else 0.0
        current_value = current_shares * price
        delta_dollar = target_dollar - current_value
        if abs(delta_dollar) < MIN_TRADE_DOLLAR:
            return
        shares_delta = delta_dollar / price
        self._execute_trade(symbol, shares_delta, price, today, reason)

    def _execute_trade(self, symbol: str, shares_delta: float, price: float, today: pd.Timestamp, reason: str) -> None:
        cost_rate = self.cost_bps / 10_000
        pos = self.positions.get(symbol)

        if shares_delta > 0:  # achat (nouvelle position ou renforcement)
            effective_price = price * (1 + cost_rate)
            # Backtest NON margé : on n'achète jamais à crédit. Le budget vient
            # du NAV (_rebalance), dont une partie est immobilisée dans les
            # positions conservées -- sans cette borne, le cash passait
            # négatif et l'exposition dépassait 100% du portefeuille.
            cost = shares_delta * effective_price
            self.buy_orders_count += 1
            if cost > self.cash:
                # Troncature COMPTÉE, plus silencieuse : c'est le symptôme
                # observable du sous-investissement décrit dans __init__.
                self.truncated_orders_count += 1
                if self.cash <= 0:
                    return
                shares_delta, cost = shares_delta * (self.cash / cost), self.cash
                if cost < MIN_TRADE_DOLLAR:
                    return
            self.cash -= cost
            if pos is None:
                self.positions[symbol] = Position(
                    symbol, shares_delta, effective_price, today,
                    stop_reference_price=effective_price,  # posée ici et nulle part ailleurs
                )
            else:
                new_shares = pos.shares + shares_delta
                # Prix de revient moyenné (P&L), référence de stop INTACTE
                # (cf. Position.stop_reference_price).
                pos.entry_price = (pos.entry_price * pos.shares + effective_price * shares_delta) / new_shares
                pos.shares = new_shares
            return

        # vente (partielle ou totale)
        if pos is None or pos.shares <= 0:
            return
        sold_shares = min(-shares_delta, pos.shares)
        effective_price = price * (1 - cost_rate)
        proceeds = sold_shares * effective_price
        self.cash += proceeds
        pnl = (effective_price - pos.entry_price) * sold_shares
        self.trades.append({
            "symbol": symbol, "entry_date": pos.entry_date, "exit_date": today,
            "shares": sold_shares, "entry_price": pos.entry_price, "exit_price": effective_price,
            "pnl": pnl, "return_pct": (effective_price - pos.entry_price) / pos.entry_price * 100,
            "holding_days": (today - pos.entry_date).days, "exit_reason": reason,
        })
        pos.shares -= sold_shares
        if pos.shares <= 1e-9:
            del self.positions[symbol]

    # ------------------------------------------------------------------ #
    # Gestion du risque : stop-loss / take-profit, données manquantes
    # ------------------------------------------------------------------ #
    def _check_stop_loss_take_profit(self, today: pd.Timestamp) -> set[str]:
        if not self.positions:
            return set()
        triggered = set()
        for symbol, pos in list(self.positions.items()):
            price = self.prices.close_at(symbol, today)
            reference = pos.stop_reference_price or pos.entry_price
            if price is None or not reference:
                continue
            move_pct = (price - reference) / reference * 100
            if move_pct <= self.stop_loss_pct:
                self._queue_order(symbol, 0.0, "stop_loss", today)
                triggered.add(symbol)
            elif move_pct >= self.take_profit_pct:
                self._queue_order(symbol, 0.0, "take_profit", today)
                triggered.add(symbol)
        return triggered

    def _handle_stale_symbols(self, today: pd.Timestamp) -> set[str]:
        """Ferme IMMÉDIATEMENT (au dernier cours connu, pas via
        pending_orders) toute position dont le symbole n'a plus de cours
        réel depuis plus de FORWARD_FILL_MAX_DAYS : au-delà de ce point, le
        forward-fill du price_panel s'arrête (cf. data_loader.build_price_panel),
        et attendre une future ouverture n'a pas de sens pour un titre qui a
        cessé d'être coté."""
        if not self.positions:
            return set()
        closed = set()
        for symbol, pos in list(self.positions.items()):
            last_valid = self.last_valid_date.get(symbol)
            if last_valid is None or pd.isna(last_valid):
                continue
            if (today - last_valid).days <= data_loader.FORWARD_FILL_MAX_DAYS:
                continue
            last_price = self.prices.close_at(symbol, last_valid)
            if last_price is None:
                continue
            logger.warning(
                "%s : plus aucun cours depuis %s (>%d jours), position clôturée au "
                "dernier cours connu (%.2f) -- probable radiation non couverte par 03b.",
                symbol, last_valid.date(), data_loader.FORWARD_FILL_MAX_DAYS, last_price,
            )
            self._execute_trade(symbol, -pos.shares, last_price, today, "data_gap")
            closed.add(symbol)
        return closed

    def _queue_order(self, symbol: str, target_dollar: float, reason: str, today: pd.Timestamp) -> None:
        self.pending_orders[symbol] = _PendingOrder(target_dollar, reason, today)

    # ------------------------------------------------------------------ #
    def execution_diagnostics(self) -> dict:
        """Écart entre ce que le rebalancement a DEMANDÉ et ce que le cash a
        permis d'exécuter, à fusionner dans metrics.json.

        La règle des positions gelées est un choix utilisateur assumé et n'est
        pas remise en cause ici -- mais elle a une conséquence qui, elle, doit
        être visible : _rebalance alloue un budget calculé sur le NAV
        (positions gelées déduites) alors que ces positions restent détenues,
        si bien que le cash disponible est souvent inférieur au budget. Les
        achats étaient alors rognés sans que rien ne l'indique, et le
        portefeuille glissait vers un buy-and-hold de lignes périmées.

        avg_cash_pct mesure l'autre face du même phénomène : la part du
        portefeuille restée en cash faute d'avoir pu exécuter les ordres."""
        truncated_pct = (
            self.truncated_orders_count / self.buy_orders_count * 100
            if self.buy_orders_count else 0.0
        )
        cash_pct = [
            row["cash"] / row["nav"] * 100
            for row in self.equity_curve_rows if row["nav"] > 0
        ]
        avg_cash_pct = sum(cash_pct) / len(cash_pct) if cash_pct else None

        if truncated_pct > TRUNCATED_ORDERS_WARNING_PCT:
            logger.warning(
                "%.1f%% des ordres d'achat (%d/%d) ont été tronqués faute de cash : le budget "
                "alloué par le rebalancement dépasse régulièrement le cash disponible, parce que "
                "les positions gelées immobilisent du capital sans être vendues. Le portefeuille "
                "est donc moins investi que la stratégie ne le demande (cash moyen %.1f%%). "
                "Réduire --max-positions, ou accepter ce sous-investissement comme faisant "
                "partie de la règle des positions gelées.",
                truncated_pct, self.truncated_orders_count, self.buy_orders_count,
                avg_cash_pct if avg_cash_pct is not None else float("nan"),
            )

        return {
            "buy_orders_count": self.buy_orders_count,
            "truncated_orders_count": self.truncated_orders_count,
            "truncated_orders_pct": float(truncated_pct),
            "avg_cash_pct": float(avg_cash_pct) if avg_cash_pct is not None else None,
        }

    # ------------------------------------------------------------------ #
    # Signaux et rebalancement
    # ------------------------------------------------------------------ #
    def _update_known_signals(self, todays_events: list[dict], today: pd.Timestamp) -> None:
        for row in todays_events:
            self.known_signals[row["symbol"]] = row
            self.signals_history_rows.append({
                "date": today, "symbol": row["symbol"], "sector": row.get("sector"),
                "fiscal_year": row.get("fiscal_year"), "gap_pct": row.get("gap_pct"),
                "valuation_dcf_per_share": row.get("valuation_dcf_per_share"),
            })

    def _momentum_ok(self, symbol: str, today: pd.Timestamp) -> bool:
        """Filtre "value trap" : écarte une NOUVELLE entrée sur un titre en
        tendance nettement baissière (cf. config.BACKTEST_MOMENTUM_MIN_PCT).
        Un titre dont l'historique est trop court pour mesurer le momentum
        n'est PAS écarté : l'absence de mesure n'est pas un signal négatif."""
        if self.momentum_min_pct is None:
            return True
        momentum = self.prices.momentum_12_1(symbol, today)
        return momentum is None or momentum * 100 >= self.momentum_min_pct

    def _rebalance(self, today: pd.Timestamp, exclude: set[str]) -> None:
        universe_today = self.universe.asof(today)
        # Un signal périmé (pas de 10-K plus récent que signal_max_age_days)
        # n'est éligible à une NOUVELLE entrée que s'il concerne une position
        # déjà détenue (dans ce cas elle est de toute façon gelée, pas
        # rebalancée sur la base de ce vieux signal -- cf. docstring module) :
        # seules les positions PAS encore ouvertes sont réellement filtrées ici.
        eligible_signals = pd.DataFrame([
            s for sym, s in self.known_signals.items()
            if sym in universe_today
            and (
                sym in self.positions
                or (
                    (today - s["published_date"]).days
                    <= data_loader.signal_max_age_for(s, self.signal_max_age_days)
                    and self._momentum_ok(sym, today)
                    # Un 8-K matériel déposé depuis la publication du signal
                    # (04c) rend celui-ci périmé : les fondamentaux sur
                    # lesquels il repose ont bougé depuis.
                    and not self.material_events.has_event_between(sym, s["published_date"], today)
                )
            )
        ])
        if eligible_signals.empty:
            return

        target_weights = self.strategy.generate_target_weights(eligible_signals, set(self.positions))
        target_weights = {s: w for s, w in target_weights.items() if s not in exclude and w > 0}
        if not target_weights:
            return

        nav_now = self._current_nav(today)

        legacy_value = sum(
            pos.shares * self._mark_price(pos, today)
            for sym, pos in self.positions.items()
            if sym not in target_weights and sym not in exclude
        )
        active_budget = max(nav_now - legacy_value, 0.0)

        already_held = {s: w for s, w in target_weights.items() if s in self.positions}
        new_candidates = {s: w for s, w in target_weights.items() if s not in self.positions}

        slots_used = sum(
            1 for sym in self.positions
            if sym not in target_weights and sym not in exclude
        ) + len(already_held)
        slots_for_new = max(self.max_positions - slots_used, 0)
        if len(new_candidates) > slots_for_new:
            new_candidates = dict(
                sorted(new_candidates.items(), key=lambda kv: kv[1], reverse=True)[:slots_for_new]
            )

        final_active = {**already_held, **new_candidates}
        total_weight = sum(final_active.values())
        # Toujours renormalisé à somme=1 (pas seulement si >1) : le plafond
        # de positions/slots gelés peut retirer des candidats APRÈS que la
        # stratégie a déjà normalisé ses poids à 100% (cf. ValuationGapDCFStrategy) ;
        # sans renormalisation ici, la part du candidat écarté resterait en
        # cash au lieu d'être réallouée aux survivants -- pas une décision
        # délibérée de la stratégie, juste un artefact de la troncature.
        if total_weight > 0:
            final_active = {s: w / total_weight for s, w in final_active.items()}

        for symbol, weight in final_active.items():
            self._queue_order(symbol, weight * active_budget, "rebalance", today)

    # ------------------------------------------------------------------ #
    # Comptabilité quotidienne
    # ------------------------------------------------------------------ #
    def _mark_price(self, pos: Position, today: pd.Timestamp) -> float:
        """Cours de valorisation d'une position : la clôture du jour, ou à
        défaut son prix d'entrée (le symbole n'a pas encore/plus de cours
        exploitable ce jour-là -- une position dans ce cas est de toute façon
        en cours de fermeture par _handle_stale_symbols)."""
        price = self.prices.close_at(pos.symbol, today)
        return pos.entry_price if price is None else price

    def _current_nav(self, today: pd.Timestamp) -> float:
        return self.cash + sum(pos.shares * self._mark_price(pos, today) for pos in self.positions.values())

    def _mark_to_market(self, today: pd.Timestamp) -> None:
        invested = sum(pos.shares * self._mark_price(pos, today) for pos in self.positions.values())
        nav = self.cash + invested
        self.equity_curve_rows.append({
            "date": today, "nav": nav, "cash": self.cash, "invested_value": invested,
            "num_positions": len(self.positions),
        })

    def _record_positions(self, today: pd.Timestamp) -> None:
        for symbol, pos in self.positions.items():
            price = self._mark_price(pos, today)
            market_value = pos.shares * price
            self.positions_history_rows.append({
                "date": today, "symbol": symbol, "shares": pos.shares, "entry_price": pos.entry_price,
                "entry_date": pos.entry_date, "price": price, "market_value": market_value,
                "unrealized_pnl": (price - pos.entry_price) * pos.shares,
                "unrealized_return_pct": (price - pos.entry_price) / pos.entry_price * 100,
            })
