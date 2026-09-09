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

# Au-delà de cette part du montant D'ACHAT DEMANDÉ qui n'a pas pu être
# investie faute de cash, le sous-investissement n'est plus un arrondi mais le
# régime normal du run : il est signalé en fin de backtest (cf.
# execution_diagnostics).
#
# Le seuil porte sur des DOLLARS et non sur un nombre d'ordres, contrairement
# à la version précédente. Une part élevée d'ordres "tronqués" ne dit rien
# tant qu'on ne sait pas de combien : mesuré sur un run de référence, 53% des
# ordres étaient réduits... pour 0,12% du montant demandé, soit un portefeuille
# investi à 99,88% de ce que la stratégie demandait. L'ancien seuil déclenchait
# un avertissement alarmant sur un moteur qui faisait exactement son travail.
UNFILLED_DOLLAR_WARNING_PCT = 1.0

# En dessous de cette part de l'univers point-in-time réellement couverte par
# des signaux, la stratégie ne choisit plus dans l'indice mais dans les seuls
# survivants -- et son alpha se mesure contre un indice de référence qui, lui,
# porte les radiées (cf. _record_signal_coverage).
SIGNAL_COVERAGE_WARNING_RATIO = 0.95


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
        signal_max_age_days: int = config.BACKTEST_SIGNAL_MAX_AGE_DAYS,
        momentum_min_pct: Optional[float] = config.BACKTEST_MOMENTUM_MIN_PCT,
        material_events_8k: Optional[pd.DataFrame] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ):
        self.prices = price_panel
        self.last_valid_date = price_panel.last_valid_date
        # Les événements sont triés par date de publication une fois pour
        # toutes, puis consommés au fil de la boucle (cf. _events_up_to) : les
        # rechercher par masque booléen sur la table complète à chacun des
        # ~2500 jours de bourse d'un run coûtait plus cher que le reste de la
        # journée simulée.
        #
        # PAS un dict indexé par date, et l'index exact était un piège : une
        # `filed_date` qui ne tombe pas EXACTEMENT sur un jour du calendrier de
        # bourse n'était jamais retrouvée, donc jamais connue du moteur -- le
        # signal était perdu en silence. Le cas se produit pour de bon : la SEC
        # accepte les dépôts le Vendredi saint, où le NYSE est fermé, et le
        # calendrier vient des cours (03b), pas d'EDGAR. Un signal publié un
        # jour non coté doit être connu à la PREMIÈRE séance suivante, ce que
        # fait la consommation par curseur.
        events = signal_events.sort_values("published_date", kind="stable")
        self._events = events.to_dict("records")
        self._event_dates = pd.DatetimeIndex(events["published_date"]) if len(events) else pd.DatetimeIndex([])
        self._next_event = 0
        self.universe = data_loader.UniverseResolver(universe_history, fallback_universe_symbols)
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.cost_bps = cost_bps
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
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
        # positions ne sont pas vendues : le cash réellement disponible peut
        # être inférieur au budget alloué. Combiné à la règle "on ne sort
        # jamais sur perte de signal", le portefeuille dériverait vers un
        # buy-and-hold de positions périmées sans que rien ne le signale.
        self.buy_orders_count = 0
        self.truncated_orders_count = 0
        # Le sous-investissement en DOLLARS, seule mesure économiquement
        # lisible : un ordre "tronqué" de 0,1% et un ordre non exécuté du tout
        # comptent pareil dans truncated_orders_count, pas ici.
        self.demanded_dollar = 0.0
        self.unfilled_dollar = 0.0

        calendar = price_panel.close.index
        if start_date is not None:
            calendar = calendar[calendar >= start_date]
        if end_date is not None:
            calendar = calendar[calendar <= end_date]
        if len(calendar) == 0:
            raise ValueError("Aucun jour de bourse dans la plage demandée : vérifie start_date/end_date et les données de prix.")
        self.calendar = calendar

        # Couverture annuelle de l'univers par les signaux (cf.
        # _record_signal_coverage) : la mesure du biais de survivance
        # RÉSIDUEL, celui que l'univers point-in-time ne corrige pas. Relevée
        # au DERNIER jour de bourse de chaque année -- un relevé au premier
        # tomberait avant le moindre dépôt et afficherait 0% de couverture sur
        # la première année de n'importe quel run.
        self.signal_coverage_rows: list[dict] = []
        dates = pd.DatetimeIndex(calendar)
        self._coverage_dates = set(pd.Series(dates, index=dates.year).groupby(level=0).last())

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

            todays_events = self._events_up_to(today)
            if todays_events:
                self._update_known_signals(todays_events, today)
                self._rebalance(today, exclude=exited_today)

            self._mark_to_market(today)
            self._record_positions(today)
            self._record_signal_coverage(today)

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
        sells: list[tuple[str, float, float, str]] = []
        buys: list[tuple[str, float, float, str]] = []

        # Le SENS d'un ordre n'est pas le signe de sa cible : une cible de
        # 5 000 $ sur une ligne qui en vaut 8 000 est une VENTE. L'ancien tri
        # (`target_dollar > 0`) ne faisait donc passer devant que les
        # liquidations totales (cible = 0) ; tout rebalancement à la baisse
        # partait dans le même paquet que les achats, dans le seul ordre
        # d'insertion du dictionnaire. Des achats étaient tronqués faute de
        # cash alors que les ventes du même jour les couvraient, et le
        # résultat du run dépendait de l'ordre d'itération d'un dict.
        for symbol, order in self.pending_orders.items():
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

            pos = self.positions.get(symbol)
            delta_dollar = order.target_dollar - (pos.shares if pos else 0.0) * price
            if abs(delta_dollar) < MIN_TRADE_DOLLAR:
                continue
            side = buys if delta_dollar > 0 else sells
            side.append((symbol, delta_dollar / price, price, order.reason))

        self.pending_orders = still_pending

        # Ventes d'abord : leur produit finance les achats du même jour
        # (_rebalance raisonne en NAV, pas en cash).
        for symbol, shares_delta, price, reason in sells:
            self._execute_trade(symbol, shares_delta, price, today, reason)
        self._execute_buys(buys, today)

    def _execute_buys(self, buys: list[tuple[str, float, float, str]], today: pd.Timestamp) -> None:
        """Achats du jour, tous servis dans la MÊME proportion quand le cash
        ne suffit pas.

        Le cash restant peut être inférieur au total demandé sans qu'aucun
        bug ne soit en cause : _rebalance dimensionne sur le NAV de la CLÔTURE
        de J-1, et le portefeuille a rouvert ailleurs. Servir les ordres l'un
        après l'autre jusqu'à épuisement du cash finançait alors intégralement
        les premiers et pas du tout les derniers -- une règle de priorité
        involontaire, calquée sur l'ordre d'apparition des symboles dans le
        flux de dépôts, qu'aucune exécution réelle ne reproduirait. La
        stratégie a demandé des POIDS RELATIFS : un manque de cash doit les
        réduire tous du même facteur, pas en sacrifier une partie."""
        if not buys:
            return

        cost_rate = self.cost_bps / 10_000
        notional = sum(shares * price for _, shares, price, _ in buys)
        demanded = notional * (1 + cost_rate)
        self.buy_orders_count += len(buys)

        scale = 1.0
        if demanded > self.cash:
            # Backtest NON margé : on n'achète jamais à crédit.
            manque = demanded - self.cash
            # Un manque à hauteur des FRAIS du lot n'est pas un défaut, c'est
            # l'arithmétique : _rebalance alloue une VALEUR DE POSITION égale
            # au NAV (positions gelées déduites), alors qu'acquérir cette
            # valeur consomme en plus la commission et le slippage. Un
            # portefeuille pleinement investi est donc court d'exactement
            # cost_bps à chaque rebalancement -- 0,1% ici -- et ne peut pas ne
            # pas l'être : on n'achète pas 100% du NAV en titres ET les frais
            # avec. Le compter comme une troncature faisait afficher "100% des
            # ordres tronqués" à un moteur qui fonctionnait, et surtout
            # noyait les vraies pénuries de cash dans ce bruit.
            if manque > notional * cost_rate + MIN_TRADE_DOLLAR:
                self.truncated_orders_count += len(buys)
            self.unfilled_dollar += manque
            if self.cash <= 0:
                return
            scale = self.cash / demanded
        self.demanded_dollar += demanded

        for symbol, shares_delta, price, reason in buys:
            if shares_delta * price * scale < MIN_TRADE_DOLLAR:
                continue
            self._execute_trade(symbol, shares_delta * scale, price, today, reason)

    def _execute_trade(self, symbol: str, shares_delta: float, price: float, today: pd.Timestamp, reason: str) -> None:
        cost_rate = self.cost_bps / 10_000
        pos = self.positions.get(symbol)

        if shares_delta > 0:  # achat (nouvelle position ou renforcement)
            effective_price = price * (1 + cost_rate)
            # Filet d'arrondi seulement : le dimensionnement au cash et son
            # décompte sont faits en amont, sur le LOT d'achats du jour
            # (_execute_buys). Cette borne ne doit plus mordre que sur des
            # écarts de virgule flottante -- si elle mordait vraiment, du cash
            # partirait en négatif.
            cost = shares_delta * effective_price
            if cost > self.cash:
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
    # Couverture de l'univers par les signaux (biais de survivance résiduel)
    # ------------------------------------------------------------------ #
    def _record_signal_coverage(self, today: pd.Timestamp) -> None:
        """Un relevé par an : combien de membres RÉELS de l'indice à cette
        date le moteur pouvait-il seulement acheter ?

        Un univers point-in-time (01b) empêche d'acheter une entreprise avant
        son entrée dans l'indice, mais il ne CRÉE pas les signaux des
        entreprises radiées. Or 03b se rabat sur l'univers complet quand il
        existe, alors que 04/04b s'en tiennent par défaut à l'univers ACTUEL
        (config.UNIVERSE_FILE) : les cours des radiées sont là -- donc dans
        l'indice de référence équipondéré -- mais pas leurs fondamentaux, donc
        pas leurs signaux. La stratégie ne choisit alors QUE parmi des
        entreprises encore membres aujourd'hui, pendant que le benchmark, lui,
        porte l'indice entier. L'alpha se lit contre un repère que la
        stratégie n'avait pas le droit de perdre, et un signal "value" est
        précisément celui que ce biais flatte le plus : les entreprises les
        moins chères sont aussi celles qui sortent le plus souvent de
        l'indice."""
        if today not in self._coverage_dates:
            return
        members = self.universe.asof(today)
        if not members:
            return
        with_signal = members & self.known_signals.keys()
        self.signal_coverage_rows.append({
            "date": today, "members": len(members), "with_signal": len(with_signal),
            "coverage_ratio": len(with_signal) / len(members),
        })

    def signal_coverage(self) -> pd.DataFrame:
        """Relevé annuel de _record_signal_coverage, à des fins d'inspection."""
        return pd.DataFrame(self.signal_coverage_rows)

    def _signal_coverage_diagnostics(self) -> dict:
        rows = self.signal_coverage_rows
        if not rows:
            return {}
        ratios = [row["coverage_ratio"] for row in rows]
        moyenne = sum(ratios) / len(ratios)
        pire = min(rows, key=lambda row: row["coverage_ratio"])

        if moyenne < SIGNAL_COVERAGE_WARNING_RATIO:
            logger.warning(
                "Couverture moyenne de l'univers point-in-time par les signaux : %.1f%% "
                "(pire année %d : %.1f%%, %d membres, %d avec signal). La stratégie n'a pu "
                "choisir que parmi cette fraction de l'indice, alors que l'indice de "
                "référence en porte la totalité : l'alpha affiché est surestimé d'autant. "
                "Cause habituelle : 04/04b n'ont été lancés que sur l'univers ACTUEL. "
                "Corrige avec 04_recuperation_10k.py --tickers %s (idem 04b), puis 07.",
                moyenne * 100, pire["date"].year, pire["coverage_ratio"] * 100,
                pire["members"], pire["with_signal"], config.UNIVERSE_FULL_FILE,
            )

        return {
            "signal_coverage_avg_ratio": float(moyenne),
            "signal_coverage_min_ratio": float(pire["coverage_ratio"]),
            "signal_coverage_min_year": int(pire["date"].year),
        }

    # ------------------------------------------------------------------ #
    def execution_diagnostics(self) -> dict:
        """Écart entre ce que le rebalancement a DEMANDÉ et ce que le cash a
        permis d'exécuter, plus la couverture de l'univers par les signaux
        (_signal_coverage_diagnostics), à fusionner dans metrics.json.

        La règle des positions gelées est un choix utilisateur assumé et n'est
        pas remise en cause ici -- mais elle a une conséquence qui, elle, doit
        être visible : _rebalance alloue un budget calculé sur le NAV
        (positions gelées déduites) alors que ces positions restent détenues,
        si bien que le cash disponible peut être inférieur au budget.

        DEUX MESURES, ET UNE SEULE EST LISIBLE. truncated_orders_count compte
        des ORDRES réduits au prorata : un ordre rogné de 0,1% y pèse autant
        qu'un ordre jamais passé, et une part résiduelle est de toute façon
        normale (le budget vient du NAV de la clôture de J-1, l'exécution a
        lieu à l'ouverture de J). unfilled_dollar_pct compte des DOLLARS : la
        part du montant demandé qui n'a pas pu être investie. C'est lui qui
        déclenche l'avertissement, et c'est lui qui répond à la question
        "de combien le portefeuille est-il moins investi que la stratégie ne
        le demande ?"."""
        truncated_pct = (
            self.truncated_orders_count / self.buy_orders_count * 100
            if self.buy_orders_count else 0.0
        )
        cash_pct = [
            row["cash"] / row["nav"] * 100
            for row in self.equity_curve_rows if row["nav"] > 0
        ]
        avg_cash_pct = sum(cash_pct) / len(cash_pct) if cash_pct else None

        unfilled_pct = (
            self.unfilled_dollar / self.demanded_dollar * 100 if self.demanded_dollar else 0.0
        )

        if unfilled_pct > UNFILLED_DOLLAR_WARNING_PCT:
            logger.warning(
                "%.2f%% du montant d'achat demandé n'a pas pu être investi faute de cash "
                "(%d ordres sur %d réduits au prorata, au-delà de ce que les frais expliquent ; "
                "cash moyen %.1f%%). Le budget alloué par le rebalancement est calculé sur le NAV "
                "de la clôture de la veille alors que les positions gelées immobilisent du "
                "capital sans être vendues. Accepter ce sous-investissement comme faisant partie "
                "de la règle des positions gelées, ou réduire le nombre de candidates retenues "
                "par la stratégie.",
                unfilled_pct, self.truncated_orders_count, self.buy_orders_count,
                avg_cash_pct if avg_cash_pct is not None else float("nan"),
            )

        return {
            "buy_orders_count": self.buy_orders_count,
            "truncated_orders_count": self.truncated_orders_count,
            "truncated_orders_pct": float(truncated_pct),
            # Le vrai chiffre à lire : la part du montant DEMANDÉ qui n'a pas
            # pu être investie. truncated_orders_pct compte des ordres, celui-ci
            # des dollars -- un ordre rogné de 0,1% et un ordre jamais passé
            # pèsent identiquement dans le premier, pas dans le second.
            "unfilled_dollar_pct": float(unfilled_pct),
            "avg_cash_pct": float(avg_cash_pct) if avg_cash_pct is not None else None,
            **self._signal_coverage_diagnostics(),
        }

    # ------------------------------------------------------------------ #
    # Signaux et rebalancement
    # ------------------------------------------------------------------ #
    def _events_up_to(self, today: pd.Timestamp) -> list[dict]:
        """Signaux devenus publics depuis la dernière séance, consommés une
        seule fois chacun.

        Le premier appel absorbe aussi tout ce qui a été publié AVANT le début
        du run (--start-date) : ces signaux sont réellement connus ce jour-là.
        Ils ne déclenchent pas d'achat pour autant -- leur âge les fait écarter
        par _signal_is_actionable -- mais ils cessent d'être invisibles, ce qui
        évite qu'un run démarré tardivement se croie sans historique."""
        if self._next_event >= len(self._events):
            return []
        fin = int(self._event_dates.searchsorted(today, side="right"))
        if fin <= self._next_event:
            return []
        consommes = self._events[self._next_event:fin]
        self._next_event = fin
        return consommes

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

    def _signal_is_actionable(self, symbol: str, signal: dict, today: pd.Timestamp) -> bool:
        """Ce signal peut-il justifier de METTRE DU CAPITAL sur ce symbole
        aujourd'hui -- première entrée comme renforcement ?

        Deux familles de filtres, qui ne portent pas sur le même objet :

        - PÉREMPTION du signal (âge, 8-K matériel depuis le dépôt) : elle dit
          que l'information elle-même n'est plus une base valable. Elle
          s'applique donc AUSSI aux positions déjà ouvertes -- on n'achète pas
          davantage sur une thèse qu'on vient de déclarer invalide.
        - MOMENTUM : garde-fou "value trap" documenté comme filtre de NOUVELLE
          entrée (cf. _momentum_ok et config.BACKTEST_MOMENTUM_MIN_PCT). Il ne
          s'applique pas à une ligne déjà détenue, sans quoi il deviendrait un
          signal de sortie déguisé, alors que la règle utilisateur est que
          seuls stop-loss/take-profit ferment une position.

        CORRIGÉ. La version précédente laissait TOUTE position détenue court-
        circuiter les trois filtres, en affirmant en commentaire qu'elle était
        "de toute façon gelée, pas rebalancée sur la base de ce vieux signal".
        Elle l'était en réalité : le symbole entrait dans eligible_signals,
        recevait un poids, et sa cible était recalculée à chaque rebalancement
        -- donc RENFORCÉE dès que le NAV montait. Une entreprise qui cesse de
        produire des signaux (FCF passé négatif, EBIT négatif : 07 n'émet plus
        rien pour elle) gardait ainsi indéfiniment son dernier écart, large par
        construction, et le moteur continuait d'acheter dessus. C'est
        exactement le générateur de value trap que les filtres devaient
        empêcher. Un tel symbole devient maintenant une position GELÉE au sens
        de la docstring du module : conservée, jamais renforcée, fermée par
        stop-loss/take-profit uniquement."""
        if symbol not in self.universe.asof(today):
            return False
        max_age = data_loader.signal_max_age_for(signal, self.signal_max_age_days)
        if (today - signal["published_date"]).days > max_age:
            return False
        # Un 8-K matériel déposé depuis la publication du signal (04c) rend
        # celui-ci périmé : les fondamentaux sur lesquels il repose ont bougé.
        if self.material_events.has_event_between(symbol, signal["published_date"], today):
            return False
        return symbol in self.positions or self._momentum_ok(symbol, today)

    def _rebalance(self, today: pd.Timestamp, exclude: set[str]) -> None:
        eligible_signals = pd.DataFrame([
            s for sym, s in self.known_signals.items()
            if self._signal_is_actionable(sym, s, today)
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

        total_weight = sum(target_weights.values())
        # Renormalisé SEULEMENT si la somme dépasse 1 -- jamais vers le haut.
        #
        # La version précédente renormalisait toujours, au motif que rien ne
        # garantissait que la stratégie eût normalisé ses poids à 100%. C'était
        # vrai, et c'est devenu faux : base.capped_weights renvoie
        # DÉLIBÉRÉMENT une somme inférieure à 1 quand le plafond par position
        # mord et qu'il y a trop peu de candidates (une seule candidate ->
        # 20%, le reste en cash). Remonter cette somme à 1 ANNULAIT le
        # plafond : mesuré, une journée à candidate unique plaçait 100% du NAV
        # sur un seul titre, deux candidates 50% chacune, pour un plafond
        # pourtant demandé à BACKTEST_MAX_WEIGHT_PER_POSITION_PCT = 20%.
        #
        # Le moteur options ne renormalise pas (cf. _queue_isolated_order) :
        # les deux moteurs appliquaient donc deux règles de concentration
        # différentes à partir de la même fonction de pondération, ce qui
        # rendait leurs performances non comparables.
        if total_weight > 1:
            target_weights = {s: w / total_weight for s, w in target_weights.items()}

        for symbol, weight in target_weights.items():
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
