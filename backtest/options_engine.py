"""
Moteur de backtest pour la stratégie OPTIONS (backtest/strategies/valuation_gap_options.py) :
même philosophie que backtest/engine.py (actions), mécanique différente
(contrats à échéance, prime, greeks).

Sélection du contrat à l'ENTRÉE (jour D+1, exécution de la décision prise à
la clôture de D -- même règle "pas de look-ahead" que le moteur actions) :
    1. Cherche un snapshot RÉEL archivé par 08_recuperation_options.py à
       proximité de D+1 (data_loader.OptionSnapshotIndex, fenêtre de
       tolérance OPTIONS_REAL_SNAPSHOT_TOLERANCE_DAYS) -- accumulés au fil du
       temps par tes runs successifs de 08 sur le compte paper trading.
    2. Sinon, SIMULE par Black-Scholes (backtest/options_pricing.py) : strike
       = spot du jour (ATM, pas de grille de strikes discrets en mode
       simulé), échéance = D+1 + OPTIONS_TARGET_TENOR_DAYS, volatilité =
       volatilité réalisée glissante (repli si aucune IV réelle disponible).

Une stratégie peut imposer un AUTRE contrat que cet ATM à ~9 mois, en
ajoutant à sa cible `strike_reference_price` (le strike retenu est alors à
mi-chemin entre ce prix de référence et le spot d'exécution) et/ou
`tenor_days` -- voir backtest/strategies/options_base.py. Ces deux clés sont
optionnelles : sans elles, le comportement ci-dessus est inchangé.

Repricing QUOTIDIEN (tant que la position est ouverte) : TOUJOURS par Black-
Scholes, que l'entrée ait été réelle ou simulée -- aucune source ne fournit
un flux d'options continu au jour le jour. Strike et échéance restent ceux
fixés à l'entrée ; la volatilité (implicite réelle si l'entrée était réelle,
sinon estimée) est figée pour toute la durée de vie de la position
(simplification documentée : pas de re-estimation de la vol jour après jour).

Dimensionnement : le capital actif alloué à un symbole (même logique que le
moteur actions -- positions gelées, budget net des positions gelées) est
converti en nombre de contrats via le delta d'entrée : nb_contrats =
capital_alloué / (|delta| x spot x multiplicateur). C'est ce que
l'utilisateur appelle "se hedger grâce aux greeks" : l'exposition $ ciblée
est la même quelle que soit la delta de l'option choisie, pas un nombre de
contrats arbitraire.

Sortie d'une position : stop-loss/take-profit, expiration (réglée à la valeur
intrinsèque), ou disparition des données de prix du sous-jacent.
Stop-loss/take-profit sont, comme les rebalancements, décidés à la clôture de
J et exécutés à l'ouverture de J+1 (même règle que les entrées, cf. plus
haut) ; seules l'expiration (réglée par construction à sa date d'échéance) et
la disparition des données (rien à attendre) sont immédiates.

Trois comportements de sortie/renouvellement sont OPTIONNELS, désactivés par
défaut -- le moteur se comporte exactement comme avant si aucun n'est activé
(c'est le cas de la stratégie valuation_gap_options) :

    stop_basis="underlying"     Stop-loss/take-profit mesurés sur le COURS DU
                                SOUS-JACENT au lieu de la prime, et orientés
                                dans le sens de la position (pour un PUT, une
                                HAUSSE du titre est la perte). Sur une
                                échéance longue, des seuils serrés appliqués à
                                la prime se déclencheraient sur la seule
                                érosion de la valeur temps, avant que la thèse
                                ait le temps de se réaliser -- voir le
                                commentaire de config.OPTIONS_MULTIPLES_STOP_LOSS_PCT.

    exit_when_signal_lost=True  Une position dont le symbole n'est plus jugé
                                éligible par la stratégie (écart repassé sous
                                le seuil) est VENDUE au rebalancement suivant,
                                au lieu de rester "gelée". L'éligibilité est
                                demandée à la stratégie
                                (OptionsStrategy.eligible_symbols) et non
                                déduite de ses cibles : une position évincée
                                par le seul plafond de positions simultanées
                                ne doit pas être vendue comme si son signal
                                avait disparu. Un retournement de sens
                                (call <-> put) ferme aussi la position.

    roll_when_days_left=N       À N jours de l'échéance, une position encore
                                éligible est clôturée et immédiatement
                                rouverte sur une nouvelle échéance pleine (au
                                strike recalculé avec la valorisation
                                théorique la plus récente), à exposition $
                                inchangée. Évite de subir l'accélération de la
                                perte de valeur temps en fin de vie du contrat.

Par défaut (aucune des trois options), une position n'est JAMAIS fermée parce
que son écart de valorisation s'est refermé : elle reste "gelée" jusqu'au
stop-loss/take-profit, à l'expiration ou à la disparition des données -- même
choix utilisateur que la stratégie actions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

import config
from backtest import data_loader, options_pricing
from backtest.strategies.options_base import OptionsStrategy

logger = logging.getLogger("backtest.options_engine")

MIN_TRADE_DOLLAR = 1.0

# Bornes de la volatilité de repricing en mode "rolling" : une volatilité
# réalisée mesurée sur une fenêtre courte peut devenir aberrante (quasi nulle
# sur un titre suspendu, ou plusieurs centaines de % après un saut isolé), et
# Black-Scholes y est très sensible.
MIN_REPRICING_VOL = 0.05
MAX_REPRICING_VOL = 2.00


@dataclass
class OptionPosition:
    symbol: str
    option_type: str            # "CALL" / "PUT"
    strike: float
    expiry: pd.Timestamp
    contracts: float
    entry_premium: float        # prix effectif par action (coût de transaction déjà inclus)
    entry_date: pd.Timestamp
    vol: float                  # volatilité retenue à l'entrée
    multiplier: float
    source: str                 # "real" ou "simulated" (traçabilité, cf. positions_history)

    # Rapport entre la volatilité retenue à l'entrée et la volatilité RÉALISÉE
    # à cette même date, en mode vol_mode="rolling" : la volatilité de
    # repricing suit ensuite la volatilité réalisée du jour, multipliée par ce
    # rapport. Il capture l'écart implicite/réalisé constaté à l'entrée (une
    # entrée sur snapshot réel part d'une IV de marché, pas d'une volatilité
    # historique) et garantit qu'au jour de l'entrée le repricing redonne
    # exactement `vol` -- sans lui, la prime sauterait dès le premier jour.
    # None = volatilité figée pour cette position (mode "frozen", ou
    # volatilité réalisée indisponible à l'entrée).
    vol_ratio: Optional[float] = None

    # Cours du sous-jacent au moment de l'entrée : référence des stop-loss/
    # take-profit quand ils portent sur le sous-jacent (stop_basis="underlying")
    # plutôt que sur la prime.
    entry_spot: float = 0.0
    # Exposition $ visée à l'entrée (avant conversion en contrats par le
    # delta) : rejouée telle quelle lors d'un roulement, pour que renouveler
    # le contrat ne change pas la taille de la position.
    target_dollar: float = 0.0
    # Contrat demandé par la stratégie (None = ATM à l'échéance par défaut du
    # moteur), conservé pour pouvoir reconstruire le même type de contrat lors
    # d'un roulement.
    strike_reference_price: Optional[float] = None
    tenor_days: Optional[int] = None

    # Dernière prime reprise par Black-Scholes, et le jour où elle l'a été.
    # Le contrat (strike, échéance, vol) ne change jamais après l'entrée : la
    # prime à la clôture d'un jour donné ne dépend donc que de ce jour, alors
    # que le moteur la redemande 3 à 4 fois par journée simulée (stop-loss,
    # NAV, mark-to-market, historique des positions).
    _premium_asof: Optional[pd.Timestamp] = None
    _premium: Optional[float] = None


@dataclass
class _PendingOrder:
    target_dollar: float
    reason: str
    queued_on: pd.Timestamp
    # Roulement : clôture puis réouverture immédiate sur une échéance pleine,
    # à exposition inchangée (voir _roll_position). Distinct d'un simple
    # redimensionnement, qui garderait le contrat existant.
    roll: bool = False


class OptionsBacktestEngine:
    def __init__(
        self,
        price_panel: data_loader.PricePanel,
        signal_events: pd.DataFrame,
        universe_history: Optional[pd.DataFrame],
        fallback_universe_symbols: set[str],
        option_snapshots: pd.DataFrame,
        strategy: OptionsStrategy,
        initial_capital: float,
        commission_per_contract: float,
        slippage_pct_of_premium: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        max_positions: int,
        commission_min_per_order: float = config.OPTIONS_COMMISSION_MIN_PER_ORDER,
        commission_max_pct_of_trade: Optional[float] = config.OPTIONS_COMMISSION_MAX_PCT_OF_TRADE,
        whole_contracts: bool = config.OPTIONS_WHOLE_CONTRACTS,
        target_tenor_days: int = config.OPTIONS_TARGET_TENOR_DAYS,
        contract_multiplier: float = config.OPTIONS_CONTRACT_MULTIPLIER,
        real_snapshot_tolerance_days: int = config.OPTIONS_REAL_SNAPSHOT_TOLERANCE_DAYS,
        realized_vol_lookback_days: int = config.OPTIONS_REALIZED_VOL_LOOKBACK_DAYS,
        signal_max_age_days: int = config.BACKTEST_SIGNAL_MAX_AGE_DAYS,
        momentum_min_pct: Optional[float] = config.BACKTEST_MOMENTUM_MIN_PCT,
        material_events_8k: Optional[pd.DataFrame] = None,
        min_resize_relative_pct: Optional[float] = config.OPTIONS_MIN_RESIZE_RELATIVE_PCT,
        stop_basis: str = "premium",
        exit_when_signal_lost: bool = False,
        roll_when_days_left: Optional[int] = None,
        vol_mode: str = "frozen",
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ):
        self.prices = price_panel
        self.last_valid_date = price_panel.last_valid_date
        # Même optimisation que le moteur actions (cf. backtest/engine.py) :
        # événements regroupés par date de publication et contrats réels
        # pré-indexés, plutôt que refiltrés à chaque jour de bourse.
        self.events_by_date = {
            published_date: group.to_dict("records")
            for published_date, group in signal_events.groupby("published_date", sort=False)
        }
        self.universe = data_loader.UniverseResolver(universe_history, fallback_universe_symbols)
        self.option_index = data_loader.OptionSnapshotIndex(option_snapshots)
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_per_contract = commission_per_contract
        self.commission_min_per_order = commission_min_per_order
        self.commission_max_pct_of_trade = commission_max_pct_of_trade
        self.whole_contracts = whole_contracts
        self.slippage_rate = slippage_pct_of_premium / 100
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions
        self.target_tenor_days = target_tenor_days
        self.contract_multiplier = contract_multiplier
        self.real_snapshot_tolerance_days = real_snapshot_tolerance_days
        self.realized_vol_lookback_days = realized_vol_lookback_days
        self.signal_max_age_days = signal_max_age_days
        self.momentum_min_pct = momentum_min_pct
        self.material_events = data_loader.MaterialEventResolver(material_events_8k)
        self.min_resize_relative_pct = min_resize_relative_pct
        if stop_basis not in ("premium", "underlying"):
            raise ValueError(f"stop_basis attend 'premium' ou 'underlying', reçu {stop_basis!r}.")
        self.stop_basis = stop_basis
        self.exit_when_signal_lost = exit_when_signal_lost
        self.roll_when_days_left = roll_when_days_left
        if vol_mode not in ("frozen", "rolling"):
            raise ValueError(f"vol_mode attend 'frozen' ou 'rolling', reçu {vol_mode!r}.")
        self.vol_mode = vol_mode

        self.cash = initial_capital
        self.positions: dict[str, OptionPosition] = {}
        self.known_signals: dict[str, dict] = {}
        self.pending_orders: dict[str, _PendingOrder] = {}
        # Contrat demandé par la stratégie pour l'ordre en attente :
        # {"option_type", "strike_reference_price", "tenor_days"}.
        self._pending_spec: dict[str, dict] = {}
        # {symbole: sens} encore justifiés par le signal au dernier
        # rebalancement, indépendamment du plafond de positions simultanées.
        # Sert à la vente sur perte de signal et au roulement ; None tant
        # qu'aucun rebalancement n'a eu lieu (aucune position ne peut exister).
        self._eligible_directions: Optional[dict[str, str]] = None

        self.trades: list[dict] = []
        self.equity_curve_rows: list[dict] = []
        self.positions_history_rows: list[dict] = []
        self.signals_history_rows: list[dict] = []

        calendar = price_panel.close.index
        if start_date is not None:
            calendar = calendar[calendar >= start_date]
        if end_date is not None:
            calendar = calendar[calendar <= end_date]
        if len(calendar) == 0:
            raise ValueError("Aucun jour de bourse dans la plage demandée.")
        self.calendar = calendar

        if universe_history is None:
            logger.warning(
                "Pas d'historique d'univers (lance 01b_historique_univers_sp500.py) : "
                "l'univers ACTUEL du S&P 500 est appliqué à toutes les dates passées -- "
                "biais de survivance connu, résultats optimistes."
            )
        if option_snapshots.empty:
            logger.warning(
                "Aucun snapshot réel d'options trouvé (lance 08_recuperation_options.py pour "
                "commencer à en accumuler) : toutes les entrées seront simulées par Black-Scholes."
            )

    # ------------------------------------------------------------------ #
    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        for today in self.calendar:
            self._execute_pending_orders(today)
            exited_today = self._settle_expired_positions(today)
            exited_today |= self._handle_stale_underlyings(today)
            exited_today |= self._check_stop_loss_take_profit(today)
            exited_today |= self._check_rolls(today, exclude=exited_today)

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
    # Pricing d'une position ouverte, à une date donnée
    # ------------------------------------------------------------------ #
    def _repricing_vol(self, pos: OptionPosition, today: pd.Timestamp) -> float:
        """Volatilité utilisée pour repricer une position ouverte.

        Mode "frozen" (défaut) : celle retenue à l'entrée, figée pour toute la
        durée de vie de la position. Mode "rolling" : la volatilité réalisée
        du jour, remise à l'échelle de l'entrée (cf. OptionPosition.vol_ratio)
        -- l'approximation "volatilité constante" est alors levée, ce qui
        compte d'autant plus que l'échéance est longue. Repli sur la
        volatilité d'entrée si l'historique du jour est insuffisant."""
        if self.vol_mode != "rolling" or pos.vol_ratio is None:
            return pos.vol
        realized = self.prices.realized_vol_at(pos.symbol, today, self.realized_vol_lookback_days)
        if realized is None or realized <= 0:
            return pos.vol
        return min(max(realized * pos.vol_ratio, MIN_REPRICING_VOL), MAX_REPRICING_VOL)

    def _current_premium(self, pos: OptionPosition, today: pd.Timestamp, spot: Optional[float] = None) -> Optional[float]:
        """Repricing Black-Scholes à strike/échéance fixés (ceux de la
        position), à la volatilité donnée par _repricing_vol. spot explicite
        (prix d'OUVERTURE du jour d'exécution) si fourni -- sinon clôture du
        jour (utilisé pour les décisions/le marked-to-market, jamais pour
        exécuter un ordre, cf. les appelants)."""
        if spot is not None:
            t_years = max((pos.expiry - today).days, 0) / 365.0
            return options_pricing.bs_price(spot, pos.strike, t_years, self._repricing_vol(pos, today), pos.option_type)

        if pos._premium_asof == today:
            return pos._premium
        close = self.prices.close_at(pos.symbol, today)
        if close is None:
            premium = None
        else:
            t_years = max((pos.expiry - today).days, 0) / 365.0
            premium = options_pricing.bs_price(close, pos.strike, t_years, self._repricing_vol(pos, today), pos.option_type)
        pos._premium_asof, pos._premium = today, premium
        return premium

    # ------------------------------------------------------------------ #
    # Exécution des ordres décidés la veille
    # ------------------------------------------------------------------ #
    def _execute_pending_orders(self, today: pd.Timestamp) -> None:
        if not self.pending_orders:
            return

        still_pending: dict[str, _PendingOrder] = {}
        # Ventes d'abord, achats ensuite : leur produit finance les achats du
        # même jour (le capital libéré par une sortie est immédiatement
        # réinvestissable, cf. _rebalance). Sans cet ordre, la garde de cash
        # de _affordable tronquerait des achats pourtant couverts par une
        # vente déjà décidée, en fonction du seul ordre d'insertion.
        ordered = sorted(self.pending_orders.items(), key=lambda kv: kv[1].target_dollar > 0)
        for symbol, order in ordered:
            spot = self.prices.open_at(symbol, today)
            if spot is None:
                if (today - order.queued_on).days > data_loader.FORWARD_FILL_MAX_DAYS:
                    logger.warning(
                        "Ordre en attente pour %s abandonné (%s) : aucune ouverture "
                        "disponible depuis plus de %d jours.",
                        symbol, order.reason, data_loader.FORWARD_FILL_MAX_DAYS,
                    )
                    self._pending_spec.pop(symbol, None)
                    continue
                still_pending[symbol] = order
                continue

            if order.roll:
                self._roll_position(symbol, spot, today)
            elif order.target_dollar <= 0:
                self._close_position(symbol, today, order.reason, spot=spot)
            else:
                spec = self._pending_spec.get(symbol, {})
                self._open_or_resize(
                    symbol, spec.get("option_type"), order.target_dollar, spot, today, order.reason,
                    strike_reference_price=spec.get("strike_reference_price"),
                    tenor_days=spec.get("tenor_days"),
                )
            self._pending_spec.pop(symbol, None)
        self.pending_orders = still_pending

    def _select_contract(
        self,
        symbol: str,
        option_type: str,
        spot: float,
        today: pd.Timestamp,
        strike_reference_price: Optional[float] = None,
        tenor_days: Optional[int] = None,
    ) -> dict:
        """Contrat retenu pour une ENTRÉE. Par défaut (aucun argument
        optionnel) : ATM à l'échéance cible du moteur, comportement
        historique. Avec strike_reference_price, le strike visé est à
        mi-chemin entre ce prix de référence (typiquement la valorisation
        théorique) et le spot d'exécution."""
        tenor = int(tenor_days) if tenor_days else self.target_tenor_days
        target_strike = (strike_reference_price + spot) / 2 if strike_reference_price is not None else None
        wants_custom = strike_reference_price is not None or tenor_days is not None

        real = self.option_index.find(
            symbol, option_type, today, self.real_snapshot_tolerance_days,
            target_strike=target_strike,
            target_tenor_days=float(tenor) if wants_custom else None,
        )
        if real is not None and real["premium"] and pd.notna(real["premium"]) and real["implied_vol"]:
            return {
                "strike": real["strike"], "expiry": real["expiry"], "premium": real["premium"],
                "vol": real["implied_vol"], "delta": real["delta"],
                "multiplier": real["multiplier"], "source": "real",
            }

        vol = options_pricing.realized_volatility(
            self.prices.close_history(symbol, today), self.realized_vol_lookback_days,
        )
        if vol is None:
            vol = 0.30  # repli conservateur si même l'historique de cours est trop court
            logger.debug("%s : historique insuffisant pour estimer la vol réalisée, repli à %.0f%%.", symbol, vol * 100)

        # Simulé : strike exact demandé (ou ATM), pas de grille de strikes discrets.
        strike = target_strike if target_strike is not None else spot
        expiry = today + pd.Timedelta(days=tenor)
        t_years = tenor / 365.0
        premium = options_pricing.bs_price(spot, strike, t_years, vol, option_type)
        greeks = options_pricing.bs_greeks(spot, strike, t_years, vol, option_type)
        return {
            "strike": strike, "expiry": expiry, "premium": premium, "vol": vol,
            "delta": greeks["delta"], "multiplier": self.contract_multiplier, "source": "simulated",
        }

    def _entry_vol_ratio(self, symbol: str, today: pd.Timestamp, entry_vol: float) -> Optional[float]:
        """Rapport volatilité d'entrée / volatilité réalisée du jour, figé pour
        la position (cf. OptionPosition.vol_ratio). None en mode "frozen", ou
        quand la volatilité réalisée est indisponible à l'entrée : la position
        garde alors sa volatilité d'entrée, faute de référence à laquelle
        rapporter les variations ultérieures."""
        if self.vol_mode != "rolling":
            return None
        realized = self.prices.realized_vol_at(symbol, today, self.realized_vol_lookback_days)
        if realized is None or realized <= 0 or not entry_vol:
            return None
        return entry_vol / realized

    def _position_delta(self, pos: OptionPosition, today: pd.Timestamp, spot: float) -> Optional[float]:
        """Delta du contrat DÉTENU (son strike, son échéance, sa volatilité de
        repricing) au spot du jour -- et non celui d'un contrat qu'on
        choisirait aujourd'hui."""
        t_years = max((pos.expiry - today).days, 0) / 365.0
        if t_years <= 0 or spot <= 0:
            return None
        return options_pricing.bs_greeks(
            spot, pos.strike, t_years, self._repricing_vol(pos, today), pos.option_type,
        )["delta"]

    def _round_contracts(self, contracts: float) -> float:
        """Une option se négocie par contrats ENTIERS (cf.
        config.OPTIONS_WHOLE_CONTRACTS). Arrondi au plus proche : c'est la
        taille réalisable la plus proche de l'exposition visée. Une cible
        sous un demi-contrat donne 0, donc pas d'ordre -- une position trop
        petite pour un seul contrat n'est tout simplement pas prenable."""
        if not self.whole_contracts:
            return contracts
        return float(int(contracts + 0.5)) if contracts > 0 else 0.0

    def _order_commission(self, contracts: float, premium_per_share: float, multiplier: float) -> float:
        """Commission IBKR d'UN ordre d'options US : tarif par contrat, mais
        jamais moins que le minimum par ordre, et jamais plus que le plafond
        en % de la valeur négociée (cf. les trois constantes
        config.OPTIONS_COMMISSION_*).

        Le minimum étant PAR ORDRE, la commission n'est plus proportionnelle
        au nombre de contrats : elle ne peut donc pas être repliée dans un
        prix unitaire avant de connaître la taille de l'ordre, contrairement
        à ce que faisait le moteur jusqu'ici."""
        contracts = abs(contracts)
        if contracts <= 0:
            return 0.0
        commission = max(contracts * self.commission_per_contract, self.commission_min_per_order)
        if self.commission_max_pct_of_trade:
            trade_value = contracts * multiplier * max(premium_per_share, 0.0)
            commission = min(commission, trade_value * self.commission_max_pct_of_trade / 100)
        return commission

    def _affordable(self, contracts: float, cost_of) -> tuple[float, float]:
        """Ramène un achat au cash réellement disponible : ce backtest est
        NON margé (pas de vente à découvert ni d'achat à crédit), or rien
        n'empêchait jusqu'ici une file d'ordres de dépenser plusieurs fois le
        capital -- le cash passait négatif et le NAV avec lui, d'où des
        drawdowns inférieurs à -100%, impossibles sur un portefeuille réel.
        Les ventes du jour ayant déjà été exécutées (cf. _execute_pending_orders),
        leur produit est bien inclus dans ce cash.

        `cost_of(contracts) -> coût total` est une fonction (et non un montant)
        parce que la commission n'est plus proportionnelle au nombre de
        contrats : avec un minimum par ordre, réduire la taille de moitié ne
        réduit pas le coût de moitié. Le coût est donc REcalculé après chaque
        réduction, au lieu d'être mis à l'échelle."""
        cost = cost_of(contracts)
        if cost <= self.cash:
            return contracts, cost
        if self.cash <= 0:
            return 0.0, 0.0

        scaled = contracts * (self.cash / cost)
        if self.whole_contracts:
            scaled = float(int(scaled))  # troncature : jamais au-dessus du cash
        for _ in range(64):
            if scaled <= 0:
                return 0.0, 0.0
            cost = cost_of(scaled)
            if cost <= self.cash:
                return scaled, cost
            scaled = scaled - 1.0 if self.whole_contracts else scaled * 0.95
        return 0.0, 0.0

    def _open_or_resize(
        self,
        symbol: str,
        option_type: str,
        target_dollar: float,
        spot: float,
        today: pd.Timestamp,
        reason: str,
        strike_reference_price: Optional[float] = None,
        tenor_days: Optional[int] = None,
    ) -> None:
        existing = self.positions.get(symbol)
        if existing is not None and existing.option_type != option_type:
            # Le signal a changé de sens (call <-> put) alors qu'une position
            # existe déjà dans l'autre sens : on laisse l'ancienne gelée
            # (elle ne sera fermée que par stop/take-profit/expiration/gap de
            # données, cf. docstring module) et on n'ouvre PAS de position
            # simultanée dans les deux sens sur le même sous-jacent.
            logger.debug("%s : signal %s ignoré, une position %s existe déjà (gelée).", symbol, option_type, existing.option_type)
            return

        contract = self._select_contract(
            symbol, option_type, spot, today,
            strike_reference_price=strike_reference_price, tenor_days=tenor_days,
        )
        delta = contract["delta"] or 0.0
        if abs(delta) < 1e-6:
            return

        multiplier = contract["multiplier"]

        if existing is None:
            target_contracts = self._round_contracts(target_dollar / (abs(delta) * spot * multiplier))
            if target_contracts <= 0:
                return
            # Prime brute (slippage inclus) : la commission est ajoutée à part,
            # au niveau de l'ORDRE, puisqu'elle a un minimum forfaitaire.
            gross_premium = contract["premium"] * (1 + self.slippage_rate)

            def cost_of(n: float) -> float:
                return n * multiplier * gross_premium + self._order_commission(n, gross_premium, multiplier)

            if cost_of(target_contracts) < MIN_TRADE_DOLLAR:
                return
            target_contracts, cost = self._affordable(target_contracts, cost_of)
            if target_contracts <= 0 or cost < MIN_TRADE_DOLLAR:
                return
            self.cash -= cost
            # Prix de revient TOUT COMPRIS par action (commission d'ordre
            # répartie sur les contrats effectivement achetés) : conserve la
            # sémantique de OptionPosition.entry_premium et donc le return_pct
            # des trades.
            effective_premium = cost / (target_contracts * multiplier)
            self.positions[symbol] = OptionPosition(
                symbol=symbol, option_type=option_type, strike=contract["strike"], expiry=contract["expiry"],
                contracts=target_contracts, entry_premium=effective_premium, entry_date=today,
                vol=contract["vol"], multiplier=multiplier, source=contract["source"],
                entry_spot=spot, target_dollar=target_dollar,
                strike_reference_price=strike_reference_price, tenor_days=tenor_days,
                vol_ratio=self._entry_vol_ratio(symbol, today, contract["vol"]),
            )
        else:
            existing.target_dollar = target_dollar
            # Renforcement d'une position existante (même sens) : moyenne
            # pondérée du prix d'entrée. Strike/échéance/vol restent ceux du
            # contrat déjà détenu (pas de re-sélection de contrat en cours de vie).
            # Le dimensionnement doit donc utiliser le delta du contrat DÉTENU,
            # pas celui du contrat qu'on aurait choisi aujourd'hui : les deux
            # n'ont ni le même strike ni la même échéance. Mélanger le delta de
            # l'un et la prime de l'autre rend le coût sans rapport avec
            # l'exposition visée (une position très hors de la monnaie se
            # faisait renforcer de plusieurs fois le NAV).
            existing_delta = self._position_delta(existing, today, spot)
            if existing_delta is None or abs(existing_delta) < 1e-6:
                return
            target_contracts = self._round_contracts(target_dollar / (abs(existing_delta) * spot * multiplier))
            delta_contracts = target_contracts - existing.contracts

            # Chaque dépôt de filing (10-K/10-Q) de N'IMPORTE LAQUELLE des
            # ~500 entreprises suivies déclenche un rebalancement qui
            # recalcule les poids de TOUTES les positions détenues (cf.
            # _rebalance) : un changement minime sur une ligne renormalise
            # aussi marginalement toutes les autres, ce qui met en file un
            # micro-ajustement à chaque événement -- des centaines de trades
            # par an qui ne font que payer commission + slippage sans changer
            # la thèse. MIN_TRADE_DOLLAR (1$) ne bloque que les montants
            # absolument négligeables, pas ces resizes proportionnellement
            # mineurs. Comparaison en NOMBRE DE CONTRATS (même unité des deux
            # côtés) plutôt qu'en dollars : target_dollar est une exposition
            # NOTIONNELLE delta-équivalente, alors que la valeur de marché de
            # la position (prime x contrats) en est une fraction systématique
            # (effet de levier) -- comparer les deux directement aurait rendu
            # le seuil inopérant (l'écart de levier domine toujours l'écart
            # réel de resize, quelle que soit sa taille).
            if self.min_resize_relative_pct and existing.contracts > 0:
                if abs(delta_contracts) / existing.contracts < self.min_resize_relative_pct / 100:
                    return

            if abs(delta_contracts) * multiplier * spot < MIN_TRADE_DOLLAR:
                return
            if delta_contracts > 0:
                current_premium = self._current_premium(existing, today, spot=spot) or contract["premium"]
                gross_premium = current_premium * (1 + self.slippage_rate)

                def cost_of(n: float) -> float:
                    return n * multiplier * gross_premium + self._order_commission(n, gross_premium, multiplier)

                delta_contracts, cost = self._affordable(delta_contracts, cost_of)
                if delta_contracts <= 0 or cost < MIN_TRADE_DOLLAR:
                    return
                self.cash -= cost
                effective_premium = cost / (delta_contracts * multiplier)
                new_contracts = existing.contracts + delta_contracts
                existing.entry_premium = (existing.entry_premium * existing.contracts + effective_premium * delta_contracts) / new_contracts
                existing.contracts = new_contracts
            else:
                self._reduce_position(existing, -delta_contracts, today, reason, spot=spot)

    def _reduce_position(self, pos: OptionPosition, contracts_to_sell: float, today: pd.Timestamp, reason: str, spot: Optional[float] = None) -> None:
        contracts_to_sell = min(contracts_to_sell, pos.contracts)
        if contracts_to_sell <= 0:
            return
        current_premium = self._current_premium(pos, today, spot=spot)
        if current_premium is None:
            return
        # Plancher à 0 : quand la commission dépasse ce que vaut encore
        # l'option (prime quasi nulle), on ne PAIE pas pour vendre -- on laisse
        # expirer sans valeur. Sans ce plancher, la sortie encaissait un
        # produit négatif : le trade perdait plus que la prime investie
        # (return_pct < -100%) et le cash fuyait à chaque clôture.
        # Commission retranchée au niveau de l'ORDRE (minimum forfaitaire),
        # pas au prorata de chaque contrat.
        gross_premium = current_premium * (1 - self.slippage_rate)
        gross_value = contracts_to_sell * pos.multiplier * gross_premium
        commission = self._order_commission(contracts_to_sell, gross_premium, pos.multiplier)
        proceeds = max(gross_value - commission, 0.0)
        effective_premium = proceeds / (contracts_to_sell * pos.multiplier)
        self.cash += proceeds
        pnl = (effective_premium - pos.entry_premium) * contracts_to_sell * pos.multiplier
        self.trades.append({
            "symbol": pos.symbol, "entry_date": pos.entry_date, "exit_date": today,
            "shares": contracts_to_sell * pos.multiplier, "entry_price": pos.entry_premium, "exit_price": effective_premium,
            "pnl": pnl, "return_pct": (effective_premium - pos.entry_premium) / pos.entry_premium * 100,
            "holding_days": (today - pos.entry_date).days, "exit_reason": reason,
            "option_type": pos.option_type, "strike": pos.strike, "contracts": contracts_to_sell, "source": pos.source,
        })
        pos.contracts -= contracts_to_sell
        if pos.contracts <= 1e-9:
            del self.positions[pos.symbol]

    def _close_position(self, symbol: str, today: pd.Timestamp, reason: str, spot: Optional[float] = None) -> None:
        pos = self.positions.get(symbol)
        if pos is None:
            return
        self._reduce_position(pos, pos.contracts, today, reason, spot=spot)

    # ------------------------------------------------------------------ #
    # Gestion du risque
    # ------------------------------------------------------------------ #
    def _settle_expired_positions(self, today: pd.Timestamp) -> set[str]:
        expired = {sym for sym, pos in self.positions.items() if today >= pos.expiry}
        for symbol in expired:
            self._close_position(symbol, today, "expiry")
        return expired

    def _position_move_pct(self, pos: OptionPosition, today: pd.Timestamp) -> Optional[float]:
        """Variation à comparer aux seuils de stop-loss/take-profit, dans le
        SENS DE LA POSITION (positif = la position gagne).

        stop_basis="premium" : variation de la prime depuis l'entrée -- déjà
        orientée, une prime qui monte est un gain pour un call comme pour un
        put. stop_basis="underlying" : variation du cours du sous-jacent,
        qu'il faut donc inverser pour un put (le titre qui baisse est le
        scénario gagnant)."""
        if self.stop_basis == "premium":
            premium = self._current_premium(pos, today)
            if premium is None:
                return None
            return (premium - pos.entry_premium) / pos.entry_premium * 100

        close = self.prices.close_at(pos.symbol, today)
        if close is None or not pos.entry_spot:
            return None
        move_pct = (close - pos.entry_spot) / pos.entry_spot * 100
        return move_pct if pos.option_type == "CALL" else -move_pct

    def _check_stop_loss_take_profit(self, today: pd.Timestamp) -> set[str]:
        """Décidé à la clôture de today, exécuté à l'ouverture du jour
        suivant (mis en file via pending_orders, comme un rebalancement) --
        même règle "pas de look-ahead" que le reste du moteur : on ne
        clôture jamais une position au prix exact qui vient de déclencher
        le seuil."""
        triggered = set()
        for symbol, pos in list(self.positions.items()):
            move_pct = self._position_move_pct(pos, today)
            if move_pct is None:
                continue
            if move_pct <= self.stop_loss_pct:
                self.pending_orders[symbol] = _PendingOrder(0.0, "stop_loss", today)
                triggered.add(symbol)
            elif move_pct >= self.take_profit_pct:
                self.pending_orders[symbol] = _PendingOrder(0.0, "take_profit", today)
                triggered.add(symbol)
        return triggered

    def _check_rolls(self, today: pd.Timestamp, exclude: set[str]) -> set[str]:
        """Met en file un roulement pour toute position arrivant à moins de
        roll_when_days_left de son échéance et toujours justifiée par son
        signal. Comme les stops, décidé à la clôture de J et exécuté à
        l'ouverture de J+1."""
        if not self.roll_when_days_left or not self.positions:
            return set()

        rolled = set()
        for symbol, pos in self.positions.items():
            if symbol in exclude or symbol in self.pending_orders:
                continue
            if (pos.expiry - today).days > self.roll_when_days_left:
                continue
            # Plus éligible : inutile de renouveler un contrat que la vente
            # sur perte de signal s'apprête à clôturer (ou que le gel laissera
            # simplement expirer si cette option n'est pas active).
            if self._eligible_directions is not None and self._eligible_directions.get(symbol) != pos.option_type:
                continue
            self.pending_orders[symbol] = _PendingOrder(pos.target_dollar, "roll", today, roll=True)
            rolled.add(symbol)
        return rolled

    def _roll_position(self, symbol: str, spot: float, today: pd.Timestamp) -> None:
        """Clôture puis rouvre immédiatement, à la même exposition $ visée,
        sur une échéance pleine et un strike recalculé avec la valorisation
        théorique la plus récente connue (pas celle de l'entrée initiale)."""
        pos = self.positions.get(symbol)
        if pos is None:
            return

        option_type, target_dollar, tenor_days = pos.option_type, pos.target_dollar, pos.tenor_days
        reference = pos.strike_reference_price
        latest_signal = self.known_signals.get(symbol)
        if reference is not None and latest_signal is not None:
            refreshed = latest_signal.get("valuation_theoretical_per_share")
            if refreshed is not None and pd.notna(refreshed):
                reference = float(refreshed)

        self._close_position(symbol, today, "roll", spot=spot)
        if symbol in self.positions:  # clôture impossible (prime indisponible) : on ne rouvre rien
            return
        self._open_or_resize(
            symbol, option_type, target_dollar, spot, today, "roll",
            strike_reference_price=reference, tenor_days=tenor_days,
        )

    def _handle_stale_underlyings(self, today: pd.Timestamp) -> set[str]:
        closed = set()
        for symbol, pos in list(self.positions.items()):
            last_valid = self.last_valid_date.get(symbol)
            if last_valid is None or pd.isna(last_valid):
                continue
            if (today - last_valid).days <= data_loader.FORWARD_FILL_MAX_DAYS:
                continue
            logger.warning(
                "%s : plus aucun cours du sous-jacent depuis %s (>%d jours), position "
                "options clôturée à la dernière valorisation connue.",
                symbol, last_valid.date(), data_loader.FORWARD_FILL_MAX_DAYS,
            )
            self._close_position(symbol, last_valid, "data_gap")
            closed.add(symbol)
        return closed

    # ------------------------------------------------------------------ #
    # Signaux et rebalancement (même logique de positions gelées que le
    # moteur actions -- voir backtest/engine.py::_rebalance)
    # ------------------------------------------------------------------ #
    def _update_known_signals(self, todays_events: list[dict], today: pd.Timestamp) -> None:
        for row in todays_events:
            self.known_signals[row["symbol"]] = row
            self.signals_history_rows.append({
                "date": today, "symbol": row["symbol"], "sector": row.get("sector"),
                "fiscal_year": row.get("fiscal_year"), "gap_pct": row.get("gap_pct"),
                "valuation_theoretical_per_share": row.get("valuation_theoretical_per_share"),
                "source": row.get("source"),
            })

    def _momentum_ok(self, signal: dict, today: pd.Timestamp) -> bool:
        """Filtre "value trap", ORIENTÉ dans le sens de la position que le
        signal justifie (cf. config.BACKTEST_MOMENTUM_MIN_PCT) : un titre en
        forte baisse disqualifie un CALL (l'écart s'élargit parce que le
        marché intègre une dégradation), mais c'est au contraire la thèse
        d'un PUT -- côté vendeur, c'est donc un titre en forte HAUSSE qui
        est écarté. Historique trop court = pas de mesure, donc pas d'exclusion."""
        if self.momentum_min_pct is None:
            return True
        gap = signal.get("gap_pct")
        if gap is None or pd.isna(gap) or gap == 0:
            return True
        momentum = self.prices.momentum_12_1(signal["symbol"], today)
        if momentum is None:
            return True
        # Momentum réorienté : positif = "va dans le sens de la position".
        oriented = momentum if gap > 0 else -momentum
        return oriented * 100 >= self.momentum_min_pct

    def _rebalance(self, today: pd.Timestamp, exclude: set[str]) -> None:
        universe_today = self.universe.asof(today)
        eligible_signals = pd.DataFrame([
            s for sym, s in self.known_signals.items()
            if sym in universe_today
            and (
                sym in self.positions
                or (
                    (today - s["published_date"]).days
                    <= data_loader.signal_max_age_for(s, self.signal_max_age_days)
                    and self._momentum_ok(s, today)
                    # Cf. backtest/engine.py : un 8-K matériel (04c) déposé
                    # depuis la publication du signal le rend périmé.
                    and not self.material_events.has_event_between(sym, s["published_date"], today)
                )
            )
        ])
        if eligible_signals.empty:
            return

        current_option_types = {sym: pos.option_type for sym, pos in self.positions.items()}
        targets = self.strategy.generate_option_targets(eligible_signals, current_option_types)
        targets = {s: t for s, t in targets.items() if s not in exclude and t["weight"] > 0}

        if self.exit_when_signal_lost or self.roll_when_days_left:
            self._eligible_directions = self.strategy.eligible_directions(eligible_signals)
        if self.exit_when_signal_lost:
            # Vendues demain à l'ouverture : exclues dès maintenant du capital
            # "gelé" et des emplacements occupés, puisque leur valeur
            # redeviendra du cash au même moment que les achats du jour.
            exclude = exclude | self._close_lost_signals(today, exclude)

        if not targets:
            return

        nav_now = self._current_nav(today)

        legacy_value = sum(
            self._position_value(pos, today)
            for sym, pos in self.positions.items()
            if sym not in targets and sym not in exclude
        )
        active_budget = max(nav_now - legacy_value, 0.0)

        already_held = {s: t for s, t in targets.items() if s in self.positions}
        new_candidates = {s: t for s, t in targets.items() if s not in self.positions}

        slots_used = sum(
            1 for sym in self.positions if sym not in targets and sym not in exclude
        ) + len(already_held)
        slots_for_new = max(self.max_positions - slots_used, 0)
        if len(new_candidates) > slots_for_new:
            new_candidates = dict(
                sorted(new_candidates.items(), key=lambda kv: kv[1]["weight"], reverse=True)[:slots_for_new]
            )

        final_active = {**already_held, **new_candidates}
        total_weight = sum(t["weight"] for t in final_active.values())
        if total_weight > 0:
            for t in final_active.values():
                t["weight"] = t["weight"] / total_weight

        for symbol, t in final_active.items():
            self._pending_spec[symbol] = {
                "option_type": t["option_type"],
                "strike_reference_price": t.get("strike_reference_price"),
                "tenor_days": t.get("tenor_days"),
            }
            self.pending_orders[symbol] = _PendingOrder(t["weight"] * active_budget, "rebalance", today)

    def _close_lost_signals(self, today: pd.Timestamp, exclude: set[str]) -> set[str]:
        """Clôture les positions dont le signal ne justifie plus la présence :
        écart repassé sous le seuil d'entrée (le symbole a disparu des
        éligibles), ou écart qui a changé de sens (call détenu alors que le
        signal est devenu vendeur, ou l'inverse).

        Une position simplement évincée par le plafond de positions
        simultanées reste éligible et n'est donc PAS vendue -- c'est tout
        l'intérêt de demander les éligibles à la stratégie plutôt que de les
        déduire de ses cibles, déjà tronquées."""
        closed = set()
        for symbol, pos in self.positions.items():
            if symbol in exclude or symbol in self.pending_orders:
                continue
            wanted_direction = (self._eligible_directions or {}).get(symbol)
            if wanted_direction == pos.option_type:
                continue
            reason = "signal_lost" if wanted_direction is None else "direction_flip"
            self.pending_orders[symbol] = _PendingOrder(0.0, reason, today)
            closed.add(symbol)
        return closed

    # ------------------------------------------------------------------ #
    # Comptabilité quotidienne
    # ------------------------------------------------------------------ #
    def _position_value(self, pos: OptionPosition, today: pd.Timestamp) -> float:
        premium = self._current_premium(pos, today)
        if premium is None:
            premium = pos.entry_premium
        return pos.contracts * pos.multiplier * premium

    def _current_nav(self, today: pd.Timestamp) -> float:
        return self.cash + sum(self._position_value(pos, today) for pos in self.positions.values())

    def _mark_to_market(self, today: pd.Timestamp) -> None:
        invested = sum(self._position_value(pos, today) for pos in self.positions.values())
        nav = self.cash + invested
        self.equity_curve_rows.append({
            "date": today, "nav": nav, "cash": self.cash, "invested_value": invested,
            "num_positions": len(self.positions),
        })

    def _record_positions(self, today: pd.Timestamp) -> None:
        for symbol, pos in self.positions.items():
            premium = self._current_premium(pos, today)
            market_value = pos.contracts * pos.multiplier * (premium if premium is not None else pos.entry_premium)
            self.positions_history_rows.append({
                "date": today, "symbol": symbol, "option_type": pos.option_type, "strike": pos.strike,
                "expiry": pos.expiry, "contracts": pos.contracts, "entry_premium": pos.entry_premium,
                "entry_date": pos.entry_date, "premium": premium, "market_value": market_value,
                "unrealized_pnl": (premium - pos.entry_premium) * pos.contracts * pos.multiplier if premium is not None else None,
                "source": pos.source,
            })
