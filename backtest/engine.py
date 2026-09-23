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
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
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

    # Plus haut atteint depuis l'ouverture de la thèse, pour le stop SUIVEUR.
    # Distinct de stop_reference_price (figé à l'entrée) et de entry_price
    # (moyenné à chaque renfort) : trois références, trois usages.
    peak_price: float = 0.0


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
        # BACKTEST_STOCKS_MOMENTUM_MIN_PCT et non BACKTEST_MOMENTUM_MIN_PCT :
        # ce moteur est celui des ACTIONS, et la grille qui a désactivé le
        # filtre n'a rien mesuré du côté options, dont le moteur garde son
        # propre défaut (cf. config).
        momentum_min_pct: Optional[float] = config.BACKTEST_STOCKS_MOMENTUM_MIN_PCT,
        rebalance_band_pct: float = config.BACKTEST_REBALANCE_BAND_PCT,
        vol_lookback_days: int = config.BACKTEST_VOL_LOOKBACK_DAYS,
        max_plausible_gap_pct: float = config.BACKTEST_MAX_PLAUSIBLE_GAP_PCT,
        # Les trois sorties FACULTATIVES, toutes désactivées par défaut : le
        # moteur se comporte exactement comme avant tant qu'aucune n'est
        # demandée (cf. leurs méthodes respectives pour le raisonnement).
        trailing_stop_pct: Optional[float] = config.BACKTEST_TRAILING_STOP_PCT,
        max_holding_days: Optional[int] = config.BACKTEST_MAX_HOLDING_DAYS,
        exit_gap_threshold_pct: Optional[float] = config.BACKTEST_EXIT_GAP_THRESHOLD_PCT,
        impact_coefficient_bps: float = config.BACKTEST_IMPACT_COEFFICIENT_BPS,
        # Tarification réelle : commission minimum en dollars, plancher de
        # taille relatif au NAV, et part maximale de l'ordre que la commission
        # minimum a le droit de représenter. À 0 -- le défaut -- le moteur se
        # comporte exactement comme avant (cf. config pour le raisonnement).
        min_commission_dollar: float = config.BACKTEST_MIN_COMMISSION_DOLLAR,
        min_trade_pct_of_nav: float = config.BACKTEST_MIN_TRADE_PCT_OF_NAV,
        max_fee_pct_of_trade: float = config.BACKTEST_MAX_FEE_PCT_OF_TRADE,
        vol_target_pct: Optional[float] = config.BACKTEST_VOL_TARGET_PCT,
        vol_target_lookback_days: int = config.BACKTEST_VOL_TARGET_LOOKBACK_DAYS,
        material_events_8k: Optional[pd.DataFrame] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ):
        self.prices = price_panel
        # DICT et non la Series de price_panel : `_handle_stale_symbols`
        # interroge cette table pour CHAQUE position à CHAQUE séance, et un
        # `Series.get(symbole)` reconstruit tout un chemin d'indexation pandas
        # à chaque appel. Mesuré au profileur sur un run complet : 371 301
        # appels pour 7,3 s, soit 17% du temps total, à ne lire qu'une date.
        # Le dict rend exactement les mêmes valeurs (Timestamp ou NaT, None si
        # absent), donc aucun changement de comportement.
        self.last_valid_date = dict(price_panel.last_valid_date)
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
        self.rebalance_band_pct = rebalance_band_pct or 0.0
        self.vol_lookback_days = vol_lookback_days or 0
        self.max_plausible_gap_pct = max_plausible_gap_pct or 0.0
        self.trailing_stop_pct = trailing_stop_pct
        self.max_holding_days = max_holding_days
        self.exit_gap_threshold_pct = exit_gap_threshold_pct
        self.impact_coefficient_bps = impact_coefficient_bps or 0.0
        self.min_commission_dollar = min_commission_dollar or 0.0
        self.min_trade_pct_of_nav = min_trade_pct_of_nav or 0.0
        self.max_fee_pct_of_trade = max_fee_pct_of_trade or 0.0
        self.vol_target_pct = vol_target_pct
        self.vol_target_lookback_days = vol_target_lookback_days
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
        # Friction réellement payée, toutes exécutions confondues : commission,
        # glissement, impact de marché et commission minimum.
        self.total_friction_dollar = 0.0
        self.executions_count = 0
        self.truncated_orders_count = 0
        # Le sous-investissement en DOLLARS, seule mesure économiquement
        # lisible : un ordre "tronqué" de 0,1% et un ordre non exécuté du tout
        # comptent pareil dans truncated_orders_count, pas ici.
        self.demanded_dollar = 0.0
        self.unfilled_dollar = 0.0
        # Jours où un repesage était possible, et ceux que la zone de
        # non-négociation a laissés passer (cf. _drift_is_material) : la mesure
        # de ce que le réglage fait réellement, à lire avec
        # annualized_turnover_pct.
        self.rebalance_days_count = 0
        self.rebalance_skipped_days = 0

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

        # Le plancher relatif se lit sur le NAV du jour, pas sur le capital
        # initial : c'est ce qui le fait tenir à l'échelle quand le
        # portefeuille a été multiplié par huit.
        nav = self._current_nav(today)
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
            # UNE LIQUIDATION PASSE TOUJOURS. Stop-loss, take-profit, stop
            # suiveur, perte de signal et symbole périmé visent une cible de
            # zéro : leur opposer un plancher de taille emprisonnerait dans le
            # portefeuille toute ligne devenue plus petite que lui, sans
            # échappatoire -- le stop-loss cesserait de fonctionner sur
            # exactement les positions qui en ont le plus besoin, celles qui
            # se sont effondrées.
            minimum = MIN_TRADE_DOLLAR if order.target_dollar <= 0 else self._montant_minimal(nav)
            if abs(delta_dollar) < minimum:
                continue
            side = buys if delta_dollar > 0 else sells
            side.append((symbol, delta_dollar / price, price, order.reason))

        self.pending_orders = still_pending

        # Ventes d'abord : leur produit finance les achats du même jour
        # (_rebalance raisonne en NAV, pas en cash).
        for symbol, shares_delta, price, reason in sells:
            self._execute_trade(symbol, shares_delta, price, today, reason)
        self._execute_buys(buys, today, self._montant_minimal(nav))

    def _execute_buys(
        self, buys: list[tuple[str, float, float, str]], today: pd.Timestamp,
        minimum: float = MIN_TRADE_DOLLAR,
    ) -> None:
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

        # Frais du LOT, ordre par ordre : avec une commission minimum, le coût
        # n'est plus proportionnel au total, et provisionner `notional x bps`
        # sous-estimerait le cash nécessaire sur un lot de petits ordres --
        # exactement le régime où la commission minimum mord.
        notional = sum(shares * price for _, shares, price, _ in buys)
        frais = sum(
            shares * price * self._taux_de_cout(symbol, shares * price, today, avec_impact=False)
            for symbol, shares, price, _ in buys
        )
        demanded = notional + frais
        self.buy_orders_count += len(buys)

        # Compté AVANT toute sortie anticipée : la version précédente
        # incrémentait `unfilled_dollar` puis sortait sur `cash <= 0` sans
        # jamais ajouter ce lot au dénominateur. Le ratio ratait donc
        # exactement les journées de pénurie TOTALE -- les pires -- et
        # `unfilled_dollar_pct` en ressortait surestimé, d'autant plus que la
        # pénurie était grave.
        self.demanded_dollar += demanded

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
            if manque > frais + MIN_TRADE_DOLLAR:
                self.truncated_orders_count += len(buys)
            self.unfilled_dollar += manque
            if self.cash <= 0:
                return
            scale = self.cash / demanded

        for symbol, shares_delta, price, reason in buys:
            # Un ordre réduit par le manque de cash sous le seuil de viabilité
            # cesse de l'être : le plancher s'applique à ce qui est RÉELLEMENT
            # exécuté, pas à ce qui était demandé.
            if shares_delta * price * scale < minimum:
                continue
            self._execute_trade(symbol, shares_delta * scale, price, today, reason)

    def _impact_bps(self, symbol: str, montant: float, today: pd.Timestamp) -> float:
        """Impact de marché, en points de base, pour un ordre de `montant`
        dollars sur ce symbole.

        MODÈLE EN RACINE DE LA PARTICIPATION, la forme empirique standard
        (Almgren et al.) : l'impact croît comme la racine de la part du volume
        quotidien qu'on consomme. `impact_coefficient_bps` est l'impact d'un
        ordre égal à 100% du volume quotidien moyen ; un ordre à 1% de ce
        volume en paie donc le dixième.

        POURQUOI CE N'EST PAS UN DÉFAUT. À un million de dollars de capital
        simulé, une ligne pèse quelques dizaines de milliers de dollars contre
        un volume quotidien médian de 113 millions : l'impact est
        rigoureusement négligeable, et l'activer ne changerait rien. Son
        intérêt est ailleurs -- il répond à « jusqu'à quel ENCOURS cette
        stratégie tient », une question que le coût forfaitaire de 10 bps ne
        peut pas poser, puisqu'il ne dépend pas de la taille."""
        if not self.impact_coefficient_bps or montant <= 0:
            return 0.0
        volume = self.prices.dollar_volume_at(symbol, today)
        if not volume:
            return 0.0
        return self.impact_coefficient_bps * math.sqrt(montant / volume)

    def _taux_de_cout(
        self, symbol: str, notionnel: float, today: pd.Timestamp, avec_impact: bool = True,
    ) -> float:
        """Coût d'une exécution, rendu comme un TAUX pour rester compatible
        avec le décalage de prix qui sert de modèle d'exécution.

        Le coût est proportionnel (commission + glissement + impact), SAUF
        quand une commission minimum en dollars est demandée : elle s'y
        substitue dès que l'ordre est trop petit pour l'atteindre. Rendre un
        taux plutôt qu'un montant garde le reste du moteur inchangé -- le prix
        effectif reste `prix x (1 ± taux)` -- tout en rendant le coût
        NON LINÉAIRE en la taille, ce qui est le point : c'est cette
        non-linéarité que le forfait de 10 bps ne pouvait pas exprimer, et qui
        décide de la viabilité d'un petit portefeuille.

        `avec_impact=False` sert au PROVISIONNEMENT du cash dans
        `_execute_buys`, et la distinction n'est pas cosmétique. L'impact de
        marché est un effet de PRIX -- on déplace le marché en passant l'ordre
        --, pas des frais qu'il faudrait mettre de côté d'avance. Le
        provisionner reviendrait à rétrécir l'ordre jusqu'à ce qu'il tienne
        dans le cash IMPACT COMPRIS, donc à ne jamais payer l'impact plutôt
        qu'à le subir : mesuré, un portefeuille d'un milliard sur un marché
        étroit finissait alors avec exactement la performance d'un
        portefeuille d'un million, et toute l'étude de capacité s'effondrait.
        """
        taux = self.cost_bps / 10_000
        if avec_impact:
            taux += self._impact_bps(symbol, notionnel, today) / 10_000
        if self.min_commission_dollar > 0 and notionnel > 0:
            taux = max(taux, self.min_commission_dollar / notionnel)
        return taux

    def _montant_minimal(self, nav: float) -> float:
        """Montant en dessous duquel un ordre n'est PAS passé : le plus
        contraignant des trois planchers.

        1. `MIN_TRADE_DOLLAR`, le garde-fou anti-poussière historique ;
        2. un plancher RELATIF au NAV -- le seul qui tienne à l'échelle, un
           dollar ne voulant pas dire la même chose sur 10 000 $ et sur 8 M$ ;
        3. le seuil de VIABILITÉ déduit de la commission minimum : si l'on
           refuse qu'un ordre paie plus de x % de frais, un ordre sous
           `commission_minimum / x` n'a pas de raison d'exister.

        NE S'APPLIQUE PAS AUX LIQUIDATIONS : voir `_execute_pending_orders`.
        Un plancher filtre ce qu'on choisit de faire, jamais ce qu'on doit
        solder -- sinon une ligne devenue minuscule serait emprisonnée dans le
        portefeuille, stop-loss compris."""
        minimum = MIN_TRADE_DOLLAR
        if self.min_trade_pct_of_nav > 0 and nav > 0:
            minimum = max(minimum, nav * self.min_trade_pct_of_nav / 100.0)
        if self.min_commission_dollar > 0 and self.max_fee_pct_of_trade > 0:
            minimum = max(minimum, self.min_commission_dollar / (self.max_fee_pct_of_trade / 100.0))
        return minimum

    def _execute_trade(self, symbol: str, shares_delta: float, price: float, today: pd.Timestamp, reason: str) -> None:
        cost_rate = self._taux_de_cout(symbol, abs(shares_delta) * price, today)
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
            # Comptabilisé APRÈS le redimensionnement au cash, sur ce qui part
            # vraiment : la friction d'un ordre rogné est celle de l'ordre
            # rogné, pas celle de l'ordre demandé.
            self.total_friction_dollar += shares_delta * price * cost_rate
            self.executions_count += 1
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
        self.total_friction_dollar += sold_shares * price * cost_rate
        self.executions_count += 1
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
            # Plus haut atteint depuis l'ouverture de la thèse : sert au stop
            # SUIVEUR, et se met à jour même quand aucune règle ne se déclenche.
            pos.peak_price = max(pos.peak_price or price, price)

            move_pct = (price - reference) / reference * 100
            raison = None
            if move_pct <= self.stop_loss_pct:
                raison = "stop_loss"
            elif move_pct >= self.take_profit_pct:
                raison = "take_profit"
            elif self._trailing_stop_touche(pos, price):
                raison = "trailing_stop"
            elif self._detention_trop_longue(pos, today):
                raison = "max_holding"
            elif self._these_refermee(symbol, today):
                raison = "signal_lost"

            if raison:
                self._queue_order(symbol, 0.0, raison, today)
                triggered.add(symbol)
        return triggered

    def _trailing_stop_touche(self, pos: Position, price: float) -> bool:
        """Stop SUIVEUR : recul depuis le plus haut atteint DEPUIS L'ENTRÉE, et
        non depuis le prix d'entrée.

        Le stop fixe mesure la perte par rapport à l'ouverture de la thèse : une
        ligne montée de 60% puis redescendue de 55% n'a jamais approché son stop
        alors qu'elle a rendu presque tout son gain. Le stop suiveur protège le
        chemin parcouru ; en contrepartie il sort d'un titre volatil qui n'a rien
        fait de mal, ce qui est exactement le reproche fait au stop serré sur une
        stratégie *value*. D'où un réglage désactivé par défaut, pas une règle."""
        if not self.trailing_stop_pct or not pos.peak_price:
            return False
        return (price - pos.peak_price) / pos.peak_price * 100 <= self.trailing_stop_pct

    def _detention_trop_longue(self, pos: Position, today: pd.Timestamp) -> bool:
        """Durée de détention maximale. Une thèse de convergence qui ne s'est
        pas réalisée en N ans n'est plus une thèse : c'est une position gelée
        que rien ne ferme, puisque seuls les stops le peuvent. Équivalent
        actions de OPTIONS_MIN_HOLDING_DAYS côté options, pris par l'autre
        bout."""
        if not self.max_holding_days:
            return False
        return (today - pos.entry_date).days >= self.max_holding_days

    def _these_refermee(self, symbol: str, today: pd.Timestamp) -> bool:
        """Sortie sur PERTE DE SIGNAL : l'écart de valorisation qui justifiait
        la position s'est refermé sous le seuil de sortie.

        DÉSACTIVÉ PAR DÉFAUT, et ce n'est pas une prudence de façade. La règle
        des positions gelées -- une ligne n'est JAMAIS vendue parce que son
        écart s'est refermé, seuls un stop-loss ou une prise de gain la
        ferment -- est un choix explicite de l'utilisateur, documenté comme tel
        dans le README et dans la docstring du module. Ce réglage rend ce choix
        MESURABLE sans le renverser : à None, le moteur se comporte exactement
        comme avant.

        Le seuil est en points d'écart, comme celui d'entrée : à 0, on sort dès
        que la valeur théorique repasse sous le cours."""
        if self.exit_gap_threshold_pct is None:
            return False
        signal = self.known_signals.get(symbol)
        if signal is None:
            return False
        gap = signal.get("gap_pct")
        if gap is None or gap != gap:
            return False
        # Un signal PÉRIMÉ ne dit plus rien : il ne doit pas déclencher une
        # sortie au motif que sa dernière valeur connue était basse. La
        # péremption gèle la ligne, elle ne la vend pas (cf. _signal_is_actionable).
        max_age = data_loader.signal_max_age_for(signal, self.signal_max_age_days)
        if (today - signal["published_date"]).days > max_age:
            return False
        return gap < self.exit_gap_threshold_pct

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
            # CE QUE LA STRATÉGIE A RÉELLEMENT PAYÉ, en dollars et en nombre.
            # Le moteur facturait sa friction sans jamais la totaliser, si bien
            # qu'on ne pouvait pas répondre à la question la plus naturelle :
            # « moins de transactions, est-ce moins de frais ? ». La réponse
            # n'est pas évidente, et c'est pour ça qu'il faut la mesurer -- la
            # friction suit les DOLLARS NÉGOCIÉS, pas le nombre d'ordres, et
            # supprimer beaucoup de petits ordres peut n'économiser presque
            # rien. Le moteur options tient ce compte depuis toujours
            # (total_commission / total_slippage) ; celui-ci ne le tenait pas.
            "total_friction_dollar": float(self.total_friction_dollar),
            "total_friction_pct_of_initial": float(
                self.total_friction_dollar / self.initial_capital * 100
            ) if self.initial_capital else None,
            "executions_count": int(self.executions_count),
            "avg_friction_per_execution_dollar": float(
                self.total_friction_dollar / self.executions_count
            ) if self.executions_count else None,
            "buy_orders_count": self.buy_orders_count,
            "truncated_orders_count": self.truncated_orders_count,
            "truncated_orders_pct": float(truncated_pct),
            # Le vrai chiffre à lire : la part du montant DEMANDÉ qui n'a pas
            # pu être investie. truncated_orders_pct compte des ordres, celui-ci
            # des dollars -- un ordre rogné de 0,1% et un ordre jamais passé
            # pèsent identiquement dans le premier, pas dans le second.
            "unfilled_dollar_pct": float(unfilled_pct),
            "avg_cash_pct": float(avg_cash_pct) if avg_cash_pct is not None else None,
            # Ce que la bande de non-négociation a réellement filtré. À lire
            # avec annualized_turnover_pct : c'est le même phénomène vu des
            # deux bouts, la part des redimensionnements évités d'un côté, ce
            # qu'ils coûtaient de l'autre.
            "rebalance_band_pct": float(self.rebalance_band_pct),
            "rebalance_days_count": int(self.rebalance_days_count),
            "rebalance_skipped_days_pct": float(
                self.rebalance_skipped_days / self.rebalance_days_count * 100
                if self.rebalance_days_count else 0.0
            ),
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
        # Un écart absurde n'est pas une conviction, c'est une erreur de
        # valorisation (cf. config.BACKTEST_MAX_PLAUSIBLE_GAP_PCT : l'archive
        # en contient jusqu'à +1 817 436 625%). Écarté ICI, au niveau du
        # moteur, et non dans chaque stratégie : le classement des candidates
        # se fait sur cette grandeur, donc une seule stratégie qui oublierait
        # le filtre se retrouverait à choisir ses plus fortes convictions
        # parmi des nombres cassés.
        if self.max_plausible_gap_pct:
            gap = signal.get("gap_pct")
            if gap is not None and gap == gap and abs(gap) > self.max_plausible_gap_pct:
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

        # VOLATILITÉ POINT-IN-TIME, ajoutée par le moteur et non par la
        # stratégie : c'est le moteur qui détient le panel de cours, et la
        # séparation des rôles veut que la stratégie ne voie que des signaux
        # (cf. docstring du module strategies). Une colonne de plus qu'une
        # stratégie est libre d'ignorer -- les trois existantes le faisaient
        # avant que la pondération par le risque n'existe.
        #
        # `realized_vol_at` lit un panel précalculé en une passe vectorisée et
        # mis en cache : le coût par ligne est un accès tableau, pas un calcul.
        if self.vol_lookback_days:
            eligible_signals = eligible_signals.assign(realized_vol=[
                self.prices.realized_vol_at(symbol, today, self.vol_lookback_days)
                for symbol in eligible_signals["symbol"]
            ])

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

        # Ciblage de volatilité : un facteur commun à toutes les cibles, donc
        # sans effet sur leurs poids RELATIFS -- il module l'exposition, pas la
        # sélection (cf. _echelle_ciblage_volatilite). À 1, rien ne change.
        echelle = self._echelle_ciblage_volatilite()
        targets = {
            symbol: weight * active_budget * echelle
            for symbol, weight in target_weights.items()
        }

        self.rebalance_days_count += 1
        if not self._drift_is_material(targets, today, nav_now):
            self.rebalance_skipped_days += 1
            return

        for symbol, target_dollar in targets.items():
            self._queue_order(symbol, target_dollar, "rebalance", today)

    def _echelle_ciblage_volatilite(self) -> float:
        """Facteur appliqué à TOUTES les cibles pour viser une volatilité de
        portefeuille constante.

        POURQUOI ÇA PEUT MARCHER SANS RIEN PRÉDIRE. La volatilité est
        GROUPÉE : une période agitée est suivie d'une période agitée, et c'est
        l'une des rares régularités robustes des marchés. Réduire l'exposition
        quand la volatilité récente est haute réduit donc la volatilité FUTURE
        plus sûrement qu'elle ne réduit le rendement futur -- ce qui est
        exactement la définition d'un gain de Sharpe. Aucune prévision de
        rendement n'y intervient.

        BORNÉ À 1 : le portefeuille peut se désinvestir quand ça secoue, jamais
        s'endetter quand c'est calme. Le moteur n'est pas margé (cf.
        _execute_buys), et un ciblage qui lèverait du levier changerait la
        nature du produit au lieu d'en lisser le risque.

        La volatilité réalisée du PORTEFEUILLE est celle de sa courbe de NAV,
        pas la moyenne de celles de ses lignes : c'est la seule qui tienne
        compte de la diversification, et elle est déjà disponible sans calcul
        supplémentaire. Elle est lue sur les DERNIÈRES séances enregistrées,
        donc sur le passé du jour simulé -- la méthode ne prend volontairement
        pas de date : elle ne peut lire que ce qui est déjà écrit, ce qui rend
        un look-ahead impossible par construction plutôt que par vigilance."""
        if not self.vol_target_pct:
            return 1.0
        fenetre = self.equity_curve_rows[-self.vol_target_lookback_days:]
        if len(fenetre) < 30:
            return 1.0  # historique trop court : on ne module rien
        nav = np.array([row["nav"] for row in fenetre], dtype=float)
        rendements = np.diff(nav) / nav[:-1]
        realisee = float(rendements.std()) * math.sqrt(252) * 100
        if not realisee > 0:
            return 1.0
        return min(1.0, self.vol_target_pct / realisee)

    def _drift_is_material(self, targets: dict[str, float], today: pd.Timestamp, nav: float) -> bool:
        """Le portefeuille s'est-il assez éloigné de sa cible pour qu'il vaille
        la peine de le repeser ? Zone de non-négociation, mesurée sur la
        DÉRIVE TOTALE en % du NAV.

        POURQUOI CE RÉGLAGE EXISTE. `_rebalance` est appelé dès qu'un signal
        est publié, et les poids sont proportionnels à l'écart de valorisation
        RAPPORTÉ À LA SOMME des écarts des candidates (cf.
        base.capped_weights). Un seul 10-Q déposé change donc ce dénominateur,
        et avec lui la cible de TOUTES les lignes du portefeuille -- pas
        seulement celle de l'entreprise qui a publié. Mesuré sur 2015-2026 :
        des dépôts tombent 2624 jours sur 2936, soit un repesage intégral 9
        séances sur 10, pour 722% de rotation annualisée et 62836 exécutions
        au service de 1934 thèses seulement. Chacune paie `cost_bps` à l'aller
        comme au retour, pour un ajustement de poids que la thèse n'a pas
        demandé.

        POURQUOI LA DÉRIVE TOTALE, ET NON UNE BANDE PAR LIGNE. Une bande
        appliquée ligne à ligne -- ne toucher une position que si SA cible
        s'écarte de plus de x% de SA valeur -- a été essayée et mesurée
        d'abord : elle divise bien les exécutions par 15, mais elle filtre
        aussi les ALLÈGEMENTS, qui sont exactement ce qui finance les achats du
        même jour (cf. `_execute_pending_orders`, les ventes passent avant les
        achats précisément pour cela). Le portefeuille se retrouve alors sans
        cash pour ses entrées neuves : mesuré, 52% du montant d'achat demandé
        devenait infinançable, contre 5% sans bande. Le filtre doit donc porter
        sur la DÉCISION DE REPESER, pas sur les lignes une à une : ou bien on
        rebalance le portefeuille en entier -- et les ventes financent les
        achats comme avant --, ou bien on n'y touche pas du tout.

        Le seuil se lit donc en POINTS DE NAV : à 5, on ne repèse que les jours
        où il faudrait faire bouger au moins 5% du portefeuille. Une entrée
        neuve compte sa cible entière dans la dérive, si bien qu'un signal
        vraiment neuf déclenche lui-même le repesage au lieu d'être retardé ;
        et une journée sautée ne remet rien à zéro, la dérive continuant de
        s'accumuler jusqu'à franchir le seuil.

        Sans effet sur les sorties par stop-loss/take-profit, qui ne passent
        pas par ici (cf. `_check_stop_loss_take_profit`), ni sur la règle des
        positions gelées. 0 le désactive et rend au moteur son comportement
        d'avant l'ajout du réglage.

        Côté options, le même bruit est traité par
        `OPTIONS_REBALANCE_LOG_GAP_THRESHOLD`, qui filtre sur le mouvement du
        SIGNAL depuis le dernier trade. Ici c'est la cible qui bouge sans que
        le signal de la ligne ait changé : le filtre doit donc porter sur la
        cible, pas sur le signal."""
        if self.rebalance_band_pct <= 0 or nav <= 0:
            return True

        # Déclaré par la STRATÉGIE, comme `signal_source` : une stratégie dont
        # les cibles ne bougent pas quand une candidate apparaît n'a aucune
        # raison de repeser tout le portefeuille pour l'acheter. `getattr` avec
        # True par défaut : les trois stratégies actions existantes, et toute
        # classe de test qui n'hérite pas de Strategy, gardent EXACTEMENT le
        # comportement d'avant.
        force_sur_entree = getattr(self.strategy, "entree_neuve_force_repesage", True)

        # AMORÇAGE, et seulement quand le coupe-circuit est levé. Sans position
        # en portefeuille, la dérive ne peut plus s'accumuler : une candidate
        # dont la cible reste sous le seuil ne serait alors JAMAIS achetée, et
        # la zone cesserait d'être un filtre de coût pour devenir un filtre de
        # signal. C'est le défaut que le coupe-circuit corrigeait ; le lever
        # exige de le corriger autrement.
        if not force_sur_entree and not self.positions:
            return True

        drift = 0.0
        for symbol, target_dollar in targets.items():
            pos = self.positions.get(symbol)
            if pos is None:
                # UNE ENTRÉE NEUVE N'EST JAMAIS UN AJUSTEMENT DE CONFORT, et la
                # zone ne gouverne que le RE-DIMENSIONNEMENT. Compter sa cible
                # dans la dérive et s'arrêter là avait un défaut que seul un
                # portefeuille à candidate unique révèle : avec un plafond par
                # ligne à 10% du NAV et une zone à 15 points, une candidate
                # SEULE pèse 10 points de dérive, donc reste sous le seuil --
                # et comme rien d'autre ne bouge, elle n'est JAMAIS achetée. La
                # zone cessait d'être un filtre de coût pour devenir un filtre
                # de signal, ce qu'elle n'a jamais eu vocation à être.
                #
                # En portefeuille fourni le cas ne se voit pas (la dérive
                # agrégée franchit le seuil de toute façon) : c'est précisément
                # ce qui en faisait un défaut latent, visible seulement dans les
                # régimes à signal rare.
                # Le seuil de VIABILITÉ, pas le garde-fou anti-poussière : une
                # candidate que la commission minimum rend inachetable n'a
                # aucune raison de déclencher un repesage pour être achetée --
                # elle ne le serait pas.
                if target_dollar >= self._montant_minimal(nav):
                    if force_sur_entree:
                        return True
                    # Coupe-circuit levé : l'entrée neuve reste un ÉCART -- une
                    # position absente est bien une déviation à la cible -- mais
                    # elle ne décide plus à elle seule. Une grosse candidate
                    # franchit encore le seuil toute seule ; une petite attend
                    # que la dérive s'accumule, ce qui est le comportement
                    # demandé.
                    drift += target_dollar
                continue
            current = pos.shares * self._mark_price(pos, today)
            drift += abs(target_dollar - current)
        return drift / nav * 100 >= self.rebalance_band_pct

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
