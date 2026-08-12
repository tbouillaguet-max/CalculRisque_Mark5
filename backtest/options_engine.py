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
       simulé), échéance = D+1 + OPTIONS_TARGET_TENOR_DAYS (2 ans), volatilité
       = volatilité réalisée glissante (repli si aucune IV réelle disponible).

Une stratégie peut imposer un AUTRE contrat que cet ATM à 2 ans, en
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

Levier : ce dimensionnement vise une exposition delta d'environ 1x le NAV,
mais le redéploiement du cash oisif (_deploy_idle_cash) le contredisait
frontalement -- remplir 90% du NAV de primes, à ~10% de prime pour 100% de
spot, porte 8 à 10x le NAV. config.OPTIONS_MAX_DELTA_NOTIONAL_PCT plafonne
désormais explicitement cette exposition, et la colonne `delta_notional` de
l'equity_curve la rend lisible dans les sorties.

Sortie d'une position : stop-loss/take-profit, expiration (réglée à la valeur
intrinsèque), ou disparition des données de prix du sous-jacent.
Stop-loss/take-profit sont, comme les rebalancements, décidés à la clôture de
J et exécutés à l'ouverture de J+1 (même règle que les entrées, cf. plus
haut) ; seules l'expiration (réglée par construction à sa date d'échéance) et
la disparition des données (rien à attendre) sont immédiates.

Les stops sur le sous-jacent et le roulement sont actifs par défaut ; les deux
autres comportements de sortie/renouvellement ci-dessous restent OPTIONNELS et
désactivés par défaut :

Référence des stops : dans les DEUX bases ci-dessous, le seuil se mesure
depuis l'ouverture de la thèse (OptionPosition.stop_reference_premium /
stop_reference_spot, figées à la première ouverture), et non depuis le prix de
revient courant -- lequel continue d'être moyenné à chaque renfort pour le
P&L. Un renfort ne déplace donc jamais le stop.

    stop_basis="underlying"     DÉFAUT (config.OPTIONS_STOP_BASIS).
                                Stop-loss/take-profit mesurés sur le COURS DU
                                SOUS-JACENT, orientés dans le sens de la
                                position (pour un PUT, une HAUSSE du titre est
                                la perte). Sur les échéances 2 ans visées par
                                les deux stratégies, des seuils appliqués à la
                                prime se déclenchent sur la seule érosion de la
                                valeur temps, avant que la thèse ait le temps de
                                se réaliser -- voir les commentaires de
                                config.OPTIONS_STOP_BASIS et
                                config.OPTIONS_MULTIPLES_STOP_LOSS_PCT.
                                stop_basis="premium" rétablit la mesure sur la
                                prime de l'option.

    exit_when_signal_lost=True  Une position dont le symbole n'est plus jugé
                                éligible par la stratégie (écart repassé sous
                                le seuil) est VENDUE au rebalancement suivant,
                                au lieu de rester "gelée". Un retournement de
                                sens (call <-> put) ferme aussi la position.

    roll_when_days_left=N       POINT DE DÉCISION à N jours de l'échéance
                                (config.OPTIONS_ROLL_WHEN_DAYS_LEFT, 9 mois par
                                défaut). Position encore éligible : clôturée et
                                immédiatement rouverte sur une nouvelle
                                échéance pleine (au strike recalculé avec la
                                valorisation théorique la plus récente), à
                                exposition $ inchangée. Position qui ne passe
                                plus les filtres de la stratégie : clôturée
                                (exit_reason "signal_lost"). Dans les deux cas
                                on ne porte jamais un contrat sur sa dernière
                                année de vie, là où la valeur temps s'érode le
                                plus vite. None désactive le réexamen : les
                                positions vont alors jusqu'à l'expiration.

    daily_rebalance=True        Réévalue l'éligibilité CHAQUE jour de bourse
                                (avec le cours DU JOUR), pas seulement les
                                jours où un nouveau signal (10-Q/10-K) tombe
                                -- mais voir DEUX MÉCANISMES DISTINCTS
                                ci-dessous : un jour sans nouveau dépôt
                                n'ouvre QUE des positions neuves, il ne
                                redimensionne JAMAIS l'existant. La
                                valorisation théorique, elle, reste celle du
                                dernier signal connu (inchangée tant qu'aucun
                                nouveau dépôt n'arrive -- ce n'est pas une
                                fuite d'information future). Par défaut
                                (False), seuls les jours d'événement
                                déclenchent un rebalancement, et l'écart y est
                                calculé avec le cours DU DÉPÔT (comportement
                                historique, inchangé).

DEUX MÉCANISMES DISTINCTS DE REBALANCEMENT, à des périmètres volontairement
différents -- un seul point d'entrée qui recalculait tout à chaque occasion
produisait un churn massif (frais + slippage) sans lien avec de l'information
réellement nouvelle :

    1. MÉCANISME JOURNALIER (_rebalance_daily, jours SANS nouveau dépôt,
       daily_rebalance=True) : UNIQUEMENT des OUVERTURES de symboles
       nouvellement éligibles, à leur poids ISOLÉ (conviction_X / somme des
       convictions de tous les éligibles du jour -- ce que la stratégie
       calcule déjà via base.capped_weights sur l'ensemble des candidats).
       AUCUN redimensionnement sur une position déjà détenue : un mouvement
       de cours pur, sans nouvelle publication, ouvre une opportunité neuve
       mais ne justifie pas de retoucher une thèse déjà engagée.

    2. MÉCANISME SUR DÉPÔT SEC (_rebalance_on_signals, 10-K/10-Q/8-K) :
       scopé au(x) SEUL(S) symbole(s) dont le dépôt du jour vient de mettre à
       jour known_signals -- jamais aux autres positions détenues (A, B, C...
       ne sont pas touchées, même si une renormalisation globale les aurait
       affectées). Nouveau candidat : ouvert sans condition. Position déjà
       détenue : redimensionnée seulement si l'écart en log a bougé de plus
       de rebalance_log_gap_threshold (ε, config.OPTIONS_REBALANCE_LOG_GAP_
       THRESHOLD, 0.15 par défaut) DEPUIS LE DERNIER TRADE RÉEL sur cette
       position (OptionPosition.last_rebalance_log_gap), pas depuis le
       dernier signal connu -- en dessous, known_signals est mis à jour mais
       aucun ordre n'est généré, et last_rebalance_log_gap ne bouge pas : le
       seuil suivant continue de se mesurer depuis ce dernier trade, pour
       qu'une dérive lente qui ne franchit jamais ε d'un coup s'accumule
       correctement au fil des dépôts.

    Dans LES DEUX cas, exit_when_signal_lost (sortie sur perte de signal ou
    retournement de direction) reste actif à son périmètre habituel -- ce
    n'est pas un redimensionnement, c'est une décision d'exposition
    orthogonale à ce que ces deux mécanismes contraignent.

Hors du point de décision de roulement, et sans exit_when_signal_lost, une
position n'est pas fermée parce que son écart de valorisation s'est refermé :
elle reste "gelée" jusqu'au stop-loss/take-profit, au réexamen de roulement, à
l'expiration ou à la disparition des données.
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

import config
from backtest import data_loader, options_pricing
from backtest import engine as engine_mod
from backtest.strategies.options_base import OptionsStrategy

logger = logging.getLogger("backtest.options_engine")

MIN_TRADE_DOLLAR = 1.0

# Plancher de prime pour un DIMENSIONNEMENT par division (montant $ visé /
# prime) -- distinct du garde-fou `premium <= 0` déjà en place ailleurs.
# math.erfc (utilisé par options_pricing._norm_cdf) traverse une bande de
# flottants SUBNORMAUX (~1e-310 à 5e-324, non nuls) avant de tomber
# exactement à 0.0 : une option assez loin de la monnaie -- strike extrême,
# lui-même non borné (cf. OPTIONS_MULTIPLES_GAP_BASIS, valuation_theoretical_
# per_share peut être aberrante) -- peut donc pricer une prime qui PASSE le
# test "> 0" tout en étant trop petite pour diviser sans déborder : share /
# (multiplier x 1e-310) dépasse le plus grand double représentable et donne
# `inf`, que int() refuse de convertir (OverflowError). En dessous de ce
# plancher, la position est traitée comme si sa prime était nulle : aucune
# taille finie ne représente correctement un contrat à ce point hors de la
# monnaie.
MIN_PREMIUM_FOR_SIZING = 1e-6

# Sorties qui ont lieu QUOI QU'IL ARRIVE, même quand les frais dépassent la
# valeur résiduelle du contrat : à l'échéance il n'y a plus rien à détenir (le
# contrat est abandonné sans frais), et sur une disparition des cours du
# sous-jacent il n'y a aucune cotation future à espérer. Voir _reduce_position.
FORCED_EXIT_REASONS = frozenset({"expiry", "data_gap"})

# Motifs d'achat EXEMPTÉS du plafond par ordre (config.OPTIONS_MAX_TRADE_DOLLAR).
#
# Un roulement n'ouvre pas une position, il en RENOUVELLE une : il clôture le
# contrat arrivé à son point de décision et le remplace immédiatement par une
# échéance pleine, à exposition inchangée. Lui appliquer le plafond par ordre
# revient à plafonner une CONTINUATION -- exactement le raisonnement pour
# lequel le plafond n'est déjà jamais appliqué aux ventes : une ligne devenue
# grosse doit pouvoir être soldée, et donc aussi rétablie.
#
# Sans cette exemption, toute position dont la prime dépasse le plafond était
# liquidée en un ordre puis rachetée 15 000 $ à la fois sur les jours suivants
# (mesuré : 2 300 contrats -> 165 au premier roulement), en payant le slippage
# et la commission minimum à chaque tranche. Le plafond protégeait d'une
# concentration ; il produisait ici une friction pure.
UNCAPPED_BUY_REASONS = frozenset({"roll"})

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

    # Cours du sous-jacent au moment de l'entrée (traçabilité).
    entry_spot: float = 0.0

    # RÉFÉRENCES DE STOP, figées à la PREMIÈRE ouverture et jamais recalculées
    # ensuite -- à distinguer de entry_premium, prix de revient comptable qui
    # continue lui d'être moyenné à chaque renfort (c'est ce qu'attend le P&L
    # des trades).
    #
    # Sans cette distinction, les deux bases de stop étaient incohérentes
    # entre elles : en stop_basis="premium", entry_premium étant moyenné à la
    # baisse à chaque renfort, la référence du stop baissait avec lui et le
    # stop ne se déclenchait pratiquement jamais tant qu'on moyennait à la
    # baisse -- exactement la situation où il devrait protéger. En
    # stop_basis="underlying", entry_spot n'était au contraire jamais mis à
    # jour, donc le stop restait calé sur la toute première entrée. Les deux
    # modes mesuraient donc deux choses différentes.
    #
    # SÉMANTIQUE RETENUE, désormais commune aux deux modes : le stop mesure la
    # perte depuis l'ouverture de la THÈSE, pas depuis le prix de revient
    # courant. Un renfort ne déplace jamais le seuil ; seule une nouvelle
    # ouverture (position fermée puis rouverte, y compris par roulement) pose
    # une nouvelle référence.
    stop_reference_premium: float = 0.0
    stop_reference_spot: float = 0.0
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

    # log(V/P) au moment du DERNIER TRADE RÉEL sur cette position (ouverture,
    # redimensionnement ou renforcement -- jamais le redéploiement de cash
    # oisif de _deploy_idle_cash, qui la laisse intacte). Sert de référence
    # au filtre ε du mécanisme de rebalancement sur dépôt SEC
    # (_rebalance_on_signals, cf. la docstring du module et
    # config.OPTIONS_REBALANCE_LOG_GAP_THRESHOLD) : None si V ou P était
    # indisponible au moment du trade -- le filtre échoue alors ouvert
    # (redimensionne) plutôt que de geler la position sans pouvoir jamais
    # mesurer son churn.
    last_rebalance_log_gap: Optional[float] = None
    # D'où vient cette position -- "rebalance" (dépôt SEC, comportement
    # historique), "rebalance_daily" (mécanisme journalier, ouverture sur
    # mouvement de cours pur sans nouveau dépôt) ou "roll" (réouverture après
    # roulement). Propagé tel quel dans trades.parquet (colonne open_reason)
    # à la sortie : sans ça, un trade issu du mécanisme journalier était
    # indiscernable d'un trade issu d'un dépôt SEC, alors que les deux
    # répondent à des degrés d'information très différents.
    open_reason: str = "rebalance"


@dataclass
class _PendingOrder:
    target_dollar: float
    reason: str
    queued_on: pd.Timestamp
    # Roulement : clôture puis réouverture immédiate sur une échéance pleine,
    # à exposition inchangée (voir _roll_position). Distinct d'un simple
    # redimensionnement, qui garderait le contrat existant.
    roll: bool = False
    # Vente d'un nombre PRÉCIS de contrats, sans passer par une exposition
    # cible (voir _check_delever) : la réduction au prorata du dé-levier
    # s'exprime naturellement en contrats, pas en dollars de notionnel.
    sell_contracts: Optional[float] = None


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
        commission_min_per_order: float = config.OPTIONS_COMMISSION_MIN_PER_ORDER,
        commission_tiers: tuple = config.OPTIONS_COMMISSION_TIERS,
        fee_bump_max_extra_pct: Optional[float] = config.OPTIONS_FEE_BUMP_MAX_EXTRA_PCT,
        max_fee_pct_of_trade: Optional[float] = config.OPTIONS_MAX_FEE_PCT_OF_TRADE,
        min_deployment_pct: Optional[float] = config.OPTIONS_MIN_DEPLOYMENT_PCT,
        max_delta_notional_pct: Optional[float] = config.OPTIONS_MAX_DELTA_NOTIONAL_PCT,
        delever_tolerance_pct: Optional[float] = config.OPTIONS_DELEVER_TOLERANCE_PCT,
        max_trade_dollar: Optional[float] = config.OPTIONS_MAX_TRADE_DOLLAR,
        max_trade_pct_of_nav: Optional[float] = config.OPTIONS_MAX_TRADE_PCT_OF_NAV,
        take_profit_convergence_fraction: Optional[float] = config.OPTIONS_TAKE_PROFIT_CONVERGENCE_FRACTION,
        whole_contracts: bool = config.OPTIONS_WHOLE_CONTRACTS,
        target_tenor_days: int = config.OPTIONS_TARGET_TENOR_DAYS,
        contract_multiplier: float = config.OPTIONS_CONTRACT_MULTIPLIER,
        real_snapshot_tolerance_days: int = config.OPTIONS_REAL_SNAPSHOT_TOLERANCE_DAYS,
        realized_vol_lookback_days: int = config.OPTIONS_REALIZED_VOL_LOOKBACK_DAYS,
        signal_max_age_days: int = config.BACKTEST_SIGNAL_MAX_AGE_DAYS,
        momentum_min_pct: Optional[float] = config.BACKTEST_MOMENTUM_MIN_PCT,
        material_events_8k: Optional[pd.DataFrame] = None,
        min_resize_relative_pct: Optional[float] = config.OPTIONS_MIN_RESIZE_RELATIVE_PCT,
        rebalance_log_gap_threshold: float = config.OPTIONS_REBALANCE_LOG_GAP_THRESHOLD,
        stop_basis: str = config.OPTIONS_STOP_BASIS,
        exit_when_signal_lost: bool = False,
        roll_when_days_left: Optional[int] = config.OPTIONS_ROLL_WHEN_DAYS_LEFT,
        daily_rebalance: bool = False,
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
        # Secteur par symbole, pour le rendement du dividende du pricing
        # (config.SECTOR_DIVIDEND_YIELD). Le secteur est porté par les signaux
        # eux-mêmes, il n'y a donc rien à recharger. Un symbole sans secteur
        # connu retombe sur "_default".
        self._sector_of: dict[str, object] = (
            signal_events.drop_duplicates("symbol").set_index("symbol")["sector"].to_dict()
            if "sector" in signal_events.columns and not signal_events.empty else {}
        )
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_per_contract = commission_per_contract
        self.commission_min_per_order = commission_min_per_order
        self.commission_tiers = commission_tiers
        self.fee_bump_max_extra_pct = fee_bump_max_extra_pct
        self.max_fee_pct_of_trade = max_fee_pct_of_trade
        self.min_deployment_pct = min_deployment_pct
        self.max_delta_notional_pct = max_delta_notional_pct
        self.delever_tolerance_pct = delever_tolerance_pct or 0.0
        # 0 / None : pas de plafond par ordre (cf. config.OPTIONS_MAX_TRADE_DOLLAR).
        self.max_trade_dollar = max_trade_dollar if max_trade_dollar and max_trade_dollar > 0 else None
        self.max_trade_pct_of_nav = (
            max_trade_pct_of_nav if max_trade_pct_of_nav and max_trade_pct_of_nav > 0 else None
        )
        self.take_profit_convergence_fraction = (
            take_profit_convergence_fraction
            if take_profit_convergence_fraction and take_profit_convergence_fraction > 0 else None
        )
        self.whole_contracts = whole_contracts
        # Contrats négociés par mois civil : détermine le palier dégressif.
        self._monthly_contracts: dict[tuple[int, int], float] = {}
        self.slippage_rate = slippage_pct_of_premium / 100
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.target_tenor_days = target_tenor_days
        self.contract_multiplier = contract_multiplier
        self.real_snapshot_tolerance_days = real_snapshot_tolerance_days
        self.realized_vol_lookback_days = realized_vol_lookback_days
        self.signal_max_age_days = signal_max_age_days
        self.momentum_min_pct = momentum_min_pct
        self.material_events = data_loader.MaterialEventResolver(material_events_8k)
        self.min_resize_relative_pct = min_resize_relative_pct
        # ε du filtre de churn sur le mécanisme sur dépôt SEC -- voir la
        # docstring du module et config.OPTIONS_REBALANCE_LOG_GAP_THRESHOLD.
        self.rebalance_log_gap_threshold = rebalance_log_gap_threshold
        if stop_basis not in ("premium", "underlying"):
            raise ValueError(f"stop_basis attend 'premium' ou 'underlying', reçu {stop_basis!r}.")
        self.stop_basis = stop_basis
        self.exit_when_signal_lost = exit_when_signal_lost
        self.roll_when_days_left = roll_when_days_left
        self.daily_rebalance = daily_rebalance
        if vol_mode not in ("frozen", "rolling"):
            raise ValueError(f"vol_mode attend 'frozen' ou 'rolling', reçu {vol_mode!r}.")
        self.vol_mode = vol_mode

        self.cash = initial_capital
        self.positions: dict[str, OptionPosition] = {}
        # Incrémentée à chaque fill : sert de clé de mémoïsation à
        # _portfolio_greeks (cf. sa docstring).
        self._positions_version = 0
        self._greeks_cache_key: Optional[tuple] = None
        self._greeks_cache: dict = {"delta_notional": 0.0, "theta_per_day": 0.0, "vega_notional": 0.0}
        self._invested_cache_key: Optional[tuple] = None
        self._invested_cache = 0.0
        # (symbole, année) -> (taux sans risque, rendement du dividende) : deux
        # recherches de dictionnaire par pricing, mais appelées ~700 000 fois
        # par run.
        self._pricing_context_cache: dict[tuple, tuple[float, float]] = {}
        self.known_signals: dict[str, dict] = {}
        self.pending_orders: dict[str, _PendingOrder] = {}
        # Contrat demandé par la stratégie pour l'ordre en attente :
        # {"option_type", "strike_reference_price", "tenor_days"}.
        self._pending_spec: dict[str, dict] = {}
        # {symbole: sens} encore justifiés par le signal au dernier
        # rebalancement. Sert à la vente sur perte de signal et au roulement ;
        # None tant qu'aucun rebalancement n'a eu lieu (aucune position ne
        # peut exister).
        self._eligible_directions: Optional[dict[str, str]] = None

        self.trades: list[dict] = []
        # Journal des exécutions (achats ET ventes) -- cf. _record_execution :
        # trades.parquet n'enregistre que les ventes, ce qui rendait la
        # conservation des quantités invérifiable depuis les sorties.
        self.executions: list[dict] = []
        self.equity_curve_rows: list[dict] = []
        self.positions_history_rows: list[dict] = []
        self.signals_history_rows: list[dict] = []

        # Voir execution_diagnostics (même raisonnement que le moteur actions).
        self.buy_orders_count = 0
        self.truncated_orders_count = 0
        # Ordres ramenés au plafond par ordre (max_trade_dollar), comptés à
        # part de truncated_orders_count : les deux causes de réduction n'ont
        # rien à voir (manque de cash contre plafond volontaire) et les
        # confondre rendrait le diagnostic de friction illisible.
        self.capped_orders_count = 0
        # Journées où le plafond de levier a déclenché une réduction au
        # prorata (cf. _check_delever).
        self.delever_events_count = 0

        # ORDRES D'ACHAT PUREMENT ABANDONNÉS, par motif. _open_or_resize
        # compte six sorties anticipées ; seules deux (troncature au cash,
        # plafond par ordre) étaient comptées, et l'abandon pour frais
        # excessifs (_fee_ratio_ok) n'apparaissait NULLE PART -- ni dans
        # buy_orders_count, incrémenté plus loin dans _affordable, ni dans les
        # logs, ni dans metrics.json. Un run pouvait donc redemander la même
        # position chaque jour de bourse pendant des années, échouer chaque
        # fois, rester intégralement en cash, et le rapporter comme un run
        # normal. Tant qu'un ordre peut disparaître sans trace, aucune
        # optimisation de paramètre n'est interprétable.
        self.dropped_orders: dict[str, int] = {}

        # FRICTION CUMULÉE, décomposée. Tant qu'on ne sait pas si la perte
        # vient de la thèse ou du coût de la mettre en oeuvre, tout réglage se
        # fait à l'aveugle : ces deux compteurs sont publiés jour par jour dans
        # l'equity_curve (colonnes total_commission / total_slippage) et en
        # cumul dans execution_diagnostics.
        #
        # Le slippage est compté DES DEUX CÔTÉS : le moteur paie la prime
        # majorée de slippage_rate à l'achat et l'encaisse minorée à la vente
        # (cf. _open_or_resize et _reduce_position). Ne compter que l'achat
        # sous-estimerait la friction de moitié.
        self.total_commission = 0.0
        self.total_slippage = 0.0

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
        # Aucune sortie entre le message de démarrage (10_backtest_options.py)
        # et la fin de cette méthode : sur un historique de plusieurs
        # milliers de jours de bourse -- surtout avec daily_rebalance=True
        # (réévalue TOUT l'univers suivi CHAQUE jour, pas seulement aux dates
        # de dépôt SEC, cf. la docstring du module), qui est désormais le
        # réglage par défaut de la stratégie par défaut -- un run de
        # plusieurs minutes ne se distingue pas d'un run bloqué. Un point
        # d'avancement périodique (~20 lignes au total, quelle que soit la
        # taille du run) suffit à lever le doute sans coût mesurable.
        n_days = len(self.calendar)
        log_every = max(n_days // 20, 1)
        started_at = time.monotonic()
        for i, today in enumerate(self.calendar):
            self._execute_pending_orders(today)
            # Le cash resté oisif est remis au travail sur les positions
            # ouvertes, TOUS LES JOURS. Cet appel vivait dans
            # _execute_pending_orders, qui sort tôt sur
            # `if not self.pending_orders: return` : le redéploiement n'avait
            # donc lieu que les jours où un ordre était déjà en file -- une
            # intermittence involontaire, qui laissait dormir des mois durant
            # le cash libéré par une expiration ou un stop, jusqu'à ce qu'un
            # ordre sans rapport le réveille.
            #
            # Placé juste après l'exécution des ordres (et non en fin de
            # journée simulée) pour conserver l'ordre intra-journalier
            # d'origine : le renforcement s'exécute au prix d'OUVERTURE, il ne
            # doit donc pas être décidé après des étapes qui, elles, lisent la
            # clôture du jour (expiration, stops, rebalancement).
            self._deploy_idle_cash(today)
            exited_today = self._settle_expired_positions(today)
            exited_today |= self._handle_stale_underlyings(today)
            exited_today |= self._check_stop_loss_take_profit(today)
            exited_today |= self._check_rolls(today, exclude=exited_today)
            # Dé-levier APRÈS les sorties déjà décidées (elles réduisent
            # peut-être déjà assez l'exposition) et AVANT le rebalancement (on
            # ne veut pas qu'un renforcement du jour reparte à la hausse une
            # ligne qu'on vient de décider de réduire).
            exited_today |= self._check_delever(today, exclude=exited_today)

            # DEUX mécanismes de rebalancement disjoints (cf. la docstring du
            # module) : sur dépôt SEC quand today en porte un (scopé aux
            # seuls symboles concernés, ε-filtré sur l'existant) ; sinon,
            # journalier si daily_rebalance est actif et qu'il existe déjà au
            # moins un signal connu (ouvertures neuves uniquement, jamais de
            # redimensionnement).
            todays_events = self.events_by_date.get(today)
            if todays_events:
                self._update_known_signals(todays_events, today)
                exited_today |= self._rebalance_on_signals(
                    {row["symbol"] for row in todays_events}, today, exclude=exited_today,
                )
            elif self.daily_rebalance and self.known_signals:
                exited_today |= self._rebalance_daily(today, exclude=exited_today)

            self._mark_to_market(today)
            self._record_positions(today)

            if (i + 1) % log_every == 0 or i + 1 == n_days:
                elapsed = time.monotonic() - started_at
                # Débit en jours/s : à ce stade il permet d'estimer le temps
                # restant sans avoir à attendre la fin pour savoir si le run
                # progresse ou est bloqué.
                rate = (i + 1) / elapsed if elapsed > 0 else float("inf")
                remaining = (n_days - (i + 1)) / rate if rate > 0 else float("inf")
                logger.info(
                    "%d/%d jours de bourse (%.0f%%) -- %s, %d positions ouvertes, "
                    "%.0fs écoulées, ~%.0fs restantes.",
                    i + 1, n_days, (i + 1) / n_days * 100, today.date(),
                    len(self.positions), elapsed, remaining,
                )

        equity_curve = pd.DataFrame(self.equity_curve_rows)
        positions_history = pd.DataFrame(self.positions_history_rows)
        trades = pd.DataFrame(self.trades)
        signals_history = pd.DataFrame(self.signals_history_rows)
        return equity_curve, positions_history, trades, signals_history

    # ------------------------------------------------------------------ #
    # Pricing d'une position ouverte, à une date donnée
    # ------------------------------------------------------------------ #
    def _pricing_context(self, symbol: str, today: pd.Timestamp) -> tuple[float, float]:
        """(taux sans risque, rendement du dividende) à utiliser pour pricer ce
        symbole à cette date.

        Le taux vient de la courbe annuelle (config.RISK_FREE_RATE_BY_YEAR) :
        une constante à 4% sur 2010-2026 était fausse des deux côtés, le taux
        réel étant allé de ~0,05% à ~5,3%. Le rendement du dividende vient du
        SECTEUR de l'entreprise (config.SECTOR_DIVIDEND_YIELD) faute de donnée
        par titre -- approximation assumée, mais qui corrige un biais
        systématique : sans dividende, Black-Scholes surévalue les calls et
        sous-évalue les puts, d'autant plus que le rendement est élevé, donc
        précisément sur les secteurs qu'un signal "value" sélectionne.

        Mémoïsé par (symbole, année) : le couple ne dépend de rien d'autre, et
        le moteur le redemande à chaque pricing d'option."""
        key = (symbol, today.year)
        cached = self._pricing_context_cache.get(key)
        if cached is None:
            cached = (
                config.risk_free_rate_for(today.year),
                config.dividend_yield_for(self._sector_of.get(symbol)),
            )
            self._pricing_context_cache[key] = cached
        return cached

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
        r, q = self._pricing_context(pos.symbol, today)
        if spot is not None:
            t_years = max((pos.expiry - today).days, 0) / 365.0
            return options_pricing.bs_price(
                spot, pos.strike, t_years, self._repricing_vol(pos, today), pos.option_type, r=r, q=q,
            )

        if pos._premium_asof == today:
            return pos._premium
        close = self.prices.close_at(pos.symbol, today)
        if close is None:
            premium = None
        else:
            t_years = max((pos.expiry - today).days, 0) / 365.0
            premium = options_pricing.bs_price(
                close, pos.strike, t_years, self._repricing_vol(pos, today), pos.option_type, r=r, q=q,
            )
        pos._premium_asof, pos._premium = today, premium
        return premium

    # ------------------------------------------------------------------ #
    # Exécution des ordres décidés la veille
    # ------------------------------------------------------------------ #
    def _order_direction(self, symbol: str) -> Optional[str]:
        """Sens d'un ordre en attente. Il vient de la spec déposée par le
        rebalancement quand il y en a une, sinon de la position détenue : un
        roulement est mis en file sans spec (cf. _check_rolls) et garde
        forcément le sens du contrat qu'il renouvelle."""
        spec = self._pending_spec.get(symbol)
        if spec and spec.get("option_type"):
            return spec["option_type"]
        pos = self.positions.get(symbol)
        return pos.option_type if pos else None

    def _execution_order(self) -> list[tuple[str, _PendingOrder]]:
        """Ordre d'exécution de la file du jour.

        1. VENTES D'ABORD. Leur produit finance les achats du même jour (le
           capital libéré par une sortie est immédiatement réinvestissable,
           cf. _rebalance) : sans cela, la garde de cash de _affordable
           tronquerait des achats pourtant couverts par une vente déjà décidée.

        2. ACHATS EN ALTERNANCE CALL / PUT, chaque côté trié par montant
           décroissant. Les achats étaient auparavant exécutés dans l'ordre
           d'insertion, qui est celui du classement par conviction de la
           stratégie -- or ce classement n'est pas neutre en direction. Quand
           le cash s'épuise en cours de file, _affordable tronque les DERNIERS
           servis : un classement qui place systématiquement une direction en
           tête la finance intégralement et laisse l'autre absorber toute la
           troncature. Le classement devenait ainsi un filtre directionnel,
           alors qu'il n'est censé exprimer qu'une conviction.

           Alternance DÉTERMINISTE plutôt qu'aléatoire : un mélange par tirage
           rendrait un run non reproductible d'une exécution à l'autre, ce qui
           interdirait toute comparaison de paramètres. Ici la première CALL et
           la première PUT passent avant la deuxième de chaque côté, et le
           résultat ne dépend que de la file.

           Une direction vide ne réserve rien : l'autre prend tout le budget."""
        sells, calls, puts, unknown = [], [], [], []
        for symbol, order in self.pending_orders.items():
            if order.target_dollar <= 0:
                sells.append((symbol, order))
                continue
            direction = self._order_direction(symbol)
            if direction == "CALL":
                calls.append((symbol, order))
            elif direction == "PUT":
                puts.append((symbol, order))
            else:
                unknown.append((symbol, order))

        calls.sort(key=lambda kv: kv[1].target_dollar, reverse=True)
        puts.sort(key=lambda kv: kv[1].target_dollar, reverse=True)

        interleaved = [
            order
            for pair in itertools.zip_longest(calls, puts)
            for order in pair
            if order is not None
        ]
        return sells + interleaved + unknown

    def _execute_pending_orders(self, today: pd.Timestamp) -> None:
        if not self.pending_orders:
            return

        still_pending: dict[str, _PendingOrder] = {}
        ordered = self._execution_order()
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
            elif order.sell_contracts is not None:
                pos = self.positions.get(symbol)
                if pos is not None:
                    self._reduce_position(pos, order.sell_contracts, today, order.reason, spot=spot)
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

    def _spot_for_execution(self, symbol: str, today: pd.Timestamp) -> Optional[float]:
        """Prix d'ouverture du jour (celui auquel un ordre s'exécute), à
        défaut la clôture -- utilisé pour mesurer une exposition, jamais pour
        exécuter un ordre."""
        spot = self.prices.open_at(symbol, today)
        return spot if spot is not None else self.prices.close_at(symbol, today)

    def _portfolio_greeks(self, today: pd.Timestamp) -> dict:
        """Exposition agrégée du portefeuille, en UNE passe et mémoïsée :

            delta_notional : somme de |delta| x spot x contrats x multiplicateur.
                La mesure honnête du levier -- combien de dollars de
                sous-jacent le portefeuille suit réellement. La valeur des
                primes n'en est qu'une fraction (une option ATM à 9 mois vaut
                ~8-12% du spot), ce qui rendait le levier invisible.
            theta_per_day  : perte de valeur temps du jour, en dollars. C'est
                le COÛT DE PORTAGE de la stratégie -- le seuil que la thèse
                doit battre avant de gagner quoi que ce soit (mesuré à ~9% de
                NAV par an sur une échéance 2 ans), et il n'apparaissait dans
                aucune sortie.
            vega_notional  : sensibilité à +1 point de volatilité implicite.
                La stratégie achetant CALL *et* PUT, elle est LONGUE DE VEGA
                des deux côtés. Le repricing ne suit jamais l'implicite (le
                pipeline ne collecte aucune surface), donc ce P&L n'est pas
                simulé -- mais l'exposition, elle, doit au moins être visible.

        MÉMOÏSATION par (date, version du portefeuille). _deploy_idle_cash
        redemandait le delta notionnel À CHAQUE symbole de sa boucle, et
        chaque appel reparcourait toutes les positions en appelant
        bs_greeks : le coût était en O(n²) par passe, sur 8 passes par jour de
        bourse. La version ne change qu'au moment où un fill modifie réellement
        le portefeuille, donc les recalculs se comptent désormais en nombre de
        FILLS du jour, pas en nombre de symboles scannés."""
        key = (today, self._positions_version)
        if self._greeks_cache_key == key:
            return self._greeks_cache

        delta_notional = theta = vega = 0.0
        for symbol, pos in self.positions.items():
            spot = self._spot_for_execution(symbol, today)
            if spot is None or spot <= 0:
                continue
            greeks = self._position_greeks(pos, today, spot)
            if greeks is None:
                continue
            quantity = pos.contracts * pos.multiplier
            delta_notional += abs(greeks["delta"]) * spot * quantity
            theta += greeks["theta"] * quantity
            vega += greeks["vega"] * quantity

        self._greeks_cache_key = key
        self._greeks_cache = {
            "delta_notional": delta_notional,
            "theta_per_day": theta,
            "vega_notional": vega,
        }
        return self._greeks_cache

    def _delta_notional(self, today: pd.Timestamp) -> float:
        return self._portfolio_greeks(today)["delta_notional"]

    def _touch_positions(self) -> None:
        """À appeler dès qu'un fill modifie le portefeuille : invalide la
        mémoïsation de _portfolio_greeks."""
        self._positions_version += 1

    def _delta_budget(self, today: pd.Timestamp, nav: Optional[float] = None) -> Optional[float]:
        """Exposition delta-équivalente encore autorisée, en dollars. None
        quand le plafond est désactivé."""
        if not self.max_delta_notional_pct:
            return None
        nav = self._current_nav(today) if nav is None else nav
        return max(nav, 0.0) * self.max_delta_notional_pct / 100

    def _contracts_under_remaining_delta(
        self, delta: float, spot: float, multiplier: float, remaining: float,
    ) -> float:
        """Nombre de contrats que le budget de levier restant laisse ouvrir sur
        un contrat de ce delta. Troncature, jamais d'arrondi au plus proche :
        un plafond qu'on arrondit vers le haut n'en est pas un."""
        if abs(delta) < 1e-6 or spot <= 0 or multiplier <= 0:
            return 0.0
        allowed = remaining / (abs(delta) * spot * multiplier)
        if allowed <= 0:
            return 0.0
        return float(int(allowed)) if self.whole_contracts else allowed

    def _contracts_under_delta_cap(
        self, pos: OptionPosition, spot: float, today: pd.Timestamp, remaining_delta_notional: float,
    ) -> float:
        """Nombre de contrats supplémentaires que le plafond de levier laisse
        encore ajouter sur cette position. Troncature (jamais d'arrondi au
        plus proche) : un plafond qu'on arrondit vers le haut n'en est pas un."""
        delta = self._position_delta(pos, today, spot)
        if delta is None or abs(delta) < 1e-6 or spot <= 0:
            return 0.0
        allowed = remaining_delta_notional / (abs(delta) * spot * pos.multiplier)
        if allowed <= 0:
            return 0.0
        return float(int(allowed)) if self.whole_contracts else allowed

    def _deploy_idle_cash(self, today: pd.Timestamp) -> None:
        """Renforce les positions ouvertes tant que la part du NAV investie en
        primes reste sous config.OPTIONS_MIN_DEPLOYMENT_PCT, ET que le levier
        reste sous config.OPTIONS_MAX_DELTA_NOTIONAL_PCT.

        Le dimensionnement du moteur vise une exposition NOTIONNELLE
        delta-équivalente ; la prime effectivement décaissée n'en représente
        qu'une fraction (l'effet de levier de l'option), si bien qu'allouer
        100% du budget laissait en pratique l'essentiel du capital en cash.
        Ce complément remet ce cash au travail sur les convictions DÉJÀ
        retenues -- au prorata de leur taille actuelle, donc sans modifier la
        hiérarchie décidée par la stratégie, et sans ouvrir de ligne que la
        stratégie n'a pas choisie.

        MAIS ce plancher de primes, poussé à 90% du NAV, ANNULAIT le
        dimensionnement par delta qu'il est censé compléter : à ~10% de prime
        pour 100% de spot, remplir 90% du NAV de primes porte une exposition
        delta de 8 à 10x le NAV. Le plafond de delta-notionnel est donc
        vérifié AVANT toute passe et RE-vérifié à chaque renforcement : dès
        qu'il est atteint, le redéploiement s'arrête, même s'il reste du cash
        et que le plancher de primes n'est pas satisfait. Le plafond prime sur
        le plancher, qui n'est qu'une préférence.

        Renforcer une position se fait sur SON contrat (strike et échéance
        inchangés), au prix d'ouverture du jour, comme n'importe quel
        renforcement.

        Un renfort dont la taille reste sous min_resize_relative_pct des
        contrats déjà détenus est IGNORÉ (même garde-fou que le
        redimensionnement sur dépôt SEC, cf. _open_or_resize) : cette méthode
        est appelée CHAQUE jour de bourse, sans mémoire d'un renfort récent,
        donc une position à peine sous le plancher de déploiement se
        ferait sinon renforcer d'un ou deux contrats indéfiniment, jour après
        jour, en payant plein tarif de frais et de slippage à chaque fois --
        de quoi consommer tout le capital en friction pure sur un backtest
        de plusieurs années, sans rapport avec la performance de la thèse."""
        if not self.min_deployment_pct or not self.positions:
            return

        for _ in range(8):  # quelques passes suffisent ; borne dure anti-boucle
            invested = sum(self._position_value(pos, today) for pos in self.positions.values())
            nav = self.cash + invested
            if nav <= 0 or invested >= nav * self.min_deployment_pct / 100:
                return
            missing = nav * self.min_deployment_pct / 100 - invested
            if missing < MIN_TRADE_DOLLAR or self.cash < MIN_TRADE_DOLLAR:
                return

            delta_budget = nav * self.max_delta_notional_pct / 100 if self.max_delta_notional_pct else None
            if delta_budget is not None and self._delta_notional(today) >= delta_budget:
                return

            values = {sym: self._position_value(pos, today) for sym, pos in self.positions.items()}
            total_value = sum(values.values())
            if total_value <= 0:
                return

            progressed = False
            for symbol, value in sorted(values.items(), key=lambda kv: kv[1], reverse=True):
                pos = self.positions.get(symbol)
                spot = self.prices.open_at(symbol, today)
                if pos is None or spot is None:
                    continue
                premium = self._current_premium(pos, today, spot=spot)
                # Pas seulement `<= 0` : une prime SUBNORMALE (non nulle mais
                # proche de la limite basse des flottants) fait déborder la
                # division ci-dessous vers l'infini avant même d'atteindre
                # _round_contracts (cf. MIN_PREMIUM_FOR_SIZING).
                if premium is None or premium < MIN_PREMIUM_FOR_SIZING:
                    continue

                gross_premium = premium * (1 + self.slippage_rate)
                share = missing * (value / total_value)
                extra = self._round_contracts(share / (pos.multiplier * gross_premium))
                if delta_budget is not None:
                    # Recalculé à chaque symbole : les renforcements déjà
                    # passés dans cette même boucle consomment le budget.
                    # _delta_notional est mémoïsé par version du portefeuille
                    # (cf. _portfolio_greeks), donc ce rappel est gratuit tant
                    # qu'aucun fill n'a eu lieu depuis le précédent.
                    remaining = delta_budget - self._delta_notional(today)
                    if remaining <= 0:
                        return
                    extra = min(extra, self._contracts_under_delta_cap(pos, spot, today, remaining))
                if extra <= 0:
                    continue
                # Même filtre que le redimensionnement sur dépôt SEC
                # (min_resize_relative_pct, cf. _open_or_resize) : sans lui,
                # CE renfort tourne CHAQUE jour de bourse tant que le plancher
                # n'est pas atteint (la boucle qui l'appelle, cf. run(), n'a
                # pas de notion de "déjà renforcé récemment") -- une position
                # à peine sous le plancher se voit ainsi renforcée d'un ou
                # deux contrats par jour, indéfiniment, en payant plein tarif
                # de slippage aller-retour (config.OPTIONS_SLIPPAGE_PCT_OF_
                # PREMIUM) et la commission minimum par ordre à chaque fois.
                # Cumulée sur des années de rebalancement quotidien, cette
                # friction à elle seule peut consommer tout le capital
                # initial, sans qu'aucune thèse n'ait perdu quoi que ce soit.
                if self.min_resize_relative_pct and pos.contracts > 0:
                    if extra / pos.contracts < self.min_resize_relative_pct / 100:
                        continue

                def cost_of(n: float, _p=gross_premium, _m=pos.multiplier) -> float:
                    return n * _m * _p + self._order_commission(n, _p, _m, today, "buy")

                extra, cost = self._affordable(extra, cost_of)
                if extra <= 0 or cost < MIN_TRADE_DOLLAR:
                    continue
                self.cash -= cost
                self._record_contract_volume(extra, today)
                effective_premium = cost / (extra * pos.multiplier)
                new_contracts = pos.contracts + extra
                # Ce renfort paie commission et slippage comme n'importe quel
                # achat : tant que sa friction n'était pas comptabilisée, la
                # friction du chemin de code LE PLUS ACTIF du moteur (appelé
                # chaque jour de bourse) n'apparaissait dans aucun total, et
                # total_friction_dollar sous-estimait lourdement le coût réel.
                self._record_fill(
                    pos.symbol, "buy", extra, effective_premium, -cost, today,
                    "deploy_idle_cash", pos.option_type, pos.multiplier,
                    mid_premium=premium,
                    commission=cost - extra * pos.multiplier * gross_premium,
                )
                # Prix de revient moyenné (P&L), référence de stop INTACTE
                # (cf. OptionPosition.stop_reference_premium).
                pos.entry_premium = (pos.entry_premium * pos.contracts + effective_premium * extra) / new_contracts
                # EXPOSITION VISÉE mise à jour au prorata du renfort. Sans
                # cela, target_dollar restait figé à sa valeur d'ouverture
                # alors que la position avait grossi -- et le roulement, qui
                # rejoue target_dollar, la ramenait brutalement à sa taille
                # d'origine (mesuré : 2 253 -> 161 contrats), en payant
                # l'aller-retour complet pour une "exposition inchangée" qui ne
                # l'était pas. Les deux mécanismes doivent parler de la même
                # grandeur.
                pos.target_dollar *= new_contracts / pos.contracts
                pos.contracts = new_contracts
                self._touch_positions()
                progressed = True

            if not progressed:
                return

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

        # L'échéance visée est transmise même quand la stratégie n'en impose
        # aucune : sans elle, un snapshot réel disponible ferait entrer sur le
        # contrat le plus proche de la monnaie quelle que soit sa maturité,
        # alors que l'échéance fait partie de la thèse (cf. docstring module).
        r, q = self._pricing_context(symbol, today)

        real = self.option_index.find(
            symbol, option_type, today, self.real_snapshot_tolerance_days,
            target_strike=target_strike,
            target_tenor_days=float(tenor),
        )
        if real is not None and real["implied_vol"] and real["implied_vol"] > 0:
            # CE QU'ON RETIENT DU SNAPSHOT : son strike, son échéance et son IV
            # -- pas sa prime.
            #
            # La prime était reprise telle quelle, alors qu'elle a été cotée à
            # la date du snapshot (jusqu'à real_snapshot_tolerance_days avant
            # aujourd'hui) et donc à un AUTRE spot que celui auquel l'ordre
            # s'exécute. Deux effets, tous deux faux :
            #   - la position s'ouvrait à un prix sans rapport avec le
            #     sous-jacent du jour, et le repricing Black-Scholes du
            #     lendemain faisait apparaître un saut de P&L fantôme ;
            #   - le problème existe même à date égale, puisque
            #     08_recuperation_options.py peut associer une IV venue
            #     d'Alpha Vantage à une prime bid/ask venue d'IBKR : les deux
            #     ne sont pas cohérentes entre elles, et BS(spot, K, T, IV_AV)
            #     ne redonne pas prime_IBKR.
            # L'IV est la grandeur TRANSPORTABLE d'une date à l'autre, le prix
            # ne l'est pas : on garde donc l'information de marché (la
            # volatilité réellement cotée sur ce contrat) et on en dérive le
            # prix au spot d'exécution. Le repricing quotidien part ainsi
            # exactement du prix d'entrée, sans discontinuité.
            strike_real = float(real["strike"])
            expiry_real = real["expiry"]
            t_years = max((expiry_real - today).days, 0) / 365.0
            if t_years > 0:
                vol_real = float(real["implied_vol"])
                greeks = options_pricing.bs_greeks(
                    spot, strike_real, t_years, vol_real, option_type, r=r, q=q,
                )
                return {
                    "strike": strike_real, "expiry": expiry_real,
                    "premium": options_pricing.bs_price(
                        spot, strike_real, t_years, vol_real, option_type, r=r, q=q,
                    ),
                    "vol": vol_real, "delta": greeks["delta"],
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
        premium = options_pricing.bs_price(spot, strike, t_years, vol, option_type, r=r, q=q)
        greeks = options_pricing.bs_greeks(spot, strike, t_years, vol, option_type, r=r, q=q)
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

    def _position_greeks(self, pos: OptionPosition, today: pd.Timestamp, spot: float) -> Optional[dict]:
        """Greeks du contrat DÉTENU (son strike, son échéance, sa volatilité de
        repricing) au spot du jour -- et non ceux d'un contrat qu'on
        choisirait aujourd'hui."""
        t_years = max((pos.expiry - today).days, 0) / 365.0
        if t_years <= 0 or spot <= 0:
            return None
        r, q = self._pricing_context(pos.symbol, today)
        return options_pricing.bs_greeks(
            spot, pos.strike, t_years, self._repricing_vol(pos, today), pos.option_type, r=r, q=q,
        )

    def _position_delta(self, pos: OptionPosition, today: pd.Timestamp, spot: float) -> Optional[float]:
        greeks = self._position_greeks(pos, today, spot)
        return None if greeks is None else greeks["delta"]

    def _round_contracts(self, contracts: float) -> float:
        """Une option se négocie par contrats ENTIERS (cf.
        config.OPTIONS_WHOLE_CONTRACTS). Arrondi au plus proche : c'est la
        taille réalisable la plus proche de l'exposition visée. Une cible
        sous un demi-contrat donne 0, donc pas d'ordre -- une position trop
        petite pour un seul contrat n'est tout simplement pas prenable.

        Filet de sécurité : `contracts` est le résultat d'une division par une
        prime, et une prime Black-Scholes peut légitimement sous-couler vers
        un flottant SUBNORMAL non nul (cf. MIN_PREMIUM_FOR_SIZING) -- un appel
        qui n'aurait pas encore ce garde-fou en amont ferait déborder la
        division vers `inf`, que `int()` refuse de convertir (OverflowError).
        Point de passage unique avant tout `int()` : neutraliser ici couvre
        aussi un futur appelant qui l'oublierait."""
        if not self.whole_contracts:
            return contracts
        if not math.isfinite(contracts):
            return 0.0
        return float(int(contracts + 0.5)) if contracts > 0 else 0.0

    def _size_contracts(self, raw_contracts: float, premium_per_share: float, today: pd.Timestamp) -> float:
        """Taille d'ordre retenue : arrondi aux contrats entiers, puis
        remontée éventuelle à la taille où le minimum par ordre cesse de
        gaspiller (cf. config.OPTIONS_FEE_BUMP_MAX_EXTRA_PCT).

        L'écart de la remontée est mesuré par rapport à l'exposition VISÉE
        (`raw_contracts`, fractionnaire), pas à la taille déjà arrondie :
        c'est bien de la cible que l'on s'écarte, et arrondir puis comparer à
        l'arrondi exagérerait l'écart d'un demi-contrat."""
        contracts = self._round_contracts(raw_contracts)
        if contracts <= 0 or not self.whole_contracts or not self.fee_bump_max_extra_pct:
            return contracts

        rate = self._commission_rate(max(premium_per_share, 0.0), today)
        if rate <= 0:
            return contracts
        # Première taille où nb_contrats x taux atteint le minimum par ordre :
        # en dessous, on paie le minimum sans en avoir "pour son argent".
        efficient = math.ceil(self.commission_min_per_order / rate)
        if contracts >= efficient or raw_contracts <= 0:
            return contracts
        if (efficient - raw_contracts) / raw_contracts * 100 > self.fee_bump_max_extra_pct:
            return contracts
        return float(efficient)

    def _fee_ratio_ok(
        self, contracts: float, premium_per_share: float, multiplier: float,
        today: pd.Timestamp, side: str,
    ) -> bool:
        """Un ordre discrétionnaire (entrée/renforcement) dont les frais
        pèsent plus que config.OPTIONS_MAX_FEE_PCT_OF_TRADE de sa propre
        valeur détruit plus qu'il n'apporte : on ne le passe pas. Jamais
        appliqué à une SORTIE -- refuser de clôturer une position parce que
        les frais sont élevés reviendrait à désactiver le stop-loss sur les
        positions devenues quasi sans valeur, exactement là où il protège."""
        if not self.max_fee_pct_of_trade:
            return True
        trade_value = contracts * multiplier * max(premium_per_share, 0.0)
        if trade_value <= 0:
            return False
        fees = self._order_commission(contracts, premium_per_share, multiplier, today, side)
        return fees <= trade_value * self.max_fee_pct_of_trade / 100

    def _commission_rate(self, premium_per_share: float, today: pd.Timestamp) -> float:
        """Taux par contrat de la grille dégressive IBKR
        (config.OPTIONS_COMMISSION_TIERS) : dépend du VOLUME MENSUEL cumulé
        et du niveau de PRIME (une option quasi sans valeur est facturée
        0,25$/contrat au lieu de 0,65$).

        Simplification assumée : le palier de volume est celui atteint AVANT
        l'ordre, appliqué à l'ordre entier. IBKR facture par tranches
        (les 10 000 premiers contrats du mois au plein tarif, les suivants au
        tarif réduit) ; découper un même ordre entre deux tranches ne change
        rien tant que le volume mensuel reste loin du premier seuil, ce qui
        est le cas d'un portefeuille de cette taille."""
        volume = self._monthly_contracts.get((today.year, today.month), 0.0)
        for volume_max, premium_bands in self.commission_tiers:
            if volume_max is None or volume <= volume_max:
                for premium_max, rate in premium_bands:
                    if premium_max is None or premium_per_share < premium_max:
                        return rate
        return self.commission_per_contract  # grille vide/mal formée : repli

    def _order_commission(
        self, contracts: float, premium_per_share: float, multiplier: float,
        today: pd.Timestamp, side: str,
    ) -> float:
        """Coût total facturé sur UN ordre d'options US : commission IBKR
        (grille dégressive, jamais moins que le minimum PAR ORDRE) + frais
        tiers (ORF, CAT, compensation OCC des deux côtés ; TAF FINRA et frais
        de transaction SEC à la vente seulement).

        Le minimum étant PAR ORDRE et les frais tiers PAR CONTRAT, le coût
        n'est pas proportionnel au nombre de contrats : il ne peut donc pas
        être replié dans un prix unitaire avant de connaître la taille de
        l'ordre, contrairement à ce que faisait le moteur jusqu'ici."""
        contracts = abs(contracts)
        if contracts <= 0:
            return 0.0

        premium = max(premium_per_share, 0.0)
        # Le minimum par ordre porte sur la COMMISSION seule ; les frais tiers
        # s'ajoutent par-dessus (cf. "Frais tiers" dans la grille IBKR).
        commission = max(contracts * self._commission_rate(premium, today), self.commission_min_per_order)

        fees = contracts * (
            config.OPTIONS_FEE_ORF_PER_CONTRACT
            + config.OPTIONS_FEE_CAT_PER_CONTRACT
            + config.OPTIONS_FEE_OCC_PER_CONTRACT
        )
        if side == "sell":
            fees += contracts * config.OPTIONS_FEE_FINRA_TAF_PER_CONTRACT
            fees += contracts * multiplier * premium * config.OPTIONS_FEE_SEC_PCT_OF_SALE
        return commission + fees

    def _record_contract_volume(self, contracts: float, today: pd.Timestamp) -> None:
        """Volume mensuel cumulé, qui détermine le palier dégressif (IBKR
        raisonne en contrats négociés dans le MOIS, achats et ventes
        confondus)."""
        key = (today.year, today.month)
        self._monthly_contracts[key] = self._monthly_contracts.get(key, 0.0) + abs(contracts)

    def _slippage_amount(
        self, contracts: float, mid_premium: float, multiplier: float,
        slippage_rate: Optional[float] = None,
    ) -> float:
        """Écart payé au marché sur un fill, par rapport au mid : la prime est
        payée majorée à l'achat et encaissée minorée à la vente (cf.
        _open_or_resize / _reduce_position), donc le montant est le même des
        deux côtés et toujours positif.

        `slippage_rate` explicite pour les fills qui ne traversent PAS de
        fourchette -- un règlement à l'échéance, où le taux est nul."""
        rate = self.slippage_rate if slippage_rate is None else slippage_rate
        return abs(contracts) * multiplier * max(mid_premium, 0.0) * rate

    def _record_fill(
        self, symbol: str, side: str, contracts: float, price: float, cash_flow: float,
        today: pd.Timestamp, reason: str, option_type: str, multiplier: float,
        mid_premium: float, commission: float, slippage_rate: Optional[float] = None,
    ) -> None:
        """Enregistre un fill RÉELLEMENT EXÉCUTÉ : sa friction (cumuls
        total_commission / total_slippage) ET sa ligne au journal des
        exécutions, en un seul appel.

        Les deux sont volontairement indissociables : quand la friction se
        comptabilisait à part, le renfort de _deploy_idle_cash -- le chemin de
        code le plus actif du moteur, appelé chaque jour de bourse -- avait été
        oublié, et total_friction_dollar sous-estimait donc lourdement le coût
        réel sans que rien ne le signale. Un seul point d'entrée par fill rend
        cet oubli impossible.

        POURQUOI UN JOURNAL D'EXÉCUTIONS, alors que trades.parquet existe.
        Celui-ci n'enregistre que les VENTES, et présente chacune comme un
        aller-retour : `entry_date` y est la date de PREMIÈRE ouverture de la
        position et `entry_price` le prix de revient MOYEN de tous les achats
        qui l'ont constituée. Or une position se construit en plusieurs fois
        (renforcement au rebalancement, redéploiement du cash oisif) sans
        qu'aucun de ces achats n'apparaisse nulle part -- si bien qu'un trade
        peut légitimement solder 285 contrats alors que 94 seulement ont été
        achetés à son `entry_date`. La comptabilité du moteur est juste (aucun
        contrat n'est vendu sans avoir été acheté, cf. tests), mais elle était
        INVÉRIFIABLE depuis les sorties : ce journal la rend auditable ligne à
        ligne.

        `cash_flow` est signé : négatif pour un achat (décaissement), positif
        pour une vente (encaissement), frais et slippage déjà inclus des deux
        côtés -- sa somme sur tout le run vaut donc exactement la variation de
        cash due aux options.

        Appelé après _affordable et _cap_order_size (donc après toute
        réduction) et seulement là où le cash bouge réellement : un ordre
        abandonné ne coûte rien, et l'inclure gonflerait la friction d'ordres
        qui n'ont jamais eu lieu."""
        slippage = self._slippage_amount(contracts, mid_premium, multiplier, slippage_rate)
        self.total_commission += commission
        self.total_slippage += slippage
        self.executions.append({
            "date": today, "symbol": symbol, "side": side, "option_type": option_type,
            "contracts": contracts, "price": price, "cash_flow": cash_flow,
            "multiplier": multiplier, "reason": reason,
            "commission": commission, "slippage": slippage,
        })

    def _drop_order(self, reason: str) -> None:
        """Enregistre l'ABANDON pur et simple d'un ordre d'achat, par motif
        (cf. self.dropped_orders). Distinct d'une réduction : ici rien n'est
        exécuté du tout."""
        self.dropped_orders[reason] = self.dropped_orders.get(reason, 0) + 1

    def _max_trade_dollar_at(self, today: pd.Timestamp) -> Optional[float]:
        """Plafond par ordre applicable aujourd'hui : le plus contraignant du
        plafond relatif (% du NAV) et du plafond absolu, ceux qui sont actifs.
        None si aucun des deux ne l'est."""
        caps = []
        if self.max_trade_pct_of_nav is not None:
            caps.append(max(self._current_nav(today), 0.0) * self.max_trade_pct_of_nav / 100)
        if self.max_trade_dollar is not None:
            caps.append(self.max_trade_dollar)
        return min(caps) if caps else None

    def _cap_order_size(self, contracts: float, cost_of, today: pd.Timestamp) -> float:
        """Ramène un ACHAT au plafond par ordre (config.OPTIONS_MAX_TRADE_PCT_OF_NAV
        et/ou config.OPTIONS_MAX_TRADE_DOLLAR).

        Distinct de _affordable, qui borne au cash disponible : ici le capital
        existe, on refuse simplement de le concentrer sur un seul ordre. Même
        forme d'appel (`cost_of` est une fonction, pas un montant) parce que la
        commission a un minimum par ordre et n'est donc pas proportionnelle à
        la taille -- réduire de moitié ne divise pas le coût par deux."""
        cap = self._max_trade_dollar_at(today)
        if cap is None or contracts <= 0:
            return contracts
        if cost_of(contracts) <= cap:
            return contracts

        self.capped_orders_count += 1
        scaled = contracts * (cap / cost_of(contracts))
        if self.whole_contracts:
            scaled = float(int(scaled))
        for _ in range(64):
            if scaled <= 0:
                return 0.0
            if cost_of(scaled) <= cap:
                return scaled
            scaled = scaled - 1.0 if self.whole_contracts else scaled * 0.95
        return 0.0

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
        self.buy_orders_count += 1
        cost = cost_of(contracts)
        if cost <= self.cash:
            return contracts, cost
        # Même diagnostic que le moteur actions (cf. engine.execution_diagnostics) :
        # le budget vient du NAV net des positions gelées, qui restent pourtant
        # détenues -- le cash disponible est donc souvent inférieur.
        self.truncated_orders_count += 1
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
        if existing is not None and option_type is not None and existing.option_type != option_type:
            # Le signal a changé de sens (call <-> put) alors qu'une position
            # existe déjà dans l'autre sens : on laisse l'ancienne gelée
            # (elle ne sera fermée que par stop/take-profit/expiration/gap de
            # données, cf. docstring module) et on n'ouvre PAS de position
            # simultanée dans les deux sens sur le même sous-jacent.
            logger.debug("%s : signal %s ignoré, une position %s existe déjà (gelée).", symbol, option_type, existing.option_type)
            self._drop_order("sens_oppose_deja_detenu")
            return
        if option_type is None:
            option_type = existing.option_type if existing is not None else None
            if option_type is None:
                self._drop_order("sens_inconnu")
                return

        # Écart en log au moment de CE trade (V = dernière valorisation
        # théorique connue, P = spot d'EXÉCUTION) : référence posée pour le
        # filtre ε du prochain dépôt SEC sur cette position (cf.
        # last_rebalance_log_gap, _rebalance_on_signals). Calculé une fois
        # ici et appliqué à chaque point où un trade RÉEL se conclut plus
        # bas -- jamais sur une tentative avortée par les gardes ci-dessous.
        new_log_gap = self._log_gap_at(symbol, spot)

        contract = self._select_contract(
            symbol, option_type, spot, today,
            strike_reference_price=strike_reference_price, tenor_days=tenor_days,
        )
        delta = contract["delta"] or 0.0
        if abs(delta) < 1e-6:
            self._drop_order("delta_negligeable")
            return

        multiplier = contract["multiplier"]
        # Budget de levier ENCORE DISPONIBLE, appliqué ici et pas seulement
        # dans _deploy_idle_cash : le plafond ne contraignait jusqu'ici aucun
        # ordre issu du rebalancement, c'est-à-dire le chemin de
        # dimensionnement principal (cf. config.OPTIONS_MAX_DELTA_NOTIONAL_PCT).
        delta_budget = self._delta_budget(today)
        remaining_delta = (
            None if delta_budget is None else delta_budget - self._delta_notional(today)
        )

        if existing is None:
            # Prime brute (slippage inclus) : la commission est ajoutée à part,
            # au niveau de l'ORDRE, puisqu'elle a un minimum forfaitaire.
            gross_premium = contract["premium"] * (1 + self.slippage_rate)
            target_contracts = self._size_contracts(
                target_dollar / (abs(delta) * spot * multiplier), gross_premium, today,
            )
            if remaining_delta is not None:
                target_contracts = min(
                    target_contracts,
                    self._contracts_under_remaining_delta(delta, spot, multiplier, remaining_delta),
                )
                if target_contracts <= 0:
                    self._drop_order("plafond_de_levier")
                    return
            if target_contracts <= 0:
                self._drop_order("taille_nulle")
                return
            if not self._fee_ratio_ok(target_contracts, gross_premium, multiplier, today, "buy"):
                self._drop_order("frais_excessifs")
                return

            def cost_of(n: float) -> float:
                return n * multiplier * gross_premium + self._order_commission(n, gross_premium, multiplier, today, "buy")

            if cost_of(target_contracts) < MIN_TRADE_DOLLAR:
                self._drop_order("montant_negligeable")
                return
            # Plafond par ordre AVANT la contrainte de cash : le plafond est
            # une règle de gestion, le cash une limite matérielle -- appliquer
            # le second en premier laisserait un ordre au-dessus du plafond
            # dès que le cash est abondant.
            if reason not in UNCAPPED_BUY_REASONS:
                target_contracts = self._cap_order_size(target_contracts, cost_of, today)
            if target_contracts <= 0:
                self._drop_order("plafond_par_ordre")
                return
            target_contracts, cost = self._affordable(target_contracts, cost_of)
            if target_contracts <= 0 or cost < MIN_TRADE_DOLLAR:
                self._drop_order("cash_insuffisant")
                return
            self.cash -= cost
            self._record_contract_volume(target_contracts, today)
            # Prix de revient TOUT COMPRIS par action (commission d'ordre
            # répartie sur les contrats effectivement achetés) : conserve la
            # sémantique de OptionPosition.entry_premium et donc le return_pct
            # des trades.
            effective_premium = cost / (target_contracts * multiplier)
            self._record_fill(
                symbol, "buy", target_contracts, effective_premium, -cost, today,
                reason, option_type, multiplier,
                mid_premium=contract["premium"],
                commission=cost - target_contracts * multiplier * gross_premium,
            )
            self.positions[symbol] = OptionPosition(
                symbol=symbol, option_type=option_type, strike=contract["strike"], expiry=contract["expiry"],
                contracts=target_contracts, entry_premium=effective_premium, entry_date=today,
                vol=contract["vol"], multiplier=multiplier, source=contract["source"],
                # EXPOSITION RÉELLEMENT ÉTABLIE, et non l'exposition demandée.
                # target_dollar est rejoué tel quel au roulement (cf.
                # _roll_position) : y stocker la cible alors que l'ordre a été
                # rogné -- par le plafond par ordre, par le cash, par le
                # plafond de levier -- ferait BONDIR la position au premier
                # roulement, qui se retrouverait à établir d'un coup ce que
                # l'ouverture n'avait pas pu prendre. "À exposition inchangée"
                # ne peut vouloir dire qu'une chose : la même que celle
                # effectivement détenue.
                entry_spot=spot,
                target_dollar=abs(delta) * spot * multiplier * target_contracts,
                # Références de stop posées ICI et nulle part ailleurs.
                stop_reference_premium=effective_premium, stop_reference_spot=spot,
                strike_reference_price=strike_reference_price, tenor_days=tenor_days,
                vol_ratio=self._entry_vol_ratio(symbol, today, contract["vol"]),
                last_rebalance_log_gap=new_log_gap, open_reason=reason,
            )
            self._touch_positions()
        else:
            # target_dollar n'est PAS mis à jour ici : il l'est plus bas, à
            # partir du nombre de contrats réellement détenu après le fill
            # (même raison que pour l'ouverture -- c'est l'exposition établie
            # qui est rejouée au roulement, pas celle qui avait été demandée).
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
                self._drop_order("delta_negligeable")
                return
            target_contracts = self._size_contracts(
                target_dollar / (abs(existing_delta) * spot * multiplier),
                self._current_premium(existing, today, spot=spot) or contract["premium"], today,
            )
            # Le plafond de levier borne la CIBLE, pas seulement l'incrément :
            # sinon un renforcement pourrait porter la position au-dessus du
            # plafond dès lors que l'incrément, lui, reste sous le budget.
            if remaining_delta is not None:
                room = existing.contracts + self._contracts_under_remaining_delta(
                    existing_delta, spot, multiplier, remaining_delta,
                )
                target_contracts = min(target_contracts, room)
            delta_contracts = target_contracts - existing.contracts

            # Second filet de churn, APRÈS le filtre ε qui a déjà décidé
            # qu'un redimensionnement était justifié (cf.
            # _rebalance_on_signals) : un changement de conviction réel peut
            # encore ne se traduire que par un micro-ajustement de contrats
            # (NAV qui a un peu bougé). MIN_TRADE_DOLLAR (1$) ne bloque que
            # les montants absolument négligeables, pas ces resizes
            # proportionnellement mineurs. Comparaison en NOMBRE DE CONTRATS
            # (même unité des deux côtés) plutôt qu'en dollars : target_dollar
            # est une exposition
            # NOTIONNELLE delta-équivalente, alors que la valeur de marché de
            # la position (prime x contrats) en est une fraction systématique
            # (effet de levier) -- comparer les deux directement aurait rendu
            # le seuil inopérant (l'écart de levier domine toujours l'écart
            # réel de resize, quelle que soit sa taille).
            if self.min_resize_relative_pct and existing.contracts > 0:
                if abs(delta_contracts) / existing.contracts < self.min_resize_relative_pct / 100:
                    self._drop_order("resize_sous_le_seuil")
                    return

            if abs(delta_contracts) * multiplier * spot < MIN_TRADE_DOLLAR:
                self._drop_order("montant_negligeable")
                return
            if delta_contracts > 0:
                current_premium = self._current_premium(existing, today, spot=spot) or contract["premium"]
                gross_premium = current_premium * (1 + self.slippage_rate)

                def cost_of(n: float) -> float:
                    return n * multiplier * gross_premium + self._order_commission(n, gross_premium, multiplier, today, "buy")

                if not self._fee_ratio_ok(delta_contracts, gross_premium, multiplier, today, "buy"):
                    self._drop_order("frais_excessifs")
                    return
                if reason not in UNCAPPED_BUY_REASONS:
                    delta_contracts = self._cap_order_size(delta_contracts, cost_of, today)
                if delta_contracts <= 0:
                    self._drop_order("plafond_par_ordre")
                    return
                delta_contracts, cost = self._affordable(delta_contracts, cost_of)
                if delta_contracts <= 0 or cost < MIN_TRADE_DOLLAR:
                    self._drop_order("cash_insuffisant")
                    return
                self.cash -= cost
                self._record_contract_volume(delta_contracts, today)
                effective_premium = cost / (delta_contracts * multiplier)
                new_contracts = existing.contracts + delta_contracts
                self._record_fill(
                    symbol, "buy", delta_contracts, effective_premium, -cost, today,
                    reason, option_type, multiplier,
                    mid_premium=current_premium,
                    commission=cost - delta_contracts * multiplier * gross_premium,
                )
                # Idem : le prix de revient suit le renfort, le seuil de stop
                # non (cf. OptionPosition.stop_reference_premium).
                existing.entry_premium = (existing.entry_premium * existing.contracts + effective_premium * delta_contracts) / new_contracts
                existing.contracts = new_contracts
                existing.target_dollar = abs(existing_delta) * spot * multiplier * new_contracts
                self._touch_positions()
                if new_log_gap is not None:
                    existing.last_rebalance_log_gap = new_log_gap
            else:
                self._reduce_position(existing, -delta_contracts, today, reason, spot=spot)
                # Réduction PARTIELLE (la position peut avoir été fermée en
                # entier par _reduce_position, auquel cas il n'y a plus rien
                # à mettre à jour) : même référence que la branche renfort.
                if symbol in self.positions:
                    remaining = self.positions[symbol]
                    remaining.target_dollar = (
                        abs(existing_delta) * spot * multiplier * remaining.contracts
                    )
                    if new_log_gap is not None:
                        remaining.last_rebalance_log_gap = new_log_gap

    def _reduce_position(self, pos: OptionPosition, contracts_to_sell: float, today: pd.Timestamp, reason: str, spot: Optional[float] = None) -> None:
        contracts_to_sell = min(contracts_to_sell, pos.contracts)
        if contracts_to_sell <= 0:
            return
        current_premium = self._current_premium(pos, today, spot=spot)
        if current_premium is None:
            return
        # À l'ÉCHÉANCE, le contrat est réglé à sa valeur intrinsèque par
        # exercice ou abandon : il n'y a pas de fourchette bid/ask à traverser,
        # donc pas de slippage. L'appliquer quand même faisait payer 2,5% de
        # l'intrinsèque sur chaque expiration dans la monnaie, une friction qui
        # n'existe pas.
        slippage_rate = 0.0 if reason == "expiry" else self.slippage_rate
        gross_premium = current_premium * (1 - slippage_rate)
        gross_value = contracts_to_sell * pos.multiplier * gross_premium
        commission = self._order_commission(contracts_to_sell, gross_premium, pos.multiplier, today, "sell")

        # Quand la commission dépasse ce que vaut encore l'option (prime quasi
        # nulle), on ne PAIE pas pour vendre : on laisse expirer. Le
        # raisonnement est juste, mais l'ancien `max(gross_value - commission,
        # 0.0)` en tirait la mauvaise conclusion -- il sortait la position à
        # zéro tout en ne payant jamais la commission, c'est-à-dire qu'il
        # comptabilisait une vente qui n'avait pas lieu.
        #
        # La position reste donc OUVERTE jusqu'à son échéance. Deux exceptions,
        # où il n'y a rien à attendre : l'expiration elle-même (le contrat est
        # abandonné sans frais, produit nul) et la disparition des données du
        # sous-jacent (aucune cotation future à espérer).
        if gross_value <= commission and reason not in FORCED_EXIT_REASONS:
            logger.debug(
                "%s : vente non passée (%s) -- valeur résiduelle %.2f$ inférieure aux frais "
                "%.2f$. Position conservée jusqu'à l'échéance du %s.",
                pos.symbol, reason, gross_value, commission, pos.expiry.date(),
            )
            return

        proceeds = max(gross_value - commission, 0.0)
        # FRICTION RÉELLEMENT DÉCAISSÉE. Quand les frais dépassent la valeur
        # résiduelle -- le cas d'une expiration hors de la monnaie, où
        # gross_value vaut 0 -- le produit est plancher à zéro et la
        # commission n'est donc PAS payée : le contrat est abandonné, pas
        # vendu. La comptabiliser en entier gonflait total_friction_dollar,
        # c'est-à-dire précisément le chiffre censé arbitrer "thèse contre
        # friction" dans 13_diagnostic_friction.py.
        commission_paid = min(commission, gross_value)
        effective_premium = proceeds / (contracts_to_sell * pos.multiplier)
        self.cash += proceeds
        self._record_contract_volume(contracts_to_sell, today)
        self._record_fill(
            pos.symbol, "sell", contracts_to_sell, effective_premium, proceeds, today,
            reason, pos.option_type, pos.multiplier,
            mid_premium=current_premium, commission=commission_paid,
            slippage_rate=slippage_rate,
        )
        pnl = (effective_premium - pos.entry_premium) * contracts_to_sell * pos.multiplier
        self.trades.append({
            "symbol": pos.symbol, "entry_date": pos.entry_date, "exit_date": today,
            "shares": contracts_to_sell * pos.multiplier, "entry_price": pos.entry_premium, "exit_price": effective_premium,
            "pnl": pnl, "return_pct": (effective_premium - pos.entry_premium) / pos.entry_premium * 100,
            "holding_days": (today - pos.entry_date).days, "exit_reason": reason,
            "option_type": pos.option_type, "strike": pos.strike, "contracts": contracts_to_sell, "source": pos.source,
            # D'où venait la position (cf. OptionPosition.open_reason) :
            # distingue un trade issu du mécanisme journalier (ouverture sur
            # mouvement de cours pur) d'un trade issu d'un dépôt SEC, deux
            # degrés d'information très différents que exit_reason seul ne dit
            # pas (il décrit la SORTIE, pas l'origine de la position sortie).
            "open_reason": pos.open_reason,
        })
        pos.contracts -= contracts_to_sell
        if pos.contracts <= 1e-9:
            del self.positions[pos.symbol]
        self._touch_positions()

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
            reference = pos.stop_reference_premium or pos.entry_premium
            if premium is None or not reference:
                return None
            return (premium - reference) / reference * 100

        close = self.prices.close_at(pos.symbol, today)
        reference = pos.stop_reference_spot or pos.entry_spot
        if close is None or not reference:
            return None
        move_pct = (close - reference) / reference * 100
        return move_pct if pos.option_type == "CALL" else -move_pct

    def _convergence_fraction(self, pos: OptionPosition, today: pd.Timestamp) -> Optional[float]:
        """Part du chemin parcourue vers la valeur théorique depuis l'entrée :
        0 = rien, 1 = convergence complète. None si la position ne vise pas de
        convergence (contrat ATM, aucun strike_reference_price) ou si l'écart
        d'entrée est dégénéré.

        Naturellement orientée, sans distinction call/put : les deux termes du
        rapport changent de signe ensemble. Pour un put, la valeur théorique
        est SOUS le cours d'entrée (écart négatif) et une baisse du cours donne
        un numérateur négatif -- le rapport reste positif et croît vers 1."""
        if not self.take_profit_convergence_fraction or pos.strike_reference_price is None:
            return None
        entry_spot = pos.stop_reference_spot or pos.entry_spot
        if not entry_spot:
            return None
        gap_at_entry = pos.strike_reference_price - entry_spot
        # Écart quasi nul : la convergence n'a plus de sens (et le rapport
        # exploserait). Le seuil d'entrée l'exclut en pratique, mais un
        # roulement repose la référence sur une théorique rafraîchie qui peut,
        # elle, avoir rejoint le cours.
        if abs(gap_at_entry) < 1e-9 * max(abs(entry_spot), 1.0):
            return None
        close = self.prices.close_at(pos.symbol, today)
        if close is None:
            return None
        return (close - entry_spot) / gap_at_entry

    def _check_stop_loss_take_profit(self, today: pd.Timestamp) -> set[str]:
        """Décidé à la clôture de today, exécuté à l'ouverture du jour
        suivant (mis en file via pending_orders, comme un rebalancement) --
        même règle "pas de look-ahead" que le reste du moteur : on ne
        clôture jamais une position au prix exact qui vient de déclencher
        le seuil.

        Le STOP-LOSS se mesure toujours en variation (cf. _position_move_pct).
        Le TAKE-PROFIT, lui, se mesure en fraction de convergence dès que la
        position vise une valeur théorique : un seuil fixe en % n'est pas
        atteignable de la même façon des deux côtés, si bien que la jambe put
        ne prenait pratiquement jamais ses gains (cf.
        config.OPTIONS_TAKE_PROFIT_CONVERGENCE_FRACTION)."""
        triggered = set()
        for symbol, pos in list(self.positions.items()):
            move_pct = self._position_move_pct(pos, today)
            if move_pct is not None and move_pct <= self.stop_loss_pct:
                self.pending_orders[symbol] = _PendingOrder(0.0, "stop_loss", today)
                triggered.add(symbol)
                continue

            convergence = self._convergence_fraction(pos, today)
            if convergence is not None:
                if convergence >= self.take_profit_convergence_fraction:
                    self.pending_orders[symbol] = _PendingOrder(0.0, "take_profit", today)
                    triggered.add(symbol)
            elif move_pct is not None and move_pct >= self.take_profit_pct:
                self.pending_orders[symbol] = _PendingOrder(0.0, "take_profit", today)
                triggered.add(symbol)
        return triggered

    def _check_rolls(self, today: pd.Timestamp, exclude: set[str]) -> set[str]:
        """Réexamine toute position arrivant à moins de roll_when_days_left de
        son échéance : encore justifiée par son signal -> roulement sur une
        échéance pleine ; plus justifiée -> clôture. Comme les stops, décidé à
        la clôture de J et exécuté à l'ouverture de J+1.

        La clôture n'est pas redondante avec exit_when_signal_lost : pour une
        stratégie qui gèle ses positions (l'option n'est pas activée), c'est
        ici, et seulement ici, que l'écart refermé est constaté -- sinon un
        contrat que plus rien ne justifie serait porté jusqu'à son expiration,
        soit toute la dernière année où la valeur temps s'érode le plus vite."""
        if not self.roll_when_days_left or not self.positions:
            return set()

        decided = set()
        for symbol, pos in self.positions.items():
            if symbol in exclude or symbol in self.pending_orders:
                continue
            if (pos.expiry - today).days > self.roll_when_days_left:
                continue
            # _eligible_directions vaut None tant qu'aucun rebalancement n'a eu
            # lieu : sans avis de la stratégie, on renouvelle plutôt que de
            # liquider sur une absence d'information.
            still_wanted = (
                self._eligible_directions is None
                or self._eligible_directions.get(symbol) == pos.option_type
            )
            if still_wanted:
                self.pending_orders[symbol] = _PendingOrder(pos.target_dollar, "roll", today, roll=True)
            else:
                self.pending_orders[symbol] = _PendingOrder(0.0, "signal_lost", today)
            decided.add(symbol)
        return decided

    def _check_delever(self, today: pd.Timestamp, exclude: set[str]) -> set[str]:
        """Ramène le portefeuille sous le plafond de levier
        (config.OPTIONS_MAX_DELTA_NOTIONAL_PCT) en réduisant TOUTES les
        positions au même prorata. Décidé à la clôture de J, exécuté à
        l'ouverture de J+1, comme les stops.

        POURQUOI IL A FALLU L'AJOUTER. Le plafond n'était vérifié que dans
        _deploy_idle_cash, donc uniquement au moment d'un renfort de cash
        oisif. Rien ne le vérifiait à l'ouverture d'une position, et surtout
        rien ne le vérifiait APRÈS COUP : le delta d'un contrat dérive avec le
        sous-jacent (gamma), et le moteur ne vendait jamais pour se
        désendetter. Mesuré sur un run : 281% de delta notionnel pour un
        plafond déclaré à 100%, sans qu'aucun mécanisme ne cherche à le
        ramener.

        RÉDUCTION AU PRORATA, et non liquidation des plus grosses lignes : le
        plafond est une contrainte de levier global, il n'exprime aucune
        opinion sur laquelle des thèses mérite d'être abandonnée. Réduire tout
        le monde du même facteur préserve la hiérarchie décidée par la
        stratégie.

        La bande de tolérance (config.OPTIONS_DELEVER_TOLERANCE_PCT) évite de
        vendre quelques contrats à chaque oscillation du marché autour du
        plafond -- exactement le churn que la refonte du rebalancement a
        supprimé ailleurs."""
        if not self.max_delta_notional_pct or not self.positions:
            return set()

        nav = self._current_nav(today)
        budget = self._delta_budget(today, nav=nav)
        if not budget or budget <= 0:
            return set()

        current = self._delta_notional(today)
        if current <= budget * (1 + self.delever_tolerance_pct / 100):
            return set()

        # On ramène à 100% du plafond, pas au bord de la bande de tolérance :
        # sinon la position suivante repasse au-dessus au premier mouvement.
        keep_ratio = budget / current
        decided = set()
        for symbol, pos in self.positions.items():
            if symbol in exclude or symbol in self.pending_orders:
                continue
            target = self._round_contracts(pos.contracts * keep_ratio)
            to_sell = pos.contracts - target
            if to_sell <= 0:
                continue
            self.pending_orders[symbol] = _PendingOrder(
                0.0, "delever", today, sell_contracts=to_sell,
            )
            decided.add(symbol)

        if decided:
            logger.debug(
                "%s : dé-levier -- delta notionnel %.0f%% du NAV (plafond %.0f%%), "
                "%d positions réduites de %.0f%%.",
                today.date(), current / nav * 100 if nav > 0 else float("nan"),
                self.max_delta_notional_pct, len(decided), (1 - keep_ratio) * 100,
            )
            self.delever_events_count += 1
        return decided

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
            # Valorisée à `last_valid` (dernier cours réel connu), mais DATÉE
            # de `today` -- l'alignement sur backtest/engine.py, qui fait déjà
            # _execute_trade(..., today, "data_gap") au dernier prix connu.
            #
            # L'appel d'origine passait `last_valid` comme date du trade : la
            # sortie et holding_days étaient datées du dernier cours connu (des
            # jours, parfois des semaines, avant la journée simulée) et surtout
            # le cash était crédité à une date ANTÉRIEURE à celle où le moteur
            # se trouve -- une position pouvait ainsi financer des achats déjà
            # passés dans la simulation.
            spot = self.prices.close_at(symbol, last_valid)
            self._close_position(symbol, today, "data_gap", spot=spot)
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

    def _signal_row_for_rebalance(self, sym: str, s: dict, today: pd.Timestamp) -> dict:
        """La ligne de signal telle quelle, sauf en daily_rebalance : le
        `close` y est alors remplacé par le cours du JOUR (au lieu du cours
        figé à la date du dernier dépôt), pour que l'écart de valorisation
        reflète le mouvement du titre depuis la dernière publication -- sinon
        réévaluer tous les jours ne changerait rien (cf. docstring du
        module). La valorisation théorique, elle, n'est pas touchée : elle ne
        bouge qu'au prochain signal, comme avant."""
        if not self.daily_rebalance:
            return s
        close_today = self.prices.close_at(sym, today)
        if close_today is None or close_today <= 0:
            return s
        row = {**s, "close": close_today}
        # gap_pct SUIT le cours, lui aussi. Il ne le suivait pas : seul `close`
        # était rafraîchi, si bien que tout ce qui s'appuie sur gap_pct
        # raisonnait encore sur la photographie de la date de dépôt. Deux
        # conséquences, l'une visible et l'autre non :
        #   - le filtre momentum (_momentum_ok) s'ORIENTE sur le signe de
        #     gap_pct. Une entreprise sous-évaluée au dépôt dont le titre
        #     s'envole devient survalorisée : la stratégie veut un PUT, mais le
        #     filtre appliquait encore la règle CALL ("écarter si le titre
        #     chute") et laissait passer. Le garde-fou "ne pas vendre à
        #     découvert une fusée" ne se déclenchait donc jamais dans le seul
        #     cas où il compte ;
        #   - valuation_gap_options lit gap_pct DIRECTEMENT (contrairement à
        #     valuation_gap_multiples_options, qui recalcule tout depuis
        #     valuation_theoretical_per_share et close) : en réévaluation
        #     quotidienne, elle voyait donc un écart figé et le mode
        #     daily_rebalance n'avait aucun effet sur elle.
        # La valorisation théorique, elle, ne bouge toujours qu'au prochain
        # dépôt : ce n'est pas une fuite d'information future.
        theoretical = s.get("valuation_theoretical_per_share")
        if theoretical is not None and pd.notna(theoretical):
            row["gap_pct"] = (float(theoretical) - close_today) / close_today * 100
        return row

    def _current_targets(self, today: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, dict], dict[str, str]]:
        """(eligible_signals, targets, eligible_directions) au jour `today` --
        même construction de l'univers éligible que l'ancien _rebalance
        (comportement historique inchangé), factorisée pour être partagée par
        les deux mécanismes de rebalancement (journalier / sur dépôt SEC, cf.
        la docstring du module).

        `targets[symbole]["weight"]` EST DÉJÀ le poids ISOLÉ conviction du
        symbole / somme des convictions de TOUS les éligibles du jour : c'est
        ce que calcule base.capped_weights, appliqué par la stratégie sur
        l'ensemble `eligible_signals` (nouveaux candidats ET positions
        détenues confondus). Aucune renormalisation supplémentaire n'a donc
        lieu ici, contrairement à l'ancien _rebalance qui renormalisait une
        SECONDE fois sur le seul sous-ensemble effectivement redimensionné ce
        jour-là -- une étape propre au modèle "tout recalculer", sans objet
        ici puisqu'on ne touche jamais qu'un symbole à la fois (cf.
        _queue_isolated_order)."""
        universe_today = self.universe.asof(today)
        rows = []
        for sym, s in self.known_signals.items():
            row = self._signal_row_for_rebalance(sym, s, today)
            if sym in self.positions:
                # UNE POSITION DÉJÀ OUVERTE PASSE TOUJOURS. Le test
                # d'appartenance à l'indice portait auparavant sur TOUTES les
                # lignes, positions comprises : un titre sorti du S&P 500
                # disparaissait donc des éligibles, donc de
                # eligible_directions, et _close_lost_signals le liquidait
                # sous le motif "signal_lost" -- alors que son écart de
                # valorisation était inchangé, que le README annonce
                # explicitement le contraire, et que le motif enregistré
                # faussait exits_by_reason. Le comportement dépendait même
                # d'un AUTRE symbole : si plus rien n'était éligible ce
                # jour-là, eligible_signals était vide, le rebalancement
                # sortait tôt et la position survivait. L'appartenance à
                # l'indice est un filtre d'ENTRÉE, et rien d'autre.
                rows.append(row)
                continue
            if sym not in universe_today:
                continue
            if (today - s["published_date"]).days > data_loader.signal_max_age_for(
                s, self.signal_max_age_days
            ):
                continue
            # Cf. backtest/engine.py : un 8-K matériel (04c) déposé depuis la
            # publication du signal le rend périmé.
            if self.material_events.has_event_between(sym, s["published_date"], today):
                continue
            # Momentum évalué sur la ligne RAFRAÎCHIE, dont le gap_pct suit le
            # cours du jour (cf. _signal_row_for_rebalance) : c'est le signe de
            # cet écart qui oriente le filtre.
            if not self._momentum_ok(row, today):
                continue
            rows.append(row)

        eligible_signals = pd.DataFrame(rows)
        if eligible_signals.empty:
            return eligible_signals, {}, {}

        current_option_types = {sym: pos.option_type for sym, pos in self.positions.items()}
        if self.exit_when_signal_lost or self.roll_when_days_left:
            # UN SEUL passage : les cibles et les sens encore justifiés
            # partagent tout le pipeline de sélection de la stratégie, et le
            # demander en deux appels le faisait tourner deux fois à
            # l'identique -- 20% du temps de run total, pour rien.
            targets, eligible_directions = self.strategy.evaluate(
                eligible_signals, current_option_types,
            )
        else:
            targets = self.strategy.generate_option_targets(eligible_signals, current_option_types)
            eligible_directions = {}
        targets = {s: t for s, t in targets.items() if t["weight"] > 0}

        return eligible_signals, targets, eligible_directions

    def _log_gap_at(self, symbol: str, price: Optional[float]) -> Optional[float]:
        """log(V/P) au sens de la refonte du rebalancement (voir la
        docstring du module) : V = dernière valorisation théorique CONNUE
        (known_signals), P = le prix fourni par l'appelant.

        GÉNÉRIQUE : ne dépend PAS de la convention gap_basis interne de la
        stratégie ("log", "theoretical", "close") -- c'est une métrique de
        CHURN commune à toute stratégie options exposant
        valuation_theoretical_per_share dans son signal (les deux stratégies
        du dépôt aujourd'hui), pas un signal de sélection. None si V ou P est
        indisponible ou non positif : le filtre ε échoue alors ouvert
        (redimensionne) plutôt que de geler silencieusement une position sans
        pouvoir jamais mesurer son churn -- voir _rebalance_on_signals."""
        if price is None or price <= 0:
            return None
        signal = self.known_signals.get(symbol)
        if not signal:
            return None
        theoretical = signal.get("valuation_theoretical_per_share")
        if theoretical is None or pd.isna(theoretical) or theoretical <= 0:
            return None
        return math.log(float(theoretical) / price)

    def _queue_isolated_order(self, symbol: str, target: dict, today: pd.Timestamp, reason: str) -> None:
        """Met en file un ordre sur CE SEUL symbole -- jamais de
        redimensionnement en cascade sur les autres positions détenues,
        contrairement à l'ancien _rebalance (voir la docstring du module).

        `target["weight"]` est déjà le poids ISOLÉ (cf. _current_targets).
        Le budget actif est le NAV net de la valeur de TOUTES LES AUTRES
        positions détenues, gelées par construction : même notion de
        "capital disponible pour la ligne qu'on s'apprête réellement à
        bouger" que l'ancien _rebalance, restreinte ici à un singleton
        plutôt qu'à l'ensemble des cibles du jour."""
        nav_now = self._current_nav(today)
        # Valeur des AUTRES positions = total investi moins celle-ci, plutôt
        # qu'une somme refaite position par position à CHAQUE ordre mis en
        # file (cf. _invested_value).
        own = self.positions.get(symbol)
        legacy_value = self._invested_value(today) - (
            self._position_value(own, today) if own is not None else 0.0
        )
        active_budget = max(nav_now - legacy_value, 0.0)
        self._pending_spec[symbol] = {
            "option_type": target["option_type"],
            "strike_reference_price": target.get("strike_reference_price"),
            "tenor_days": target.get("tenor_days"),
        }
        self.pending_orders[symbol] = _PendingOrder(target["weight"] * active_budget, reason, today)

    def _rebalance_daily(self, today: pd.Timestamp, exclude: set[str]) -> set[str]:
        """MÉCANISME JOURNALIER (daily_rebalance=True, jours SANS nouveau
        dépôt SEC) : uniquement des OUVERTURES de symboles nouvellement
        éligibles. Aucun redimensionnement sur une position déjà détenue --
        c'est le mécanisme sur dépôt (_rebalance_on_signals) qui s'en charge,
        filtré par rebalance_log_gap_threshold. Voir la docstring du module.

        exit_when_signal_lost reste actif ici : une sortie sur perte de
        signal n'est pas un redimensionnement, c'est une décision
        d'exposition orthogonale à ce que cette refonte contraint."""
        eligible_signals, targets, eligible_directions = self._current_targets(today)
        if eligible_signals.empty:
            return set()
        self._eligible_directions = eligible_directions

        closed = self._close_lost_signals(today, exclude) if self.exit_when_signal_lost else set()
        exclude = exclude | closed

        for symbol, target in targets.items():
            if symbol in exclude or symbol in self.positions:
                continue  # déjà détenu : le mécanisme journalier n'ouvre que du NEUF
            self._queue_isolated_order(symbol, target, today, reason="rebalance_daily")

        return closed

    def _rebalance_on_signals(self, filed_symbols: set[str], today: pd.Timestamp, exclude: set[str]) -> set[str]:
        """MÉCANISME SUR DÉPÔT SEC (10-K/10-Q/8-K) : n'agit QUE sur les
        symboles dont known_signals vient d'être mis à jour aujourd'hui
        (potentiellement plusieurs le même jour). eligible_signals/targets/
        eligible_directions sont calculés UNE SEULE FOIS pour tous (même
        photographie de marché, cf. _current_targets), _close_lost_signals de
        même -- seule la décision d'ouverture/redimensionnement est ensuite
        scopée symbole par symbole : le dépôt de X ne génère jamais d'ordre
        sur A, B, C... même si la renormalisation globale les aurait
        affectés. Voir la docstring du module.

        NOUVEAU CANDIDAT (symbole pas encore détenu) : ouvert SANS test ε, il
        n'y a pas de trade antérieur auquel comparer.

        POSITION DÉJÀ DÉTENUE : redimensionnée seulement si l'écart en log a
        bougé de plus de rebalance_log_gap_threshold DEPUIS LE DERNIER TRADE
        RÉEL sur cette position (last_rebalance_log_gap) -- pas depuis le
        dernier signal connu. En dessous du seuil, known_signals est déjà à
        jour (fait par _update_known_signals, appelé avant ce mécanisme) mais
        last_rebalance_log_gap NE BOUGE PAS : le seuil suivant continue de se
        mesurer depuis ce dernier trade, pas depuis ce signal ignoré -- sans
        quoi une dérive lente qui ne franchit jamais ε d'un coup mais
        s'accumule sur plusieurs dépôts successifs ne se rattraperait
        jamais."""
        eligible_signals, targets, eligible_directions = self._current_targets(today)
        if eligible_signals.empty:
            return set()
        self._eligible_directions = eligible_directions

        closed = self._close_lost_signals(today, exclude) if self.exit_when_signal_lost else set()
        exclude = exclude | closed

        for symbol in filed_symbols:
            if symbol in exclude:
                continue
            target = targets.get(symbol)
            if target is None:
                continue  # plus (ou toujours pas) éligible : rien à ouvrir/redimensionner

            existing = self.positions.get(symbol)
            if existing is None:
                self._queue_isolated_order(symbol, target, today, reason="rebalance")
                continue

            new_log_gap = self._log_gap_at(symbol, self.prices.close_at(symbol, today))
            if new_log_gap is None or existing.last_rebalance_log_gap is None:
                delta = None  # référence indisponible : échoue ouvert, cf. _log_gap_at
            else:
                delta = abs(new_log_gap - existing.last_rebalance_log_gap)

            if delta is not None and delta <= self.rebalance_log_gap_threshold:
                continue  # sous le seuil : known_signals à jour, last_rebalance_log_gap intact

            self._queue_isolated_order(symbol, target, today, reason="rebalance")

        return closed

    def _close_lost_signals(self, today: pd.Timestamp, exclude: set[str]) -> set[str]:
        """Clôture les positions dont le signal ne justifie plus la présence :
        écart repassé sous le seuil d'entrée (le symbole a disparu des
        éligibles), ou écart qui a changé de sens (call détenu alors que le
        signal est devenu vendeur, ou l'inverse)."""
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
    def execution_diagnostics(self) -> dict:
        """Mêmes diagnostics que backtest/engine.py::execution_diagnostics,
        plus l'exposition delta moyenne et maximale -- le levier n'a pas à
        être relu à la main dans l'equity_curve pour être constaté."""
        truncated_pct = (
            self.truncated_orders_count / self.buy_orders_count * 100
            if self.buy_orders_count else 0.0
        )
        rows = [row for row in self.equity_curve_rows if row["nav"] > 0]
        cash_pct = [row["cash"] / row["nav"] * 100 for row in rows]
        delta_pct = [row["delta_notional"] / row["nav"] * 100 for row in rows]

        if truncated_pct > engine_mod.TRUNCATED_ORDERS_WARNING_PCT:
            logger.warning(
                "%.1f%% des ordres d'achat (%d/%d) ont été tronqués faute de cash : le budget "
                "alloué dépasse régulièrement le cash disponible, les positions gelées "
                "immobilisant du capital sans être vendues.",
                truncated_pct, self.truncated_orders_count, self.buy_orders_count,
            )

        dropped_total = sum(self.dropped_orders.values())
        if dropped_total:
            logger.warning(
                "%d ordres d'achat ABANDONNÉS (aucune exécution, aucun coût) : %s. "
                "Un ordre abandonné ne laisse aucune trace dans trades.parquet ni dans "
                "l'equity_curve -- si 'frais_excessifs' ou 'cash_insuffisant' domine, la "
                "stratégie a voulu être investie et ne l'a pas été, et les métriques "
                "ci-dessous décrivent un portefeuille que personne n'a choisi.",
                dropped_total,
                ", ".join(f"{k}={v}" for k, v in sorted(
                    self.dropped_orders.items(), key=lambda kv: -kv[1],
                )),
            )

        # Coût de portage cumulé (theta) : le seuil que la thèse doit battre
        # avant de gagner quoi que ce soit. Somme des thetas quotidiens
        # publiés dans l'equity_curve.
        theta_total = sum(row.get("theta_per_day") or 0.0 for row in self.equity_curve_rows)
        vega_pct = [
            row["vega_notional"] / row["nav"] * 100
            for row in rows if row.get("vega_notional") is not None
        ]

        total_friction = self.total_commission + self.total_slippage
        initial_nav = self.equity_curve_rows[0]["nav"] if self.equity_curve_rows else None
        return {
            "buy_orders_count": self.buy_orders_count,
            "truncated_orders_count": self.truncated_orders_count,
            "truncated_orders_pct": float(truncated_pct),
            "capped_orders_count": self.capped_orders_count,
            "dropped_orders_count": int(dropped_total),
            "dropped_orders_by_reason": dict(sorted(self.dropped_orders.items())),
            "delever_events_count": self.delever_events_count,
            # THETA : coût de portage cumulé sur tout le run, et son poids
            # rapporté au capital de départ. Négatif (c'est une perte).
            "total_theta_decay_dollar": float(theta_total),
            "total_theta_decay_pct_of_initial": (
                float(theta_total / initial_nav * 100) if initial_nav else None
            ),
            # VEGA : exposition moyenne à +1 point de volatilité implicite, en
            # % du NAV. Le repricing ne suit JAMAIS l'implicite (cf.
            # options_pricing), donc ce P&L n'est pas simulé -- ce chiffre dit
            # de combien le run se trompe pour chaque point d'implicite.
            "avg_vega_notional_pct": float(sum(vega_pct) / len(vega_pct)) if vega_pct else None,
            "avg_cash_pct": float(sum(cash_pct) / len(cash_pct)) if cash_pct else None,
            "avg_delta_notional_pct": float(sum(delta_pct) / len(delta_pct)) if delta_pct else None,
            "max_delta_notional_pct_observed": float(max(delta_pct)) if delta_pct else None,
            # Friction totale, et son poids rapporté au capital de départ : le
            # chiffre qui dit si une perte vient de la thèse ou de son coût
            # d'exécution.
            "total_commission_dollar": float(self.total_commission),
            "total_slippage_dollar": float(self.total_slippage),
            "total_friction_dollar": float(total_friction),
            "total_friction_pct_of_initial": (
                float(total_friction / initial_nav * 100) if initial_nav else None
            ),
        }

    # ------------------------------------------------------------------ #
    # Comptabilité quotidienne
    # ------------------------------------------------------------------ #
    def _position_value(self, pos: OptionPosition, today: pd.Timestamp) -> float:
        premium = self._current_premium(pos, today)
        if premium is None:
            premium = pos.entry_premium
        return pos.contracts * pos.multiplier * premium

    def _invested_value(self, today: pd.Timestamp) -> float:
        """Valeur de marché de toutes les positions, mémoïsée par (date,
        version du portefeuille) -- même mécanique que _portfolio_greeks.

        _queue_isolated_order en a besoin pour CHAQUE ordre mis en file, et
        recalculait à chaque fois la somme sur toutes les positions : encore un
        O(n²) par jour, invisible tant que le portefeuille restait petit."""
        key = (today, self._positions_version)
        if self._invested_cache_key != key:
            self._invested_cache_key = key
            self._invested_cache = sum(
                self._position_value(pos, today) for pos in self.positions.values()
            )
        return self._invested_cache

    def _current_nav(self, today: pd.Timestamp) -> float:
        return self.cash + self._invested_value(today)

    def _mark_to_market(self, today: pd.Timestamp) -> None:
        invested = sum(self._position_value(pos, today) for pos in self.positions.values())
        nav = self.cash + invested
        # delta_notional : le levier réellement porté, invisible jusqu'ici
        # dans toutes les sorties du moteur (invested_value ne mesure que les
        # primes décaissées, soit ~10% de l'exposition d'une option ATM).
        # delta_notional_pct le rapporte au NAV, l'unité dans laquelle
        # config.OPTIONS_MAX_DELTA_NOTIONAL_PCT est exprimé.
        greeks = self._portfolio_greeks(today)
        delta_notional = greeks["delta_notional"]
        self.equity_curve_rows.append({
            "date": today, "nav": nav, "cash": self.cash, "invested_value": invested,
            "num_positions": len(self.positions),
            "delta_notional": delta_notional,
            "delta_notional_pct": (delta_notional / nav * 100) if nav > 0 else None,
            # Coût de portage du jour (négatif) et exposition à la volatilité
            # implicite : le premier est le seuil que la thèse doit battre, le
            # second la part du P&L que ce backtest ne simule pas du tout
            # (cf. execution_diagnostics et options_pricing).
            "theta_per_day": greeks["theta_per_day"],
            "vega_notional": greeks["vega_notional"],
            # Friction payée depuis le début du run (cumuls monotones : la
            # friction d'un jour donné s'obtient par simple différence).
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
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
