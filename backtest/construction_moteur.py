"""Construction du moteur ACTIONS depuis la ligne de commande -- partagée par
09_backtest.py et 17_paper_trading.py.

POURQUOI UN MODULE. Le compte paper doit rejouer EXACTEMENT la configuration du
backtest : mêmes coûts, même zone de non-négociation, même ciblage de
volatilité, mêmes sorties. Recopier la liste des réglages dans un second
script, c'est garantir qu'un jour l'un bouge sans l'autre -- et le compte paper
testerait alors une autre stratégie que celle qui a été mesurée, sans que rien
ne le dise. Les options du moteur et leurs défauts n'existent donc qu'ici.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import pandas as pd

import config
from backtest import data_loader
from backtest.engine import BacktestEngine
from backtest.strategies import STRATEGY_REGISTRY
from backtest.strategies.base import Strategy


def parse_strategy_params(pairs: list[str]) -> dict:
    params = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--strategy-param attend 'clé=valeur', reçu: {pair!r}")
        key, raw_value = pair.split("=", 1)
        try:
            value = float(raw_value)
            if value.is_integer():
                value = int(value)
        except ValueError:
            value = raw_value
        params[key] = value
    return params


def ajouter_options_moteur(parser: argparse.ArgumentParser) -> None:
    """Les réglages de la stratégie et du moteur, avec les défauts de config.

    Ni `--strategy` ni les dates : leurs défauts diffèrent d'un script à
    l'autre (09 balaie l'historique, le compte paper rejoue une stratégie
    choisie jusqu'à aujourd'hui), chacun les déclare donc lui-même."""
    parser.add_argument("--initial-capital", type=float, default=config.BACKTEST_INITIAL_CAPITAL)
    parser.add_argument("--commission-bps", type=float, default=config.BACKTEST_COMMISSION_BPS)
    parser.add_argument("--slippage-bps", type=float, default=config.BACKTEST_SLIPPAGE_BPS)
    parser.add_argument("--stop-loss-pct", type=float, default=config.BACKTEST_STOP_LOSS_PCT, help="Négatif, ex: -15.")
    parser.add_argument("--take-profit-pct", type=float, default=config.BACKTEST_TAKE_PROFIT_PCT)
    parser.add_argument(
        "--entry-threshold-pct", type=float, default=None,
        # LES `%%` SONT OBLIGATOIRES, et le `%` de formatage ne l'est pas. Ce
        # help était interpolé ICI puis ré-interpolé par argparse, qui applique
        # `help % params` au moment d'afficher --help : le « (20%) » produit par
        # la première passe devenait une directive invalide pour la seconde, et
        # `python 09_backtest.py --help` plantait sur ValueError. Une f-string
        # supprime la première passe ; les `%%` survivent à la seconde.
        help="Seuil d'entrée passé à la stratégie. Non précisé, CHAQUE stratégie garde son "
             "propre défaut -- ils ne se lisent pas pareil : valuation_gap_dcf attend un écart "
             f"au cours ({config.BACKTEST_ENTRY_THRESHOLD_PCT:.0f}%%), "
             "valuation_gap_sector_neutral un écart à la médiane de son secteur "
             f"({config.BACKTEST_SECTOR_NEUTRAL_ENTRY_THRESHOLD_PCT:.0f}%%).",
    )
    parser.add_argument("--strategy-param", action="append", default=[], metavar="KEY=VALUE", help="Paramètre supplémentaire spécifique à la stratégie (répétable).")
    parser.add_argument(
        "--momentum-min-pct", type=float, default=config.BACKTEST_STOCKS_MOMENTUM_MIN_PCT,
        help="Momentum 12-1 minimal (en %%) pour une NOUVELLE entrée, filtre anti-value-trap. "
             "Ex: -10. Utiliser --no-momentum-filter pour le désactiver.",
    )
    parser.add_argument(
        "--no-momentum-filter", dest="momentum_min_pct", action="store_const", const=None,
        help="Désactive le filtre momentum.",
    )
    parser.add_argument(
        "--rebalance-band-pct", type=float, default=config.BACKTEST_REBALANCE_BAND_PCT,
        help="Zone de non-négociation, en POINTS DE NAV : le portefeuille n'est repesé que les "
             "jours où il faudrait faire bouger au moins ce %%%% de sa valeur. Sans elle, un seul "
             "dépôt SEC repèse tout le portefeuille (défaut: %(default)s, 0 désactive).",
    )
    parser.add_argument(
        "--trailing-stop-pct", type=float, default=config.BACKTEST_TRAILING_STOP_PCT,
        help="Stop SUIVEUR : recul maximal depuis le plus haut atteint depuis l'entrée "
             "(négatif, ex. -25). Défaut: désactivé.",
    )
    parser.add_argument(
        "--max-holding-days", type=int, default=config.BACKTEST_MAX_HOLDING_DAYS,
        help="Durée de détention maximale, en jours. Défaut: désactivé.",
    )
    parser.add_argument(
        "--exit-gap-threshold-pct", type=float, default=config.BACKTEST_EXIT_GAP_THRESHOLD_PCT,
        help="Vend une ligne dont l'écart est repassé sous ce seuil. TOUCHE À LA RÈGLE DES "
             "POSITIONS GELÉES (une position n'est sinon jamais vendue sur refermeture de "
             "l'écart) : désactivé par défaut, à activer en connaissance de cause.",
    )
    parser.add_argument(
        "--impact-coefficient-bps", type=float, default=config.BACKTEST_IMPACT_COEFFICIENT_BPS,
        help="Impact de marche, en bps, d'un ordre egal a 100%% du volume quotidien moyen du "
             "titre (l'impact suit la RACINE de la part de volume consommee). 0 = desactive. "
             "Sert surtout a chiffrer la CAPACITE : jusqu'a quel encours la strategie tient.",
    )
    parser.add_argument(
        "--min-commission-dollar", type=float, default=config.BACKTEST_MIN_COMMISSION_DOLLAR,
        help="Commission MINIMUM par exécution, en dollars : le coût d'un ordre devient "
             "max(notionnel x bps, ce minimum). 1 = 1 $ à l'achat et 1 $ à la vente. "
             "0 = coût purement proportionnel (défaut, comportement historique).",
    )
    parser.add_argument(
        "--min-trade-pct-of-nav", type=float, default=config.BACKTEST_MIN_TRADE_PCT_OF_NAV,
        help="Plancher de taille d'ordre en %% du NAV. Le plancher absolu de 1 $ ne coupe "
             "rien à l'échelle (0,000036 %% d'un NAV de 2,8 M$) ; celui-ci tient. Ne "
             "s'applique JAMAIS aux liquidations. 0 = désactivé.",
    )
    parser.add_argument(
        "--max-fee-pct-of-trade", type=float, default=config.BACKTEST_MAX_FEE_PCT_OF_TRADE,
        help="Part maximale d'un ordre que la commission minimum a le droit de représenter : "
             "c'est le critère de VIABILITÉ. Avec 1 $ de commission minimum et 1, un ordre "
             "sous 100 $ n'est pas passé. 0 = aucun seuil.",
    )
    parser.add_argument(
        "--vol-target-pct", type=float, default=config.BACKTEST_VOL_TARGET_PCT,
        help="Cible de volatilite annualisee du portefeuille, en %%. L'exposition est REDUITE "
             "quand la volatilite realisee recente depasse la cible, jamais augmentee au-dela "
             "de 100%% (le moteur n'est pas marge). Defaut: desactive.",
    )


@dataclass
class DonneesActions:
    """Tout ce que le moteur actions lit, chargé une seule fois."""
    price_panel: data_loader.PricePanel
    signal_events: pd.DataFrame
    universe_history: Optional[pd.DataFrame]
    fallback_symbols: set
    material_events: Optional[pd.DataFrame]


def charger_donnees(strategy_name: str) -> DonneesActions:
    daily_prices = data_loader.load_daily_prices()
    price_panel = data_loader.build_price_panel(daily_prices)
    # La source du signal est déclarée par la STRATÉGIE (Strategy.signal_source)
    # et non choisie ici : c'est une propriété de sa thèse. Tester le nom de la
    # stratégie en dur obligerait à modifier ce fichier à chaque ajout, ce que
    # le registre sert précisément à éviter.
    signal_events = data_loader.build_strategy_signal_events(
        STRATEGY_REGISTRY[strategy_name].signal_source)
    return DonneesActions(
        price_panel=price_panel,
        signal_events=signal_events,
        universe_history=data_loader.load_universe_history(),
        fallback_symbols=data_loader.load_current_universe_symbols(),
        material_events=data_loader.load_material_events_8k(),
    )


def construire_strategie(strategy_name: str, args: argparse.Namespace) -> Strategy:
    strategy_cls = STRATEGY_REGISTRY[strategy_name]
    # `entry_threshold_pct` n'est transmis QUE s'il a été demandé. Le passer
    # systématiquement écrasait le défaut propre à chaque stratégie par celui
    # de valuation_gap_dcf (20%) : valuation_gap_sector_neutral, dont le seuil
    # porte sur l'écart à la MÉDIANE DU SECTEUR et vaut 10%, tournait en
    # silence à 20 -- deux fois trop sélectif, sans que rien ne l'indique.
    strategy_params: dict = {}
    if args.entry_threshold_pct is not None:
        strategy_params["entry_threshold_pct"] = args.entry_threshold_pct
    strategy_params.update(parse_strategy_params(args.strategy_param))
    return strategy_cls(**strategy_params)


def construire_moteur(
    args: argparse.Namespace,
    donnees: DonneesActions,
    strategy: Strategy,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> BacktestEngine:
    return BacktestEngine(
        price_panel=donnees.price_panel,
        signal_events=donnees.signal_events,
        universe_history=donnees.universe_history,
        fallback_universe_symbols=donnees.fallback_symbols,
        strategy=strategy,
        initial_capital=args.initial_capital,
        cost_bps=args.commission_bps + args.slippage_bps,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        momentum_min_pct=args.momentum_min_pct,
        rebalance_band_pct=args.rebalance_band_pct,
        trailing_stop_pct=args.trailing_stop_pct,
        max_holding_days=args.max_holding_days,
        exit_gap_threshold_pct=args.exit_gap_threshold_pct,
        impact_coefficient_bps=args.impact_coefficient_bps,
        min_commission_dollar=args.min_commission_dollar,
        min_trade_pct_of_nav=args.min_trade_pct_of_nav,
        max_fee_pct_of_trade=args.max_fee_pct_of_trade,
        vol_target_pct=args.vol_target_pct,
        material_events_8k=donnees.material_events,
        start_date=start_date,
        end_date=end_date,
    )
