"""
Grid-search sur N'IMPORTE QUEL paramètre scalaire, de la stratégie OU du moteur.

Rejoue le backtest de 10_backtest_options.py pour chaque valeur du paramètre
demandé, sur les MÊMES données chargées UNE SEULE FOIS -- même structure que
11b/11c/11d, dont ce script reprend les conventions. Tous les autres réglages,
moteur ET stratégie, sont FIXÉS à leurs valeurs courantes (engine_defaults de
la stratégie sinon config) : le paramètre balayé est le SEUL qui varie.

Le script trouve tout seul si le nom demandé est un paramètre de la STRATÉGIE
(constructeur de la classe, héritage compris) ou du MOTEUR
(OptionsBacktestEngine.__init__), et refuse un nom qui n'est ni l'un ni
l'autre. Les deux familles couvrent ensemble tout ce qui est réglable sans
toucher au code.

POURQUOI UN SCRIPT GÉNÉRIQUE PLUTÔT QU'UN 11e, 11f, 11g...
-----------------------------------------------------------
11, 11b, 11c et 11d sont quatre déclinaisons de la même mécanique pour quatre
paramètres. En ajouter une par nouveau réglage optimisable multiplie 400 lignes
de structure identique -- et surtout, un paramètre sans script reste de facto
non optimisable, ce qui est la vraie perte. Ce script couvre tout paramètre
scalaire du constructeur d'une stratégie, présent ou futur, sans nouveau
fichier. Les quatre scripts dédiés restent en place : ils portent des grilles,
des garde-fous et des commentaires propres à leur paramètre (11 balaye un
COUPLE stop/take, 11d partage sa grille avec le seuil de sortie), que
généraliser aurait dilué.

LE PIÈGE QUE CE SCRIPT FERME
-----------------------------
`Strategy.__init__(**params)` stocke tout ce qu'on lui passe sans rien valider
(backtest/strategies/base.py). Une faute de frappe est donc AVALÉE EN SILENCE :

    ValuationGapExpectedValueOptionsStrategy(min_kely_fraction=0.42)
    -> params["min_kely_fraction"] = 0.42   (et jusque dans run_config.json)
    -> self.min_kelly_fraction  = 0.05      (le vrai réglage, inchangé)

Un balayage lancé sur ce nom-là produirait N runs RIGOUREUSEMENT IDENTIQUES,
puis recommanderait fièrement une valeur. Ce script refuse donc tout paramètre
qui n'apparaît pas explicitement dans la signature du constructeur de la
stratégie (héritage compris), et signale en fin de course une grille dont
toutes les lignes sont identiques -- soit un plateau réel, soit un réglage qui
n'a jamais été branché.

Usage :
    python 11e_optimize_strategy_param.py                       # min_kelly_fraction
    python 11e_optimize_strategy_param.py --param strike_grid_n_sigma
    python 11e_optimize_strategy_param.py --param min_holding_days       # paramètre MOTEUR
    python 11e_optimize_strategy_param.py --param min_kelly_fraction --grid 0 0.05 0.1 0.2
    python 11e_optimize_strategy_param.py --list-params         # tout ce qui est balayable
    python 11e_optimize_strategy_param.py --workers 4           # process en parallèle (fork)

Le CSV complet (une ligne par valeur, toutes les métriques, y compris TOUS les
compteurs que la stratégie expose via diagnostics()) est écrit sous
data/backtest_options/optimize_<paramètre>_<stratégie>_<horodatage>.csv.

PAS DE WALK-FORWARD, comme 11b/11c/11d : in-sample assumé. Une valeur choisie
ici est choisie sur les données qui servent ensuite à la juger -- c'est un
point de départ, pas une validation.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import pandas as pd

import config
from backtest import data_loader, metrics as metrics_mod
from backtest.options_engine import OptionsBacktestEngine
from backtest.strategies import OPTIONS_STRATEGY_REGISTRY

logger = logging.getLogger("optimize_strategy_param")

# 10_backtest_options.py n'est pas un nom de module Python valide (commence par
# un chiffre) : import explicite, comme le font déjà 11, 11b, 11c et 11d.
_cli = importlib.import_module("10_backtest_options")

DEFAULT_STRATEGY = "valuation_gap_expected_value_options"
DEFAULT_PARAM = "min_kelly_fraction"


def _dans(borne_basse: float, borne_haute: float, inclus_haut: bool = True):
    """Validateur de grille, rendu sous forme de (test, message)."""
    haut = "]" if inclus_haut else "["
    libelle = f"[{borne_basse}, {borne_haute}{haut}"

    def test(v: float) -> bool:
        return borne_basse <= v <= borne_haute if inclus_haut else borne_basse <= v < borne_haute

    return test, libelle


def _positif():
    return (lambda v: v > 0), "]0, +inf["


# Paramètres connus : grille par défaut, bornes, et ce que le réglage décide.
# Un paramètre ABSENT d'ici reste balayable -- il faut simplement fournir
# --grid explicitement, et aucune borne n'est vérifiée.
KNOWN_PARAMS: dict[str, dict] = {
    "min_kelly_fraction": {
        "grid": [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
        "bounds": _dans(0.0, 1.0, inclus_haut=False),
        "aide": "Plancher de matérialité sur la mise log-optimale f* : une ligne que Kelly "
                "dimensionne sous ce seuil n'est pas ouverte. 0 désactive. Ne mord que sur "
                "les sous-jacents très volatils (cf. config.OPTIONS_EV_MIN_KELLY_FRACTION).",
        "compteur": "dropped_kelly_below_floor_count",
    },
    "strike_grid_n_sigma": {
        "grid": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
        "bounds": _positif(),
        "aide": "Demi-largeur de la grille de strikes candidats, en écarts-types du log-prix "
                "à l'échéance. Trop étroite, l'optimum se colle à un bord ; trop large, elle "
                "propose des contrats que personne ne cote.",
    },
    "strike_grid_step_sigma": {
        "grid": [0.10, 0.15, 0.20, 0.25, 0.35, 0.50],
        "bounds": _positif(),
        "aide": "Pas de la grille de strikes, en écarts-types. Plus fin = strike plus proche "
                "de l'optimum théorique, au prix d'un contrat moins susceptible d'être coté.",
    },
    "exit_threshold_ratio": {
        "grid": [0.0, 0.25, 0.50, 0.75, 1.0],
        "bounds": _dans(0.0, 1.0),
        "aide": "Hystérésis : le seuil de SORTIE vaut cette fraction du seuil d'entrée. "
                "1,0 supprime l'hystérésis (on sort dès qu'on n'entrerait plus).",
    },
    # --- paramètres du MOTEUR (OptionsBacktestEngine) -----------------------
    # Aucun n'avait de script d'optimisation avant celui-ci : 11 couvre le
    # couple stop/take, 11b l'epsilon de rebalancement, et c'est tout.
    "min_holding_days": {
        "grid": [0, 5, 10, 21, 42, 63],
        "bounds": _dans(0, 10_000),
        "aide": "Durée minimale de détention avant qu'un stop puisse sortir. Protège d'un "
                "aller-retour immédiat, au prix d'un stop qui ne coupe pas tout de suite.",
    },
    "take_profit_convergence_fraction": {
        "grid": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "bounds": _dans(0.0, 1.0, inclus_haut=True),
        "aide": "Part du chemin vers la valeur théorique au-delà de laquelle on prend le "
                "gain, au lieu d'un seuil fixe en pourcentage.",
    },
    "max_delta_notional_pct": {
        "grid": [50.0, 75.0, 100.0, 150.0, 200.0, 300.0],
        "bounds": _positif(),
        "aide": "Plafond de levier : exposition delta-équivalente maximale, en % du NAV. "
                "C'est la mesure honnête du levier -- la prime n'en est qu'une fraction.",
    },
    "delever_tolerance_pct": {
        "grid": [0.0, 5.0, 10.0, 20.0, 30.0],
        "bounds": _dans(0.0, 1000.0),
        "aide": "Bande de tolérance avant que le dé-levier ne déclenche une réduction au "
                "prorata. Trop serrée, elle fait vendre sur du bruit de marché.",
    },
    "min_deployment_pct": {
        "grid": [0.0, 25.0, 50.0, 75.0, 90.0],
        "bounds": _dans(0.0, 100.0),
        "aide": "Plancher de primes : part du NAV maintenue investie par redéploiement du "
                "cash oisif. 0 = désactivé (défaut depuis l'audit : c'était une martingale "
                "sur les positions perdantes).",
    },
    "roll_when_days_left": {
        "grid": [90, 180, 270, 365, 540],
        "bounds": _positif(),
        "aide": "Point de décision du roulement, en jours avant l'échéance. Plus haut = on "
                "ne porte jamais le contrat sur sa dernière année, là où la valeur temps "
                "s'érode le plus vite, mais on paie plus souvent le roulement.",
    },
    "target_tenor_days": {
        "grid": [365, 545, 730, 900, 1095],
        "bounds": _positif(),
        "aide": "Échéance visée à l'entrée. Arbitrage direct entre coût de portage (theta) "
                "et temps laissé à la thèse pour se réaliser.",
    },
}

# Rempli une fois par process. Sur Linux les workers l'héritent par fork ;
# ailleurs, _pool_initializer est le SEUL moyen par lequel ils le reçoivent
# (cf. le même commentaire dans 11c).
_DATA: dict = {}


def _pool_initializer(data: dict) -> None:
    global _DATA
    _DATA = data


def _parametres_acceptes(strategy_cls) -> list[str]:
    """Paramètres NOMMÉS des constructeurs de la classe et de ses ancêtres.

    `**kwargs` ne compte pas : c'est précisément ce qui rend une faute de
    frappe silencieuse (cf. le pavé en tête de module). Seul un nom déclaré
    explicitement prouve que le réglage est branché quelque part."""
    noms: set[str] = set()
    for cls in inspect.getmro(strategy_cls):
        init = cls.__dict__.get("__init__")
        if init is None:
            continue
        for nom, p in inspect.signature(init).parameters.items():
            if nom in ("self",) or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            noms.add(nom)
    return sorted(noms)


def _parametres_moteur() -> list[str]:
    """Paramètres nommés d'OptionsBacktestEngine, hors ceux que ce script
    fournit lui-même (données, dates, stratégie) : les faire varier n'aurait
    pas de sens dans un balayage."""
    reserves = {
        "self", "price_panel", "signal_events", "universe_history",
        "fallback_universe_symbols", "option_snapshots", "strategy",
        "material_events_8k", "start_date", "end_date",
    }
    return sorted(
        nom for nom, p in inspect.signature(OptionsBacktestEngine.__init__).parameters.items()
        if nom not in reserves and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    )


def _load_data(benchmark_symbol: str) -> dict:
    logger.info("Chargement des données (une seule fois pour toute la grille)...")
    daily_prices = data_loader.load_daily_prices()
    price_panel = data_loader.build_price_panel(daily_prices)
    valorisation_combinee = data_loader.load_valorisation_combinee_history()
    signal_events = data_loader.build_options_signal_events(valorisation_combinee)
    universe_history = data_loader.load_universe_history()
    fallback_symbols = data_loader.load_current_universe_symbols()
    option_snapshots = data_loader.load_option_snapshots_history()
    material_events = data_loader.load_material_events_8k()
    universe = data_loader.UniverseResolver(universe_history, fallback_symbols)
    benchmark_prices, benchmark_label = data_loader.build_benchmark_series(
        price_panel, universe, symbol=benchmark_symbol,
    )
    return {
        "price_panel": price_panel,
        "signal_events": signal_events,
        "universe_history": universe_history,
        "fallback_symbols": fallback_symbols,
        "option_snapshots": option_snapshots,
        "material_events": material_events,
        "benchmark_prices": benchmark_prices,
        "benchmark_label": benchmark_label,
    }


# Métriques retenues pour toutes les grilles. Les compteurs PROPRES à la
# stratégie ne sont pas listés ici : ils sont ramassés dynamiquement (voir
# _run_one), pour qu'un nouveau compteur apparaisse dans le CSV sans que ce
# script ait à le connaître.
METRIC_KEYS = (
    "num_trades", "win_rate_pct", "profit_factor", "total_return_pct",
    "cagr_pct", "annualized_volatility_pct", "sharpe_ratio", "sortino_ratio",
    "max_drawdown_pct", "calmar_ratio", "avg_holding_days",
    "total_commission_dollar", "total_slippage_dollar",
    "total_friction_dollar", "total_friction_pct_of_initial",
)


def _run_one(
    param: str,
    cible: str,
    value: float,
    strategy_name: str,
    strategy_params: dict,
    baseline_settings: dict,
    engine_kwargs: dict,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> dict:
    """Un run complet pour cette valeur. `cible` vaut "strategy" ou "engine".
    Fonction de module (et non closure) pour rester picklable par
    ProcessPoolExecutor ; _DATA ne transite pas par les arguments, il vient du
    fork ou de _pool_initializer."""
    row = {param: value, "error": None}
    try:
        strategy_cls = OPTIONS_STRATEGY_REGISTRY[strategy_name]
        params_strategie = dict(strategy_params)
        if cible == "strategy":
            params_strategie[param] = value
        strategy = strategy_cls(**params_strategie)

        # Un SEUL dictionnaire de réglages moteur, pour que l'injection du
        # paramètre balayé ne dépende pas de la case où il se range : certains
        # (min_holding_days, roll_when_days_left, target_tenor_days...) sont
        # résolus par resolve_engine_settings, les autres viennent de la CLI.
        # Les passer en deux endroits ferait qu'une moitié des paramètres
        # moteur serait silencieusement ignorée par le balayage.
        kwargs = {
            "stop_loss_pct": baseline_settings["stop_loss_pct"],
            "take_profit_pct": baseline_settings["take_profit_pct"],
            "target_tenor_days": baseline_settings["target_tenor_days"],
            "stop_basis": baseline_settings["stop_basis"],
            "exit_when_signal_lost": baseline_settings["exit_when_signal_lost"],
            "roll_when_days_left": baseline_settings["roll_when_days_left"],
            "daily_rebalance": baseline_settings["daily_rebalance"],
            "vol_mode": baseline_settings["vol_mode"],
            "min_resize_relative_pct": baseline_settings["min_resize_relative_pct"],
            "min_holding_days": baseline_settings["min_holding_days"],
            **engine_kwargs,
        }
        if cible == "engine":
            kwargs[param] = value

        engine = OptionsBacktestEngine(
            price_panel=_DATA["price_panel"],
            signal_events=_DATA["signal_events"],
            universe_history=_DATA["universe_history"],
            fallback_universe_symbols=_DATA["fallback_symbols"],
            option_snapshots=_DATA["option_snapshots"],
            material_events_8k=_DATA["material_events"],
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
        equity_curve, _positions_history, trades, _signals_history = engine.run()
        diagnostics = engine.execution_diagnostics()
        run_metrics = metrics_mod.compute_metrics(
            equity_curve, trades,
            risk_free_rate=config.RISK_FREE_RATE,
            benchmark_prices=_DATA["benchmark_prices"],
            extra=diagnostics,
        )
        for key in METRIC_KEYS:
            row[key] = run_metrics.get(key)

        # Compteurs de la stratégie, ramassés SANS liste en dur : c'est eux qui
        # disent si le réglage balayé a mordu, et ils diffèrent d'une stratégie
        # à l'autre (cf. options_engine._strategy_diagnostics).
        for key, val in getattr(strategy, "diagnostics", dict)().items():
            if isinstance(val, (int, float, type(None))):
                row[key] = val

        if not trades.empty and "exit_reason" in trades.columns:
            row["exits_by_reason"] = trades["exit_reason"].value_counts().to_dict()
    except Exception as exc:  # noqa: BLE001 -- une valeur qui plante ne doit pas tuer la grille
        logger.exception("Échec pour %s=%s", param, value)
        row["error"] = str(exc)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, choices=sorted(OPTIONS_STRATEGY_REGISTRY))
    parser.add_argument(
        "--param", default=DEFAULT_PARAM,
        help=f"Paramètre du constructeur de la stratégie à balayer (défaut: %(default)s). "
             f"Grille par défaut connue pour : {', '.join(sorted(KNOWN_PARAMS))}. "
             "Tout autre paramètre accepté par la stratégie marche aussi, avec --grid.",
    )
    parser.add_argument(
        "--target", choices=("strategy", "engine"), default=None,
        help="Où appliquer --param. Déduit automatiquement ; à préciser seulement si le nom "
             "existe des deux côtés.",
    )
    parser.add_argument(
        "--list-params", action="store_true",
        help="Affiche les paramètres que la stratégie accepte, puis sort. Aucun backtest.",
    )
    parser.add_argument(
        "--grid", type=float, nargs="+", default=None,
        help="Valeurs à tester. Obligatoire pour un paramètre hors de la liste connue.",
    )
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument(
        "--min-trades", type=int, default=20,
        help="Nombre de trades sous lequel une valeur n'est pas recommandable, quel que soit "
             "son Sharpe (défaut: %(default)s). Un réglage qui n'ouvre presque rien peut "
             "afficher un Sharpe flatteur sur une poignée de positions.",
    )
    parser.add_argument("--objective", default="sharpe_ratio", help="Métrique de classement (défaut: %(default)s).")
    parser.add_argument("--entry-threshold-pct", type=float, default=None)
    parser.add_argument("--benchmark-symbol", default=config.BENCHMARK_SYMBOL)
    parser.add_argument("--initial-capital", type=float, default=config.OPTIONS_INITIAL_CAPITAL)
    parser.add_argument("--commission-per-contract", type=float, default=config.OPTIONS_COMMISSION_PER_CONTRACT)
    parser.add_argument("--slippage-pct-of-premium", type=float, default=config.OPTIONS_SLIPPAGE_PCT_OF_PREMIUM)
    parser.add_argument("--commission-min-per-order", type=float, default=config.OPTIONS_COMMISSION_MIN_PER_ORDER)
    parser.add_argument("--max-fee-pct-of-trade", type=float, default=config.OPTIONS_MAX_FEE_PCT_OF_TRADE)
    parser.add_argument("--fee-bump-max-extra-pct", type=float, default=config.OPTIONS_FEE_BUMP_MAX_EXTRA_PCT)
    parser.add_argument("--min-deployment-pct", type=float, default=config.OPTIONS_MIN_DEPLOYMENT_PCT)
    parser.add_argument("--max-delta-notional-pct", type=float, default=config.OPTIONS_MAX_DELTA_NOTIONAL_PCT)
    parser.add_argument("--max-trade-pct-of-nav", type=float, default=config.OPTIONS_MAX_TRADE_PCT_OF_NAV)
    parser.add_argument("--max-trade-dollar", type=float, default=config.OPTIONS_MAX_TRADE_DOLLAR)
    parser.add_argument(
        "--fractional-contracts", dest="whole_contracts", action="store_false",
        default=config.OPTIONS_WHOLE_CONTRACTS,
    )
    parser.add_argument("--real-snapshot-tolerance-days", type=int, default=config.OPTIONS_REAL_SNAPSHOT_TOLERANCE_DAYS)
    parser.add_argument("--momentum-min-pct", type=float, default=config.BACKTEST_MOMENTUM_MIN_PCT)
    parser.add_argument("--no-momentum-filter", dest="momentum_min_pct", action="store_const", const=None)
    parser.add_argument("--workers", type=int, default=1, help="Process en parallèle (fork). 1 = séquentiel.")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    strategy_cls = OPTIONS_STRATEGY_REGISTRY[args.strategy]
    acceptes_strategie = _parametres_acceptes(strategy_cls)
    acceptes_moteur = _parametres_moteur()

    if args.list_params:
        for titre, noms in (
            (f"Paramètres de la stratégie '{args.strategy}'", acceptes_strategie),
            ("Paramètres du moteur (OptionsBacktestEngine)", acceptes_moteur),
        ):
            print(f"\n{titre} :\n")
            for nom in noms:
                connu = KNOWN_PARAMS.get(nom)
                print(f"  {nom}{'  (grille par défaut)' if connu else ''}")
                if connu:
                    print(f"      {connu['aide']}")
        print()
        return

    param = args.param
    # LE GARDE-FOU CENTRAL. Sans lui, un nom mal orthographié se range dans
    # `params` sans rien changer, et toute la grille rend le MÊME run.
    dans_strategie = param in acceptes_strategie
    dans_moteur = param in acceptes_moteur
    if not (dans_strategie or dans_moteur):
        tous = sorted(set(acceptes_strategie) | set(acceptes_moteur))
        proches = [n for n in tous if n.replace("_", "") == param.replace("_", "")] or [
            n for n in tous if param[:6] in n
        ]
        raise SystemExit(
            f"'{param}' n'est un paramètre ni de la stratégie '{args.strategy}', ni du moteur.\n"
            f"Aucun des deux constructeurs ne le déclare, et Strategy.__init__(**params) "
            f"l'accepterait pourtant sans erreur : la grille entière rendrait le même run.\n"
            + (f"Vouliez-vous dire : {', '.join(proches)} ?\n" if proches else "")
            + f"Liste complète : python {sys.argv[0]} --strategy {args.strategy} --list-params"
        )
    if dans_strategie and dans_moteur:
        # Homonymie réelle possible (ex: un réglage porté des deux côtés) :
        # deviner ferait balayer le mauvais.
        if args.target is None:
            raise SystemExit(
                f"'{param}' existe des DEUX côtés (stratégie et moteur) : précise "
                "--target strategy ou --target engine."
            )
        cible = args.target
    else:
        cible = "strategy" if dans_strategie else "engine"
        if args.target and args.target != cible:
            raise SystemExit(
                f"--target {args.target} demandé, mais '{param}' est un paramètre "
                f"{'de la stratégie' if dans_strategie else 'du moteur'}."
            )
    logger.info("'%s' est un paramètre %s.", param, "de la stratégie" if cible == "strategy" else "du moteur")

    connu = KNOWN_PARAMS.get(param, {})
    grid = sorted(args.grid) if args.grid else connu.get("grid")
    if not grid:
        raise SystemExit(
            f"Aucune grille par défaut pour '{param}' : fournis --grid. "
            f"Grilles connues : {', '.join(sorted(KNOWN_PARAMS))}."
        )
    if "bounds" in connu:
        test, libelle = connu["bounds"]
        hors = [v for v in grid if not test(v)]
        if hors:
            raise SystemExit(f"--grid attend des valeurs dans {libelle} pour '{param}', reçu {hors}.")

    # Réglages moteur FIXÉS pour toute la grille, résolus UNE FOIS via la même
    # logique que 10_backtest_options.py.
    settings_args = argparse.Namespace(
        stop_loss_pct=None, take_profit_pct=None, target_tenor_days=None, stop_basis=None,
        exit_when_signal_lost=None, roll_when_days_left=None, daily_rebalance=None,
        vol_mode=None, min_resize_relative_pct=None,
    )
    baseline_settings, imposed = _cli.resolve_engine_settings(strategy_cls, settings_args)
    if imposed:
        logger.info("Réglages imposés par la stratégie '%s' : %s.", args.strategy, ", ".join(imposed))

    strategy_params = {}
    if args.entry_threshold_pct is not None:
        strategy_params["entry_threshold_pct"] = args.entry_threshold_pct

    start_date = pd.Timestamp(args.start_date) if args.start_date else None
    end_date = pd.Timestamp(args.end_date) if args.end_date else None

    engine_kwargs = dict(
        initial_capital=args.initial_capital,
        commission_per_contract=args.commission_per_contract,
        slippage_pct_of_premium=args.slippage_pct_of_premium,
        commission_min_per_order=args.commission_min_per_order,
        max_fee_pct_of_trade=args.max_fee_pct_of_trade,
        min_deployment_pct=args.min_deployment_pct,
        max_delta_notional_pct=args.max_delta_notional_pct,
        fee_bump_max_extra_pct=args.fee_bump_max_extra_pct,
        whole_contracts=args.whole_contracts,
        real_snapshot_tolerance_days=args.real_snapshot_tolerance_days,
        momentum_min_pct=args.momentum_min_pct,
        max_trade_dollar=args.max_trade_dollar,
        max_trade_pct_of_nav=args.max_trade_pct_of_nav,
    )

    global _DATA
    _DATA = _load_data(args.benchmark_symbol)

    logger.info("Backtest options '%s' : %d valeurs de %s %s...", args.strategy, len(grid), param, grid)

    rows: list[dict] = []
    if args.workers <= 1:
        for i, value in enumerate(grid, 1):
            logger.info("[%d/%d] %s=%s", i, len(grid), param, value)
            rows.append(_run_one(
                param, cible, value, args.strategy, strategy_params, baseline_settings,
                engine_kwargs, start_date, end_date,
            ))
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_pool_initializer, initargs=(_DATA,),
        ) as pool:
            futures = {
                pool.submit(
                    _run_one, param, cible, value, args.strategy, strategy_params,
                    baseline_settings, engine_kwargs, start_date, end_date,
                ): value
                for value in grid
            }
            for i, future in enumerate(as_completed(futures), 1):
                logger.info("[%d/%d] terminé : %s=%s", i, len(grid), param, futures[future])
                rows.append(future.result())

    results = pd.DataFrame(rows).sort_values(param).reset_index(drop=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(args.output_csv or (
        config.DIR_BACKTEST_OPTIONS / f"optimize_{param}_{args.strategy}_{run_id}.csv"
    ))
    config.DIR_BACKTEST_OPTIONS.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    logger.info("Résultats complets (%d lignes) sauvegardés dans %s", len(results), out_path)

    n_errors = results["error"].notna().sum()
    if n_errors:
        logger.warning("%d/%d valeurs ont échoué (voir la colonne 'error' du CSV).", n_errors, len(results))

    display_cols = [
        param, "num_trades", "win_rate_pct", "cagr_pct", args.objective,
        "max_drawdown_pct", "calmar_ratio", "total_friction_pct_of_initial",
    ]
    if connu.get("compteur") in results.columns:
        display_cols.append(connu["compteur"])
    display_cols = list(dict.fromkeys(c for c in display_cols if c in results.columns))
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:,.3f}".format):
        print(f"\n=== {param} -- stratégie: {args.strategy} ===")
        print(results[display_cols].to_string(index=False))

    usable = results[results["error"].isna()].copy()
    if usable.empty:
        logger.error("Aucune valeur exploitable (toutes en erreur).")
        sys.exit(1)

    # UNE GRILLE PLATE N'EST PAS UN RÉSULTAT. Soit le paramètre n'a aucun effet
    # sur cette période (plateau réel : information utile), soit il n'est pas
    # branché là où on le croit. Dans les deux cas, recommander une valeur
    # « meilleure » serait du bruit présenté comme un choix.
    varie = [c for c in ("num_trades", args.objective, "cagr_pct") if c in usable.columns]
    if varie and all(usable[c].nunique(dropna=False) <= 1 for c in varie):
        logger.warning(
            "TOUTES les valeurs de %s donnent des résultats IDENTIQUES (%s). Soit le "
            "paramètre est sans effet sur cette période -- élargis --grid ou la période --, "
            "soit il n'agit pas là où tu le crois. Aucune recommandation n'a de sens ici.",
            param, ", ".join(varie),
        )

    if args.objective not in usable.columns:
        raise SystemExit(f"--objective '{args.objective}' absent des métriques. Colonnes : {list(usable.columns)}")

    ranked = usable.sort_values(args.objective, ascending=False, na_position="last")
    with pd.option_context("display.width", 220, "display.float_format", "{:,.3f}".format):
        print(f"\n=== Top {min(args.top_n, len(ranked))} par {args.objective} ===")
        print(ranked[display_cols].head(args.top_n).to_string(index=False))

    credibles = ranked[ranked["num_trades"].fillna(0) >= args.min_trades]
    if credibles.empty:
        logger.error(
            "Aucune valeur n'atteint %d trades sur cette période. Élargis la période, "
            "baisse --entry-threshold-pct, ou abaisse --min-trades en connaissance de cause.",
            args.min_trades,
        )
        sys.exit(1)

    best = credibles.iloc[0]
    if len(credibles) < len(ranked):
        logger.warning(
            "%d valeur(s) écartée(s) du classement pour moins de %d trades.",
            len(ranked) - len(credibles), args.min_trades,
        )

    if best[param] in (min(grid), max(grid)):
        logger.warning(
            "L'optimum (%s=%s) est en BORD de grille -- élargis --grid pour vérifier qu'il "
            "ne s'agit pas d'un optimum tronqué par la plage testée.",
            param, best[param],
        )

    logger.info(
        "Meilleure valeur par %s : %s=%s (%s=%.3f, %d trades, CAGR=%.1f%%, max_drawdown=%.1f%%).",
        args.objective, param, best[param], args.objective,
        best.get(args.objective) or float("nan"), best.get("num_trades") or 0,
        best.get("cagr_pct") or float("nan"), best.get("max_drawdown_pct") or float("nan"),
    )
    logger.info(
        "Rappel : ce classement est IN-SAMPLE (pas de walk-forward). La valeur retenue a "
        "été choisie sur les données qui servent à la juger."
    )


if __name__ == "__main__":
    main()
