"""
Grid-search sur (stop-loss, prise de gain) pour une stratégie options.

Rejoue le backtest de 10_backtest_options.py pour CHAQUE binôme stop-loss /
take-profit d'une grille, sur les MÊMES données chargées UNE SEULE FOIS (le
chargement -- cours quotidiens, valorisation combinée, univers point-in-time,
snapshots d'options réels -- est de loin le poste le plus coûteux, il ne doit
pas être repayé à chaque combinaison), puis classe les résultats par la
métrique choisie (Sharpe par défaut).

Tous les autres réglages du moteur (échéance, base des stops, roulement,
etc.) restent ceux résolus par 10_backtest_options.resolve_engine_settings --
donc ceux imposés par la stratégie (engine_defaults) sauf override explicite
ici. Seuls le stop-loss et la prise de gain varient d'une combinaison à l'autre.

La PRISE DE GAIN n'a pas la même unité selon la stratégie (--take-profit-mode) :
un %% de mouvement du sous-jacent pour une stratégie ATM, une FRACTION du chemin
vers la valeur théorique pour une stratégie de convergence -- où take_profit_pct
est inerte et où le balayer ne mesurerait rien. Le mode retenu est journalisé au
lancement et reporté dans la colonne `take_profit_mode` du CSV.

Usage :
    python 11_optimize_options_stops.py
    python 11_optimize_options_stops.py --strategy valuation_gap_multiples_options
    python 11_optimize_options_stops.py --start-date 2015-01-01 --objective calmar_ratio
    python 11_optimize_options_stops.py \
        --stop-loss-grid -10 -15 -20 -25 -30 -35 -40 -50 \
        --take-profit-grid 20 30 40 60 80 100 150 200
    python 11_optimize_options_stops.py --workers 4          # parallélise sur des process (fork)
    python 11_optimize_options_stops.py --min-trades 30 --top-n 10

Le CSV complet (une ligne par binôme, toutes les métriques) est écrit sous
data/backtest_options/optimize_<stratégie>_<horodatage>.csv -- pratique pour
retracer une heatmap ou recouper avec d'autres critères après coup.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
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

logger = logging.getLogger("optimize_options_stops")

# 10_backtest_options.py n'est pas un nom de module Python valide (commence
# par un chiffre) : import explicite par chemin de fichier, comme le fait
# déjà tests/test_options_engine.py pour le même besoin (réutiliser
# resolve_engine_settings au lieu de dupliquer sa logique de fallback).
_cli = importlib.import_module("10_backtest_options")

DEFAULT_STOP_LOSS_GRID = [-10.0, -15.0, -20.0, -25.0, -30.0, -35.0, -40.0, -50.0]
DEFAULT_TAKE_PROFIT_GRID = [20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0]
# Fractions du chemin vers la valeur théorique, pour les stratégies qui visent
# une convergence (cf. OptionsStrategy.targets_convergence). Au-delà de 1,0 on
# exige un DÉPASSEMENT de la valeur théorique -- inclus pour vérifier que
# l'optimum ne s'y trouve pas, pas parce que c'est recommandé.
DEFAULT_CONVERGENCE_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]

# Métriques pour lesquelles une valeur PLUS GRANDE est toujours meilleure --
# y compris max_drawdown_pct, négatif, où -10 (peu de perte) > -35. Un tri
# descendant est donc correct pour toutes, sans cas particulier.
OBJECTIVE_CHOICES = [
    "sharpe_ratio", "sortino_ratio", "calmar_ratio", "cagr_pct", "total_return_pct",
    "profit_factor", "win_rate_pct", "max_drawdown_pct",
]

# Rempli une fois par process AVANT de forker le pool (voir main()) : les
# workers héritent de ces objets par copy-on-write, sans repasser par le
# chargement disque ni par une sérialisation coûteuse d'un DataFrame par tâche.
_DATA: dict = {}


def _load_data(args: argparse.Namespace) -> dict:
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
        price_panel, universe, symbol=args.benchmark_symbol,
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


def _run_combo(
    stop_loss_pct: float,
    take_profit: float,
    strategy_name: str,
    strategy_params: dict,
    baseline_settings: dict,
    engine_kwargs: dict,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    take_profit_is_convergence: bool = False,
    train_fraction: Optional[float] = None,
) -> dict:
    """Une combinaison = un run complet. Isolé en fonction de module (plutôt
    que méthode/closure) pour rester picklable par ProcessPoolExecutor --
    mais les données volumineuses (_DATA) ne sont PAS des arguments : elles
    viennent du fork, voir le commentaire sur _DATA plus haut.

    `take_profit` porte selon le mode soit un seuil fixe en % (take_profit_pct),
    soit une fraction de convergence -- voir --take-profit-mode."""
    row = {"stop_loss_pct": stop_loss_pct, "take_profit": take_profit, "error": None}
    try:
        strategy_cls = OPTIONS_STRATEGY_REGISTRY[strategy_name]
        strategy = strategy_cls(**strategy_params)
        settings = {**baseline_settings, "stop_loss_pct": stop_loss_pct}
        if take_profit_is_convergence:
            engine_kwargs = {**engine_kwargs, "take_profit_convergence_fraction": take_profit}
        else:
            settings["take_profit_pct"] = take_profit

        engine = OptionsBacktestEngine(
            price_panel=_DATA["price_panel"],
            signal_events=_DATA["signal_events"],
            universe_history=_DATA["universe_history"],
            fallback_universe_symbols=_DATA["fallback_symbols"],
            option_snapshots=_DATA["option_snapshots"],
            strategy=strategy,
            material_events_8k=_DATA["material_events"],
            stop_loss_pct=settings["stop_loss_pct"],
            take_profit_pct=settings["take_profit_pct"],
            target_tenor_days=settings["target_tenor_days"],
            stop_basis=settings["stop_basis"],
            exit_when_signal_lost=settings["exit_when_signal_lost"],
            roll_when_days_left=settings["roll_when_days_left"],
            daily_rebalance=settings["daily_rebalance"],
            vol_mode=settings["vol_mode"],
            min_resize_relative_pct=settings["min_resize_relative_pct"],
            start_date=start_date,
            end_date=end_date,
            **engine_kwargs,
        )
        equity_curve, _positions_history, trades, _signals_history = engine.run()
        run_metrics = metrics_mod.compute_metrics(
            equity_curve, trades,
            risk_free_rate=config.RISK_FREE_RATE,
            benchmark_prices=_DATA["benchmark_prices"],
            extra=engine.execution_diagnostics(),
        )
        for key in (
            "num_trades", "win_rate_pct", "avg_trade_return_pct", "profit_factor",
            "total_return_pct", "cagr_pct", "annualized_volatility_pct",
            "sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "calmar_ratio",
            "avg_holding_days", "alpha_pct", "avg_exposure_pct", "truncated_orders_pct",
        ):
            row[key] = run_metrics.get(key)
        # Séparation apprentissage / test : c'est le seul chiffre qui dit si
        # l'optimum de la grille survit à des données qui n'ont pas servi à le
        # choisir (cf. metrics.split_period_metrics).
        if train_fraction:
            row.update(metrics_mod.split_period_metrics(
                equity_curve, trades,
                metrics_mod.resolve_split_date(equity_curve, train_fraction),
                risk_free_rate=config.RISK_FREE_RATE,
                benchmark_prices=_DATA["benchmark_prices"],
            ))
        if not trades.empty and "exit_reason" in trades.columns:
            row["exits_by_reason"] = trades["exit_reason"].value_counts().to_dict()
    except Exception as exc:  # noqa: BLE001 -- une combinaison qui plante ne doit pas tuer la grille
        logger.exception("Échec pour stop_loss=%s take_profit=%s", stop_loss_pct, take_profit)
        row["error"] = str(exc)
    return row


def rank_results(results: pd.DataFrame, objective: str, min_trades: int) -> pd.DataFrame:
    """Classe les combinaisons par `objective` (ordre décroissant, voir
    OBJECTIVE_CHOICES), en écartant celles trop peu tradées pour être
    significatives ET celles en erreur -- mais sans les supprimer du DataFrame
    complet retourné par l'appelant, seulement de ce classement.

    Le classement se fait sur la PÉRIODE D'APPRENTISSAGE (`train_<objectif>`)
    dès qu'elle existe : choisir sur l'historique complet puis lire la
    performance sur ce même historique ne mesure rien d'autre que la capacité
    de la grille à s'y ajuster. La colonne `test_<objectif>` reste affichée à
    côté, et c'est elle qu'il faut regarder."""
    usable = results[results["error"].isna()].copy()
    if "num_trades" in usable.columns:
        too_few = usable[usable["num_trades"].fillna(0) < min_trades]
        if not too_few.empty:
            logger.info(
                "%d/%d combinaisons écartées du classement (moins de %d trades) : "
                "risque de sur-ajustement à un échantillon trop petit.",
                len(too_few), len(usable), min_trades,
            )
        usable = usable[usable["num_trades"].fillna(0) >= min_trades]
    ranking_key = f"train_{objective}"
    if ranking_key not in usable.columns or usable[ranking_key].isna().all():
        ranking_key = objective
    return usable.sort_values(ranking_key, ascending=False, na_position="last")


def print_heatmap(results: pd.DataFrame, objective: str) -> None:
    """Grille stop_loss (lignes) x take_profit (colonnes) -> valeur de
    l'objectif, pour repérer d'un coup d'œil un plateau ou un optimum en
    bord de grille (signe qu'il faut l'élargir)."""
    if results.empty:
        return
    pivot = results.pivot_table(index="stop_loss_pct", columns="take_profit", values=objective)
    pivot = pivot.sort_index(ascending=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50, "display.float_format", "{:.2f}".format):
        print(f"\n=== Heatmap {objective} (lignes=stop_loss_pct, colonnes=take_profit) ===")
        print(pivot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="valuation_gap_options", choices=sorted(OPTIONS_STRATEGY_REGISTRY))
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--stop-loss-grid", type=float, nargs="+", default=None, help="Valeurs négatives, ex: -10 -20 -30.")
    parser.add_argument(
        "--take-profit-grid", type=float, nargs="+", default=None,
        help="Unité selon --take-profit-mode : des %% de mouvement du sous-jacent en mode 'pct', "
             "des FRACTIONS de convergence (ex: 0.6 0.8 1.0) en mode 'convergence'.",
    )
    parser.add_argument(
        "--take-profit-mode", choices=("auto", "pct", "convergence"), default="auto",
        help="Quel paramètre de prise de gain balayer. 'auto' (défaut) choisit 'convergence' pour "
             "les stratégies qui visent une valeur théorique et 'pct' pour les autres : sur une "
             "stratégie de convergence, take_profit_pct est INERTE et le balayer ne mesurerait rien.",
    )
    parser.add_argument("--objective", default="sharpe_ratio", choices=OBJECTIVE_CHOICES)
    parser.add_argument(
        "--train-fraction", type=float, default=0.60,
        help="Part initiale de l'historique servant à CHOISIR la combinaison ; le reste sert "
             "à la juger (défaut: %(default)s). Le classement se fait sur train_<objectif>, et "
             "le CSV comme le tableau affichent test_<objectif> à côté -- c'est ce second "
             "chiffre qui dit si l'optimum survit à des données qui ne l'ont pas choisi.",
    )
    parser.add_argument(
        "--no-walk-forward", dest="train_fraction", action="store_const", const=None,
        help="Classe sur l'historique COMPLET (comportement d'avant). Le résultat est alors "
             "un optimum in-sample, à ne pas lire comme une performance attendue.",
    )
    parser.add_argument("--min-trades", type=int, default=15, help="Combinaisons avec moins de trades écartées du classement (pas du CSV).")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--workers", type=int, default=1, help="Process en parallèle (fork). 1 = séquentiel.")
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
    parser.add_argument(
        "--max-trade-pct-of-nav", type=float, default=config.OPTIONS_MAX_TRADE_PCT_OF_NAV,
        help="Montant maximal décaissé par ORDRE D'ACHAT, en %% du NAV (défaut: %(default)s). Remplace le plafond en dollars absolus, qui ne suivait pas la taille du portefeuille. 0 pour désactiver.",
    )
    parser.add_argument("--max-trade-dollar", type=float, default=config.OPTIONS_MAX_TRADE_DOLLAR)
    parser.add_argument(
        "--fractional-contracts", dest="whole_contracts", action="store_false",
        default=config.OPTIONS_WHOLE_CONTRACTS,
    )
    parser.add_argument("--real-snapshot-tolerance-days", type=int, default=config.OPTIONS_REAL_SNAPSHOT_TOLERANCE_DAYS)
    parser.add_argument("--momentum-min-pct", type=float, default=config.BACKTEST_MOMENTUM_MIN_PCT)
    parser.add_argument("--no-momentum-filter", dest="momentum_min_pct", action="store_const", const=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    stop_loss_grid = args.stop_loss_grid or DEFAULT_STOP_LOSS_GRID
    if any(v >= 0 for v in stop_loss_grid):
        logger.warning("stop_loss_grid contient des valeurs >= 0 : un stop-loss se définit normalement en négatif.")

    strategy_cls = OPTIONS_STRATEGY_REGISTRY[args.strategy]

    # Sur une stratégie de convergence, c'est la fraction de convergence qui
    # déclenche la prise de gain : balayer take_profit_pct produirait une
    # grille entière de résultats IDENTIQUES, qu'on lirait comme un plateau
    # alors que c'est simplement un paramètre sans effet.
    convergence_strategy = getattr(strategy_cls, "targets_convergence", False)
    mode = args.take_profit_mode
    if mode == "auto":
        mode = "convergence" if convergence_strategy else "pct"
    if mode == "convergence" and not convergence_strategy:
        raise SystemExit(
            f"--take-profit-mode convergence demandé, mais '{args.strategy}' ne vise pas de valeur "
            f"théorique : sa prise de gain passe par take_profit_pct (mode 'pct')."
        )
    if mode == "pct" and convergence_strategy:
        logger.warning(
            "'%s' vise une convergence : take_profit_pct y est INERTE tant que "
            "--take-profit-convergence-fraction n'est pas mis à 0. Les résultats de cette "
            "grille seront identiques d'une colonne à l'autre.", args.strategy,
        )
    take_profit_is_convergence = mode == "convergence"
    take_profit_grid = args.take_profit_grid or (
        DEFAULT_CONVERGENCE_GRID if take_profit_is_convergence else DEFAULT_TAKE_PROFIT_GRID
    )
    logger.info(
        "Prise de gain balayée en mode '%s' (%s).",
        mode, "fraction de convergence" if take_profit_is_convergence else "%% du sous-jacent",
    )

    # Réglages moteur autres que stop_loss/take_profit, résolus UNE FOIS via
    # la même logique que 10_backtest_options.py (engine_defaults de la
    # stratégie sinon fallback config) -- seuls stop_loss_pct/take_profit_pct
    # varient ensuite d'une combinaison à l'autre.
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
    _DATA = _load_data(args)

    grid = list(itertools.product(stop_loss_grid, take_profit_grid))
    logger.info(
        "Backtest options '%s' : %d combinaisons (%d stop-loss x %d take-profit)...",
        args.strategy, len(grid), len(stop_loss_grid), len(take_profit_grid),
    )

    unite = "" if take_profit_is_convergence else "%"
    rows: list[dict] = []
    if args.workers <= 1:
        for i, (sl, tp) in enumerate(grid, 1):
            logger.info("[%d/%d] stop_loss=%s%% take_profit=%s%s", i, len(grid), sl, tp, unite)
            rows.append(_run_combo(
                sl, tp, args.strategy, strategy_params, baseline_settings, engine_kwargs,
                start_date, end_date, take_profit_is_convergence, args.train_fraction,
            ))
    else:
        # ProcessPoolExecutor par défaut utilise fork() sur Linux : les
        # workers héritent de _DATA (déjà rempli ci-dessus) par
        # copy-on-write, sans le repasser en argument ni le recharger.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _run_combo, sl, tp, args.strategy, strategy_params,
                    baseline_settings, engine_kwargs, start_date, end_date,
                    take_profit_is_convergence, args.train_fraction,
                ): (sl, tp)
                for sl, tp in grid
            }
            for i, future in enumerate(as_completed(futures), 1):
                sl, tp = futures[future]
                logger.info("[%d/%d] terminé : stop_loss=%s%% take_profit=%s%s", i, len(grid), sl, tp, unite)
                rows.append(future.result())

    results = pd.DataFrame(rows).sort_values(["stop_loss_pct", "take_profit"]).reset_index(drop=True)
    results.insert(2, "take_profit_mode", mode)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_csv or (config.DIR_BACKTEST_OPTIONS / f"optimize_{args.strategy}_{run_id}.csv")
    out_path = str(out_path)
    config.DIR_BACKTEST_OPTIONS.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    logger.info("Résultats complets (%d lignes) sauvegardés dans %s", len(results), out_path)

    n_errors = results["error"].notna().sum()
    if n_errors:
        logger.warning("%d/%d combinaisons ont échoué (voir la colonne 'error' du CSV).", n_errors, len(results))

    ranked = rank_results(results, args.objective, args.min_trades)
    if ranked.empty:
        logger.error("Aucune combinaison exploitable (toutes en erreur ou sous le seuil --min-trades).")
        sys.exit(1)

    display_cols = [
        "stop_loss_pct", "take_profit", "num_trades", "win_rate_pct", "profit_factor",
        "cagr_pct", "annualized_volatility_pct", "sharpe_ratio", "sortino_ratio",
        "max_drawdown_pct", "calmar_ratio", "avg_holding_days", "alpha_pct",
        # Apprentissage / test côte à côte : le classement se fait sur la
        # première, la seconde est celle qui compte.
        f"train_{args.objective}", f"test_{args.objective}",
    ]
    display_cols = [c for c in display_cols if c in ranked.columns]
    with pd.option_context("display.width", 240, "display.max_columns", 40, "display.float_format", "{:.2f}".format):
        titre = (
            f"classé sur train_{args.objective}" if f"train_{args.objective}" in ranked.columns
            else f"classé sur {args.objective} — IN-SAMPLE"
        )
        print(f"\n=== Top {args.top_n} combinaisons ({titre}, stratégie: {args.strategy}) ===")
        print(ranked[display_cols].head(args.top_n).to_string(index=False))

    print_heatmap(results, args.objective)

    best = ranked.iloc[0]
    at_stop_edge = best["stop_loss_pct"] in (min(stop_loss_grid), max(stop_loss_grid))
    at_tp_edge = best["take_profit"] in (min(take_profit_grid), max(take_profit_grid))
    if at_stop_edge or at_tp_edge:
        logger.warning(
            "L'optimum (stop_loss=%s%%, take_profit=%s%%) est en BORD de grille -- "
            "élargis --stop-loss-grid/--take-profit-grid pour vérifier qu'il ne s'agit "
            "pas d'un optimum tronqué par la plage testée.",
            best["stop_loss_pct"], best["take_profit"],
        )

    # LE TEST QUI COMPTE : la combinaison retenue tient-elle hors échantillon ?
    train_key, test_key = f"train_{args.objective}", f"test_{args.objective}"
    if train_key in ranked.columns and test_key in ranked.columns:
        train_value, test_value = best.get(train_key), best.get(test_key)
        if pd.notna(train_value) and pd.notna(test_value):
            logger.info(
                "Hors échantillon (coupure au %s) : %s = %.3f en apprentissage -> %.3f en test.",
                best.get("split_date", "?"), args.objective, train_value, test_value,
            )
            if test_value < 0 or test_value < train_value * 0.5:
                logger.warning(
                    "L'optimum ne SURVIT PAS hors échantillon (%s : %.3f -> %.3f). La "
                    "combinaison retenue décrit surtout la période qui l'a choisie : ne "
                    "l'utilise pas comme réglage de production sur la seule foi de ce run.",
                    args.objective, train_value, test_value,
                )
    else:
        logger.warning(
            "Classement IN-SAMPLE (--no-walk-forward) : la combinaison retenue est celle qui "
            "colle le mieux à l'historique qui a servi à la choisir, ce qui n'est pas une "
            "performance attendue. Relance sans --no-walk-forward pour la juger."
        )

    logger.info(
        "Meilleure combinaison (%s=%.3f) : stop_loss=%s%%, take_profit=%s%%, "
        "%d trades, sharpe=%.2f, calmar=%.2f, max_drawdown=%.1f%%.",
        args.objective, best[args.objective], best["stop_loss_pct"], best["take_profit"],
        best["num_trades"], best.get("sharpe_ratio") or float("nan"),
        best.get("calmar_ratio") or float("nan"), best.get("max_drawdown_pct") or float("nan"),
    )


if __name__ == "__main__":
    main()
