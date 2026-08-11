"""
Grid-search sur ε (rebalance_log_gap_threshold) pour une stratégie options.

Rejoue le backtest de 10_backtest_options.py pour CHAQUE valeur de ε d'une
grille, sur les MÊMES données chargées UNE SEULE FOIS (comme
11_optimize_options_stops.py, dont ce script reprend la structure). Tous les
autres réglages du moteur (stop-loss, take-profit, échéance, base des stops,
roulement, etc.) sont FIXÉS à leurs valeurs courantes -- résolues comme dans
10_backtest_options.py (engine_defaults de la stratégie sinon config) -- ε
est le SEUL paramètre qui varie d'un run à l'autre. Pas de surface 2D : ce
script n'optimise que ε, cf. 11_optimize_options_stops.py pour le grid-search
sur (stop-loss, take-profit).

ε (config.OPTIONS_REBALANCE_LOG_GAP_THRESHOLD, 0.15 par défaut) contrôle le
mécanisme de rebalancement SUR DÉPÔT SEC (backtest/options_engine.py,
OptionsBacktestEngine._rebalance_on_signals) : une position déjà détenue
n'est redimensionnée que si |log(V/P) - last_rebalance_log_gap| dépasse ε
depuis le dernier trade réel sur cette position. ε=0 s'approche du
comportement historique (redimensionne dès que le signal a le moindre effet,
mais toujours scopé au seul symbole concerné -- voir la docstring
d'options_engine.py pour la différence avec l'ancien _rebalance, qui
redimensionnait TOUTES les positions détenues à chaque événement).

Pour chaque ε, ce script enregistre :
    - total_friction_dollar / total_friction_pct_of_initial (commission +
      slippage cumulés, cf. la session d'instrumentation de la friction)
    - num_trades total, et par motif (exits_by_reason)
    - num_rebalance_trades : trades dont le motif est "rebalance" -- le
      churn RÉSIDUEL du mécanisme sur dépôt SEC après filtrage par ε
    - cagr_pct, sharpe_ratio, max_drawdown_pct

Usage :
    python 11b_optimize_rebalance_threshold.py
    python 11b_optimize_rebalance_threshold.py --strategy valuation_gap_multiples_options
    python 11b_optimize_rebalance_threshold.py --epsilon-grid 0 0.05 0.10 0.15 0.20 0.30
    python 11b_optimize_rebalance_threshold.py --workers 4   # parallélise sur des process (fork)

Le CSV complet (une ligne par ε, toutes les métriques) est écrit sous
data/backtest_options/optimize_rebalance_<stratégie>_<horodatage>.csv.
"""

from __future__ import annotations

import argparse
import importlib
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

logger = logging.getLogger("optimize_rebalance_threshold")

# 10_backtest_options.py n'est pas un nom de module Python valide (commence
# par un chiffre) : import explicite par chemin de fichier, comme le fait
# déjà 11_optimize_options_stops.py pour le même besoin.
_cli = importlib.import_module("10_backtest_options")

DEFAULT_EPSILON_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

# Rempli une fois par process AVANT de forker le pool (voir main()) : les
# workers héritent de ces objets par copy-on-write, sans repasser par le
# chargement disque ni par une sérialisation coûteuse d'un DataFrame par tâche.
_DATA: dict = {}


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


def _run_one(
    epsilon: float,
    strategy_name: str,
    strategy_params: dict,
    baseline_settings: dict,
    engine_kwargs: dict,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> dict:
    """Un run complet pour cette valeur de ε. Isolé en fonction de module
    (plutôt que méthode/closure) pour rester picklable par
    ProcessPoolExecutor -- les données volumineuses (_DATA) ne sont PAS des
    arguments : elles viennent du fork, voir le commentaire sur _DATA
    plus haut."""
    row = {"rebalance_log_gap_threshold": epsilon, "error": None}
    try:
        strategy_cls = OPTIONS_STRATEGY_REGISTRY[strategy_name]
        strategy = strategy_cls(**strategy_params)

        engine = OptionsBacktestEngine(
            price_panel=_DATA["price_panel"],
            signal_events=_DATA["signal_events"],
            universe_history=_DATA["universe_history"],
            fallback_universe_symbols=_DATA["fallback_symbols"],
            option_snapshots=_DATA["option_snapshots"],
            material_events_8k=_DATA["material_events"],
            strategy=strategy,
            rebalance_log_gap_threshold=epsilon,
            stop_loss_pct=baseline_settings["stop_loss_pct"],
            take_profit_pct=baseline_settings["take_profit_pct"],
            target_tenor_days=baseline_settings["target_tenor_days"],
            stop_basis=baseline_settings["stop_basis"],
            exit_when_signal_lost=baseline_settings["exit_when_signal_lost"],
            roll_when_days_left=baseline_settings["roll_when_days_left"],
            daily_rebalance=baseline_settings["daily_rebalance"],
            vol_mode=baseline_settings["vol_mode"],
            min_resize_relative_pct=baseline_settings["min_resize_relative_pct"],
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
            "num_trades", "win_rate_pct", "profit_factor", "total_return_pct",
            "cagr_pct", "annualized_volatility_pct", "sharpe_ratio", "sortino_ratio",
            "max_drawdown_pct", "calmar_ratio", "avg_holding_days",
            "total_commission_dollar", "total_slippage_dollar",
            "total_friction_dollar", "total_friction_pct_of_initial",
        ):
            row[key] = run_metrics.get(key)

        if not trades.empty and "exit_reason" in trades.columns:
            counts = trades["exit_reason"].value_counts().to_dict()
            row["exits_by_reason"] = counts
            # Le churn RÉSIDUEL du mécanisme sur dépôt SEC après filtrage par
            # ε : c'est le chiffre qui dit si ε a réellement coupé le churn,
            # pas seulement déplacé le problème vers un autre motif de sortie.
            row["num_rebalance_trades"] = int(counts.get("rebalance", 0))
        else:
            row["num_rebalance_trades"] = 0
    except Exception as exc:  # noqa: BLE001 -- un ε qui plante ne doit pas tuer la grille
        logger.exception("Échec pour rebalance_log_gap_threshold=%s", epsilon)
        row["error"] = str(exc)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="valuation_gap_multiples_options", choices=sorted(OPTIONS_STRATEGY_REGISTRY))
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument(
        "--epsilon-grid", type=float, nargs="+", default=None,
        help="Valeurs de ε à tester, en points de log(V/P) (défaut: "
             f"{DEFAULT_EPSILON_GRID}). 0 s'approche du comportement sans filtre.",
    )
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

    epsilon_grid = sorted(args.epsilon_grid or DEFAULT_EPSILON_GRID)
    if any(e < 0 for e in epsilon_grid):
        raise SystemExit("--epsilon-grid ne peut pas contenir de valeur négative (ε mesure un écart absolu).")

    strategy_cls = OPTIONS_STRATEGY_REGISTRY[args.strategy]

    # Réglages moteur FIXÉS pour toute la grille, résolus UNE FOIS via la même
    # logique que 10_backtest_options.py (engine_defaults de la stratégie
    # sinon fallback config) -- rebalance_log_gap_threshold n'y figure pas
    # volontairement : c'est le seul paramètre qui varie, injecté séparément
    # par _run_one à chaque point de la grille.
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
    )

    global _DATA
    _DATA = _load_data(args.benchmark_symbol)

    logger.info("Backtest options '%s' : %d valeurs de ε %s...", args.strategy, len(epsilon_grid), epsilon_grid)

    rows: list[dict] = []
    if args.workers <= 1:
        for i, eps in enumerate(epsilon_grid, 1):
            logger.info("[%d/%d] rebalance_log_gap_threshold=%s", i, len(epsilon_grid), eps)
            rows.append(_run_one(
                eps, args.strategy, strategy_params, baseline_settings, engine_kwargs, start_date, end_date,
            ))
    else:
        # ProcessPoolExecutor par défaut utilise fork() sur Linux : les
        # workers héritent de _DATA (déjà rempli ci-dessus) par
        # copy-on-write, sans le repasser en argument ni le recharger.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _run_one, eps, args.strategy, strategy_params,
                    baseline_settings, engine_kwargs, start_date, end_date,
                ): eps
                for eps in epsilon_grid
            }
            for i, future in enumerate(as_completed(futures), 1):
                eps = futures[future]
                logger.info("[%d/%d] terminé : rebalance_log_gap_threshold=%s", i, len(epsilon_grid), eps)
                rows.append(future.result())

    results = pd.DataFrame(rows).sort_values("rebalance_log_gap_threshold").reset_index(drop=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_csv or (config.DIR_BACKTEST_OPTIONS / f"optimize_rebalance_{args.strategy}_{run_id}.csv")
    out_path = str(out_path)
    config.DIR_BACKTEST_OPTIONS.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    logger.info("Résultats complets (%d lignes) sauvegardés dans %s", len(results), out_path)

    n_errors = results["error"].notna().sum()
    if n_errors:
        logger.warning("%d/%d valeurs de ε ont échoué (voir la colonne 'error' du CSV).", n_errors, len(results))

    display_cols = [
        "rebalance_log_gap_threshold", "num_trades", "num_rebalance_trades",
        "total_friction_dollar", "total_friction_pct_of_initial",
        "cagr_pct", "sharpe_ratio", "max_drawdown_pct", "calmar_ratio",
    ]
    display_cols = [c for c in display_cols if c in results.columns]
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:,.3f}".format):
        print(f"\n=== ε (rebalance_log_gap_threshold) -- stratégie: {args.strategy} ===")
        print(results[display_cols].to_string(index=False))

    usable = results[results["error"].isna()].copy()
    if usable.empty:
        logger.error("Aucune valeur de ε exploitable (toutes en erreur).")
        sys.exit(1)

    ranked = usable.sort_values("sharpe_ratio", ascending=False, na_position="last")
    best = ranked.iloc[0]
    with pd.option_context("display.width", 220, "display.float_format", "{:,.3f}".format):
        print(f"\n=== Top {min(args.top_n, len(ranked))} par sharpe_ratio ===")
        print(ranked[display_cols].head(args.top_n).to_string(index=False))

    if best["rebalance_log_gap_threshold"] in (min(epsilon_grid), max(epsilon_grid)):
        logger.warning(
            "L'optimum (ε=%s) est en BORD de grille -- élargis --epsilon-grid pour vérifier "
            "qu'il ne s'agit pas d'un optimum tronqué par la plage testée.",
            best["rebalance_log_gap_threshold"],
        )
    logger.info(
        "Meilleur ε par Sharpe : %.3f (sharpe=%.2f, %d trades dont %d 'rebalance', "
        "friction=%.1f%% du capital initial, max_drawdown=%.1f%%).",
        best["rebalance_log_gap_threshold"], best.get("sharpe_ratio") or float("nan"),
        best.get("num_trades") or 0, best.get("num_rebalance_trades") or 0,
        best.get("total_friction_pct_of_initial") or float("nan"),
        best.get("max_drawdown_pct") or float("nan"),
    )


if __name__ == "__main__":
    main()
