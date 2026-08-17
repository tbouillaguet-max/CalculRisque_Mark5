"""
Grid-search sur `convergence_fraction` pour la stratégie « espérance de gain ».

Rejoue le backtest de 10_backtest_options.py pour CHAQUE valeur de la fraction
de convergence, sur les MÊMES données chargées UNE SEULE FOIS (comme
11b_optimize_rebalance_threshold.py, dont ce script reprend la structure). Tous
les autres réglages -- moteur ET stratégie -- sont FIXÉS à leurs valeurs
courantes, résolus comme dans 10_backtest_options.py (engine_defaults de la
stratégie sinon config) : la fraction est le SEUL paramètre qui varie d'un run
à l'autre.

CE QUE MESURE LA FRACTION. La stratégie traduit la thèse de valorisation en une
dérive annualisée mu = fraction x ln(V / S0) / T, puis choisit le strike qui
maximise le taux de croissance log-optimal du contrat sous cette dérive (voir
backtest/strategies/valuation_gap_expected_value_options.py). La fraction dit
donc quelle PART du chemin vers la valeur théorique on suppose parcourue à
l'échéance :

    fraction = 1,0  le cours atteint exactement sa valeur théorique -- hypothèse
                    que rien n'étaye, et qui transformerait chaque écart de
                    valorisation en gain certain ;
    fraction = 0,5  la moitié du chemin, soit l'hypothèse déjà implicite dans
                    valuation_gap_multiples_options (dont le strike est posé à
                    mi-chemin), rendue ici explicite ;
    fraction = 0,2  convergence marginale -- la dérive est faible, Kelly refuse
                    de payer de la convexité et se rabat sur des strikes dans la
                    monnaie, voire n'ouvre rien du tout.

Le dernier cas est la raison d'être de la colonne
`dropped_expected_value_negative_count` : à fraction basse, la contrainte
d'espérance positive écarte de plus en plus de lignes, et une fraction qui
n'ouvrirait presque aucune position peut afficher un excellent Sharpe sur trois
trades. Le CSV porte donc num_trades à côté du Sharpe, et ce script refuse de
recommander une fraction sous --min-trades.

PAS DE WALK-FORWARD à ce stade (contrairement à 11 et 11b, qui séparent
apprentissage et test) : première itération, exécution simple, in-sample
assumé. Une fraction choisie ici est donc choisie sur les données qui servent
ensuite à la juger -- le chiffre est un point de départ, pas une validation.

Pour chaque fraction, ce script enregistre :
    - cagr_pct, sharpe_ratio, max_drawdown_pct, calmar_ratio, win_rate_pct
    - num_trades total, et par motif (exits_by_reason)
    - total_friction_dollar / total_friction_pct_of_initial
    - les compteurs propres à la stratégie : lignes écartées pour espérance
      négative, part des ouvertures faites sur volatilité implicite réelle

Usage :
    python 11c_optimize_convergence_fraction.py
    python 11c_optimize_convergence_fraction.py --fraction-grid 0.3 0.5 0.7 1.0
    python 11c_optimize_convergence_fraction.py --start-date 2015-01-01
    python 11c_optimize_convergence_fraction.py --workers 4   # parallélise sur des process (fork)

Le CSV complet (une ligne par fraction, toutes les métriques) est écrit sous
data/backtest_options/optimize_convergence_<stratégie>_<horodatage>.csv.
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

logger = logging.getLogger("optimize_convergence_fraction")

# 10_backtest_options.py n'est pas un nom de module Python valide (commence
# par un chiffre) : import explicite par chemin de fichier, comme le font déjà
# 11_optimize_options_stops.py et 11b_optimize_rebalance_threshold.py.
_cli = importlib.import_module("10_backtest_options")

DEFAULT_FRACTION_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

DEFAULT_STRATEGY = "valuation_gap_expected_value_options"

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


def _init_worker(benchmark_symbol: str) -> None:
    """Garantit que _DATA est rempli DANS le worker, quelle que soit la façon
    dont le pool a démarré le process.

    Sous fork (défaut Linux), le worker hérite du _DATA déjà rempli par le
    parent : il n'y a rien à charger, et le test ci-dessous court-circuite.
    Sous spawn (défaut Windows, et macOS depuis 3.8), le worker ré-importe le
    module à neuf -- _DATA y vaut {} et CHAQUE tâche échouerait en
    KeyError: 'price_panel'. C'est exactement ce qui se produisait tant que ce
    module supposait le fork."""
    global _DATA
    if _DATA:
        return
    _DATA = _load_data(benchmark_symbol)


def _run_one(
    fraction: float,
    strategy_name: str,
    strategy_params: dict,
    baseline_settings: dict,
    engine_kwargs: dict,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> dict:
    """Un run complet pour cette fraction. Isolé en fonction de module (plutôt
    que méthode/closure) pour rester picklable par ProcessPoolExecutor -- les
    données volumineuses (_DATA) ne sont PAS des arguments : elles viennent du
    fork, voir le commentaire sur _DATA plus haut."""
    row = {"convergence_fraction": fraction, "error": None}
    try:
        strategy_cls = OPTIONS_STRATEGY_REGISTRY[strategy_name]
        strategy = strategy_cls(**{**strategy_params, "convergence_fraction": fraction})

        engine = OptionsBacktestEngine(
            price_panel=_DATA["price_panel"],
            signal_events=_DATA["signal_events"],
            universe_history=_DATA["universe_history"],
            fallback_universe_symbols=_DATA["fallback_symbols"],
            option_snapshots=_DATA["option_snapshots"],
            material_events_8k=_DATA["material_events"],
            strategy=strategy,
            stop_loss_pct=baseline_settings["stop_loss_pct"],
            take_profit_pct=baseline_settings["take_profit_pct"],
            target_tenor_days=baseline_settings["target_tenor_days"],
            stop_basis=baseline_settings["stop_basis"],
            exit_when_signal_lost=baseline_settings["exit_when_signal_lost"],
            roll_when_days_left=baseline_settings["roll_when_days_left"],
            daily_rebalance=baseline_settings["daily_rebalance"],
            vol_mode=baseline_settings["vol_mode"],
            min_resize_relative_pct=baseline_settings["min_resize_relative_pct"],
            min_holding_days=baseline_settings["min_holding_days"],
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
            # Compteurs propres à la stratégie (cf. options_engine._strategy_diagnostics) :
            # ils disent si la contrainte d'espérance positive a MORDU, et sur
            # quelle volatilité les contrats ont été choisis.
            "expected_value_evaluations_count", "dropped_expected_value_negative_count",
            "expected_value_implied_vol_pct", "expected_value_quoted_grid_count",
        ):
            row[key] = run_metrics.get(key)

        if not trades.empty and "exit_reason" in trades.columns:
            row["exits_by_reason"] = trades["exit_reason"].value_counts().to_dict()
    except Exception as exc:  # noqa: BLE001 -- une fraction qui plante ne doit pas tuer la grille
        logger.exception("Échec pour convergence_fraction=%s", fraction)
        row["error"] = str(exc)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, choices=sorted(OPTIONS_STRATEGY_REGISTRY))
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument(
        "--fraction-grid", type=float, nargs="+", default=None,
        help="Fractions de convergence à tester, dans ]0, 1] (défaut: "
             f"{DEFAULT_FRACTION_GRID}). 1,0 suppose la convergence complète.",
    )
    parser.add_argument(
        "--min-trades", type=int, default=20,
        help="Nombre de trades sous lequel une fraction n'est pas recommandable, quel que "
             "soit son Sharpe (défaut: %(default)s). Une fraction basse écarte beaucoup de "
             "lignes pour espérance négative et peut afficher un Sharpe flatteur sur une "
             "poignée de positions.",
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
    parser.add_argument(
        "--max-trade-pct-of-nav", type=float, default=config.OPTIONS_MAX_TRADE_PCT_OF_NAV,
        help="Montant maximal décaissé par ORDRE D'ACHAT, en %% du NAV (défaut: %(default)s). 0 pour désactiver.",
    )
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

    fraction_grid = sorted(args.fraction_grid or DEFAULT_FRACTION_GRID)
    hors_bornes = [f for f in fraction_grid if not 0.0 < f <= 1.0]
    if hors_bornes:
        raise SystemExit(
            f"--fraction-grid attend des valeurs dans ]0, 1], reçu {hors_bornes}. "
            "0 supprimerait toute dérive (aucune thèse), au-delà de 1 supposerait un "
            "dépassement de la valeur théorique."
        )

    strategy_cls = OPTIONS_STRATEGY_REGISTRY[args.strategy]
    if not hasattr(strategy_cls, "convergence_fraction") and args.strategy != DEFAULT_STRATEGY:
        logger.warning(
            "La stratégie '%s' n'expose pas convergence_fraction : le paramètre sera "
            "transmis à son constructeur et échouera probablement. Ce script vise '%s'.",
            args.strategy, DEFAULT_STRATEGY,
        )

    # Réglages moteur FIXÉS pour toute la grille, résolus UNE FOIS via la même
    # logique que 10_backtest_options.py (engine_defaults de la stratégie
    # sinon fallback config).
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

    logger.info(
        "Backtest options '%s' : %d fractions de convergence %s...",
        args.strategy, len(fraction_grid), fraction_grid,
    )

    rows: list[dict] = []
    if args.workers <= 1:
        for i, fraction in enumerate(fraction_grid, 1):
            logger.info("[%d/%d] convergence_fraction=%s", i, len(fraction_grid), fraction)
            rows.append(_run_one(
                fraction, args.strategy, strategy_params, baseline_settings, engine_kwargs,
                start_date, end_date,
            ))
    else:
        # initializer plutôt que fork implicite : sous fork le worker hérite
        # de _DATA et _init_worker ne fait rien, sous spawn il le charge. Sans
        # cela, tout le pool échoue sur les plateformes qui ne forkent pas.
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.benchmark_symbol,),
        ) as pool:
            futures = {
                pool.submit(
                    _run_one, fraction, args.strategy, strategy_params,
                    baseline_settings, engine_kwargs, start_date, end_date,
                ): fraction
                for fraction in fraction_grid
            }
            for i, future in enumerate(as_completed(futures), 1):
                fraction = futures[future]
                logger.info("[%d/%d] terminé : convergence_fraction=%s", i, len(fraction_grid), fraction)
                rows.append(future.result())

    results = pd.DataFrame(rows).sort_values("convergence_fraction").reset_index(drop=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_csv or (
        config.DIR_BACKTEST_OPTIONS / f"optimize_convergence_{args.strategy}_{run_id}.csv"
    )
    out_path = str(out_path)
    config.DIR_BACKTEST_OPTIONS.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    logger.info("Résultats complets (%d lignes) sauvegardés dans %s", len(results), out_path)

    n_errors = results["error"].notna().sum()
    if n_errors:
        logger.warning("%d/%d fractions ont échoué (voir la colonne 'error' du CSV).", n_errors, len(results))

    display_cols = [
        "convergence_fraction", "num_trades", "win_rate_pct",
        "cagr_pct", "sharpe_ratio", "max_drawdown_pct", "calmar_ratio",
        "total_friction_pct_of_initial",
        "dropped_expected_value_negative_count", "expected_value_implied_vol_pct",
    ]
    display_cols = [c for c in display_cols if c in results.columns]
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:,.3f}".format):
        print(f"\n=== convergence_fraction -- stratégie: {args.strategy} ===")
        print(results[display_cols].to_string(index=False))

    usable = results[results["error"].isna()].copy()
    if usable.empty:
        logger.error("Aucune fraction exploitable (toutes en erreur).")
        sys.exit(1)

    ranked = usable.sort_values("sharpe_ratio", ascending=False, na_position="last")
    with pd.option_context("display.width", 220, "display.float_format", "{:,.3f}".format):
        print(f"\n=== Top {min(args.top_n, len(ranked))} par sharpe_ratio ===")
        print(ranked[display_cols].head(args.top_n).to_string(index=False))

    # Une fraction qui n'ouvre presque rien n'est pas comparable aux autres :
    # son Sharpe porte sur une poignée de trades. On la signale plutôt que de
    # la recommander en silence.
    credibles = ranked[ranked["num_trades"].fillna(0) >= args.min_trades]
    if credibles.empty:
        logger.error(
            "Aucune fraction n'atteint %d trades : la contrainte d'espérance positive "
            "écarte l'essentiel des lignes sur cette période. Élargis la période, baisse "
            "--entry-threshold-pct, ou abaisse --min-trades en connaissance de cause.",
            args.min_trades,
        )
        sys.exit(1)

    best = credibles.iloc[0]
    if len(credibles) < len(ranked):
        logger.warning(
            "%d fraction(s) écartée(s) du classement pour moins de %d trades.",
            len(ranked) - len(credibles), args.min_trades,
        )

    if best["convergence_fraction"] in (min(fraction_grid), max(fraction_grid)):
        logger.warning(
            "L'optimum (fraction=%s) est en BORD de grille -- élargis --fraction-grid pour "
            "vérifier qu'il ne s'agit pas d'un optimum tronqué par la plage testée. "
            "Attention : 1,0 est un bord DUR (convergence complète), pas une troncature.",
            best["convergence_fraction"],
        )

    logger.info(
        "Meilleure fraction par Sharpe : %.2f (sharpe=%.2f, %d trades, CAGR=%.1f%%, "
        "max_drawdown=%.1f%%, %d lignes écartées pour espérance négative).",
        best["convergence_fraction"], best.get("sharpe_ratio") or float("nan"),
        best.get("num_trades") or 0, best.get("cagr_pct") or float("nan"),
        best.get("max_drawdown_pct") or float("nan"),
        best.get("dropped_expected_value_negative_count") or 0,
    )
    logger.info(
        "Rappel : ce classement est IN-SAMPLE (pas de walk-forward à ce stade). "
        "La fraction retenue a été choisie sur les données qui servent à la juger."
    )


if __name__ == "__main__":
    main()
