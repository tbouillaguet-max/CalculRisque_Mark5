"""
Backtest d'une stratégie OPTIONS construite sur la valorisation combinée
(multiples sectoriels par année en priorité, DCF en repli -- voir
06b_calcul_valorisation_combinee.py) : achète des CALL sur les entreprises
sous-évaluées, des PUT sur les survalorisées, dimensionnés par le delta pour
une exposition $ cible. Voir backtest/options_engine.py pour les hypothèses
détaillées (repli Black-Scholes, positions gelées, stop-loss/take-profit sur
la prime, expiration).

Distinct de 09_backtest.py (stratégie actions, DCF seul, inchangée).

Prérequis : 01(b), 02, 03b, 04, 05, 07, 06b_calcul_valorisation_combinee.py.
08_recuperation_options.py est optionnel mais recommandé : plus tu accumules
de snapshots réels (data/options/history/), moins le backtest s'appuie sur
du Black-Scholes simulé.

Usage :
    python 10_backtest_options.py
    python 10_backtest_options.py --start-date 2015-01-01
    python 10_backtest_options.py --entry-threshold-pct 25 --stop-loss-pct -40 --take-profit-pct 80
    python 10_backtest_options.py --list-strategies
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

import pandas as pd

import config
from backtest import data_loader, metrics as metrics_mod
from backtest.options_engine import OptionsBacktestEngine
from backtest.strategies import OPTIONS_STRATEGY_REGISTRY

logger = logging.getLogger("backtest_options.cli")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="valuation_gap_options")
    parser.add_argument("--list-strategies", action="store_true")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--initial-capital", type=float, default=config.OPTIONS_INITIAL_CAPITAL)
    parser.add_argument("--commission-per-contract", type=float, default=config.OPTIONS_COMMISSION_PER_CONTRACT)
    parser.add_argument("--slippage-pct-of-premium", type=float, default=config.OPTIONS_SLIPPAGE_PCT_OF_PREMIUM)
    parser.add_argument("--stop-loss-pct", type=float, default=config.OPTIONS_STOP_LOSS_PCT, help="Sur la prime, négatif, ex: -50.")
    parser.add_argument("--take-profit-pct", type=float, default=config.OPTIONS_TAKE_PROFIT_PCT)
    parser.add_argument("--max-positions", type=int, default=config.OPTIONS_MAX_POSITIONS)
    parser.add_argument("--entry-threshold-pct", type=float, default=config.OPTIONS_ENTRY_THRESHOLD_PCT)
    parser.add_argument("--target-tenor-days", type=int, default=config.OPTIONS_TARGET_TENOR_DAYS)
    parser.add_argument("--real-snapshot-tolerance-days", type=int, default=config.OPTIONS_REAL_SNAPSHOT_TOLERANCE_DAYS)
    parser.add_argument("--strategy-param", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.list_strategies:
        for name in sorted(OPTIONS_STRATEGY_REGISTRY):
            print(name)
        return

    if args.strategy not in OPTIONS_STRATEGY_REGISTRY:
        logger.error("Stratégie options inconnue: %s. Disponibles: %s", args.strategy, sorted(OPTIONS_STRATEGY_REGISTRY))
        sys.exit(1)

    logger.info("Chargement des données...")
    daily_prices = data_loader.load_daily_prices()
    price_panel = data_loader.build_price_panel(daily_prices)
    valorisation_combinee = data_loader.load_valorisation_combinee_history()
    signal_events = data_loader.build_options_signal_events(valorisation_combinee)
    universe_history = data_loader.load_universe_history()
    fallback_symbols = data_loader.load_current_universe_symbols()
    option_snapshots = data_loader.load_option_snapshots_history()

    strategy_cls = OPTIONS_STRATEGY_REGISTRY[args.strategy]
    strategy_params = {"entry_threshold_pct": args.entry_threshold_pct, "max_positions": args.max_positions}
    strategy_params.update(parse_strategy_params(args.strategy_param))
    strategy = strategy_cls(**strategy_params)

    start_date = pd.Timestamp(args.start_date) if args.start_date else None
    end_date = pd.Timestamp(args.end_date) if args.end_date else None

    engine = OptionsBacktestEngine(
        price_panel=price_panel,
        signal_events=signal_events,
        universe_history=universe_history,
        fallback_universe_symbols=fallback_symbols,
        option_snapshots=option_snapshots,
        strategy=strategy,
        initial_capital=args.initial_capital,
        commission_per_contract=args.commission_per_contract,
        slippage_pct_of_premium=args.slippage_pct_of_premium,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        max_positions=args.max_positions,
        target_tenor_days=args.target_tenor_days,
        real_snapshot_tolerance_days=args.real_snapshot_tolerance_days,
        start_date=start_date,
        end_date=end_date,
    )

    logger.info(
        "Backtest options '%s' du %s au %s (%d jours de bourse)...",
        args.strategy, engine.calendar[0].date(), engine.calendar[-1].date(), len(engine.calendar),
    )
    equity_curve, positions_history, trades, signals_history = engine.run()
    run_metrics = metrics_mod.compute_metrics(equity_curve, trades, risk_free_rate=config.RISK_FREE_RATE)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = config.DIR_BACKTEST_OPTIONS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_curve.to_parquet(out_dir / "equity_curve.parquet", index=False, engine="pyarrow")
    positions_history.to_parquet(out_dir / "positions_history.parquet", index=False, engine="pyarrow")
    trades.to_parquet(out_dir / "trades.parquet", index=False, engine="pyarrow")
    signals_history.to_parquet(out_dir / "signals_history.parquet", index=False, engine="pyarrow")
    (out_dir / "metrics.json").write_text(json.dumps(run_metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps({
        "strategy": args.strategy, "strategy_params": strategy_params,
        "initial_capital": args.initial_capital, "commission_per_contract": args.commission_per_contract,
        "slippage_pct_of_premium": args.slippage_pct_of_premium, "stop_loss_pct": args.stop_loss_pct,
        "take_profit_pct": args.take_profit_pct, "max_positions": args.max_positions,
        "target_tenor_days": args.target_tenor_days, "real_snapshot_tolerance_days": args.real_snapshot_tolerance_days,
        "start_date": str(engine.calendar[0].date()), "end_date": str(engine.calendar[-1].date()),
        "has_pit_universe": universe_history is not None, "n_real_option_snapshots": len(option_snapshots),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Résultats sauvegardés dans %s", out_dir)
    logger.info("--- Résumé ---")
    for key in [
        "total_return_pct", "cagr_pct", "annualized_volatility_pct", "sharpe_ratio", "sortino_ratio",
        "max_drawdown_pct", "calmar_ratio", "num_trades", "win_rate_pct", "profit_factor", "avg_exposure_pct",
    ]:
        if key in run_metrics:
            logger.info("%s: %s", key, run_metrics[key])
    if not trades.empty and "exit_reason" in trades.columns:
        logger.info("Répartition des sorties: %s", trades["exit_reason"].value_counts().to_dict())
    if not positions_history.empty and "source" in positions_history.columns:
        logger.info("Positions par source: %s", positions_history.drop_duplicates("symbol")["source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
