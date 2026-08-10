"""
Décomposition de la perte : thèse contre friction contre churn.

Rejoue la même stratégie sur les 4 combinaisons du plan 2x2

    slippage x {réel, 0}   x   rebalancement quotidien x {activé, désactivé}

et met les résultats côte à côte. Les données ne sont chargées qu'une fois.

CE QUE CHAQUE CASE MESURE. Le run nominal (slippage réel, rebalancement
quotidien) est ce que la stratégie fait réellement. Le run "pure thèse"
(slippage 0, pas de rebalancement quotidien) est ce qu'elle vaudrait sans
aucun coût d'exécution ni sur-réaction aux mouvements de cours intra-
trimestre : c'est le rendement du SIGNAL seul.

    thèse    = rendement du run "pure thèse"
    friction = thèse - rendement à slippage réel, à rebalancement égal
    churn    = rendement à rebalancement désactivé - rendement à rebalancement
               quotidien, à slippage égal

Séparer les deux importe parce qu'ils ne se corrigent pas de la même façon :
la friction se réduit en tradant moins gros ou moins souvent (plafond par
ordre, seuil de resize), le churn en ne réagissant qu'aux publications. Tant
que les deux sont confondus dans un seul chiffre de perte, tout réglage se
fait à l'aveugle -- et c'est précisément pour rendre cette case lisible que
le moteur accumule désormais total_commission / total_slippage.

Le slippage n'est pas la seule friction : les commissions et frais tiers
restent payés dans TOUTES les cases (les annuler ne décrirait plus aucun
courtier réel). La colonne "pure thèse" est donc un plafond, pas un
contrefactuel atteignable.

Usage :
    python 13_diagnostic_friction.py
    python 13_diagnostic_friction.py --strategy valuation_gap_options --start-date 2015-01-01
    python 13_diagnostic_friction.py --export
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from datetime import datetime

import pandas as pd

import config
from backtest import data_loader, metrics as metrics_mod
from backtest.options_engine import OptionsBacktestEngine
from backtest.strategies import OPTIONS_STRATEGY_REGISTRY

logger = logging.getLogger("diagnostic_friction")

_cli = importlib.import_module("10_backtest_options")


def _load_data(benchmark_symbol: str) -> dict:
    logger.info("Chargement des données (une seule fois pour les 4 runs)...")
    price_panel = data_loader.build_price_panel(data_loader.load_daily_prices())
    signal_events = data_loader.build_options_signal_events(
        data_loader.load_valorisation_combinee_history()
    )
    universe_history = data_loader.load_universe_history()
    fallback_symbols = data_loader.load_current_universe_symbols()
    benchmark_prices, benchmark_label = data_loader.build_benchmark_series(
        price_panel, data_loader.UniverseResolver(universe_history, fallback_symbols),
        symbol=benchmark_symbol,
    )
    return {
        "price_panel": price_panel,
        "signal_events": signal_events,
        "universe_history": universe_history,
        "fallback_symbols": fallback_symbols,
        "option_snapshots": data_loader.load_option_snapshots_history(),
        "material_events": data_loader.load_material_events_8k(),
        "benchmark_prices": benchmark_prices,
        "benchmark_label": benchmark_label,
    }


def run_one(
    data: dict, strategy_cls, strategy_params: dict, settings: dict, engine_kwargs: dict,
    slippage_pct: float, daily_rebalance: bool, args: argparse.Namespace,
) -> dict:
    engine = OptionsBacktestEngine(
        price_panel=data["price_panel"],
        signal_events=data["signal_events"],
        universe_history=data["universe_history"],
        fallback_universe_symbols=data["fallback_symbols"],
        option_snapshots=data["option_snapshots"],
        material_events_8k=data["material_events"],
        strategy=strategy_cls(**strategy_params),
        slippage_pct_of_premium=slippage_pct,
        daily_rebalance=daily_rebalance,
        stop_loss_pct=settings["stop_loss_pct"],
        take_profit_pct=settings["take_profit_pct"],
        target_tenor_days=settings["target_tenor_days"],
        stop_basis=settings["stop_basis"],
        exit_when_signal_lost=settings["exit_when_signal_lost"],
        roll_when_days_left=settings["roll_when_days_left"],
        vol_mode=settings["vol_mode"],
        min_resize_relative_pct=settings["min_resize_relative_pct"],
        start_date=pd.Timestamp(args.start_date) if args.start_date else None,
        end_date=pd.Timestamp(args.end_date) if args.end_date else None,
        **engine_kwargs,
    )
    equity_curve, _positions, trades, _signals = engine.run()
    run_metrics = metrics_mod.compute_metrics(
        equity_curve, trades, risk_free_rate=config.RISK_FREE_RATE,
        benchmark_prices=data["benchmark_prices"], extra=engine.execution_diagnostics(),
    )
    initial_nav = float(equity_curve["nav"].iloc[0])
    pnl = float(equity_curve["nav"].iloc[-1]) - initial_nav
    friction = run_metrics.get("total_friction_dollar") or 0.0
    return {
        "slippage_pct": slippage_pct,
        "daily_rebalance": daily_rebalance,
        "pnl_dollar": pnl,
        "total_return_pct": run_metrics.get("total_return_pct"),
        "cagr_pct": run_metrics.get("cagr_pct"),
        "sharpe_ratio": run_metrics.get("sharpe_ratio"),
        "max_drawdown_pct": run_metrics.get("max_drawdown_pct"),
        "num_trades": run_metrics.get("num_trades"),
        "commission_dollar": run_metrics.get("total_commission_dollar"),
        "slippage_dollar": run_metrics.get("total_slippage_dollar"),
        "friction_dollar": friction,
        "friction_pct_of_initial": run_metrics.get("total_friction_pct_of_initial"),
        # Ce que la friction représente rapportée à la perte : le chiffre qui
        # dit "60% ou 20%". Non défini quand le run gagne de l'argent (une
        # part de perte n'a alors aucun sens) -- laissé à None plutôt que
        # calculé sur une valeur absolue qui se lirait comme un ratio de perte.
        "friction_pct_of_loss": float(friction / abs(pnl) * 100) if pnl < 0 else None,
        "annualized_turnover_pct": run_metrics.get("annualized_turnover_pct"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="valuation_gap_multiples_options", choices=sorted(OPTIONS_STRATEGY_REGISTRY))
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--entry-threshold-pct", type=float, default=None)
    parser.add_argument("--benchmark-symbol", default=config.BENCHMARK_SYMBOL)
    parser.add_argument("--initial-capital", type=float, default=config.OPTIONS_INITIAL_CAPITAL)
    parser.add_argument("--max-trade-dollar", type=float, default=config.OPTIONS_MAX_TRADE_DOLLAR)
    parser.add_argument(
        "--slippage-pct-of-premium", type=float, default=config.OPTIONS_SLIPPAGE_PCT_OF_PREMIUM,
        help="Slippage du run NOMINAL ; l'autre branche du 2x2 est toujours 0.",
    )
    parser.add_argument("--export", action="store_true", help="Écrit le tableau en CSV/JSON dans data/backtest_options/.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Les deux branches du plan seraient identiques : le 2x2 dégénère en 1x2 et
    # la décomposition ne mesurerait plus rien (l'écart imputé au slippage
    # vaudrait zéro par construction, pas par constat).
    if args.slippage_pct_of_premium == 0:
        raise SystemExit(
            "--slippage-pct-of-premium 0 : le run nominal et le run sans slippage seraient "
            "identiques. Passe le slippage RÉEL attendu (défaut : "
            f"{config.OPTIONS_SLIPPAGE_PCT_OF_PREMIUM}%) -- la branche à 0 est déjà jouée "
            "automatiquement."
        )

    strategy_cls = OPTIONS_STRATEGY_REGISTRY[args.strategy]
    settings_args = argparse.Namespace(
        stop_loss_pct=None, take_profit_pct=None, target_tenor_days=None, stop_basis=None,
        exit_when_signal_lost=None, roll_when_days_left=None, daily_rebalance=None,
        vol_mode=None, min_resize_relative_pct=None,
    )
    settings, imposed = _cli.resolve_engine_settings(strategy_cls, settings_args)
    if imposed:
        logger.info("Réglages imposés par la stratégie '%s' : %s.", args.strategy, ", ".join(imposed))

    strategy_params = {}
    if args.entry_threshold_pct is not None:
        strategy_params["entry_threshold_pct"] = args.entry_threshold_pct

    engine_kwargs = dict(
        initial_capital=args.initial_capital,
        commission_per_contract=config.OPTIONS_COMMISSION_PER_CONTRACT,
        commission_min_per_order=config.OPTIONS_COMMISSION_MIN_PER_ORDER,
        max_fee_pct_of_trade=config.OPTIONS_MAX_FEE_PCT_OF_TRADE,
        min_deployment_pct=config.OPTIONS_MIN_DEPLOYMENT_PCT,
        max_delta_notional_pct=config.OPTIONS_MAX_DELTA_NOTIONAL_PCT,
        fee_bump_max_extra_pct=config.OPTIONS_FEE_BUMP_MAX_EXTRA_PCT,
        whole_contracts=config.OPTIONS_WHOLE_CONTRACTS,
        max_trade_dollar=args.max_trade_dollar,
        momentum_min_pct=config.BACKTEST_MOMENTUM_MIN_PCT,
    )

    data = _load_data(args.benchmark_symbol)
    rows = []
    for slippage in (args.slippage_pct_of_premium, 0.0):
        for daily in (True, False):
            logger.info("Run slippage=%.2f%% daily_rebalance=%s...", slippage, daily)
            rows.append(run_one(
                data, strategy_cls, strategy_params, settings, engine_kwargs, slippage, daily, args,
            ))

    results = pd.DataFrame(rows)
    results.insert(0, "case", [
        f"slippage={r.slippage_pct:g}% / daily={'oui' if r.daily_rebalance else 'non'}"
        for r in results.itertuples()
    ])

    with pd.option_context("display.width", 220, "display.max_columns", 40, "display.float_format", "{:,.2f}".format):
        print(f"\n=== Plan 2x2 friction / churn ({args.strategy}) ===")
        print(results.drop(columns=["slippage_pct", "daily_rebalance"]).to_string(index=False))

    def pnl_of(slippage_zero: bool, daily: bool) -> float:
        mask = (results["slippage_pct"] == 0.0) == slippage_zero
        return float(results[mask & (results["daily_rebalance"] == daily)]["pnl_dollar"].iloc[0])

    nominal = pnl_of(slippage_zero=False, daily=True)
    these = pnl_of(slippage_zero=True, daily=False)
    cout_slippage = pnl_of(slippage_zero=True, daily=True) - nominal
    cout_churn = pnl_of(slippage_zero=False, daily=False) - nominal

    print(
        f"\n>>> P&L nominal      : {nominal:+,.0f}$"
        f"\n>>> P&L 'pure thèse' : {these:+,.0f}$   (slippage 0, pas de rebalancement quotidien)"
        f"\n>>> dont slippage    : {cout_slippage:+,.0f}$   (écart imputable au seul slippage)"
        f"\n>>> dont churn       : {cout_churn:+,.0f}$   (écart imputable au seul rebalancement quotidien)"
    )
    if nominal < 0:
        print(
            f">>> La friction (slippage + commissions) représente "
            f"{results[results['slippage_pct'] != 0]['friction_pct_of_loss'].iloc[0]:.0f}% "
            f"de la perte du run nominal."
        )
    print(
        "\nLes commissions et frais tiers sont payés dans les QUATRE cases : la colonne "
        "'pure thèse' est un plafond, pas un contrefactuel atteignable."
    )

    if args.export:
        config.DIR_BACKTEST_OPTIONS.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = config.DIR_BACKTEST_OPTIONS / f"friction_{args.strategy}_{run_id}"
        results.to_csv(f"{base}.csv", index=False)
        pd.Series({
            "pnl_nominal": nominal, "pnl_these": these,
            "cout_slippage": cout_slippage, "cout_churn": cout_churn,
        }).to_json(f"{base}.json", indent=2)
        logger.info("Résultats exportés dans %s.csv / .json", base)


if __name__ == "__main__":
    main()
