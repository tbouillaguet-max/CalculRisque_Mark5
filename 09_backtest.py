"""
Backtest une stratégie construite sur les données du pipeline (01b cours
quotidiens exclus -- voir 03b, univers point-in-time -- voir 01b, écarts de
valorisation DCF -- voir 07). Sans biais de survivance si 01b a été lancé
(sinon repli sur l'univers actuel, voir warning au démarrage).

Hypothèses du moteur (backtest/engine.py), à garder en tête pour interpréter
les résultats :
    - Décision à la clôture, exécution à l'ouverture du jour suivant (pas de
      look-ahead), sauf radiation totale (clôturée au dernier cours connu).
    - Une position ne sort JAMAIS uniquement parce que son écart de
      valorisation s'est refermé : seuls stop-loss/take-profit (ou une
      disparition des données) la clôturent (voir doc du choix utilisateur
      dans backtest/engine.py).
    - Coûts de transaction (commission + slippage) fusionnés en un seul
      cost_bps appliqué symétriquement à l'achat et à la vente.

Prérequis : avoir lancé au moins 01_build_universe.py, 03b (ou 03 à défaut,
mais 03 ne fournit que des clôtures annuelles -- insuffisant pour un
backtest quotidien), 04_recuperation_10k.py et 07_calcul_dcf.py. 01b est
optionnel mais fortement recommandé (sinon biais de survivance).

Usage :
    python 09_backtest.py
    python 09_backtest.py --strategy valuation_gap_dcf --start-date 2015-01-01
    python 09_backtest.py --entry-threshold-pct 25 --stop-loss-pct -10 --take-profit-pct 40
    python 09_backtest.py --list-strategies
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from backtest import data_loader, metrics as metrics_mod
from backtest.construction_moteur import (  # noqa: F401 -- parse_strategy_params : API de ce script
    ajouter_options_moteur, charger_donnees, construire_moteur, construire_strategie,
    parse_strategy_params,
)
from backtest.strategies import STRATEGY_REGISTRY

logger = logging.getLogger("backtest.cli")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="valuation_gap_dcf", help="Nom de la stratégie enregistrée (voir --list-strategies).")
    parser.add_argument("--list-strategies", action="store_true", help="Liste les stratégies disponibles et quitte.")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD (défaut: début des données de prix).")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD (défaut: fin des données de prix).")
    # Réglages de la stratégie et du moteur : partagés avec 17_paper_trading.py,
    # pour que le compte paper rejoue exactement la configuration mesurée ici.
    ajouter_options_moteur(parser)
    parser.add_argument("--run-id", default=None, help="Nom du sous-dossier de sortie (défaut: horodatage).")
    parser.add_argument("--risk-free-rate", type=float, default=config.RISK_FREE_RATE)
    parser.add_argument(
        "--benchmark-symbol", default=config.BENCHMARK_SYMBOL,
        help="Indice de référence (défaut: %(default)s). Doit être présent dans les cours "
             "quotidiens ; sinon un indice équipondéré de l'univers point-in-time est reconstruit.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=1,
        help="Nombre de configurations essayées avant de retenir celle-ci (défaut: %(default)s, "
             "run isolé). Après un grid-search, passe la TAILLE DE LA GRILLE : le Sharpe du "
             "meilleur point est celui d'un maximum sur autant de tirages, et metrics.json "
             "rapporte alors sharpe_noise_floor et deflated_sharpe_ratio en regard.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.list_strategies:
        for name in sorted(STRATEGY_REGISTRY):
            print(name)
        return

    if args.strategy not in STRATEGY_REGISTRY:
        logger.error("Stratégie inconnue: %s. Disponibles: %s", args.strategy, sorted(STRATEGY_REGISTRY))
        sys.exit(1)

    logger.info("Chargement des données...")
    donnees = charger_donnees(args.strategy)
    universe_history = donnees.universe_history
    price_panel = donnees.price_panel
    strategy = construire_strategie(args.strategy, args)

    start_date = pd.Timestamp(args.start_date) if args.start_date else None
    end_date = pd.Timestamp(args.end_date) if args.end_date else None
    engine = construire_moteur(args, donnees, strategy, start_date=start_date, end_date=end_date)

    logger.info(
        "Backtest '%s' du %s au %s (%d jours de bourse)...",
        args.strategy, engine.calendar[0].date(), engine.calendar[-1].date(), len(engine.calendar),
    )
    equity_curve, positions_history, trades, signals_history = engine.run()
    benchmark_prices, benchmark_label = data_loader.build_benchmark_series(
        price_panel, engine.universe, symbol=args.benchmark_symbol,
    )
    run_metrics = metrics_mod.compute_metrics(
        equity_curve, trades,
        risk_free_rate=args.risk_free_rate,
        benchmark_prices=benchmark_prices,
        extra=engine.execution_diagnostics(),
        n_trials=args.n_trials,
    )
    run_metrics["benchmark_label"] = benchmark_label

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = config.DIR_BACKTEST / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_curve.to_parquet(out_dir / "equity_curve.parquet", index=False, engine="pyarrow")
    positions_history.to_parquet(out_dir / "positions_history.parquet", index=False, engine="pyarrow")
    trades.to_parquet(out_dir / "trades.parquet", index=False, engine="pyarrow")
    signals_history.to_parquet(out_dir / "signals_history.parquet", index=False, engine="pyarrow")
    (out_dir / "metrics.json").write_text(json.dumps(run_metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps({
        # strategy.params et non strategy_params : ce sont les valeurs
        # EFFECTIVES, défauts de la stratégie compris. Ne consigner que ce que
        # la ligne de commande portait laissait un run irreproductible dès
        # qu'un défaut bougeait.
        "strategy": args.strategy, "strategy_params": strategy.params,
        "initial_capital": args.initial_capital, "commission_bps": args.commission_bps,
        "slippage_bps": args.slippage_bps, "stop_loss_pct": args.stop_loss_pct,
        "take_profit_pct": args.take_profit_pct,
        "momentum_min_pct": args.momentum_min_pct, "rebalance_band_pct": args.rebalance_band_pct,
        "start_date": str(engine.calendar[0].date()), "end_date": str(engine.calendar[-1].date()),
        "risk_free_rate": args.risk_free_rate, "has_pit_universe": universe_history is not None,
        "benchmark_symbol": args.benchmark_symbol, "benchmark_label": benchmark_label,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Résultats sauvegardés dans %s", out_dir)
    logger.info("--- Résumé ---")
    for key in [
        "total_return_pct", "cagr_pct", "annualized_volatility_pct", "sharpe_ratio", "sortino_ratio",
        "max_drawdown_pct", "calmar_ratio", "annualized_turnover_pct", "avg_exposure_pct",
        # num_trades/win_rate_pct/profit_factor comptent des EXÉCUTIONS (ventes
        # partielles de rebalancement comprises) ; les variantes *_positions
        # comptent des THÈSES. Les deux sont affichées côte à côte parce que
        # l'écart entre elles est en soi une information (voir
        # metrics._position_level_metrics) -- ce sont les secondes qui disent
        # si la stratégie a raison, les premières combien elle exécute.
        "num_trades", "win_rate_pct", "profit_factor",
        "num_positions_closed", "win_rate_positions_pct", "profit_factor_positions",
        "truncated_orders_count", "truncated_orders_pct", "avg_cash_pct",
        "signal_coverage_avg_ratio", "signal_coverage_min_ratio", "signal_coverage_min_year",
    ]:
        if key in run_metrics:
            logger.info("%s: %s", key, run_metrics[key])
    logger.info("%s", metrics_mod.format_benchmark_summary(run_metrics, benchmark_label))
    logger.info(
        "Relis ce run en détail (couverture de l'univers, alpha par sous-période, "
        "sensibilité aux coûts) : python 14_audit_backtest.py --run-id %s", run_id,
    )


if __name__ == "__main__":
    main()
