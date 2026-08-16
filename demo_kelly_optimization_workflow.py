#!/usr/bin/env python3
"""
Démonstration complète du workflow d'optimisation de la stratégie Kelly.

Ce script exécute les quatre phases recommandées d'optimisation de
ValuationGapExpectedValueOptionsStrategy:

    Phase 1 (Baseline):
        - Backtest de référence avec les défauts
        - Comparaison des trois stratégies (Kelly vs ATM vs Multiples)

    Phase 2 (Convergence):
        - Grid-search sur convergence_fraction
        - Identification du pic optimal

    Phase 3 (Stops):
        - Grid-search stop-loss/take-profit
        - Heatmap de performance

    Phase 4 (Validation):
        - Backtest out-of-sample sur données non-vues
        - Confirmation de la stabilité de l'optimisation

Utilisation:
    python demo_kelly_optimization_workflow.py \\
        --start-date 2015-01-01 \\
        --end-date 2024-01-01 \\
        --phase 1          # Exécuter seulement la phase 1 (baseline)

    python demo_kelly_optimization_workflow.py \\
        --phase all        # Exécuter toutes les phases (séquentiel)

Sortie:
    - Logs détaillés pour chaque phase
    - CSV comparatifs en data/backtest/comparisons/
    - CSV de grilles en data/backtest/optimization/
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("demo_kelly_optimization")


def run_command(cmd: list, description: str) -> int:
    """Exécute une commande et log le résultat."""
    logger.info("=" * 80)
    logger.info(f"[PHASE] {description}")
    logger.info(f"[CMD] {' '.join(cmd)}")
    logger.info("=" * 80)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"❌ Erreur lors de: {description}")
        return result.returncode
    logger.info(f"✓ Succès: {description}\n")
    return 0


def phase_1_baseline(args: argparse.Namespace) -> int:
    """
    Phase 1: Établir la baseline.

    Exécute:
    1. Backtest de référence Kelly avec tous les défauts
    2. Comparaison côte à côte des trois stratégies
    """
    logger.info("\n" + "🔵 " * 20)
    logger.info("PHASE 1: ÉTABLIR LA BASELINE (10-15 min)")
    logger.info("🔵 " * 20 + "\n")

    cmd_baseline = [
        sys.executable, "10_backtest_options.py",
        "--strategy", "valuation_gap_expected_value_options",
    ]
    if args.start_date:
        cmd_baseline += ["--start-date", args.start_date]
    if args.end_date:
        cmd_baseline += ["--end-date", args.end_date]

    if run_command(cmd_baseline, "Backtest de référence Kelly") != 0:
        return 1

    cmd_compare = [
        sys.executable, "compare_options_strategies.py",
    ]
    if args.start_date:
        cmd_compare += ["--start-date", args.start_date]
    if args.end_date:
        cmd_compare += ["--end-date", args.end_date]

    if run_command(cmd_compare, "Comparaison des trois stratégies") != 0:
        return 1

    logger.info("\n📊 RÉSULTATS PHASE 1 (Baseline):")
    logger.info("   - Backtest Kelly: data/backtest/options/<run_id>/")
    logger.info("   - Comparaison: data/backtest/comparisons/comparison_*.csv")
    logger.info("\n   Interprétation:")
    logger.info("   → CAGR: rendement annualisé de chaque stratégie")
    logger.info("   → Sharpe: rendement/risque")
    logger.info("   → P&L par motif: où vient le gain (profit, expiry, stop)")
    logger.info("\n   Actions recommandées:")
    logger.info("   1. Comparer le CAGR Kelly vs ATM vs Multiples")
    logger.info("   2. Analyser la décomposition P&L par motif")
    logger.info("   3. Si Kelly surperforme: continuer phase 2 (optimisation)")
    logger.info("   4. Si Kelly sous-performe: investiguer convergence_fraction\n")

    return 0


def phase_2_convergence(args: argparse.Namespace) -> int:
    """
    Phase 2: Optimiser convergence_fraction.

    Exécute un grid-search complet sur la fraction de convergence
    qui contrôle la dérive alimentée au critère de Kelly.
    """
    logger.info("\n" + "🟢 " * 20)
    logger.info("PHASE 2: OPTIMISER CONVERGENCE_FRACTION (20-45 min)")
    logger.info("🟢 " * 20 + "\n")

    cmd = [
        sys.executable, "11c_optimize_convergence_fraction.py",
    ]
    if args.start_date:
        cmd += ["--start-date", args.start_date]
    if args.end_date:
        cmd += ["--end-date", args.end_date]
    if args.workers:
        cmd += ["--workers", str(args.workers)]

    if run_command(cmd, "Grid-search sur convergence_fraction") != 0:
        return 1

    logger.info("\n📈 RÉSULTATS PHASE 2 (Convergence):")
    logger.info("   - Grille d'optimisation: data/backtest/optimization/convergence_*.csv")
    logger.info("\n   Interprétation:")
    logger.info("   → Plotter CAGR vs fraction (doit être unimodal)")
    logger.info("   → Identifier le pic (fraction optimale)")
    logger.info("   → Vérifier que le pic n'est pas au bord de la grille")
    logger.info("\n   Actions recommandées:")
    logger.info("   1. Lire la CSV de grille")
    logger.info("   2. Finder la fraction avec CAGR maximal")
    logger.info("   3. Si pic au bord: élargir --fraction-grid")
    logger.info("   4. Mettre à jour le défaut dans valuation_gap_expected_value_options.py")
    logger.info("   5. Re-run phase 1 pour comparer nouveau baseline\n")

    return 0


def phase_3_stops(args: argparse.Namespace) -> int:
    """
    Phase 3: Optimiser stops (stop-loss et take-profit).

    Exécute un grid-search joint sur les paires (stop_loss, take_profit)
    pour identifier la combinaison optimale de sortie.
    """
    logger.info("\n" + "🟡 " * 20)
    logger.info("PHASE 3: OPTIMISER STOP-LOSS & TAKE-PROFIT (30-60 min)")
    logger.info("🟡 " * 20 + "\n")

    cmd = [
        sys.executable, "11_optimize_options_stops.py",
        "--strategy", "valuation_gap_expected_value_options",
    ]
    if args.start_date:
        cmd += ["--start-date", args.start_date]
    if args.end_date:
        cmd += ["--end-date", args.end_date]

    if run_command(cmd, "Grid-search stop-loss/take-profit") != 0:
        return 1

    logger.info("\n📊 RÉSULTATS PHASE 3 (Stops):")
    logger.info("   - Grille stops: data/backtest/optimization/stops_*.csv")
    logger.info("\n   Interprétation:")
    logger.info("   → Heatmap (stop_loss vs take_profit) avec Sharpe")
    logger.info("   → Identifier la cellule optimale")
    logger.info("   → Vérifier stabilité (pic vs plateau)")
    logger.info("\n   Actions recommandées:")
    logger.info("   1. Créer une heatmap CSV → Excel/matplotlib")
    logger.info("   2. Identifier la combinaison (stop, profit) optimale")
    logger.info("   3. Affiner autour du pic si nécessaire")
    logger.info("   4. Mettre à jour les engine_defaults dans la stratégie")
    logger.info("   5. Re-run phase 1 pour mesurer l'impact total\n")

    return 0


def phase_4_validation(args: argparse.Namespace) -> int:
    """
    Phase 4: Validation out-of-sample.

    Teste les paramètres optimisés sur une période non-vue pendant
    l'optimisation, pour vérifier la généralisation.
    """
    logger.info("\n" + "🔴 " * 20)
    logger.info("PHASE 4: VALIDATION OUT-OF-SAMPLE (10-30 min)")
    logger.info("🔴 " * 20 + "\n")

    # OOS = data post-optimization
    # Si optimization sur 2015-2024, tester sur 2024+
    oos_start = "2024-01-01"
    oos_end = "2025-08-16"

    cmd = [
        sys.executable, "10_backtest_options.py",
        "--strategy", "valuation_gap_expected_value_options",
        "--start-date", oos_start,
        "--end-date", oos_end,
    ]

    logger.info(f"   Testant sur période OOS: {oos_start} → {oos_end}")
    logger.info("   (données NON vues pendant l'optimisation)\n")

    if run_command(cmd, "Backtest validation out-of-sample") != 0:
        return 1

    logger.info("\n✅ RÉSULTATS PHASE 4 (Validation OOS):")
    logger.info("   - Backtest OOS: data/backtest/options/<run_id>/")
    logger.info("   - Metrics OOS: data/backtest/options/<run_id>/metrics.json")
    logger.info("\n   Interprétation:")
    logger.info("   → Comparer CAGR (in-sample vs OOS)")
    logger.info("   → Comparer Sharpe (in-sample vs OOS)")
    logger.info("   → Écart < 10% = généralisation bonne")
    logger.info("   → Écart > 20% = overfitting possible")
    logger.info("\n   Actions recommandées:")
    logger.info("   1. Charger metrics.json OOS et IS")
    logger.info("   2. Calculer écarts (|OOS - IS| / IS)")
    logger.info("   3. Si écarts petits: paramètres sont robustes")
    logger.info("   4. Si écarts importants: revoir la grille d'optimisation")
    logger.info("   5. Production ready: intégrer au pipeline trimestriel\n")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        default="1",
        choices=["1", "2", "3", "4", "all"],
        help="Quelle(s) phase(s) exécuter (défaut: 1)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2015-01-01",
        help="Date de début (défaut: 2015-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-01-01",
        help="Date de fin pour l'optimisation (défaut: 2024-01-01)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Nombre de workers parallèles pour grid-search (défaut: 2)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("\n" + "=" * 80)
    logger.info("DÉMONSTRATION WORKFLOW D'OPTIMISATION KELLY")
    logger.info("=" * 80)
    logger.info(f"Phase(s): {args.phase}")
    logger.info(f"Période: {args.start_date} → {args.end_date}")
    logger.info("=" * 80 + "\n")

    phases = {
        "1": phase_1_baseline,
        "2": phase_2_convergence,
        "3": phase_3_stops,
        "4": phase_4_validation,
    }

    if args.phase == "all":
        phases_to_run = ["1", "2", "3", "4"]
    else:
        phases_to_run = [args.phase]

    for phase_num in phases_to_run:
        if phases[phase_num](args) != 0:
            logger.error(f"\n❌ Workflow interrompu à phase {phase_num}")
            sys.exit(1)

    logger.info("\n" + "✅ " * 20)
    logger.info("WORKFLOW COMPLET RÉUSSI")
    logger.info("✅ " * 20)
    logger.info("\nRésumé des sorties:")
    logger.info("   1. Baseline: data/backtest/comparisons/comparison_*.csv")
    logger.info("   2. Convergence: data/backtest/optimization/convergence_*.csv")
    logger.info("   3. Stops: data/backtest/optimization/stops_*.csv")
    logger.info("   4. Validation OOS: data/backtest/options/<run_id>/metrics.json")
    logger.info("\nProchaines étapes:")
    logger.info("   1. Analyser chaque CSV pour trouver les optimaux")
    logger.info("   2. Mettre à jour les défauts dans valuation_gap_expected_value_options.py")
    logger.info("   3. Exécuter la pipeline trimestrielle en conditions réelles")
    logger.info("   4. Monitorer la performance réelle vs prédite\n")


if __name__ == "__main__":
    main()
