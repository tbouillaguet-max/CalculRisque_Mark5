"""
Décomposition CALL / PUT d'un backtest options déjà joué.

Le résumé de 10_backtest_options.py agrège les deux paris en un seul NAV : il
dit combien la stratégie gagne, jamais laquelle des deux jambes le gagne. Ce
script rouvre le run sauvegardé et sépare tout ce qui peut l'être :

    1. Métriques habituelles, en trois colonnes : ALL / CALL / PUT (mêmes
       définitions que metrics.py, donc lisibles ligne à ligne).
    2. Répartition des VALORISATIONS : prime immobilisée par chaque jambe, en
       moyenne et au pic, et part du portefeuille.
    3. Répartition des VOLUMES DE TRADE : nombre de trades, contrats,
       montants décaissés/encaissés, P&L.
    4. P&L par (jambe, motif de sortie) -- le tableau qui localise réellement
       la perte : un stop-loss négatif ne se lit pas comme une expiration
       négative.
    5. P&L réalisé par année et par jambe.

Aucun backtest n'est rejoué : le script LIT `data/backtest_options/<run_id>/`.
Sans --run-id, il prend le run le plus récent de la stratégie demandée.

Usage :
    python 10_backtest_options.py --strategy valuation_gap_multiples_options --start-date 2015-01-01
    python 12_analyse_put_call.py                      # dernier run de cette stratégie
    python 12_analyse_put_call.py --run-id 20260810_143000
    python 12_analyse_put_call.py --strategy valuation_gap_options
    python 12_analyse_put_call.py --export             # écrit aussi les CSV dans le run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

import config
from backtest import put_call_analysis as pca

logger = logging.getLogger("analyse_put_call")


def resolve_run_dir(strategy: str, run_id: str | None) -> Path:
    """Répertoire du run à analyser. Sans run_id, le plus récent de cette
    stratégie -- déterminé en lisant run_config.json, pas en se fiant au nom
    du dossier (un --run-id explicite passé à 10_backtest_options.py peut être
    n'importe quoi)."""
    root = config.DIR_BACKTEST_OPTIONS
    if run_id:
        run_dir = root / run_id
        if not run_dir.is_dir():
            raise SystemExit(f"Run introuvable : {run_dir}")
        return run_dir

    candidates = []
    for path in root.glob("*/run_config.json"):
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if cfg.get("strategy") == strategy:
            candidates.append((path.parent.stat().st_mtime, path.parent))
    if not candidates:
        raise SystemExit(
            f"Aucun run de la stratégie '{strategy}' dans {root}.\n"
            f"Lance d'abord :\n"
            f"    python 10_backtest_options.py --strategy {strategy} --start-date 2015-01-01"
        )
    return max(candidates)[1]


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    equity_curve = pd.read_parquet(run_dir / "equity_curve.parquet")
    positions_history = pd.read_parquet(run_dir / "positions_history.parquet")
    trades = pd.read_parquet(run_dir / "trades.parquet")
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    return equity_curve, positions_history, trades, run_config


def _show(title: str, frame: pd.DataFrame, floatfmt: str = "{:,.2f}") -> None:
    print(f"\n=== {title} ===")
    if frame is None or frame.empty:
        print("(vide)")
        return
    with pd.option_context(
        "display.width", 200, "display.max_columns", 40, "display.max_rows", 200,
        "display.float_format", floatfmt.format,
    ):
        print(frame.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="valuation_gap_multiples_options")
    parser.add_argument("--run-id", default=None, help="Défaut : le run le plus récent de la stratégie.")
    parser.add_argument("--export", action="store_true", help="Écrit aussi les tableaux en CSV dans le répertoire du run.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    run_dir = resolve_run_dir(args.strategy, args.run_id)
    equity_curve, positions_history, trades, run_config = load_run(run_dir)
    logger.info(
        "Run %s : stratégie '%s', du %s au %s.",
        run_dir.name, run_config.get("strategy"), run_config.get("start_date"), run_config.get("end_date"),
    )
    if equity_curve.empty:
        raise SystemExit("equity_curve vide : rien à analyser.")
    if trades.empty:
        logger.warning("Aucun trade dans ce run : seules les métriques de courbe seront exploitables.")

    # La décomposition suppose NAV = capital initial + réalisé + latent. On le
    # VÉRIFIE plutôt que de le supposer : au-delà de 0,01% du capital, les
    # tableaux ci-dessous ne décrivent plus le run et il faut le dire.
    residual = pca.reconciliation_residual(equity_curve, positions_history, trades)
    tolerance_pct = 0.01
    if (residual["max_abs_residual_pct_of_initial"] or 0) > tolerance_pct:
        logger.warning(
            "Réconciliation imparfaite : écart max %.2f$ (%.4f%% du capital initial) entre le NAV "
            "du moteur et 'capital + réalisé + latent'. La décomposition CALL/PUT ci-dessous est "
            "donc approximative -- voir la docstring de backtest/put_call_analysis.py.",
            residual["max_abs_residual_dollar"], residual["max_abs_residual_pct_of_initial"],
        )
    else:
        logger.info(
            "Réconciliation exacte (écart max %.4f$) : la somme des deux jambes reproduit le NAV du moteur.",
            residual["max_abs_residual_dollar"],
        )

    per_side = pca.side_metrics(
        equity_curve, positions_history, trades, risk_free_rate=config.RISK_FREE_RATE,
    )
    tables = {
        "metriques_par_jambe": pca.metrics_table(per_side),
        "repartition_valorisations": pca.valuation_breakdown(positions_history),
        "repartition_volumes": pca.volume_breakdown(trades),
        "pnl_par_motif_de_sortie": pca.pnl_by_exit_reason(trades),
        "pnl_realise_par_annee": pca.annual_pnl_by_side(trades),
    }

    _show("Métriques : ALL vs CALL vs PUT", tables["metriques_par_jambe"], "{:,.3f}")
    print(
        "\nLecture : les % d'une jambe sont rapportés au CAPITAL INITIAL COMPLET (chaque jambe est\n"
        "rejouée comme un portefeuille recevant le seul P&L de cette jambe). Seule la ligne\n"
        "leg_pnl_dollar s'additionne entre CALL et PUT pour retrouver ALL."
    )
    _show("Répartition des valorisations (prime immobilisée)", tables["repartition_valorisations"])
    _show("Répartition des volumes de trade", tables["repartition_volumes"])
    _show("P&L par jambe et motif de sortie", tables["pnl_par_motif_de_sortie"], "{:,.2f}")
    _show("P&L réalisé par année de sortie", tables["pnl_realise_par_annee"])

    call_pnl = per_side.get("CALL", {}).get("leg_pnl_dollar")
    put_pnl = per_side.get("PUT", {}).get("leg_pnl_dollar")
    if call_pnl is not None and put_pnl is not None:
        perdante = "PUT" if put_pnl < call_pnl else "CALL"
        print(
            f"\n>>> CALL : {call_pnl:+,.0f}$   |   PUT : {put_pnl:+,.0f}$   "
            f"|   total : {per_side['ALL']['leg_pnl_dollar']:+,.0f}$"
        )
        print(f">>> Jambe la moins contributive : {perdante}.")

    if args.export:
        for name, frame in tables.items():
            if frame is not None and not frame.empty:
                frame.to_csv(run_dir / f"put_call_{name}.csv")
        pca.pnl_contribution_over_time(equity_curve, positions_history, trades).to_csv(
            run_dir / "put_call_contribution_quotidienne.csv", index=False,
        )
        (run_dir / "put_call_metrics.json").write_text(
            json.dumps({"reconciliation": residual, "per_side": per_side}, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("Tableaux exportés dans %s", run_dir)


if __name__ == "__main__":
    sys.exit(main())
