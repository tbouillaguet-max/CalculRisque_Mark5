"""
Comparaison côte à côte des trois stratégies OPTIONS, sur la même période et
le même univers.

    valuation_gap_options                  strike ATM
    valuation_gap_multiples_options        strike à mi-chemin cours / valeur théorique
    valuation_gap_expected_value_options   strike qui maximise la croissance log-optimale

CHAQUE STRATÉGIE TOURNE AVEC SES PROPRES engine_defaults. C'est délibéré, et
c'est le point le plus important de ce script : les stops, l'échéance, le
roulement et le mode de volatilité font partie de la THÈSE de chaque stratégie
(un strike hors de la monnaie justifie un stop plus resserré, une échéance à
deux ans justifie une volatilité de repricing glissante). Les uniformiser
donnerait une comparaison « toutes choses égales par ailleurs » qui ne
mesurerait plus aucune des trois telles qu'elles sont pensées -- seulement
trois variantes d'un cadre commun que personne n'a conçu. Ce qui EST uniformisé
et doit l'être : la période, l'univers, le capital initial, les coûts
d'exécution et le benchmark.

CE QUE LE TABLEAU NE DIT PAS. Trois runs sur une même période, ce sont trois
tirages d'une même histoire de marché, pas trois échantillons indépendants. Un
écart de Sharpe de 0,1 entre deux stratégies sur quinze ans n'est pas une
preuve de supériorité ; la décomposition du P&L par motif de sortie est
souvent plus informative que le classement lui-même, parce qu'elle dit POURQUOI
une stratégie gagne (le pari se réalise) ou perd (les stops coupent au mauvais
moment, ou la valeur temps mange le gain).

Chaque stratégie est rejouée via 10_backtest_options.py -- le vrai pipeline,
pas une réimplémentation : les runs produits sont donc inspectables ensuite
avec 12_analyse_put_call.py, et les métriques lues ici sont exactement celles
de metrics.json.

Usage :
    python compare_options_strategies.py --start-date 2015-01-01
    python compare_options_strategies.py --start-date 2015-01-01 --end-date 2024-01-01
    python compare_options_strategies.py --strategies valuation_gap_multiples_options valuation_gap_expected_value_options
    python compare_options_strategies.py --reuse-existing      # ne rejoue rien, lit les derniers runs

Sorties :
    - tableau console (métriques clés, puis P&L par motif de sortie)
    - CSV sous data/backtest/comparisons/comparison_<horodatage>.csv
      (+ ..._by_exit_reason.csv pour la décomposition)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from backtest import put_call_analysis as pca
from backtest.strategies import OPTIONS_STRATEGY_REGISTRY

logger = logging.getLogger("compare_options_strategies")

DEFAULT_STRATEGIES = [
    "valuation_gap_options",
    "valuation_gap_multiples_options",
    "valuation_gap_expected_value_options",
]

# Métriques retenues pour le tableau côte à côte. L'ordre est celui de la
# lecture : d'abord ce que la stratégie rapporte, puis ce qu'elle risque, puis
# ce qu'elle coûte, puis ce qu'elle fait (volume d'activité).
COMPARISON_METRICS = [
    ("cagr_pct", "CAGR %"),
    ("total_return_pct", "Rendement total %"),
    ("sharpe_ratio", "Sharpe"),
    ("sortino_ratio", "Sortino"),
    ("max_drawdown_pct", "Max drawdown %"),
    ("calmar_ratio", "Calmar"),
    ("win_rate_pct", "Taux de réussite %"),
    ("profit_factor", "Profit factor"),
    ("total_friction_pct_of_initial", "Friction % capital initial"),
    ("total_friction_dollar", "Friction $"),
    ("total_theta_decay_pct_of_initial", "Theta % capital initial"),
    ("num_trades", "Trades"),
    ("avg_holding_days", "Durée moyenne (j)"),
    ("avg_exposure_pct", "Exposition moyenne %"),
    ("dropped_orders_count", "Ordres écartés"),
    # Propres à la stratégie « espérance de gain » : absents des deux autres,
    # affichés en tiret plutôt qu'en zéro (ne rien mesurer n'est pas mesurer zéro).
    ("dropped_expected_value_negative_count", "Lignes écartées (espérance <= 0)"),
    ("expected_value_implied_vol_pct", "Ouvertures sur IV réelle %"),
]


def _run_backtest(strategy: str, run_id: str, args: argparse.Namespace) -> Path:
    """Rejoue 10_backtest_options.py pour cette stratégie et rend son
    répertoire de sortie.

    Sous-processus plutôt qu'import : c'est ce qui garantit que chaque
    stratégie voit EXACTEMENT la résolution d'engine_defaults du vrai
    pipeline, y compris ses avertissements, sans qu'on ait à la réimplémenter
    ici (et donc sans risque de la voir diverger silencieusement)."""
    commande = [
        sys.executable, "10_backtest_options.py",
        "--strategy", strategy,
        "--run-id", run_id,
        "--initial-capital", str(args.initial_capital),
        "--benchmark-symbol", args.benchmark_symbol,
    ]
    if args.start_date:
        commande += ["--start-date", args.start_date]
    if args.end_date:
        commande += ["--end-date", args.end_date]
    if args.extra_arg:
        commande += args.extra_arg

    logger.info("[%s] %s", strategy, " ".join(commande))
    resultat = subprocess.run(commande, cwd=Path(__file__).parent)
    if resultat.returncode != 0:
        raise SystemExit(
            f"Le backtest de '{strategy}' a échoué (code {resultat.returncode}). "
            "La comparaison est interrompue : un tableau à deux colonnes sur trois "
            "serait plus trompeur qu'utile."
        )
    return config.DIR_BACKTEST_OPTIONS / run_id


def _latest_run_dir(strategy: str) -> Path | None:
    """Dernier run de cette stratégie, identifié par run_config.json et non par
    le nom du dossier (cf. 12_analyse_put_call.resolve_run_dir)."""
    candidats = []
    for chemin in config.DIR_BACKTEST_OPTIONS.glob("*/run_config.json"):
        try:
            cfg = json.loads(chemin.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if cfg.get("strategy") == strategy:
            candidats.append((chemin.parent.stat().st_mtime, chemin.parent))
    return max(candidats)[1] if candidats else None


def _load_metrics(run_dir: Path) -> dict:
    chemin = run_dir / "metrics.json"
    if not chemin.exists():
        raise SystemExit(f"metrics.json introuvable dans {run_dir}.")
    return json.loads(chemin.read_text(encoding="utf-8"))


def _load_trades(run_dir: Path) -> pd.DataFrame:
    chemin = run_dir / "trades.parquet"
    return pd.read_parquet(chemin) if chemin.exists() else pd.DataFrame()


def _comparison_table(par_strategie: dict[str, dict]) -> pd.DataFrame:
    """Une ligne par métrique, une colonne par stratégie -- l'orientation qui
    se lit à l'œil quand les stratégies sont peu nombreuses et les métriques
    nombreuses."""
    lignes = []
    for cle, libelle in COMPARISON_METRICS:
        ligne = {"metrique": libelle}
        for strategie, metriques in par_strategie.items():
            ligne[strategie] = metriques.get(cle)
        # Une métrique qu'AUCUNE stratégie ne renseigne n'apporte rien au
        # tableau (c'est le cas des compteurs propres à une stratégie quand
        # elle n'est pas dans la comparaison).
        if any(ligne[s] is not None for s in par_strategie):
            lignes.append(ligne)
    return pd.DataFrame(lignes).set_index("metrique")


def _exit_reason_table(trades_par_strategie: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """P&L par (stratégie, jambe, motif de sortie), empilé.

    Même fonction que celle qu'emploie 12_analyse_put_call.py, pour que les
    chiffres de la comparaison et ceux de l'analyse détaillée d'un run soient
    les mêmes à la ligne près."""
    morceaux = []
    for strategie, trades in trades_par_strategie.items():
        table = pca.pnl_by_exit_reason(trades)
        if table.empty:
            continue
        morceaux.append(table.assign(strategy=strategie))
    if not morceaux:
        return pd.DataFrame()
    colonnes = ["strategy", "option_type", "exit_reason", "num_trades", "pnl_dollar",
                "avg_return_pct", "median_return_pct", "win_rate_pct", "avg_holding_days"]
    empile = pd.concat(morceaux, ignore_index=True)
    return empile[[c for c in colonnes if c in empile.columns]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--strategies", nargs="+", default=DEFAULT_STRATEGIES,
        choices=sorted(OPTIONS_STRATEGY_REGISTRY),
        help=f"Stratégies à comparer (défaut: les trois -- {' '.join(DEFAULT_STRATEGIES)}).",
    )
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--initial-capital", type=float, default=config.OPTIONS_INITIAL_CAPITAL)
    parser.add_argument("--benchmark-symbol", default=config.BENCHMARK_SYMBOL)
    parser.add_argument(
        "--extra-arg", action="append", default=[],
        help="Argument supplémentaire transmis TEL QUEL à 10_backtest_options.py pour "
             "CHAQUE stratégie (répétable). À réserver aux réglages qui doivent rester "
             "communs : passer ici un stop-loss reviendrait à uniformiser les thèses, "
             "ce que ce script cherche justement à éviter.",
    )
    parser.add_argument(
        "--reuse-existing", action="store_true",
        help="Ne rejoue rien : lit le dernier run de chaque stratégie. Rapide, mais les "
             "runs peuvent porter sur des périodes DIFFÉRENTES -- le script le vérifie et "
             "avertit.",
    )
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if len(args.strategies) < 2:
        raise SystemExit("Comparer demande au moins deux stratégies.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    repertoires: dict[str, Path] = {}

    for strategie in args.strategies:
        if args.reuse_existing:
            run_dir = _latest_run_dir(strategie)
            if run_dir is None:
                raise SystemExit(
                    f"Aucun run existant pour '{strategie}'. Relance sans --reuse-existing, "
                    f"ou joue d'abord :\n"
                    f"    python 10_backtest_options.py --strategy {strategie} --start-date 2015-01-01"
                )
            logger.info("[%s] run existant réutilisé : %s", strategie, run_dir.name)
        else:
            run_dir = _run_backtest(strategie, f"{run_id}_{strategie}", args)
        repertoires[strategie] = run_dir

    metriques = {s: _load_metrics(d) for s, d in repertoires.items()}
    trades = {s: _load_trades(d) for s, d in repertoires.items()}

    # Les runs réutilisés peuvent porter sur des périodes différentes : la
    # comparaison n'aurait alors aucun sens, et l'erreur serait indétectable
    # dans le tableau final.
    periodes = {}
    for strategie, run_dir in repertoires.items():
        cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        periodes[strategie] = (cfg.get("start_date"), cfg.get("end_date"))
    if len(set(periodes.values())) > 1:
        logger.warning(
            "Les runs comparés NE PORTENT PAS sur la même période : %s. Le tableau "
            "ci-dessous compare des choses non comparables -- relance sans "
            "--reuse-existing.",
            ", ".join(f"{s} = {a} -> {b}" for s, (a, b) in periodes.items()),
        )
    else:
        debut, fin = next(iter(periodes.values()))
        logger.info("Période commune : %s -> %s", debut, fin)

    tableau = _comparison_table(metriques)
    par_motif = _exit_reason_table(trades)

    sortie = Path(args.output_csv) if args.output_csv else (
        config.DIR_BACKTEST / "comparisons" / f"comparison_{run_id}.csv"
    )
    sortie.parent.mkdir(parents=True, exist_ok=True)
    tableau.to_csv(sortie)
    logger.info("Tableau comparatif sauvegardé dans %s", sortie)

    if not par_motif.empty:
        sortie_motifs = sortie.with_name(f"{sortie.stem}_by_exit_reason.csv")
        par_motif.to_csv(sortie_motifs, index=False)
        logger.info("P&L par motif de sortie sauvegardé dans %s", sortie_motifs)

    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:,.3f}".format):
        print("\n=== Comparaison des stratégies options ===")
        print(tableau.to_string(na_rep="-"))

        if not par_motif.empty:
            print("\n=== P&L par motif de sortie ===")
            print(par_motif.to_string(index=False, na_rep="-"))

            # Le tableau qui localise réellement la performance : total par
            # stratégie et par motif, toutes jambes confondues.
            pivot = par_motif.pivot_table(
                index="exit_reason", columns="strategy", values="pnl_dollar", aggfunc="sum",
            )
            print("\n=== P&L ($) par motif de sortie, toutes jambes confondues ===")
            print(pivot.to_string(na_rep="-"))

    print(
        "\nRappel : trois runs sur une même période sont trois tirages d'une même "
        "histoire de marché, pas trois échantillons indépendants. Un écart de Sharpe "
        "faible ne départage rien -- la décomposition par motif de sortie dit POURQUOI "
        "une stratégie gagne, ce que le classement ne dit pas."
    )


if __name__ == "__main__":
    main()
