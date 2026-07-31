"""
Orchestrateur du pipeline sur une base TRIMESTRIELLE plutôt qu'annuelle.

Deux modes, mutuellement exclusifs :

MODE LIVE (défaut, sans --as-of-date)
---------------------------------------
Enchaîne les étapes en conditions réelles (nouveaux appels réseau SEC/IBKR/
Mistral), dans l'ordre :
    04b (10-Q + TTM) -> 04c (8-K entre trimestres) -> 05 (multiples) ->
    06 (multiples moyens) -> 06b (valorisation combinée) -> 07 (DCF) ->
    07b (validation qualitative) -> 08 (options, si le filtre d'écart de
    valorisation retient des entreprises)
Chaque étape est un sous-processus indépendant (aucun script numéroté
n'expose de main(args) appelable en mémoire -- ils lisent tous sys.argv via
argparse) : un échec arrête proprement l'enchaînement plutôt que de
continuer aveuglément sur des données potentiellement incomplètes.

--skip-options saute l'étape 08 (nécessite IB Gateway ouvert -- pas toujours
disponible dans un contexte automatisé/cron, voir section Cron plus bas).

MODE REPLAY (--as-of-date AAAA-MM-JJ)
----------------------------------------
Ne fait AUCUN appel réseau. Reconstitue ce que 05/06/06b/07 auraient produit
avec UNIQUEMENT les données déjà connues (filed_date <= as_of_date) à cette
date passée -- utile pour une inspection point-in-time ponctuelle hors du
moteur de backtest complet (qui, lui, fait déjà ce filtrage nativement au
jour le jour, voir backtest/engine.py).

Aucune écriture dans data/ réel : les fichiers sources (financials, TTM,
cours) sont filtrés puis copiés dans un répertoire de travail temporaire, et
05/06/06b/07 y sont exécutés avec ce répertoire comme working directory --
config.py résout BASE_DIR="./data" relativement au répertoire d'exécution
(voir README.md), donc un sous-processus lancé avec cwd=<temp> lit/écrit
entièrement isolé de data/, sans monkey-patch ni variable d'environnement.
Les résultats (DCF_FILE, VALORISATION_COMBINEE_FILE) sont recopiés à côté du
répertoire temporaire pour inspection, puis le répertoire est journalisé
(pas supprimé automatiquement, pour recopie/debug manuel si besoin).

Cron (production, pas de daemon Python custom)
--------------------------------------------------
Un 10-Q est déposé ~45 jours après la fin de trimestre (délai SEC habituel) :
exemple de crontab déclenchant ce script début-mai/août/novembre/février
(après la fenêtre de dépôt la plus probable) :

    0 6 5 2,5,8,11 *  cd /chemin/vers/CalculRisque_Mark3 && \
        /usr/bin/python3 run_pipeline_quarterly.py --skip-options >> logs/quarterly.log 2>&1

(--skip-options si IB Gateway n'est pas géré par le même cron -- relance
alors 08_recuperation_options.py séparément une fois Gateway disponible.)

Usage :
    python run_pipeline_quarterly.py                       # mode live, toutes les étapes
    python run_pipeline_quarterly.py --skip-options         # mode live, sans 08 (pas d'IBKR)
    python run_pipeline_quarterly.py --limit 10             # test rapide (04b/04c/07b limités)
    python run_pipeline_quarterly.py --as-of-date 2024-06-30  # mode replay, aucun appel réseau
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

import config

logger = logging.getLogger("run_pipeline_quarterly")

REPO_ROOT = Path(__file__).resolve().parent

LIVE_STEPS = [
    "04b_recuperation_10q.py", "04c_recuperation_8k.py", "05_calcul_multiples.py",
    "06_calcul_multiples_moyens.py", "06b_calcul_valorisation_combinee.py",
    "07_calcul_dcf.py", "07b_validation_qualitative.py",
]
OPTIONS_STEP = "08_recuperation_options.py"
# Étapes qui acceptent --limit (pour un test rapide du pipeline complet) :
# celles qui bouclent sur l'univers/les périodes (04b/04c/07b), pas 05/06/06b/07
# (calcul vectorisé sur ce qui existe déjà en cache, --limit n'aurait pas de sens).
STEPS_ACCEPTING_LIMIT = {"04b_recuperation_10q.py", "04c_recuperation_8k.py", "07b_validation_qualitative.py"}

REPLAY_STEPS = ["05_calcul_multiples.py", "06_calcul_multiples_moyens.py", "06b_calcul_valorisation_combinee.py", "07_calcul_dcf.py"]


def run_step(script: str, extra_args: List[str], cwd: Optional[Path] = None) -> None:
    cmd = [sys.executable, str(REPO_ROOT / script), *extra_args]
    logger.info("=== %s %s (cwd=%s) ===", script, " ".join(extra_args), cwd or REPO_ROOT)
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise RuntimeError(f"{script} a échoué (code {result.returncode}) -- pipeline arrêté.")


def run_live(limit: Optional[int], skip_options: bool) -> None:
    for script in LIVE_STEPS:
        extra_args = ["--limit", str(limit)] if (limit and script in STEPS_ACCEPTING_LIMIT) else []
        run_step(script, extra_args)

    if skip_options:
        logger.info("--skip-options : étape 08 (chaînes d'options, IBKR) sautée.")
        return
    run_step(OPTIONS_STEP, [])


# ----------------------------------------------------------------------------
# Mode replay : reconstitution point-in-time, aucun appel réseau
# ----------------------------------------------------------------------------

def _filter_by_filed_date(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if "filed_date" not in df.columns or df.empty:
        return df
    filed = pd.to_datetime(df["filed_date"], errors="coerce")
    return df[filed <= as_of].copy()


def _prepare_replay_workspace(as_of: pd.Timestamp) -> Path:
    """Copie FILTRÉE (filed_date <= as_of) des fichiers sources nécessaires
    à 05/06/06b/07, dans une arborescence data/ isolée sous un répertoire
    temporaire -- jamais d'écriture dans data/ réel (voir docstring)."""
    scratch = Path(tempfile.mkdtemp(prefix="quarterly_replay_"))
    scratch_data = scratch / "data"

    for d in (config.DIR_UNIVERSE, config.DIR_PRICES, config.DIR_FINANCIALS, config.DIR_MULTIPLES, config.DIR_DCF):
        rel = d.relative_to(config.BASE_DIR)
        (scratch_data / rel).mkdir(parents=True, exist_ok=True)

    def _dest(path: Path) -> Path:
        return scratch_data / path.relative_to(config.BASE_DIR)

    if config.UNIVERSE_FILE.exists():
        shutil.copy(config.UNIVERSE_FILE, _dest(config.UNIVERSE_FILE))
    else:
        raise FileNotFoundError(f"{config.UNIVERSE_FILE} introuvable -- lance d'abord 01_build_universe.py.")

    if config.FINANCIALS_FILE.exists():
        annual = _filter_by_filed_date(pd.read_parquet(config.FINANCIALS_FILE), as_of)
        annual.to_parquet(_dest(config.FINANCIALS_FILE), index=False)
    else:
        raise FileNotFoundError(f"{config.FINANCIALS_FILE} introuvable -- lance d'abord 04_recuperation_10k.py.")

    if config.FINANCIALS_TTM_FILE.exists():
        ttm = _filter_by_filed_date(pd.read_parquet(config.FINANCIALS_TTM_FILE), as_of)
        if not ttm.empty:
            ttm.to_parquet(_dest(config.FINANCIALS_TTM_FILE), index=False)

    if config.PRICES_FILE.exists():
        prices = pd.read_parquet(config.PRICES_FILE)
        # Un cours de clôture de fin d'année N est réputé connu dès le
        # 31/12/N (pas de délai de dépôt SEC pour un cours de marché,
        # contrairement aux données financières) -- filtré sur cette base.
        year_end_known = pd.to_datetime(prices["year"].astype(int).astype(str) + "-12-31")
        prices[year_end_known <= as_of].to_parquet(_dest(config.PRICES_FILE), index=False)

    if config.DAILY_PRICES_FILE.exists():
        daily = pd.read_parquet(config.DAILY_PRICES_FILE)
        daily = daily[pd.to_datetime(daily["date"]) <= as_of]
        (scratch_data / config.DAILY_PRICES_FILE.relative_to(config.BASE_DIR)).parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(_dest(config.DAILY_PRICES_FILE), index=False)

    return scratch


def run_replay(as_of_date: str) -> None:
    try:
        as_of = pd.Timestamp(as_of_date)
    except ValueError as exc:
        raise ValueError(f"--as-of-date invalide (attendu AAAA-MM-JJ) : {as_of_date}") from exc

    logger.info("Mode replay point-in-time : reconstitution au %s (aucun appel réseau).", as_of.date())
    scratch = _prepare_replay_workspace(as_of)
    logger.info("Espace de travail isolé : %s", scratch)

    for script in REPLAY_STEPS:
        run_step(script, [], cwd=scratch)

    scratch_dcf = scratch / "data" / config.DCF_FILE.relative_to(config.BASE_DIR)
    scratch_combined = scratch / "data" / config.VALORISATION_COMBINEE_FILE.relative_to(config.BASE_DIR)
    logger.info("Terminé. Résultats point-in-time disponibles sous :")
    if scratch_dcf.exists():
        logger.info("  - DCF : %s", scratch_dcf)
    if scratch_combined.exists():
        logger.info("  - Valorisation combinée : %s", scratch_combined)
    logger.info("Répertoire temporaire conservé (pas de suppression automatique) pour inspection/recopie manuelle.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of-date", type=str, default=None, metavar="YYYY-MM-DD", help="Mode replay point-in-time (aucun appel réseau) au lieu du mode live.")
    parser.add_argument("--limit", type=int, default=None, help="Transmis à 04b/04c/07b (mode live uniquement) pour un test rapide.")
    parser.add_argument("--skip-options", action="store_true", help="Saute l'étape 08 (chaînes d'options, nécessite IB Gateway) -- mode live uniquement.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    start = datetime.now()
    try:
        if args.as_of_date:
            run_replay(args.as_of_date)
        else:
            run_live(limit=args.limit, skip_options=args.skip_options)
    except RuntimeError as exc:
        logger.error("Pipeline arrêté : %s", exc)
        sys.exit(1)

    logger.info("Pipeline terminé en %s.", datetime.now() - start)


if __name__ == "__main__":
    main()
