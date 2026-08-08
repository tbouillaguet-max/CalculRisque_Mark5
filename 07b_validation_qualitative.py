"""
Validation QUALITATIVE, par LLM (Mistral), du signal quantitatif de
valorisation (07_calcul_dcf.py / 06b_calcul_valorisation_combinee.py) :
pour chaque (symbol, période) déjà valorisé, récupère le texte du 10-K/10-Q
DE CE DÉPÔT PRÉCIS (jamais un filing plus récent) et demande au modèle si le
narratif de l'entreprise à cette date (risques, perspectives) est cohérent
avec, à surveiller, ou contradictoire vis-à-vis de l'écart de valorisation
calculé (gap_pct). Un DCF peut être arithmétiquement correct tout en reposant
sur des hypothèses que l'entreprise elle-même contredit dans son propre
texte (ex: dépréciation d'actif annoncée, procédure judiciaire matérielle,
guidance revue à la baisse) -- ce script sert de garde-fou qualitatif, pas de
remplacement du signal quantitatif.

Contrainte anti-anticipation (point-in-time)
----------------------------------------------
Le modèle ne reçoit QUE le texte du filing déposé À la filed_date de la
période évaluée -- jamais un filing postérieur, jamais un résumé agrégé de
plusieurs dépôts. Cette garantie est structurelle, pas une simple consigne de
prompt : sec_filings_text.py::get_filing_text_asof ne peut physiquement
renvoyer que le document déposé à cette date exacte (voir sa docstring). Le
verdict produit pour une période N est donc exactement ce qu'un lecteur du
10-K/10-Q de l'époque aurait pu établir à l'époque -- aucune connaissance
d'événements survenus depuis n'est jamais transmise au modèle.

Ce module ne crée volontairement PAS de mécanisme générique séparé : la
logique de recherche/téléchargement/extraction de texte SEC et l'appel LLM
générique vivent dans sec_filings_text.py (réutilisé aussi par
04c_recuperation_8k.py, qui applique la même contrainte anti-anticipation à
la détection d'événements matériels).

Prérequis :
    pip install requests beautifulsoup4
    export MISTRAL_API_KEY="ta_cle"   (voir 02_categoriser_secteurs.py -- sans
    cette variable, le script journalise chaque ligne comme "non_evalue" et
    n'appelle jamais Mistral, plutôt que de planter)

Usage :
    python 07b_validation_qualitative.py
    python 07b_validation_qualitative.py --limit 10
    python 07b_validation_qualitative.py --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

import config
import sec_filings_text as sft

logger = logging.getLogger("validation_qualitative")

CHECKPOINT_EVERY = 10
FORM_BY_PERIOD_TYPE = {"FY": ("10-K",), "TTM": ("10-Q", "10-K")}  # TTM peut être "complété" par un 10-K si le dernier trimestre du TTM est un Q4 (voir 04b)

PROMPT_TEMPLATE = """Tu es un analyste financier. Voici un extrait du début du \
document SEC ({form}, déposé le {filed_date}) de l'entreprise {symbol}, et \
l'écart de valorisation calculé par un modèle quantitatif (DCF/multiples) à \
cette même date : {gap_pct:.1f}% (positif = jugée sous-évaluée par le \
modèle, négatif = jugée survalorisée).

Analyse UNIQUEMENT le texte ci-dessous (ignore tout ce que tu pourrais savoir \
par ailleurs sur cette entreprise après cette date) : ce texte reflète-t-il \
une situation cohérente avec cet écart de valorisation, ou signale-t-il des \
risques/événements qui pourraient le remettre en cause (dépréciation \
d'actifs, procédure judiciaire matérielle, révision de guidance, doute sur \
la continuité d'exploitation...) ?

Texte du document :
{text}

Réponds UNIQUEMENT avec un JSON valide au format :
{{"verdict": "coherent" ou "a_surveiller" ou "contradictoire", \
"justification": "une phrase courte", "risques_cites": ["liste courte de \
risques mentionnés dans le texte, vide si aucun notable"]}}
"""


def build_prompt(symbol: str, form: str, filed_date: str, gap_pct: float, text: str) -> str:
    return PROMPT_TEMPLATE.format(symbol=symbol, form=form, filed_date=filed_date, gap_pct=gap_pct, text=text)


def load_signal_periods(limit: Optional[int] = None) -> pd.DataFrame:
    """Une ligne par (symbol, period_type, fiscal_year, fiscal_quarter) déjà
    valorisée : DCF_HISTORY_FILE (07) en priorité (toujours présent),
    complété par VALORISATION_COMBINEE_FILE (06b, gap_pct par multiples
    éventuellement différent) si disponible -- dédupliqué sur la clé de
    période, la valeur de gap_pct utilisée n'a pas besoin d'être identique
    entre les deux (l'objectif ici est juste de couvrir les périodes déjà
    signalées comme intéressantes par AU MOINS une des deux méthodes)."""
    if not config.DCF_HISTORY_FILE.exists():
        raise FileNotFoundError(f"{config.DCF_HISTORY_FILE} introuvable. Lance d'abord 07_calcul_dcf.py.")
    dcf = pd.read_parquet(config.DCF_HISTORY_FILE)
    dcf = dcf.dropna(subset=["gap_pct", "cik"]) if "cik" in dcf.columns else dcf.dropna(subset=["gap_pct"])

    keep_cols = ["symbol", "cik", "period_type", "fiscal_year", "fiscal_quarter", "filed_date", "gap_pct"]
    keep_cols = [c for c in keep_cols if c in dcf.columns]
    df = dcf[keep_cols].copy()

    if "cik" not in df.columns:
        # DCF_HISTORY_FILE ne porte pas le CIK (ajouté par 04/04b, mais
        # perdu en route dans build_input_table -- cf. 07 : le CIK n'est
        # utile qu'ici) : impossible de retrouver le filing sans lui.
        raise ValueError(
            f"{config.DCF_HISTORY_FILE} n'a pas de colonne 'cik' : relance 04_recuperation_10k.py "
            "--force-refresh puis 07_calcul_dcf.py pour régénérer un historique avec le CIK."
        )

    df = df.dropna(subset=["cik", "filed_date"]).drop_duplicates(subset=["symbol", "period_type", "fiscal_year", "fiscal_quarter"])
    df = df.sort_values("filed_date", ascending=False)
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


def evaluate_period(row: pd.Series) -> dict:
    symbol, cik = row["symbol"], str(row["cik"])
    period_type = row.get("period_type") or "FY"
    filed_date = str(row["filed_date"])[:10]
    forms = FORM_BY_PERIOD_TYPE.get(period_type, ("10-K", "10-Q"))

    filing = sft.get_filing_text_asof(cik, filed_date, forms=forms)
    if filing is None:
        # Distingué de "pas de clé API" ci-dessous : les deux produisaient le
        # même "non_evalue", si bien qu'un trou de couverture SEC et une
        # absence de configuration se lisaient pareil dans le parquet de
        # sortie -- impossible de savoir laquelle des deux corriger.
        return {
            "verdict": "non_evalue_filing_introuvable",
            "justification": "Aucun 10-K/10-Q trouvé à cette date de dépôt exacte.",
            "risques_cites": None,
        }

    if not os.environ.get(sft.MISTRAL_API_KEY_ENV):
        return {
            "verdict": "non_evalue_pas_de_cle_api",
            "justification": f"{sft.MISTRAL_API_KEY_ENV} non définie : aucun appel au modèle.",
            "risques_cites": None,
            "accession_number": filing["accession_number"],
            "form": filing["form"],
            "extraction_mode": filing.get("extraction_mode"),
        }

    prompt = build_prompt(symbol, filing["form"], filed_date, float(row["gap_pct"]), filing["text"])
    result = sft.analyser_texte_mistral(prompt)
    if result is None or "verdict" not in result:
        return {
            "verdict": "non_evalue_reponse_invalide",
            "justification": "Réponse Mistral indisponible ou invalide.",
            "risques_cites": None,
            "accession_number": filing["accession_number"],
            "form": filing["form"],
            "extraction_mode": filing.get("extraction_mode"),
        }

    return {
        "verdict": result.get("verdict"),
        "justification": result.get("justification"),
        "risques_cites": json.dumps(result.get("risques_cites"), ensure_ascii=False) if result.get("risques_cites") is not None else None,
        "accession_number": filing["accession_number"],
        "form": filing["form"],
        # Qualité de l'extraction, remontée jusqu'au parquet : un verdict
        # rendu sur le début du document ne porte pas sur la même chose qu'un
        # verdict rendu sur les Items 1A/3/7 (cf. sec_filings_text.MAX_TEXT_CHARS).
        "extraction_mode": filing.get("extraction_mode"),
    }


# ----------------------------------------------------------------------------
# Checkpoint/reprise façon 08_recuperation_options.py
# ----------------------------------------------------------------------------

def _progress_path(output_dir: Path) -> Path:
    return output_dir / "progress_qualitative.json"


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint_qualitative.jsonl"


def load_progress(output_dir: Path) -> set:
    path = _progress_path(output_dir)
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")).get("processed", []))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Fichier de progression illisible (%s), on repart de zéro.", e)
        return set()


def save_progress(output_dir: Path, processed: set) -> None:
    path = _progress_path(output_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"processed": sorted(processed), "updated_at": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_checkpoint(output_dir: Path, row: dict) -> None:
    with _checkpoint_path(output_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")


def load_checkpoint_rows(output_dir: Path) -> list:
    path = _checkpoint_path(output_dir)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=config.DIR_DCF, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not os.environ.get(sft.MISTRAL_API_KEY_ENV):
        logger.warning(
            "MISTRAL_API_KEY non définie : toutes les périodes seront journalisées comme "
            "'non_evalue_pas_de_cle_api' (pas d'appel Mistral). export MISTRAL_API_KEY=... "
            "pour activer la validation qualitative."
        )

    periods = load_signal_periods(limit=args.limit)
    if periods.empty:
        logger.warning("Aucune période à évaluer (DCF_HISTORY_FILE vide ou sans gap_pct exploitable).")
        return

    processed_keys: set = set()
    if args.resume:
        processed_keys = load_progress(args.output_dir)
        logger.info("Reprise : %d périodes déjà traitées.", len(processed_keys))
    else:
        _progress_path(args.output_dir).unlink(missing_ok=True)
        _checkpoint_path(args.output_dir).unlink(missing_ok=True)

    def _key(row: pd.Series) -> str:
        return f"{row['symbol']}|{row.get('period_type')}|{row.get('fiscal_year')}|{row.get('fiscal_quarter')}"

    to_process = [row for _, row in periods.iterrows() if _key(row) not in processed_keys]
    logger.info("%d/%d périodes à évaluer (%d déjà traitées).", len(to_process), len(periods), len(processed_keys))

    since_checkpoint = 0
    try:
        for i, row in enumerate(to_process, start=1):
            key = _key(row)
            # pd.notna et non la seule valeur de vérité : fiscal_quarter est
            # NaN sur les lignes annuelles, et un NaN est "vrai" en Python --
            # le log affichait donc "INTU FY 2016 nan".
            quarter = row.get("fiscal_quarter")
            suffixe = f" {quarter}" if pd.notna(quarter) and quarter else ""
            logger.info(
                "[%d/%d] %s %s %s%s...", i, len(to_process),
                row["symbol"], row.get("period_type"), row.get("fiscal_year"), suffixe,
            )
            try:
                result = evaluate_period(row)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  -> ECHEC pour %s : %s (période ignorée, on continue)", key, exc)
                result = {"verdict": "non_evalue", "justification": str(exc), "risques_cites": None}

            checkpoint_row = {
                "symbol": row["symbol"], "period_type": row.get("period_type"),
                "fiscal_year": row.get("fiscal_year"), "fiscal_quarter": row.get("fiscal_quarter"),
                "filed_date": row["filed_date"], "gap_pct": row["gap_pct"],
                "evaluated_at": datetime.now().isoformat(timespec="seconds"),
                **result,
            }
            append_checkpoint(args.output_dir, checkpoint_row)
            processed_keys.add(key)

            since_checkpoint += 1
            if since_checkpoint >= CHECKPOINT_EVERY:
                save_progress(args.output_dir, processed_keys)
                since_checkpoint = 0
    finally:
        save_progress(args.output_dir, processed_keys)

    rows = load_checkpoint_rows(args.output_dir)
    if not rows:
        logger.warning("Aucun résultat produit, pas de fichier de sortie généré.")
        return

    df = pd.DataFrame(rows)
    config.QUALITATIVE_VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.QUALITATIVE_VALIDATION_FILE, index=False, engine="pyarrow")
    logger.info("Validation qualitative sauvegardée : %s (%d lignes).", config.QUALITATIVE_VALIDATION_FILE, len(df))
    logger.info("Répartition des verdicts : %s", df["verdict"].value_counts().to_dict())


if __name__ == "__main__":
    main()
