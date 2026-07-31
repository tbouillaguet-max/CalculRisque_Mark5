"""
Détecte les événements MATÉRIELS annoncés par 8-K entre deux trimestres TTM
déjà connus (04b_recuperation_10q.py) : rachats d'actions, changement de
guidance, départ de dirigeant, procédure judiciaire, M&A -- des événements
qui pourraient invalider une thèse de valorisation AVANT le prochain
recalcul trimestriel (07_calcul_dcf.py).

Pour chaque entreprise, la fenêtre de recherche est calculée à partir des
filed_date déjà connues dans FINANCIALS_TTM_FILE (04b) : entre chaque paire
de trimestres consécutifs, plus une fenêtre "ouverte" du dernier trimestre
connu jusqu'à aujourd'hui (pour détecter les 8-K récents pas encore suivis
d'un nouveau TTM). Les 8-K eux-mêmes sont listés et téléchargés via
sec_filings_text.py (réutilisé, pas dupliqué -- même module que
07b_validation_qualitative.py).

Contrainte anti-anticipation
------------------------------
Chaque 8-K est un document déjà intrinsèquement point-in-time (il ne peut
par construction parler que d'événements connus à SA date de dépôt) : la
classification par Mistral ne porte QUE sur le texte de CE 8-K, jamais sur un
résumé agrégé ou une connaissance d'événements postérieurs -- même garantie
structurelle que 07b_validation_qualitative.py (sec_filings_text.py::
fetch_filing_text ne télécharge qu'UN document à la fois).

Pré-classification par regex (codes "Item X.XX", boilerplate standardisé du
formulaire 8-K -- ex: Item 5.02 = départ/nomination de dirigeant, Item 8.01 =
autres événements, Item 1.01 = accord matériel) : gratuite et fiable, gardée
en plus du verdict LLM (pas à sa place) pour un recoupement rapide côté
rapport, sans dépendre uniquement de la classification sémantique du modèle.

Prérequis :
    pip install requests beautifulsoup4
    export MISTRAL_API_KEY="ta_cle"   (voir 02_categoriser_secteurs.py -- sans
    cette variable, les 8-K sont journalisés avec category="non_evalue"
    plutôt que de planter)

Usage :
    python 04c_recuperation_8k.py
    python 04c_recuperation_8k.py --limit 10
    python 04c_recuperation_8k.py --resume
    python 04c_recuperation_8k.py --ticker AAPL
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

import config
import sec_filings_text as sft

logger = logging.getLogger("recuperation_8k")

CHECKPOINT_EVERY = 10
ITEM_CODE_PATTERN = re.compile(r"Item\s+\d+\.\d+", re.IGNORECASE)

CATEGORIES = (
    "rachat_actions", "changement_guidance", "depart_dirigeant",
    "procedure_judiciaire", "fusion_acquisition", "autre_materiel", "non_materiel",
)

PROMPT_TEMPLATE = """Tu es un analyste financier. Voici le texte d'un 8-K \
déposé par {symbol} le {filed_date} (codes Item détectés dans le document : \
{item_codes}). Analyse UNIQUEMENT ce texte (ignore tout ce que tu pourrais \
savoir par ailleurs sur cette entreprise après cette date).

Texte du document :
{text}

Cet événement est-il matériel pour une thèse de valorisation (susceptible de \
changer significativement la valeur intrinsèque ou le risque perçu de \
l'entreprise) ? Classifie-le dans UNE seule catégorie parmi : {categories}.

Réponds UNIQUEMENT avec un JSON valide au format :
{{"category": "une des catégories ci-dessus", "materiality": true ou false, \
"summary": "résumé en une phrase courte"}}
"""


def build_prompt(symbol: str, filed_date: str, item_codes: List[str], text: str) -> str:
    return PROMPT_TEMPLATE.format(
        symbol=symbol, filed_date=filed_date, item_codes=", ".join(item_codes) or "aucun détecté",
        text=text, categories=", ".join(CATEGORIES),
    )


def extract_item_codes(text: str) -> List[str]:
    """Codes "Item X.XX" détectés (recherche insensible à la casse, mais
    normalisés en "Item X.XX" avant dédoublonnage -- sinon "Item 5.02" et
    "item 5.02" compteraient comme deux codes distincts)."""
    matches = ITEM_CODE_PATTERN.findall(text)
    normalized = {re.sub(r"^item", "Item", m, flags=re.IGNORECASE) for m in matches}
    return sorted(normalized)


def compute_search_windows(ttm: pd.DataFrame, symbol: str, today: datetime) -> List[tuple]:
    """Fenêtres (start_date, end_date) entre trimestres TTM consécutifs
    déjà connus pour ce symbole, plus une fenêtre ouverte du dernier
    trimestre connu jusqu'à aujourd'hui. Vide si aucun trimestre TTM connu
    pour ce symbole (04b jamais lancé pour lui)."""
    dates = sorted(pd.to_datetime(ttm.loc[ttm["symbol"] == symbol, "filed_date"]).dt.strftime("%Y-%m-%d").unique())
    if not dates:
        return []
    windows = [(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]
    windows.append((dates[-1], today.strftime("%Y-%m-%d")))
    return windows


def classify_8k(symbol: str, filed_date: str, text: str) -> dict:
    item_codes = extract_item_codes(text)
    prompt = build_prompt(symbol, filed_date, item_codes, text)
    result = sft.analyser_texte_mistral(prompt)
    if result is None or "category" not in result:
        return {"item_codes": item_codes, "category": "non_evalue", "materiality": None, "summary": None}
    return {
        "item_codes": item_codes, "category": result.get("category"),
        "materiality": result.get("materiality"), "summary": result.get("summary"),
    }


def process_ticker_8k(symbol: str, cik: str, windows: List[tuple]) -> List[dict]:
    rows = []
    seen_accessions = set()
    for start_date, end_date in windows:
        filings = sft.list_company_filings(cik, forms=("8-K",), start_date=start_date, end_date=end_date)
        for filing in filings:
            if filing["accession_number"] in seen_accessions:
                continue  # deux fenêtres adjacentes peuvent se recouvrir sur leur borne commune
            seen_accessions.add(filing["accession_number"])

            url = sft.filing_document_url(cik, filing["accession_number"], filing["primary_document"])
            text = sft.fetch_filing_text(url)
            if text is None:
                continue

            classification = classify_8k(symbol, filing["filing_date"], text)
            rows.append({
                "symbol": symbol, "cik": cik, "filed_date": filing["filing_date"],
                "accession_number": filing["accession_number"],
                **classification,
                "fetch_timestamp": datetime.now().isoformat(timespec="seconds"),
            })
    return rows


# ----------------------------------------------------------------------------
# Checkpoint/reprise façon 08_recuperation_options.py
# ----------------------------------------------------------------------------

def _progress_path(output_dir: Path) -> Path:
    return output_dir / "progress_8k.json"


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint_8k.jsonl"


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


def append_checkpoint(output_dir: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with _checkpoint_path(output_dir).open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")


def load_checkpoint_rows(output_dir: Path) -> List[dict]:
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
    parser.add_argument("--tickers", type=Path, default=config.UNIVERSE_FILE)
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--output-dir", default=config.DIR_FINANCIALS, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not os.environ.get(sft.MISTRAL_API_KEY_ENV):
        logger.warning(
            "MISTRAL_API_KEY non définie : les 8-K seront journalisés avec category='non_evalue' "
            "(pas d'appel Mistral). export MISTRAL_API_KEY=... pour activer la classification."
        )

    if not config.FINANCIALS_TTM_FILE.exists():
        logger.error(
            "%s introuvable. Lance d'abord 04b_recuperation_10q.py : ce script a besoin des "
            "trimestres TTM déjà connus pour délimiter les fenêtres de recherche des 8-K.",
            config.FINANCIALS_TTM_FILE,
        )
        return
    ttm = pd.read_parquet(config.FINANCIALS_TTM_FILE)

    if args.ticker:
        symbols_ric = [args.ticker.upper()]
    else:
        universe = pd.read_csv(args.tickers, encoding="utf-8-sig")
        symbols_ric = universe["RIC"].dropna().unique().tolist()
        if args.limit:
            symbols_ric = symbols_ric[: args.limit]

    # cik par symbole : porté par FINANCIALS_TTM_FILE (04b), pas besoin de
    # re-résoudre via company_tickers.json comme 04/04b.
    cik_by_symbol = ttm.drop_duplicates(subset=["symbol"]).set_index("symbol")["cik"].astype(str).to_dict()

    processed_keys: set = set()
    if args.resume:
        processed_keys = load_progress(args.output_dir)
        logger.info("Reprise : %d tickers déjà traités.", len(processed_keys))
    else:
        _progress_path(args.output_dir).unlink(missing_ok=True)
        _checkpoint_path(args.output_dir).unlink(missing_ok=True)

    today = datetime.now()
    to_process = []
    for ric in symbols_ric:
        symbol = config.to_ib_symbol(ric)
        if symbol in processed_keys:
            continue
        if symbol not in cik_by_symbol:
            continue  # pas de trimestre TTM connu pour ce symbole -- rien à borner, ignoré silencieusement
        to_process.append(symbol)

    logger.info("%d/%d tickers avec un historique TTM à interroger.", len(to_process), len(symbols_ric))

    ok_count, fail_count, event_count = 0, 0, 0
    since_checkpoint = 0
    try:
        for i, symbol in enumerate(to_process, start=1):
            cik = cik_by_symbol[symbol]
            windows = compute_search_windows(ttm, symbol, today)
            logger.info("[%d/%d] %s (CIK %s, %d fenêtre(s))...", i, len(to_process), symbol, cik, len(windows))
            try:
                rows = process_ticker_8k(symbol, cik, windows)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  -> ECHEC pour %s : %s (ticker ignoré, on continue)", symbol, exc)
                rows = []
                fail_count += 1
            else:
                ok_count += 1
                event_count += len(rows)
                if rows:
                    append_checkpoint(args.output_dir, rows)

            processed_keys.add(symbol)
            since_checkpoint += 1
            if since_checkpoint >= CHECKPOINT_EVERY:
                save_progress(args.output_dir, processed_keys)
                since_checkpoint = 0
    finally:
        save_progress(args.output_dir, processed_keys)

    logger.info("Terminé. OK: %d | Échecs: %d | 8-K détectés: %d", ok_count, fail_count, event_count)

    rows = load_checkpoint_rows(args.output_dir)
    if not rows:
        logger.warning("Aucun 8-K collecté, pas de fichier de sortie généré.")
        return

    df = pd.DataFrame(rows)
    config.MATERIAL_EVENTS_8K_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.MATERIAL_EVENTS_8K_FILE, index=False, engine="pyarrow")
    logger.info("8-K sauvegardés : %s (%d lignes, %d entreprises).", config.MATERIAL_EVENTS_8K_FILE, len(df), df["symbol"].nunique())
    if "materiality" in df.columns:
        logger.info("Répartition matérialité : %s", df["materiality"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
