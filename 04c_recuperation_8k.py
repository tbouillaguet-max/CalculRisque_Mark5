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

Mémoire des classifications (cache_8k_mistral.jsonl)
----------------------------------------------------
Un 8-K est un document FIGÉ : son texte ne changera plus, donc sa
classification non plus. Chaque 8-K classifié AVEC SUCCÈS par Mistral est
mémorisé (clé : symbole + numéro d'accession) dans un cache JSONL persistant,
et n'est jamais ré-analysé -- ni son texte re-téléchargé auprès de la SEC.
Seuls les 8-K réellement classifiés y entrent : un "non_evalue" (quota Mistral
épuisé, clé absente, réponse hors format) n'est PAS mémorisé, sinon un incident
passager gèlerait définitivement un trou dans les données.

Ce cache est indépendant de --resume : --resume reprend un run interrompu (au
grain du ticker), le cache survit à TOUS les runs (au grain du document). Un
run complet relancé après une interruption ne repaie donc pas les milliers
d'appels déjà effectués. --no-llm-cache force la ré-analyse.

Prérequis :
    pip install requests beautifulsoup4
    export MISTRAL_API_KEY="ta_cle"   (voir 02_categoriser_secteurs.py -- sans
    cette variable, les 8-K sont journalisés avec category="non_evalue"
    plutôt que de planter)
    export MISTRAL_REQUESTS_PER_SECOND="1"   (facultatif : débit sortant vers
    Mistral, voir sec_filings_text.MISTRAL_RATE_LIMITER)

Usage :
    python 04c_recuperation_8k.py
    python 04c_recuperation_8k.py --limit 10
    python 04c_recuperation_8k.py --resume
    python 04c_recuperation_8k.py --ticker AAPL
    python 04c_recuperation_8k.py --no-llm-cache
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import config
import sec_filings_text as sft

logger = logging.getLogger("recuperation_8k")

CHECKPOINT_EVERY = 10
ITEM_CODE_PATTERN = re.compile(r"Item\s+\d+\.\d+", re.IGNORECASE)

# Cache persistant des 8-K DÉJÀ classifiés par Mistral (voir le docstring).
# JSONL append-only : écrit ligne par ligne au fil du run, donc utilisable même
# après un Ctrl-C ou une coupure -- un JSON réécrit en bloc en fin de run
# perdrait tout le travail d'un run interrompu, exactement le cas qu'il s'agit
# d'éviter.
LLM_CACHE_FILENAME = "cache_8k_mistral.jsonl"
# Classifications qui ne valent PAS mémorisation : ce sont des non-réponses
# (quota épuisé, clé absente, format illisible), pas des verdicts.
NON_CACHEABLE_CATEGORIES = frozenset({None, "", "non_evalue"})

# Au-delà de cette part d'entreprises en échec RÉSEAU, le run échoue
# bruyamment sans rien écrire. Un material_events_8k.parquet incomplet est
# pire qu'absent : load_material_events_8k rend None sur un fichier vide (le
# filtre reste alors sans effet, ce qui se voit), mais un fichier À MOITIÉ
# rempli passe pour complet et laisse entrer des signaux que des 8-K matériels
# auraient dû périmer.
DEFAULT_MAX_FAILURE_RATIO = 0.10

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


# ----------------------------------------------------------------------------
# Mémoire des 8-K déjà classifiés (cache_8k_mistral.jsonl)
# ----------------------------------------------------------------------------

def llm_cache_path(output_dir: Path) -> Path:
    return output_dir / LLM_CACHE_FILENAME


def cache_key(symbol: str, accession_number: str) -> str:
    return f"{symbol}:{accession_number}"


def is_cacheable(classification: dict) -> bool:
    """Vrai seulement si Mistral a rendu un verdict exploitable. Mémoriser un
    "non_evalue" reviendrait à graver dans le marbre l'échec du jour (quota
    Mistral atteint) : le 8-K ne serait plus jamais reproposé à l'analyse."""
    return classification.get("category") not in NON_CACHEABLE_CATEGORIES


def load_llm_cache(output_dir: Path) -> Dict[str, dict]:
    """Cache des classifications déjà obtenues, indexé par symbole:accession.

    Tolérant aux lignes corrompues (un run tué en plein write laisse une ligne
    tronquée) : on ignore la ligne fautive plutôt que de perdre tout le cache
    -- une entrée manquante coûte un appel Mistral, un cache illisible en
    coûte des milliers."""
    path = llm_cache_path(output_dir)
    if not path.exists():
        return {}
    cache: Dict[str, dict] = {}
    ignorees = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                ignorees += 1
                continue
            symbol, accession = entry.get("symbol"), entry.get("accession_number")
            if not symbol or not accession:
                ignorees += 1
                continue
            # Dernière écriture gagnante : une ré-analyse (--no-llm-cache)
            # remplace l'ancien verdict sans qu'il faille réécrire le fichier.
            cache[cache_key(symbol, accession)] = entry
    if ignorees:
        logger.warning("%d ligne(s) illisible(s) ignorée(s) dans %s.", ignorees, path)
    logger.info("Mémoire des classifications : %d 8-K déjà analysés dans %s.", len(cache), path)
    return cache


def append_llm_cache(output_dir: Path, entry: dict) -> None:
    """Écriture IMMÉDIATE, une ligne par 8-K classifié. L'appel Mistral vient
    d'être payé : il ne doit pas être reperdu par un Ctrl-C dix secondes plus
    tard."""
    path = llm_cache_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
        f.flush()


def row_from_cache(entry: dict, symbol: str, cik: str, filing: dict) -> dict:
    """Ligne de sortie reconstruite depuis le cache. Les champs d'identité
    viennent du filing courant (source de vérité), le verdict du cache."""
    return {
        "symbol": symbol, "cik": cik, "filed_date": filing["filing_date"],
        "accession_number": filing["accession_number"],
        "item_codes": entry.get("item_codes") or [],
        "category": entry.get("category"),
        "materiality": entry.get("materiality"),
        "summary": entry.get("summary"),
        "fetch_timestamp": entry.get("fetch_timestamp"),
        "from_cache": True,
    }


def process_ticker_8k(
    symbol: str, cik: str, windows: List[tuple],
    llm_cache: Optional[Dict[str, dict]] = None, output_dir: Optional[Path] = None,
) -> tuple:
    """Lignes 8-K du ticker, plus le nombre de classifications servies par le
    cache. `llm_cache` à None désactive complètement la mémoire (--no-llm-cache)."""
    rows = []
    cache_hits = 0
    seen_accessions = set()
    # UNE seule requête submissions par entreprise, puis filtrage en mémoire.
    # Auparavant, list_company_filings était appelée une fois PAR FENÊTRE :
    # une entreprise avec 40 trimestres TTM connus téléchargeait 40 fois le
    # même JSON (~20 000 requêtes SEC pour ~500 nécessaires à l'échelle du
    # S&P 500).
    submissions = sft.fetch_submissions_strict(cik)
    for start_date, end_date in windows:
        filings = sft.filter_filings(submissions, forms=("8-K",), start_date=start_date, end_date=end_date)
        for filing in filings:
            if filing["accession_number"] in seen_accessions:
                continue  # deux fenêtres adjacentes peuvent se recouvrir sur leur borne commune
            seen_accessions.add(filing["accession_number"])

            # Déjà classifié lors d'un run précédent : ni téléchargement SEC,
            # ni appel Mistral. Le test vient AVANT fetch_filing_text, sinon
            # l'économie se limiterait au LLM.
            if llm_cache is not None:
                connu = llm_cache.get(cache_key(symbol, filing["accession_number"]))
                if connu is not None:
                    rows.append(row_from_cache(connu, symbol, cik, filing))
                    cache_hits += 1
                    continue

            if not filing.get("primary_document"):
                logger.warning(
                    "%s : filing %s du %s sans primary_document, ignoré (filing ancien ?).",
                    symbol, filing["accession_number"], filing["filing_date"],
                )
                continue

            url = sft.filing_document_url(cik, filing["accession_number"], filing["primary_document"])
            # form="8-K" : pas de tentative d'extraction par section (document
            # court, dont l'objet est annoncé dès les premières lignes).
            extracted = sft.fetch_filing_text(url, form=filing["form"])
            if extracted is None:
                continue
            text, _extraction_mode = extracted

            classification = classify_8k(symbol, filing["filing_date"], text)
            row = {
                "symbol": symbol, "cik": cik, "filed_date": filing["filing_date"],
                "accession_number": filing["accession_number"],
                **classification,
                "fetch_timestamp": datetime.now().isoformat(timespec="seconds"),
                "from_cache": False,
            }
            rows.append(row)

            if llm_cache is not None and is_cacheable(classification):
                llm_cache[cache_key(symbol, filing["accession_number"])] = row
                if output_dir is not None:
                    append_llm_cache(output_dir, row)
    return rows, cache_hits


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
    parser.add_argument(
        "--no-llm-cache", action="store_true",
        help="Ignore " + LLM_CACHE_FILENAME + " et re-soumet à Mistral des 8-K déjà classifiés "
             "(à réserver à un changement de prompt ou de modèle : chaque appel est payant).",
    )
    parser.add_argument(
        "--max-failure-ratio", type=float, default=DEFAULT_MAX_FAILURE_RATIO,
        help="Part maximale d'entreprises en échec RÉSEAU tolérée avant d'abandonner le run "
             "sans rien écrire (défaut: %(default)s). Un material_events_8k.parquet incomplet "
             "désactive silencieusement le filtre d'événements matériels du backtest.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if sft.sec_http.require_contact_email(logger) is None:
        sys.exit(1)

    if not os.environ.get(sft.MISTRAL_API_KEY_ENV):
        logger.warning(
            "MISTRAL_API_KEY non définie : les 8-K seront journalisés avec category='non_evalue' "
            "(pas d'appel Mistral). export MISTRAL_API_KEY=... pour activer la classification."
        )
    else:
        logger.info(
            "Débit Mistral : un appel toutes les %.2fs (%s pour l'ajuster au quota de ton plan). "
            "Le débit se resserre automatiquement en cas de 429.",
            sft.MISTRAL_RATE_LIMITER.interval, sft.MISTRAL_REQUESTS_PER_SECOND_ENV,
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
        # Progression et checkpoint sont propres à UN run et repartent de zéro.
        # Le cache des classifications, lui, est délibérément conservé : c'est
        # une mémoire de documents figés, pas l'état d'avancement d'un run.
        _progress_path(args.output_dir).unlink(missing_ok=True)
        _checkpoint_path(args.output_dir).unlink(missing_ok=True)

    llm_cache: Optional[Dict[str, dict]] = None
    if args.no_llm_cache:
        logger.warning(
            "--no-llm-cache : les 8-K déjà classifiés seront re-téléchargés et re-soumis à Mistral."
        )
    else:
        llm_cache = load_llm_cache(args.output_dir)

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

    # Trois compteurs distincts, et non un seul "échec" : une entreprise sans
    # 8-K et une entreprise dont la SEC n'a pas répondu ne veulent pas dire la
    # même chose. Seuls les échecs RÉSEAU remettent en cause la complétude du
    # fichier de sortie (cf. le seuil en fin de run).
    ok_count, not_found_count, network_fail_count, event_count = 0, 0, 0, 0
    cache_hit_count = 0
    since_checkpoint = 0
    interrompu = False
    try:
        for i, symbol in enumerate(to_process, start=1):
            cik = cik_by_symbol[symbol]
            windows = compute_search_windows(ttm, symbol, today)
            logger.info("[%d/%d] %s (CIK %s, %d fenêtre(s))...", i, len(to_process), symbol, cik, len(windows))
            hits = 0
            try:
                rows, hits = process_ticker_8k(symbol, cik, windows, llm_cache, args.output_dir)
            except KeyboardInterrupt:
                # Ctrl-C pendant une attente de quota : sortie propre (le
                # cache et le checkpoint sont déjà sur disque), pas une trace
                # de pile de vingt lignes.
                logger.warning("Interruption demandée : arrêt après %d/%d tickers.", i - 1, len(to_process))
                interrompu = True
                break
            except sft.sec_http.SecNotFound:
                # CIK inconnu de la SEC : réponse définitive, il n'y a rien à
                # récupérer et rien à réessayer. Ce n'est pas un incident.
                logger.info("  -> CIK %s inconnu de la SEC pour %s, ignoré.", cik, symbol)
                rows = []
                not_found_count += 1
            except sft.sec_http.SecUnavailable as exc:
                # La SEC n'a pas répondu : les 8-K de cette entreprise existent
                # peut-être. C'est CE cas qui creusait des trous silencieux.
                logger.warning("  -> SEC indisponible pour %s : %s (à relancer)", symbol, exc)
                rows = []
                network_fail_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("  -> ECHEC pour %s : %s (ticker ignoré, on continue)", symbol, exc)
                rows = []
                network_fail_count += 1
            else:
                ok_count += 1
                event_count += len(rows)
                cache_hit_count += hits
                if rows:
                    append_checkpoint(args.output_dir, rows)

            processed_keys.add(symbol)
            since_checkpoint += 1
            if since_checkpoint >= CHECKPOINT_EVERY:
                save_progress(args.output_dir, processed_keys)
                since_checkpoint = 0
    finally:
        save_progress(args.output_dir, processed_keys)

    logger.info(
        "Terminé. OK: %d | CIK introuvables: %d | Échecs réseau: %d | 8-K détectés: %d "
        "(dont %d servis par la mémoire, %d nouvellement analysés)",
        ok_count, not_found_count, network_fail_count, event_count,
        cache_hit_count, event_count - cache_hit_count,
    )
    if llm_cache is not None:
        logger.info("Mémoire des classifications : %d 8-K au total dans %s.",
                    len(llm_cache), llm_cache_path(args.output_dir))

    if interrompu:
        # Le fichier de sortie serait tronqué à l'endroit exact du Ctrl-C, et
        # un material_events_8k.parquet partiel désactive silencieusement le
        # filtre d'événements matériels du backtest. Rien n'est écrit.
        logger.warning(
            "Run interrompu : aucun fichier de sortie généré. Le travail déjà payé est "
            "conservé (%s) ; relance avec --resume pour reprendre là où tu en étais.",
            llm_cache_path(args.output_dir),
        )
        sys.exit(130)

    failure_ratio = network_fail_count / len(to_process) if to_process else 0.0
    if failure_ratio > args.max_failure_ratio:
        logger.error(
            "%.0f%% des entreprises (%d/%d) ont échoué pour cause d'indisponibilité SEC, "
            "au-delà du seuil de %.0f%%. Le fichier de sortie serait INCOMPLET, et un "
            "material_events_8k.parquet incomplet désactive silencieusement le filtre "
            "d'événements matériels du backtest (load_material_events_8k rend None et le "
            "run continue). Rien n'est écrit : relance 04c_recuperation_8k.py --resume "
            "quand la SEC répond de nouveau.",
            failure_ratio * 100, network_fail_count, len(to_process), args.max_failure_ratio * 100,
        )
        sys.exit(1)

    rows = load_checkpoint_rows(args.output_dir)
    if not rows:
        logger.warning("Aucun 8-K collecté, pas de fichier de sortie généré.")
        return

    df = pd.DataFrame(rows)
    if "from_cache" in df.columns:
        # Un checkpoint écrit par une version antérieure n'a pas la colonne :
        # sans normalisation, le mélange bool/NaN part en colonne "object" et
        # pyarrow refuse d'inférer un type.
        df["from_cache"] = df["from_cache"].fillna(False).astype(bool)
    config.MATERIAL_EVENTS_8K_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.MATERIAL_EVENTS_8K_FILE, index=False, engine="pyarrow")
    logger.info("8-K sauvegardés : %s (%d lignes, %d entreprises).", config.MATERIAL_EVENTS_8K_FILE, len(df), df["symbol"].nunique())
    if "materiality" in df.columns:
        logger.info("Répartition matérialité : %s", df["materiality"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
