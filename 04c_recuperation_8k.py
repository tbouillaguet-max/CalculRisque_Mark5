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
    export GEMINI_API_KEY="ta_cle"    (ou MISTRAL_API_KEY, voir
                                      sec_filings_text.fournisseur_llm -- sans
    cette variable, les 8-K sont journalisés avec category="non_evalue"
    plutôt que de planter)
    export MISTRAL_REQUESTS_PER_SECOND="1"   (facultatif : débit sortant vers le
    LLM, Gemini ou Mistral, voir sec_filings_text.MISTRAL_RATE_LIMITER)

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
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import config
import ecriture_atomique
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


# ----------------------------------------------------------------------------
# Classification SANS modèle, à partir du texte du document
# ----------------------------------------------------------------------------
# POURQUOI. Sans MISTRAL_API_KEY, `classify_8k` renvoyait `non_evalue` et jetait
# le texte qu'il venait de télécharger. Mesuré sur l'archive du dépôt : 99 147
# dépôts, `category` à `non_evalue` sur 100% des lignes, `materiality` et
# `summary` vides partout. Le filtre d'événements matériels -- l'une des deux
# protections anti-value-trap du moteur -- ne s'appliquait donc à RIEN, en
# silence à un avertissement près. Le coûteux (télécharger le document) était
# déjà payé ; seul le jugement manquait.
#
# CE QUE LA RÈGLE LIT, ET POURQUOI C'EST LE DOCUMENT. Deux niveaux, tous deux
# extraits du texte :
#   1. les CODES D'ITEM que le déposant déclare lui-même en tête du 8-K
#      (`extract_item_codes` les parse depuis le document). La SEC les
#      normalise : la matérialité d'un « Item 4.02 » -- non-fiabilité d'états
#      financiers déjà publiés -- tient à la définition du code, pas à une
#      lecture ;
#   2. des FORMULATIONS caractéristiques, cherchées dans le corps du document,
#      qui tranchent là où le code seul est ambigu. C'est le cas décisif de
#      l'Item 5.02, qui couvre aussi bien le départ d'un directeur général que
#      l'élection routinière d'un administrateur.
#
# Déterministe et auditable, donc reproductible d'un run à l'autre -- ce que la
# classification par modèle n'était pas. Elle reste prioritaire quand une clé
# est disponible : la règle est un repli, pas un remplacement.

# Tables lues depuis config : la relecture d'archive (backtest.data_loader) se
# sert des mêmes, et deux copies auraient fini par diverger en silence.
# `_ITEMS_MATERIELS` : matériels par définition SEC, le texte n'ajoute rien.
# `_ITEMS_AMBIGUS` : le texte décide -- un Item 1.01 peut annoncer une fusion
# comme un contrat de fourniture, un Item 5.02 un départ de dirigeant comme une
# élection d'administrateur.
_ITEMS_MATERIELS = dict(config.MATERIAL_8K_ITEM_CATEGORIES)
_ITEMS_AMBIGUS = frozenset(config.AMBIGUOUS_8K_ITEM_CODES)

# Formulations cherchées dans le CORPS du document. L'ordre compte : la
# première catégorie dont un motif est trouvé l'emporte, du plus spécifique au
# plus général.
_MOTIFS_PAR_CATEGORIE = (
    ("fusion_acquisition", (
        r"merger agreement", r"agreement and plan of merger", r"business combination",
        r"definitive agreement to (?:acquire|purchase)", r"tender offer",
        r"agreed to (?:acquire|be acquired)", r"asset purchase agreement",
    )),
    ("procedure_judiciaire", (
        r"chapter 11", r"chapter 7", r"bankruptcy", r"receivership",
        r"class action", r"securities litigation", r"sec investigation",
        r"department of justice", r"subpoena", r"consent decree",
        r"settlement agreement", r"civil penalty",
    )),
    ("changement_guidance", (
        r"(?:revis|updat|lower|rais|reduc|increas)\w*\s+(?:its\s+)?(?:full[- ]year\s+)?(?:financial\s+)?(?:guidance|outlook)",
        r"withdraw\w*\s+(?:its\s+)?(?:guidance|outlook)",
        r"no longer expects", r"now expects", r"suspend\w*\s+(?:its\s+)?guidance",
    )),
    ("rachat_actions", (
        r"share repurchase (?:program|authorization)", r"stock repurchase (?:program|authorization)",
        r"repurchase up to", r"authorized the repurchase", r"buyback program",
    )),
    ("depart_dirigeant", (
        # Restreint aux dirigeants exécutifs ET à un départ : une élection
        # d'administrateur au conseil ne change pas une thèse de valorisation.
        r"(?:chief executive officer|chief financial officer|president|chairman)[^.]{0,120}?"
        r"(?:resign|step(?:ped|ping)? down|depart|terminat|will leave|retire)",
        r"(?:resign|step(?:ped|ping)? down|depart|terminat)\w*[^.]{0,120}?"
        r"(?:chief executive officer|chief financial officer)",
    )),
    ("autre_materiel", (
        r"impairment charge", r"goodwill impairment", r"restructuring (?:plan|charge|program)",
        r"non[- ]reliance", r"should no longer be relied upon", r"material weakness",
        r"restate\w*\s+(?:its\s+)?(?:financial statements|prior)",
        r"delisting", r"notice of noncompliance", r"going concern",
        r"dismissed\s+\w+\s+as (?:its\s+)?independent registered public accounting firm",
    )),
)

_MOTIFS_COMPILES = tuple(
    (categorie, tuple(re.compile(motif, re.IGNORECASE) for motif in motifs))
    for categorie, motifs in _MOTIFS_PAR_CATEGORIE
)

# Une phrase du document, reprise telle quelle comme résumé. Vaut mieux qu'un
# résumé fabriqué : c'est vérifiable contre la source.
_PHRASE = re.compile(r"[^.\n]{40,400}\.")


def _numero_item(code: str) -> str:
    """"Item 5.02" -> "5.02". L'archive contient "Item 9.01", "Item  9.01" et
    "Item\\n9.01" comme trois valeurs distinctes : comparer les chaînes
    entières n'en verrait qu'une."""
    trouve = re.search(r"(\d+\.\d+)", str(code))
    return trouve.group(1) if trouve else ""


def classify_8k_par_regles(item_codes: List[str], text: str) -> dict:
    """Catégorie, matérialité et résumé déduits du DOCUMENT, sans modèle.

    Voir le pavé ci-dessus pour le raisonnement. Rend les mêmes clés que la
    voie modèle, plus `classification_source` -- sans quoi on ne saurait plus,
    en relisant le parquet, lequel des deux chemins a produit une ligne."""
    numeros = {_numero_item(c) for c in item_codes}
    corps = text or ""

    categorie = None
    preuve = None
    for candidate, motifs in _MOTIFS_COMPILES:
        for motif in motifs:
            trouve = motif.search(corps)
            if trouve:
                categorie, preuve = candidate, trouve
                break
        if categorie:
            break

    # Le texte n'a rien dit : les codes non ambigus tranchent seuls.
    if categorie is None:
        for numero in sorted(numeros):
            if numero in _ITEMS_MATERIELS:
                categorie = _ITEMS_MATERIELS[numero]
                break

    # Un motif trouvé dans un document qui ne déclare QUE des codes
    # administratifs (9.01 pièces jointes, 5.07 vote en assemblée) est très
    # probablement une mention de passage, pas l'objet du dépôt.
    if categorie and not (numeros & (set(_ITEMS_MATERIELS) | _ITEMS_AMBIGUS)):
        categorie = None

    if categorie is None:
        return {
            "item_codes": item_codes, "category": "non_materiel", "materiality": False,
            "summary": None, "classification_source": "regles_document",
        }

    resume = None
    if preuve is not None:
        fenetre = corps[max(0, preuve.start() - 200): preuve.end() + 200]
        phrase = _PHRASE.search(fenetre)
        resume = " ".join(phrase.group(0).split())[:300] if phrase else None

    return {
        "item_codes": item_codes, "category": categorie, "materiality": True,
        "summary": resume, "classification_source": "regles_document",
    }


def classify_8k(symbol: str, filed_date: str, text: str) -> dict:
    """Classification d'un 8-K à partir de son texte.

    Le modèle est prioritaire quand une clé est disponible ; à défaut, la
    règle documentaire (`classify_8k_par_regles`) prend le relais plutôt que de
    renvoyer `non_evalue` et de jeter le document. Voir le pavé plus haut."""
    item_codes = extract_item_codes(text)
    prompt = build_prompt(symbol, filed_date, item_codes, text)
    result = sft.analyser_texte_llm(prompt)
    if result is None or "category" not in result:
        return classify_8k_par_regles(item_codes, text)
    # Le fournisseur qui a réellement répondu ("gemini" ou "mistral"), pas un
    # libellé figé : c'est la trace qui permet de comparer les verdicts.
    return {
        "item_codes": item_codes, "category": result.get("category"),
        "materiality": result.get("materiality"), "summary": result.get("summary"),
        "classification_source": sft.fournisseur_llm() or "llm",
    }


# ----------------------------------------------------------------------------
# Mémoire des 8-K déjà classifiés (cache_8k_mistral.jsonl)
# ----------------------------------------------------------------------------

def llm_cache_path(output_dir: Path) -> Path:
    return output_dir / LLM_CACHE_FILENAME


def cache_key(symbol: str, accession_number: str) -> str:
    return f"{symbol}:{accession_number}"


def is_cacheable(classification: dict) -> bool:
    """Vrai seulement si un verdict exploitable a été rendu. Mémoriser un
    "non_evalue" reviendrait à graver dans le marbre l'échec du jour (quota
    Mistral atteint) : le 8-K ne serait plus jamais reproposé à l'analyse.

    Un verdict PAR RÈGLES est mémorisé comme les autres -- il est déterministe,
    et c'est le téléchargement du document qu'on évite de repayer, pas le
    calcul. Il reste distingué par `classification_source`, ce dont
    `load_llm_cache` se sert pour le remettre en jeu le jour où une clé d'API
    devient disponible (cf. sa docstring)."""
    return classification.get("category") not in NON_CACHEABLE_CATEGORIES


def load_llm_cache(output_dir: Path) -> Dict[str, dict]:
    """Cache des classifications déjà obtenues, indexé par symbole:accession.

    Tolérant aux lignes corrompues (un run tué en plein write laisse une ligne
    tronquée) : on ignore la ligne fautive plutôt que de perdre tout le cache
    -- une entrée manquante coûte un appel Mistral, un cache illisible en
    coûte des milliers.

    Les verdicts rendus PAR RÈGLES (`classification_source == "regles_document"`)
    sont ignorés dès qu'une clé d'API est disponible : ils ont été produits
    faute de mieux, et les garder empêcherait le modèle de reprendre la main
    le jour où la clé arrive -- un repli qui se transformerait en plafond."""
    path = llm_cache_path(output_dir)
    if not path.exists():
        return {}
    cache: Dict[str, dict] = {}
    ignorees = 0
    # Gemini OU Mistral : une seule des deux clés suffit à rendre la main au
    # modèle (voir sft.fournisseur_llm).
    llm_disponible = sft.llm_disponible()
    remis_en_jeu = 0
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
            if llm_disponible and entry.get("classification_source") == "regles_document":
                remis_en_jeu += 1
                continue
            # Dernière écriture gagnante : une ré-analyse (--no-llm-cache)
            # remplace l'ancien verdict sans qu'il faille réécrire le fichier.
            cache[cache_key(symbol, accession)] = entry
    if ignorees:
        logger.warning("%d ligne(s) illisible(s) ignorée(s) dans %s.", ignorees, path)
    if remis_en_jeu:
        logger.info(
            "%d 8-K classés par règles remis en jeu : %s est disponible, "
            "le modèle reprend la main dessus.", remis_en_jeu, sft.description_llm(),
        )
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
        # Reporté depuis le cache : sans lui, une ligne relue ne dirait plus
        # lequel des deux chemins l'a produite, et le parquet deviendrait
        # ininterprétable dès qu'un run mélange les deux.
        "classification_source": entry.get("classification_source"),
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
    ecriture_atomique.remplacer(tmp, path)


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
    parser.add_argument(
        "--tickers", type=Path, default=None,
        help="CSV d'univers (défaut : univers point-in-time de 01b s'il existe, "
             "sinon univers actuel de 01 -- voir config.default_universe_file).",
    )
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

    if not sft.llm_disponible():
        logger.warning(
            "Aucune clé LLM (%s ou %s) : les 8-K seront journalisés avec category='non_evalue' "
            "(pas d'appel au modèle). Définis l'une des deux pour activer la classification.",
            sft.GEMINI_API_KEY_ENV, sft.MISTRAL_API_KEY_ENV,
        )
    else:
        logger.info(
            "Classification par %s. Débit : un appel toutes les %.2fs (%s pour l'ajuster au "
            "quota de ton offre). Le débit se resserre automatiquement en cas de 429.",
            sft.description_llm(), sft.MISTRAL_RATE_LIMITER.interval,
            sft.MISTRAL_REQUESTS_PER_SECOND_ENV,
        )

    if not config.FINANCIALS_TTM_FILE.exists():
        logger.error(
            "%s introuvable. Lance d'abord 04b_recuperation_10q.py : ce script a besoin des "
            "trimestres TTM déjà connus pour délimiter les fenêtres de recherche des 8-K.",
            config.FINANCIALS_TTM_FILE,
        )
        sys.exit(1)
    ttm = pd.read_parquet(config.FINANCIALS_TTM_FILE)

    if args.ticker:
        symbols_ric = [args.ticker.upper()]
    else:
        tickers_file = args.tickers or config.default_universe_file()
        if args.tickers is None and tickers_file == config.UNIVERSE_FULL_FILE:
            logger.info(
                "Univers point-in-time retenu (%s) : les entreprises RADIÉES sont incluses. "
                "Sans elles, le backtest ne peut choisir que parmi des survivantes alors que "
                "son indice de référence porte l'indice entier -- biais de survivance. "
                "Le premier run est plus long ; les suivants ignorent les tickers en cache.",
                tickers_file,
            )
        universe = pd.read_csv(tickers_file, encoding="utf-8-sig")
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
