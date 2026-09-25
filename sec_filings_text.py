"""
Module partagé : recherche et extraction de texte de filings SEC EDGAR, pour
07b_validation_qualitative.py et 04c_recuperation_8k.py. Centralisé ici pour
ne pas dupliquer la logique de recherche/téléchargement/extraction entre les
deux, et pour que tout script qui a besoin du texte d'un filing SEC suive la
même contrainte anti-anticipation (voir get_filing_text_asof ci-dessous).

Liste les filings via l'API "submissions" de la SEC
(data.sec.gov/submissions/CIK*.json) plutôt que la recherche plein texte
(efts.sec.gov) : "submissions" renvoie directement les MÉTADONNÉES de CHAQUE
filing (formulaire, date de dépôt, document principal) pour un CIK donné --
énumération complète et fiable pour "tous les 8-K de cette entreprise entre
telle et telle date". La recherche plein texte, elle, est conçue pour une
recherche par MOT-CLÉ dans le contenu (moins adaptée à une énumération
exhaustive par formulaire/date, et sujette à un classement par pertinence qui
peut faire manquer des filings).

L'historique COMPLET est couvert : filings.recent (les ~1000 dépôts les plus
récents) ET les pages anciennes référencées par filings.files. La distinction
compte, parce que `recent` compte tous formulaires confondus et que les
submissions d'un émetteur incluent tous les Form 3/4/5 de ses dirigeants --
une grande capitalisation en dépose plusieurs centaines par an, si bien que
`recent` ne couvre parfois que trois à cinq ans. S'en tenir là faisait
renvoyer None à get_filing_text_asof sur toute la partie ancienne d'un
backtest 2010-2026.

Contrainte anti-anticipation (utilisée par 07b et 04c)
--------------------------------------------------------
get_filing_text_asof(cik, filed_date, ...) ne retourne JAMAIS que le texte du
filing déposé EXACTEMENT à filed_date -- jamais un filing plus récent, jamais
une connaissance agrégée de plusieurs filings. C'est la garantie structurelle
qui empêche un LLM appelé sur ce texte de "voir" des événements postérieurs à
la date simulée : le texte transmis au modèle est physiquement celui d'un
document déposé à cette date-là, rien d'autre.

Prérequis :
    pip install requests beautifulsoup4
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple

from pathlib import Path

import requests
from bs4 import BeautifulSoup

import config
import sec_http

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
# Pages d'historique ancien, référencées par nom dans filings.files.
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{accession_nodash}/{document}"

# Cache des submissions : mémoire pour un run, disque pour les runs suivants.
# Un émetteur ne dépose pas plusieurs fois par jour, et un run interrompu puis
# repris (--resume) ne doit pas tout retélécharger.
SUBMISSIONS_CACHE_DIR = config.DIR_FINANCIALS / "sec_submissions"
SUBMISSIONS_CACHE_TTL_SECONDS = 24 * 3600
_submissions_memory_cache: Dict[str, List[Dict]] = {}

# Budget de texte transmis au LLM (voir analyser_texte_mistral) : un 10-K
# peut faire 100+ pages, très au-delà du contexte utile/payant pour une
# classification. Ce budget est désormais dépensé sur les SECTIONS
# pertinentes, plus sur le début du document (voir fetch_filing_text).
MAX_TEXT_CHARS = 15_000

# Plafond de téléchargement par document. Un 10-K moderne en iXBRL fait 10 à
# 30 Mo de balisage pour quelques dizaines de milliers de caractères de texte
# utile : le flux est coupé au-delà. Large devant MAX_TEXT_CHARS, parce que le
# ratio balisage/texte est de l'ordre de 10 à 30 pour 1 et que les sections
# recherchées se trouvent après le corps du document.
MAX_DOWNLOAD_BYTES = 12_000_000

# Sections d'un 10-K/10-Q sur lesquelles porte réellement le prompt de 07b.
# La regex tolère la casse, les espaces insécables et les variantes de
# ponctuation ("ITEM 1A.", "Item&nbsp;1A —", "Item 1A:"). Le point d'ancrage
# est le début d'intitulé, pas la mention en toutes lettres : les documents
# SEC n'ont aucun balisage sémantique exploitable pour ça.
_ITEM_SEPARATOR = r"[\s   ]*"
_SECTION_PATTERNS = [
    ("Item 1A - Facteurs de risque",
     re.compile(rf"ITEM{_ITEM_SEPARATOR}1A{_ITEM_SEPARATOR}[.:\-–—]?", re.IGNORECASE)),
    ("Item 3 - Procedures judiciaires",
     re.compile(rf"ITEM{_ITEM_SEPARATOR}3{_ITEM_SEPARATOR}[.:\-–—]?(?!\d)", re.IGNORECASE)),
    ("Item 7 - Analyse de la direction",
     re.compile(rf"ITEM{_ITEM_SEPARATOR}7{_ITEM_SEPARATOR}[.:\-–—]?(?!A|\d)", re.IGNORECASE)),
]

# Parser BeautifulSoup retenu, résolu une seule fois (voir _best_parser).
_PARSER: Optional[str] = None

# --------------------------------------------------------------------------- #
# Fournisseur du LLM : Gemini (Google) ou Mistral
# --------------------------------------------------------------------------- #
# 04c, 07b et 02 ne parlent qu'à analyser_texte_llm : le fournisseur se choisit
# ici, par variables d'environnement, sans toucher aux scripts.
#
#   GEMINI_API_KEY définie   -> Gemini (prioritaire)
#   sinon MISTRAL_API_KEY    -> Mistral (comportement historique)
#   LLM_PROVIDER=gemini|mistral force le choix quand les deux clés existent.
#
# Les CLÉS ne s'écrivent jamais dans ce fichier : les constantes *_ENV
# ci-dessous sont les NOMS des variables d'environnement où les lire. Y coller
# une clé revient à chercher une variable qui porterait ce nom -- elle
# n'existe pas, et le LLM est alors jugé indisponible.
LLM_PROVIDER_ENV = "LLM_PROVIDER"

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_ENV = "GEMINI_MODEL"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Les modèles Gemini « à réflexion » décomptent leurs jetons de réflexion de
# maxOutputTokens : avec le budget de 500 jetons prévu pour une réponse JSON
# courte, la réflexion peut tout consommer et la réponse revenir vide. On la
# coupe sur les modèles flash de la génération 2.5 (budget 0 accepté), et on
# réserve une marge sur les autres, où elle ne se désactive pas toujours.
GEMINI_THINKING_HEADROOM_TOKENS = 2048

MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-large-latest"
# Reprises RÉSEAU (dont les 429). Trois ne suffisaient pas : avec le backoff
# ci-dessous elles épuisaient la patience du script en ~15 secondes, alors
# qu'une fenêtre de quota Mistral se compte en dizaines de secondes. Le
# 04c_recuperation_8k.py abandonnait donc la classification (category=
# "non_evalue") dès la première rafale de 429, sur des milliers de documents.
MISTRAL_MAX_RETRIES = 6
# Reprises sur réponse NON PARSABLE (distinct des reprises réseau ci-dessus) :
# une seule. L'ancienne version n'en faisait aucune (abandon immédiat), mais
# insister sur un modèle qui vient de répondre hors format coûte des appels
# payants pour un gain marginal.
MISTRAL_MAX_PARSE_RETRIES = 2
MISTRAL_RETRY_DELAY = 2
MISTRAL_MAX_RETRY_DELAY = 90.0
MISTRAL_TEMPERATURE = 0.1

# Codes HTTP qui justifient un réessai. Tout le reste (401 clé invalide, 403
# compte suspendu, 422 prompt refusé) est DÉFINITIF : réessayer trois fois ne
# fait que retarder l'inévitable en brûlant des appels.
MISTRAL_RETRYABLE_STATUS = frozenset((408, 409, 425, 429, 500, 502, 503, 504))

# Débit sortant vers Mistral. Le vrai correctif du 429 n'est pas de mieux
# réessayer : c'est de ne pas dépasser le quota. Sans limiteur, 04c enchaînait
# ses appels aussi vite que le réseau le permettait et se faisait jeter dès le
# premier ticker. 1 requête/seconde tient dans le quota de tous les plans
# Mistral ; ajustable via MISTRAL_REQUESTS_PER_SECOND pour un plan plus large.
MISTRAL_REQUESTS_PER_SECOND_ENV = "MISTRAL_REQUESTS_PER_SECOND"
MISTRAL_DEFAULT_REQUESTS_PER_SECOND = 1.0
# Plafond de l'auto-freinage : au-delà, ce n'est plus une rafale à lisser mais
# un quota épuisé, et il vaut mieux échouer visiblement que ramper.
MISTRAL_MAX_INTERVAL = 30.0
# Succès consécutifs avant de resserrer l'intervalle élargi par un 429.
MISTRAL_SUCCESSES_BEFORE_SPEEDUP = 20

# Délimiteurs de bloc de code Markdown, que les modèles ajoutent volontiers
# autour d'un JSON ("```json\n{...}\n```") -- l'ancienne exigence
# `content.startswith("{")` les rejetait purement et simplement.
_CODE_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")

logger = logging.getLogger("sec_filings_text")


def _get_json(url: str) -> Optional[dict]:
    """Conservé pour les appelants existants : None quel que soit le motif.
    Le code neuf passe par sec_http.get_json, qui distingue "n'existe pas"
    (SecNotFound) de "pas de réponse" (SecUnavailable)."""
    return sec_http.get_json_or_none(url)


def _normalize_filing_block(block: dict) -> List[Dict]:
    """Un bloc "colonnaire" de l'API submissions (des listes parallèles, une
    par champ) -> une liste de dicts. Les listes peuvent être de longueurs
    différentes sur des filings anciens : zip s'arrête à la plus courte, ce
    qui vaut mieux qu'un IndexError ou qu'un décalage silencieux entre
    colonnes."""
    return [
        {"form": form, "filing_date": filing_date,
         "accession_number": accession, "primary_document": doc}
        for form, filing_date, accession, doc in zip(
            block.get("form", []), block.get("filingDate", []),
            block.get("accessionNumber", []), block.get("primaryDocument", []),
        )
    ]


def _submissions_cache_path(cik: str) -> Path:
    return SUBMISSIONS_CACHE_DIR / f"CIK{cik}.json"


def _read_submissions_cache(cik: str) -> Optional[List[Dict]]:
    path = _submissions_cache_path(cik)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > SUBMISSIONS_CACHE_TTL_SECONDS:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cache submissions illisible pour CIK %s (%s), il sera réinterrogé.", cik, exc)
        return None


def _write_submissions_cache(cik: str, filings: List[Dict]) -> None:
    try:
        SUBMISSIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _submissions_cache_path(cik).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(filings, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_submissions_cache_path(cik))
    except OSError as exc:  # un cache non écrit n'est pas une erreur bloquante
        logger.debug("Cache submissions non écrit pour CIK %s : %s", cik, exc)


def fetch_submissions_strict(cik: str, use_cache: bool = True) -> List[Dict]:
    """Comme fetch_submissions, mais laisse remonter SecNotFound (CIK
    inexistant : rien à récupérer, et rien à signaler comme incident) et
    SecUnavailable (la SEC n'a pas répondu : les données existent peut-être).

    Les appelants qui écrivent un fichier de sortie DOIVENT distinguer les
    deux : confondues, une panne réseau passagère creuse des trous
    silencieux dans le parquet -- et pour material_events_8k.parquet, ces
    trous désactivent ensuite le filtre d'événements matériels sans que rien
    ne le signale."""
    if use_cache:
        cached = _submissions_memory_cache.get(cik)
        if cached is not None:
            return cached
        on_disk = _read_submissions_cache(cik)
        if on_disk is not None:
            _submissions_memory_cache[cik] = on_disk
            return on_disk

    data = sec_http.get_json(SUBMISSIONS_URL.format(cik=cik))
    filings = _assemble_filings(cik, data)
    if use_cache:
        _submissions_memory_cache[cik] = filings
        _write_submissions_cache(cik, filings)
    return filings


def _assemble_filings(cik: str, data: dict) -> List[Dict]:
    blocks = data.get("filings", {})
    filings = _normalize_filing_block(blocks.get("recent", {}))
    filings.extend(_fetch_older_filings(cik, blocks.get("files", [])))

    # Dédoublonnage par accession : les pages anciennes et `recent` peuvent se
    # recouvrir sur leur borne, et un même filing ne doit pas être classifié
    # deux fois par 04c.
    unique: Dict[str, Dict] = {}
    for filing in filings:
        accession = filing.get("accession_number")
        if accession and accession not in unique:
            unique[accession] = filing
    return sorted(unique.values(), key=lambda f: f.get("filing_date") or "")


def fetch_submissions(cik: str, use_cache: bool = True) -> List[Dict]:
    """TOUS les filings connus d'un CIK, normalisés en
    [{form, filing_date, accession_number, primary_document}, ...].

    Séparée de filter_filings parce que 04c_recuperation_8k.py appelait
    list_company_filings UNE FOIS PAR FENÊTRE de recherche : une entreprise
    avec 40 trimestres TTM connus déclenchait 40 téléchargements du MÊME JSON
    (à l'échelle du S&P 500, ~20 000 requêtes pour ~500 nécessaires). Le
    filtrage par formulaire et par dates est une opération purement mémoire :
    il n'a jamais eu besoin de retourner sur le réseau.

    Deux niveaux de cache : mémoire (pour un run) et disque avec TTL (pour
    les runs successifs -- un émetteur ne dépose pas plusieurs fois par jour,
    et un run interrompu puis repris ne doit pas tout retélécharger)."""
    if use_cache:
        cached = _submissions_memory_cache.get(cik)
        if cached is not None:
            return cached
        on_disk = _read_submissions_cache(cik)
        if on_disk is not None:
            _submissions_memory_cache[cik] = on_disk
            return on_disk

    data = _get_json(SUBMISSIONS_URL.format(cik=cik))
    if not data:
        return []

    filings = _assemble_filings(cik, data)

    if use_cache:
        _submissions_memory_cache[cik] = filings
        _write_submissions_cache(cik, filings)
    return filings


def _fetch_older_filings(cik: str, files: List[dict]) -> List[Dict]:
    """Pages d'historique ancien référencées par data["filings"]["files"].

    `recent` ne contient que les ~1000 derniers dépôts, TOUS FORMULAIRES
    CONFONDUS. Or les submissions d'un émetteur incluent tous les Form 3/4/5
    de ses dirigeants : une grande capitalisation en dépose plusieurs
    centaines par an, si bien que `recent` ne couvre parfois que trois à cinq
    ans. Sur un backtest 2010-2026, get_filing_text_asof renvoyait donc None
    pour toute la partie ancienne -- et 07b journalisait "non_evalue" sans
    distinguer ce cas d'un vrai manque de verdict.

    Une page qu'on ne sait pas récupérer est journalisée et sautée : mieux
    vaut un historique partiel signalé qu'un échec de tout le ticker."""
    older: List[Dict] = []
    for page in files or []:
        name = page.get("name") if isinstance(page, dict) else None
        if not name:
            continue
        data = _get_json(SUBMISSIONS_PAGE_URL.format(name=name))
        if not data:
            logger.warning(
                "Page d'historique %s inaccessible pour CIK %s : les filings qu'elle contient "
                "resteront introuvables (verdicts 'non_evalue_filing_introuvable').", name, cik,
            )
            continue
        # Les pages anciennes portent les mêmes listes colonnaires que
        # `recent`, mais à la RACINE du document, sans enveloppe "filings".
        older.extend(_normalize_filing_block(data))
    return older


def filter_filings(
    filings: List[Dict], forms: tuple = ("10-K", "10-Q"),
    start_date: Optional[str] = None, end_date: Optional[str] = None,
) -> List[Dict]:
    """Filtre EN MÉMOIRE par formulaire et fenêtre de dates de DÉPÔT
    (filingDate SEC, format YYYY-MM-DD, bornes incluses).

    Le résultat est trié par date de dépôt, puis par l'ORDRE DE PRÉFÉRENCE de
    la tuple `forms` -- voir get_filing_text_asof pour ce que ce second
    critère résout."""
    preference = {form: rank for rank, form in enumerate(forms)}
    results = [
        f for f in filings
        if f.get("form") in preference
        and not (start_date and (f.get("filing_date") or "") < start_date)
        and not (end_date and (f.get("filing_date") or "") > end_date)
    ]
    results.sort(key=lambda f: (f.get("filing_date") or "", preference[f["form"]]))
    return results


def list_company_filings(
    cik: str, forms: tuple = ("10-K", "10-Q"),
    start_date: Optional[str] = None, end_date: Optional[str] = None,
) -> List[Dict]:
    """Filings d'une entreprise (CIK, 10 chiffres) filtrés par formulaire et
    fenêtre de dates de dépôt. Liste vide si le CIK est introuvable ou n'a
    aucun filing correspondant -- pas une erreur bloquante.

    Conservée telle quelle pour les appelants existants ; elle n'est
    désormais qu'un raccourci sur fetch_submissions + filter_filings, et le
    JSON n'est plus retéléchargé à chaque appel (voir fetch_submissions)."""
    return filter_filings(fetch_submissions(cik), forms=forms, start_date=start_date, end_date=end_date)


def filing_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    """URL du document principal d'un filing dans les archives EDGAR.
    Attention : contrairement à l'API companyfacts (CIK avec zéros de tête,
    10 chiffres), les archives utilisent le CIK SANS zéros de tête."""
    cik_nolead = str(int(cik))
    accession_nodash = accession_number.replace("-", "")
    return ARCHIVES_URL.format(cik_nolead=cik_nolead, accession_nodash=accession_nodash, document=primary_document)


def _best_parser() -> str:
    """"lxml" s'il est installé, sinon "html.parser".

    Un 10-K moderne en iXBRL fait plusieurs mégaoctets de balisage :
    html.parser y est plusieurs fois plus lent que lxml, et ce document est
    parsé une fois par période évaluée."""
    global _PARSER
    if _PARSER is None:
        try:
            import lxml  # noqa: F401
            _PARSER = "lxml"
        except ImportError:
            _PARSER = "html.parser"
            logger.debug("lxml indisponible : repli sur html.parser (plus lent sur les gros filings).")
    return _PARSER


def _download_bounded(url: str, max_bytes: int) -> Optional[bytes]:
    """Télécharge au plus max_bytes, en flux, avec arrêt anticipé.

    Les filings modernes en iXBRL font 10 à 30 Mo, intégralement téléchargés
    jusqu'ici pour n'en garder que quelques milliers de caractères de texte.
    L'arrêt anticipé borne le transfert ; le budget est pris large devant le
    budget de TEXTE, parce que le ratio balisage/texte d'un iXBRL est de
    l'ordre de 10 à 30 pour 1 et que les sections recherchées (Items 1A, 3, 7)
    se trouvent après le corps du document."""
    try:
        resp = sec_http.request(url, stream=True)
    except (sec_http.SecNotFound, sec_http.SecUnavailable) as e:
        logger.error("Échec du téléchargement du filing %s: %s", url, e)
        return None

    chunks, total = [], 0
    try:
        for chunk in resp.iter_content(chunk_size=65_536):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= max_bytes:
                logger.debug("%s tronqué à %d octets (budget de téléchargement atteint).", url, total)
                break
    except requests.exceptions.RequestException as e:
        # Coupure en cours de flux : on garde ce qui est déjà arrivé plutôt
        # que de tout perdre -- un document partiel reste exploitable, et
        # l'extraction par section dira si les sections voulues y sont.
        logger.warning("Flux interrompu pour %s (%s) : %d octets exploités.", url, e, total)
        if not chunks:
            return None
    finally:
        resp.close()

    return b"".join(chunks)


def _extract_sections(text: str, max_chars: int) -> Optional[str]:
    """Concatène les sections d'un 10-K/10-Q qui portent réellement ce que le
    prompt de 07b cherche : Item 1A (facteurs de risque), Item 3 (procédures
    judiciaires) et Item 7 (analyse de la direction, dont la guidance).

    POURQUOI. La troncature d'origine gardait les max_chars PREMIERS
    caractères du document. Or les 15 000 premiers caractères d'un 10-K sont
    la page de garde, la table des matières et le début de l'Item 1
    (Business) : une dépréciation d'actif, un litige matériel, une révision de
    guidance ou un doute sur la continuité d'exploitation se trouvent
    typiquement entre 30 000 et 80 000 caractères plus loin. Les verdicts
    produits ne portaient donc pas sur ce qu'on croyait mesurer.

    Le budget est réparti ENTRE LES SECTIONS TROUVÉES, pas alloué au premier
    arrivé : trouver l'Item 1A ne doit pas consommer tout l'espace au point
    de faire disparaître l'Item 7.

    None si aucune section n'est localisée -- à l'appelant de retomber sur le
    début du document, en le signalant."""
    matches = []
    for label, pattern in _SECTION_PATTERNS:
        found = list(pattern.finditer(text))
        if found:
            # La table des matières cite les mêmes intitulés que le corps du
            # document : la DERNIÈRE occurrence est la bonne dans la grande
            # majorité des cas, la première étant celle du sommaire.
            matches.append((label, found[-1].start()))

    if not matches:
        return None

    matches.sort(key=lambda item: item[1])
    budget = max_chars // len(matches)
    morceaux = []
    for index, (label, start) in enumerate(matches):
        # Fin = début de la section suivante trouvée, sinon fin du document.
        end = matches[index + 1][1] if index + 1 < len(matches) else len(text)
        extrait = text[start:min(end, start + budget)].strip()
        if extrait:
            morceaux.append(f"[{label}]\n{extrait}")

    return "\n\n".join(morceaux)[:max_chars] if morceaux else None


def fetch_filing_text(
    url: str, max_chars: int = MAX_TEXT_CHARS, form: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """Télécharge un document de filing et en extrait le texte transmis au
    modèle. Retourne (texte, extraction_mode) où extraction_mode vaut :

        "sections"        Items 1A/3/7 localisés et concaténés (10-K/10-Q).
        "debut_document"  Aucune section localisée : repli sur le début du
                          document, comportement historique.

    Ce mode est remonté jusqu'au parquet de sortie de 07b : un verdict rendu
    sur le début du document ne porte pas sur la même chose qu'un verdict
    rendu sur les sections de risque, et la différence doit être auditable
    plutôt que devinée.

    Les 8-K sont laissés au repli "début de document" sans même tenter la
    localisation : ce sont des documents courts, dont l'objet est annoncé dès
    les premières lignes -- la troncature y est sans effet.

    None si le téléchargement échoue."""
    content = _download_bounded(url, MAX_DOWNLOAD_BYTES)
    if content is None:
        return None

    soup = BeautifulSoup(content, _best_parser())
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)

    if form is None or form.upper().startswith(("10-K", "10-Q")):
        sections = _extract_sections(text, max_chars)
        if sections:
            return sections, "sections"
        logger.debug("%s : aucune section Item 1A/3/7 localisée, repli sur le début du document.", url)

    return text[:max_chars], "debut_document"


def get_filing_text_asof(cik: str, filed_date: str, forms: tuple = ("10-K", "10-Q")) -> Optional[Dict]:
    """Le filing EXACT déposé à filed_date -- jamais un autre, ni plus
    récent ni plus ancien (voir "Contrainte anti-anticipation" en tête de
    fichier). filed_date doit correspondre à une vraie date de dépôt (cas
    normal : c'est déjà la filed_date XBRL extraite par 04/04b pour la même
    période). None si aucun filing de ce type n'a été déposé exactement à
    cette date (CIK introuvable, désynchronisation de cache, etc.).

    Quand PLUSIEURS filings partagent la même date de dépôt, l'ordre de la
    tuple `forms` départage : elle exprime une préférence (07b passe
    ("10-Q", "10-K") pour une période TTM, ("10-K",) pour un exercice), mais
    le filtre `if form not in forms` ne la respectait pas -- c'était
    `filings[0]`, donc l'ordre du JSON SEC, qui décidait. filter_filings trie
    désormais sur ce critère (cf. D3), et le cas est journalisé en debug.

    Retourne {form, filing_date, accession_number, primary_document, text,
    extraction_mode}."""
    filings = list_company_filings(cik, forms=forms, start_date=filed_date, end_date=filed_date)
    if not filings:
        return None
    if len(filings) > 1:
        logger.debug(
            "CIK %s : %d filings déposés le %s (%s) -- le premier dans l'ordre de préférence "
            "%s est retenu.", cik, len(filings), filed_date,
            ", ".join(f"{f['form']}/{f['accession_number']}" for f in filings), list(forms),
        )
    filing = filings[0]

    if not filing.get("primary_document"):
        # Certains filings anciens n'ont pas de primaryDocument : l'URL
        # construite serait invalide (".../accession/") et le téléchargement
        # ramènerait la page d'index, pas le document.
        logger.warning(
            "CIK %s, filing %s du %s : primary_document vide, document introuvable "
            "(filing ancien ?). Période non évaluée.",
            cik, filing.get("accession_number"), filed_date,
        )
        return None

    url = filing_document_url(cik, filing["accession_number"], filing["primary_document"])
    extracted = fetch_filing_text(url, form=filing.get("form"))
    if extracted is None:
        return None
    text, extraction_mode = extracted
    return {**filing, "text": text, "extraction_mode": extraction_mode}


def _parse_json_reponse(content) -> Optional[dict]:
    """Objet JSON contenu dans une réponse de modèle, ou None.

    L'ancienne version exigeait `content.startswith("{")` et
    `content.endswith("}")` : une réponse encadrée de ```json ... ``` -- ce
    que les modèles produisent spontanément -- était rejetée sans réessai,
    alors que le JSON attendu était bien là. Trois niveaux de tolérance, du
    plus strict au plus permissif, pour ne jamais accepter autre chose qu'un
    objet JSON complet."""
    if not isinstance(content, str):
        return None
    texte = _CODE_FENCE.sub("", content.strip()).strip()

    try:
        parsed = json.loads(texte)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Dernier recours : le modèle a encadré son JSON de prose. On isole la
    # première accolade ouvrante et la dernière fermante.
    debut, fin = texte.find("{"), texte.rfind("}")
    if debut == -1 or fin <= debut:
        return None
    try:
        parsed = json.loads(texte[debut:fin + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


class AdaptiveRateLimiter:
    """Limiteur de débit qui se resserre tout seul quand le serveur dit non.

    Même principe que sec_http.RateLimiter (attente sous verrou, donc débit
    global borné quel que soit le nombre de threads), avec une différence :
    l'intervalle n'est pas figé. Le quota Mistral dépend du plan du compte et
    n'est annoncé nulle part -- le seul moyen de le connaître est de s'y
    cogner. Chaque 429 DOUBLE donc l'intervalle (jusqu'à `max_interval`), et
    une série de succès le ramène progressivement vers sa valeur nominale.

    Sans ça, réessayer après un 429 ne fait que déplacer le problème : la
    requête suivante repart au même rythme et se fait refuser de nouveau."""

    def __init__(self, rate_per_second: float, max_interval: float = MISTRAL_MAX_INTERVAL,
                 successes_before_speedup: int = MISTRAL_SUCCESSES_BEFORE_SPEEDUP):
        self._base_interval = 1.0 / rate_per_second if rate_per_second > 0 else 0.0
        self._interval = self._base_interval
        self._max_interval = max_interval
        self._successes_before_speedup = successes_before_speedup
        self._successes = 0
        self._lock = threading.Lock()
        self._next_slot = 0.0

    @property
    def interval(self) -> float:
        return self._interval

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait > 0:
                time.sleep(wait)
                now = self._next_slot
            self._next_slot = now + self._interval

    def penalize(self, pause: float = 0.0) -> float:
        """Un 429 vient d'arriver : on ralentit, et on décale le prochain
        créneau d'au moins `pause` (typiquement la valeur de Retry-After) pour
        que la reprise ne parte pas avant la fin de la fenêtre de quota."""
        with self._lock:
            self._successes = 0
            self._interval = min(max(self._interval * 2, self._base_interval or 1.0), self._max_interval)
            self._next_slot = max(self._next_slot, time.monotonic() + max(pause, 0.0))
            return self._interval

    def reward(self) -> None:
        """Appel réussi : après une série assez longue, on desserre d'un cran
        (jamais en dessous de l'intervalle nominal)."""
        with self._lock:
            if self._interval <= self._base_interval:
                return
            self._successes += 1
            if self._successes >= self._successes_before_speedup:
                self._successes = 0
                self._interval = max(self._interval / 2, self._base_interval)


def _mistral_rate_per_second() -> float:
    brut = os.environ.get(MISTRAL_REQUESTS_PER_SECOND_ENV, "").strip()
    if not brut:
        return MISTRAL_DEFAULT_REQUESTS_PER_SECOND
    try:
        valeur = float(brut)
    except ValueError:
        logger.warning("%s='%s' illisible, valeur par défaut (%s req/s).",
                       MISTRAL_REQUESTS_PER_SECOND_ENV, brut, MISTRAL_DEFAULT_REQUESTS_PER_SECOND)
        return MISTRAL_DEFAULT_REQUESTS_PER_SECOND
    if valeur <= 0:
        logger.warning("%s doit être > 0 (reçu %s), valeur par défaut (%s req/s).",
                       MISTRAL_REQUESTS_PER_SECOND_ENV, valeur, MISTRAL_DEFAULT_REQUESTS_PER_SECOND)
        return MISTRAL_DEFAULT_REQUESTS_PER_SECOND
    return valeur


# Limiteur GLOBAL au processus, partagé par 04c et 07b : deux modules qui
# appelleraient Mistral en parallèle avec chacun le sien doubleraient le débit
# réel, donc le taux de 429.
MISTRAL_RATE_LIMITER = AdaptiveRateLimiter(_mistral_rate_per_second())


def _retry_after_seconds(response: Optional["requests.Response"]) -> Optional[float]:
    """Valeur de l'en-tête Retry-After, en secondes. L'en-tête admet deux
    formes (un nombre de secondes ou une date HTTP) et Mistral utilise la
    première ; la seconde est gérée pour ne pas dépendre de ce détail."""
    if response is None:
        return None
    brut = (getattr(response, "headers", None) or {}).get("Retry-After")
    if not brut:
        return None
    brut = str(brut).strip()
    try:
        return max(float(brut), 0.0)
    except ValueError:
        pass
    try:
        cible = parsedate_to_datetime(brut)
    except (TypeError, ValueError):
        return None
    if cible is None:
        return None
    maintenant = datetime.now(timezone.utc) if cible.tzinfo else datetime.now()
    return max((cible - maintenant).total_seconds(), 0.0)


def _retry_delay_from_body(response: Optional["requests.Response"]) -> Optional[float]:
    """Délai de reprise annoncé DANS LE CORPS de la réponse, en secondes.

    Gemini n'envoie pas d'en-tête Retry-After sur un 429 : il place le délai
    dans `error.details[].retryDelay`, sous la forme "36s". None si absent ou
    illisible."""
    if response is None:
        return None
    try:
        corps = response.json()
    except (ValueError, AttributeError):
        return None
    if not isinstance(corps, dict) or not isinstance(corps.get("error"), dict):
        return None
    for detail in corps["error"].get("details") or []:
        brut = detail.get("retryDelay") if isinstance(detail, dict) else None
        if isinstance(brut, str) and brut.endswith("s"):
            try:
                return max(float(brut[:-1]), 0.0)
            except ValueError:
                continue
    return None


def _mistral_retry_delay(response: Optional["requests.Response"], attempt: int) -> float:
    """Retry-After s'il est fourni (le serveur sait mieux que nous), sinon
    backoff exponentiel plafonné avec jitter -- le jitter évite que plusieurs
    appelants repartis en même temps ne se resynchronisent sur le quota."""
    retry_after = _retry_after_seconds(response)
    if retry_after is None:
        retry_after = _retry_delay_from_body(response)
    if retry_after is not None:
        return min(retry_after + random.uniform(0, 1), MISTRAL_MAX_RETRY_DELAY)
    return min(MISTRAL_RETRY_DELAY * (2 ** attempt), MISTRAL_MAX_RETRY_DELAY) + random.uniform(0, 1)


def fournisseur_llm() -> Optional[str]:
    """Fournisseur du LLM à utiliser : "gemini", "mistral", ou None si aucune
    clé n'est disponible (voir l'en-tête « Fournisseur du LLM »).

    LLM_PROVIDER force le choix ; il ne crée pas de clé pour autant : forcer
    "gemini" sans GEMINI_API_KEY rend None plutôt que de basculer en silence
    sur l'autre fournisseur, et le journal dit alors pourquoi."""
    force = os.environ.get(LLM_PROVIDER_ENV, "").strip().lower()
    cles = {"gemini": GEMINI_API_KEY_ENV, "mistral": MISTRAL_API_KEY_ENV}
    if force:
        if force not in cles:
            logger.warning("%s='%s' inconnu (attendu : gemini ou mistral) : choix automatique.",
                           LLM_PROVIDER_ENV, force)
        else:
            return force if os.environ.get(cles[force]) else None
    for nom in ("gemini", "mistral"):
        if os.environ.get(cles[nom]):
            return nom
    return None


def llm_disponible() -> bool:
    """Vrai si une clé d'API permet d'appeler un LLM (Gemini ou Mistral)."""
    return fournisseur_llm() is not None


def aide_cle_absente() -> str:
    """Quoi vérifier quand aucune clé n'est vue. Le cas fréquent n'est pas une
    clé manquante, mais une clé posée là où ce processus ne la voit pas."""
    import env_local

    noms = (GEMINI_API_KEY_ENV, MISTRAL_API_KEY_ENV)
    if any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", nom) for nom in noms):
        return ("sec_filings_text.py a été modifié : GEMINI_API_KEY_ENV et MISTRAL_API_KEY_ENV "
                "doivent contenir le NOM d'une variable ('GEMINI_API_KEY'), pas la clé. Restaure "
                "le fichier (git checkout -- sec_filings_text.py) et mets la clé dans .env.")
    return (f"Ajoute la ligne {GEMINI_API_KEY_ENV}=ta_cle au fichier {env_local.FICHIER} (lu par "
            f"tous les scripts, jamais poussé sur git), ou, sous PowerShell, "
            f"$env:{GEMINI_API_KEY_ENV} = \"ta_cle\" dans CE terminal. Après setx, seul un "
            "NOUVEAU terminal voit la variable.")


def description_llm() -> str:
    """Libellé pour les journaux : fournisseur et modèle, ou comment en activer un."""
    fournisseur = fournisseur_llm()
    if fournisseur == "gemini":
        return f"Gemini ({_gemini_model()})"
    if fournisseur == "mistral":
        return f"Mistral ({MISTRAL_MODEL})"
    return f"aucun LLM ({GEMINI_API_KEY_ENV} ou {MISTRAL_API_KEY_ENV} à définir)"


def _gemini_model() -> str:
    return os.environ.get(GEMINI_MODEL_ENV, "").strip() or GEMINI_DEFAULT_MODEL


def _requete_gemini(prompt: str, max_tokens: int, api_key: str) -> Tuple[str, dict, dict]:
    """(url, en-têtes, corps) d'un appel generateContent de l'API Gemini."""
    model = _gemini_model()
    generation = {
        "temperature": MISTRAL_TEMPERATURE,
        # Mode JSON natif, l'équivalent du response_format de Mistral.
        "responseMimeType": "application/json",
    }
    if model.startswith("gemini-2.5-flash"):
        generation["maxOutputTokens"] = max_tokens
        generation["thinkingConfig"] = {"thinkingBudget": 0}
    else:
        generation["maxOutputTokens"] = max_tokens + GEMINI_THINKING_HEADROOM_TOKENS
    return (
        GEMINI_URL.format(model=model),
        # Clé dans un en-tête plutôt que dans l'URL (?key=...) : une URL finit
        # dans les messages d'erreur de requests, donc dans les journaux.
        {"x-goog-api-key": api_key, "Content-Type": "application/json"},
        {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": generation},
    )


def _requete_mistral(prompt: str, max_tokens: int, api_key: str) -> Tuple[str, dict, dict]:
    """(url, en-têtes, corps) d'un appel chat/completions de l'API Mistral."""
    return (
        MISTRAL_URL,
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        {
            "model": MISTRAL_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": MISTRAL_TEMPERATURE,
            "max_tokens": max_tokens,
            # Mode JSON natif de l'API : le modèle ne peut plus encadrer sa
            # réponse d'un bloc de code ni la préfixer d'une phrase. C'est la
            # correction à la racine ; _parse_json_reponse reste en garde-fou.
            "response_format": {"type": "json_object"},
        },
    )


def _contenu_gemini(data: dict) -> str:
    """Texte de la réponse Gemini. KeyError/IndexError si la réponse n'en porte
    pas -- prompt bloqué (promptFeedback), ou réponse tronquée sans texte --,
    traité par l'appelant comme une enveloppe inexploitable."""
    parts = data["candidates"][0]["content"]["parts"]
    texte = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
    if not texte:
        raise KeyError("text")
    return texte


def _contenu_mistral(data: dict) -> str:
    return data["choices"][0]["message"]["content"]


def _extrait_erreur(response: Optional["requests.Response"]) -> str:
    """Message d'erreur renvoyé par l'API, tronqué : c'est lui qui dit POURQUOI
    un 403 ou un 400 est refusé (offre non activée, modèle inaccessible, clé
    révoquée...), là où le seul code HTTP ne le dit pas."""
    if response is None:
        return ""
    try:
        corps = response.json()
        if isinstance(corps, dict):
            erreur = corps.get("error")
            if isinstance(erreur, dict) and erreur.get("message"):
                return str(erreur["message"])[:300]
            if corps.get("message"):
                return str(corps["message"])[:300]
            if corps.get("detail"):
                return str(corps["detail"])[:300]
    except (ValueError, AttributeError):
        pass
    return str(getattr(response, "text", "") or "")[:300]


# --------------------------------------------------------------------------- #
# Disjoncteur de quota
# --------------------------------------------------------------------------- #
# POURQUOI. Un quota PAR MINUTE se résorbe pendant les réessais ; un quota PAR
# JOUR, non. Une fois celui du palier gratuit de Gemini atteint, chaque appel
# épuisait ses MISTRAL_MAX_RETRIES tentatives, espacées jusqu'à
# MISTRAL_MAX_RETRY_DELAY secondes, avant de rendre None : plusieurs minutes
# par document, pour rien. Sur les ~97 500 8-K que 04c avait à classer, le run
# rampait des jours au lieu de finir.
#
# Après LLM_REFUS_QUOTA_AVANT_COUPURE analyses de suite refusées pour quota
# (429 jusqu'à la dernière tentative), plus aucun appel pour le reste du
# PROCESSUS : analyser_texte_llm rend None tout de suite, et chaque appelant
# fait ce qu'il fait déjà sans modèle (04c classe par règles -- verdicts que
# le modèle reprend au run suivant --, 07b journalise non_evalue). Un seul
# succès, ou un échec d'une autre nature, remet le compte à zéro.
LLM_REFUS_QUOTA_AVANT_COUPURE = 3
_refus_quota_consecutifs = 0
_llm_coupe_pour_ce_run = False


def reinitialiser_disjoncteur_llm() -> None:
    """Réarme le disjoncteur (tests ; un run de production est un processus)."""
    global _refus_quota_consecutifs, _llm_coupe_pour_ce_run
    _refus_quota_consecutifs = 0
    _llm_coupe_pour_ce_run = False


def llm_coupe_pour_ce_run() -> bool:
    return _llm_coupe_pour_ce_run


def analyser_texte_llm(prompt: str, max_tokens: int = 500) -> Optional[dict]:
    """Appelle le LLM configuré (Gemini ou Mistral, voir fournisseur_llm) avec
    un prompt demandant une réponse JSON stricte -- retries avec backoff
    exponentiel, la réponse DOIT être un objet JSON valide.
    Généraliste (pas de schéma imposé ici) : chaque appelant (07b, 04c, 02)
    construit son propre prompt et valide les clés qu'il attend dans le dict
    retourné. None si aucune clé n'est définie, ou après épuisement des
    tentatives -- l'appelant doit traiter ce cas comme "pas de verdict",
    jamais planter.

    Les appels passent par MISTRAL_RATE_LIMITER (voir AdaptiveRateLimiter),
    commun aux deux fournisseurs : espacés en amont pour ne pas provoquer de
    429, et espacés DAVANTAGE dès qu'un 429 survient malgré tout."""
    global _refus_quota_consecutifs, _llm_coupe_pour_ce_run
    fournisseur = fournisseur_llm()
    if fournisseur is None or _llm_coupe_pour_ce_run:
        return None
    if fournisseur == "gemini":
        url, headers, payload = _requete_gemini(prompt, max_tokens, os.environ[GEMINI_API_KEY_ENV])
        extraire, nom = _contenu_gemini, "Gemini"
    else:
        url, headers, payload = _requete_mistral(prompt, max_tokens, os.environ[MISTRAL_API_KEY_ENV])
        extraire, nom = _contenu_mistral, "Mistral"

    parse_failures = 0
    network_failures = 0
    derniere_erreur: Optional[Exception] = None

    while network_failures < MISTRAL_MAX_RETRIES:
        MISTRAL_RATE_LIMITER.acquire()
        resp = None
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=45)
            statut = getattr(resp, "status_code", None)
            if statut is not None and statut >= 400:
                raise requests.exceptions.HTTPError(f"HTTP {statut}", response=resp)
            _refus_quota_consecutifs = 0     # le fournisseur répond : pas de quota épuisé
            content = extraire(resp.json())
        except requests.exceptions.RequestException as e:
            statut = getattr(getattr(e, "response", None), "status_code", None)
            if statut is not None and statut not in MISTRAL_RETRYABLE_STATUS:
                # 401 (clé invalide), 403, 422... : la réponse ne changera pas.
                logger.error("Appel %s refusé définitivement (HTTP %s), aucun réessai : %s",
                             nom, statut, _extrait_erreur(getattr(e, "response", None)) or e)
                _refus_quota_consecutifs = 0
                return None

            network_failures += 1
            derniere_erreur = e
            if network_failures >= MISTRAL_MAX_RETRIES:
                break

            delay = _mistral_retry_delay(resp, network_failures - 1)
            if statut == 429:
                intervalle = MISTRAL_RATE_LIMITER.penalize(pause=delay)
                logger.warning(
                    "Quota %s atteint (429, tentative %d/%d). Débit ramené à un appel "
                    "toutes les %.1fs ; nouvel essai dans %.1fs...",
                    nom, network_failures, MISTRAL_MAX_RETRIES, intervalle, delay,
                )
            else:
                logger.warning("Tentative %s %d/%d échouée: %s. Nouvel essai dans %.1fs...",
                               nom, network_failures, MISTRAL_MAX_RETRIES, e, delay)
            time.sleep(delay)
            continue
        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.error("Réponse %s inexploitable (enveloppe): %s", nom, e)
            return None

        MISTRAL_RATE_LIMITER.reward()
        parsed = _parse_json_reponse(content)
        if parsed is not None:
            return parsed

        # Réponse non parsable : UNE seule reprise, pas trois. L'ancienne
        # version abandonnait immédiatement (return None sans réessai), mais
        # insister davantage sur un modèle qui vient de répondre hors format
        # coûte trois appels payants pour un gain marginal.
        parse_failures += 1
        if parse_failures >= MISTRAL_MAX_PARSE_RETRIES:
            logger.warning("Réponse %s non exploitable après %d essais : %s", nom, parse_failures, str(content)[:200])
            return None
        logger.warning("Réponse %s non parsable, une nouvelle tentative : %s", nom, str(content)[:200])

    logger.error("Échec après %d tentatives %s : %s", MISTRAL_MAX_RETRIES, nom, derniere_erreur)
    statut_final = getattr(getattr(derniere_erreur, "response", None), "status_code", None)
    if statut_final != 429:
        _refus_quota_consecutifs = 0
        return None
    _refus_quota_consecutifs += 1
    if _refus_quota_consecutifs >= LLM_REFUS_QUOTA_AVANT_COUPURE:
        _llm_coupe_pour_ce_run = True
        logger.error(
            "Quota %s épuisé : %d analyses de suite refusées (429) malgré %d tentatives chacune. "
            "Plus aucun appel au modèle jusqu'à la fin de ce run -- les documents restants sont "
            "traités sans lui (04c les classe par règles, et le modèle les reprendra au prochain "
            "run). Relance plus tard, ou passe à une offre payante et relève %s.",
            nom, _refus_quota_consecutifs, MISTRAL_MAX_RETRIES, MISTRAL_REQUESTS_PER_SECOND_ENV)
    return None


# Nom historique, conservé pour les appelants qui ne sont pas encore passés à
# analyser_texte_llm : il suit le même choix de fournisseur.
analyser_texte_mistral = analyser_texte_llm
