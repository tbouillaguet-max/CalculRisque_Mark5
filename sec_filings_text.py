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

import functools
import json
import logging
import os
import random
import re
import threading
import time
import warnings
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, NamedTuple, Optional, Tuple
from urllib.parse import urljoin

from pathlib import Path

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

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

# Budget de texte transmis au LLM (voir analyser_document) : un 10-K
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
# Le LLM : Gemini (Google), API generateContent
# --------------------------------------------------------------------------- #
# 04c, 07b et 02 ne parlent qu'à analyser_document (un document) et
# analyser_documents (plusieurs par requête) : le modèle se règle ici, par
# variables d'environnement, sans toucher aux scripts.
#
# UNE REQUÊTE, TROIS ZONES. Pour soumettre un document avec une consigne et le
# format de la réponse, generateContent sépare les trois -- corps identique à
# celui qu'envoie le SDK officiel google-genai 2.25 (voir
# tests/test_appel_gemini.py::test_le_corps_est_celui_du_sdk_officiel) :
#
#   systemInstruction          la CONSIGNE : rôle, tâche, critères ;
#   contents                   le DOCUMENT, tel quel ;
#   generationConfig           le FORMAT : responseMimeType "application/json"
#     .responseJsonSchema      et le schéma JSON de la réponse.
#
# Avec un schéma, Gemini génère SOUS CONTRAINTE : il ne peut produire qu'un
# JSON conforme -- champs obligatoires présents, valeur d'un `enum` prise dans
# la liste, vrai booléen pour un booléen. Le mode JSON seul, sans schéma, ne
# garantissait que du JSON : la mémoire des 8-K de 04c contenait ainsi une
# catégorie inventée (« aut_materiel »), acceptée telle quelle. Le schéma ne
# se répète PAS dans la consigne, exemple de JSON compris : Google le
# déconseille, la qualité de la réponse baisse. La réponse est revérifiée ici
# (_ecart_au_schema) : une réponse hors format n'est jamais rendue.
#
# Les CLÉS ne s'écrivent jamais dans ce fichier : les constantes *_ENV
# ci-dessous sont les NOMS des variables d'environnement où les lire. Y coller
# une clé revient à chercher une variable qui porterait ce nom -- elle
# n'existe pas, et le LLM est alors jugé indisponible.
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_ENV = "GEMINI_MODEL"
# MODÈLES, DANS L'ORDRE OÙ ILS SONT ESSAYÉS. Au palier gratuit, chaque modèle a
# SON quota quotidien, et c'est lui qui borne un run : relevé en septembre 2026
# (sources tierces, la page officielle variant par compte), environ 20
# requêtes par jour pour les modèles Flash, 500 à 1 500 pour les Flash-Lite.
# gemini-3.8-flash seul, l'ancien défaut, demandait donc des mois pour les
# 8-K récents. Flash-Lite d'abord ; quand le quota du jour d'un modèle est
# épuisé, les analyses passent au suivant (voir « Modèles écartés »), et
# gemini-3.8-flash, le plus fin mais le plus rationné, ferme la marche.
# GEMINI_MODEL remplace la liste : un nom, ou plusieurs séparés par des
# virgules. Les vrais quotas de ta clé s'affichent dans AI Studio.
GEMINI_DEFAULT_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash")
GEMINI_DEFAULT_MODEL = GEMINI_DEFAULT_MODELS[0]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# RÉFLEXION. Les modèles Gemini réfléchissent avant de répondre, et leurs jetons
# de réflexion se décomptent de maxOutputTokens : non réglée, la réflexion peut
# consommer tout le budget, et la réponse revenir vide ou coupée
# (finishReason=MAX_TOKENS) -- un verdict perdu. Depuis la génération 3, elle
# se règle par NIVEAU ; "minimal" suffit pour classer un document court, et
# c'est le niveau qui consomme le moins de jetons. Les modèles 2.x ne
# connaissent que le budget en jetons (coupé à 0 sur les 2.5-flash) et
# refuseraient un niveau.
GEMINI_THINKING_LEVEL_ENV = "GEMINI_THINKING_LEVEL"
GEMINI_THINKING_LEVELS = ("minimal", "low", "medium", "high")
GEMINI_DEFAULT_THINKING_LEVEL = "minimal"
# Marge de jetons laissée à la réflexion, en plus du budget de la réponse. Un
# plafond, pas une dépense : seuls les jetons réellement produits sont facturés.
GEMINI_THINKING_HEADROOM_TOKENS = 8192
# Google demande de retirer `temperature` (et top_p, top_k) des requêtes aux
# modèles 3.8 : aucun de ces réglages n'est envoyé.

# Reprises RÉSEAU (dont les 429 et les 503). Trois ne suffisaient pas : avec le
# backoff ci-dessous elles épuisaient la patience du script en ~15 secondes,
# alors qu'une fenêtre de quota se compte en dizaines de secondes. Le
# 04c_recuperation_8k.py abandonnait donc la classification dès la première
# rafale de 429, sur des milliers de documents.
GEMINI_MAX_RETRIES = 6
# Reprises sur réponse HORS FORMAT -- JSON illisible, ou non conforme au schéma
# -- distinctes des reprises réseau ci-dessus : une seule. Insister sur un
# modèle qui vient de répondre hors format coûte des appels pour un gain
# marginal.
GEMINI_MAX_PARSE_RETRIES = 2
GEMINI_RETRY_DELAY = 2
GEMINI_MAX_RETRY_DELAY = 90.0
# Attente d'une réponse. Large : un modèle qui réfléchit répond en quelques
# secondes, parfois beaucoup plus sous charge, et une attente coupée trop tôt
# renvoie la même requête en entier.
GEMINI_TIMEOUT_S = 120

# Codes HTTP qui justifient un réessai. Tout le reste (400 requête invalide,
# 401/403 clé refusée, 404 modèle inconnu) est DÉFINITIF : réessayer ne fait
# que retarder l'inévitable en brûlant des appels.
GEMINI_RETRYABLE_STATUS = frozenset((408, 409, 425, 429, 500, 502, 503, 504))

# Débit sortant vers Gemini. Le vrai correctif du 429 n'est pas de mieux
# réessayer : c'est de ne pas dépasser le quota. Sans limiteur, 04c enchaînait
# ses appels aussi vite que le réseau le permettait et se faisait jeter dès le
# premier ticker. 0,2 requête/seconde par défaut, soit 12 par minute : sous
# les 15 par minute du palier gratuit des Flash-Lite, et une requête refusée
# est une requête perdue. GEMINI_REQUESTS_PER_SECOND l'ajuste au quota de
# l'offre (plus haut sur une clé facturée).
GEMINI_REQUESTS_PER_SECOND_ENV = "GEMINI_REQUESTS_PER_SECOND"
GEMINI_DEFAULT_REQUESTS_PER_SECOND = 0.2
# Plafond de l'auto-freinage : au-delà, ce n'est plus une rafale à lisser mais
# un quota épuisé, et il vaut mieux échouer visiblement que ramper.
GEMINI_MAX_INTERVAL = 30.0
# Succès consécutifs avant de resserrer l'intervalle élargi par un 429.
GEMINI_SUCCESSES_BEFORE_SPEEDUP = 20

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

    # Les dépôts récents sont du XHTML (iXBRL) ouvert par une déclaration
    # <?xml ...?> : BeautifulSoup avertit alors qu'il lit du XML avec un parser
    # HTML (XMLParsedAsHTMLWarning). C'est voulu : le parser HTML tolère un
    # document coupé par MAX_DOWNLOAD_BYTES ou mal balisé, là où un parser XML
    # s'arrête à la première erreur, et seul le texte est gardé. Filtré ICI
    # seulement, pour ne rien masquer ailleurs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
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


# --------------------------------------------------------------------------- #
# Pièce jointe d'un 8-K : le communiqué de presse (Exhibit 99)
# --------------------------------------------------------------------------- #
# Un 8-K de résultats (Item 2.02) ou d'annonce (7.01, 8.01) tient souvent en
# une phrase -- « the Company issued a press release, attached as Exhibit
# 99.1 » -- et c'est la pièce jointe qui porte l'information : chiffres,
# prévisions, opération. Sans elle, le modèle jugeait une page de couverture.
# Mesuré le 2026-09-27 : sur les 3 240 8-K suivis d'une réaction de cours
# au-delà de 10 %, 2 598 avaient été jugés non matériels, dont 82 % de
# résultats trimestriels.
#
# La page d'index du dépôt liste ses documents AVEC LEUR TYPE (8-K, EX-99.1,
# EX-99.2...) : c'est elle qui dit où est le communiqué, là où les noms de
# fichiers varient d'un déposant à l'autre.
INDEX_DEPOT_URL = ("https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{accession_nodash}/"
                   "{accession_number}-index.htm")
_TYPE_EX99 = re.compile(r"^EX-99(?:\.(\d+))?\b", re.IGNORECASE)
# Une pièce jointe en PDF ou en image n'a pas de texte à lire.
_EXTENSIONS_TEXTE = (".htm", ".html", ".txt")


def pieces_jointes_99(cik: str, accession_number: str) -> List[str]:
    """URLs des pièces jointes EX-99 d'un dépôt, dans l'ordre de leur numéro
    (99.1, 99.2...). Liste vide si la page d'index est introuvable ou n'en
    déclare aucune : le 8-K est alors lu seul, comme avant."""
    url = INDEX_DEPOT_URL.format(
        cik_nolead=str(int(cik)), accession_nodash=accession_number.replace("-", ""),
        accession_number=accession_number)
    try:
        reponse = sec_http.request(url)
    except (sec_http.SecNotFound, sec_http.SecUnavailable, requests.exceptions.RequestException) as e:
        # La pièce jointe est un plus : son absence ne doit jamais coûter le
        # 8-K, ni l'entreprise entière.
        logger.debug("Index du dépôt %s illisible (%s) : 8-K lu sans pièce jointe.", accession_number, e)
        return []
    soup = BeautifulSoup(reponse.content, _best_parser())
    trouvees = []
    for ligne in soup.find_all("tr"):
        cellules = ligne.find_all("td")
        if len(cellules) < 4:
            continue
        type_document = _TYPE_EX99.match(cellules[3].get_text(strip=True))
        lien = cellules[2].find("a", href=True)
        if type_document is None or lien is None:
            continue
        # Un document iXBRL est lié via la visionneuse (« /ix?doc=/Archives/... »).
        chemin = lien["href"].replace("/ix?doc=", "")
        if not chemin.lower().endswith(_EXTENSIONS_TEXTE):
            continue
        trouvees.append((int(type_document.group(1) or 0), urljoin("https://www.sec.gov/", chemin)))
    return [u for _, u in sorted(trouvees)]


def texte_piece_jointe(cik: str, accession_number: str, max_chars: int = MAX_TEXT_CHARS) -> Optional[str]:
    """Texte du communiqué joint à un dépôt -- sa première pièce jointe EX-99
    lisible --, borné à `max_chars`. None s'il n'y en a pas."""
    for url in pieces_jointes_99(cik, accession_number):
        try:
            extrait = fetch_filing_text(url, max_chars=max_chars, form="8-K")
        except requests.exceptions.RequestException as e:
            logger.debug("Pièce jointe %s illisible (%s).", url, e)
            continue
        if extrait is not None and extrait[0].strip():
            return extrait[0]
    return None


def find_filing_asof(cik: str, filed_date: str, forms: tuple = ("10-K", "10-Q")) -> Optional[Dict]:
    """Le filing EXACT déposé à filed_date -- jamais un autre, ni plus
    récent ni plus ancien (voir "Contrainte anti-anticipation" en tête de
    fichier) --, SANS télécharger son document : ses seules métadonnées
    {form, filing_date, accession_number, primary_document}, lues dans l'index
    des dépôts. 07b s'en sert pour reconnaître un filing déjà jugé avant de
    payer son téléchargement.

    filed_date doit correspondre à une vraie date de dépôt (cas normal : c'est
    déjà la filed_date XBRL extraite par 04/04b pour la même période). None si
    aucun filing de ce type n'a été déposé exactement à cette date (CIK
    introuvable, désynchronisation de cache, etc.), ou s'il n'a pas de document
    principal.

    Quand PLUSIEURS filings partagent la même date de dépôt, l'ordre de la
    tuple `forms` départage : elle exprime une préférence (07b passe
    ("10-Q", "10-K") pour une période TTM, ("10-K",) pour un exercice), mais
    le filtre `if form not in forms` ne la respectait pas -- c'était
    `filings[0]`, donc l'ordre du JSON SEC, qui décidait. filter_filings trie
    désormais sur ce critère (cf. D3), et le cas est journalisé en debug."""
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
    return filing


def get_filing_text_asof(cik: str, filed_date: str, forms: tuple = ("10-K", "10-Q"),
                         max_chars: int = MAX_TEXT_CHARS) -> Optional[Dict]:
    """find_filing_asof, puis le texte de son document (au plus max_chars).

    Retourne {form, filing_date, accession_number, primary_document, text,
    extraction_mode}, ou None (filing introuvable, ou téléchargement échoué)."""
    filing = find_filing_asof(cik, filed_date, forms=forms)
    if filing is None:
        return None
    url = filing_document_url(cik, filing["accession_number"], filing["primary_document"])
    extracted = fetch_filing_text(url, max_chars=max_chars, form=filing.get("form"))
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
    l'intervalle n'est pas figé. Le quota Gemini dépend de l'offre du compte
    (palier gratuit ou payant) et du modèle, que le script ne connaît pas --
    le seul moyen de le connaître est de s'y cogner. Chaque 429 DOUBLE donc
    l'intervalle (jusqu'à `max_interval`), et une série de succès le ramène
    progressivement vers sa valeur nominale.

    Sans ça, réessayer après un 429 ne fait que déplacer le problème : la
    requête suivante repart au même rythme et se fait refuser de nouveau."""

    def __init__(self, rate_per_second: float, max_interval: float = GEMINI_MAX_INTERVAL,
                 successes_before_speedup: int = GEMINI_SUCCESSES_BEFORE_SPEEDUP):
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


def _gemini_rate_per_second() -> float:
    brut = os.environ.get(GEMINI_REQUESTS_PER_SECOND_ENV, "").strip()
    if not brut:
        return GEMINI_DEFAULT_REQUESTS_PER_SECOND
    try:
        valeur = float(brut)
    except ValueError:
        logger.warning("%s='%s' illisible, valeur par défaut (%s req/s).",
                       GEMINI_REQUESTS_PER_SECOND_ENV, brut, GEMINI_DEFAULT_REQUESTS_PER_SECOND)
        return GEMINI_DEFAULT_REQUESTS_PER_SECOND
    if valeur <= 0:
        logger.warning("%s doit être > 0 (reçu %s), valeur par défaut (%s req/s).",
                       GEMINI_REQUESTS_PER_SECOND_ENV, valeur, GEMINI_DEFAULT_REQUESTS_PER_SECOND)
        return GEMINI_DEFAULT_REQUESTS_PER_SECOND
    return valeur


# Limiteur GLOBAL au processus, partagé par 04c, 07b et 02 : deux modules qui
# appelleraient Gemini en parallèle avec chacun le sien doubleraient le débit
# réel, donc le taux de 429.
GEMINI_RATE_LIMITER = AdaptiveRateLimiter(_gemini_rate_per_second())


def _retry_after_seconds(response: Optional["requests.Response"]) -> Optional[float]:
    """Valeur de l'en-tête HTTP standard Retry-After, en secondes. Gemini
    annonce plutôt son délai dans le corps (voir _retry_delay_from_body), mais
    une passerelle ou un proxy peut poser l'en-tête. Il admet deux formes : un
    nombre de secondes ou une date HTTP."""
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


def _gemini_retry_delay(response: Optional["requests.Response"], attempt: int) -> float:
    """Le délai annoncé par le serveur s'il y en a un (il sait mieux que nous),
    sinon backoff exponentiel plafonné avec jitter -- le jitter évite que
    plusieurs appelants repartis en même temps ne se resynchronisent sur le
    quota."""
    retry_after = _retry_after_seconds(response)
    if retry_after is None:
        retry_after = _retry_delay_from_body(response)
    if retry_after is not None:
        return min(retry_after + random.uniform(0, 1), GEMINI_MAX_RETRY_DELAY)
    return min(GEMINI_RETRY_DELAY * (2 ** attempt), GEMINI_MAX_RETRY_DELAY) + random.uniform(0, 1)


def llm_disponible() -> bool:
    """Vrai si une clé Gemini est disponible. Une clé vide compte comme absente."""
    return bool(os.environ.get(GEMINI_API_KEY_ENV, "").strip())


def aide_cle_absente() -> str:
    """Quoi vérifier quand aucune clé n'est vue. Le cas fréquent n'est pas une
    clé manquante, mais une clé posée là où ce processus ne la voit pas."""
    import env_local

    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", GEMINI_API_KEY_ENV):
        return ("sec_filings_text.py a été modifié : GEMINI_API_KEY_ENV doit contenir le NOM "
                "d'une variable ('GEMINI_API_KEY'), pas la clé. Restaure le fichier "
                "(git checkout -- sec_filings_text.py) et mets la clé dans .env.")
    return (f"Ajoute la ligne {GEMINI_API_KEY_ENV}=ta_cle au fichier {env_local.FICHIER} (lu par "
            f"tous les scripts, jamais poussé sur git), ou, sous PowerShell, "
            f"$env:{GEMINI_API_KEY_ENV} = \"ta_cle\" dans CE terminal. Après setx, seul un "
            "NOUVEAU terminal voit la variable.")


def description_llm() -> str:
    """Libellé pour les journaux : modèles et réflexion, ou comment activer Gemini."""
    if not llm_disponible():
        return f"aucun LLM ({GEMINI_API_KEY_ENV} à définir)"
    modeles = _modeles_configures()
    # " > " et non une flèche : un journal redirigé vers un fichier sous Windows
    # s'écrit en cp1252, qui n'a pas de flèche.
    liste = " > ".join(modeles)
    if all(_generation_2(m) for m in modeles):
        return f"Gemini ({liste})"
    return f"Gemini ({liste}, réflexion {_niveau_reflexion()})"


def _modeles_configures() -> List[str]:
    """Les modèles de GEMINI_MODEL, dans l'ordre, sinon la liste par défaut.
    Sans le préfixe "models/" que Google écrit dans ses messages et listes :
    recopié tel quel, il donnait une URL invalide."""
    noms = [nom.strip().removeprefix("models/") for nom in os.environ.get(GEMINI_MODEL_ENV, "").split(",")]
    return list(dict.fromkeys(nom for nom in noms if nom)) or list(GEMINI_DEFAULT_MODELS)


def _modeles_restants() -> List[str]:
    """Les modèles de la liste qui n'ont pas été écartés pendant ce run."""
    return [m for m in _modeles_configures() if m not in _modeles_ecartes]


def _gemini_model() -> str:
    """Le modèle à appeler : le premier qui n'a pas été écarté pendant ce run
    (le premier de la liste s'ils l'ont tous été, pour les messages)."""
    restants = _modeles_restants()
    return restants[0] if restants else _modeles_configures()[0]


def dernier_modele_utilise() -> Optional[str]:
    """Le modèle qui a rendu le dernier verdict, à consigner avec lui : quand
    un modèle passe la main au suivant, un même run mêle leurs verdicts."""
    return _dernier_modele


def _generation_2(model: str) -> bool:
    """Modèle 2.x : réflexion réglée par budget de jetons, pas par niveau."""
    return model.startswith("gemini-2.")


def _niveau_reflexion() -> str:
    """Niveau de réflexion demandé par GEMINI_THINKING_LEVEL, "minimal" par défaut."""
    return _niveau_valide(os.environ.get(GEMINI_THINKING_LEVEL_ENV, "").strip().lower())


@functools.lru_cache(maxsize=None)
def _niveau_valide(brut: str) -> str:
    """En cache : une valeur inconnue n'est signalée qu'une fois par run, pas à
    chaque document."""
    if not brut:
        return GEMINI_DEFAULT_THINKING_LEVEL
    if brut not in GEMINI_THINKING_LEVELS:
        logger.warning("%s='%s' inconnu (attendu : %s) : niveau %s.", GEMINI_THINKING_LEVEL_ENV,
                       brut, ", ".join(GEMINI_THINKING_LEVELS), GEMINI_DEFAULT_THINKING_LEVEL)
        return GEMINI_DEFAULT_THINKING_LEVEL
    return brut


def _requete_gemini(
    document: str, consigne: str, schema: dict, max_tokens: int, api_key: str,
    model: Optional[str] = None,
) -> Tuple[str, dict, dict]:
    """(url, en-têtes, corps) d'un appel generateContent au modèle `model` (par
    défaut le modèle courant) : le document dans contents, la consigne dans
    systemInstruction, le schéma dans generationConfig -- voir « Le LLM »."""
    model = model or _gemini_model()
    # Mode JSON sous la contrainte du schéma : la réponse est un JSON conforme,
    # sans bloc de code ni phrase autour.
    generation: dict = {"responseMimeType": "application/json", "responseJsonSchema": schema}
    if not _generation_2(model):
        generation["maxOutputTokens"] = max_tokens + GEMINI_THINKING_HEADROOM_TOKENS
        # Clé et valeur telles que les envoie le SDK officiel : le nom de champ
        # du protocole (l'API l'accepte comme sa forme thinkingLevel) et
        # l'énumération en majuscules.
        generation["thinkingConfig"] = {"thinking_level": _niveau_reflexion().upper()}
    elif model.startswith("gemini-2.5-flash"):
        generation["maxOutputTokens"] = max_tokens
        generation["thinkingConfig"] = {"thinkingBudget": 0}
    else:
        generation["maxOutputTokens"] = max_tokens + GEMINI_THINKING_HEADROOM_TOKENS
    corps: dict = {"contents": [{"role": "user", "parts": [{"text": document}]}]}
    if consigne:
        # Rôle "user" sur l'instruction système, comme le SDK officiel.
        corps["systemInstruction"] = {"role": "user", "parts": [{"text": consigne}]}
    corps["generationConfig"] = generation
    return (
        GEMINI_URL.format(model=model),
        # Clé dans un en-tête plutôt que dans l'URL (?key=...) : une URL finit
        # dans les messages d'erreur de requests, donc dans les journaux.
        {"x-goog-api-key": api_key.strip(), "Content-Type": "application/json"},
        corps,
    )


def _contenu_gemini(data: dict) -> str:
    """Texte de la réponse Gemini. KeyError si la réponse n'en porte pas, ou
    une réponse coupée -- traité par l'appelant comme une enveloppe
    inexploitable. Le message dit laquelle : prompt bloqué (promptFeedback),
    ou finishReason=MAX_TOKENS, une réflexion qui a mangé tout le budget (voir
    GEMINI_THINKING_HEADROOM_TOKENS) -- un JSON coupé ne se relit pas."""
    if not isinstance(data, dict):
        raise TypeError(f"réponse inattendue ({type(data).__name__})")
    candidats = data.get("candidates") or []
    if not candidats:
        bloque = (data.get("promptFeedback") or {}).get("blockReason")
        raise KeyError(f"aucune réponse, prompt bloqué ({bloque})" if bloque else "aucune réponse")
    fin = candidats[0].get("finishReason")
    parts = (candidats[0].get("content") or {}).get("parts") or []
    # Les parties de réflexion (thought) ne sont pas la réponse.
    texte = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
    if fin == "MAX_TOKENS":
        raise KeyError("réponse coupée par la limite de jetons (finishReason=MAX_TOKENS) : "
                       f"baisse {GEMINI_THINKING_LEVEL_ENV} ou relève GEMINI_THINKING_HEADROOM_TOKENS")
    if not texte:
        raise KeyError(f"réponse sans texte (finishReason={fin})")
    return texte


# Types JSON d'un schéma et leur équivalent Python. bool est un int pour
# Python : il est exclu à part des types numériques (voir _ecart_au_schema).
_TYPES_JSON = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "integer": int, "number": (int, float), "null": type(None),
}


def _ecart_au_schema(valeur, schema: dict, chemin: str = "réponse") -> Optional[str]:
    """Premier écart entre une valeur JSON et son schéma, None si conforme.

    Garde-fou, pas validateur complet : il vérifie ce que la génération sous
    contrainte promet et que les appelants exploitent -- type, champs
    obligatoires, propriétés, liste de valeurs (enum), éléments d'un tableau.
    Les autres mots-clés (minimum, format...) ne sont pas contrôlés."""
    if not isinstance(schema, dict):
        return None
    attendu = schema.get("type")
    if isinstance(attendu, str):
        attendu = attendu.lower()          # "STRING" (forme OpenAPI) vaut "string"
        python = _TYPES_JSON.get(attendu)
        if python is not None and (not isinstance(valeur, python)
                                   or (isinstance(valeur, bool) and attendu in ("integer", "number"))):
            return f"{chemin} : {attendu} attendu, reçu {valeur!r:.80}"
    if "enum" in schema and valeur not in schema["enum"]:
        return f"{chemin} : {valeur!r:.80} hors de la liste autorisée"
    if isinstance(valeur, dict):
        for cle in schema.get("required") or ():
            if cle not in valeur:
                return f"{chemin} : champ obligatoire « {cle} » absent"
        for cle, sous_schema in (schema.get("properties") or {}).items():
            if cle in valeur:
                ecart = _ecart_au_schema(valeur[cle], sous_schema, f"{chemin}.{cle}")
                if ecart:
                    return ecart
    if isinstance(valeur, list) and isinstance(schema.get("items"), dict):
        for rang, element in enumerate(valeur):
            ecart = _ecart_au_schema(element, schema["items"], f"{chemin}[{rang}]")
            if ecart:
                return ecart
    return None


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
    # Corps non JSON (page HTML d'un proxy...) : une ligne, pas un pavé.
    return " ".join(str(getattr(response, "text", "") or "").split())[:300]


# --------------------------------------------------------------------------- #
# Disjoncteur du modèle
# --------------------------------------------------------------------------- #
# POURQUOI. Une analyse sans réponse épuise ses GEMINI_MAX_RETRIES tentatives
# avant de rendre None : une bonne minute sur un 503 (attentes de 2, 4, 8, 16
# et 32 s), davantage sur un 429. C'est la bonne patience pour un incident
# passager. Face à une panne qui DURE, elle se paie sur chaque document --
# quota quotidien atteint (palier gratuit de Gemini), modèle « overloaded »
# (503) des heures durant -- et les ~6 300 8-K récents de 04c prenaient des
# jours.
#
# PAUSE. Après LLM_ECHECS_AVANT_PAUSE analyses de suite sans réponse (429, 5xx
# ou réseau jusqu'à la dernière tentative), le modèle est mis en pause :
# analyser_document rend None sans appel, et chaque appelant fait ce qu'il
# fait déjà sans modèle (04c classe par règles, verdicts que le modèle reprend
# au run suivant ; 07b journalise non_evalue ; 02 laisse « indetermine »). La
# première analyse après la pause sert de test : une réponse rouvre tout, un
# nouvel échec relance une pause deux fois plus longue, jusqu'à
# LLM_PAUSE_MAX_S. Toute réponse de Gemini, même un refus propre à un
# document, remet le compte à zéro.
#
# MODÈLES ÉCARTÉS. Un refus qui ne vise qu'UN modèle l'écarte pour le reste du
# run, et le suivant de la liste (GEMINI_DEFAULT_MODELS, ou GEMINI_MODEL)
# reprend la même requête, sans attendre : son quota du jour épuisé (un 429
# dont le quota est « PerDay » -- attendre des heures n'y changerait rien),
# modèle inconnu ou retiré pour cette clé (404), ou réglage qu'il refuse (400
# sur la réflexion, le schéma, un champ de la requête). Chaque modèle ayant son
# propre quota, la liste multiplie ce qu'un run peut faire en une journée.
#
# COUPURE. Un refus qui vise la CLÉ (invalide, accès refusé, paiement requis,
# pays non couvert) coupe Gemini pour tout le run, dès le premier ; de même
# quand le dernier modèle de la liste est écarté. La réponse serait la même
# pour chaque document, et l'ancien comportement -- un appel et une ligne
# d'erreur par document -- noyait la cause sous des milliers de lignes
# identiques.
LLM_ECHECS_AVANT_PAUSE = 3
LLM_PAUSE_INITIALE_S = 15 * 60.0
LLM_PAUSE_MAX_S = 2 * 3600.0
# 401 clé invalide, 402 paiement requis, 403 accès refusé. Un 400 ne vise la
# clé que s'il le dit (clé invalide, pays non couvert, facturation).
LLM_STATUTS_CLE = frozenset((401, 402, 403))
_MOTIFS_400_CLE = ("api key", "api_key", "location is not supported", "billing")
# 404 modèle inconnu ou retiré. Un 400 ne vise le modèle que s'il le dit
# (réglage inconnu du modèle) ; sinon il vise la requête, donc ce document-là.
LLM_STATUTS_MODELE = frozenset((404,))
_MOTIFS_400_MODELE = ("thinking", "schema", "unknown name", "invalid json payload")
# Sauf « The specified schema produces a constraint that has too many states
# for serving » : le schéma de CETTE requête est trop lourd -- un lot trop grand,
# que analyser_documents recoupe --, pas le modèle. Lu comme un refus du
# modèle, il les écartait l'un après l'autre et coupait Gemini pour le run.
_MOTIFS_400_REQUETE = ("too many states",)
# « ... Please update your code to use models/gemini-3.8-flash ... »
_MODELE_PROPOSE = re.compile(r"\buse\s+(?:models/)?(gemini-[\w.\-]+)", re.IGNORECASE)

_horloge = time.monotonic          # remplaçable par les tests
_echecs_consecutifs = 0
_pause_jusqu_a: Optional[float] = None
_duree_pause = LLM_PAUSE_INITIALE_S
_coupure: Optional[str] = None
# Modèle -> raison de son écart, pour le reste du run.
_modeles_ecartes: Dict[str, str] = {}
_dernier_modele: Optional[str] = None
# Vrai quand la dernière analyse a échoué à cause de ce qu'elle soumettait --
# requête refusée pour son contenu, réponse bloquée ou hors format --, et non
# faute de réponse, de quota ou de clé (voir analyser_documents).
_echec_du_contenu = False
# Ce que le modèle a réellement fait pendant le run (voir bilan_llm).
_bilan: Dict[str, int] = {}


def reinitialiser_disjoncteur_llm() -> None:
    """Réarme le disjoncteur, rend leur chance aux modèles écartés et vide le
    bilan (tests ; un run de production est un processus)."""
    global _echecs_consecutifs, _pause_jusqu_a, _duree_pause, _coupure, _dernier_modele, _echec_du_contenu
    _echecs_consecutifs = 0
    _pause_jusqu_a = None
    _duree_pause = LLM_PAUSE_INITIALE_S
    _coupure = None
    _modeles_ecartes.clear()
    _dernier_modele = None
    _echec_du_contenu = False
    _bilan.clear()


def llm_coupe_pour_ce_run() -> bool:
    """Vrai après un refus de configuration : plus aucun appel de tout le run."""
    return _coupure is not None


def llm_en_pause() -> bool:
    """Vrai pendant une pause (le modèle ne répondait plus). Faux dès qu'elle
    est écoulée : l'analyse suivante teste alors le modèle."""
    return _pause_jusqu_a is not None and _horloge() < _pause_jusqu_a


def bilan_llm() -> str:
    """Une ligne pour la fin d'un run : ce que le modèle a réellement fait."""
    morceaux = [f"{_bilan.get('verdicts', 0)} verdict(s)"]
    if _bilan.get("inexploitables"):
        morceaux.append(f"{_bilan['inexploitables']} réponse(s) inexploitable(s) ou refusée(s)")
    if _bilan.get("sans_reponse"):
        morceaux.append(f"{_bilan['sans_reponse']} analyse(s) sans réponse malgré les réessais")
    if _bilan.get("ecartes"):
        pourquoi = f"modèle coupé, {_coupure}" if _coupure else f"{_bilan.get('pauses', 0)} pause(s)"
        morceaux.append(f"{_bilan['ecartes']} document(s) traité(s) sans le modèle ({pourquoi})")
    if _bilan.get("requetes"):
        morceaux.append(f"{_bilan['requetes']} requête(s) envoyée(s)")
    if _modeles_ecartes:
        morceaux.append("écarté(s) en cours de run : " + ", ".join(
            f"{modele} ({raison})" for modele, raison in _modeles_ecartes.items()))
    return f"{description_llm()} -- " + ", ".join(morceaux) + "."


def documents_sans_modele() -> int:
    """Documents rendus sans verdict faute de réponse du modèle (réessais
    épuisés, pause, coupure) pendant ce run."""
    return _bilan.get("sans_reponse", 0) + _bilan.get("ecartes", 0)


def _compter(evenement: str, n: int = 1) -> None:
    _bilan[evenement] = _bilan.get(evenement, 0) + n


def _message(erreur: Exception) -> str:
    """str(KeyError("x")) vaut "'x'" : le message sans ses guillemets."""
    if isinstance(erreur, KeyError) and erreur.args:
        return str(erreur.args[0])
    return str(erreur)


def _duree_lisible(secondes: float) -> str:
    minutes = round(secondes / 60)
    if minutes < 60:
        return f"{minutes} min"
    heures, reste = divmod(minutes, 60)
    return f"{heures} h" if not reste else f"{heures} h {reste:02d}"


def _raison_echec(erreur: Optional[Exception]) -> str:
    """Pourquoi une tentative n'a pas abouti, avec les mots du fournisseur :
    « HTTP 503 : The model is overloaded... » dit que la panne est chez lui,
    là où « HTTP 503 » seul se lisait comme une panne du code."""
    if erreur is None:
        return "raison inconnue"
    reponse = getattr(erreur, "response", None)
    statut = getattr(reponse, "status_code", None)
    if statut is None:
        return f"{type(erreur).__name__} : {str(erreur)[:200]}"
    message = _extrait_erreur(reponse)
    return f"HTTP {statut} : {message}" if message else f"HTTP {statut}"


def _refus_de_la_cle(statut: int, message: str) -> bool:
    """Le refus vise-t-il la clé (même réponse pour tout modèle et tout document) ?"""
    if statut in LLM_STATUTS_CLE:
        return True
    return statut == 400 and any(motif in message.lower() for motif in _MOTIFS_400_CLE)


def _refus_du_modele(statut: int, message: str) -> bool:
    """Le refus vise-t-il le modèle appelé (un autre modèle pourrait répondre) ?"""
    if statut in LLM_STATUTS_MODELE:
        return True
    message = message.lower()
    return (statut == 400 and any(motif in message for motif in _MOTIFS_400_MODELE)
            and not any(motif in message for motif in _MOTIFS_400_REQUETE))


def _refus_de_configuration(statut: int, message: str) -> bool:
    """Le refus vise-t-il la configuration -- clé ou modèle -- plutôt que le
    document ? La même réponse reviendrait pour chaque document."""
    return _refus_de_la_cle(statut, message) or _refus_du_modele(statut, message)


def _quota_du_jour(reponse: Optional["requests.Response"]) -> bool:
    """Ce 429 vient-il du quota QUOTIDIEN, épuisé jusqu'au lendemain, plutôt
    que du quota par minute ? Gemini le dit dans le corps, par l'identifiant du
    quota dépassé (« GenerateRequestsPerDayPerProjectPerModel-FreeTier »)."""
    if reponse is None:
        return False
    try:
        corps = reponse.json()
    except (ValueError, AttributeError):
        return False
    erreur = corps.get("error") if isinstance(corps, dict) else None
    if not isinstance(erreur, dict):
        return False
    for detail in erreur.get("details") or []:
        for violation in (detail.get("violations") or []) if isinstance(detail, dict) else []:
            if isinstance(violation, dict) and "PerDay" in str(violation.get("quotaId", "")):
                return True
    return bool(re.search(r"per\s*day|daily", str(erreur.get("message", "")), re.IGNORECASE))


def _ecarter_modele(modele: str, raison: str) -> bool:
    """Écarte `modele` pour le reste du run. Vrai s'il reste un modèle pour
    reprendre la requête, faux si c'était le dernier."""
    _modeles_ecartes[modele] = raison
    restants = _modeles_restants()
    if restants:
        logger.warning("Gemini : %s écarté pour ce run (%s) -- %s prend le relais.",
                       modele, raison, restants[0])
        return True
    return False


def _aide_configuration(statut: int, message: str, modele: Optional[str] = None) -> str:
    """Quoi changer. Sur un modèle retiré, Google nomme son successeur : on le
    reprend tel quel."""
    modele = modele or _gemini_model()
    if statut == 404:
        propose = _MODELE_PROPOSE.search(message)
        if propose:
            successeur = propose.group(1).rstrip(".")
            return f"Google propose {successeur} : mets la ligne {GEMINI_MODEL_ENV}={successeur} dans .env."
        return (f"Le modèle {modele} n'existe pas ou n'est pas ouvert à cette clé : choisis-en "
                f"un autre avec {GEMINI_MODEL_ENV} dans .env (python diagnostic_llm.py --modeles "
                "liste ceux de ta clé).")
    if statut == 400 and "thinking" in message.lower():
        return (f"Le modèle {modele} ne règle pas sa réflexion par niveau "
                f"({GEMINI_THINKING_LEVEL_ENV}) : choisis un modèle Gemini 3 ou plus récent avec "
                f"{GEMINI_MODEL_ENV} dans .env.")
    return "python diagnostic_llm.py teste la clé et chaque modèle en quelques secondes."


def _le_modele_repond() -> None:
    """Gemini a répondu : compte remis à zéro, pause levée."""
    global _echecs_consecutifs, _pause_jusqu_a, _duree_pause
    if _pause_jusqu_a is not None:
        logger.info("Gemini répond de nouveau : fin de la pause, le modèle reprend la main.")
    _echecs_consecutifs = 0
    _pause_jusqu_a = None
    _duree_pause = LLM_PAUSE_INITIALE_S


def _sans_reponse(derniere_erreur: Optional[Exception], test_de_reprise: bool,
                  nb_documents: int = 1) -> None:
    """Une analyse vient d'épuiser ses réessais : pause au troisième échec de
    suite, ou dès le premier si c'était l'essai qui suit une pause."""
    global _echecs_consecutifs, _pause_jusqu_a, _duree_pause
    _compter("sans_reponse", nb_documents)
    _echecs_consecutifs += 1
    raison = _raison_echec(derniere_erreur)
    if not test_de_reprise and _echecs_consecutifs < LLM_ECHECS_AVANT_PAUSE:
        logger.warning("Gemini sans réponse après %d tentatives (%s) : ce document est traité sans le modèle.",
                       GEMINI_MAX_RETRIES, raison)
        return

    duree = _duree_pause
    _pause_jusqu_a = _horloge() + duree
    _duree_pause = min(duree * 2, LLM_PAUSE_MAX_S)
    _compter("pauses")
    sans_lui = ("entre-temps, les documents sont traités sans lui (04c les classe par règles, et le "
                "modèle les reprendra au prochain run)")
    statut = getattr(getattr(derniere_erreur, "response", None), "status_code", None)
    if test_de_reprise:
        logger.warning("Toujours aucune réponse de Gemini (%s) : nouvelle pause de %s, %s.",
                       raison, _duree_lisible(duree), sans_lui)
    elif statut == 429:
        logger.error(
            "Quota Gemini épuisé : %d analyses de suite refusées (429) malgré %d tentatives chacune (%s). "
            "Modèle en pause %s puis réessayé ; %s. Un quota quotidien ne revient que le lendemain ; "
            "sur une offre payante, relève %s.",
            _echecs_consecutifs, GEMINI_MAX_RETRIES, raison, _duree_lisible(duree), sans_lui,
            GEMINI_REQUESTS_PER_SECOND_ENV)
    else:
        logger.error(
            "Gemini ne répond plus : %d analyses de suite sans réponse malgré %d tentatives chacune "
            "(dernière erreur : %s). La panne est chez Google ou sur le réseau, pas dans le code. "
            "Modèle en pause %s puis réessayé ; %s. Si ça dure, essaie un autre modèle : %s=... dans "
            ".env (python diagnostic_llm.py --modeles liste ceux de ta clé).",
            _echecs_consecutifs, GEMINI_MAX_RETRIES, raison, _duree_lisible(duree), sans_lui,
            GEMINI_MODEL_ENV)


def analyser_document(document: str, consigne: str, schema: dict, max_tokens: int = 500) -> Optional[dict]:
    """Soumet à Gemini un DOCUMENT, une CONSIGNE et le FORMAT de la réponse --
    un schéma JSON --, et rend la réponse : un dict conforme au schéma.

    Les trois partent séparément (voir « Le LLM ») : Gemini génère sous la
    contrainte du schéma, et la réponse est revérifiée ici -- une réponse hors
    format est redemandée une fois, puis abandonnée, jamais rendue.
    `max_tokens` est le budget de la RÉPONSE ; la marge de réflexion s'y ajoute.

    Réessais avec backoff sur les incidents de réseau et de quota ; les appels
    passent par GEMINI_RATE_LIMITER (voir AdaptiveRateLimiter) : espacés en
    amont pour ne pas provoquer de 429, et DAVANTAGE dès qu'un 429 survient
    malgré tout. Un modèle écarté passe la main au suivant de la liste (voir
    « Modèles écartés »).

    None si aucune clé n'est définie, après épuisement des tentatives, ou
    pendant une pause du disjoncteur (voir « Disjoncteur du modèle ») :
    l'appelant doit traiter ce cas comme « pas de verdict », jamais planter."""
    return _analyser(document, consigne, schema, max_tokens, nb_documents=1)


def analyser_documents(documents: Dict[str, str], consigne: str, schema_element: dict,
                       max_tokens_par_document: int = 150) -> Optional[Dict[str, dict]]:
    """Plusieurs documents en UNE requête, et leurs réponses par identifiant.

    L'économie qui compte au palier gratuit : c'est le nombre de REQUÊTES par
    jour qui borne un run, pas leur taille. La consigne et le format partent
    une fois pour tout le lot ; chaque document part entre balises
    <document id="...">, et la réponse attendue est un objet avec un champ
    OBLIGATOIRE par identifiant, conforme à schema_element -- aucun document
    ne peut être oublié. Des identifiants courts (d1, d2...) coûtent moins de
    jetons qu'un numéro d'accession.

    La consigne doit demander de juger chaque document séparément.

    UN DOCUMENT NE FAIT PAS TOMBER SON LOT. Une requête qui échoue à cause de
    ce qu'elle soumet -- refusée pour son contenu, réponse bloquée ou hors
    format -- peut ne tenir qu'à un document, et le même lot échouerait de
    nouveau, à l'identique, à chaque run. Il est alors coupé en deux, chaque
    moitié soumise à part ; une moitié en échec face à une moitié qui répond
    est recoupée à son tour, jusqu'à isoler le document en cause -- quelques
    requêtes de plus, en cas d'échec seulement. Quand les DEUX moitiés
    échouent, la cause n'est pas un document isolé : on s'arrête là, plutôt
    que de finir document par document et de multiplier les requêtes.

    Rend les réponses obtenues par identifiant -- un document sans verdict en
    est absent --, {} pour un lot vide, None quand aucun document n'a de
    verdict (voir analyser_document)."""
    if not documents:
        return {}
    obtenues = _requete_de_lot(documents, consigne, schema_element, max_tokens_par_document)
    if obtenues is not None or not _echec_du_contenu:
        return obtenues
    if len(documents) == 1:
        _compter("inexploitables", 1)
        return None

    obtenues = {}
    en_cause = dict(documents)
    while len(en_cause) > 1:
        logger.warning("Gemini : %d documents sans réponse exploitable ensemble -- coupés en deux pour "
                       "isoler celui qui fait échouer la requête.", len(en_cause))
        idents = list(en_cause)
        moities = [{ident: en_cause[ident] for ident in part}
                   for part in (idents[:len(idents) // 2], idents[len(idents) // 2:])]
        en_echec, repondu = [], False
        for moitie in moities:
            reponses = _requete_de_lot(moitie, consigne, schema_element, max_tokens_par_document)
            if reponses is not None:
                obtenues.update(reponses)
                repondu = True
            elif _echec_du_contenu:
                en_echec.append(moitie)
        if not (repondu and len(en_echec) == 1):
            # Deux moitiés en échec : la cause n'est pas un document isolé. Une
            # moitié sans réponse (quota, panne) : rien à isoler -- _analyser
            # l'a comptée.
            _compter("inexploitables", sum(len(moitie) for moitie in en_echec))
            return obtenues or None
        en_cause = en_echec[0]
    logger.warning("Gemini : document %s isolé -- la requête échoue à cause de lui ; les autres du lot "
                   "ont leur verdict.", next(iter(en_cause)))
    _compter("inexploitables", len(en_cause))
    return obtenues or None


def _requete_de_lot(documents: Dict[str, str], consigne: str, schema_element: dict,
                    max_tokens_par_document: int) -> Optional[Dict[str, dict]]:
    """UNE requête pour tout le lot (voir analyser_documents). Un échec dû au
    contenu n'est pas compté ici : le lot sera peut-être recoupé."""
    texte = "\n\n".join(f'<document id="{ident}">\n{contenu}\n</document>'
                         for ident, contenu in documents.items())
    schema = {
        "type": "object",
        "properties": {ident: schema_element for ident in documents},
        "required": list(documents),
        "propertyOrdering": list(documents),
    }
    reponse = _analyser(texte, consigne, schema, max_tokens_par_document * len(documents),
                        nb_documents=len(documents), compter_inexploitables=False)
    if reponse is None:
        return None
    return {ident: reponse[ident] for ident in documents}


def _analyser(document: str, consigne: str, schema: dict, max_tokens: int,
              nb_documents: int, compter_inexploitables: bool = True) -> Optional[dict]:
    """L'appel lui-même (voir analyser_document). `nb_documents` : combien de
    documents la requête porte, pour que le bilan compte des documents.
    `compter_inexploitables` : faux quand l'appelant compte lui-même un échec
    dû au contenu (voir _echec_du_contenu)."""
    global _coupure, _dernier_modele, _echec_du_contenu
    _echec_du_contenu = False
    if not llm_disponible():
        return None
    if not document or not document.strip():
        # Gemini refuserait un contenu vide (400) : un appel payé pour rien.
        logger.warning("Document vide : rien à soumettre à Gemini.")
        return None
    if _coupure is not None or llm_en_pause():
        _compter("ecartes", nb_documents)
        return None
    cle = os.environ[GEMINI_API_KEY_ENV]
    test_de_reprise = _pause_jusqu_a is not None     # pause écoulée : cette analyse la teste
    if test_de_reprise:
        logger.info("Fin de la pause : nouvel essai de Gemini.")

    parse_failures = 0
    network_failures = 0
    derniere_erreur: Optional[Exception] = None

    while network_failures < GEMINI_MAX_RETRIES:
        # Recalculée à chaque tentative : un modèle écarté entre-temps cède
        # la place au suivant, avec les réglages qui lui conviennent.
        modele = _gemini_model()
        url, headers, payload = _requete_gemini(document, consigne, schema, max_tokens, cle, modele)
        GEMINI_RATE_LIMITER.acquire()
        _compter("requetes")
        resp = None
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT_S)
            statut = getattr(resp, "status_code", None)
            if statut is not None and statut >= 400:
                raise requests.exceptions.HTTPError(f"HTTP {statut}", response=resp)
            if network_failures:
                # Referme la série « tentative k/6 sans réponse » : sans cette
                # ligne, rien ne disait que le document avait fini par passer,
                # et une surcharge absorbée se lisait comme un échec.
                logger.info("Gemini a répondu à la tentative %d/%d.", network_failures + 1, GEMINI_MAX_RETRIES)
            _le_modele_repond()
            content = _contenu_gemini(resp.json())
        except requests.exceptions.RequestException as e:
            reponse = getattr(e, "response", None)
            statut = getattr(reponse, "status_code", None)
            if statut == 429 and _quota_du_jour(reponse):
                # Attendre ne servirait à rien avant le lendemain : au suivant.
                if _ecarter_modele(modele, "quota du jour épuisé"):
                    continue
                _coupure = "quotas du jour épuisés"
                _compter("ecartes", nb_documents)
                logger.error(
                    "Gemini : quota du jour épuisé pour chaque modèle de la liste (%s) -- plus aucun "
                    "appel jusqu'à la fin de ce run. Les documents restants sont traités sans lui, et "
                    "le modèle les reprendra au prochain run : les quotas quotidiens reviennent le "
                    "lendemain. Une clé facturée (Cloud Billing) les relève.",
                    ", ".join(_modeles_configures()))
                return None
            if statut is not None and statut not in GEMINI_RETRYABLE_STATUS:
                message = _extrait_erreur(reponse)
                if _refus_de_la_cle(statut, message):
                    _coupure = f"HTTP {statut}"
                    _compter("ecartes", nb_documents)
                    logger.error(
                        "Gemini refuse la configuration (HTTP %s) : %s -- plus aucun appel au modèle "
                        "jusqu'à la fin de ce run, les documents restants sont traités sans lui. %s",
                        statut, message or e, _aide_configuration(statut, message, modele))
                    return None
                if _refus_du_modele(statut, message):
                    logger.error("Gemini refuse le modèle %s (HTTP %s) : %s -- %s",
                                 modele, statut, message or e, _aide_configuration(statut, message, modele))
                    if _ecarter_modele(modele, f"HTTP {statut}"):
                        continue
                    _coupure = f"HTTP {statut}"
                    _compter("ecartes", nb_documents)
                    logger.error("Plus aucun modèle de la liste n'est utilisable : plus aucun appel "
                                 "jusqu'à la fin de ce run, les documents restants sont traités sans lui.")
                    return None
                # 400 propre à CE document : la réponse ne changera pas.
                logger.error("Appel Gemini refusé définitivement (HTTP %s), aucun réessai : %s",
                             statut, message or e)
                _le_modele_repond()
                _echec_du_contenu = True
                if compter_inexploitables:
                    _compter("inexploitables", nb_documents)
                return None

            network_failures += 1
            derniere_erreur = e
            if network_failures >= GEMINI_MAX_RETRIES:
                break

            delay = _gemini_retry_delay(resp, network_failures - 1)
            if statut == 429:
                intervalle = GEMINI_RATE_LIMITER.penalize(pause=delay)
                logger.warning(
                    "Quota Gemini atteint (429, tentative %d/%d). Débit ramené à un appel "
                    "toutes les %.1fs ; nouvel essai dans %.1fs...",
                    network_failures, GEMINI_MAX_RETRIES, intervalle, delay,
                )
            else:
                # Surcharge (503), erreur serveur ou coupure réseau : un
                # incident chez Google, que les réessais absorbent le plus
                # souvent. En INFO et avec ses mots : l'ancien « Tentative
                # échouée: HTTP 503 » en WARNING se lisait comme une panne du code.
                logger.info("Gemini : tentative %d/%d sans réponse (%s). Nouvel essai dans %.1fs.",
                            network_failures, GEMINI_MAX_RETRIES, _raison_echec(e), delay)
            time.sleep(delay)
            continue
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            logger.error("Réponse Gemini inexploitable : %s", _message(e))
            _echec_du_contenu = True
            if compter_inexploitables:
                _compter("inexploitables", nb_documents)
            return None

        GEMINI_RATE_LIMITER.reward()
        parsed = _parse_json_reponse(content)
        ecart = "JSON illisible" if parsed is None else _ecart_au_schema(parsed, schema)
        if ecart is None:
            _compter("verdicts", nb_documents)
            _dernier_modele = modele
            return parsed

        # Réponse hors format : UNE seule reprise. Insister davantage sur un
        # modèle qui vient de répondre hors format coûte des appels pour un
        # gain marginal.
        parse_failures += 1
        if parse_failures >= GEMINI_MAX_PARSE_RETRIES:
            logger.warning("Réponse Gemini hors format après %d essais (%s) : %s",
                           parse_failures, ecart, str(content)[:200])
            _echec_du_contenu = True
            if compter_inexploitables:
                _compter("inexploitables", nb_documents)
            return None
        logger.warning("Réponse Gemini hors format (%s), une nouvelle tentative : %s",
                       ecart, str(content)[:200])

    _sans_reponse(derniere_erreur, test_de_reprise, nb_documents)
    return None


class EssaiLLM(NamedTuple):
    """Résultat d'essai_unique_llm."""
    statut: Optional[int]      # code HTTP, None sans réponse du tout
    ok: bool                   # une réponse conforme au schéma
    texte: str                 # la réponse, ou la raison de l'échec
    duree_s: float


def essai_unique_llm(document: str, consigne: str, schema: dict, max_tokens: int = 64,
                     modele: Optional[str] = None) -> EssaiLLM:
    """UNE requête à Gemini -- la même que celle d'analyser_document, sans
    réessai, limiteur ni disjoncteur -- pour diagnostic_llm.py, qui doit
    montrer la réponse brute. `modele` : celui à essayer (défaut : le premier
    de la liste). Une réponse qui ne respecte pas le schéma compte comme un
    échec : aucun script ne s'en servirait."""
    if not llm_disponible():
        return EssaiLLM(None, False, "aucune clé", 0.0)
    url, headers, payload = _requete_gemini(
        document, consigne, schema, max_tokens, os.environ[GEMINI_API_KEY_ENV], modele)
    debut = time.monotonic()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT_S)
    except requests.exceptions.RequestException as e:
        return EssaiLLM(None, False, _raison_echec(e), time.monotonic() - debut)
    duree = time.monotonic() - debut
    if resp.status_code >= 400:
        return EssaiLLM(resp.status_code, False, _extrait_erreur(resp), duree)
    try:
        contenu = _contenu_gemini(resp.json())
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
        return EssaiLLM(resp.status_code, False, f"réponse inexploitable : {_message(e)}", duree)
    parsed = _parse_json_reponse(contenu)
    ecart = "JSON illisible" if parsed is None else _ecart_au_schema(parsed, schema)
    if ecart:
        return EssaiLLM(resp.status_code, False, f"réponse hors format ({ecart}) : {contenu}", duree)
    return EssaiLLM(resp.status_code, True, contenu, duree)
