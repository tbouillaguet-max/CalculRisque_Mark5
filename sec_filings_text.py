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
import time
from typing import Dict, List, Optional

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
# classification -- tronqué au début du document (page de garde + souvent le
# début des risk factors / MD&A pour un 10-K, l'objet de l'annonce pour un
# 8-K), pragmatique plutôt qu'une extraction sémantique de section.
MAX_TEXT_CHARS = 15_000

MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-large-latest"
MISTRAL_MAX_RETRIES = 3
MISTRAL_RETRY_DELAY = 2
MISTRAL_TEMPERATURE = 0.1

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
    filings = sorted(unique.values(), key=lambda f: f.get("filing_date") or "")

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


def fetch_filing_text(url: str, max_chars: int = MAX_TEXT_CHARS) -> Optional[str]:
    """Télécharge un document de filing (HTML dans la quasi-totalité des cas
    depuis le début des années 2000) et en extrait le texte brut, tronqué à
    max_chars. None si le téléchargement échoue."""
    try:
        resp = sec_http.request(url)
    except (sec_http.SecNotFound, sec_http.SecUnavailable) as e:
        logger.error("Échec du téléchargement du filing %s: %s", url, e)
        return None

    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return text[:max_chars]


def get_filing_text_asof(cik: str, filed_date: str, forms: tuple = ("10-K", "10-Q")) -> Optional[Dict]:
    """Le filing EXACT déposé à filed_date -- jamais un autre, ni plus
    récent ni plus ancien (voir "Contrainte anti-anticipation" en tête de
    fichier). filed_date doit correspondre à une vraie date de dépôt (cas
    normal : c'est déjà la filed_date XBRL extraite par 04/04b pour la même
    période). None si aucun filing de ce type n'a été déposé exactement à
    cette date (CIK introuvable, désynchronisation de cache, etc.).

    Retourne {form, filing_date, accession_number, primary_document, text}."""
    filings = list_company_filings(cik, forms=forms, start_date=filed_date, end_date=filed_date)
    if not filings:
        return None
    filing = filings[0]
    url = filing_document_url(cik, filing["accession_number"], filing["primary_document"])
    text = fetch_filing_text(url)
    if text is None:
        return None
    return {**filing, "text": text}


def analyser_texte_mistral(prompt: str, max_tokens: int = 500) -> Optional[dict]:
    """Appelle Mistral avec un prompt demandant une réponse JSON stricte --
    même pattern que 02_categoriser_secteurs.py::appeler_mistral (retries
    avec backoff exponentiel, la réponse DOIT être un objet JSON valide).
    Généraliste (pas de schéma imposé ici) : chaque appelant (07b, 04c)
    construit son propre prompt et valide les clés qu'il attend dans le dict
    retourné. None si MISTRAL_API_KEY est absente, ou après épuisement des
    tentatives -- l'appelant doit traiter ce cas comme "pas de verdict",
    jamais planter."""
    api_key = os.environ.get(MISTRAL_API_KEY_ENV)
    if not api_key:
        return None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": MISTRAL_TEMPERATURE,
        "max_tokens": max_tokens,
    }

    for attempt in range(MISTRAL_MAX_RETRIES):
        try:
            resp = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=45)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if not content.startswith("{") or not content.endswith("}"):
                logger.warning("Réponse Mistral non-JSON : %s", content[:200])
                return None
            return json.loads(content)
        except requests.exceptions.RequestException as e:
            delay = MISTRAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
            logger.warning("Tentative Mistral %d échouée: %s. Nouvel essai dans %.1fs...", attempt + 1, e, delay)
            time.sleep(delay)
        except (KeyError, json.JSONDecodeError) as e:
            logger.error("Erreur de parsing de la réponse Mistral: %s", e)
            return None

    logger.error("Échec après %d tentatives Mistral.", MISTRAL_MAX_RETRIES)
    return None
