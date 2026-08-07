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

Limite connue : seule filings.recent est consultée (les ~1000 dépôts les plus
récents par CIK, largement suffisant pour du 10-K/10-Q/8-K sur quelques
années) -- l'historique très ancien au-delà (filings.files, paginé côté SEC)
n'est pas couvert par ce module.

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

import requests
from bs4 import BeautifulSoup

import sec_http

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{accession_nodash}/{document}"
REQUEST_DELAY = 0.15

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


def list_company_filings(
    cik: str, forms: tuple = ("10-K", "10-Q"),
    start_date: Optional[str] = None, end_date: Optional[str] = None,
) -> List[Dict]:
    """Filings d'une entreprise (CIK, 10 chiffres) filtrés par formulaire et
    fenêtre de dates de DÉPÔT (filingDate SEC, format YYYY-MM-DD, bornes
    incluses). Une entrée : {form, filing_date, accession_number,
    primary_document}. Liste vide si le CIK est introuvable ou n'a aucun
    filing correspondant -- pas une erreur bloquante."""
    data = _get_json(SUBMISSIONS_URL.format(cik=cik))
    if not data:
        return []
    recent = data.get("filings", {}).get("recent", {})
    forms_list = recent.get("form", [])
    dates_list = recent.get("filingDate", [])
    accessions_list = recent.get("accessionNumber", [])
    docs_list = recent.get("primaryDocument", [])

    results = []
    for form, filing_date, accession, doc in zip(forms_list, dates_list, accessions_list, docs_list):
        if form not in forms:
            continue
        if start_date and filing_date < start_date:
            continue
        if end_date and filing_date > end_date:
            continue
        results.append({
            "form": form, "filing_date": filing_date,
            "accession_number": accession, "primary_document": doc,
        })
    return results


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
