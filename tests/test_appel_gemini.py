"""Appel du LLM par Gemini, et choix du fournisseur (sec_filings_text)."""

from __future__ import annotations

import pytest
import requests

import sec_filings_text as sft


class _ReponseGemini:
    """Réponse de l'API Gemini : texte dans candidates[0].content.parts, ou
    corps d'erreur {"error": {...}} quand status_code >= 400."""

    def __init__(self, texte="", status_code=200, corps=None):
        self.status_code = status_code
        self.headers = {}
        self._corps = corps if corps is not None else {
            "candidates": [{"content": {"role": "model", "parts": [{"text": texte}]}}],
        }
        self.text = str(self._corps)

    def json(self):
        return self._corps


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setenv(sft.GEMINI_API_KEY_ENV, "cle-gemini-de-test")
    monkeypatch.setattr(sft.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sft, "MISTRAL_RATE_LIMITER", sft.AdaptiveRateLimiter(1000.0))

    appels = []

    def poser(reponses):
        def faux_post(url, headers=None, json=None, timeout=None):
            appels.append({"url": url, "headers": headers, "json": json})
            item = reponses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item if isinstance(item, _ReponseGemini) else _ReponseGemini(item)
        monkeypatch.setattr(sft.requests, "post", faux_post)
        return appels

    return poser


# --------------------------------------------------------------------------- #
# Choix du fournisseur
# --------------------------------------------------------------------------- #

def test_sans_aucune_cle_pas_de_fournisseur():
    assert sft.fournisseur_llm() is None
    assert not sft.llm_disponible()


def test_mistral_seul(monkeypatch):
    monkeypatch.setenv(sft.MISTRAL_API_KEY_ENV, "m")
    assert sft.fournisseur_llm() == "mistral"


def test_gemini_prioritaire_quand_les_deux_cles_existent(monkeypatch):
    monkeypatch.setenv(sft.MISTRAL_API_KEY_ENV, "m")
    monkeypatch.setenv(sft.GEMINI_API_KEY_ENV, "g")
    assert sft.fournisseur_llm() == "gemini"


def test_llm_provider_force_le_choix(monkeypatch):
    monkeypatch.setenv(sft.MISTRAL_API_KEY_ENV, "m")
    monkeypatch.setenv(sft.GEMINI_API_KEY_ENV, "g")
    monkeypatch.setenv(sft.LLM_PROVIDER_ENV, "mistral")
    assert sft.fournisseur_llm() == "mistral"


def test_forcer_un_fournisseur_sans_sa_cle_ne_bascule_pas_sur_l_autre(monkeypatch):
    """Forcer gemini sans clé Gemini doit se voir, pas partir en silence chez
    Mistral (qui facture, et peut refuser)."""
    monkeypatch.setenv(sft.MISTRAL_API_KEY_ENV, "m")
    monkeypatch.setenv(sft.LLM_PROVIDER_ENV, "gemini")
    assert sft.fournisseur_llm() is None


def test_une_cle_vide_compte_comme_absente(monkeypatch):
    monkeypatch.setenv(sft.GEMINI_API_KEY_ENV, "")
    assert sft.fournisseur_llm() is None


# --------------------------------------------------------------------------- #
# Requête et réponse Gemini
# --------------------------------------------------------------------------- #

def test_requete_gemini(gemini):
    appels = gemini(['{"category": "non_materiel"}'])
    assert sft.analyser_texte_llm("mon prompt", max_tokens=321) == {"category": "non_materiel"}

    appel = appels[0]
    assert appel["url"] == sft.GEMINI_URL.format(model=sft.GEMINI_DEFAULT_MODEL)
    # Clé dans l'en-tête, jamais dans l'URL (qui finit dans les journaux).
    assert appel["headers"]["x-goog-api-key"] == "cle-gemini-de-test"
    assert "cle-gemini-de-test" not in appel["url"]
    assert appel["json"]["contents"][0]["parts"][0]["text"] == "mon prompt"
    generation = appel["json"]["generationConfig"]
    assert generation["responseMimeType"] == "application/json"
    assert generation["maxOutputTokens"] == 321
    assert generation["thinkingConfig"] == {"thinkingBudget": 0}


def test_modele_choisi_par_variable_d_environnement(gemini, monkeypatch):
    """Un modèle dont la réflexion ne se coupe pas reçoit une marge de jetons,
    sans quoi elle consommerait tout le budget de la réponse."""
    monkeypatch.setenv(sft.GEMINI_MODEL_ENV, "gemini-2.5-pro")
    appels = gemini(['{"ok": true}'])
    sft.analyser_texte_llm("p", max_tokens=500)
    assert "/models/gemini-2.5-pro:generateContent" in appels[0]["url"]
    generation = appels[0]["json"]["generationConfig"]
    assert "thinkingConfig" not in generation
    assert generation["maxOutputTokens"] == 500 + sft.GEMINI_THINKING_HEADROOM_TOKENS


def test_les_parties_de_reflexion_sont_ignorees(gemini):
    corps = {"candidates": [{"content": {"parts": [
        {"text": "je réfléchis...", "thought": True},
        {"text": '{"verdict": "coherent"}'},
    ]}}]}
    gemini([_ReponseGemini(corps=corps)])
    assert sft.analyser_texte_llm("p") == {"verdict": "coherent"}


def test_prompt_bloque_ne_plante_pas(gemini):
    """Un prompt bloqué par les filtres renvoie un 200 SANS candidates."""
    gemini([_ReponseGemini(corps={"promptFeedback": {"blockReason": "SAFETY"}})])
    assert sft.analyser_texte_llm("p") is None


def test_reponse_tronquee_sans_texte_ne_plante_pas(gemini):
    gemini([_ReponseGemini(corps={"candidates": [{"finishReason": "MAX_TOKENS", "content": {}}]})])
    assert sft.analyser_texte_llm("p") is None


def test_le_message_d_erreur_de_l_api_est_journalise(gemini, caplog):
    """Le seul code HTTP (403) ne dit pas POURQUOI l'appel est refusé ; le
    message de l'API, si."""
    erreur = {"error": {"code": 403, "message": "Method doesn't allow unregistered callers",
                        "status": "PERMISSION_DENIED"}}
    appels = gemini([_ReponseGemini(status_code=403, corps=erreur), '{"ok": true}'])
    with caplog.at_level("ERROR", logger="sec_filings_text"):
        assert sft.analyser_texte_llm("p") is None
    assert len(appels) == 1
    assert "unregistered callers" in caplog.text
    assert "Gemini" in caplog.text


def test_le_retry_delay_du_corps_est_respecte(gemini, monkeypatch):
    """Gemini annonce son délai de reprise dans le corps d'un 429, pas dans
    un en-tête Retry-After."""
    attentes = []
    monkeypatch.setattr(sft.time, "sleep", attentes.append)
    quota = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "36s"},
    ]}}
    gemini([_ReponseGemini(status_code=429, corps=quota), '{"ok": true}'])
    assert sft.analyser_texte_llm("p") == {"ok": True}
    assert any(36 <= a <= 37 for a in attentes), attentes


def test_erreur_reseau_reessayee(gemini):
    appels = gemini([requests.exceptions.ConnectionError("coupé"), '{"ok": true}'])
    assert sft.analyser_texte_llm("p") == {"ok": True}
    assert len(appels) == 2


def test_l_ancien_nom_suit_le_fournisseur(gemini):
    """analyser_texte_mistral reste appelable et passe lui aussi par Gemini."""
    appels = gemini(['{"ok": true}'])
    assert sft.analyser_texte_mistral("p") == {"ok": True}
    assert "generativelanguage.googleapis.com" in appels[0]["url"]
