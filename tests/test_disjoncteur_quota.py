"""Un quota LLM épuisé coupe les appels pour le reste du run, au lieu de les laisser ramper.

Une fois le quota QUOTIDIEN atteint (palier gratuit de Gemini), chaque appel
épuisait ses réessais -- jusqu'à 90 s d'attente chacun -- avant de rendre None :
plusieurs minutes par document. Sur les ~97 500 8-K de 04c, des jours. Voir le
pavé « Disjoncteur de quota » de sec_filings_text.py.
"""

from __future__ import annotations

import importlib

import pytest

import sec_filings_text as sft

_04c = importlib.import_module("04c_recuperation_8k")


class _Reponse:
    def __init__(self, status_code=200, texte='{"category": "rachat_actions"}'):
        self.status_code = status_code
        self.headers = {}
        self._corps = ({"candidates": [{"content": {"parts": [{"text": texte}]}}]}
                       if status_code < 400 else {"error": {"code": status_code, "status": "RESOURCE_EXHAUSTED"}})
        self.text = str(self._corps)

    def json(self):
        return self._corps


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setenv(sft.GEMINI_API_KEY_ENV, "cle-de-test")
    monkeypatch.setattr(sft.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sft, "MISTRAL_RATE_LIMITER", sft.AdaptiveRateLimiter(1000.0))
    appels = []

    def poser(statuts):
        file = list(statuts)

        def faux_post(url, headers=None, json=None, timeout=None):
            appels.append(url)
            return _Reponse(file.pop(0))
        monkeypatch.setattr(sft.requests, "post", faux_post)
        return appels
    return poser


def _refus_complet():
    """Une analyse refusée pour quota jusqu'à sa dernière tentative."""
    return [429] * sft.MISTRAL_MAX_RETRIES


def test_trois_analyses_refusees_pour_quota_coupent_le_modele(gemini, caplog):
    appels = gemini(_refus_complet() * sft.LLM_REFUS_QUOTA_AVANT_COUPURE)
    for _ in range(sft.LLM_REFUS_QUOTA_AVANT_COUPURE):
        assert sft.analyser_texte_llm("p") is None
    assert sft.llm_coupe_pour_ce_run()
    avant = len(appels)
    assert sft.analyser_texte_llm("p") is None
    assert len(appels) == avant, "un appel est parti alors que le quota est épuisé"
    assert "Quota Gemini épuisé" in caplog.text


def test_un_pic_resorbe_pendant_les_reessais_ne_compte_pas(gemini):
    """429, 429, puis réponse : c'est un quota par minute, il se résorbe."""
    gemini(([429, 429, 200]) * 5)
    for _ in range(5):
        assert sft.analyser_texte_llm("p") == {"category": "rachat_actions"}
    assert not sft.llm_coupe_pour_ce_run()


def test_un_succes_remet_le_compte_a_zero(gemini):
    gemini(_refus_complet() * 2 + [200] + _refus_complet() * 2)
    for _ in range(5):
        sft.analyser_texte_llm("p")
    assert not sft.llm_coupe_pour_ce_run()


def test_une_panne_d_une_autre_nature_ne_coupe_rien(gemini):
    gemini([503] * sft.MISTRAL_MAX_RETRIES * 4)
    for _ in range(4):
        assert sft.analyser_texte_llm("p") is None
    assert not sft.llm_coupe_pour_ce_run()


def test_04c_classe_par_regles_des_que_le_modele_est_coupe(gemini):
    """Le repli de 04c prend le relais tout de suite : pas une requête de plus,
    et le verdict par règles sera repris par le modèle au run suivant."""
    appels = gemini(_refus_complet() * sft.LLM_REFUS_QUOTA_AVANT_COUPURE)
    for _ in range(sft.LLM_REFUS_QUOTA_AVANT_COUPURE):
        sft.analyser_texte_llm("p")
    avant = len(appels)
    verdict = _04c.classify_8k("AAPL", "2021-05-03", "Item 2.06 Material Impairments\nimpairment charge")
    assert verdict["classification_source"] == "regles_document"
    assert len(appels) == avant
