"""Un modèle qui ne répond plus est mis en pause puis réessayé ; un modèle refusé est coupé.

Constaté : un quota quotidien atteint (429), puis un modèle « overloaded »
(503), faisaient attendre à CHAQUE document ses six réessais -- une bonne
minute sur un 503 -- avant de le rendre sans verdict : les ~6 300 8-K récents
de 04c prenaient des jours. Et un modèle retiré (404) donnait un appel et une
ligne d'erreur par document. Voir le pavé « Disjoncteur du modèle » de
sec_filings_text.py.
"""

from __future__ import annotations

import importlib

import pytest
import requests

import sec_filings_text as sft

_04c = importlib.import_module("04c_recuperation_8k")

SURCHARGE = "The model is overloaded. Please try again later."
MODELE_RETIRE = ("This model models/gemini-2.5-flash is no longer available to new users. Please update "
                 "your code to use models/gemini-3.8-flash for the latest features and improvements.")


# Un schéma qui accepte tout objet : ces tests portent sur le disjoncteur,
# pas sur le format de la réponse.
OBJET = {"type": "object"}


def _analyser(max_tokens: int = 500):
    return sft.analyser_document("document", "consigne", OBJET, max_tokens=max_tokens)


class _Reponse:
    def __init__(self, status_code=200, message="", texte='{"category": "rachat_actions"}'):
        self.status_code = status_code
        self.headers = {}
        self._corps = ({"candidates": [{"content": {"parts": [{"text": texte}]}}]}
                       if status_code < 400 else {"error": {"code": status_code, "message": message}})
        self.text = str(self._corps)

    def json(self):
        return self._corps


class _Horloge:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def avancer(self, secondes):
        self.t += secondes


@pytest.fixture
def horloge(monkeypatch):
    h = _Horloge()
    monkeypatch.setattr(sft, "_horloge", h)
    return h


@pytest.fixture
def gemini(monkeypatch):
    """Pose une file de réponses : un code HTTP, (code, message), une exception."""
    monkeypatch.setenv(sft.GEMINI_API_KEY_ENV, "cle-de-test")
    monkeypatch.setattr(sft.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sft, "GEMINI_RATE_LIMITER", sft.AdaptiveRateLimiter(1000.0))
    appels = []

    def poser(reponses):
        file = list(reponses)

        def faux_post(url, headers=None, json=None, timeout=None):
            appels.append(url)
            item = file.pop(0)
            if isinstance(item, Exception):
                raise item
            if isinstance(item, tuple):
                return _Reponse(*item)
            return _Reponse(item)
        monkeypatch.setattr(sft.requests, "post", faux_post)
        return appels
    return poser


def _sans_reponse(statut=503, message=SURCHARGE):
    """Une analyse sans réponse jusqu'à sa dernière tentative."""
    return [(statut, message)] * sft.GEMINI_MAX_RETRIES


def _mettre_en_pause(gemini, statut=503, message=SURCHARGE, ensuite=()):
    appels = gemini(_sans_reponse(statut, message) * sft.LLM_ECHECS_AVANT_PAUSE + list(ensuite))
    for _ in range(sft.LLM_ECHECS_AVANT_PAUSE):
        assert _analyser() is None
    return appels


# --------------------------------------------------------------------------- #
# Pause
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("statut, message, annonce", [
    (503, SURCHARGE, "Gemini ne répond plus"),
    (429, "Quota exceeded for metric generate_content_free_tier_requests", "Quota Gemini épuisé"),
])
def test_trois_analyses_sans_reponse_mettent_le_modele_en_pause(gemini, horloge, caplog, statut, message, annonce):
    appels = _mettre_en_pause(gemini, statut, message)
    assert sft.llm_en_pause() and not sft.llm_coupe_pour_ce_run()
    avant = len(appels)
    assert _analyser() is None
    assert len(appels) == avant, "un appel est parti pendant la pause"
    assert annonce in caplog.text
    assert message in caplog.text, "la raison donnée par le fournisseur doit être journalisée"


def test_une_erreur_reseau_persistante_met_aussi_en_pause(gemini, horloge):
    coupure = requests.exceptions.ConnectionError("réseau coupé")
    gemini([coupure] * sft.GEMINI_MAX_RETRIES * sft.LLM_ECHECS_AVANT_PAUSE)
    for _ in range(sft.LLM_ECHECS_AVANT_PAUSE):
        _analyser()
    assert sft.llm_en_pause()


def test_deux_analyses_sans_reponse_ne_suffisent_pas(gemini, horloge):
    gemini(_sans_reponse() * 2 + [200])
    _analyser()
    _analyser()
    assert not sft.llm_en_pause()
    assert _analyser() == {"category": "rachat_actions"}


def test_un_pic_resorbe_pendant_les_reessais_ne_compte_pas(gemini, horloge):
    """429 ou 503, puis réponse : l'incident se résorbe pendant les réessais."""
    gemini([429, 429, 200, 503, 503, 200] * 3)
    for _ in range(6):
        assert _analyser() == {"category": "rachat_actions"}
    assert not sft.llm_en_pause()


def test_un_succes_remet_le_compte_a_zero(gemini, horloge):
    gemini(_sans_reponse() * 2 + [200] + _sans_reponse() * 2)
    for _ in range(5):
        _analyser()
    assert not sft.llm_en_pause()


def test_la_pause_ecoulee_le_modele_est_reessaye_et_reprend(gemini, horloge, caplog):
    caplog.set_level("INFO", logger="sec_filings_text")
    appels = _mettre_en_pause(gemini, ensuite=[200, 200])
    horloge.avancer(sft.LLM_PAUSE_INITIALE_S - 1)
    assert _analyser() is None, "la pause n'est pas écoulée"
    horloge.avancer(1)
    avant = len(appels)
    assert _analyser() == {"category": "rachat_actions"}
    assert len(appels) == avant + 1
    assert not sft.llm_en_pause()
    assert "répond de nouveau" in caplog.text
    assert _analyser() == {"category": "rachat_actions"}


def test_un_echec_apres_la_pause_la_double_jusqu_au_plafond(gemini, horloge):
    """L'essai qui suit une pause la relance tout de suite, deux fois plus
    longue : 15 min, 30, 1 h, 2 h, puis 2 h."""
    gemini(_sans_reponse() * (sft.LLM_ECHECS_AVANT_PAUSE + 5))
    for _ in range(sft.LLM_ECHECS_AVANT_PAUSE):
        _analyser()
    pauses = [sft.LLM_PAUSE_INITIALE_S]
    for _ in range(4):
        horloge.avancer(pauses[-1])
        assert not sft.llm_en_pause()
        _analyser()                 # l'essai de reprise échoue
        assert sft.llm_en_pause()
        debut = horloge.t
        while sft.llm_en_pause():
            horloge.avancer(60)
        pauses.append(horloge.t - debut)
    assert pauses == [900.0, 1800.0, 3600.0, 7200.0, 7200.0]


def test_le_message_du_fournisseur_accompagne_chaque_reessai_sans_alarmer(gemini, horloge, caplog):
    """Un 503 absorbé par les réessais est un incident chez Google, pas une
    panne du code : journalisé en INFO, avec la raison donnée par Google."""
    gemini([(503, SURCHARGE), 200])
    with caplog.at_level("INFO", logger="sec_filings_text"):
        assert _analyser() == {"category": "rachat_actions"}
    reessai = [r for r in caplog.records if "tentative 1/" in r.getMessage()]
    assert reessai and reessai[0].levelname == "INFO"
    assert SURCHARGE in reessai[0].getMessage()
    # Et la fin de la série se voit : le document est passé.
    reprise = [r for r in caplog.records if "a répondu à la tentative 2/" in r.getMessage()]
    assert reprise and reprise[0].levelname == "INFO"


def test_une_reponse_du_premier_coup_ne_journalise_rien(gemini, horloge, caplog):
    gemini([200])
    with caplog.at_level("INFO", logger="sec_filings_text"):
        _analyser()
    assert "a répondu" not in caplog.text


# --------------------------------------------------------------------------- #
# Coupure sur un refus de configuration
# --------------------------------------------------------------------------- #

def test_un_modele_retire_coupe_tout_de_suite_et_nomme_son_successeur(gemini, horloge, caplog):
    appels = gemini([(404, MODELE_RETIRE), 200])
    assert _analyser() is None
    assert sft.llm_coupe_pour_ce_run()
    assert _analyser() is None
    assert len(appels) == 1, "un modèle retiré ne revient pas d'un document à l'autre"
    assert "GEMINI_MODEL=gemini-3.8-flash" in caplog.text


def test_un_404_sans_successeur_renvoie_a_la_liste_des_modeles(gemini, horloge, caplog):
    gemini([(404, "models/gemini-9 is not found for API version v1beta")])
    _analyser()
    assert sft.llm_coupe_pour_ce_run()
    assert "diagnostic_llm.py --modeles" in caplog.text


def test_une_cle_invalide_coupe_mais_pas_un_refus_propre_au_document(gemini, horloge):
    """Gemini répond 400 à une clé invalide ; un 400 sur la taille d'un
    document, lui, ne dit rien des documents suivants."""
    gemini([(400, "The input token count exceeds the maximum number of tokens allowed."), 200])
    assert _analyser() is None
    assert not sft.llm_coupe_pour_ce_run()
    assert _analyser() == {"category": "rachat_actions"}

    sft.reinitialiser_disjoncteur_llm()
    gemini([(400, "API key not valid. Please pass a valid API key.")])
    _analyser()
    assert sft.llm_coupe_pour_ce_run()


@pytest.mark.parametrize("message", [
    "Thinking level is not supported for this model.",
    'Invalid JSON payload received. Unknown name "responseJsonSchema" at \'generation_config\': '
    "Cannot find field.",
])
def test_un_reglage_refuse_par_le_modele_coupe_tout_de_suite(gemini, horloge, caplog, message):
    """Un 400 sur le niveau de réflexion ou un champ de la requête vise le
    réglage, pas le document : la même réponse reviendrait pour chaque 8-K."""
    appels = gemini([(400, message), 200])
    assert _analyser() is None
    assert sft.llm_coupe_pour_ce_run()
    assert _analyser() is None
    assert len(appels) == 1


def test_un_niveau_de_reflexion_refuse_dit_quoi_changer(gemini, horloge, caplog, monkeypatch):
    monkeypatch.setenv(sft.GEMINI_MODEL_ENV, "gemma-4-31b-it")
    gemini([(400, "Thinking level is not supported for this model.")])
    _analyser()
    assert "gemma-4-31b-it" in caplog.text and "Gemini 3" in caplog.text


# --------------------------------------------------------------------------- #
# Ce que voient les appelants
# --------------------------------------------------------------------------- #

def test_04c_classe_par_regles_pendant_la_pause(gemini, horloge):
    """Le repli de 04c prend le relais tout de suite : pas une requête de plus,
    et le verdict par règles sera repris par le modèle au run suivant."""
    appels = _mettre_en_pause(gemini)
    avant = len(appels)
    verdict = _04c.classify_8k("AAPL", "2021-05-03", "Item 2.06 Material Impairments\nimpairment charge")
    assert verdict["classification_source"] == "regles_document"
    assert len(appels) == avant


def test_le_bilan_dit_ce_que_le_modele_a_fait(gemini, horloge):
    _mettre_en_pause(gemini, ensuite=[])
    gemini([200, 200])
    horloge.avancer(sft.LLM_PAUSE_INITIALE_S)
    _analyser()
    _analyser()
    bilan = sft.bilan_llm()
    assert bilan.startswith("Gemini (gemini-3.8-flash, réflexion low) -- 2 verdict(s)")
    assert "3 analyse(s) sans réponse" in bilan

    sft.reinitialiser_disjoncteur_llm()
    _mettre_en_pause(gemini)
    _analyser()
    _analyser()
    assert "2 document(s) traité(s) sans le modèle (1 pause(s))" in sft.bilan_llm()
