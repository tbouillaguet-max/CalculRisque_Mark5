"""D7 : robustesse du parsing et des reprises de l'appel LLM."""

from __future__ import annotations

import json

import pytest
import requests

import sec_filings_text as sft

ATTENDU = {"verdict": "coherent", "justification": "rien à signaler"}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("reponse", [
    '{"verdict": "coherent", "justification": "rien à signaler"}',
    '```json\n{"verdict": "coherent", "justification": "rien à signaler"}\n```',
    '```\n{"verdict": "coherent", "justification": "rien à signaler"}\n```',
    '  \n {"verdict": "coherent", "justification": "rien à signaler"}  \n ',
    'Voici mon analyse :\n{"verdict": "coherent", "justification": "rien à signaler"}\nFin.',
])
def test_les_enrobages_courants_sont_acceptes(reponse):
    """L'exigence `content.startswith("{")` rejetait tout ça, alors que le
    JSON attendu était bien présent."""
    assert sft._parse_json_reponse(reponse) == ATTENDU


@pytest.mark.parametrize("reponse", [
    "", "   ", "Je ne peux pas répondre.", "[1, 2, 3]", '{"incomplet": ', None, 42,
])
def test_les_reponses_reellement_inexploitables_sont_refusees(reponse):
    """La tolérance ne doit pas aller jusqu'à accepter autre chose qu'un
    objet JSON complet."""
    assert sft._parse_json_reponse(reponse) is None


# --------------------------------------------------------------------------- #
# Reprises
# --------------------------------------------------------------------------- #

class _ReponseHttp:
    def __init__(self, contenu):
        self._contenu = contenu

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._contenu}}]}


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv(sft.MISTRAL_API_KEY_ENV, "cle-de-test")
    monkeypatch.setattr(sft.time, "sleep", lambda _s: None)

    appels = []

    def poser(reponses):
        def faux_post(url, headers=None, json=None, timeout=None):
            appels.append(json)
            item = reponses.pop(0)
            if isinstance(item, Exception):
                raise item
            return _ReponseHttp(item)
        monkeypatch.setattr(sft.requests, "post", faux_post)
        return appels

    return poser


def test_le_mode_json_natif_est_demande(api):
    appels = api(['{"verdict": "coherent"}'])
    sft.analyser_texte_mistral("prompt")
    assert appels[0]["response_format"] == {"type": "json_object"}


def test_une_reprise_sur_reponse_non_parsable(api):
    """L'ancienne version abandonnait immédiatement ; on retente UNE fois."""
    appels = api(["pas du json", '{"verdict": "coherent"}'])
    assert sft.analyser_texte_mistral("prompt") == {"verdict": "coherent"}
    assert len(appels) == 2


def test_pas_plus_d_une_reprise_sur_reponse_non_parsable(api):
    """Insister sur un modèle hors format coûte des appels payants pour rien."""
    appels = api(["pas du json", "toujours pas", "encore moins"])
    assert sft.analyser_texte_mistral("prompt") is None
    assert len(appels) == 2


def test_les_erreurs_reseau_gardent_leurs_trois_tentatives(api):
    appels = api([
        requests.exceptions.ConnectionError("coupé"),
        requests.exceptions.ConnectionError("encore"),
        '{"verdict": "coherent"}',
    ])
    assert sft.analyser_texte_mistral("prompt") == {"verdict": "coherent"}
    assert len(appels) == 3


def test_sans_cle_api_aucun_appel(monkeypatch):
    monkeypatch.delenv(sft.MISTRAL_API_KEY_ENV, raising=False)

    def interdit(*a, **k):
        raise AssertionError("aucun appel réseau ne doit être tenté sans clé")

    monkeypatch.setattr(sft.requests, "post", interdit)
    assert sft.analyser_texte_mistral("prompt") is None


def test_enveloppe_inattendue_ne_plante_pas(monkeypatch):
    """Une réponse HTTP 200 dont le corps n'a pas la forme attendue est un
    échec définitif, pas une réponse à reparser indéfiniment."""
    monkeypatch.setenv(sft.MISTRAL_API_KEY_ENV, "cle-de-test")
    monkeypatch.setattr(sft.time, "sleep", lambda _s: None)

    class _SansChoices:
        def raise_for_status(self):
            pass

        def json(self):
            return {"pas_de_choices": True}

    monkeypatch.setattr(
        sft.requests, "post", lambda url, headers=None, json=None, timeout=None: _SansChoices(),
    )
    assert sft.analyser_texte_mistral("prompt") is None
