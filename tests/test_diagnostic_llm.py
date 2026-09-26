"""diagnostic_llm.py : la réponse brute du fournisseur en quelques secondes.

Sans lui, un 404 (modèle retiré) ou un 503 (modèle surchargé) ne se voyait
qu'en lançant 04c, après ses premiers téléchargements SEC, au milieu de
centaines de lignes.
"""

from __future__ import annotations

import pytest
import requests

import diagnostic_llm as diag
import sec_filings_text as sft

CLE = "cle-secrete-de-test"


class _Reponse:
    def __init__(self, status_code=200, message="", corps=None):
        self.status_code = status_code
        self.headers = {}
        if corps is None:
            corps = ({"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}
                     if status_code < 400 else {"error": {"code": status_code, "message": message}})
        self._corps = corps
        self.text = str(corps)

    def json(self):
        return self._corps


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setenv(sft.GEMINI_API_KEY_ENV, CLE)
    # Posée puis écrasée par --modele : l'annulation du test la retire.
    monkeypatch.setenv(sft.GEMINI_MODEL_ENV, sft.GEMINI_DEFAULT_MODEL)
    monkeypatch.setattr(diag.time, "sleep", lambda _s: None)
    appels = []

    def poser(reponses):
        file = list(reponses)

        def faux_post(url, headers=None, json=None, timeout=None):
            appels.append({"url": url, "headers": headers})
            item = file.pop(0)
            if isinstance(item, Exception):
                raise item
            return item if isinstance(item, _Reponse) else _Reponse(*item) if isinstance(item, tuple) else _Reponse(item)
        monkeypatch.setattr(requests, "post", faux_post)
        return appels
    return poser


def test_sans_cle_rien_ne_part_et_l_aide_s_affiche(capsys, monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("appel sans clé"))
    assert diag.main([]) == 1
    sortie = capsys.readouterr().out
    assert "Aucune clé vue" in sortie and ".env" in sortie


def test_un_modele_qui_repond(gemini, capsys):
    gemini([200, 200, 200])
    assert diag.main([]) == 0
    sortie = capsys.readouterr().out
    assert sortie.count(": OK en") == 3
    assert "Le modèle répond" in sortie
    assert CLE not in sortie, "la clé ne doit jamais s'afficher"


def test_un_modele_surcharge_se_voit_avec_les_mots_de_google(gemini, capsys):
    gemini([(503, "The model is overloaded. Please try again later."), 200, (503, "The model is overloaded.")])
    assert diag.main(["--essais", "3"]) == 0
    sortie = capsys.readouterr().out
    assert "HTTP 503" in sortie and "The model is overloaded" in sortie
    assert "1/3 réponse(s)" in sortie and "surchargé" in sortie


def test_un_modele_retire_donne_la_ligne_a_mettre_dans_env(gemini, capsys):
    message = ("This model models/gemini-2.5-flash is no longer available to new users. "
               "Please update your code to use models/gemini-3.8-flash for the latest features.")
    gemini([(404, message)])
    assert diag.main(["--essais", "1", "--modele", "gemini-2.5-flash"]) == 1
    sortie = capsys.readouterr().out
    assert "GEMINI_MODEL=gemini-3.8-flash" in sortie
    assert "option --modele" in sortie


def test_modele_essaye_sans_toucher_a_la_configuration(gemini):
    appels = gemini([200])
    diag.main(["--essais", "1", "--modele", "gemini-autre"])
    assert "/models/gemini-autre:generateContent" in appels[0]["url"]
    assert appels[0]["headers"]["x-goog-api-key"] == CLE
    assert CLE not in appels[0]["url"]


def test_une_coupure_reseau_ne_plante_pas(gemini, capsys):
    gemini([requests.exceptions.ConnectionError("réseau coupé")])
    assert diag.main(["--essais", "1"]) == 1
    sortie = capsys.readouterr().out
    assert "pas de réponse" in sortie and "ConnectionError" in sortie


def test_la_liste_des_modeles(gemini, monkeypatch, capsys):
    vus = []

    def faux_get(url, headers=None, params=None, timeout=None):
        vus.append({"url": url, "headers": headers, "params": params})
        return _Reponse(corps={"models": [
            {"name": "models/gemini-3.8-flash", "displayName": "Gemini 3.8 Flash",
             "supportedGenerationMethods": ["generateContent", "countTokens"]},
            {"name": "models/gemini-3.8-flash-lite", "displayName": "Gemini 3.8 Flash-Lite",
             "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/text-embedding-9", "supportedGenerationMethods": ["embedContent"]},
        ]})
    monkeypatch.setattr(requests, "get", faux_get)
    assert diag.main(["--modeles"]) == 0
    sortie = capsys.readouterr().out
    assert "2 modèle(s)" in sortie
    assert "gemini-3.8-flash-lite" in sortie and "text-embedding-9" not in sortie
    assert "<-- utilisé" in sortie
    assert vus[0]["headers"]["x-goog-api-key"] == CLE
    assert CLE not in vus[0]["url"] and CLE not in str(vus[0]["params"])
