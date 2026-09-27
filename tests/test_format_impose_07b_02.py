"""07b et 02 envoient à Gemini un document, une consigne et un format, comme 04c.

Avant, les deux scripts envoyaient un prompt unique terminé par un exemple de
JSON, et ne vérifiaient que la présence d'une clé : un verdict hors liste
(07b) ou un secteur inventé (02) passait tel quel. Désormais, le format est un
schéma JSON que Gemini respecte sous contrainte, et qui est revérifié ; ces
tests passent par le vrai chemin HTTP, simulé.
"""

from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest

import config
import sec_filings_text as sft

_04c = importlib.import_module("04c_recuperation_8k")
_07b = importlib.import_module("07b_validation_qualitative")
_02 = importlib.import_module("02_categoriser_secteurs")


class _Reponse:
    def __init__(self, texte):
        self.status_code = 200
        self.headers = {}
        self._texte = texte

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": self._texte}]}, "finishReason": "STOP"}]}


@pytest.fixture
def gemini(monkeypatch):
    """Pose une file de réponses (textes JSON) et rend les corps envoyés."""
    monkeypatch.setenv(sft.GEMINI_API_KEY_ENV, "cle-de-test")
    monkeypatch.setattr(sft.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sft, "GEMINI_RATE_LIMITER", sft.AdaptiveRateLimiter(1000.0))
    corps = []

    def poser(textes):
        file = list(textes)

        def faux_post(url, headers=None, json=None, timeout=None):
            corps.append(json)
            return _Reponse(file.pop(0))
        monkeypatch.setattr(sft.requests, "post", faux_post)
        return corps
    return poser


# --------------------------------------------------------------------------- #
# 07b
# --------------------------------------------------------------------------- #
PERIODE = pd.Series({
    "symbol": "AAPL", "cik": "320193", "period_type": "FY", "fiscal_year": 2025,
    "fiscal_quarter": None, "filed_date": "2025-10-31", "gap_pct": 23.456,
})


@pytest.fixture
def filing(monkeypatch):
    monkeypatch.setattr(_07b.sft, "get_filing_text_asof", lambda cik, filed_date, forms=(): {
        "form": "10-K", "filing_date": filed_date, "accession_number": "0000320193-25-000079",
        "primary_document": "aapl-20250927.htm", "extraction_mode": "sections",
        "text": "[Item 1A - Facteurs de risque]\nThe Company is subject to litigation.",
    })


def test_07b_envoie_l_extrait_la_consigne_et_le_schema_separement(gemini, filing):
    corps = gemini([json.dumps({"verdict": "a_surveiller", "justification": "Un litige est en cours.",
                                "risques_cites": ["litige"]})])

    verdict = _07b.evaluate_period(PERIODE)

    envoye = corps[0]
    assert envoye["contents"][0]["parts"][0]["text"].startswith("[Item 1A - Facteurs de risque]")
    consigne = envoye["systemInstruction"]["parts"][0]["text"]
    assert "AAPL" in consigne and "10-K" in consigne and "2025-10-31" in consigne and "23.5 %" in consigne
    assert "sections" in consigne, "la consigne dit ce que contient l'extrait"
    assert "litigation" not in consigne, "le document ne part pas dans la consigne"
    assert envoye["generationConfig"]["responseJsonSchema"] == _07b.SCHEMA_REPONSE
    assert verdict["verdict"] == "a_surveiller"
    assert json.loads(verdict["risques_cites"]) == ["litige"]
    assert verdict["extraction_mode"] == "sections"


def test_07b_un_verdict_hors_liste_n_est_jamais_enregistre(gemini, filing):
    hors_liste = json.dumps({"verdict": "incoherent", "justification": "x", "risques_cites": []})
    corps = gemini([hors_liste, hors_liste])

    verdict = _07b.evaluate_period(PERIODE)

    assert verdict["verdict"] == "non_evalue_reponse_invalide"
    assert len(corps) == 2, "une reprise, pas davantage"


def test_07b_le_modele_peut_rendre_chaque_verdict_que_le_backtest_exclut():
    """Un verdict exclu par le filtre qualitatif mais absent de la liste ne
    pourrait jamais être rendu : le filtre ne filtrerait plus rien, en silence."""
    assert set(config.QUALITATIVE_GATE_EXCLUDED_VERDICTS) <= set(_07b.VERDICTS)
    assert _07b.SCHEMA_REPONSE["properties"]["verdict"]["enum"] == list(_07b.VERDICTS)


@pytest.mark.parametrize("mode, attendu", [
    ("sections", "les sections repérées dans le document"),
    ("debut_document", "le début du document"),
    (None, "un extrait du document"),
])
def test_07b_la_consigne_dit_ce_que_contient_l_extrait(mode, attendu):
    """L'ancien prompt annonçait toujours « le début du document »."""
    assert attendu in _07b.build_consigne("AAPL", "10-K", "2025-10-31", 10.0, mode)


# --------------------------------------------------------------------------- #
# 02
# --------------------------------------------------------------------------- #
LOT = ["Apple Inc.", "AT&T Inc.", "Moody's Corporation"]


def test_02_chaque_entreprise_du_lot_est_un_champ_obligatoire():
    schema = _02.schema_reponse(LOT)
    assert list(schema["properties"]) == LOT and schema["required"] == LOT
    for champ in schema["properties"].values():
        assert champ["enum"] == [*_02.SECTEURS, _02.INDETERMINE]


def test_02_envoie_la_liste_la_consigne_et_le_schema_separement(gemini):
    reponse = {"Apple Inc.": "Technologie", "AT&T Inc.": "Télécommunications",
               "Moody's Corporation": "Services financiers"}
    corps = gemini([json.dumps(reponse, ensure_ascii=False)])

    assert _02.appeler_llm(LOT) == reponse

    envoye = corps[0]
    assert envoye["contents"][0]["parts"][0]["text"] == "Apple Inc.\nAT&T Inc.\nMoody's Corporation"
    assert envoye["systemInstruction"]["parts"][0]["text"] == _02.CONSIGNE
    assert envoye["generationConfig"]["responseJsonSchema"] == _02.schema_reponse(LOT)


@pytest.mark.parametrize("reponse", [
    {"Apple Inc.": "Informatique", "AT&T Inc.": "Télécommunications",
     "Moody's Corporation": "Services financiers"},                   # secteur inventé
    {"Apple Inc.": "Technologie", "AT&T Inc.": "Télécommunications"},  # entreprise oubliée
])
def test_02_une_reponse_hors_format_n_est_jamais_retenue(gemini, reponse):
    texte = json.dumps(reponse, ensure_ascii=False)
    gemini([texte, texte])
    assert _02.appeler_llm(LOT) == {}, "le lot reste « indetermine », à reprendre au prochain run"


# --------------------------------------------------------------------------- #
# Les trois consignes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script, consigne", [
    ("04c", _04c.build_consigne("AAPL", "2026-06-01", ["Item 5.02"])),
    ("07b", _07b.build_consigne("AAPL", "10-K", "2025-10-31", 23.4, "sections")),
    ("02", _02.CONSIGNE),
])
def test_aucune_consigne_ne_repete_le_format(script, consigne):
    """Google le déconseille : le schéma répété dans la consigne, exemple de
    JSON compris, fait baisser la qualité de la réponse."""
    assert "{" not in consigne and "}" not in consigne, script
    if script == "02":
        assert not [s for s in _02.SECTEURS if s in consigne], "la liste des secteurs est dans le schéma"


def test_02_un_nom_manquant_ne_fait_pas_echouer_le_lot(gemini):
    """01b peut laisser le nom d'une radiée vide (NaN) : il n'est pas soumis,
    et les autres entreprises du lot sont classées normalement."""
    corps = gemini([json.dumps({"Apple Inc.": "Technologie"})])
    assert _02.appeler_llm([float("nan"), "Apple Inc.", "  "]) == {"Apple Inc.": "Technologie"}
    assert corps[0]["contents"][0]["parts"][0]["text"] == "Apple Inc."
