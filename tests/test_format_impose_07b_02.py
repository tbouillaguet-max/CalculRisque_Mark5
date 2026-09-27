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


EXTRAIT = "[Item 1A - Facteurs de risque]\nThe Company is subject to litigation."


def _periode(symbol="AAPL", cik="320193", filed_date="2025-10-31", gap_pct=23.456):
    return pd.Series({**PERIODE.to_dict(), "symbol": symbol, "cik": cik,
                      "filed_date": filed_date, "gap_pct": gap_pct})


@pytest.fixture
def filing(monkeypatch):
    """Un 10-K par CIK, déposé à la date demandée ; rend les URL téléchargées."""
    monkeypatch.setattr(_07b.sft, "find_filing_asof", lambda cik, filed_date, forms=(): {
        "form": "10-K", "filing_date": filed_date, "accession_number": f"{cik}-25-000079",
        "primary_document": f"{cik}.htm",
    })
    telecharges = []

    def telecharger(url, max_chars=None, form=None):
        telecharges.append(url)
        return EXTRAIT, "sections"
    monkeypatch.setattr(_07b.sft, "fetch_filing_text", telecharger)
    return telecharges


def _juger(periodes, memoire=None, output_dir=None):
    """Le chemin de 07b : chaque période préparée, puis les lots soumis."""
    resultats, en_attente = {}, []
    for periode in periodes:
        resultat, attente = _07b.preparer_periode(periode, memoire)
        if attente is None:
            resultats[periode["symbol"]] = resultat
        else:
            en_attente.append(attente)
    for attente, resultat in _07b.classer_en_attente(en_attente, memoire, output_dir):
        resultats[attente.row["symbol"]] = resultat
    return resultats


A_SURVEILLER = {"verdict": "a_surveiller", "justification": "Un litige est en cours.", "risques_cites": ["litige"]}
COHERENT = {"verdict": "coherent", "justification": "Rien à signaler.", "risques_cites": []}


def test_07b_envoie_les_extraits_la_consigne_et_le_schema_separement(gemini, filing):
    corps = gemini([json.dumps({"d1": A_SURVEILLER})])

    verdict = _juger([PERIODE])["AAPL"]

    envoye = corps[0]
    document = envoye["contents"][0]["parts"][0]["text"]
    assert document.startswith('<document id="d1">') and EXTRAIT in document
    assert "AAPL" in document and "10-K" in document and "+23.5 %" in document
    assert "Extrait : sections facteurs de risque" in document, "le document dit ce que contient l'extrait"
    consigne = envoye["systemInstruction"]["parts"][0]["text"]
    assert "2025-10-31" in consigne
    assert "litigation" not in consigne and "AAPL" not in consigne, "le document ne part pas dans la consigne"
    assert envoye["generationConfig"]["responseJsonSchema"]["properties"] == {"d1": _07b.SCHEMA_VERDICT}
    assert verdict["verdict"] == "a_surveiller"
    assert json.loads(verdict["risques_cites"]) == ["litige"]
    assert verdict["extraction_mode"] == "sections"


def test_07b_un_verdict_hors_liste_n_est_jamais_enregistre(gemini, filing, tmp_path):
    hors_liste = json.dumps({"d1": {"verdict": "incoherent", "justification": "x", "risques_cites": []}})
    corps = gemini([hors_liste, hors_liste])

    verdict = _juger([PERIODE], _07b.charger_memoire(tmp_path), tmp_path)["AAPL"]

    assert verdict["verdict"] == "non_evalue_reponse_invalide"
    assert len(corps) == 2, "une reprise, pas davantage"
    assert not _07b.memoire_path(tmp_path).exists(), "rien de mémorisé : la période repartira"


def test_07b_le_modele_peut_rendre_chaque_verdict_que_le_backtest_exclut():
    """Un verdict exclu par le filtre qualitatif mais absent de la liste ne
    pourrait jamais être rendu : le filtre ne filtrerait plus rien, en silence."""
    assert set(config.QUALITATIVE_GATE_EXCLUDED_VERDICTS) <= set(_07b.VERDICTS)
    assert _07b.SCHEMA_VERDICT["properties"]["verdict"]["enum"] == list(_07b.VERDICTS)


@pytest.mark.parametrize("mode, attendu", [
    ("sections", "sections facteurs de risque"),
    ("debut_document", "début du document"),
    (None, "extrait du document"),
])
def test_07b_le_document_dit_ce_que_contient_l_extrait(mode, attendu):
    """L'ancien prompt annonçait toujours « le début du document »."""
    attente = _07b.PeriodeEnAttente(row=PERIODE, texte=EXTRAIT, filing={
        "form": "10-K", "accession_number": "1", "extraction_mode": mode})
    assert f"Extrait : {attendu}" in _07b._document_pour_le_modele(attente)


def test_07b_les_periodes_deposees_le_meme_jour_partent_ensemble(gemini, filing):
    """Une requête par date de dépôt, les plus récentes d'abord : jamais un
    filing plus récent à côté d'un plus ancien, qu'il pourrait éclairer."""
    corps = gemini([json.dumps({"d1": COHERENT, "d2": A_SURVEILLER}), json.dumps({"d1": COHERENT})])

    resultats = _juger([_periode("IBM", "51143", filed_date="2025-02-25"),
                        _periode("AAPL"), _periode("MSFT", "789019")])

    assert len(corps) == 2
    assert "2025-10-31" in corps[0]["systemInstruction"]["parts"][0]["text"]
    assert list(corps[0]["generationConfig"]["responseJsonSchema"]["properties"]) == ["d1", "d2"]
    assert "2025-02-25" in corps[1]["systemInstruction"]["parts"][0]["text"]
    assert {s: r["verdict"] for s, r in resultats.items()} == {
        "AAPL": "coherent", "MSFT": "a_surveiller", "IBM": "coherent"}


def test_07b_une_periode_isolee_sans_reponse_reste_a_juger(filing, monkeypatch, tmp_path):
    """Le lot a répondu, sauf pour la période qui le faisait échouer (voir
    sec_filings_text.analyser_documents) : elle seule sort sans verdict, et
    rien n'est mémorisé pour elle."""
    monkeypatch.setenv(sft.GEMINI_API_KEY_ENV, "cle-de-test")
    monkeypatch.setattr(_07b.sft, "analyser_documents", lambda documents, *a, **k: {"d2": COHERENT})

    resultats = _juger([_periode("AAPL"), _periode("MSFT", "789019")], _07b.charger_memoire(tmp_path), tmp_path)

    assert (resultats["AAPL"]["verdict"], resultats["MSFT"]["verdict"]) == ("non_evalue_reponse_invalide", "coherent")
    assert list(_07b.charger_memoire(tmp_path)) == [_07b.cle_memoire("789019-25-000079", 23.456)]


def test_07b_un_filing_deja_juge_ne_coute_ni_requete_ni_telechargement(gemini, filing, tmp_path):
    corps = gemini([json.dumps({"d1": A_SURVEILLER})])
    premier = _juger([PERIODE], _07b.charger_memoire(tmp_path), tmp_path)["AAPL"]
    assert len(corps) == 1 and len(filing) == 1

    # Run suivant, mémoire relue du disque : même filing, écart recalculé
    # mais de même sens.
    second = _juger([_periode(gap_pct=11.0)], _07b.charger_memoire(tmp_path), tmp_path)["AAPL"]

    assert len(corps) == 1 and len(filing) == 1, "ni requête Gemini, ni téléchargement SEC"
    assert second["from_cache"] is True and premier["from_cache"] is False
    assert [second[k] for k in ("verdict", "justification", "risques_cites", "extraction_mode")] == \
           [premier[k] for k in ("verdict", "justification", "risques_cites", "extraction_mode")]


def test_07b_un_ecart_de_sens_oppose_est_rejuge(gemini, filing, tmp_path):
    """Une sous-évaluation et une survalorisation ne se jugent pas pareil."""
    corps = gemini([json.dumps({"d1": COHERENT}), json.dumps({"d1": A_SURVEILLER})])
    _juger([PERIODE], _07b.charger_memoire(tmp_path), tmp_path)

    verdict = _juger([_periode(gap_pct=-5.0)], _07b.charger_memoire(tmp_path), tmp_path)["AAPL"]

    assert len(corps) == 2 and verdict["from_cache"] is False and verdict["verdict"] == "a_surveiller"


def test_07b_la_memoire_sert_meme_sans_cle(gemini, filing, tmp_path, monkeypatch):
    gemini([json.dumps({"d1": A_SURVEILLER})])
    _juger([PERIODE], _07b.charger_memoire(tmp_path), tmp_path)
    monkeypatch.delenv(sft.GEMINI_API_KEY_ENV)

    resultats = _juger([PERIODE, _periode("MSFT", "789019")], _07b.charger_memoire(tmp_path), tmp_path)

    assert resultats["AAPL"]["verdict"] == "a_surveiller"
    assert resultats["MSFT"]["verdict"] == "non_evalue_pas_de_cle_api"


# --------------------------------------------------------------------------- #
# 02
# --------------------------------------------------------------------------- #
LOT = ["Apple Inc.", "AT&T Inc.", "Moody's Corporation"]


def test_02_envoie_les_entreprises_la_consigne_et_le_schema_separement(gemini):
    corps = gemini([json.dumps({"d1": "Technologie", "d2": "Télécommunications",
                                "d3": "Services financiers"}, ensure_ascii=False)])

    assert _02.appeler_llm(LOT) == {"Apple Inc.": "Technologie", "AT&T Inc.": "Télécommunications",
                                    "Moody's Corporation": "Services financiers"}

    envoye = corps[0]
    assert envoye["contents"][0]["parts"][0]["text"] == (
        '<document id="d1">\nApple Inc.\n</document>\n\n<document id="d2">\nAT&T Inc.\n</document>\n\n'
        '<document id="d3">\nMoody\'s Corporation\n</document>')
    assert envoye["systemInstruction"]["parts"][0]["text"] == _02.CONSIGNE
    schema = envoye["generationConfig"]["responseJsonSchema"]
    assert schema["required"] == ["d1", "d2", "d3"], "chaque entreprise du lot est un champ obligatoire"
    assert all(champ == {"type": "string", "enum": [*_02.SECTEURS, _02.INDETERMINE]}
               for champ in schema["properties"].values())


@pytest.mark.parametrize("reponse, en_cause", [
    ({"d1": "Informatique", "d2": "Télécommunications", "d3": "Services financiers"}, "Apple Inc."),
    ({"d1": "Technologie", "d2": "Télécommunications"}, "Moody's Corporation"),   # entreprise oubliée
])
def test_02_une_reponse_hors_format_n_est_jamais_retenue(gemini, reponse, en_cause):
    """Un secteur inventé ou une entreprise oubliée : le lot est redemandé,
    puis recoupé jusqu'à isoler l'entreprise en cause (voir
    sec_filings_text.analyser_documents). Elle seule reste « indetermine », à
    reprendre au prochain run ; les autres ont leur secteur."""
    gemini([json.dumps(reponse, ensure_ascii=False)] * 10)

    secteurs = _02.appeler_llm(LOT)

    assert secteurs.pop(en_cause) == _02.INDETERMINE
    assert secteurs and set(secteurs.values()) <= set(_02.SECTEURS)


# --------------------------------------------------------------------------- #
# Les trois consignes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script, consigne", [
    ("04c", _04c.build_consigne("2026-06-01")),
    ("07b", _07b.build_consigne("2025-10-31")),
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
    corps = gemini([json.dumps({"d1": "Technologie"})])
    assert _02.appeler_llm([float("nan"), "Apple Inc.", "  "]) == {"Apple Inc.": "Technologie"}
    assert corps[0]["contents"][0]["parts"][0]["text"] == '<document id="d1">\nApple Inc.\n</document>'
