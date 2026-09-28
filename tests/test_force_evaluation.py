"""04c --force-evaluation : Gemini relit les 8-K récents qu'il a lus sans leur communiqué joint.

Constaté le 2026-09-28 : les 3 274 8-K classés par Gemini la veille l'avaient
été sur le 8-K seul. La mémoire les servait tels quels -- un 8-K est figé --,
et rien ne permettait de les lui rendre sans tout retélécharger
(--no-llm-cache, 99 790 8-K). Voir VERSION_LECTURE_MODELE.
"""

from __future__ import annotations

import importlib
import json

_c8k = importlib.import_module("04c_recuperation_8k")

LIMITE = "2025-08-22"


def _memoire(tmp_path, entrees):
    chemin = _c8k.llm_cache_path(tmp_path)
    chemin.write_text("".join(json.dumps(e) + "\n" for e in entrees), encoding="utf-8")


def _gemini(accession, filed_date="2026-06-01", items=("Item 8.01",), **champs):
    return {"symbol": "AAPL", "accession_number": accession, "filed_date": filed_date,
            "item_codes": list(items), "category": "non_materiel", "materiality": False,
            "classification_source": "gemini", **champs}


ENTREES = [
    _gemini("SANS_COMMUNIQUE"),                                               # lu sans son communiqué
    _gemini("AVEC_COMMUNIQUE", version_lecture=_c8k.VERSION_LECTURE_MODELE),  # déjà relu
    _gemini("ANCIEN", filed_date="2015-03-02"),                               # hors fenêtre du modèle
    _gemini("MATERIEL_D_OFFICE", items=("Item 2.06",)),                       # que Gemini ne lirait pas
]


def test_sans_l_option_les_verdicts_de_gemini_restent_servis(tmp_path, monkeypatch):
    """Le quota de requêtes est compté : pas de relecture sans la demander."""
    monkeypatch.setattr(_c8k.sft, "llm_disponible", lambda: True)
    _memoire(tmp_path, ENTREES)
    assert len(_c8k.load_llm_cache(tmp_path, LIMITE)) == 4


def test_l_option_rend_a_gemini_les_seuls_8k_lus_sans_leur_communique(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(_c8k.sft, "llm_disponible", lambda: True)
    _memoire(tmp_path, ENTREES)
    with caplog.at_level("INFO"):
        cache = _c8k.load_llm_cache(tmp_path, LIMITE, forcer_modele=True)
    assert set(cache) == {"AAPL:AVEC_COMMUNIQUE", "AAPL:ANCIEN", "AAPL:MATERIEL_D_OFFICE"}
    assert "--force-evaluation : 1 8-K récents" in caplog.text


def test_sans_cle_l_option_est_sans_effet(tmp_path, monkeypatch):
    monkeypatch.setattr(_c8k.sft, "llm_disponible", lambda: False)
    _memoire(tmp_path, ENTREES)
    assert len(_c8k.load_llm_cache(tmp_path, LIMITE, forcer_modele=True)) == 4


def test_un_8k_relu_ne_l_est_pas_deux_fois(tmp_path, monkeypatch):
    """De bout en bout : le 8-K forcé est retéléchargé, relu avec son
    communiqué, et son nouveau verdict porte la version courante -- relancer
    l'option (quota épuisé la veille) ne le relit pas."""
    monkeypatch.setenv(_c8k.sft.GEMINI_API_KEY_ENV, "une-cle")
    monkeypatch.setattr(_c8k.sft, "fetch_submissions_strict", lambda cik: [
        {"form": "8-K", "filing_date": "2026-06-01", "accession_number": "SANS_COMMUNIQUE",
         "primary_document": "r.htm"}])
    telecharges = []
    monkeypatch.setattr(_c8k.sft, "fetch_filing_text", lambda url, form=None: telecharges.append(url) or (
        "Item 8.01 Other Events. The Company issued a press release.", "debut_document"))
    monkeypatch.setattr(_c8k.sft, "texte_piece_jointe", lambda *a, **k: "The Company withdrew its guidance.")
    lus = []
    monkeypatch.setattr(_c8k.sft, "analyser_documents", lambda documents, *a, **k: lus.extend(
        documents.values()) or {i: {"category": "changement_guidance", "materiality": True,
                                    "summary": "Prévisions retirées."} for i in documents})
    _memoire(tmp_path, [_gemini("SANS_COMMUNIQUE")])

    for _ in range(2):
        cache = _c8k.load_llm_cache(tmp_path, LIMITE, forcer_modele=True)
        attente = []
        _c8k.process_ticker_8k("AAPL", "320193", [("2026-01-01", "2026-12-31")], cache, tmp_path,
                               LIMITE, attente)
        _c8k.classer_en_attente(attente, cache, tmp_path)

    assert len(telecharges) == 1 and len(lus) == 1, "relu une seule fois"
    assert "withdrew its guidance" in lus[0]
    verdict = _c8k.load_llm_cache(tmp_path, LIMITE)["AAPL:SANS_COMMUNIQUE"]
    assert (verdict["category"], verdict["version_lecture"]) == (
        "changement_guidance", _c8k.VERSION_LECTURE_MODELE)
