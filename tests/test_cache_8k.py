"""Mémoire des 8-K déjà classifiés par Mistral (04c_recuperation_8k.py).

Un 8-K est un document figé : une fois classifié avec succès, il ne doit plus
jamais être re-téléchargé ni re-soumis au LLM.
"""

from __future__ import annotations

import importlib
import json

import pytest

module_04c = importlib.import_module("04c_recuperation_8k")


FILING = {
    "form": "8-K", "filing_date": "2023-05-04",
    "accession_number": "0000066740-23-000042", "primary_document": "d8k.htm",
}
VERDICT = {
    "item_codes": ["Item 5.02"], "category": "depart_dirigeant",
    "materiality": True, "summary": "Départ du directeur financier.",
}


@pytest.fixture
def sec(monkeypatch):
    """Remplace la couche SEC + LLM de 04c, et compte ce qui a réellement été
    téléchargé et classifié."""
    compteurs = {"telechargements": 0, "classifications": 0}
    sft = module_04c.sft

    monkeypatch.setattr(sft, "fetch_submissions_strict", lambda cik: [FILING])
    monkeypatch.setattr(sft, "filter_filings", lambda subs, **kw: list(subs))
    monkeypatch.setattr(sft, "filing_document_url", lambda *a: "https://sec.example/d8k.htm")

    def faux_fetch(url, form=None):
        compteurs["telechargements"] += 1
        return ("Item 5.02 Departure of Officers.", "brut")

    def faux_classify(prompt, max_tokens=500):
        compteurs["classifications"] += 1
        return {"category": "depart_dirigeant", "materiality": True,
                "summary": "Départ du directeur financier."}

    monkeypatch.setattr(sft, "fetch_filing_text", faux_fetch)
    monkeypatch.setattr(sft, "analyser_texte_mistral", faux_classify)
    return compteurs


def test_un_8k_deja_classifie_n_est_ni_retelecharge_ni_reanalyse(sec, tmp_path):
    cache = {}
    rows, hits = module_04c.process_ticker_8k(
        "MMM", "0000066740", [("2023-01-01", "2023-12-31")], cache, tmp_path)
    assert hits == 0
    assert sec == {"telechargements": 1, "classifications": 1}
    assert rows[0]["category"] == "depart_dirigeant"

    # Second run : le cache relu depuis le disque, comme au démarrage réel.
    relu = module_04c.load_llm_cache(tmp_path)
    rows2, hits2 = module_04c.process_ticker_8k(
        "MMM", "0000066740", [("2023-01-01", "2023-12-31")], relu, tmp_path)

    assert hits2 == 1
    assert sec == {"telechargements": 1, "classifications": 1}, "aucun nouvel appel attendu"
    assert rows2[0]["category"] == "depart_dirigeant"
    assert rows2[0]["materiality"] is True
    assert rows2[0]["from_cache"] is True


def test_un_verdict_manquant_n_est_pas_memorise(sec, tmp_path, monkeypatch):
    """Quota Mistral épuisé -> category="non_evalue". Le mémoriser gèlerait
    définitivement le trou : ce 8-K ne serait plus jamais reproposé."""
    monkeypatch.setattr(module_04c.sft, "analyser_texte_mistral", lambda *a, **k: None)

    cache = {}
    rows, _ = module_04c.process_ticker_8k(
        "MMM", "0000066740", [("2023-01-01", "2023-12-31")], cache, tmp_path)

    assert rows[0]["category"] == "non_evalue"
    assert cache == {}
    assert not module_04c.llm_cache_path(tmp_path).exists()


def test_no_llm_cache_force_la_reanalyse(sec, tmp_path):
    """llm_cache=None (--no-llm-cache) : ni lecture ni écriture de la mémoire."""
    module_04c.append_llm_cache(tmp_path, {"symbol": "MMM", **FILING, **VERDICT})

    rows, hits = module_04c.process_ticker_8k(
        "MMM", "0000066740", [("2023-01-01", "2023-12-31")], None, tmp_path)

    assert hits == 0
    assert sec == {"telechargements": 1, "classifications": 1}
    assert rows[0]["from_cache"] is False


def test_le_cache_est_ecrit_au_fil_de_l_eau(sec, tmp_path):
    """Un appel payé doit survivre à un Ctrl-C immédiat : l'écriture ne peut
    pas attendre la fin du run."""
    module_04c.process_ticker_8k(
        "MMM", "0000066740", [("2023-01-01", "2023-12-31")], {}, tmp_path)

    lignes = module_04c.llm_cache_path(tmp_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lignes) == 1
    entree = json.loads(lignes[0])
    assert entree["accession_number"] == FILING["accession_number"]
    assert entree["symbol"] == "MMM"


def test_une_ligne_corrompue_ne_perd_pas_tout_le_cache(tmp_path, caplog):
    path = module_04c.llm_cache_path(tmp_path)
    path.write_text(
        json.dumps({"symbol": "MMM", **FILING, **VERDICT}) + "\n"
        + '{"symbol": "AAPL", "accession_num'  # ligne tronquée par un kill
        + "\n" + json.dumps({"symbol": "IBM", **FILING, **VERDICT}) + "\n",
        encoding="utf-8",
    )
    cache = module_04c.load_llm_cache(tmp_path)
    assert set(cache) == {
        module_04c.cache_key("MMM", FILING["accession_number"]),
        module_04c.cache_key("IBM", FILING["accession_number"]),
    }


def test_la_derniere_ecriture_gagne(tmp_path):
    """Une ré-analyse (--no-llm-cache) doit primer sur l'ancien verdict sans
    qu'il faille réécrire le fichier."""
    path = module_04c.llm_cache_path(tmp_path)
    path.write_text(
        json.dumps({"symbol": "MMM", **FILING, **VERDICT}) + "\n"
        + json.dumps({"symbol": "MMM", **FILING, **{**VERDICT, "category": "non_materiel"}}) + "\n",
        encoding="utf-8",
    )
    cache = module_04c.load_llm_cache(tmp_path)
    assert cache[module_04c.cache_key("MMM", FILING["accession_number"])]["category"] == "non_materiel"


@pytest.mark.parametrize("categorie, memorisable", [
    ("depart_dirigeant", True), ("non_materiel", True),
    ("non_evalue", False), (None, False), ("", False),
])
def test_seuls_les_verdicts_reels_sont_memorisables(categorie, memorisable):
    assert module_04c.is_cacheable({"category": categorie}) is memorisable
