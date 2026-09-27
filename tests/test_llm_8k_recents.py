"""Gemini ne classe que les 8-K récents ; les anciens passent par les règles.

Un 8-K ne sert qu'à périmer un signal encore actionnable. Au-delà de la plus
longue durée de vie d'un signal, il ne touche plus aucune décision : lui payer
un appel au modèle -- 97 500 appels pour l'historique entier, bien au-delà du
quota quotidien du palier gratuit -- n'apportait rien. Voir
config.LLM_8K_FENETRE_JOURS.
"""

from __future__ import annotations

import importlib
import json
from datetime import datetime

import pytest

import config

_04c = importlib.import_module("04c_recuperation_8k")

# Un 8-K que Gemini lirait (Item 8.01) : seule sa date décide ici.
TEXTE = "Item 8.01 Other Events\nThe Company entered into an Agreement and Plan of Merger."
VERDICT_MODELE = {"category": "rachat_actions", "materiality": True, "summary": "verdict du modèle"}


def test_la_fenetre_est_la_plus_longue_duree_de_vie_d_un_signal():
    assert config.LLM_8K_FENETRE_JOURS == max(
        config.BACKTEST_SIGNAL_MAX_AGE_DAYS, *config.BACKTEST_SIGNAL_MAX_AGE_DAYS_BY_PERIOD.values())


def test_la_date_limite_et_le_tri():
    limite = _04c.date_limite_llm(datetime(2026, 9, 26), 400)
    assert limite == "2025-08-22"
    assert _04c.llm_pour("2025-08-22", limite) and _04c.llm_pour("2026-01-05", limite)
    assert not _04c.llm_pour("2025-08-21", limite)
    assert _04c.llm_pour(None, limite), "sans date, le doute profite au modèle"
    assert _04c.date_limite_llm(datetime(2026, 9, 26), 0) is None      # 0 : tout l'historique
    assert _04c.llm_pour("2011-01-01", None)


def _filings(monkeypatch):
    monkeypatch.setattr(_04c.sft, "fetch_submissions_strict", lambda cik: [
        {"form": "8-K", "filing_date": "2015-03-02", "accession_number": "ANCIEN", "primary_document": "a.htm"},
        {"form": "8-K", "filing_date": "2026-06-01", "accession_number": "RECENT", "primary_document": "r.htm"},
    ])
    monkeypatch.setattr(_04c.sft, "fetch_filing_text", lambda url, form=None: (TEXTE, "debut_document"))


def test_un_8k_ancien_est_classe_par_regles_sans_mise_en_attente(tmp_path, monkeypatch):
    _filings(monkeypatch)
    attente = []
    lignes, _ = _04c.process_ticker_8k(
        "AAPL", "320193", [("2015-01-01", "2026-12-31")], llm_cache={}, output_dir=tmp_path,
        limite_llm="2025-08-22", en_attente=attente)
    assert {l["accession_number"]: l["classification_source"] for l in lignes} == {
        "ANCIEN": "regles_document", "RECENT": "regles_document"}
    assert [a.ligne["accession_number"] for a in attente] == ["RECENT"]


def test_seul_le_8k_recent_part_au_modele(tmp_path, monkeypatch):
    _filings(monkeypatch)
    lots = []
    monkeypatch.setattr(_04c.sft, "analyser_documents", lambda documents, *a, **k: lots.append(
        len(documents)) or {i: dict(VERDICT_MODELE) for i in documents})

    attente = []
    lignes, _ = _04c.process_ticker_8k(
        "AAPL", "320193", [("2015-01-01", "2026-12-31")], llm_cache={}, output_dir=tmp_path,
        limite_llm="2025-08-22", en_attente=attente)
    classees = _04c.classer_en_attente(attente, {}, tmp_path)

    sources = {l["accession_number"]: l["classification_source"] for l in lignes}
    sources.update({l["accession_number"]: l["classification_source"] for l in classees})
    assert sources == {"ANCIEN": "regles_document", "RECENT": "gemini"}
    assert lots == [1]


def _memoire(tmp_path, entrees):
    chemin = _04c.llm_cache_path(tmp_path)
    chemin.write_text("".join(json.dumps(e) + "\n" for e in entrees), encoding="utf-8")


def test_la_memoire_ne_rend_au_modele_que_les_verdicts_recents(tmp_path, monkeypatch):
    """Un ancien 8-K classé par règles reste servi par la mémoire, même avec
    une clé ; un récent repart au modèle. Sans limite, les deux repartent."""
    monkeypatch.setattr(_04c.sft, "llm_disponible", lambda: True)
    _memoire(tmp_path, [
        {"symbol": "AAPL", "accession_number": "ANCIEN", "filed_date": "2015-03-02",
         "item_codes": ["Item 8.01"], "classification_source": "regles_document"},
        {"symbol": "AAPL", "accession_number": "RECENT", "filed_date": "2026-06-01",
         "item_codes": ["Item 8.01"], "classification_source": "regles_document"},
        {"symbol": "AAPL", "accession_number": "MODELE", "filed_date": "2026-06-02",
         "item_codes": ["Item 8.01"], "classification_source": "gemini"},
    ])
    avec_limite = _04c.load_llm_cache(tmp_path, "2025-08-22")
    assert set(avec_limite) == {"AAPL:ANCIEN", "AAPL:MODELE"}
    assert set(_04c.load_llm_cache(tmp_path, None)) == {"AAPL:MODELE"}


def test_l_option_du_script_vaut_la_fenetre_de_config():
    import pathlib
    import re
    texte = pathlib.Path(_04c.__file__).read_text(encoding="utf-8")
    assert re.search(r'"--llm-depuis-jours", type=int, default=config\.LLM_8K_FENETRE_JOURS', texte)
    assert "load_llm_cache(args.output_dir, limite_llm)" in texte
    assert "process_ticker_8k(symbol, cik, windows, llm_cache, args.output_dir, limite_llm," in texte
