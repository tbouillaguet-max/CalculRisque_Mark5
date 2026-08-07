"""Bloc D : cache des submissions, historique complet, choix déterministe du
filing, extraction par section, robustesse de l'appel LLM."""

from __future__ import annotations

import importlib
import json

import pytest

import sec_filings_text as sft


@pytest.fixture(autouse=True)
def cache_isole(tmp_path, monkeypatch):
    """Aucun test ne doit lire ni écrire le cache réel du pipeline."""
    monkeypatch.setattr(sft, "SUBMISSIONS_CACHE_DIR", tmp_path / "sec_submissions")
    sft._submissions_memory_cache.clear()
    yield
    sft._submissions_memory_cache.clear()


def submissions_json(entrees, files=None) -> dict:
    """Reproduit le format "colonnaire" de l'API submissions."""
    def bloc(items):
        return {
            "form": [e[0] for e in items],
            "filingDate": [e[1] for e in items],
            "accessionNumber": [e[2] for e in items],
            "primaryDocument": [e[3] for e in items],
        }

    payload = {"filings": {"recent": bloc(entrees)}}
    if files is not None:
        payload["filings"]["files"] = files
    return payload


ENTREES = [
    ("8-K", "2022-03-01", "0001-22-000001", "a.htm"),
    ("10-K", "2022-02-15", "0001-22-000002", "b.htm"),
    ("8-K", "2022-06-01", "0001-22-000003", "c.htm"),
    ("4",    "2022-06-02", "0001-22-000004", "d.xml"),
    ("10-Q", "2022-05-05", "0001-22-000005", "e.htm"),
]


# --------------------------------------------------------------------------- #
# D3 : une requête par entreprise
# --------------------------------------------------------------------------- #

def test_fetch_submissions_ne_telecharge_qu_une_fois(monkeypatch):
    appels = []

    def faux_get(url):
        appels.append(url)
        return submissions_json(ENTREES)

    monkeypatch.setattr(sft, "_get_json", faux_get)

    submissions = sft.fetch_submissions("0000000001")
    for _ in range(40):   # 40 fenêtres de recherche, comme 04c
        sft.filter_filings(submissions, forms=("8-K",), start_date="2022-01-01", end_date="2022-12-31")
    sft.fetch_submissions("0000000001")   # ré-interrogation : sert le cache

    assert len(appels) == 1, f"{len(appels)} téléchargements du même JSON"


def test_le_cache_disque_survit_a_un_nouveau_run(monkeypatch):
    appels = []

    def faux_get(url):
        appels.append(url)
        return submissions_json(ENTREES)

    monkeypatch.setattr(sft, "_get_json", faux_get)
    sft.fetch_submissions("0000000001")
    sft._submissions_memory_cache.clear()      # simule un nouveau processus
    sft.fetch_submissions("0000000001")

    assert len(appels) == 1


def test_cache_expire_apres_le_ttl(monkeypatch):
    appels = []
    monkeypatch.setattr(sft, "_get_json", lambda url: appels.append(url) or submissions_json(ENTREES))
    sft.fetch_submissions("0000000001")

    sft._submissions_memory_cache.clear()
    monkeypatch.setattr(sft, "SUBMISSIONS_CACHE_TTL_SECONDS", -1)
    sft.fetch_submissions("0000000001")

    assert len(appels) == 2


def test_filter_filings_filtre_par_formulaire_et_dates():
    filings = sft._normalize_filing_block(submissions_json(ENTREES)["filings"]["recent"])
    resultat = sft.filter_filings(filings, forms=("8-K",), start_date="2022-02-01", end_date="2022-04-01")
    assert [f["accession_number"] for f in resultat] == ["0001-22-000001"]


def test_filter_filings_ignore_les_formulaires_non_demandes():
    filings = sft._normalize_filing_block(submissions_json(ENTREES)["filings"]["recent"])
    formulaires = {f["form"] for f in sft.filter_filings(filings, forms=("10-K", "10-Q"))}
    assert formulaires == {"10-K", "10-Q"}


def test_blocs_de_longueurs_inegales_ne_plantent_pas():
    """Des colonnes dépareillées existent sur des filings anciens : mieux
    vaut perdre la queue que lever un IndexError en plein run."""
    bloc = {
        "form": ["8-K", "10-K"], "filingDate": ["2022-03-01"],
        "accessionNumber": ["0001-22-000001"], "primaryDocument": ["a.htm"],
    }
    assert len(sft._normalize_filing_block(bloc)) == 1


def test_submissions_introuvable_donne_une_liste_vide(monkeypatch):
    monkeypatch.setattr(sft, "_get_json", lambda url: None)
    assert sft.fetch_submissions("0000000009") == []
