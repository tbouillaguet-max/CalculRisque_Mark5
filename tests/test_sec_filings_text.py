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


# --------------------------------------------------------------------------- #
# D2 : historique complet (filings.files)
# --------------------------------------------------------------------------- #

ANCIENS = [
    ("10-K", "2011-02-20", "0001-11-000001", "vieux.htm"),
    ("10-Q", "2012-05-10", "0001-12-000002", "vieux2.htm"),
]


def _routeur(recent, pages: dict, cik: str = "0000000001"):
    """Sert le document principal du CIK, puis les pages d'historique
    référencées -- les pages portent leurs listes colonnaires à la RACINE,
    sans enveloppe "filings", comme l'API réelle."""
    url_principale = sft.SUBMISSIONS_URL.format(cik=cik)

    def faux_get(url):
        if url == url_principale:
            return submissions_json(recent, files=[{"name": nom} for nom in pages])
        for nom, entrees in pages.items():
            if url == sft.SUBMISSIONS_PAGE_URL.format(name=nom):
                return submissions_json(entrees)["filings"]["recent"]
        return None
    return faux_get


def test_les_pages_anciennes_sont_suivies(monkeypatch):
    """recent ne couvre que les ~1000 derniers dépôts, tous formulaires
    confondus : sans les pages anciennes, un backtest 2010-2026 n'a aucun
    filing sur sa première moitié."""
    monkeypatch.setattr(sft, "_get_json", _routeur(ENTREES, {"CIK0000000001-submissions-001.json": ANCIENS}))
    filings = sft.fetch_submissions("0000000001")
    accessions = {f["accession_number"] for f in filings}
    assert "0001-11-000001" in accessions
    assert "0001-22-000002" in accessions


def test_un_filing_ancien_devient_retrouvable(monkeypatch):
    monkeypatch.setattr(sft, "_get_json", _routeur(ENTREES, {"page.json": ANCIENS}))
    trouves = sft.list_company_filings(
        "0000000001", forms=("10-K",), start_date="2011-02-20", end_date="2011-02-20",
    )
    assert [f["accession_number"] for f in trouves] == ["0001-11-000001"]


def test_les_doublons_entre_pages_sont_ecartes(monkeypatch):
    """Les pages et `recent` se recouvrent sur leur borne : un même 8-K ne
    doit pas être classifié deux fois par 04c."""
    monkeypatch.setattr(sft, "_get_json", _routeur(ENTREES, {"page.json": ENTREES[:2] + ANCIENS}))
    filings = sft.fetch_submissions("0000000001")
    accessions = [f["accession_number"] for f in filings]
    assert len(accessions) == len(set(accessions))


def test_page_inaccessible_ne_perd_pas_le_reste(monkeypatch, caplog):
    url_principale = sft.SUBMISSIONS_URL.format(cik="0000000001")

    def faux_get(url):
        if url == url_principale:
            return submissions_json(ENTREES, files=[{"name": "page-cassee.json"}])
        return None   # la page d'historique est indisponible

    monkeypatch.setattr(sft, "_get_json", faux_get)
    with caplog.at_level("WARNING"):
        filings = sft.fetch_submissions("0000000001")
    assert len(filings) == len(ENTREES)
    assert any("Page d'historique" in m for m in caplog.messages)


def test_les_filings_sont_tries_par_date(monkeypatch):
    monkeypatch.setattr(sft, "_get_json", _routeur(ENTREES, {"page.json": ANCIENS}))
    dates = [f["filing_date"] for f in sft.fetch_submissions("0000000001")]
    assert dates == sorted(dates)
