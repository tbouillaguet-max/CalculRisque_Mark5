"""Le communiqué joint d'un 8-K (Exhibit 99), lu par Gemini avec le 8-K.

Un 8-K de résultats tient en une phrase -- « the Company issued a press
release, attached as Exhibit 99.1 » -- et c'est le communiqué qui porte
l'information. Mesuré le 2026-09-27 : sur les 3 240 8-K suivis d'une réaction
de cours au-delà de 10 %, 2 598 avaient été jugés non matériels, dont 82 % de
résultats trimestriels -- ni les règles ni Gemini n'en voyaient le contenu.
"""

from __future__ import annotations

import importlib

import pytest
import requests

import sec_filings_text as sft

_c8k = importlib.import_module("04c_recuperation_8k")

INDEX = """<html><body>
<table class="tableFile" summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>1</td><td>8-K</td><td><a href="/ix?doc=/Archives/edgar/data/320193/000032019324000123/aapl-20241031.htm">aapl-20241031.htm</a> iXBRL</td><td>8-K</td><td>32711</td></tr>
<tr><td>3</td><td>DATA</td><td><a href="/Archives/edgar/data/320193/000032019324000123/ex992.htm">ex992.htm</a></td><td>EX-99.2</td><td>5000</td></tr>
<tr><td>2</td><td>PRESS RELEASE</td><td><a href="/Archives/edgar/data/320193/000032019324000123/a8-kex991.htm">a8-kex991.htm</a></td><td>EX-99.1</td><td>123456</td></tr>
<tr><td>4</td><td>SLIDES</td><td><a href="/Archives/edgar/data/320193/000032019324000123/ex993.pdf">ex993.pdf</a></td><td>EX-99.3</td><td>999</td></tr>
<tr><td>5</td><td>GRAPHIC</td><td><a href="/Archives/edgar/data/320193/000032019324000123/logo.jpg">logo.jpg</a></td><td>GRAPHIC</td><td>10</td></tr>
</table></body></html>"""

BASE = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/"


class _Reponse:
    def __init__(self, contenu: str):
        self.content = contenu.encode()


def test_les_pieces_jointes_99_sont_trouvees_dans_l_index_dans_l_ordre(monkeypatch):
    demandes = []
    monkeypatch.setattr(sft.sec_http, "request", lambda url, **k: demandes.append(url) or _Reponse(INDEX))
    assert sft.pieces_jointes_99("0000320193", "0000320193-24-000123") == [
        BASE + "a8-kex991.htm", BASE + "ex992.htm"]   # 99.1 d'abord ; le PDF et l'image écartés
    assert demandes == [BASE + "0000320193-24-000123-index.htm"]


@pytest.mark.parametrize("erreur", [
    sft.sec_http.SecNotFound("index"), sft.sec_http.SecUnavailable("index"),
    requests.exceptions.ConnectionError("coupé"),
])
def test_un_index_illisible_ne_coute_pas_le_8k(monkeypatch, erreur):
    def echec(url, **k):
        raise erreur
    monkeypatch.setattr(sft.sec_http, "request", echec)
    assert sft.pieces_jointes_99("320193", "0000320193-24-000123") == []
    assert sft.texte_piece_jointe("320193", "0000320193-24-000123") is None


def test_le_texte_est_celui_de_la_premiere_piece_lisible(monkeypatch):
    monkeypatch.setattr(sft, "pieces_jointes_99", lambda cik, acc: ["u1", "u2", "u3"])
    textes = {"u1": None, "u2": ("Revenue fell 20% and the Company lowered its outlook.", "debut_document"),
              "u3": ("jamais lu", "debut_document")}
    lus = []
    monkeypatch.setattr(sft, "fetch_filing_text",
                        lambda url, max_chars=None, form=None: lus.append((url, max_chars)) or textes[url])
    assert sft.texte_piece_jointe("320193", "acc", max_chars=500) == textes["u2"][0]
    assert lus == [("u1", 500), ("u2", 500)]


def test_gemini_lit_le_8k_puis_le_communique():
    texte = _c8k.texte_pour_le_modele(
        "FORM 8-K Item 2.02 Results of Operations. The Company issued a press release. SIGNATURES ...",
        "Q3 revenue fell 20%. " + "x " * 10_000)
    corps, communique = texte.split("\n[Communiqué joint (Exhibit 99)]\n")
    assert corps == "Item 2.02 Results of Operations. The Company issued a press release."
    assert communique.startswith("Q3 revenue fell 20%.")
    assert len(communique) == _c8k.MAX_CARACTERES_PIECE_JOINTE
    assert _c8k.texte_pour_le_modele("Item 8.01 Other Events.", None) == "Item 8.01 Other Events."


def test_seuls_les_8k_lus_par_gemini_font_telecharger_leur_communique(tmp_path, monkeypatch):
    """Un 8-K ancien -- classé par règles, sur le 8-K seul -- ne coûte aucune
    requête de plus à la SEC."""
    monkeypatch.setattr(_c8k.sft, "fetch_submissions_strict", lambda cik: [
        {"form": "8-K", "filing_date": "2015-03-02", "accession_number": "ANCIEN", "primary_document": "a.htm"},
        {"form": "8-K", "filing_date": "2026-06-01", "accession_number": "RECENT", "primary_document": "r.htm"},
    ])
    monkeypatch.setattr(_c8k.sft, "fetch_filing_text", lambda url, form=None: (
        "Item 2.02 Results of Operations. The Company issued a press release. Item 9.01 Exhibits.",
        "debut_document"))
    demandes = []
    monkeypatch.setattr(_c8k.sft, "texte_piece_jointe", lambda cik, acc, max_chars: demandes.append(acc) or
                        "Q2 revenue fell 30% and the Company withdrew its guidance.")

    attente = []
    lignes, _ = _c8k.process_ticker_8k("AAPL", "320193", [("2015-01-01", "2026-12-31")], {}, tmp_path,
                                       "2025-08-22", attente)

    assert demandes == ["RECENT"]
    [recent] = attente
    assert "withdrew its guidance" in recent.texte
    # Les règles, elles, n'ont lu que le 8-K : leurs motifs y ont été mesurés.
    assert {l["accession_number"]: l["category"] for l in lignes} == {
        "ANCIEN": "non_materiel", "RECENT": "non_materiel"}
