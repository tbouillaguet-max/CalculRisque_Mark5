"""D1 et D6 : extraction par section des 10-K/10-Q, et robustesse du
téléchargement."""

from __future__ import annotations

import pytest
import requests

import sec_filings_text as sft


# Un 10-K réaliste en miniature : page de garde et table des matières
# d'abord, Item 1 (Business) très long, puis les sections qui portent
# réellement ce que 07b cherche. C'est exactement la disposition qui faisait
# que la troncature au début ne voyait jamais les Items 1A/3/7.
GARDE = "UNITED STATES SECURITIES AND EXCHANGE COMMISSION Form 10-K Annual Report " * 40
SOMMAIRE = "TABLE OF CONTENTS Item 1 Business Item 1A Risk Factors Item 3 Legal Proceedings Item 7 MD&A "
ITEM_1 = "ITEM 1. BUSINESS " + ("Nous concevons et vendons des produits. " * 400)
ITEM_1A = "ITEM 1A. RISK FACTORS " + ("Une dépréciation d'actif pourrait survenir. " * 60)
ITEM_3 = "ITEM 3. LEGAL PROCEEDINGS " + ("Un litige matériel est en cours. " * 60)
ITEM_7 = "ITEM 7. MANAGEMENT'S DISCUSSION " + ("La guidance est révisée à la baisse. " * 60)

DOCUMENT_10K = f"<html><body><p>{GARDE}{SOMMAIRE}{ITEM_1}{ITEM_1A}{ITEM_3}{ITEM_7}</p></body></html>".encode()


class _FluxFactice:
    def __init__(self, content: bytes, erreur_apres: int | None = None):
        self._content = content
        self._erreur_apres = erreur_apres

    def iter_content(self, chunk_size=65_536):
        envoyes = 0
        for i in range(0, len(self._content), chunk_size):
            if self._erreur_apres is not None and envoyes >= self._erreur_apres:
                raise requests.exceptions.ChunkedEncodingError("flux coupé")
            morceau = self._content[i:i + chunk_size]
            envoyes += len(morceau)
            yield morceau

    def close(self):
        pass


@pytest.fixture
def sert(monkeypatch):
    def _sert(content, erreur_apres=None):
        monkeypatch.setattr(
            sft.sec_http, "request",
            lambda url, session=None, stream=False, timeout=None: _FluxFactice(content, erreur_apres),
        )
    return _sert


# --------------------------------------------------------------------------- #
# D1 : extraction par section
# --------------------------------------------------------------------------- #

def test_les_sections_de_risque_sont_extraites(sert):
    sert(DOCUMENT_10K)
    texte, mode = sft.fetch_filing_text("https://sec.gov/x.htm", form="10-K")

    assert mode == "sections"
    assert "dépréciation d'actif" in texte
    assert "litige matériel" in texte
    assert "guidance est révisée" in texte


def test_l_ancienne_troncature_ne_voyait_aucune_de_ces_sections(sert):
    """Contrôle explicite du bug : les 15 000 premiers caractères du même
    document ne contiennent que la garde, le sommaire et l'Item 1."""
    sert(DOCUMENT_10K)
    texte_complet, _ = sft.fetch_filing_text("https://sec.gov/x.htm", form="8-K")  # repli début
    debut = texte_complet[:sft.MAX_TEXT_CHARS]

    assert "dépréciation d'actif" not in debut
    assert "litige matériel" not in debut
    assert "Nous concevons et vendons" in debut


def test_le_budget_est_reparti_entre_les_sections(sert):
    """Trouver l'Item 1A ne doit pas consommer tout l'espace au point de faire
    disparaître l'Item 7."""
    sert(DOCUMENT_10K)
    texte, _ = sft.fetch_filing_text("https://sec.gov/x.htm", form="10-K")

    assert len(texte) <= sft.MAX_TEXT_CHARS
    for etiquette in ("Item 1A", "Item 3", "Item 7"):
        assert etiquette in texte


def test_repli_sur_le_debut_si_aucune_section(sert):
    sert(b"<html><body>Un document sans aucun intitule structure.</body></html>")
    texte, mode = sft.fetch_filing_text("https://sec.gov/x.htm", form="10-K")
    assert mode == "debut_document"
    assert "document sans aucun" in texte


def test_les_8k_ne_tentent_pas_l_extraction_par_section(sert):
    """Documents courts : la troncature au début est sans effet, inutile d'y
    chercher des sections."""
    sert(DOCUMENT_10K)
    _, mode = sft.fetch_filing_text("https://sec.gov/x.htm", form="8-K")
    assert mode == "debut_document"


@pytest.mark.parametrize("variante", [
    "ITEM 1A.", "Item 1A.", "item 1a:", "ITEM  1A", "Item 1A —",
])
def test_variantes_d_intitule_reconnues(variante, sert):
    corps = "garde " * 200 + variante + " RISK FACTORS " + "risque cite " * 50
    sert(f"<html><body>{corps}</body></html>".encode())
    _, mode = sft.fetch_filing_text("https://sec.gov/x.htm", form="10-K")
    assert mode == "sections", f"variante non reconnue : {variante!r}"


def test_le_sommaire_ne_prime_pas_sur_le_corps(sert):
    """La table des matières cite les mêmes intitulés : c'est la dernière
    occurrence qui doit être retenue, pas celle du sommaire."""
    sert(DOCUMENT_10K)
    texte, _ = sft.fetch_filing_text("https://sec.gov/x.htm", form="10-K")
    # Si le sommaire l'emportait, on récupérerait l'Item 1 (Business), pas les
    # facteurs de risque.
    assert "dépréciation d'actif" in texte


def test_le_mode_est_remonte_par_get_filing_text_asof(monkeypatch, tmp_path):
    monkeypatch.setattr(sft, "SUBMISSIONS_CACHE_DIR", tmp_path)
    sft._submissions_memory_cache.clear()
    monkeypatch.setattr(sft, "_get_json", lambda url: {"filings": {"recent": {
        "form": ["10-K"], "filingDate": ["2022-02-15"],
        "accessionNumber": ["0001-22-000010"], "primaryDocument": ["a.htm"],
    }}})
    monkeypatch.setattr(
        sft, "fetch_filing_text", lambda url, max_chars=None, form=None: ("texte", "sections"),
    )
    filing = sft.get_filing_text_asof("0000000001", "2022-02-15", forms=("10-K",))
    assert filing["extraction_mode"] == "sections"


# --------------------------------------------------------------------------- #
# D6 : robustesse du téléchargement
# --------------------------------------------------------------------------- #

def test_le_telechargement_est_borne(sert, monkeypatch):
    """Un iXBRL de 30 Mo ne doit pas être intégralement transféré pour en
    garder quelques milliers de caractères."""
    monkeypatch.setattr(sft, "MAX_DOWNLOAD_BYTES", 100_000)
    enorme = b"<html><body>" + b"x" * 5_000_000 + b"</body></html>"
    sert(enorme)
    contenu = sft._download_bounded("https://sec.gov/gros.htm", 100_000)
    assert len(contenu) < 200_000


def test_flux_interrompu_garde_ce_qui_est_arrive(sert, caplog):
    # Assez gros pour couvrir plusieurs morceaux de flux, sinon la coupure
    # n'a jamais l'occasion de se produire.
    sert(b"<html><body>" + b"y" * 300_000 + b"</body></html>", erreur_apres=65_536)
    with caplog.at_level("WARNING"):
        contenu = sft._download_bounded("https://sec.gov/x.htm", sft.MAX_DOWNLOAD_BYTES)
    assert contenu is not None and len(contenu) > 0
    assert any("Flux interrompu" in m for m in caplog.messages)


def test_echec_de_telechargement_donne_none(monkeypatch):
    def echoue(url, session=None, stream=False, timeout=None):
        raise sft.sec_http.SecUnavailable(url)

    monkeypatch.setattr(sft.sec_http, "request", echoue)
    assert sft.fetch_filing_text("https://sec.gov/x.htm", form="10-K") is None


def test_le_parser_lxml_est_prefere():
    """lxml est plusieurs fois plus rapide que html.parser sur un document de
    cette taille ; le repli reste possible s'il n'est pas installé."""
    assert sft._best_parser() in ("lxml", "html.parser")
    try:
        import lxml  # noqa: F401
    except ImportError:
        pytest.skip("lxml non installé dans cet environnement")
    assert sft._best_parser() == "lxml"
