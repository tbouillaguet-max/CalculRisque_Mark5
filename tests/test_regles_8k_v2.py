"""Règles de classification des 8-K, version 2 : les faux « matériels » mesurés.

Confrontée à la réaction du cours autour de chaque dépôt (99 787 8-K classés le
2026-09-27), la version 1 tirait 9 863 de ses 22 153 verdicts « matériels » de
six pièges -- aussi inertes, pour le cours, qu'un 8-K de routine. Les phrases
ci-dessous sont celles qui les déclenchaient, reprises des dépôts eux-mêmes.
Voir le pavé « POURQUOI LA VERSION 2 » de 04c_recuperation_8k.py.
"""

from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest

import config
from backtest import data_loader

_c8k = importlib.import_module("04c_recuperation_8k")


def classer(texte: str) -> dict:
    return _c8k.classify_8k_par_regles(_c8k.extract_item_codes(texte), texte)


# --------------------------------------------------------------------------- #
# Les six pièges : plus d'alerte
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("texte", [
    # 1. La mention légale des déclarations prospectives.
    "Item 8.01 Other Events. The Company announced the pricing of its offering. This Current Report "
    "contains forward-looking statements within the meaning of the Private Securities Litigation "
    "Reform Act of 1995.",
    # 2. Les clauses de défaut d'un contrat de dette.
    "Item 1.01 Entry into a Material Definitive Agreement. Item 2.03 Creation of a Direct Financial "
    "Obligation. The Credit Agreement contains customary events of default, including certain events "
    "of bankruptcy, insolvency, reorganization, administration or similar proceedings.",
    "Item 1.01 Entry into a Material Definitive Agreement. Events of default include, among others, "
    "nonpayment of principal or interest when due, breach of covenants or other agreements in the "
    "Indenture, defaults in payment of certain other indebtedness and certain events of bankruptcy or "
    "insolvency.",
    # 3. Une émission d'obligations (Item 2.03), matérielle d'office en version 1.
    "Item 2.03 Creation of a Direct Financial Obligation. On May 1, 2024, the Company issued $1.0 billion "
    "aggregate principal amount of 4.500% Senior Notes due 2034. Item 9.01 Financial Statements and Exhibits.",
    # 4. Un rachat d'OBLIGATIONS, et l'étiquette XBRL des co-déposants.
    "Item 8.01 Other Events. On June 3, 2024, the Company announced cash tender offers for certain "
    "outstanding debt securities.",
    "Item 8.01 Other Events. The Company has amended its previously announced tender offer and consent "
    "solicitation in respect of its 5.250% Senior Notes due 2025.",
    "Item 8.01 Other Events. Co-Registrant Form Type 8-K Co-Registrant Written Communications false "
    "Co-Registrant Solicitating Materials false Co-Registrant PreCommencement Tender Offer false",
    # 5. Item 5.02 sans départ de directeur général ni financier.
    "Item 5.02 Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain "
    "Officers; Compensatory Arrangements of Certain Officers. On March 1, 2024, the Board appointed "
    "Jane Doe as Chief Financial Officer, effective April 1, 2024.",
    "Item 5.02 Departure of Directors or Certain Officers. John Smith, Executive Vice President and "
    "President of the Americas segment, will retire effective June 30, 2024.",
    "Item 5.02 Compensatory Arrangements of Certain Officers. If the Chief Executive Officer retires "
    "after age 60, his outstanding equity awards will continue to vest.",
    "Item 5.02 Compensatory Arrangements of Certain Officers. The agreement provides severance if the "
    "Chief Financial Officer's employment is terminated without cause or he resigns for good reason.",
])
def test_les_pieges_de_la_version_1_ne_donnent_plus_l_alerte(texte):
    verdict = classer(texte)
    assert verdict["materiality"] is False, verdict


def test_un_rachat_d_actions_est_note_mais_ne_donne_pas_l_alerte():
    """6. Une bonne nouvelle : réaction moyenne positive, dérive +0,8 % à 60
    séances. La catégorie reste, pour mémoire."""
    verdict = classer("Item 8.01 Other Events. On May 1, 2024, the Board authorized a new share "
                      "repurchase program of up to $5 billion of the Company's common stock.")
    assert (verdict["category"], verdict["materiality"]) == ("rachat_actions", False)
    assert "repurchase program" in verdict["summary"]


def test_un_rachat_d_actions_ne_masque_pas_un_item_materiel():
    """La catégorie sans alerte ne passe qu'après les Items matériels par
    définition : annoncé avec une dépréciation (2.06), le rachat ne la cache pas."""
    verdict = classer("Item 2.06 Material Impairments. Item 8.01 Other Events. The Board authorized "
                      "a new share repurchase program of up to $1 billion.")
    assert (verdict["category"], verdict["materiality"]) == ("autre_materiel", True)


# --------------------------------------------------------------------------- #
# Ce qui portait un vrai signal : toujours détecté
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("texte, attendue", [
    ("Item 1.03 Bankruptcy or Receivership. On May 1, 2024, the Company and certain of its subsidiaries "
     "filed voluntary petitions for relief under Chapter 11 of the United States Bankruptcy Code.",
     "procedure_judiciaire"),
    ("Item 8.01 Other Events. On June 1, 2024, the Company received a subpoena from the Department of "
     "Justice requesting documents relating to its pricing practices.", "procedure_judiciaire"),
    ("Item 8.01 Other Events. Parent commenced a tender offer to purchase all of the outstanding shares "
     "of common stock of the Company for $42.00 per share in cash.", "fusion_acquisition"),
    ("Item 1.01 Entry into a Material Definitive Agreement. On July 8, 2024, the Company entered into "
     "an Agreement and Plan of Merger with Acme Corp.", "fusion_acquisition"),
    ("Item 5.02 Departure of Directors or Certain Officers. On May 3, 2024, John Smith notified the "
     "Board of his decision to resign as Chief Executive Officer, effective June 30, 2024.",
     "depart_dirigeant"),
    ("Item 5.02 Departure of Directors or Certain Officers. On May 3, 2024, Jane Doe stepped down as "
     "Chief Financial Officer of the Company.", "depart_dirigeant"),
    ("Item 2.02 Results of Operations and Financial Condition. The Company lowered its full-year "
     "guidance for fiscal 2024.", "changement_guidance"),
    ("Item 8.01 Other Events. The Company expects to record a goodwill impairment charge of "
     "approximately $1.2 billion in the third quarter.", "autre_materiel"),
])
def test_les_vrais_evenements_restent_detectes(texte, attendue):
    verdict = classer(texte)
    assert (verdict["category"], verdict["materiality"]) == (attendue, True)


def test_le_resume_est_la_phrase_qui_a_decide():
    """Version 1 : la première phrase trouvée AUTOUR du mot-clé, souvent une
    autre (« 2 are copies of the information provided to the ASX... ») --
    invérifiable. Version 2 : la phrase même."""
    verdict = classer("Item 8.01 Other Events. The meeting was held in Denver. On June 1, 2024, the "
                      "Company received a subpoena from the SEC. The Company is cooperating.")
    assert verdict["summary"] == "On June 1, 2024, the Company received a subpoena from the SEC."


def test_une_mention_prospective_n_empeche_pas_de_lire_le_reste():
    """Seule la phrase de formulaire est écartée, pas le document."""
    verdict = classer(
        "Item 8.01 Other Events. On June 1, 2024, a securities class action was filed against the "
        "Company. This report contains forward-looking statements within the meaning of the Private "
        "Securities Litigation Reform Act of 1995.")
    assert (verdict["category"], verdict["materiality"]) == ("procedure_judiciaire", True)
    assert "class action" in verdict["summary"]


def test_chaque_verdict_porte_la_version_des_regles():
    assert classer("Item 9.01 Exhibits.")["version_regles"] == _c8k.VERSION_REGLES
    assert classer("Item 2.06 Material Impairments.")["version_regles"] == _c8k.VERSION_REGLES


def test_l_item_2_03_n_est_plus_materiel_d_office():
    assert "2.03" not in config.MATERIAL_8K_ITEM_CATEGORIES
    assert "2.03" not in config.MATERIAL_8K_ITEM_CODES


# --------------------------------------------------------------------------- #
# Les verdicts d'une version antérieure sont reclassés
# --------------------------------------------------------------------------- #
def _memoire(tmp_path, entrees):
    chemin = _c8k.llm_cache_path(tmp_path)
    chemin.write_text("".join(json.dumps(e) + "\n" for e in entrees), encoding="utf-8")
    return chemin


def test_les_verdicts_par_regles_perimes_sont_ecartes_ceux_de_gemini_gardes(tmp_path, caplog):
    commun = {"symbol": "AAPL", "filed_date": "2015-03-02", "item_codes": ["Item 8.01"]}
    chemin = _memoire(tmp_path, [
        {**commun, "accession_number": "V1", "category": "procedure_judiciaire",
         "classification_source": "regles_document"},                       # sans version : v1
        {**commun, "accession_number": "V1BIS", "category": "autre_materiel",
         "classification_source": "regles_document", "version_regles": 1},
        {**commun, "accession_number": "V2", "category": "non_materiel",
         "classification_source": "regles_document", "version_regles": _c8k.VERSION_REGLES},
        {**commun, "accession_number": "G", "category": "fusion_acquisition",
         "classification_source": "gemini"},
    ])
    with caplog.at_level("INFO"):
        cache = _c8k.load_llm_cache(tmp_path)
    assert set(cache) == {"AAPL:V2", "AAPL:G"}
    assert "2 verdict(s) rendu(s) par une version antérieure des règles" in caplog.text
    relues = [json.loads(l)["accession_number"] for l in chemin.read_text(encoding="utf-8").splitlines()]
    assert relues == ["V2", "G"], "effacés du fichier : le 8-K est retéléchargé et reclassé"


def test_un_8k_au_verdict_perime_est_retelecharge_et_reclasse(tmp_path, monkeypatch):
    telecharges = []
    monkeypatch.setattr(_c8k.sft, "fetch_submissions_strict", lambda cik: [
        {"form": "8-K", "filing_date": "2015-03-02", "accession_number": "V1", "primary_document": "a.htm"}])
    monkeypatch.setattr(_c8k.sft, "fetch_filing_text", lambda url, form=None: telecharges.append(url) or (
        "Item 8.01 Other Events. This report contains forward-looking statements within the meaning of "
        "the Private Securities Litigation Reform Act of 1995.", "debut_document"))
    _memoire(tmp_path, [{"symbol": "AAPL", "accession_number": "V1", "filed_date": "2015-03-02",
                         "item_codes": ["Item 8.01"], "category": "procedure_judiciaire",
                         "materiality": True, "classification_source": "regles_document"}])

    cache = _c8k.load_llm_cache(tmp_path)
    lignes, servis = _c8k.process_ticker_8k("AAPL", "320193", [("2015-01-01", "2015-12-31")], cache, tmp_path)

    assert servis == 0 and len(telecharges) == 1
    assert (lignes[0]["category"], lignes[0]["materiality"], lignes[0]["version_regles"]) == (
        "non_materiel", False, _c8k.VERSION_REGLES)


# --------------------------------------------------------------------------- #
# Le filtre du backtest : un rachat d'actions ne bloque plus
# --------------------------------------------------------------------------- #
def test_le_filtre_du_backtest_ignore_les_rachats_d_actions_meme_juges_materiels(tmp_path):
    """Gemini, ou la version 1 des règles, ont pu juger un rachat « matériel » :
    le filtre l'écarte quand même -- il protège d'un piège, pas d'une bonne
    nouvelle."""
    chemin = tmp_path / "material_events_8k.parquet"
    pd.DataFrame([
        {"symbol": "AAPL", "filed_date": "2024-05-01", "category": "rachat_actions", "materiality": True,
         "item_codes": ["Item 8.01"]},
        {"symbol": "MSFT", "filed_date": "2024-05-02", "category": "fusion_acquisition", "materiality": True,
         "item_codes": ["Item 1.01"]},
        {"symbol": "IBM", "filed_date": "2024-05-03", "category": "non_materiel", "materiality": False,
         "item_codes": ["Item 9.01"]},
    ]).to_parquet(chemin, index=False)

    evenements = data_loader.load_material_events_8k(chemin)

    assert evenements["symbol"].tolist() == ["MSFT"]
