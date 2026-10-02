"""Recalcul de la valorisation à partir des faits d'un 8-K (recalcul_8k.py).

CE QUE CES TESTS PROTÈGENT. Le module modifie la valeur théorique sur laquelle
le moteur engage du capital. Trois garanties comptent plus que les autres :

  - SANS FAIT, RIEN NE BOUGE : un multiple réappliqué aux mêmes fondamentaux
    redonne exactement le prix stocké par 06b, un DCF sans changement redonne
    celui de 07. Sinon le recalcul déplacerait des signaux que rien n'a touchés.
  - UN FAIT DÉJÀ COMPTÉ NE L'EST PAS DEUX FOIS : un trimestre déjà inclus dans
    le TTM d'origine est ignoré.
  - LES CHIFFRES DE LA DIRECTION NE PASSENT QUE SI ON LE DEMANDE : en mode
    "sectoriel", guidance, économies et contrats sont sans effet.
"""

from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
import pytest

import config
import hierarchie_multiples
import recalcul_8k as r8
from backtest import data_loader

dcf = importlib.import_module("07_calcul_dcf")

M = 1e6


# --------------------------------------------------------------------------- #
# Données synthétiques
# --------------------------------------------------------------------------- #
def fondamentaux_bruts(**changements) -> dict:
    ligne = {
        "symbol": "TST", "period_type": "TTM", "year": 2025, "fiscal_quarter": "Q2",
        "filed_date": pd.Timestamp("2025-08-01"), "period_end": "2025-06-30",
        "revenue": 1000 * M, "ebitda": 200 * M, "ebit": 150 * M, "net_income": 100 * M,
        "net_debt": 300 * M, "shares_outstanding": 100 * M, "da": 50 * M, "capex": 40 * M,
        "working_capital": 0.0, "tax_rate": 0.21,
    }
    ligne.update(changements)
    return ligne


def ligne_combinee(close: float = 100.0, **changements) -> dict:
    """Ligne de 06b cohérente avec fondamentaux_bruts : EV/EBITDA 10x,
    EV/Sales 2x, P/E 15x -> 17, 17, 15 $ par action ; hiérarchie "tiers" :
    médiane de P/E et EV/EBITDA = 16 $."""
    ligne = {
        "symbol": "TST", "sector": "Technologie", "sector_current": "Technologie",
        "period_type": "TTM", "year": 2025, "fiscal_quarter": "Q2",
        "filed_date": pd.Timestamp("2025-08-01"), "close": close,
        "valuation_multiples_per_share": 16.0, "valuation_dcf_per_share": 20.0,
        "valuation_theoretical_per_share": 16.0, "source": "multiples",
        "gap_pct": (16.0 - close) / close * 100, "n_multiples_used": 3, "n_peers": 30.0,
        "price_from_ev_ebitda": 17.0, "price_from_ev_sales": 17.0, "price_from_pe": 15.0,
    }
    ligne.update(changements)
    return ligne


def extraction(faits=(), date="2025-10-20", accession="0000000000-25-000001", **changements) -> dict:
    ligne = {
        "symbol": "TST", "accession_number": accession, "filed_date": date,
        "faits": json.dumps(list(faits)), "fin_periode_resultats": "", "prime_risque": "aucune",
        "veto": False, "direction": "neutre",
    }
    ligne.update(changements)
    return ligne


RESULTATS = [
    {"type": "ca_trimestre", "valeur": 300.0, "citation": "x"},
    {"type": "ca_trimestre_n1", "valeur": 250.0, "citation": "x"},
    {"type": "ebit_trimestre", "valeur": 50.0, "citation": "x"},
    {"type": "ebit_trimestre_n1", "valeur": 40.0, "citation": "x"},
    {"type": "resultat_net_trimestre", "valeur": 35.0, "citation": "x"},
    {"type": "resultat_net_trimestre_n1", "valeur": 28.0, "citation": "x"},
]


def cours_constants(close: float, symbol: str = "TST") -> r8.CoursQuotidiens:
    dates = pd.bdate_range("2025-07-01", "2026-03-31")
    return r8.CoursQuotidiens(pd.DataFrame({"symbol": symbol, "date": dates, "close": close}))


def ajuster(historique_lignes, extractions_lignes, reglages, close=100.0, multiples=None, source="combinee"):
    multiples = multiples if multiples is not None else pd.DataFrame([fondamentaux_bruts()])
    return r8.ajuster_historique(
        pd.DataFrame(historique_lignes), r8.preparer_fondamentaux(multiples),
        pd.DataFrame(extractions_lignes), cours_constants(close), reglages, source=source)


# --------------------------------------------------------------------------- #
# Sans fait, rien ne bouge
# --------------------------------------------------------------------------- #
def test_le_dcf_a_croissance_constante_reproduit_celui_de_07():
    valeur_07, details = dcf.calculer_dcf(100.0, 0.07, 0.03, 0.10, 5, dette_nette=0)
    croissances = r8.trajectoire_croissance(0.07, None, 5)
    assert r8.valeur_entreprise_dcf(100.0, croissances, 0.03, 0.10) == pytest.approx(details["Enterprise_Value"])


def test_les_multiples_reappliques_aux_memes_fondamentaux_redonnent_les_prix_stockes():
    f = r8.Fondamentaux.depuis_ligne(fondamentaux_bruts())
    prix = r8.prix_implicites(ligne_combinee(), f, f)
    assert prix == pytest.approx({"EV/EBITDA": 17.0, "EV/Sales": 17.0, "P/E": 15.0})


def test_sans_fait_aucune_ligne_ajustee():
    ajustees, vetos = ajuster([ligne_combinee()], [extraction()], r8.Reglages("renforcer"))
    assert ajustees.empty and vetos.empty


def test_le_mode_veto_ne_recalcule_rien():
    ajustees, _ = ajuster([ligne_combinee()],
                          [extraction(RESULTATS, fin_periode_resultats="2025-09-30")], r8.Reglages("veto"))
    assert ajustees.empty


# --------------------------------------------------------------------------- #
# Application des faits
# --------------------------------------------------------------------------- #
def test_des_resultats_trimestriels_font_glisser_le_ttm():
    f = r8.Fondamentaux.depuis_ligne(fondamentaux_bruts())
    nouveau, postes = r8.appliquer_faits(f, r8.faits_par_type(RESULTATS), "2025-09-30")
    assert nouveau.revenue == pytest.approx(1050 * M)
    assert nouveau.ebit == pytest.approx(160 * M)
    assert nouveau.ebitda == pytest.approx(210 * M)
    assert nouveau.net_income == pytest.approx(107 * M)
    assert set(postes) == {"resultats_ca", "resultats_ebit", "resultats_resultat_net"}


def test_un_trimestre_deja_dans_le_ttm_n_est_pas_compte_deux_fois():
    f = r8.Fondamentaux.depuis_ligne(fondamentaux_bruts())
    nouveau, postes = r8.appliquer_faits(f, r8.faits_par_type(RESULTATS), "2025-06-30")
    assert postes == [] and nouveau == f


def test_des_resultats_sans_date_de_cloture_ne_sont_pas_appliques():
    f = r8.Fondamentaux.depuis_ligne(fondamentaux_bruts())
    _, postes = r8.appliquer_faits(f, r8.faits_par_type(RESULTATS), "")
    assert postes == []


def test_une_ligne_annuelle_sans_cloture_l_estime_avant_son_depot():
    f = r8.Fondamentaux.depuis_ligne(fondamentaux_bruts(period_end=None, filed_date=pd.Timestamp("2025-02-20")))
    assert f.period_end == pd.Timestamp("2025-01-21")


def test_une_emission_dilue_et_apporte_de_la_tresorerie():
    f = r8.Fondamentaux.depuis_ligne(fondamentaux_bruts())
    nouveau, postes = r8.appliquer_faits(f, {"actions_emises": 10.0, "produit_emission": 150.0})
    assert nouveau.shares == pytest.approx(110 * M)
    assert nouveau.net_debt == pytest.approx(150 * M)
    assert postes == ["emission"]


def test_une_acquisition_finalisee_ajoute_dette_et_perimetre():
    f = r8.Fondamentaux.depuis_ligne(fondamentaux_bruts())
    nouveau, _ = r8.appliquer_faits(f, {"acquisition_prix_cash": 500.0, "acquisition_ca_cible": 200.0,
                                        "acquisition_ebitda_cible": 40.0})
    assert nouveau.net_debt == pytest.approx(800 * M)
    assert nouveau.revenue == pytest.approx(1200 * M)
    assert nouveau.ebitda == pytest.approx(240 * M)
    # L'EBIT suit dans le rapport EBIT/EBITDA de l'acquéreur (150/200).
    assert nouveau.ebit == pytest.approx(150 * M + 40 * M * 0.75)


def test_une_charge_ponctuelle_touche_la_tresorerie_pas_le_resultat():
    f = r8.Fondamentaux.depuis_ligne(fondamentaux_bruts())
    nouveau, _ = r8.appliquer_faits(f, {"charge_cash_ponctuelle": 60.0})
    assert nouveau.net_debt == pytest.approx(360 * M)
    assert (nouveau.ebit, nouveau.net_income) == (f.ebit, f.net_income)


def test_les_chiffres_de_la_direction_sont_ignores_en_mode_sectoriel():
    f = r8.Fondamentaux.depuis_ligne(fondamentaux_bruts())
    valeurs = {"economies_annuelles": 20.0, "contrat_valeur_totale": 500.0, "contrat_duree_annees": 5.0}
    nouveau, postes = r8.appliquer_faits(f, valeurs, reglages=r8.Reglages("renforcer", "sectoriel"))
    assert postes == [] and nouveau == f


def test_les_chiffres_de_la_direction_sont_ponderes_par_la_prudence():
    f = r8.Fondamentaux.depuis_ligne(fondamentaux_bruts())
    nouveau, postes = r8.appliquer_faits(f, {"economies_annuelles": 20.0},
                                         reglages=r8.Reglages("renforcer", "guidance", 0.5))
    assert postes == ["economies"]
    assert nouveau.ebitda == pytest.approx(210 * M)


def test_la_croissance_implicite_d_une_guidance():
    f = r8.Fondamentaux.depuis_ligne(fondamentaux_bruts())
    assert r8.croissance_guidance({"guidance_ca_bas": 1100.0, "guidance_ca_haut": 1200.0}, f) == pytest.approx(0.15)
    # À défaut de CA, le BPA : 1,10 $ prévu pour 1,00 $ réalisé.
    assert r8.croissance_guidance({"guidance_bpa_bas": 1.10}, f) == pytest.approx(0.10)
    # Hors bornes : plus probablement une erreur d'unité qu'une prévision.
    assert r8.croissance_guidance({"guidance_ca_bas": 5000.0}, f) is None


def test_la_trajectoire_de_guidance_revient_au_secteur_en_fin_de_prevision():
    trajectoire = r8.trajectoire_croissance(0.07, 0.15, 5)
    assert trajectoire[0] == pytest.approx(0.15)
    assert trajectoire[-1] == pytest.approx(0.07)
    assert np.all(np.diff(trajectoire) < 0)


def test_une_prime_de_risque_decote_et_rien_sans_prime():
    hyp = r8.hypotheses_dcf("Technologie", 2025)
    assert r8.facteur_risque(hyp, 0) == 1.0
    assert 0 < r8.facteur_risque(hyp, 75) < 1


# --------------------------------------------------------------------------- #
# Lignes de signal ajustées
# --------------------------------------------------------------------------- #
def test_renforcer_recalcule_la_valeur_a_partir_des_resultats():
    ajustees, _ = ajuster([ligne_combinee(close=10.0)],
                          [extraction(RESULTATS, fin_periode_resultats="2025-09-30")],
                          r8.Reglages("renforcer"), close=10.0)
    assert len(ajustees) == 1
    ligne = ajustees.iloc[0]
    # EV/EBITDA : (10 x 210 - 300) / 100 = 18 ; P/E : 15 x 1,07 = 16,05.
    assert ligne["price_from_ev_ebitda"] == pytest.approx(18.0)
    assert ligne["price_from_pe"] == pytest.approx(16.05)
    attendu = float(hierarchie_multiples.combiner(
        pd.DataFrame([{"EV/EBITDA": 18.0, "EV/Sales": ligne["price_from_ev_sales"], "P/E": 16.05}]),
        config.MULTIPLE_RELIABILITY_TIERS).iloc[0])
    assert ligne["valuation_theoretical_per_share"] == pytest.approx(attendu)
    assert ligne["gap_pct"] == pytest.approx((attendu - 10.0) / 10.0 * 100)
    # Daté du 8-K, valorisé au cours de ce jour, vieillissant avec le 10-Q.
    assert ligne["filed_date"] == pd.Timestamp("2025-10-20")
    assert ligne["age_reference_date"] == pd.Timestamp("2025-08-01")
    assert ligne["close"] == 10.0
    assert ligne["source"] == "multiples"


def test_garder_une_bonne_nouvelle_ne_renforce_pas_la_these():
    """Thèse acheteuse (16 $ pour un cours de 10 $) : de bons résultats
    portent la valeur à 17 $ en "renforcer", la laissent à 16 $ en "garder"."""
    args = ([ligne_combinee(close=10.0)], [extraction(RESULTATS, fin_periode_resultats="2025-09-30")])
    renforcer, _ = ajuster(*args, r8.Reglages("renforcer"), close=10.0)
    garder, _ = ajuster(*args, r8.Reglages("garder"), close=10.0)
    assert renforcer.iloc[0]["valuation_theoretical_per_share"] > 16.0
    assert garder.iloc[0]["valuation_theoretical_per_share"] == pytest.approx(16.0)
    assert garder.iloc[0]["valeur_8k_brute"] == pytest.approx(renforcer.iloc[0]["valeur_8k_brute"])


def test_garder_une_mauvaise_nouvelle_reduit_la_these():
    mauvais = [dict(f, valeur=f["valeur"] * (0.5 if f["type"].endswith("trimestre") else 1.0)) for f in RESULTATS]
    garder, _ = ajuster([ligne_combinee(close=10.0)],
                        [extraction(mauvais, fin_periode_resultats="2025-09-30")],
                        r8.Reglages("garder"), close=10.0)
    assert garder.iloc[0]["valuation_theoretical_per_share"] < 16.0


def test_un_8k_de_veto_n_est_pas_recalcule_mais_signale():
    ajustees, vetos = ajuster([ligne_combinee()],
                              [extraction(RESULTATS, fin_periode_resultats="2025-09-30", veto=True)],
                              r8.Reglages("renforcer"))
    assert ajustees.empty
    assert list(vetos["accession_number"]) == ["0000000000-25-000001"]


def test_un_risque_non_chiffrable_decote_la_valeur():
    ajustees, _ = ajuster([ligne_combinee(close=10.0)], [extraction(prime_risque="moyenne")],
                          r8.Reglages("renforcer"), close=10.0)
    assert len(ajustees) == 1
    assert ajustees.iloc[0]["valuation_theoretical_per_share"] < 16.0
    assert ajustees.iloc[0]["postes_8k"] == "prime_risque"


def test_les_8k_d_une_meme_periode_se_cumulent():
    emission = [{"type": "actions_emises", "valeur": 10.0, "citation": "x"},
                {"type": "produit_emission", "valeur": 100.0, "citation": "x"}]
    ajustees, _ = ajuster(
        [ligne_combinee(close=10.0)],
        [extraction(RESULTATS, fin_periode_resultats="2025-09-30"),
         extraction(emission, date="2025-11-05", accession="0000000000-25-000002")],
        r8.Reglages("renforcer"), close=10.0)
    assert len(ajustees) == 2
    second = ajustees.iloc[1]
    # EBITDA 210 (résultats), dette nette 200, 110 M d'actions.
    assert second["price_from_ev_ebitda"] == pytest.approx((10 * 210 - 200) / 110)


def test_un_nouveau_depot_periodique_remet_les_compteurs_a_zero():
    suivante = ligne_combinee(close=10.0, fiscal_quarter="Q3", filed_date=pd.Timestamp("2025-11-01"))
    multiples = pd.DataFrame([
        fondamentaux_bruts(),
        fondamentaux_bruts(fiscal_quarter="Q3", filed_date=pd.Timestamp("2025-11-01"), period_end="2025-09-30"),
    ])
    ajustees, _ = ajuster(
        [ligne_combinee(close=10.0), suivante],
        [extraction(RESULTATS, fin_periode_resultats="2025-09-30"),
         extraction(prime_risque="faible", date="2025-11-10", accession="0000000000-25-000002")],
        r8.Reglages("renforcer"), close=10.0, multiples=multiples)
    assert len(ajustees) == 2
    # Le second 8-K repart des fondamentaux du Q3, sans les résultats déjà
    # intégrés par le 10-Q : seul le risque le décote.
    assert ajustees.iloc[1]["age_reference_date"] == pd.Timestamp("2025-11-01")
    assert ajustees.iloc[1]["postes_8k"] == "prime_risque"


def test_un_8k_depose_le_jour_du_10q_suivant_lui_cede_la_place():
    suivante = ligne_combinee(fiscal_quarter="Q3", filed_date=pd.Timestamp("2025-10-20"))
    ajustees, _ = ajuster([ligne_combinee(), suivante],
                          [extraction(RESULTATS, fin_periode_resultats="2025-09-30")],
                          r8.Reglages("renforcer"))
    assert ajustees.empty


def test_la_source_dcf_recalcule_par_ecart_au_dcf_stocke():
    ligne = ligne_combinee(close=10.0)
    historique_dcf = {k: ligne[k] for k in ("symbol", "sector", "period_type", "year", "fiscal_quarter",
                                            "filed_date", "close", "valuation_dcf_per_share", "gap_pct")}
    ajustees, _ = ajuster([historique_dcf], [extraction(RESULTATS, fin_periode_resultats="2025-09-30")],
                          r8.Reglages("renforcer"), close=10.0, source="dcf")
    f_base = r8.Fondamentaux.depuis_ligne(fondamentaux_bruts())
    f_nouveau, _ = r8.appliquer_faits(f_base, r8.faits_par_type(RESULTATS), "2025-09-30")
    hyp = r8.hypotheses_dcf("Technologie", 2025)
    attendu = 20.0 + r8.valeur_dcf_par_action(f_nouveau, hyp) - r8.valeur_dcf_par_action(f_base, hyp)
    assert ajustees.iloc[0]["valuation_dcf_per_share"] == pytest.approx(attendu)
    assert ajustees.iloc[0]["gap_pct"] == pytest.approx((attendu - 10.0) / 10.0 * 100)


# --------------------------------------------------------------------------- #
# Péremption et réglages
# --------------------------------------------------------------------------- #
def test_un_signal_ajuste_vieillit_avec_sa_periode_d_origine():
    jour = pd.Timestamp("2025-11-01")
    ajuste = {"published_date": pd.Timestamp("2025-10-20"), "age_reference_date": pd.Timestamp("2025-08-01")}
    assert data_loader.signal_age_days(ajuste, jour) == 92
    assert data_loader.signal_age_days({"published_date": pd.Timestamp("2025-10-20")}, jour) == 12
    assert data_loader.signal_age_days({"published_date": pd.Timestamp("2025-10-20"),
                                        "age_reference_date": pd.NaT}, jour) == 12


@pytest.mark.parametrize("arguments", [
    {"mode": "inconnu"}, {"projections": "inconnu"}, {"prudence": 1.5},
])
def test_un_reglage_invalide_est_refuse(arguments):
    with pytest.raises(ValueError):
        r8.Reglages(**arguments)


def test_le_defaut_ne_change_rien_au_comportement_historique():
    assert config.AJUSTEMENT_8K_MODE == "veto"
    assert not r8.Reglages.depuis_config().actif


def test_les_vetos_rejoignent_les_evenements_materiels_quand_le_recalcul_est_actif(tmp_path, monkeypatch):
    extractions = tmp_path / "extractions_8k.parquet"
    pd.DataFrame([extraction(veto=True)]).to_parquet(extractions)
    monkeypatch.setattr(config, "EXTRACTIONS_8K_FILE", extractions)
    absent = tmp_path / "absent.parquet"
    assert data_loader.load_material_events_8k(absent, reglages_8k=r8.Reglages("veto")) is None
    evenements = data_loader.load_material_events_8k(absent, reglages_8k=r8.Reglages("garder"))
    assert list(evenements["symbol"]) == ["TST"]
    assert evenements["filed_date"].iloc[0] == pd.Timestamp("2025-10-20")
