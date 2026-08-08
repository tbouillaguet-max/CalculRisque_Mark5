"""Reconstruction de l'EBIT quand l'entreprise ne tague pas
OperatingIncomeLoss, et correction de l'appel cassé de compute_derived dans
04b (qui faisait échouer TOUS les tickers, donc ne produisait jamais de TTM)."""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

import sec_xbrl

module_04 = importlib.import_module("04_recuperation_10k")
module_04b = importlib.import_module("04b_recuperation_10q")


def frame(**colonnes) -> pd.DataFrame:
    return pd.DataFrame([colonnes])


# --------------------------------------------------------------------------- #
# Les quatre voies
# --------------------------------------------------------------------------- #

def test_ebit_direct_reste_prioritaire():
    """Un OperatingIncomeLoss tagué n'est JAMAIS remplacé par une
    reconstruction, même quand les postes de repli sont disponibles."""
    out = module_04.compute_derived(frame(
        ebit=500.0, pretax_income=999.0, interest_expense=999.0,
        gross_profit=999.0, operating_expenses=0.0,
    ))
    assert out["ebit"].iloc[0] == 500.0
    assert out["ebit_source"].iloc[0] == "operating_income"


def test_reconstruction_par_le_resultat_avant_impot():
    """EBIT = résultat avant impôt + charges d'intérêts."""
    out = module_04.compute_derived(frame(ebit=None, pretax_income=480.0, interest_expense=20.0))
    assert out["ebit"].iloc[0] == pytest.approx(500.0)
    assert out["ebit_source"].iloc[0] == "pretax_plus_interest"


def test_interets_manquants_sous_estiment_l_ebit():
    """Traités comme nuls : conservateur pour le DCF, jamais optimiste."""
    out = module_04.compute_derived(frame(ebit=None, pretax_income=480.0))
    assert out["ebit"].iloc[0] == pytest.approx(480.0)
    assert out["ebit_source"].iloc[0] == "pretax_plus_interest"


def test_reconstruction_par_la_marge_brute():
    out = module_04.compute_derived(frame(
        ebit=None, pretax_income=None, gross_profit=700.0, operating_expenses=250.0,
    ))
    assert out["ebit"].iloc[0] == pytest.approx(450.0)
    assert out["ebit_source"].iloc[0] == "gross_profit_less_opex"


def test_reconstruction_par_les_couts_totaux():
    out = module_04.compute_derived(frame(
        ebit=None, pretax_income=None, revenue=1000.0, costs_and_expenses=600.0,
    ))
    assert out["ebit"].iloc[0] == pytest.approx(400.0)
    assert out["ebit_source"].iloc[0] == "revenue_less_costs"


def test_ordre_de_preference_des_voies():
    """Le résultat avant impôt prime sur la marge brute, qui prime sur les
    coûts totaux."""
    out = module_04.compute_derived(frame(
        ebit=None, pretax_income=480.0, interest_expense=20.0,
        gross_profit=700.0, operating_expenses=250.0,
        revenue=1000.0, costs_and_expenses=600.0,
    ))
    assert out["ebit_source"].iloc[0] == "pretax_plus_interest"


def test_aucune_voie_disponible_laisse_l_ebit_vide():
    """Ne rien inventer : 07 écartera la ligne, ce qui est le comportement
    voulu."""
    out = module_04.compute_derived(frame(ebit=None, pretax_income=None, revenue=1000.0))
    assert pd.isna(out["ebit"].iloc[0])
    assert pd.isna(out["ebit_source"].iloc[0])


def test_provenance_deja_connue_est_conservee():
    """04b rappelle compute_derived sur les agrégats TTM : un EBIT reconstruit
    au trimestre ne doit pas être réétiqueté "operating_income"."""
    out = module_04.compute_derived(frame(ebit=440.0, ebit_source="pretax_plus_interest"))
    assert out["ebit_source"].iloc[0] == "pretax_plus_interest"


def test_ebitda_suit_l_ebit_reconstruit():
    out = module_04.compute_derived(frame(ebit=None, pretax_income=480.0, interest_expense=20.0, da=100.0))
    assert out["ebitda"].iloc[0] == pytest.approx(600.0)


def test_cache_ancien_sans_les_nouvelles_colonnes(caplog):
    """Rétrocompatibilité : un financials.parquet antérieur n'a ni
    pretax_income ni gross_profit -- compute_derived doit s'en accommoder."""
    out = module_04.compute_derived(frame(ebit=500.0, da=50.0, net_income=300.0))
    assert out["ebit"].iloc[0] == 500.0
    assert out["ebit_source"].iloc[0] == "operating_income"


# --------------------------------------------------------------------------- #
# Tags partagés
# --------------------------------------------------------------------------- #

def test_les_deux_scripts_partagent_les_memes_tags():
    """Ils en avaient chacun une copie, à synchroniser à la main."""
    assert module_04.XBRL_TAGS is sec_xbrl.XBRL_TAGS
    assert module_04b.XBRL_TAGS is sec_xbrl.XBRL_TAGS


def test_toute_metrique_est_classee_flux_ou_stock():
    assert sec_xbrl.FLOW_METRICS | sec_xbrl.STOCK_METRICS == set(sec_xbrl.XBRL_TAGS)
    assert not (sec_xbrl.FLOW_METRICS & sec_xbrl.STOCK_METRICS)


def test_les_postes_de_reconstruction_sont_des_flux():
    for metrique in ("pretax_income", "gross_profit", "operating_expenses", "costs_and_expenses"):
        assert metrique in sec_xbrl.FLOW_METRICS, metrique


# --------------------------------------------------------------------------- #
# 04b : compute_derived appelé sur le DataFrame, pas ligne par ligne
# --------------------------------------------------------------------------- #

def _facts_sans_operating_income() -> dict:
    """Émetteur qui publie son résultat avant impôt mais jamais
    OperatingIncomeLoss -- le cas qui privait de DCF."""
    periodes = [
        (2022, "Q1", "2022-01-01", "2022-03-31", "2022-05-01"),
        (2022, "Q2", "2022-04-01", "2022-06-30", "2022-08-01"),
        (2022, "Q3", "2022-07-01", "2022-09-30", "2022-11-01"),
        (2022, "FY", "2022-01-01", "2022-12-31", "2023-02-15"),
        (2023, "Q1", "2023-01-01", "2023-03-31", "2023-05-01"),
        (2023, "Q2", "2023-04-01", "2023-06-30", "2023-08-01"),
        (2023, "Q3", "2023-07-01", "2023-09-30", "2023-11-01"),
        (2023, "FY", "2023-01-01", "2023-12-31", "2024-02-15"),
    ]
    pretax, interet, instants = [], [], []
    for fy, fp, start, end, filed in periodes:
        form = "10-K" if fp == "FY" else "10-Q"
        commun = {"fy": fy, "fp": fp, "form": form, "filed": filed}
        pretax.append({"start": start, "end": end, "val": 400.0 if fp == "FY" else 100.0, **commun})
        interet.append({"start": start, "end": end, "val": 40.0 if fp == "FY" else 10.0, **commun})
        instants.append({"end": end, "val": 5000.0, **commun})

    return {"facts": {"us-gaap": {
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest":
            {"units": {"USD": pretax}},
        "InterestExpense": {"units": {"USD": interet}},
        "Assets": {"units": {"USD": instants}},
        "CommonStockSharesOutstanding": {"units": {"shares": instants}},
    }}}


def test_04b_produit_bien_des_trimestres():
    """L'appel `df.apply(compute_derived, axis=1)` passait chaque LIGNE en
    Series ; `series.columns` n'existe pas, donc AttributeError -- avalée par
    le try/except par ticker de main(). TOUS les tickers échouaient et
    financials_ttm.parquet n'était jamais produit."""
    quarterly = module_04b.build_quarterly_table(_facts_sans_operating_income(), "AAA", "0000000001")
    assert not quarterly.empty
    assert len(quarterly) == 8
    assert "net_debt" in quarterly.columns      # preuve que compute_derived a bien tourné


def test_04b_produit_bien_un_ttm():
    quarterly = module_04b.build_quarterly_table(_facts_sans_operating_income(), "AAA", "0000000001")
    ttm = module_04b.compute_ttm(quarterly)
    assert not ttm.empty
    assert (ttm["ebit"] > 0).all()


def test_04b_reconstruit_l_ebit_et_en_garde_la_trace():
    quarterly = module_04b.build_quarterly_table(_facts_sans_operating_income(), "AAA", "0000000001")
    assert (quarterly["ebit_source"] == "pretax_plus_interest").all()

    ttm = module_04b.compute_ttm(quarterly)
    # 4 trimestres à 110 (100 avant impôt + 10 d'intérêts).
    assert ttm["ebit"].iloc[0] == pytest.approx(440.0)
    assert (ttm["ebit_source"] == "pretax_plus_interest").all(), (
        "le TTM réétiquette un EBIT reconstruit comme s'il était tagué directement"
    )


def test_compute_derived_refuse_une_ligne_isolee():
    """Garde-fou explicite : la fonction est vectorisée et prend le DataFrame
    entier. C'est l'oubli qui a cassé 04b."""
    with pytest.raises(AttributeError):
        module_04.compute_derived(pd.Series({"ebit": 100.0}))
