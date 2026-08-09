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


# --------------------------------------------------------------------------- #
# net_debt : ne plus se vider dès qu'une composante manque
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("colonnes,attendu", [
    ({"long_term_debt": 100.0, "short_term_debt": 20.0, "cash": 50.0}, 70.0),
    ({"long_term_debt": 100.0, "short_term_debt": None, "cash": 50.0}, 50.0),
    ({"long_term_debt": 100.0, "short_term_debt": 20.0, "cash": None}, 120.0),
    ({"long_term_debt": None, "short_term_debt": None, "cash": 50.0}, -50.0),
])
def test_net_debt_somme_ce_qui_est_connu(colonnes, attendu):
    """Une seule composante non taguée suffisait à vider net_debt (renseigné
    sur 61% des lignes seulement). Or 07 fait `net_debt or 0` : ces NaN
    devenaient "ni dette ni trésorerie" au calcul -- une colonne vide dans le
    parquet ET une hypothèse silencieuse au DCF."""
    out = module_04.compute_derived(frame(**colonnes))
    assert out["net_debt"].iloc[0] == pytest.approx(attendu)


def test_net_debt_reste_vide_si_rien_n_est_connu():
    """On ne fabrique pas un zéro : sans aucun des trois postes, la valeur est
    réellement inconnue."""
    out = module_04.compute_derived(frame(long_term_debt=None, short_term_debt=None, cash=None))
    assert pd.isna(out["net_debt"].iloc[0])


# --------------------------------------------------------------------------- #
# Diagnostic de couverture
# --------------------------------------------------------------------------- #

def _lignes_couverture() -> pd.DataFrame:
    lignes = []
    for annee in range(2010, 2016):
        # PARTIELLE : rien avant 2013, exploitable ensuite.
        lignes.append({"symbol": "PARTIELLE", "year": annee, "ebit": None if annee < 2013 else 100.0})
        # AUCUNE : jamais d'EBIT.
        lignes.append({"symbol": "AUCUNE", "year": annee, "ebit": None})
        # COMPLETE : aucun trou.
        lignes.append({"symbol": "COMPLETE", "year": annee, "ebit": 80.0})
    return pd.DataFrame(lignes)


def test_le_resume_distingue_manque_partiel_et_total(caplog):
    """Compter les entreprises ayant AU MOINS UNE période manquante affichait
    quasiment tout l'univers comme perdu, alors qu'il ne manquait que les
    exercices les plus anciens."""
    with caplog.at_level("WARNING"):
        module_04.log_coverage(_lignes_couverture())

    resume = next(m for m in caplog.messages if "lignes sans EBIT" in m)
    assert "2 entreprises touchées" in resume       # COMPLETE n'est pas comptée
    assert "1 gardent au moins un exercice" in resume
    assert "1 n'en ont AUCUN" in resume
    assert any("AUCUNE" in m and "COMPLETE" not in m for m in caplog.messages)


def test_le_resume_ventile_par_exercice(caplog):
    """Un déficit concentré sur les exercices anciens est normal (le balisage
    XBRL ne s'est généralisé qu'à partir de ~2011) et ne demande aucune
    action ; sur les années récentes, si."""
    with caplog.at_level("INFO"):
        module_04.log_coverage(_lignes_couverture())
    assert any("par exercice" in m for m in caplog.messages)


def test_une_seule_categorie_pour_la_provenance_absente(caplog):
    """La fusion avec le cache existant mélange pd.NA (lignes de ce run) et
    np.nan (lignes antérieures à la colonne) : value_counts les affichait
    comme deux catégories distinctes."""
    import numpy as np

    df = pd.DataFrame([
        {"symbol": "A", "year": 2020, "ebit": 100.0, "ebit_source": pd.NA},
        {"symbol": "B", "year": 2020, "ebit": 100.0, "ebit_source": np.nan},
        {"symbol": "C", "year": 2020, "ebit": 100.0, "ebit_source": "operating_income"},
    ])
    with caplog.at_level("INFO"):
        module_04.log_coverage(df)

    provenance = next(m for m in caplog.messages if "Provenance de l'EBIT" in m)
    repartition = provenance.split(":", 1)[1]
    assert repartition.count("non renseigné") == 1
    # Les deux libellés que value_counts affichait en catégories distinctes.
    assert "'nan'" not in repartition and "<NA>" not in repartition


def test_aucun_avertissement_quand_la_couverture_est_complete(caplog):
    df = pd.DataFrame([{"symbol": "A", "year": 2020, "ebit": 100.0, "ebit_source": "operating_income"}])
    with caplog.at_level("WARNING"):
        module_04.log_coverage(df)
    assert not any("sans EBIT" in m for m in caplog.messages)


def test_compute_derived_refuse_une_ligne_isolee():
    """Garde-fou explicite : la fonction est vectorisée et prend le DataFrame
    entier. C'est l'oubli qui a cassé 04b."""
    with pytest.raises(AttributeError):
        module_04.compute_derived(pd.Series({"ebit": 100.0}))
