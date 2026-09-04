"""Composition point-in-time du groupe de pairs sectoriel (sector_history.py) :
reclassements GICS et appartenance à l'indice.
"""

from __future__ import annotations

import importlib
from datetime import date

import pandas as pd
import pytest

import sector_history

module_06b = importlib.import_module("06b_calcul_valorisation_combinee")


# --------------------------------------------------------------------------- #
# Reclassements GICS
# --------------------------------------------------------------------------- #

def test_alphabet_etait_dans_la_technologie_avant_2018():
    """Le cas cité en exemple : la création de Communication Services en
    septembre 2018 a sorti Alphabet de la technologie. Avant cette date, la
    comparer aux médias est un anachronisme."""
    assert sector_history.sector_asof("GOOGL", "Medias", "2015-03-01") == "Technologie"
    assert sector_history.sector_asof("GOOGL", "Medias", "2019-03-01") == "Medias"


def test_la_charniere_est_exactement_a_la_date_d_effet():
    veille = pd.Timestamp("2018-09-27")
    jour_j = pd.Timestamp("2018-09-28")
    assert sector_history.sector_asof("META", "Medias", veille) == "Technologie"
    assert sector_history.sector_asof("META", "Medias", jour_j) == "Medias"


def test_les_operateurs_telecom_n_ont_pas_bouge_en_2018():
    """Communication Services absorbe les médias et l'internet, mais les
    opérateurs y étaient déjà (sous le nom Telecommunication Services)."""
    assert sector_history.sector_asof("T", "Télécommunications", "2015-01-01") == "Télécommunications"


def test_un_mouvement_de_secteur_entier_ne_demande_pas_de_liste_de_tickers():
    """L'immobilier sort des financières en 2016 : la règle porte sur le
    secteur, donc s'applique à une foncière jamais énumérée nulle part."""
    assert sector_history.sector_asof("XYZREIT", "Immobilier", "2014-06-30") == "Services financiers"
    assert sector_history.sector_asof("XYZREIT", "Immobilier", "2017-06-30") == "Immobilier"


def test_les_paiements_sont_repasses_en_technologie_avant_2023():
    assert sector_history.sector_asof("V", "Services financiers", "2020-01-01") == "Technologie"
    assert sector_history.sector_asof("V", "Services financiers", "2024-01-01") == "Services financiers"


def test_deux_remaniements_successifs_se_defont_dans_le_bon_ordre():
    """Une entreprise aujourd'hui en immobilier, valorisée avant 2016, doit
    remonter jusqu'aux financières -- et la règle 2018 (qui ne la concerne
    pas) ne doit pas s'interposer."""
    assert sector_history.sector_asof("PLD", "Immobilier", "2010-01-01") == "Services financiers"


@pytest.mark.parametrize("as_of", [None, pd.NaT, "pas une date"])
def test_une_date_illisible_laisse_le_secteur_actuel(as_of):
    """Inventer un secteur d'époque sur une ligne non datée serait pire que
    de garder l'actuel."""
    assert sector_history.sector_asof("GOOGL", "Medias", as_of) == "Medias"


def test_un_secteur_inconnu_reste_inchange():
    assert sector_history.sector_asof("GOOGL", None, "2015-01-01") is None
    assert sector_history.sector_asof("INCONNU", "Santé", "2015-01-01") == "Santé"


def test_apply_sector_asof_sur_un_dataframe():
    df = pd.DataFrame({
        "symbol": ["GOOGL", "GOOGL", "DIS", "JNJ"],
        "sector": ["Medias", "Medias", "Medias", "Santé"],
        "filed_date": ["2015-02-01", "2021-02-01", "2015-02-01", "2015-02-01"],
    })
    out = sector_history.apply_sector_asof(df)
    assert list(out["sector_asof"]) == ["Technologie", "Medias", "Distribution", "Santé"]


# --------------------------------------------------------------------------- #
# Appartenance à l'indice
# --------------------------------------------------------------------------- #

@pytest.fixture
def history():
    """Trois profils : une survivante, une radiée en 2015, une entrante de
    2020. Ce sont les trois cas qui composent le biais de survivance."""
    return pd.DataFrame({
        "symbol": ["SURVIE", "RADIEE", "ENTRANTE"],
        "ric": ["SURVIE", "RADIEE", "ENTRANTE"],
        "start_date": pd.to_datetime(["2000-01-01", "2005-01-01", "2020-06-01"]),
        "end_date": pd.to_datetime([None, "2015-03-01", None]),
        "is_current_member": [True, False, True],
    })


def test_membres_a_une_date_passee(history):
    assert sector_history.members_asof(history, "2012-01-01") == {"SURVIE", "RADIEE"}
    assert sector_history.members_asof(history, "2021-01-01") == {"SURVIE", "ENTRANTE"}


def test_une_radiee_est_membre_avant_sa_sortie(history):
    index = sector_history.MembershipIndex(history)
    assert index.is_member("RADIEE", pd.Timestamp("2012-01-01"))
    assert not index.is_member("RADIEE", pd.Timestamp("2016-01-01"))


def test_une_entrante_n_est_pas_membre_avant_son_entree(history):
    """L'autre moitié du biais, rarement relevée : une entrée au S&P 500
    récompense un parcours boursier, donc laisser les entrantes peser sur les
    médianes anciennes les pousse vers le haut."""
    index = sector_history.MembershipIndex(history)
    assert not index.is_member("ENTRANTE", pd.Timestamp("2012-01-01"))
    assert index.is_member("ENTRANTE", pd.Timestamp("2021-01-01"))


def test_un_symbole_inconnu_de_l_historique_reste_admis(history):
    """01b ne remonte qu'à ~1996-2000 : écarter par défaut remplacerait un
    biais mesuré par une amputation silencieuse."""
    index = sector_history.MembershipIndex(history)
    assert index.is_member("JAMAIS_VUE", pd.Timestamp("2012-01-01"))


def test_une_ligne_non_datee_n_est_pas_filtree(history):
    index = sector_history.MembershipIndex(history)
    assert index.is_member("RADIEE", pd.NaT)


def test_coverage_report_chiffre_ce_qui_manque(history):
    rapport = sector_history.coverage_report(history, {"SURVIE"}, "2012-01-01")
    assert rapport["members"] == 2
    assert rapport["present"] == 1
    assert rapport["coverage_ratio"] == pytest.approx(0.5)
    assert rapport["missing_symbols"] == ["RADIEE"]


def test_sans_historique_aucune_couverture_n_est_affirmee():
    assert sector_history.coverage_report(None, {"SURVIE"}, "2012-01-01") is None


# --------------------------------------------------------------------------- #
# Effet sur la médiane sectorielle (06b)
# --------------------------------------------------------------------------- #

def _multiples(symbols, pe_values, sector="Technologie"):
    """Millésime 2012 dont les pairs déposent en février, un jour chacun, et
    dont la DERNIÈRE ligne ("CIBLE") dépose en mars : elle voit donc tous les
    autres (compute_pit_sector_multiples ne montre à une ligne que les pairs
    déjà déposés à sa propre date)."""
    symbols = list(symbols) + ["CIBLE"]
    pe_values = list(pe_values) + [99.0]
    dates = [f"2013-02-{1 + i:02d}" for i in range(len(symbols) - 1)] + ["2013-03-15"]
    return pd.DataFrame({
        "symbol": symbols, "sector": [sector] * len(symbols),
        "period_type": ["FY"] * len(symbols), "fiscal_year": [2012] * len(symbols),
        "fiscal_quarter": [None] * len(symbols),
        "filed_date": pd.to_datetime(dates),
        "EV/EBITDA": [None] * len(symbols), "EV/Sales": [None] * len(symbols),
        "P/E": pe_values,
    })


def _historique(symbols, starts, ends):
    return pd.DataFrame({
        "symbol": symbols, "ric": symbols,
        "start_date": pd.to_datetime(starts), "end_date": pd.to_datetime(ends),
        "is_current_member": [e is None for e in ends],
    })


def test_une_entrante_posterieure_ne_pese_plus_sur_une_mediane_ancienne():
    """Sans le filtre d'appartenance, CHERE (entrée en 2020) tire vers le haut
    la médiane du millésime 2012."""
    df = _multiples(["A", "B", "C", "D", "E", "F", "CHERE"], [10, 11, 12, 13, 14, 15, 55])
    history = _historique(
        ["A", "B", "C", "D", "E", "F", "CHERE", "CIBLE"],
        ["2000-01-01"] * 6 + ["2020-01-01"] + ["2000-01-01"],
        [None] * 8,
    )

    sans_filtre = module_06b.compute_pit_sector_multiples(df, None)
    avec_filtre = module_06b.compute_pit_sector_multiples(
        df, sector_history.MembershipIndex(history))

    cible = df.index[df["symbol"] == "CIBLE"][0]
    assert sans_filtre.loc[cible, "P/E_n_peers"] == 7      # tout le monde sauf soi
    assert avec_filtre.loc[cible, "P/E_n_peers"] == 6      # CHERE écartée
    assert avec_filtre.loc[cible, "P/E_median"] < sans_filtre.loc[cible, "P/E_median"]


def test_une_radiee_compte_dans_les_millesimes_ou_elle_etait_membre():
    """Le cœur de la correction : une fois les données backfillées, une
    entreprise radiée en 2015 doit peser sur la médiane 2012."""
    df = _multiples(["A", "B", "C", "D", "E", "RADIEE"], [20, 21, 22, 23, 24, 4])
    history = _historique(
        ["A", "B", "C", "D", "E", "RADIEE", "CIBLE"],
        ["2000-01-01"] * 7,
        [None] * 5 + ["2015-01-01"] + [None],
    )

    resultat = module_06b.compute_pit_sector_multiples(
        df, sector_history.MembershipIndex(history))
    cible = df.index[df["symbol"] == "CIBLE"][0]
    assert resultat.loc[cible, "P/E_n_peers"] == 6

    # La radiée (P/E 4) doit peser sur l'agrégation, pas seulement figurer au
    # compteur. Comparé aux DEUX groupes de pairs plutôt qu'à une constante :
    # le test porte sur la composition du groupe, pas sur la façon de le
    # résumer (cf. config.SECTOR_MULTIPLE_AGGREGATOR, commutable).
    survivantes = pd.Series([20.0, 21.0, 22.0, 23.0, 24.0])
    avec_radiee = pd.Series([20.0, 21.0, 22.0, 23.0, 24.0, 4.0])
    assert resultat.loc[cible, "P/E_median"] == pytest.approx(
        module_06b.aggregate_multiple(avec_radiee))
    assert resultat.loc[cible, "P/E_median"] < module_06b.aggregate_multiple(survivantes)


def test_le_secteur_d_epoque_repartit_les_pairs_autrement():
    """Alphabet valorisée en 2015 doit se retrouver groupée avec la
    technologie, pas avec les médias."""
    df = pd.DataFrame({
        "symbol": ["GOOGL", "MSFT", "DIS"],
        "sector": ["Medias", "Technologie", "Medias"],
        "filed_date": pd.to_datetime(["2015-02-01"] * 3),
    })
    out = module_06b.apply_point_in_time_sectors(df)
    assert list(out["sector"]) == ["Technologie", "Technologie", "Distribution"]
    assert list(out["sector_current"]) == ["Medias", "Technologie", "Medias"]
