"""Look-ahead de la chaîne de VALORISATION (07), et sa correction.

Deux entrées du DCF venaient de l'avenir :

    LA1  le SECTEUR. La colonne `sector` porte la classification GICS
         d'aujourd'hui, et elle pilote le WACC, les taux de croissance et
         l'éligibilité même au DCF. 06b la ramenait déjà au secteur d'époque ;
         07 ne le faisait pas.
    LA2  le TAUX SANS RISQUE. RISK_FREE_RATE_BY_YEAR porte des MOYENNES
         annuelles : actualiser un dépôt de février N au taux moyen de N,
         c'est utiliser un chiffre qui n'existera qu'en décembre.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

import config
import sector_history

module_07 = importlib.import_module("07_calcul_dcf")


def _ligne(symbol: str, secteur: str, filed: str, **overrides) -> dict:
    base = {
        "symbol": symbol, "sector": secteur, "ebit": 1000.0, "capex": 200.0,
        "working_capital_change": 0.0, "da": 300.0, "net_debt": 0.0,
        "shares_outstanding": 100.0, "tax_rate": 0.21, "close": 50.0,
        "year": pd.Timestamp(filed).year - 1, "cik": "0000000001", "period_type": "FY",
        "fiscal_year": pd.Timestamp(filed).year - 1, "fiscal_quarter": None,
        "filed_date": pd.Timestamp(filed),
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# LA1 : secteur d'époque
# --------------------------------------------------------------------------- #
def test_LA1_visa_est_valorisee_avant_son_reclassement_de_2023():
    """Visa est aujourd'hui une financière (GICS a déplacé les paiements en
    mars 2023), donc classée "Services financiers"... mais en 2015 elle était
    en technologie.

    Le point n'est pas l'étiquette : c'est qu'en 2015, un analyste faisait un
    DCF sur Visa. Refuser de la valoriser au motif d'une décision de
    nomenclature prise huit ans plus tard, c'est décider avec l'avenir."""
    avant = pd.DataFrame([_ligne("V", "Services financiers", "2015-11-20")])
    resultat = module_07.calculer_dcf_par_entreprise(avant)

    assert len(resultat) == 1, "Visa reste écartée en 2015 : le secteur d'époque n'est pas appliqué"
    assert resultat["Secteur"].iloc[0] == "Technologie", "secteur d'époque attendu"
    assert resultat["Secteur_actuel"].iloc[0] == "Services financiers"


def test_LA1_une_vraie_banque_reste_ecartee_a_toute_date():
    """Contrôle négatif : la correction ne doit pas rouvrir le DCF aux
    métiers de bilan. JPMorgan n'a jamais été autre chose qu'une banque."""
    for annee in ("2012-02-20", "2019-02-20", "2025-02-20"):
        df = pd.DataFrame([_ligne("JPM", "Banques", annee)])
        assert module_07.calculer_dcf_par_entreprise(df).empty, annee


def test_LA1_le_wacc_suit_le_secteur_d_epoque():
    """Alphabet : "Medias" aujourd'hui (WACC calibré 9%, croissance 5%),
    "Technologie" avant septembre 2018 (10% et 7%). Le DCF d'un dépôt de 2015
    doit utiliser les hypothèses TECHNO."""
    avant = module_07.calculer_dcf_par_entreprise(
        pd.DataFrame([_ligne("GOOGL", "Medias", "2015-02-20")])
    )
    apres = module_07.calculer_dcf_par_entreprise(
        pd.DataFrame([_ligne("GOOGL", "Medias", "2020-02-20")])
    )
    assert avant["Secteur"].iloc[0] == "Technologie"
    assert apres["Secteur"].iloc[0] == "Medias"

    prime_techno = (
        config.SECTOR_DCF_PARAMS["Technologie"]["wacc"] - config.WACC_CALIBRATION_RISK_FREE_RATE
    )
    attendu = config.risk_free_rate_known_at(2015) + prime_techno
    assert avant["WACC"].iloc[0] == pytest.approx(attendu, abs=1e-9)


def test_LA1_sans_date_de_depot_le_secteur_actuel_est_conserve():
    """sector_asof rend le secteur actuel tel quel quand la date est
    indatable : inventer serait pire que ne rien corriger."""
    df = pd.DataFrame([_ligne("GOOGL", "Medias", "2015-02-20", filed_date=pd.NaT)])
    resultat = module_07.calculer_dcf_par_entreprise(df)
    assert resultat["Secteur"].iloc[0] == "Medias"


def test_LA1_le_secteur_d_epoque_est_propage_dans_l_historique():
    """C'est cette colonne que les stratégies de backtest reçoivent : une
    stratégie neutre au secteur doit comparer une entreprise à ses pairs
    D'ALORS, pas à ceux d'aujourd'hui."""
    assert sector_history.sector_asof("V", "Services financiers", pd.Timestamp("2015-06-01")) == "Technologie"
    assert sector_history.sector_asof("V", "Services financiers", pd.Timestamp("2024-06-01")) == "Services financiers"


# --------------------------------------------------------------------------- #
# LA2 : taux sans risque connu à la date
# --------------------------------------------------------------------------- #
def test_LA2_le_taux_retenu_est_celui_de_l_annee_precedente():
    assert config.risk_free_rate_known_at(2020) == config.RISK_FREE_RATE_BY_YEAR[2019]
    assert config.risk_free_rate_known_at(pd.Timestamp("2020-02-14")) == config.RISK_FREE_RATE_BY_YEAR[2019]
    assert config.risk_free_rate_known_at(None) == config.RISK_FREE_RATE
    assert config.risk_free_rate_known_at(pd.NaT) == config.RISK_FREE_RATE
    # Hors table : repli sur la constante, jamais sur un chiffre inventé.
    assert config.risk_free_rate_known_at(1950) == config.RISK_FREE_RATE


def test_LA2_le_biais_de_2020_allait_dans_le_sens_qui_flatte_le_backtest():
    """2020 est le cas d'école : la moyenne annuelle (0,37%) est écrasée par
    l'effondrement de mars, alors qu'en février le T-Bill cotait encore
    ~1,55%. Le taux moyen donnait donc un WACC trop BAS, donc une valeur
    théorique trop HAUTE, donc plus de signaux d'achat -- juste avant le
    krach."""
    contemporain = config.sector_dcf_params("Technologie", 2020, point_in_time=False)
    connu = config.sector_dcf_params("Technologie", 2020)
    assert contemporain["wacc"] < connu["wacc"]


def test_LA2_le_dcf_utilise_le_taux_connu():
    resultat = module_07.calculer_dcf_par_entreprise(
        pd.DataFrame([_ligne("AAA", "Technologie", "2020-02-20")])
    )
    prime = config.SECTOR_DCF_PARAMS["Technologie"]["wacc"] - config.WACC_CALIBRATION_RISK_FREE_RATE
    assert resultat["WACC"].iloc[0] == pytest.approx(
        config.RISK_FREE_RATE_BY_YEAR[2019] + prime, abs=1e-9,
    )


def test_LA2_le_sharpe_garde_le_taux_contemporain():
    """Contrepoint volontaire : une mesure EX-POST doit comparer le rendement
    réalisé au taux sans risque réellement servi sur la même période. Décaler
    d'un an y serait une erreur, pas une correction."""
    from backtest import metrics as metrics_mod

    dates = pd.Series(pd.to_datetime(["2020-03-02", "2020-03-03"]))
    quotidien = metrics_mod._risk_free_daily(dates, config.RISK_FREE_RATE)
    attendu = config.RISK_FREE_RATE_BY_YEAR[2020] / metrics_mod.TRADING_DAYS_PER_YEAR
    assert quotidien.iloc[0] == pytest.approx(attendu)
