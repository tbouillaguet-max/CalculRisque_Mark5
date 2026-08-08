"""B4 : les financières et les foncières sortent du DCF, et restent valorisées
par les multiples sectoriels de 06b."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

import config

module_07 = importlib.import_module("07_calcul_dcf")
module_06b = importlib.import_module("06b_calcul_valorisation_combinee")


def ligne(symbol: str, secteur: str, **overrides) -> dict:
    base = {
        "symbol": symbol, "sector": secteur, "ebit": 1000.0, "capex": 200.0,
        "working_capital_change": 0.0, "da": 300.0, "net_debt": 0.0,
        "shares_outstanding": 100.0, "tax_rate": 0.21, "close": 50.0,
        "year": 2022, "cik": "0000000001", "period_type": "FY",
        "fiscal_year": 2022, "fiscal_quarter": None, "filed_date": pd.Timestamp("2023-02-15"),
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("secteur", ["Banques", "Assurance", "Services financiers", "Immobilier"])
def test_les_secteurs_sans_dcf_sont_ecartes(secteur):
    """Avant, il suffisait qu'OperatingIncomeLoss soit tagué pour qu'un
    chiffre sorte -- alors qu'un FCFF n'a pas de sens sur ces métiers."""
    df = pd.DataFrame([ligne("AAA", secteur)])
    assert module_07.calculer_dcf_par_entreprise(df).empty


def test_les_autres_secteurs_restent_valorises():
    df = pd.DataFrame([ligne("AAA", "Technologie")])
    assert len(module_07.calculer_dcf_par_entreprise(df)) == 1


def test_comptage_des_lignes_ecartees(caplog):
    df = pd.DataFrame([
        ligne("BNK", "Banques"), ligne("INS", "Assurance"),
        ligne("REIT", "Immobilier"), ligne("TECH", "Technologie"),
    ])
    with caplog.at_level("INFO"):
        result = module_07.calculer_dcf_par_entreprise(df)
    assert list(result["Ticker"]) == ["TECH"]
    assert any("SECTORS_SANS_DCF" in m for m in caplog.messages)


def test_la_liste_couvre_le_minimum_demande():
    for secteur in ("Banques", "Assurance", "Services financiers", "Immobilier"):
        assert secteur in config.SECTORS_SANS_DCF
    # Cohérence : chacun de ces secteurs doit avoir au moins un multiple
    # pertinent, sinon l'écarter du DCF le priverait de toute valorisation.
    for secteur in config.SECTORS_SANS_DCF:
        assert config.SECTOR_MULTIPLES.get(secteur), secteur


def test_06b_valorise_une_banque_par_les_multiples_sans_dcf():
    """Le repli doit fonctionner APRÈS le changement : une banque sans ligne
    DCF garde une valorisation théorique, issue du P/E sectoriel."""
    # Assez de pairs pour que la médiane sectorielle soit jugée robuste.
    pairs = pd.DataFrame([
        {
            "symbol": f"BNK{i}", "sector": "Banques", "period_type": "FY",
            "fiscal_year": 2022, "fiscal_quarter": None, "year": 2022,
            "close": 50.0, "shares_outstanding": 100.0, "net_debt": 0.0,
            "ebitda": 900.0, "revenue": 5000.0, "net_income": 500.0,
            "EV/EBITDA": 8.0, "EV/Sales": 1.5, "P/E": 10.0,
            "filed_date": pd.Timestamp("2023-02-15"),
        }
        for i in range(module_06b.MIN_PEERS_PER_SECTOR_YEAR + 1)
    ])

    medianes = module_06b.compute_sector_year_multiples(pairs)
    valorise = module_06b.compute_implied_valuations(pairs, medianes)

    # P/E médian 10 x BPA 5 = 50 par action ; EV/EBITDA et EV/Sales sont
    # écartés pour ce secteur par config.SECTOR_MULTIPLES.
    assert valorise["valuation_multiples_per_share"].iloc[0] == pytest.approx(50.0)
    assert (valorise["n_multiples_used"] == 1).all()

    # Sans DCF du tout, la valorisation théorique reste celle des multiples.
    has_multiples = valorise["valuation_multiples_per_share"].notna()
    theorique = valorise["valuation_multiples_per_share"].where(has_multiples, np.nan)
    assert theorique.notna().all()
