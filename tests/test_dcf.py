"""A2 : la D&A est bien réintégrée au FCF, et le paramètre mort `interets`
a disparu."""

from __future__ import annotations

import importlib
import inspect

import pandas as pd
import pytest

dcf = importlib.import_module("07_calcul_dcf")


def test_fcf_reintegre_la_da():
    """FCFF = EBIT x (1-t) + D&A - CapEx - ΔBFR."""
    fcf = dcf.calculer_fcf(ebit=1000.0, taux_imposition=0.21, capex=200.0, variation_bfr=50.0, da=300.0)
    assert fcf == pytest.approx(1000 * 0.79 + 300 - 200 - 50)


def test_fcf_sans_da_vaut_l_ancien_calcul():
    """Le défaut da=0 redonne exactement l'ancienne formule : une entreprise
    sans D&A tagué garde le comportement d'avant (conservateur)."""
    avec = dcf.calculer_fcf(ebit=1000.0, taux_imposition=0.21, capex=200.0, variation_bfr=50.0, da=0.0)
    sans = dcf.calculer_fcf(ebit=1000.0, taux_imposition=0.21, capex=200.0, variation_bfr=50.0)
    assert avec == sans == pytest.approx(1000 * 0.79 - 200 - 50)


def test_da_augmente_le_fcf_donc_la_valeur_par_action():
    """L'effet observable du bug : sans D&A, l'entreprise ressort
    mécaniquement moins bien valorisée."""
    faible = dcf.calculer_fcf(ebit=1000.0, taux_imposition=0.21, capex=200.0, variation_bfr=0.0, da=0.0)
    fort = dcf.calculer_fcf(ebit=1000.0, taux_imposition=0.21, capex=200.0, variation_bfr=0.0, da=400.0)
    equity_faible, _ = dcf.calculer_dcf(faible, 0.05, 0.02, 0.10, 5, dette_nette=0.0)
    equity_fort, _ = dcf.calculer_dcf(fort, 0.05, 0.02, 0.10, 5, dette_nette=0.0)
    assert equity_fort > equity_faible


def test_parametre_interets_supprime():
    """Un paramètre qu'aucun appelant ne renseigne et qui vaut donc toujours
    0 est un piège : il ne doit plus exister."""
    assert "interets" not in inspect.signature(dcf.calculer_fcf).parameters


def _ligne(**overrides) -> pd.DataFrame:
    base = {
        "symbol": "AAA", "ebit": 1000.0, "capex": 200.0, "working_capital_change": 50.0,
        "da": 300.0, "net_debt": 0.0, "shares_outstanding": 100.0, "tax_rate": 0.21,
        "close": 50.0, "sector": "Technologie", "year": 2020, "cik": "0000000001",
        "period_type": "FY", "fiscal_year": 2020, "fiscal_quarter": None,
        "filed_date": pd.Timestamp("2021-02-15"),
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_pipeline_branche_bien_la_colonne_da():
    """La colonne `da` de financials.parquet doit atteindre le calcul : sans
    branchement, les deux appels donneraient la même valeur par action."""
    avec_da = dcf.calculer_dcf_par_entreprise(_ligne(da=300.0))
    sans_da = dcf.calculer_dcf_par_entreprise(_ligne(da=0.0))
    assert not avec_da.empty and not sans_da.empty
    assert avec_da["Valeur_par_action_DCF"].iloc[0] > sans_da["Valeur_par_action_DCF"].iloc[0]


def test_da_manquante_traitee_comme_zero_avec_avertissement(caplog):
    """Une D&A manquante ne doit pas faire disparaître la ligne, mais ne doit
    pas non plus passer en silence."""
    with caplog.at_level("WARNING"):
        result = dcf.calculer_dcf_par_entreprise(_ligne(da=None))
    assert len(result) == 1
    assert any("D&A absente" in message for message in caplog.messages)
