"""Rattachement du secteur aux entreprises RADIÉES (05_calcul_multiples.py).

Sans secteur, une entreprise radiée backfillée est écartée des médianes de 06b
(dropna(subset=["sector"])) : le backfill aurait coûté des milliers de requêtes
SEC sans rien corriger au biais de survivance.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

module_05 = importlib.import_module("05_calcul_multiples")


@pytest.fixture
def univers(tmp_path, monkeypatch):
    """Univers courant (une survivante) + univers complet (la survivante et
    une radiée), tous deux déjà catégorisés par 02."""
    courant = tmp_path / "sp500_universe.csv"
    complet = tmp_path / "sp500_universe_full.csv"
    pd.DataFrame({"RIC": ["SURVIE"], "sector": ["Technologie"]}).to_csv(
        courant, index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "RIC": ["SURVIE", "RADIEE"], "sector": ["Technologie", "Santé"],
    }).to_csv(complet, index=False, encoding="utf-8-sig")

    monkeypatch.setattr(module_05.config, "UNIVERSE_FILE", courant)
    monkeypatch.setattr(module_05.config, "UNIVERSE_FULL_FILE", complet)
    return courant, complet


def test_une_radiee_recupere_son_secteur_depuis_l_univers_complet(univers):
    df = pd.DataFrame({"symbol": ["SURVIE", "RADIEE"]})
    out = module_05.merge_sectors(df)
    assert list(out["sector"]) == ["Technologie", "Santé"]


def test_l_univers_courant_reste_prioritaire(univers):
    """Le secteur d'une entreprise encore membre vient de 01+02, pas du
    superset de 01b qui peut être plus ancien."""
    courant, _ = univers
    pd.DataFrame({"RIC": ["SURVIE"], "sector": ["Santé"]}).to_csv(
        courant, index=False, encoding="utf-8-sig")
    out = module_05.merge_sectors(pd.DataFrame({"symbol": ["SURVIE"]}))
    assert out.loc[0, "sector"] == "Santé"


def test_une_entreprise_sans_secteur_est_signalee(univers, caplog):
    with caplog.at_level("WARNING"):
        out = module_05.merge_sectors(pd.DataFrame({"symbol": ["SURVIE", "INCONNUE"]}))
    assert pd.isna(out.loc[1, "sector"])
    assert "SANS SECTEUR" in caplog.text
    assert "INCONNUE" in caplog.text


def test_sans_univers_complet_le_comportement_est_inchange(tmp_path, monkeypatch):
    courant = tmp_path / "sp500_universe.csv"
    pd.DataFrame({"RIC": ["SURVIE"], "sector": ["Technologie"]}).to_csv(
        courant, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(module_05.config, "UNIVERSE_FILE", courant)
    monkeypatch.setattr(module_05.config, "UNIVERSE_FULL_FILE", tmp_path / "absent.csv")

    out = module_05.merge_sectors(pd.DataFrame({"symbol": ["SURVIE", "RADIEE"]}))
    assert out.loc[0, "sector"] == "Technologie"
    assert pd.isna(out.loc[1, "sector"])


def test_les_classes_d_actions_sont_normalisees(tmp_path, monkeypatch):
    """BRK.B dans l'univers doit rejoindre "BRK B" côté multiples."""
    courant = tmp_path / "sp500_universe.csv"
    pd.DataFrame({"RIC": ["BRK.B"], "sector": ["Services financiers"]}).to_csv(
        courant, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(module_05.config, "UNIVERSE_FILE", courant)
    monkeypatch.setattr(module_05.config, "UNIVERSE_FULL_FILE", tmp_path / "absent.csv")

    attendu = module_05.config.to_ib_symbol("BRK.B")
    out = module_05.merge_sectors(pd.DataFrame({"symbol": [attendu]}))
    assert out.loc[0, "sector"] == "Services financiers"
