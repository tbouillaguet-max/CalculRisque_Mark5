"""B1 : le repli annuel ne doit jamais servir un cours postérieur à
filed_date."""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

import config

module_05 = importlib.import_module("05_calcul_multiples")


@pytest.fixture
def fichiers(tmp_path, monkeypatch):
    """Redirige config vers un tmp_path : les tests ne doivent jamais lire ni
    écrire les vraies données du pipeline."""
    monkeypatch.setattr(config, "DAILY_PRICES_FILE", tmp_path / "daily.parquet")
    monkeypatch.setattr(config, "PRICES_FILE", tmp_path / "annual.parquet")
    return tmp_path


def ecrire_annuel(path, lignes):
    pd.DataFrame(lignes).to_parquet(path, index=False)


def test_repli_prend_la_cloture_anterieure_a_filed_date(fichiers):
    """Un 10-K FY2022 déposé en mars 2023 doit être valorisé contre la
    clôture du 31/12/2022 -- et surtout PAS contre celle du 31/12/2023, qui
    n'existait pas encore au moment du dépôt."""
    ecrire_annuel(config.PRICES_FILE, [
        {"symbol": "AAA", "year": 2021, "date": pd.Timestamp("2021-12-31"), "close": 80.0},
        {"symbol": "AAA", "year": 2022, "date": pd.Timestamp("2022-12-30"), "close": 100.0},
        {"symbol": "AAA", "year": 2023, "date": pd.Timestamp("2023-12-29"), "close": 150.0},
    ])
    df = pd.DataFrame([{"symbol": "AAA", "year": 2023, "filed_date": "2023-03-01"}])

    result = module_05.match_price_asof(df)
    assert result["close"].iloc[0] == 100.0
    assert result["price_source"].iloc[0] == "annuel_anterieur"


def test_ancien_comportement_aurait_pris_un_cours_futur(fichiers):
    """Contrôle explicite du bug : la jointure sur `year` renvoyait 150 (la
    clôture de fin 2023) pour un dépôt de mars 2023."""
    ecrire_annuel(config.PRICES_FILE, [
        {"symbol": "AAA", "year": 2023, "date": pd.Timestamp("2023-12-29"), "close": 150.0},
        {"symbol": "AAA", "year": 2022, "date": pd.Timestamp("2022-12-30"), "close": 100.0},
    ])
    df = pd.DataFrame([{"symbol": "AAA", "year": 2023, "filed_date": "2023-03-01"}])

    result = module_05.match_price_asof(df)
    assert result["close"].iloc[0] != 150.0, "cours postérieur à la date de dépôt (look-ahead)"


def test_cours_quotidien_prioritaire_sur_le_repli(fichiers):
    daily = pd.DataFrame([
        {"symbol": "AAA", "date": pd.Timestamp("2023-02-27"), "close": 123.0},
        {"symbol": "AAA", "date": pd.Timestamp("2023-03-01"), "close": 124.0},
    ])
    daily.to_parquet(config.DAILY_PRICES_FILE, index=False)
    ecrire_annuel(config.PRICES_FILE, [
        {"symbol": "AAA", "year": 2022, "date": pd.Timestamp("2022-12-30"), "close": 100.0},
    ])
    df = pd.DataFrame([{"symbol": "AAA", "year": 2023, "filed_date": "2023-03-01"}])

    result = module_05.match_price_asof(df)
    assert result["close"].iloc[0] == 124.0
    assert result["price_source"].iloc[0] == "quotidien"


def test_aucune_cloture_anterieure_laisse_la_ligne_sans_cours(fichiers):
    """Plutôt qu'un cours futur : une ligne sans cours est écartée plus loin
    par calculate_multiples, un cours faux se propagerait au signal."""
    ecrire_annuel(config.PRICES_FILE, [
        {"symbol": "AAA", "year": 2023, "date": pd.Timestamp("2023-12-29"), "close": 150.0},
    ])
    df = pd.DataFrame([{"symbol": "AAA", "year": 2020, "filed_date": "2020-03-01"}])

    result = module_05.match_price_asof(df)
    assert pd.isna(result["close"].iloc[0])


def test_avertissement_au_dela_du_seuil_de_repli(fichiers, caplog):
    ecrire_annuel(config.PRICES_FILE, [
        {"symbol": f"S{i}", "year": 2022, "date": pd.Timestamp("2022-12-30"), "close": 100.0}
        for i in range(10)
    ])
    df = pd.DataFrame([{"symbol": f"S{i}", "year": 2023, "filed_date": "2023-03-01"} for i in range(10)])

    with caplog.at_level("WARNING"):
        module_05.match_price_asof(df)
    assert any("repli annuel" in m for m in caplog.messages)


def test_cache_annuel_sans_colonne_date(fichiers, caplog):
    """Rétrocompatibilité : un PRICES_FILE antérieur à l'ajout de `date` est
    daté au 31/12, ce qui ne peut que rendre le cours plus ancien."""
    pd.DataFrame([
        {"symbol": "AAA", "year": 2022, "close": 100.0},
        {"symbol": "AAA", "year": 2023, "close": 150.0},
    ]).to_parquet(config.PRICES_FILE, index=False)
    df = pd.DataFrame([{"symbol": "AAA", "year": 2023, "filed_date": "2023-03-01"}])

    with caplog.at_level("WARNING"):
        result = module_05.match_price_asof(df)
    assert result["close"].iloc[0] == 100.0
    assert any("colonne 'date'" in m for m in caplog.messages)
