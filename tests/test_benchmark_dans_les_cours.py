"""L'indice de référence doit être RÉCUPÉRÉ sans devenir INVESTISSABLE.

03b_recuperation_cours_quotidiens.py alimente data/prices/daily_prices.parquet,
d'où backtest/data_loader.build_benchmark_series tire la série du benchmark.
Tant que SPY n'y figurait pas, le backtest se rabattait silencieusement sur un
indice équipondéré de l'univers -- un comparateur légitime, mais qui ne répond
pas à la même question que « bat-on le S&P 500 ? ».

Le risque du correctif est l'inverse du bug : faire entrer un ETF dans
l'univers des candidates. Ces tests tiennent les deux bouts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

import config

_RACINE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "recuperation_cours_quotidiens", _RACINE / "03b_recuperation_cours_quotidiens.py"
)
prix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prix)


def _univers() -> pd.DataFrame:
    return pd.DataFrame([
        {"RIC": "AAPL", "Instrument_Name": "Apple Inc", "Country": "United States",
         "Currency": "USD", "Exchange": "Nasdaq", "ib_symbol": "AAPL", "ib_exchange": "NASDAQ"},
        {"RIC": "BRK.B", "Instrument_Name": "Berkshire Hathaway", "Country": "United States",
         "Currency": "USD", "Exchange": "NYSE", "ib_symbol": "BRK B", "ib_exchange": "NYSE"},
    ])


def test_le_benchmark_est_ajoute_aux_tickers_a_recuperer():
    avec = prix.append_benchmark(_univers())
    assert config.BENCHMARK_SYMBOL in set(avec["ib_symbol"])
    assert len(avec) == 3


def test_la_ligne_ajoutee_porte_toutes_les_colonnes_attendues():
    """Une ligne incomplète ferait échouer process_ticker au moment de
    construire l'enregistrement, pas au moment de l'ajouter -- donc loin
    d'ici, et en plein run réseau."""
    avec = prix.append_benchmark(_univers())
    ligne = avec[avec["ib_symbol"] == config.BENCHMARK_SYMBOL].iloc[0]

    for colonne in prix.REQUIRED_COLUMNS | {"ib_symbol", "ib_exchange"}:
        assert pd.notna(ligne[colonne]), f"colonne {colonne} vide"


def test_le_benchmark_se_resout_sur_les_deux_sources():
    """IBKR passe par ib_exchange, Stooq par le RIC : les deux conventions
    doivent tomber juste, sinon le ticker se journalise en trou de couverture
    après avoir consommé ses tentatives réseau."""
    ligne = prix.append_benchmark(_univers()).iloc[-1]

    assert ligne["ib_exchange"] == "ARCA"          # resolve_stock_contract
    assert prix.to_stooq_symbol(ligne["RIC"]) == "spy.us"


def test_ajouter_le_benchmark_deux_fois_ne_le_duplique_pas():
    une_fois = prix.append_benchmark(_univers())
    deux_fois = prix.append_benchmark(une_fois)
    assert len(deux_fois) == len(une_fois)


def test_un_univers_contenant_deja_le_symbole_n_est_pas_modifie():
    deja = _univers()
    deja.loc[len(deja)] = {
        "RIC": "SPY", "Instrument_Name": "déjà là", "Country": "United States",
        "Currency": "USD", "Exchange": "NYSE Arca", "ib_symbol": "SPY", "ib_exchange": "ARCA",
    }
    assert len(prix.append_benchmark(deja)) == len(deja)


@pytest.mark.parametrize("symbole", ["IVV", "QQQ"])
def test_un_autre_indice_de_reference_est_accepte(symbole):
    avec = prix.append_benchmark(_univers(), symbole)
    ligne = avec[avec["ib_symbol"] == symbole].iloc[0]
    assert ligne["Instrument_Name"] == prix.BENCHMARK_INSTRUMENTS[symbole]


def test_un_symbole_inconnu_recoit_un_libelle_de_repli():
    avec = prix.append_benchmark(_univers(), "XYZ")
    assert "XYZ" in avec[avec["ib_symbol"] == "XYZ"].iloc[0]["Instrument_Name"]


def test_une_chaine_vide_n_ajoute_rien():
    assert len(prix.append_benchmark(_univers(), "")) == 2


def test_le_benchmark_n_entre_pas_dans_l_univers_investissable():
    """LE point du correctif. Le fichier de cours et l'univers investissable
    sont deux sources DISTINCTES : data_loader lit les membres dans
    UNIVERSE_HISTORY_FILE / UNIVERSE_FILE, jamais dans daily_prices. Ajouter
    SPY aux cours ne peut donc pas en faire une candidate -- ce test fige cette
    séparation, qui est la seule raison pour laquelle le correctif est sûr."""
    import inspect

    from backtest import data_loader

    for fonction in (data_loader.load_universe_history, data_loader.load_current_universe_symbols):
        source = inspect.getsource(fonction)
        assert "DAILY_PRICES_FILE" not in source, (
            f"{fonction.__name__} lit le fichier de cours : le benchmark deviendrait "
            "investissable, et le correctif de 03b serait dangereux."
        )
