"""A1 : OptionSnapshotIndex.find sur les quatre combinaisons de cibles.

Le NameError corrigé ici ne se déclenchait que sur UNE des quatre (strike seul)
et uniquement quand un snapshot réel existait pour le couple (symbole, type) --
d'où un bug latent qui ne se voyait pas tant que la stratégie options ne
demandait pas un strike sans échéance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.data_loader import OptionSnapshotIndex, PricePanel, UniverseResolver, build_benchmark_series


@pytest.fixture
def index() -> OptionSnapshotIndex:
    """Trois contrats CALL cotés le même jour sur AAA, volontairement classés
    différemment selon le critère : le contrat le plus proche de la monnaie
    n'est ni celui du strike 120 ni celui de l'échéance la plus longue."""
    snapshot_date = pd.Timestamp("2020-06-01")
    rows = [
        # strike, moneyness_pct, expiry
        (100.0, 0.0, "2020-09-01"),    # ATM, ~92 jours
        (120.0, 20.0, "2021-06-01"),   # loin de la monnaie, ~365 jours
        (110.0, 10.0, "2020-12-01"),   # intermédiaire, ~183 jours
    ]
    df = pd.DataFrame([
        {
            "symbol": "AAA", "option_type": "CALL", "snapshot_date": snapshot_date,
            "expiry": pd.Timestamp(expiry), "strike": strike, "moneyness_pct": moneyness,
            "bid": 4.0, "ask": 6.0, "implied_vol": 0.25, "delta": 0.5,
            "gamma": 0.01, "vega": 0.2, "theta": -0.03,
            "underlying_spot": 100.0, "multiplier": 100.0,
        }
        for strike, moneyness, expiry in rows
    ])
    return OptionSnapshotIndex(df)


AS_OF = pd.Timestamp("2020-06-02")


def test_find_sans_cible_retient_le_plus_proche_de_la_monnaie(index):
    contract = index.find("AAA", "CALL", AS_OF)
    assert contract is not None
    assert contract["strike"] == 100.0


def test_find_strike_seul(index):
    """Chemin qui levait NameError : un strike cible, aucune échéance cible."""
    contract = index.find("AAA", "CALL", AS_OF, target_strike=118.0)
    assert contract is not None
    assert contract["strike"] == 120.0


def test_find_tenor_seul(index):
    """Sans strike demandé, la monnaie reste le critère prioritaire : à
    échéance égale de pertinence, on ne retient pas un strike arbitraire."""
    contract = index.find("AAA", "CALL", AS_OF, target_tenor_days=360.0)
    assert contract is not None
    assert contract["strike"] == 100.0


def test_find_strike_et_tenor(index):
    contract = index.find("AAA", "CALL", AS_OF, target_strike=109.0, target_tenor_days=180.0)
    assert contract is not None
    assert contract["strike"] == 110.0
    assert contract["expiry"] == pd.Timestamp("2020-12-01")


def test_find_hors_tolerance_retourne_none(index):
    assert index.find("AAA", "CALL", pd.Timestamp("2021-01-01"), tolerance_days=14) is None


def test_find_symbole_inconnu_retourne_none(index):
    assert index.find("ZZZ", "CALL", AS_OF, target_strike=100.0) is None


def test_index_vide_ne_plante_pas():
    empty = OptionSnapshotIndex(pd.DataFrame())
    assert empty.find("AAA", "CALL", AS_OF, target_strike=100.0) is None


# --------------------------------------------------------------------------- #
# C6 : construction de l'indice de référence
# --------------------------------------------------------------------------- #

def _panel(closes: dict[str, list[float]]) -> PricePanel:
    dates = pd.bdate_range("2020-01-01", periods=len(next(iter(closes.values()))))
    close = pd.DataFrame(closes, index=dates)
    last_valid = close.apply(lambda col: col.last_valid_index())
    return PricePanel(close, close.copy(), last_valid)


def test_benchmark_prefere_le_symbole_demande():
    panel = _panel({"SPY": [100.0, 101.0, 102.0], "AAA": [10.0, 9.0, 8.0]})
    series, label = build_benchmark_series(panel, UniverseResolver(None, {"AAA"}), symbol="SPY")
    assert label == "SPY"
    assert list(series) == [100.0, 101.0, 102.0]


def test_benchmark_replie_sur_un_indice_equipondere():
    """Sans SPY, l'indice est la moyenne des rendements quotidiens des seuls
    membres à cette date."""
    panel = _panel({"AAA": [100.0, 110.0, 121.0], "BBB": [50.0, 45.0, 40.5]})
    series, label = build_benchmark_series(panel, UniverseResolver(None, {"AAA", "BBB"}), symbol="SPY")
    assert "équipondéré" in label
    # +10% et -10% le même jour -> rendement moyen nul, indice inchangé.
    assert series.iloc[1] == pytest.approx(1.0)
    assert series.iloc[2] == pytest.approx(1.0)


def test_benchmark_equipondere_respecte_l_univers_point_in_time():
    """Une entreprise sortie de l'indice ne doit plus peser sur le repère
    après sa date de sortie."""
    history = pd.DataFrame([
        {"ric": "AAA", "start_date": pd.Timestamp("2019-01-01"), "end_date": pd.NaT},
        {"ric": "BBB", "start_date": pd.Timestamp("2019-01-01"), "end_date": pd.Timestamp("2020-01-02")},
    ])
    panel = _panel({"AAA": [100.0, 110.0, 121.0], "BBB": [50.0, 25.0, 12.5]})
    series, _ = build_benchmark_series(panel, UniverseResolver(history, set()), symbol="SPY")
    # BBB est sortie avant le 3e jour : seul AAA (+10%) compte ce jour-là.
    assert series.iloc[2] / series.iloc[1] == pytest.approx(1.10)


def test_benchmark_none_si_rien_n_est_calculable():
    panel = _panel({"AAA": [np.nan, np.nan, np.nan]})
    series, label = build_benchmark_series(panel, UniverseResolver(None, set()), symbol="SPY")
    assert series is None and label == "aucun"
