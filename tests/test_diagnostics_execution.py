"""C4 et C5 : date de clôture sur radiation, et visibilité du
sous-investissement."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.data_loader import PricePanel
from backtest.engine import BacktestEngine
from backtest.strategies.base import Strategy
from tests.options_harness import cours, moteur, serie_bruitee, signaux


# --------------------------------------------------------------------------- #
# C4 : radiation datée du jour simulé
# --------------------------------------------------------------------------- #

def test_options_la_sortie_sur_radiation_est_datee_du_jour_simule():
    """La position est VALORISÉE au dernier cours connu mais le trade est
    DATÉ du jour simulé -- comme le fait déjà backtest/engine.py. Sinon le
    cash serait crédité à une date antérieure à la journée en cours, et
    holding_days compterait à partir du mauvais jour."""
    n = 80
    closes = serie_bruitee(100.0, n, graine=3)
    # Le titre cesse de coter à mi-parcours : au-delà, plus aucun cours réel.
    closes = closes[: n // 2] + [float("nan")] * (n - n // 2)

    panel = cours({"AAA": closes})
    engine = moteur(panel, signaux(["AAA"], "2020-01-06"), {"AAA": "CALL"}, min_deployment_pct=None)
    _, _, trades, _ = engine.run()

    gaps = trades[trades["exit_reason"] == "data_gap"]
    assert not gaps.empty, "aucune sortie sur disparition des cours"

    last_valid = panel.last_valid_date["AAA"]
    exit_date = gaps["exit_date"].iloc[0]
    assert exit_date > last_valid, (
        f"sortie datée du dernier cours connu ({last_valid.date()}) au lieu du jour simulé"
    )
    # La date de sortie doit être un vrai jour du calendrier simulé.
    assert exit_date in set(engine.calendar)
    assert gaps["holding_days"].iloc[0] == (exit_date - gaps["entry_date"].iloc[0]).days


def test_options_et_actions_datent_la_radiation_pareil():
    """Les deux moteurs doivent produire la même sémantique de sortie."""
    n = 60
    closes = serie_bruitee(100.0, n, graine=5)
    closes = closes[: n // 2] + [float("nan")] * (n - n // 2)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame({"AAA": closes}, index=dates)
    panel_actions = PricePanel(close, close.copy(), close.apply(lambda c: c.last_valid_index()))

    class ToujoursAAA(Strategy):
        def generate_target_weights(self, signals, current_positions):
            return {"AAA": 1.0}

    events = pd.DataFrame([{
        "symbol": "AAA", "published_date": dates[3], "fiscal_year": 2019, "sector": "Technologie",
        "close_at_filing": 100.0, "valuation_dcf_per_share": 200.0, "gap_pct": 100.0,
        "period_type": "FY",
    }])
    engine = BacktestEngine(
        price_panel=panel_actions, signal_events=events, universe_history=None,
        fallback_universe_symbols={"AAA"}, strategy=ToujoursAAA(),
        initial_capital=1_000_000.0, cost_bps=0.0,
        stop_loss_pct=-99.0, take_profit_pct=1000.0, momentum_min_pct=None,
    )
    _, _, trades_actions, _ = engine.run()
    gaps = trades_actions[trades_actions["exit_reason"] == "data_gap"]
    assert not gaps.empty
    assert gaps["exit_date"].iloc[0] > panel_actions.last_valid_date["AAA"]


# --------------------------------------------------------------------------- #
# C5 : ordres tronqués et cash moyen
# --------------------------------------------------------------------------- #

def _moteur_actions(n: int = 60) -> BacktestEngine:
    symbols = [f"S{i}" for i in range(8)]
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame(
        {s: serie_bruitee(100.0 + 10 * i, n, graine=i) for i, s in enumerate(symbols)},
        index=dates,
    )
    panel = PricePanel(close, close.copy(), close.apply(lambda c: c.last_valid_index()))

    class ToutLUnivers(Strategy):
        def generate_target_weights(self, signals, current_positions):
            syms = list(signals["symbol"])
            return {s: 1.0 / len(syms) for s in syms}

    # Un événement par jour et par symbole : rebalancements fréquents, donc
    # nombreux ordres d'achat -- le régime où la troncature se manifeste.
    events = pd.DataFrame([
        {"symbol": s, "published_date": d, "fiscal_year": 2019, "sector": "Technologie",
         "close_at_filing": 100.0, "valuation_dcf_per_share": 300.0, "gap_pct": 200.0,
         "period_type": "FY"}
        for d in dates[2:-2] for s in symbols
    ])
    return BacktestEngine(
        price_panel=panel, signal_events=events, universe_history=None,
        fallback_universe_symbols=set(symbols), strategy=ToutLUnivers(),
        initial_capital=1_000_000.0, cost_bps=10.0,
        stop_loss_pct=-95.0, take_profit_pct=1000.0,
        momentum_min_pct=None,
    )


def test_les_diagnostics_sont_produits():
    engine = _moteur_actions()
    engine.run()
    diagnostics = engine.execution_diagnostics()

    assert set(diagnostics) == {
        "buy_orders_count", "truncated_orders_count", "truncated_orders_pct", "avg_cash_pct",
    }
    assert diagnostics["buy_orders_count"] > 0
    assert 0.0 <= diagnostics["avg_cash_pct"] <= 100.0
    assert diagnostics["truncated_orders_count"] <= diagnostics["buy_orders_count"]


def test_les_ordres_tronques_sont_comptes():
    """Le budget alloué par _rebalance dépasse le cash dès que des positions
    immobilisent du capital sans être vendues : la troncature doit se voir."""
    engine = _moteur_actions()
    engine.run()
    diagnostics = engine.execution_diagnostics()
    assert diagnostics["truncated_orders_count"] > 0, (
        "aucune troncature détectée : le scénario ne reproduit pas le phénomène"
    )


def test_avertissement_au_dela_du_seuil(caplog):
    engine = _moteur_actions()
    engine.run()
    with caplog.at_level("WARNING"):
        diagnostics = engine.execution_diagnostics()
    if diagnostics["truncated_orders_pct"] > 10.0:
        assert any("tronqués faute de cash" in m for m in caplog.messages)


def test_diagnostics_options_exposent_aussi_le_levier():
    panel = cours({"AAA": serie_bruitee(100.0, 90, graine=1)})
    engine = moteur(panel, signaux(["AAA"], "2020-01-06"), {"AAA": "CALL"})
    engine.run()
    diagnostics = engine.execution_diagnostics()

    assert diagnostics["avg_delta_notional_pct"] is not None
    assert diagnostics["max_delta_notional_pct_observed"] >= diagnostics["avg_delta_notional_pct"]
    assert diagnostics["avg_cash_pct"] is not None


def test_diagnostics_sur_un_run_sans_aucun_ordre():
    """Aucun ordre passé : les diagnostics doivent rester définis plutôt que
    de diviser par zéro."""
    panel = cours({"AAA": serie_bruitee(100.0, 30, graine=1)})
    engine = moteur(panel, signaux(["AAA"], "2030-01-06"), {"AAA": "CALL"})
    engine.run()
    diagnostics = engine.execution_diagnostics()
    assert diagnostics["buy_orders_count"] == 0
    assert diagnostics["truncated_orders_pct"] == 0.0
