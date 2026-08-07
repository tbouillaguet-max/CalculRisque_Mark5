"""C2 : la référence des stops est figée à la première ouverture, dans les
deux moteurs.

Le bug : le prix de revient (entry_price / entry_premium) est moyenné à
chaque renfort. Adossé au stop, il le faisait DESCENDRE avec le prix à chaque
renfort en baisse -- le stop ne se déclenchait donc pratiquement jamais tant
qu'on moyennait à la baisse, exactement la situation où il protège."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.engine import Position
from backtest.options_engine import OptionPosition
from tests.options_harness import cours, moteur, serie_bruitee, signaux


# --------------------------------------------------------------------------- #
# Moteur actions
# --------------------------------------------------------------------------- #

def test_actions_le_renfort_ne_deplace_pas_le_stop():
    pos = Position("AAA", shares=100.0, entry_price=100.0, entry_date=pd.Timestamp("2020-01-01"),
                   stop_reference_price=100.0)
    # Renfort à 80 : le prix de revient descend à 90, la référence reste 100.
    nouvelles = pos.shares + 100.0
    pos.entry_price = (pos.entry_price * pos.shares + 80.0 * 100.0) / nouvelles
    pos.shares = nouvelles

    assert pos.entry_price == pytest.approx(90.0)
    assert pos.stop_reference_price == pytest.approx(100.0)

    cours_du_jour = 87.0
    # Contre le prix de revient moyenné : -3,3%, sous le seuil de -15% -> pas
    # de stop. Contre la référence figée : -13%, toujours pas... mais le
    # rapport des deux montre bien que le seuil a été déplacé de 10 points.
    assert (cours_du_jour - pos.entry_price) / pos.entry_price * 100 == pytest.approx(-3.33, abs=0.01)
    assert (cours_du_jour - pos.stop_reference_price) / pos.stop_reference_price * 100 == pytest.approx(-13.0)


def test_actions_stop_declenche_sur_la_reference_figee():
    """Bout en bout : deux renforts en baisse, puis une chute qui doit
    déclencher le stop -- ce que le prix de revient moyenné empêchait."""
    import numpy as np

    from backtest.data_loader import PricePanel
    from backtest.engine import BacktestEngine
    from backtest.strategies.base import Strategy

    class ToujoursAAA(Strategy):
        def generate_target_weights(self, signals, current_positions):
            return {"AAA": 1.0}

    # Baisse régulière : le moteur rebalance à chaque événement et renforce.
    closes = [100.0, 96.0, 92.0, 88.0, 84.0, 80.0, 76.0, 72.0, 68.0, 64.0, 60.0, 56.0]
    dates = pd.bdate_range("2020-01-01", periods=len(closes))
    close = pd.DataFrame({"AAA": closes}, index=dates)
    panel = PricePanel(close, close.copy(), close.apply(lambda c: c.last_valid_index()))

    events = pd.DataFrame([
        {"symbol": "AAA", "published_date": d, "fiscal_year": 2019, "sector": "Technologie",
         "close_at_filing": 100.0, "valuation_dcf_per_share": 200.0, "gap_pct": 100.0,
         "period_type": "FY"}
        for d in dates[:-1]
    ])

    engine = BacktestEngine(
        price_panel=panel, signal_events=events, universe_history=None,
        fallback_universe_symbols={"AAA"}, strategy=ToujoursAAA(),
        initial_capital=1_000_000.0, cost_bps=0.0,
        stop_loss_pct=-15.0, take_profit_pct=1000.0, max_positions=5,
        momentum_min_pct=None,
    )
    _, _, trades, _ = engine.run()
    assert not trades.empty, "le stop-loss n'a jamais été déclenché malgré une baisse de 44%"
    assert (trades["exit_reason"] == "stop_loss").any()


# --------------------------------------------------------------------------- #
# Moteur options
# --------------------------------------------------------------------------- #

def test_options_la_reference_de_stop_survit_aux_renforts():
    pos = OptionPosition(
        symbol="AAA", option_type="CALL", strike=100.0, expiry=pd.Timestamp("2021-01-01"),
        contracts=10.0, entry_premium=8.0, entry_date=pd.Timestamp("2020-01-01"),
        vol=0.3, multiplier=100.0, source="simulated",
        entry_spot=100.0, stop_reference_premium=8.0, stop_reference_spot=100.0,
    )
    nouveaux = pos.contracts + 10.0
    pos.entry_premium = (pos.entry_premium * pos.contracts + 4.0 * 10.0) / nouveaux
    pos.contracts = nouveaux

    assert pos.entry_premium == pytest.approx(6.0)     # prix de revient moyenné (P&L)
    assert pos.stop_reference_premium == pytest.approx(8.0)   # seuil intact
    assert pos.stop_reference_spot == pytest.approx(100.0)


@pytest.mark.parametrize("base", ["premium", "underlying"])
def test_options_le_mouvement_ignore_le_prix_de_revient(base):
    """Cœur du correctif : moyenner le prix de revient à la baisse ne doit
    RIEN changer au mouvement mesuré par le stop. Avant, en base "premium",
    diviser entry_premium par deux divisait mécaniquement la perte affichée
    par deux -- et désarmait le stop."""
    panel = cours({"AAA": serie_bruitee(100.0, 60, graine=7)})
    engine = moteur(panel, signaux(["AAA"], "2020-01-06"), {"AAA": "CALL"}, stop_basis=base)
    jour = panel.close.index[30]

    pos = OptionPosition(
        symbol="AAA", option_type="CALL", strike=100.0, expiry=panel.close.index[-1],
        contracts=10.0, entry_premium=8.0, entry_date=panel.close.index[0],
        vol=0.3, multiplier=100.0, source="simulated",
        entry_spot=100.0, stop_reference_premium=8.0, stop_reference_spot=100.0,
    )
    avant = engine._position_move_pct(pos, jour)

    # Renfort massif à la baisse : prix de revient divisé par deux.
    pos.entry_premium = 4.0
    pos.entry_spot = 50.0
    apres = engine._position_move_pct(pos, jour)

    assert avant is not None
    assert apres == pytest.approx(avant), (
        f"base {base} : le mouvement mesuré a changé ({avant} -> {apres}) alors que "
        "seul le prix de revient a bougé"
    )


def test_options_le_roulement_pose_une_nouvelle_reference():
    """Un roulement ferme puis rouvre : nouvelle thèse, donc nouvelle
    référence -- contrairement à un simple renfort, qui n'y touche pas."""
    panel = cours({"AAA": serie_bruitee(100.0, 200, graine=11)})
    engine = moteur(
        panel, signaux(["AAA"], "2020-01-06"), {"AAA": "CALL"},
        target_tenor_days=60, roll_when_days_left=20, stop_loss_pct=-95.0,
        min_deployment_pct=None,
    )
    _, _, trades, _ = engine.run()
    assert (trades["exit_reason"] == "roll").any(), "aucun roulement déclenché"

    # Sans renfort (min_deployment_pct désactivé), la position rouverte a une
    # référence égale à son propre prix de revient : elle vient d'être posée.
    for pos in engine.positions.values():
        assert pos.stop_reference_premium == pytest.approx(pos.entry_premium)
        assert pos.stop_reference_spot == pytest.approx(pos.entry_spot)
