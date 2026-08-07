"""Bloc C : profil de risque et comptabilité du moteur d'options."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.options_harness import cours, moteur, serie_bruitee, signaux

N_JOURS = 120


@pytest.fixture
def panel():
    return cours({
        "AAA": serie_bruitee(100.0, N_JOURS, graine=1),
        "BBB": serie_bruitee(80.0, N_JOURS, graine=2),
    })


@pytest.fixture
def evenements():
    return signaux(["AAA", "BBB"], "2020-01-06")


# --------------------------------------------------------------------------- #
# C1 : plafond d'exposition delta-notionnelle
# --------------------------------------------------------------------------- #

def test_le_plafond_de_delta_borne_le_levier(panel, evenements):
    engine = moteur(
        panel, evenements, {"AAA": "CALL", "BBB": "CALL"},
        min_deployment_pct=90.0, max_delta_notional_pct=100.0,
    )
    equity_curve, _, _, _ = engine.run()
    leverage = equity_curve["delta_notional"] / equity_curve["nav"]
    # Une marge de tolérance : le plafond est vérifié avant chaque
    # renforcement, mais le sous-jacent bouge ensuite librement.
    assert leverage.max() <= 1.35, f"levier max {leverage.max():.2f}x"


def test_sans_plafond_le_deploiement_a_90pct_explose_le_levier(panel, evenements):
    """Le comportement d'avant, conservé sous max_delta_notional_pct=0 : c'est
    lui qui produisait les drawdowns extrêmes."""
    engine = moteur(
        panel, evenements, {"AAA": "CALL", "BBB": "CALL"},
        min_deployment_pct=90.0, max_delta_notional_pct=0,
    )
    equity_curve, _, _, _ = engine.run()
    leverage = (equity_curve["delta_notional"] / equity_curve["nav"]).max()
    assert leverage > 2.0, f"levier max {leverage:.2f}x -- le cas dégradé devrait être bien pire"


def test_le_plafond_prime_sur_le_plancher_de_primes(panel, evenements):
    """Quand les deux se contredisent, c'est le plafond de levier qui gagne :
    le NAV reste partiellement en cash plutôt que de porter 8x."""
    engine = moteur(
        panel, evenements, {"AAA": "CALL", "BBB": "CALL"},
        min_deployment_pct=90.0, max_delta_notional_pct=100.0,
    )
    equity_curve, _, _, _ = engine.run()
    exposition_primes = (equity_curve["invested_value"] / equity_curve["nav"]).max()
    assert exposition_primes < 0.90


def test_delta_notional_est_publie_dans_l_equity_curve(panel, evenements):
    engine = moteur(panel, evenements, {"AAA": "CALL"})
    equity_curve, _, _, _ = engine.run()
    assert "delta_notional" in equity_curve.columns
    assert "delta_notional_pct" in equity_curve.columns
    ouvertes = equity_curve[equity_curve["num_positions"] > 0]
    assert (ouvertes["delta_notional"] > 0).all()
    # L'exposition delta dépasse toujours largement les primes décaissées :
    # c'est précisément l'effet de levier que la colonne rend visible.
    assert (ouvertes["delta_notional"] > ouvertes["invested_value"]).all()


def test_drawdown_maximal_superieur_a_moins_cent_pourcent(panel, evenements):
    """Critère d'acceptation : un portefeuille non margé ne peut pas perdre
    plus que son capital."""
    engine = moteur(panel, evenements, {"AAA": "CALL", "BBB": "PUT"}, min_deployment_pct=90.0)
    equity_curve, _, _, _ = engine.run()
    drawdown = (equity_curve["nav"] / equity_curve["nav"].cummax() - 1).min() * 100
    assert drawdown > -100.0
    assert (equity_curve["nav"] > 0).all()
    assert (equity_curve["cash"] >= -1e-6).all()
