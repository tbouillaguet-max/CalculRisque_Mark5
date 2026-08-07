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
    # Le plafond contraint l'ORDRE, pas la position dans la durée : entre deux
    # renforcements, le delta du contrat dérive avec le sous-jacent (gamma) et
    # le moteur ne vend jamais pour se désendetter. Le levier réalisé peut
    # donc dépasser le plafond de quelques dizaines de points -- sans commune
    # mesure avec les 6x du cas non plafonné.
    assert leverage.max() <= 1.5, f"levier max {leverage.max():.2f}x"


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


# --------------------------------------------------------------------------- #
# C3 : redéploiement du cash oisif, tous les jours
# --------------------------------------------------------------------------- #

def test_le_redeploiement_a_lieu_meme_sans_ordre_en_file(panel, evenements, monkeypatch):
    """_deploy_idle_cash vivait dans _execute_pending_orders, qui sort tôt sur
    une file vide : le redéploiement ne se produisait que les jours où un
    ordre était déjà en attente."""
    from backtest import options_engine as oe

    engine = moteur(panel, evenements, {"AAA": "CALL"}, min_deployment_pct=50.0)

    jours_avec_file, jours_appeles = [], []
    vrai_deploy = oe.OptionsBacktestEngine._deploy_idle_cash

    def espion(self, today):
        jours_appeles.append(today)
        if self.pending_orders:
            jours_avec_file.append(today)
        return vrai_deploy(self, today)

    monkeypatch.setattr(oe.OptionsBacktestEngine, "_deploy_idle_cash", espion)
    engine.run()

    assert len(jours_appeles) == len(engine.calendar), "appelé moins d'une fois par jour de bourse"
    assert len(jours_appeles) > len(jours_avec_file), (
        "aucun jour sans ordre en file : le test ne prouve rien"
    )


def test_drawdown_maximal_superieur_a_moins_cent_pourcent(panel, evenements):
    """Critère d'acceptation : un portefeuille non margé ne peut pas perdre
    plus que son capital."""
    engine = moteur(panel, evenements, {"AAA": "CALL", "BBB": "PUT"}, min_deployment_pct=90.0)
    equity_curve, _, _, _ = engine.run()
    drawdown = (equity_curve["nav"] / equity_curve["nav"].cummax() - 1).min() * 100
    assert drawdown > -100.0
    assert (equity_curve["nav"] > 0).all()
    assert (equity_curve["cash"] >= -1e-6).all()
