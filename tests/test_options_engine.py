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


# --------------------------------------------------------------------------- #
# E4 : commission de sortie
# --------------------------------------------------------------------------- #

def _position_sans_valeur(engine, panel, expiry_index=-1):
    from backtest.options_engine import OptionPosition

    pos = OptionPosition(
        symbol="AAA", option_type="CALL",
        strike=10_000.0,                       # très loin de la monnaie : prime ~0
        expiry=panel.close.index[expiry_index],
        contracts=1.0, entry_premium=5.0, entry_date=panel.close.index[0],
        vol=0.25, multiplier=100.0, source="simulated",
        entry_spot=100.0, stop_reference_premium=5.0, stop_reference_spot=100.0,
    )
    engine.positions["AAA"] = pos
    return pos


def test_une_vente_qui_couterait_plus_qu_elle_ne_rapporte_n_a_pas_lieu(panel, evenements):
    """L'ancien `max(gross_value - commission, 0)` sortait la position à zéro
    SANS jamais payer la commission : il comptabilisait une vente qui n'avait
    pas lieu."""
    engine = moteur(panel, evenements, {"AAA": "CALL"})
    _position_sans_valeur(engine, panel)
    cash_avant = engine.cash

    engine._close_position("AAA", panel.close.index[10], "stop_loss")

    assert "AAA" in engine.positions, "position sortie alors que la vente n'est pas passée"
    assert engine.cash == pytest.approx(cash_avant)
    assert engine.trades == []


def test_l_expiration_reste_une_sortie_forcee(panel, evenements):
    """À l'échéance il n'y a plus rien à détenir : le contrat est abandonné
    sans frais, produit nul."""
    engine = moteur(panel, evenements, {"AAA": "CALL"})
    _position_sans_valeur(engine, panel)
    cash_avant = engine.cash

    engine._close_position("AAA", panel.close.index[-1], "expiry")

    assert "AAA" not in engine.positions
    assert engine.cash == pytest.approx(cash_avant)      # produit nul, pas de frais payés
    assert engine.trades[-1]["exit_reason"] == "expiry"


def test_la_disparition_des_cours_reste_une_sortie_forcee(panel, evenements):
    """Aucune cotation future à espérer : conserver la position n'aurait
    aucun sens."""
    engine = moteur(panel, evenements, {"AAA": "CALL"})
    _position_sans_valeur(engine, panel)
    engine._close_position("AAA", panel.close.index[10], "data_gap")
    assert "AAA" not in engine.positions


def test_une_vente_rentable_se_fait_normalement(panel, evenements):
    """Le garde-fou ne doit pas bloquer les sorties ordinaires."""
    engine = moteur(panel, evenements, {"AAA": "CALL"})
    from backtest.options_engine import OptionPosition

    engine.positions["AAA"] = OptionPosition(
        symbol="AAA", option_type="CALL", strike=50.0,    # dans la monnaie
        expiry=panel.close.index[-1], contracts=10.0, entry_premium=5.0,
        entry_date=panel.close.index[0], vol=0.25, multiplier=100.0,
        source="simulated", entry_spot=100.0,
        stop_reference_premium=5.0, stop_reference_spot=100.0,
    )
    cash_avant = engine.cash
    engine._close_position("AAA", panel.close.index[10], "take_profit")

    assert "AAA" not in engine.positions
    assert engine.cash > cash_avant
    assert engine.trades[-1]["exit_reason"] == "take_profit"


def test_aucun_trade_ne_perd_plus_que_sa_prime(panel, evenements):
    """Conséquence attendue du plancher : return_pct >= -100% partout."""
    engine = moteur(panel, evenements, {"AAA": "CALL", "BBB": "PUT"}, min_deployment_pct=90.0)
    _, _, trades, _ = engine.run()
    if not trades.empty:
        assert trades["return_pct"].min() >= -100.0


def test_drawdown_maximal_superieur_a_moins_cent_pourcent(panel, evenements):
    """Critère d'acceptation : un portefeuille non margé ne peut pas perdre
    plus que son capital."""
    engine = moteur(panel, evenements, {"AAA": "CALL", "BBB": "PUT"}, min_deployment_pct=90.0)
    equity_curve, _, _, _ = engine.run()
    drawdown = (equity_curve["nav"] / equity_curve["nav"].cummax() - 1).min() * 100
    assert drawdown > -100.0
    assert (equity_curve["nav"] > 0).all()
    assert (equity_curve["cash"] >= -1e-6).all()


# --------------------------------------------------------------------------- #
# Point de décision à roll_when_days_left de l'échéance
# --------------------------------------------------------------------------- #

def _moteur_avec_reexamen(gap_du_second_signal: float):
    """Une entrée sur un écart large, puis un second dépôt qui ramène l'écart
    à `gap_du_second_signal`. L'échéance est courte et le seuil de réexamen
    proche pour que le point de décision tombe dans le calendrier simulé."""
    from backtest.strategies.valuation_gap_options import ValuationGapOptionsStrategy

    panel = cours({"AAA": serie_bruitee(100.0, 200, graine=13)})
    dates = panel.close.index
    evenements = pd.concat([
        signaux(["AAA"], dates[2].strftime("%Y-%m-%d"), gap_pct=80.0),
        signaux(["AAA"], dates[15].strftime("%Y-%m-%d"), gap_pct=gap_du_second_signal),
    ], ignore_index=True)

    return moteur(
        panel, evenements, {"AAA": "CALL"},
        strategy=ValuationGapOptionsStrategy(),
        target_tenor_days=60, roll_when_days_left=20,
        stop_loss_pct=-95.0, take_profit_pct=1000.0, min_deployment_pct=None,
    )


def test_le_reexamen_cloture_une_position_qui_ne_passe_plus_les_filtres():
    """Écart repassé sous le seuil d'entrée : au point de décision la position
    est VENDUE, au lieu d'être portée jusqu'à son expiration."""
    _, _, trades, _ = _moteur_avec_reexamen(gap_du_second_signal=0.0).run()
    assert (trades["exit_reason"] == "signal_lost").any()
    assert not (trades["exit_reason"] == "roll").any()


def test_le_reexamen_roule_une_position_qui_passe_encore_les_filtres():
    """Même scénario, écart toujours au-dessus du seuil : le contrat est
    renouvelé plutôt que liquidé."""
    _, _, trades, _ = _moteur_avec_reexamen(gap_du_second_signal=80.0).run()
    assert (trades["exit_reason"] == "roll").any()
    assert not (trades["exit_reason"] == "signal_lost").any()


def test_les_deux_strategies_achetent_a_deux_ans_avec_reexamen_a_neuf_mois():
    """Câblage config -> CLI -> moteur. Rien ici n'est vérifié par les tests de
    comportement ci-dessus (le banc leur impose des échéances courtes), et une
    échéance qui régresserait silencieusement à 9 mois ne casserait rien."""
    import argparse
    import importlib

    import config
    from backtest.strategies import OPTIONS_STRATEGY_REGISTRY

    resolve = importlib.import_module("10_backtest_options").resolve_engine_settings
    aucune_option = argparse.Namespace()

    for nom, cls in OPTIONS_STRATEGY_REGISTRY.items():
        settings, _ = resolve(cls, aucune_option)
        assert settings["target_tenor_days"] == 730, nom
        assert settings["roll_when_days_left"] == 270, nom
        assert settings["stop_basis"] == "underlying", nom

    # L'inflation attendue est corrigée sur l'horizon du CONTRAT : l'écart
    # d'une stratégie directionnelle dépend donc de l'échéance retenue.
    assert config.OPTIONS_TARGET_TENOR_DAYS == 730
