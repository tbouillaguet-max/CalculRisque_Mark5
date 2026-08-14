"""Refonte du rebalancement : deux mécanismes disjoints (journalier / sur
dépôt SEC), filtre ε sur l'écart en log, et provenance des positions
(open_reason). Voir la docstring de backtest/options_engine.py."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import config
from backtest.options_engine import OptionPosition, OptionsBacktestEngine
from backtest.strategies.valuation_gap_options import ValuationGapOptionsStrategy
from backtest.strategies.valuation_gap_multiples_options import ValuationGapMultiplesOptionsStrategy
from tests.options_harness import cours, moteur, serie_plate, signaux

N_JOURS = 90


@pytest.fixture(autouse=True)
def sans_inflation(monkeypatch):
    """Neutralise la correction d'inflation (base.inflation_adjusted_log_gap)
    pour tous les tests de ce fichier : elle ajoute un décalage additif au
    score en log qui dépend de l'ANNÉE de published_date, ce qui compliquerait
    le calcul à la main des écarts attendus sans rien apporter à ce qui est
    testé ici (le MÉCANISME de rebalancement, pas le calcul du gap)."""
    monkeypatch.setattr(config, "INFLATION_ADJUST_GAP", False)


def _spy_open_or_resize(monkeypatch):
    """Enregistre chaque appel à _open_or_resize (symbole, jour), sans
    changer son comportement -- même pattern que
    test_le_redeploiement_a_lieu_meme_sans_ordre_en_file dans
    test_options_engine.py."""
    appels: list[tuple[str, pd.Timestamp]] = []
    vrai = OptionsBacktestEngine._open_or_resize

    def espion(self, symbol, option_type, target_dollar, spot, today, reason, **kwargs):
        appels.append((symbol, today))
        return vrai(self, symbol, option_type, target_dollar, spot, today, reason, **kwargs)

    monkeypatch.setattr(OptionsBacktestEngine, "_open_or_resize", espion)
    return appels


# --------------------------------------------------------------------------- #
# Mécanisme journalier : ouvre du neuf, ne redimensionne jamais l'existant
# --------------------------------------------------------------------------- #

def test_le_mecanisme_journalier_n_ouvre_que_des_symboles_neufs(monkeypatch):
    """AAA est éligible dès le départ et déposé au jour 2 : il s'ouvre via le
    mécanisme SUR DÉPÔT (open_reason="rebalance"). BBB n'est éligible qu'à
    partir d'une chute de cours PURE (aucun nouveau dépôt) : il ne peut donc
    devenir candidat que via le mécanisme JOURNALIER (open_reason=
    "rebalance_daily"). Ni l'un ni l'autre n'est jamais retouché après coup."""
    appels = _spy_open_or_resize(monkeypatch)

    # BBB : théorique 150, cours à 150 les 20 premiers jours (log-gap nul,
    # pas éligible), puis 100 (log(1.5)=0.405, largement au-dessus du seuil
    # d'entrée ~0.182) -- une chute de cours PURE, sans nouveau dépôt.
    bbb = [150.0] * 20 + [100.0] * (N_JOURS - 20)
    panel = cours({"AAA": serie_plate(100.0, N_JOURS), "BBB": bbb})
    dates = panel.close.index

    evenements = pd.concat([
        signaux(["AAA"], dates[2].strftime("%Y-%m-%d"), theoretical=200.0),   # log(2.0)=0.69, éligible d'emblée
        signaux(["BBB"], dates[2].strftime("%Y-%m-%d"), theoretical=150.0),   # connu mais pas encore éligible
    ], ignore_index=True)

    engine = moteur(
        panel, evenements, {},  # StrategieFixe non utilisée : stratégie explicite ci-dessous
        strategy=ValuationGapMultiplesOptionsStrategy(),
        daily_rebalance=True, exit_when_signal_lost=True,
        stop_loss_pct=-95.0, take_profit_pct=1000.0, target_tenor_days=300,
        roll_when_days_left=None, min_deployment_pct=None, min_resize_relative_pct=None,
    )
    engine.run()

    aaa_jours = [j for s, j in appels if s == "AAA"]
    bbb_jours = [j for s, j in appels if s == "BBB"]

    assert len(aaa_jours) == 1, f"AAA retouché {len(aaa_jours)} fois, attendu 1 seule (l'ouverture initiale)"
    assert len(bbb_jours) == 1, f"BBB retouché {len(bbb_jours)} fois, attendu 1 seule (l'ouverture journalière)"
    assert bbb_jours[0] > aaa_jours[0], "BBB doit s'ouvrir APRÈS AAA (chute de cours au jour 20)"

    assert engine.positions["AAA"].open_reason == "rebalance"
    assert engine.positions["BBB"].open_reason == "rebalance_daily"


def test_le_mecanisme_journalier_ferme_toujours_sur_perte_de_signal(monkeypatch):
    """exit_when_signal_lost reste actif dans le mécanisme JOURNALIER : ce
    n'est pas un redimensionnement, donc pas concerné par la contrainte
    "aucun redimensionnement de l'existant"."""
    # AAA : théorique 200, cours 100 (éligible) puis 200 à partir du jour 20
    # (log-gap nul : n'est plus éligible), sans aucun nouveau dépôt.
    aaa = [100.0] * 20 + [200.0] * (N_JOURS - 20)
    panel = cours({"AAA": aaa})
    dates = panel.close.index
    evenements = signaux(["AAA"], dates[2].strftime("%Y-%m-%d"), theoretical=200.0)

    engine = moteur(
        panel, evenements, {},
        strategy=ValuationGapMultiplesOptionsStrategy(),
        daily_rebalance=True, exit_when_signal_lost=True,
        stop_loss_pct=-95.0, take_profit_pct=1000.0, take_profit_convergence_fraction=0,
        target_tenor_days=300, roll_when_days_left=None,
        min_deployment_pct=None, min_resize_relative_pct=None,
    )
    _, _, trades, _ = engine.run()

    assert "AAA" not in engine.positions
    assert (trades["exit_reason"] == "signal_lost").any()


# --------------------------------------------------------------------------- #
# Mécanisme sur dépôt SEC : nouveau candidat sans test ε
# --------------------------------------------------------------------------- #

def test_un_nouveau_candidat_s_ouvre_sans_test_epsilon(monkeypatch):
    """Un ε énorme ne doit JAMAIS empêcher l'ouverture d'un symbole pas
    encore détenu -- le test ε ne s'applique qu'aux positions déjà là."""
    panel = cours({"AAA": serie_plate(100.0, 30)})
    dates = panel.close.index
    evenements = signaux(["AAA"], dates[2].strftime("%Y-%m-%d"), theoretical=200.0)

    engine = moteur(
        panel, evenements, {},
        strategy=ValuationGapMultiplesOptionsStrategy(),
        rebalance_log_gap_threshold=999.0,
        stop_loss_pct=-95.0, take_profit_pct=1000.0, target_tenor_days=300,
        roll_when_days_left=None, min_deployment_pct=None,
    )
    engine.run()

    assert "AAA" in engine.positions
    assert engine.positions["AAA"].open_reason == "rebalance"
    assert engine.positions["AAA"].last_rebalance_log_gap == pytest.approx(math.log(2.0), rel=1e-6)


# --------------------------------------------------------------------------- #
# Mécanisme sur dépôt SEC : filtre ε sur une position déjà détenue
# --------------------------------------------------------------------------- #

def _moteur_six_candidats(theorique_second_signal_aaa, rebalance_log_gap_threshold, min_resize_relative_pct=None):
    """AAA + cinq positions de remplissage (BBB..FFF), toutes à conviction
    égale au départ -- assez de candidats pour que le plafond de pondération
    (BACKTEST_MAX_WEIGHT_PER_POSITION_PCT=20%) ait un effet : avec deux
    candidats seulement, capped_weights retombe en équipondération pure et
    aucun changement de conviction ne peut jamais faire bouger le poids (cf.
    base.capped_weights, "plafond inatteignable"). Ici, à poids égal initial
    (~16,7% chacun, sous le plafond), une conviction qui augmente sur UN seul
    symbole vient buter sur le plafond de 20% -- une hausse de poids garantie.

    weight_cap_pct=0 désactive le plafond D'ENTRÉE propre à la stratégie
    (qui écrête le score de conviction _abs_gap à 100 AVANT capped_weights,
    cf. sa docstring) : sans ça, un théorique très supérieur à 500 produit un
    strike candidat -- (théorique + spot)/2 -- si loin de la monnaie que son
    delta s'écrase sous 1e-6 et _open_or_resize abandonne l'ordre AVANT même
    d'atteindre la logique de redimensionnement, indépendamment de ε."""
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    panel = cours({s: serie_plate(100.0, 40) for s in symbols})
    dates = panel.close.index

    premier_depot = pd.concat([
        signaux([s], dates[2].strftime("%Y-%m-%d"), theoretical=200.0) for s in symbols
    ], ignore_index=True)
    second_depot = signaux(["AAA"], dates[20].strftime("%Y-%m-%d"), theoretical=theorique_second_signal_aaa)
    evenements = pd.concat([premier_depot, second_depot], ignore_index=True)

    engine = moteur(
        panel, evenements, {},
        strategy=ValuationGapMultiplesOptionsStrategy(entry_threshold_pct=0.01, weight_cap_pct=0),
        rebalance_log_gap_threshold=rebalance_log_gap_threshold,
        min_resize_relative_pct=min_resize_relative_pct,
        stop_loss_pct=-95.0, take_profit_pct=1000.0, target_tenor_days=300,
        roll_when_days_left=None, min_deployment_pct=None,
        initial_capital=5_000_000.0,
    )
    return engine, dates


def test_un_ecart_qui_depasse_epsilon_redimensionne_la_position(monkeypatch):
    """AAA passe de théorique 200 à 500 (log-gap 1,61 contre 0,69 pour les
    cinq autres, restés fixes) : sa conviction domine, son poids isolé bute
    sur le plafond de 20% (contre ~16,7% initial) -- un redimensionnement
    réel, au-dessus de tous les planchers ($1, min_resize_relative_pct
    désactivé ici pour isoler ε)."""
    appels = _spy_open_or_resize(monkeypatch)
    engine, dates = _moteur_six_candidats(
        theorique_second_signal_aaa=500.0, rebalance_log_gap_threshold=0.15,
    )
    engine.run()

    aaa_jours = sorted(j for s, j in appels if s == "AAA")
    autres_jours = [(s, j) for s, j in appels if s != "AAA"]

    assert len(aaa_jours) == 2, f"AAA retouché {len(aaa_jours)} fois, attendu 2 (ouverture + redimensionnement)"
    assert aaa_jours[1] > dates[20], "le redimensionnement doit suivre le second dépôt (jour 20)"
    # Aucun des cinq autres symboles n'est retouché une seconde fois : le
    # dépôt de AAA ne génère un ordre QUE sur AAA (pas de cascade), même si
    # leur poids "vrai" (recalculé globalement) a mécaniquement changé.
    for s in ("BBB", "CCC", "DDD", "EEE", "FFF"):
        assert sum(1 for sym, _ in autres_jours if sym == s) == 1, f"{s} retouché plus d'une fois"

    close_exec = engine.prices.close_at("AAA", aaa_jours[1])
    attendu = math.log(500.0 / close_exec)
    assert engine.positions["AAA"].last_rebalance_log_gap == pytest.approx(attendu, rel=1e-6)


def test_un_ecart_sous_epsilon_ne_declenche_aucun_trade_mais_met_a_jour_le_signal(monkeypatch):
    """AAA passe de théorique 200 à 210 (log(210/200)=0,049, sous ε=0,15) :
    known_signals doit refléter 210 pour les calculs futurs, mais
    last_rebalance_log_gap reste calé sur le dépôt D'ORIGINE -- le prochain
    seuil se mesure depuis le dernier TRADE, pas depuis ce signal ignoré."""
    appels = _spy_open_or_resize(monkeypatch)
    engine, dates = _moteur_six_candidats(
        theorique_second_signal_aaa=210.0, rebalance_log_gap_threshold=0.15,
    )
    engine.run()

    aaa_jours = [j for s, j in appels if s == "AAA"]
    assert len(aaa_jours) == 1, "AAA n'aurait dû être touché qu'à l'ouverture (Δ sous ε)"

    assert engine.known_signals["AAA"]["valuation_theoretical_per_share"] == pytest.approx(210.0)
    reference_initiale = math.log(200.0 / 100.0)
    assert engine.positions["AAA"].last_rebalance_log_gap == pytest.approx(reference_initiale, rel=1e-6)


def test_min_resize_relative_pct_reste_un_second_filet_apres_epsilon(monkeypatch):
    """Δ franchit largement ε, mais min_resize_relative_pct=99% bloque le
    micro-ajustement en NOMBRE DE CONTRATS -- le second filet documenté dans
    la refonte reste actif après le premier."""
    appels = _spy_open_or_resize(monkeypatch)
    # min_resize_relative_pct compare au delta BRUT (target_contracts -
    # existing.contracts), avant toute troncature par max_trade_dollar/le
    # cash disponible : avec weight_cap_pct=0, ce delta brut est énorme ici
    # (~1300% des contrats détenus, la conviction de AAA n'étant plus
    # écrêtée du tout) -- un seuil à 2000% est donc garanti de le bloquer,
    # quand un seuil à 200% ne le bloquerait PAS (le delta réellement
    # observé, une fois tronqué à 68 contrats, ne dit rien du delta brut sur
    # lequel porte le test).
    engine, dates = _moteur_six_candidats(
        theorique_second_signal_aaa=500.0, rebalance_log_gap_threshold=0.15,
        min_resize_relative_pct=2000.0,
    )
    engine.run()

    aaa_jours = [j for s, j in appels if s == "AAA"]
    # _open_or_resize est bien RETENTÉ (le filtre ε a laissé passer l'ordre)...
    assert len(aaa_jours) == 2
    # ...mais min_resize_relative_pct='99%' est conçu pour bloquer un
    # changement de conviction qui ne représente pas 99% des contrats
    # détenus : la taille de la position ne doit donc pas avoir bougé au même
    # degré que dans le test sans second filet.
    engine_sans_filet, _ = _moteur_six_candidats(
        theorique_second_signal_aaa=500.0, rebalance_log_gap_threshold=0.15,
        min_resize_relative_pct=None,
    )
    engine_sans_filet.run()
    assert engine.positions["AAA"].contracts < engine_sans_filet.positions["AAA"].contracts


# --------------------------------------------------------------------------- #
# Provenance des positions : open_reason dans trades.parquet
# --------------------------------------------------------------------------- #

def test_open_reason_distingue_depot_et_journalier():
    """Deux positions closes par le MÊME mécanisme de sortie (l'expiration,
    forcée et donc garantie), mais ouvertes par des mécanismes différents :
    trades.parquet doit permettre de les distinguer sans ambiguïté."""
    n_jours = 220
    bbb = [150.0] * 10 + [100.0] * (n_jours - 10)  # devient éligible au jour 10, sans dépôt
    panel = cours({"AAA": serie_plate(100.0, n_jours), "BBB": bbb})
    dates = panel.close.index
    evenements = pd.concat([
        signaux(["AAA"], dates[2].strftime("%Y-%m-%d"), theoretical=200.0),
        signaux(["BBB"], dates[2].strftime("%Y-%m-%d"), theoretical=150.0),
    ], ignore_index=True)

    engine = moteur(
        panel, evenements, {},
        # tenor_days est un paramètre de LA STRATÉGIE (elle le renvoie dans
        # chaque cible, cf. generate_option_targets), distinct de
        # target_tenor_days côté moteur (qui ne sert que de repli ATM sans
        # stratégie de convergence) : les deux doivent être alignés ici. Pas
        # trop court non plus : un strike à mi-chemin (150 pour AAA, 50% hors
        # de la monnaie) avec une échéance de quelques jours a un delta qui
        # s'écrase sous 1e-6 et _open_or_resize abandonne l'ouverture --
        # 60 jours laisse un delta exploitable tout en tenant dans le panel.
        strategy=ValuationGapMultiplesOptionsStrategy(tenor_days=60),
        daily_rebalance=True, stop_loss_pct=-95.0, take_profit_pct=1000.0,
        take_profit_convergence_fraction=0, target_tenor_days=60, roll_when_days_left=None,
        min_deployment_pct=None, min_resize_relative_pct=None,
        # Le contrat de ce scénario est TRÈS hors de la monnaie (strike à
        # mi-chemin, 125 pour un spot de 100, à 60 jours) : son delta tourne
        # autour de 0,04, sous le plancher de dimensionnement
        # (config.OPTIONS_MIN_DELTA_FOR_SIZING). Le plancher est justifié --
        # dimensionner par division sur un delta pareil attribue des milliers
        # de contrats -- mais ce test-ci porte sur `open_reason`, pas sur le
        # dimensionnement : on le désactive pour que les deux positions
        # existent et que l'origine de leur ouverture puisse être comparée.
        min_delta_for_sizing=0,
    )
    _, _, trades, _ = engine.run()

    assert not trades.empty
    assert "open_reason" in trades.columns
    aaa_trade = trades[trades["symbol"] == "AAA"].iloc[0]
    bbb_trade = trades[trades["symbol"] == "BBB"].iloc[0]
    assert aaa_trade["open_reason"] == "rebalance"
    assert bbb_trade["open_reason"] == "rebalance_daily"
    # exit_reason, lui, reste le code de sortie usuel -- inchangé par cette
    # refonte, qui ne touche qu'à l'origine de l'OUVERTURE.
    assert aaa_trade["exit_reason"] == "expiry"
    assert bbb_trade["exit_reason"] == "expiry"


# --------------------------------------------------------------------------- #
# _log_gap_at : cas dégénérés
# --------------------------------------------------------------------------- #

def test_log_gap_at_est_none_si_le_signal_est_inconnu():
    panel = cours({"AAA": serie_plate(100.0, 10)})
    engine = moteur(panel, signaux(["AAA"], "2020-01-06"), {"AAA": "CALL"})
    assert engine._log_gap_at("ZZZ", 100.0) is None


def test_log_gap_at_est_none_si_le_prix_est_invalide():
    panel = cours({"AAA": serie_plate(100.0, 10)})
    engine = moteur(panel, signaux(["AAA"], "2020-01-06"), {"AAA": "CALL"})
    engine.known_signals["AAA"] = {"valuation_theoretical_per_share": 150.0}
    assert engine._log_gap_at("AAA", None) is None
    assert engine._log_gap_at("AAA", 0.0) is None
    assert engine._log_gap_at("AAA", -5.0) is None


def test_log_gap_at_est_none_si_la_theorique_est_invalide():
    panel = cours({"AAA": serie_plate(100.0, 10)})
    engine = moteur(panel, signaux(["AAA"], "2020-01-06"), {"AAA": "CALL"})
    engine.known_signals["AAA"] = {"valuation_theoretical_per_share": None}
    assert engine._log_gap_at("AAA", 100.0) is None
    engine.known_signals["AAA"] = {"valuation_theoretical_per_share": 0.0}
    assert engine._log_gap_at("AAA", 100.0) is None


def test_log_gap_at_calcule_correctement_le_cas_nominal():
    panel = cours({"AAA": serie_plate(100.0, 10)})
    engine = moteur(panel, signaux(["AAA"], "2020-01-06"), {"AAA": "CALL"})
    engine.known_signals["AAA"] = {"valuation_theoretical_per_share": 150.0}
    assert engine._log_gap_at("AAA", 100.0) == pytest.approx(math.log(1.5))


# --------------------------------------------------------------------------- #
# OptionPosition : nouveaux champs, défauts rétrocompatibles
# --------------------------------------------------------------------------- #

def test_option_position_a_des_defauts_retrocompatibles():
    """Une construction sans last_rebalance_log_gap/open_reason (comme le
    fait le reste de la suite de tests, écrite avant cette refonte) ne doit
    pas casser."""
    pos = OptionPosition(
        symbol="AAA", option_type="CALL", strike=100.0, expiry=pd.Timestamp("2025-01-01"),
        contracts=1.0, entry_premium=5.0, entry_date=pd.Timestamp("2020-01-01"),
        vol=0.25, multiplier=100.0, source="simulated",
    )
    assert pos.last_rebalance_log_gap is None
    assert pos.open_reason == "rebalance"


# --------------------------------------------------------------------------- #
# Câblage moteur : rebalance_log_gap_threshold
# --------------------------------------------------------------------------- #

def test_rebalance_log_gap_threshold_par_defaut_vient_de_config():
    panel = cours({"AAA": serie_plate(100.0, 10)})
    engine = moteur(panel, signaux(["AAA"], "2020-01-06"), {"AAA": "CALL"})
    assert engine.rebalance_log_gap_threshold == config.OPTIONS_REBALANCE_LOG_GAP_THRESHOLD


def test_rebalance_log_gap_threshold_est_bien_stocke_si_fourni():
    panel = cours({"AAA": serie_plate(100.0, 10)})
    engine = moteur(
        panel, signaux(["AAA"], "2020-01-06"), {"AAA": "CALL"}, rebalance_log_gap_threshold=0.42,
    )
    assert engine.rebalance_log_gap_threshold == pytest.approx(0.42)
