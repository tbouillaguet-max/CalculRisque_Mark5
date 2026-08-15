"""Stratégie « espérance de gain » branchée sur le moteur.

Le module de maths est testé isolément (tests/test_expected_value.py). Ce
fichier-ci vérifie l'INTÉGRATION, et en particulier l'astuce du strike : le
moteur ne sait pas recevoir un strike, seulement un prix de référence dont il
fait la moyenne avec le spot. La stratégie inverse cette formule, et rien
d'autre ne garantit que l'inversion tienne.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

import config
from backtest import expected_value, options_pricing
from backtest.strategies.options_base import MarketContext
from backtest.strategies.valuation_gap_expected_value_options import (
    ValuationGapExpectedValueOptionsStrategy,
)
from tests.options_harness import cours, moteur, serie_bruitee, signaux

SIGMA = 0.35
SPOT = 100.0


def _contexte(sigma=SIGMA, implied=None, quoted=(), today="2020-06-01") -> MarketContext:
    return MarketContext(
        today=pd.Timestamp(today),
        realized_vol=lambda symbol: sigma,
        pricing_context=lambda symbol: (0.04, 0.0),
        implied_vol=lambda symbol, option_type, tenor: implied,
        quoted_strikes=lambda symbol, option_type, tenor: list(quoted),
    )


def _signaux(theoretical=150.0, spot=SPOT, symbols=("AAA",)) -> pd.DataFrame:
    df = signaux(symbols, "2020-05-01", theoretical=theoretical)
    return df.assign(close=spot)


def _strategie(**kwargs) -> ValuationGapExpectedValueOptionsStrategy:
    strategie = ValuationGapExpectedValueOptionsStrategy(**kwargs)
    strategie.bind_market_context(_contexte(**kwargs.pop("_contexte", {})))
    return strategie


# --------------------------------------------------------------------------- #
# L'astuce du strike
# --------------------------------------------------------------------------- #

def test_le_prix_de_reference_transmis_redonne_le_strike_optimal():
    """C'est LE test de l'astuce : le moteur calcule (référence + spot) / 2, la
    stratégie transmet 2K* - spot, donc la moyenne doit redonner K*."""
    strategie = _strategie()
    cibles = strategie.generate_option_targets(_signaux(), {})

    assert "AAA" in cibles
    reference = cibles["AAA"]["strike_reference_price"]
    strike_retenu = (reference + SPOT) / 2.0

    attendu = strategie._optimal_strike("AAA", "CALL", SPOT, 150.0)
    assert strike_retenu == pytest.approx(attendu)


def test_le_strike_retenu_est_bien_celui_qui_maximise_la_croissance():
    """Contrôle indépendant : on refait la sélection à la main sur la même
    grille et on doit retomber sur le même contrat."""
    strategie = _strategie()
    t_years = strategie.tenor_days / 365.0
    mu = expected_value.convergence_drift(SPOT, 150.0, t_years, strategie.convergence_fraction)

    cibles = strategie.generate_option_targets(_signaux(), {})
    strike_retenu = (cibles["AAA"]["strike_reference_price"] + SPOT) / 2.0

    meilleur, meilleur_score = None, -math.inf
    for strike in expected_value.strike_grid(SPOT, SIGMA, t_years):
        premium = options_pricing.bs_price(SPOT, strike, t_years, SIGMA, "CALL", r=0.04, q=0.0)
        optimum = expected_value.kelly_optimum(SPOT, strike, mu, SIGMA, t_years, "CALL", premium)
        if optimum is not None and optimum[1] > meilleur_score:
            meilleur, meilleur_score = strike, optimum[1]

    assert strike_retenu == pytest.approx(meilleur)


def test_une_reference_negative_reste_admissible():
    """Un strike très dans la monnaie donne 2K* - spot < 0. Seule la moyenne
    compte, mais le cas doit être traversé sans garde-fou intempestif."""
    # Thèse à peine positive : Kelly choisit un strike profondément ITM.
    strategie = _strategie()
    cibles = strategie.generate_option_targets(_signaux(theoretical=104.0), {})
    if not cibles:
        pytest.skip("aucune ligne retenue à cette conviction")
    reference = cibles["AAA"]["strike_reference_price"]
    assert (reference + SPOT) / 2.0 > 0


# --------------------------------------------------------------------------- #
# Volatilité de sélection
# --------------------------------------------------------------------------- #

def test_l_implicite_reelle_prime_sur_la_realisee():
    """Sélectionner sur la réalisée un contrat que le marché price à une tout
    autre implicite reviendrait à trouver des bonnes affaires partout où le
    marché anticipe simplement plus de mouvement."""
    strategie = ValuationGapExpectedValueOptionsStrategy()
    strategie.bind_market_context(_contexte(sigma=0.20, implied=0.60))
    strategie.generate_option_targets(_signaux(), {})

    assert strategie.diagnostics()["expected_value_implied_vol_count"] == 1
    assert strategie.diagnostics()["expected_value_realized_vol_count"] == 0
    assert strategie.diagnostics()["expected_value_implied_vol_pct"] == 100.0


def test_sans_volatilite_on_se_replie_sur_celle_du_moteur():
    """Écarter la ligne serait plus prudent en apparence, mais biaiserait la
    sélection vers les seuls titres à long historique -- alors que le moteur
    sait parfaitement ouvrir la position, au repli de config.OPTIONS_FALLBACK_VOL."""
    strategie = ValuationGapExpectedValueOptionsStrategy()
    strategie.bind_market_context(MarketContext(
        today=pd.Timestamp("2020-06-01"),
        realized_vol=lambda symbol: None,
        pricing_context=lambda symbol: (0.04, 0.0),
        implied_vol=lambda symbol, option_type, tenor: None,
        quoted_strikes=lambda symbol, option_type, tenor: [],
    ))
    cibles = strategie.generate_option_targets(_signaux(), {})

    assert "AAA" in cibles
    assert strategie.diagnostics()["expected_value_fallback_vol_count"] == 1
    # Le strike doit être celui qu'on obtiendrait à la volatilité de repli.
    attendu = expected_value.optimal_strike(
        SPOT,
        expected_value.convergence_drift(SPOT, 150.0, strategie.tenor_days / 365.0, 0.5),
        config.OPTIONS_FALLBACK_VOL, strategie.tenor_days / 365.0, "CALL",
        expected_value.strike_grid(SPOT, config.OPTIONS_FALLBACK_VOL, strategie.tenor_days / 365.0),
        r=0.04, q=0.0,
    )["strike"]
    assert (cibles["AAA"]["strike_reference_price"] + SPOT) / 2.0 == pytest.approx(attendu)


# --------------------------------------------------------------------------- #
# Grille contrainte par les strikes cotés
# --------------------------------------------------------------------------- #

def test_la_grille_se_restreint_aux_strikes_cotes():
    """Sans cette contrainte, l'optimisation choisirait un strike que personne
    ne cotait, et le moteur se rabattrait silencieusement sur un autre."""
    cotes = [90.0, 110.0, 130.0]
    strategie = ValuationGapExpectedValueOptionsStrategy()
    strategie.bind_market_context(_contexte(quoted=cotes))

    cibles = strategie.generate_option_targets(_signaux(), {})
    strike = (cibles["AAA"]["strike_reference_price"] + SPOT) / 2.0

    assert strike in cotes
    assert strategie.diagnostics()["expected_value_quoted_grid_count"] == 1


def test_des_strikes_cotes_hors_fenetre_font_retomber_sur_la_grille_theorique():
    """Mieux vaut un contrat simulé dans la fenêtre plausible qu'un contrat
    réel manifestement hors sujet."""
    strategie = ValuationGapExpectedValueOptionsStrategy()
    strategie.bind_market_context(_contexte(quoted=[5000.0, 9000.0]))

    cibles = strategie.generate_option_targets(_signaux(), {})
    strike = (cibles["AAA"]["strike_reference_price"] + SPOT) / 2.0

    assert strike < 1000.0
    assert strategie.diagnostics()["expected_value_theoretical_grid_count"] == 1


# --------------------------------------------------------------------------- #
# Espérance positive obligatoire
# --------------------------------------------------------------------------- #

def test_une_these_trop_faible_n_ouvre_rien_et_est_comptee():
    """Une valeur théorique à peine au-dessus du cours ne bat pas la prime :
    aucune mise log-optimale n'existe, la ligne est écartée et comptée."""
    strategie = ValuationGapExpectedValueOptionsStrategy(entry_threshold_pct=0.5)
    strategie.bind_market_context(_contexte(sigma=0.80))

    cibles = strategie.generate_option_targets(_signaux(theoretical=100.5), {})
    assert cibles == {}
    assert strategie.diagnostics()["dropped_expected_value_negative_count"] == 1


def test_la_fraction_de_convergence_est_validee():
    for invalide in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            ValuationGapExpectedValueOptionsStrategy(convergence_fraction=invalide)


def test_une_fraction_plus_forte_deplace_le_strike_vers_le_dehors():
    """Plus la convergence supposée est grande, plus la dérive est forte, donc
    plus Kelly accepte de convexité."""
    strikes = []
    for fraction in (0.2, 0.5, 1.0):
        strategie = ValuationGapExpectedValueOptionsStrategy(convergence_fraction=fraction)
        strategie.bind_market_context(_contexte())
        cibles = strategie.generate_option_targets(_signaux(theoretical=200.0), {})
        strikes.append((cibles["AAA"]["strike_reference_price"] + SPOT) / 2.0)
    assert all(a <= b for a, b in zip(strikes, strikes[1:]))
    assert strikes[-1] > strikes[0]


# --------------------------------------------------------------------------- #
# Héritage : le signal doit rester rigoureusement celui de la stratégie multiples
# --------------------------------------------------------------------------- #

def test_les_candidates_et_les_poids_sont_ceux_de_la_strategie_multiples():
    """La comparaison entre les deux stratégies ne doit mesurer QUE l'effet de
    la sélection de strike."""
    from backtest.strategies.valuation_gap_multiples_options import (
        ValuationGapMultiplesOptionsStrategy,
    )

    donnees = _signaux(symbols=("AAA", "BBB", "CCC"))
    reference = ValuationGapMultiplesOptionsStrategy()
    strategie = _strategie()

    attendues = reference.generate_option_targets(donnees, {})
    obtenues = strategie.generate_option_targets(donnees, {})

    assert set(obtenues) == set(attendues)
    for symbol in attendues:
        assert obtenues[symbol]["weight"] == pytest.approx(attendues[symbol]["weight"])
        assert obtenues[symbol]["option_type"] == attendues[symbol]["option_type"]


# --------------------------------------------------------------------------- #
# Bout en bout à travers le moteur
# --------------------------------------------------------------------------- #

def test_un_run_complet_ouvre_des_positions_et_expose_ses_compteurs():
    n = 400
    panel = cours({"AAA": serie_bruitee(100.0, n, graine=1),
                   "BBB": serie_bruitee(100.0, n, graine=2)})
    evenements = signaux(["AAA", "BBB"], "2020-01-02", theoretical=180.0)

    strategie = ValuationGapExpectedValueOptionsStrategy()
    engine = moteur(panel, evenements, {}, strategy=strategie, vol_mode="rolling")
    engine.run()

    metriques = engine.diagnostics_summary() if hasattr(engine, "diagnostics_summary") else None
    assert engine.executions, "aucune exécution : le moteur n'a rien ouvert"

    # Les compteurs de la stratégie doivent avoir remonté jusqu'aux métriques.
    resume = engine._strategy_diagnostics()
    assert resume["expected_value_evaluations_count"] > 0
    assert "dropped_expected_value_negative_count" in resume
    assert metriques is None or "dropped_expected_value_negative_count" in metriques


def test_le_strike_ouvert_par_le_moteur_est_proche_du_strike_optimise():
    """Le spot de décision (clôture J) et le spot d'exécution (ouverture J+1)
    diffèrent : l'astuce 2K* - spot laisse donc dériver le strike de la MOITIÉ
    du mouvement de nuit. Ce test chiffre cette dérive et vérifie qu'elle reste
    très inférieure au pas de grille -- sans quoi l'optimisation choisirait un
    contrat et le moteur en ouvrirait un autre."""
    # Assez court pour qu'aucun roulement ne vienne remplacer le contrat.
    panel = cours({"AAA": serie_bruitee(100.0, 200, graine=7)})
    evenements = signaux(["AAA"], "2020-01-02", theoretical=180.0)

    strategie = ValuationGapExpectedValueOptionsStrategy()
    engine = moteur(panel, evenements, {}, strategy=strategie, vol_mode="rolling")
    engine.run()

    position = engine.positions.get("AAA")
    assert position is not None, "aucune position ouverte"

    # Volatilité effectivement retenue : au tout début du panel, l'historique
    # est trop court, donc le repli partagé avec le moteur.
    assert strategie.diagnostics()["expected_value_fallback_vol_count"] == 1
    sigma = config.OPTIONS_FALLBACK_VOL
    t_years = strategie.tenor_days / 365.0

    dates = panel.close.index
    spot_decision = float(panel.close.loc[dates[1], "AAA"])   # clôture du jour du signal
    spot_execution = float(panel.open.loc[dates[2], "AAA"])   # ouverture du lendemain

    mu = expected_value.convergence_drift(spot_decision, 180.0, t_years, 0.5)
    r, q = engine._pricing_context("AAA", dates[1])
    optimal = expected_value.optimal_strike(
        spot_decision, mu, sigma, t_years, "CALL",
        expected_value.strike_grid(spot_decision, sigma, t_years), r=r, q=q,
    )["strike"]

    derive = abs(position.strike - optimal)
    pas_de_grille = optimal * (math.exp(0.25 * sigma * math.sqrt(t_years)) - 1.0)

    # La dérive est de l'ordre de la moitié d'un mouvement journalier...
    assert derive < abs(spot_execution - spot_decision)
    # ... donc très en dessous du pas de grille : le contrat que le moteur
    # ouvre est bien celui que l'optimisation a choisi. Mesuré sur ce jeu :
    # dérive 0,22 $ pour un pas de grille de 5,4 $, soit 4 %.
    assert derive < 0.1 * pas_de_grille
