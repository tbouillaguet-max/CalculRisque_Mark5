"""Formules fermées d'espérance/variance du payoff (backtest/expected_value.py).

Le test qui porte tout le reste est test_coherence_black_scholes_* : si
l'espérance actualisée sous mu = r redonne exactement le prix Black-Scholes,
la formule d'espérance est correcte par construction, et la variance s'appuie
sur le même changement de mesure.
"""

from __future__ import annotations

import math

import pytest

from backtest import expected_value as ev
from backtest import options_pricing

# Jeu de paramètres "ordinaire" (titre à 100 $, 30 % de vol, 2 ans -- l'échéance
# visée par le pipeline, cf. config.OPTIONS_TARGET_TENOR_DAYS).
SPOT = 100.0
SIGMA = 0.30
T = 2.0
R = 0.04


# --------------------------------------------------------------------------- #
# Cohérence avec Black-Scholes (le test central)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("strike", [50.0, 80.0, 100.0, 120.0, 200.0])
@pytest.mark.parametrize("t_years", [0.25, 1.0, 2.0, 5.0])
def test_coherence_black_scholes_call(strike, t_years):
    """Sous mu = r, E[payoff]·e^(-rT) EST le prix Black-Scholes : c'est la
    même intégrale, seule la mesure change. Toute erreur de signe, de d1/d2 ou
    d'exposant se voit ici."""
    esperance = ev.expected_payoff_call(SPOT, strike, R, SIGMA, t_years)
    actualisee = esperance * math.exp(-R * t_years)
    attendu = options_pricing.bs_price(SPOT, strike, t_years, SIGMA, "CALL", r=R, q=0.0)
    assert actualisee == pytest.approx(attendu, abs=1e-8)


@pytest.mark.parametrize("strike", [50.0, 80.0, 100.0, 120.0, 200.0])
@pytest.mark.parametrize("t_years", [0.25, 1.0, 2.0, 5.0])
def test_coherence_black_scholes_put(strike, t_years):
    esperance = ev.expected_payoff_put(SPOT, strike, R, SIGMA, t_years)
    actualisee = esperance * math.exp(-R * t_years)
    attendu = options_pricing.bs_price(SPOT, strike, t_years, SIGMA, "PUT", r=R, q=0.0)
    assert actualisee == pytest.approx(attendu, abs=1e-8)


def test_parite_call_put_sur_les_esperances():
    """E[call] - E[put] = E[S_T] - K = S0·e^(mu·T) - K, quelle que soit la
    mesure : la parité ne dépend pas du taux, seulement de la dérive."""
    mu, strike = 0.12, 110.0
    ecart = (
        ev.expected_payoff_call(SPOT, strike, mu, SIGMA, T)
        - ev.expected_payoff_put(SPOT, strike, mu, SIGMA, T)
    )
    assert ecart == pytest.approx(SPOT * math.exp(mu * T) - strike, abs=1e-9)


# --------------------------------------------------------------------------- #
# Cas limites
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("strike, attendu", [(80.0, 20.0), (100.0, 0.0), (120.0, 0.0)])
def test_maturite_nulle_donne_la_valeur_intrinseque_call(strike, attendu):
    assert ev.expected_payoff_call(SPOT, strike, 0.10, SIGMA, 0.0) == pytest.approx(attendu)


@pytest.mark.parametrize("strike, attendu", [(120.0, 20.0), (100.0, 0.0), (80.0, 0.0)])
def test_maturite_nulle_donne_la_valeur_intrinseque_put(strike, attendu):
    assert ev.expected_payoff_put(SPOT, strike, 0.10, SIGMA, 0.0) == pytest.approx(attendu)


def test_volatilite_nulle_donne_un_payoff_deterministe():
    """Sans aléa, S_T = S0·e^(mu·T) et le payoff se réduit à sa valeur
    intrinsèque à cette valeur-là."""
    mu = 0.10
    forward = SPOT * math.exp(mu * T)
    assert ev.expected_payoff_call(SPOT, 100.0, mu, 0.0, T) == pytest.approx(forward - 100.0)
    assert ev.variance_payoff_call(SPOT, 100.0, mu, 0.0, T) == 0.0


def test_deep_otm_call_ne_vaut_presque_rien():
    assert ev.expected_payoff_call(SPOT, 10 * SPOT, 0.05, SIGMA, T) < 1e-3


def test_deep_otm_put_ne_vaut_presque_rien():
    assert ev.expected_payoff_put(SPOT, 0.1 * SPOT, 0.05, SIGMA, T) < 1e-3


def test_deep_itm_call_vaut_le_forward_moins_le_strike():
    """Très dans la monnaie, l'option est quasi certainement exercée : son
    espérance tend vers E[S_T] - K."""
    mu, strike = 0.05, 0.1 * SPOT
    attendu = SPOT * math.exp(mu * T) - strike
    assert ev.expected_payoff_call(SPOT, strike, mu, SIGMA, T) == pytest.approx(attendu, rel=1e-6)


def test_un_call_de_strike_nul_vaut_le_sous_jacent():
    """ln(S0/K) serait indéfini : le cas est traité à part, et pas en laissant
    remonter une ZeroDivisionError au milieu d'un run."""
    mu = 0.07
    assert ev.expected_payoff_call(SPOT, 0.0, mu, SIGMA, T) == pytest.approx(SPOT * math.exp(mu * T))
    assert ev.expected_payoff_put(SPOT, 0.0, mu, SIGMA, T) == 0.0


def test_un_type_d_option_inconnu_est_refuse():
    with pytest.raises(ValueError):
        ev.expected_payoff(SPOT, 100.0, 0.05, SIGMA, T, "STRADDLE")


# --------------------------------------------------------------------------- #
# Variance
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("strike", [1.0, 10.0, 50.0, 100.0, 150.0, 500.0, 5000.0])
@pytest.mark.parametrize("option_type", ["CALL", "PUT"])
def test_la_variance_n_est_jamais_negative(strike, option_type):
    """E[X²] - E[X]² est une différence de grands nombres presque égaux dès que
    le payoff est quasi déterministe : en double précision elle rend
    régulièrement -1e-16, ce qui ferait planter la racine du Sharpe."""
    variance = ev.variance_payoff(SPOT, strike, 0.08, SIGMA, T, option_type)
    assert variance >= 0.0


@pytest.mark.parametrize("option_type", ["CALL", "PUT"])
def test_la_variance_est_coherente_avec_une_simulation(option_type):
    """Contrôle indépendant de la forme fermée : Monte-Carlo sur un grand
    échantillon doit retrouver espérance et variance à quelques pour mille.
    La simulation ne sert QU'ICI -- la sélection de strike, elle, reste en
    forme fermée pour ne pas dépendre d'un tirage."""
    import numpy as np

    mu, strike, n = 0.08, 110.0, 400_000
    rng = np.random.default_rng(12345)
    z = rng.standard_normal(n)
    s_t = SPOT * np.exp((mu - 0.5 * SIGMA ** 2) * T + SIGMA * math.sqrt(T) * z)
    payoff = np.maximum(s_t - strike, 0.0) if option_type == "CALL" else np.maximum(strike - s_t, 0.0)

    assert ev.expected_payoff(SPOT, strike, mu, SIGMA, T, option_type) == pytest.approx(
        payoff.mean(), rel=0.02)
    assert ev.variance_payoff(SPOT, strike, mu, SIGMA, T, option_type) == pytest.approx(
        payoff.var(), rel=0.05)


def test_le_second_moment_domine_le_carre_de_l_esperance():
    """Inégalité de Jensen : E[X²] >= E[X]², sans quoi la variance serait
    structurellement négative."""
    for strike in (60.0, 100.0, 140.0):
        moment = ev.second_moment_payoff_call(SPOT, strike, 0.06, SIGMA, T)
        esperance = ev.expected_payoff_call(SPOT, strike, 0.06, SIGMA, T)
        assert moment >= esperance ** 2


# --------------------------------------------------------------------------- #
# Monotonies
# --------------------------------------------------------------------------- #

def test_l_esperance_d_un_call_decroit_avec_le_strike():
    strikes = [60.0, 80.0, 100.0, 120.0, 140.0, 160.0]
    valeurs = [ev.expected_payoff_call(SPOT, k, 0.06, SIGMA, T) for k in strikes]
    assert all(a > b for a, b in zip(valeurs, valeurs[1:]))


def test_l_esperance_d_un_put_croit_avec_le_strike():
    strikes = [60.0, 80.0, 100.0, 120.0, 140.0, 160.0]
    valeurs = [ev.expected_payoff_put(SPOT, k, 0.06, SIGMA, T) for k in strikes]
    assert all(a < b for a, b in zip(valeurs, valeurs[1:]))


def test_l_esperance_d_un_call_croit_avec_la_derive():
    valeurs = [ev.expected_payoff_call(SPOT, 100.0, mu, SIGMA, T) for mu in (-0.05, 0.0, 0.05, 0.15)]
    assert all(a < b for a, b in zip(valeurs, valeurs[1:]))


def test_l_esperance_croit_avec_la_volatilite():
    valeurs = [ev.expected_payoff_call(SPOT, 120.0, 0.05, s, T) for s in (0.10, 0.20, 0.40, 0.80)]
    assert all(a < b for a, b in zip(valeurs, valeurs[1:]))


# --------------------------------------------------------------------------- #
# Sharpe du contrat et choix du strike
# --------------------------------------------------------------------------- #

def test_sans_edge_le_sharpe_est_nul():
    """mu = r : le marché et la thèse disent la même chose, donc l'espérance
    nette est nulle et aucun strike ne se distingue."""
    strike = 110.0
    premium = options_pricing.bs_price(SPOT, strike, T, SIGMA, "CALL", r=R, q=0.0)
    sharpe = ev.sharpe_contract(SPOT, strike, R, SIGMA, T, "CALL", premium)
    # E[payoff] = prime·e^(rT) : l'écart est la seule valeur temps du taux.
    assert sharpe > 0
    assert sharpe == pytest.approx(
        (premium * math.exp(R * T) - premium)
        / math.sqrt(ev.variance_payoff_call(SPOT, strike, R, SIGMA, T)),
        rel=1e-9,
    )


def test_un_payoff_deterministe_n_a_pas_de_sharpe():
    """Écart-type nul : retourner +inf ferait gagner ce cas dégénéré à tous
    les coups dans le classement des strikes."""
    assert ev.sharpe_contract(SPOT, 100.0, 0.05, 0.0, T, "CALL", premium=1.0) is None
    assert ev.sharpe_contract(SPOT, 100.0, 0.05, SIGMA, 0.0, "CALL", premium=1.0) is None


def test_la_grille_de_strikes_est_centree_et_adaptative():
    grille = ev.strike_grid(SPOT, SIGMA, T, n_sigma=3.0, step_sigma=0.25)
    assert len(grille) == 25                      # 24 pas + le centre
    assert grille[len(grille) // 2] == pytest.approx(SPOT)
    assert grille[0] == pytest.approx(SPOT * math.exp(-3 * SIGMA * math.sqrt(T)))
    assert grille[-1] == pytest.approx(SPOT * math.exp(3 * SIGMA * math.sqrt(T)))

    # Plus la volatilité monte, plus la grille s'élargit -- c'est tout l'objet
    # de l'adaptativité.
    large = ev.strike_grid(SPOT, 0.60, T)
    assert large[-1] > grille[-1]


def test_une_grille_degeneree_est_vide():
    assert ev.strike_grid(SPOT, 0.0, T) == []
    assert ev.strike_grid(SPOT, SIGMA, 0.0) == []


def test_le_strike_optimal_a_une_esperance_nette_positive():
    """mu franchement au-dessus de r : la thèse dit que le titre monte plus
    que le marché ne le facture, il doit donc exister un strike gagnant."""
    mu = 0.25
    grille = ev.strike_grid(SPOT, SIGMA, T)
    resultat = ev.optimal_strike(SPOT, mu, SIGMA, T, "CALL", grille, r=R)

    assert resultat is not None
    assert resultat["edge"] > 0
    assert resultat["strike"] in grille
    assert resultat["sharpe"] == pytest.approx(
        ev.sharpe_contract(SPOT, resultat["strike"], mu, SIGMA, T, "CALL", resultat["premium"]),
        rel=1e-12,
    )


def test_le_strike_optimal_maximise_bien_le_sharpe():
    mu = 0.20
    grille = ev.strike_grid(SPOT, SIGMA, T)
    resultat = ev.optimal_strike(SPOT, mu, SIGMA, T, "CALL", grille, r=R)

    for strike in grille:
        premium = options_pricing.bs_price(SPOT, strike, T, SIGMA, "CALL", r=R, q=0.0)
        if premium <= 0 or ev.expected_payoff_call(SPOT, strike, mu, SIGMA, T) - premium <= 0:
            continue
        sharpe = ev.sharpe_contract(SPOT, strike, mu, SIGMA, T, "CALL", premium)
        assert sharpe <= resultat["sharpe"] + 1e-12


def test_aucun_strike_gagnant_quand_la_these_est_baissiere():
    """mu très en dessous de r sur un CALL : aucun candidat n'a d'espérance
    nette positive, la ligne doit être écartée (None) et non ouverte au moins
    mauvais strike."""
    grille = ev.strike_grid(SPOT, SIGMA, T)
    assert ev.optimal_strike(SPOT, -0.30, SIGMA, T, "CALL", grille, r=R) is None


def test_une_prime_reelle_peut_etre_injectee():
    """premium_fn permet de classer les strikes sur des primes COTÉES plutôt
    que modélisées, sans que le module ait à connaître le carnet d'ordres."""
    mu = 0.25
    grille = [90.0, 100.0, 110.0]
    cotations = {90.0: 25.0, 100.0: 18.0, 110.0: 13.0}
    resultat = ev.optimal_strike(
        SPOT, mu, SIGMA, T, "CALL", grille, premium_fn=cotations.get)

    assert resultat is not None
    assert resultat["premium"] == cotations[resultat["strike"]]


def test_les_strikes_invalides_sont_ignores():
    grille = [None, float("nan"), -10.0, 0.0, 100.0]
    resultat = ev.optimal_strike(SPOT, 0.25, SIGMA, T, "CALL", grille, r=R)
    assert resultat is not None
    assert resultat["strike"] == 100.0
    assert resultat["n_candidates"] == 1


# --------------------------------------------------------------------------- #
# Dérive de convergence
# --------------------------------------------------------------------------- #

def test_la_convergence_totale_amene_le_cours_a_la_valeur_theorique():
    mu = ev.convergence_drift(SPOT, 150.0, T, fraction=1.0)
    assert SPOT * math.exp(mu * T) == pytest.approx(150.0)


def test_la_convergence_partielle_ne_fait_que_la_fraction_du_chemin_en_log():
    mu_moitie = ev.convergence_drift(SPOT, 150.0, T, fraction=0.5)
    mu_total = ev.convergence_drift(SPOT, 150.0, T, fraction=1.0)
    assert mu_moitie == pytest.approx(mu_total / 2)
    # À mi-chemin en log, donc sous la moyenne arithmétique des deux bornes.
    assert SPOT < SPOT * math.exp(mu_moitie * T) < 150.0


def test_une_these_baissiere_donne_une_derive_negative():
    assert ev.convergence_drift(SPOT, 70.0, T, fraction=0.5) < 0


@pytest.mark.parametrize("spot, valeur, t_years", [(0.0, 150.0, 2.0), (100.0, 0.0, 2.0), (100.0, 150.0, 0.0)])
def test_une_derive_indefinie_est_rendue_none(spot, valeur, t_years):
    assert ev.convergence_drift(spot, valeur, t_years, fraction=0.5) is None
