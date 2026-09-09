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
# Moments tronqués et risque baissier
# --------------------------------------------------------------------------- #

def test_les_moments_tronques_convergent_vers_les_moments_complets():
    """À la limite d'un seuil infini, la troncature ne coupe plus rien :
    on doit retrouver 1, E[S_T] et E[S_T²]."""
    mu = 0.08
    m0, m1, m2 = ev.truncated_moments_below(SPOT, 1e12, mu, SIGMA, T)
    assert m0 == pytest.approx(1.0)
    assert m1 == pytest.approx(SPOT * math.exp(mu * T))
    assert m2 == pytest.approx(SPOT ** 2 * math.exp((2 * mu + SIGMA ** 2) * T))


def test_les_moments_tronques_sont_vides_sous_un_seuil_negatif():
    """S_T est strictement positif : l'événement est vide. C'est cette
    convention qui permet aux formules de risque de traiter sans cas
    particulier un seuil de rentabilité situé sous zéro."""
    assert ev.truncated_moments_below(SPOT, -5.0, 0.05, SIGMA, T) == (0.0, 0.0, 0.0)
    assert ev.truncated_moments_below(SPOT, 0.0, 0.05, SIGMA, T) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("strike", [70.0, 100.0, 130.0])
@pytest.mark.parametrize("option_type", ["CALL", "PUT"])
def test_le_risque_baissier_est_coherent_avec_une_simulation(strike, option_type):
    """Contrôle indépendant de la forme fermée du semi-écart-type et de la
    probabilité de gain."""
    import numpy as np

    mu, n = 0.20, 500_000
    rng = np.random.default_rng(7)
    s_t = SPOT * np.exp((mu - 0.5 * SIGMA ** 2) * T + SIGMA * math.sqrt(T) * rng.standard_normal(n))
    payoff = np.maximum(s_t - strike, 0.0) if option_type == "CALL" else np.maximum(strike - s_t, 0.0)
    premium = options_pricing.bs_price(SPOT, strike, T, SIGMA, option_type, r=R, q=0.0)

    assert ev.downside_semivariance(SPOT, strike, mu, SIGMA, T, option_type, premium) == pytest.approx(
        np.mean(np.maximum(premium - payoff, 0.0) ** 2), rel=0.02)
    assert ev.probability_of_profit(SPOT, strike, mu, SIGMA, T, option_type, premium) == pytest.approx(
        np.mean(payoff > premium), abs=0.005)


def test_le_risque_baissier_est_plafonne_par_la_prime():
    """La perte d'une option longue est bornée par la mise : le semi-écart-type
    sous le seuil de rentabilité ne peut donc pas dépasser la prime. C'est
    exactement l'asymétrie que l'écart-type ne voit pas."""
    for strike in (50.0, 100.0, 200.0):
        premium = options_pricing.bs_price(SPOT, strike, T, SIGMA, "CALL", r=R, q=0.0)
        semi_ecart = math.sqrt(ev.downside_semivariance_call(SPOT, strike, 0.15, SIGMA, T, premium))
        assert 0.0 <= semi_ecart <= premium + 1e-9


def test_un_put_dont_la_prime_depasse_le_strike_ne_peut_jamais_gagner():
    """Payoff maximal K, mise > K : la probabilité de gain est nulle et tout le
    support contribue au risque baissier."""
    strike, premium = 20.0, 25.0
    assert ev.probability_of_profit(SPOT, strike, -0.20, SIGMA, T, "PUT", premium) == 0.0
    assert ev.downside_semivariance_put(SPOT, strike, -0.20, SIGMA, T, premium) > 0.0


def test_sans_prime_il_n_y_a_pas_de_risque_baissier():
    """Un contrat gratuit ne peut pas perdre d'argent."""
    assert ev.downside_semivariance_call(SPOT, 100.0, 0.05, SIGMA, T, premium=0.0) == 0.0
    assert ev.downside_semivariance_put(SPOT, 100.0, 0.05, SIGMA, T, premium=0.0) == 0.0


def test_la_probabilite_de_gain_decroit_avec_le_strike_pour_un_call():
    valeurs = []
    for strike in (80.0, 100.0, 120.0, 160.0, 200.0):
        premium = options_pricing.bs_price(SPOT, strike, T, SIGMA, "CALL", r=R, q=0.0)
        valeurs.append(ev.probability_of_profit(SPOT, strike, 0.10, SIGMA, T, "CALL", premium))
    assert all(a > b for a, b in zip(valeurs, valeurs[1:]))


def test_le_risque_baissier_refuse_un_type_inconnu():
    with pytest.raises(ValueError):
        ev.downside_semivariance(SPOT, 100.0, 0.05, SIGMA, T, "STRANGLE", 5.0)
    with pytest.raises(ValueError):
        ev.probability_of_profit(SPOT, 100.0, 0.05, SIGMA, T, "STRANGLE", 5.0)


# --------------------------------------------------------------------------- #
# Sharpe du contrat et choix du strike
# --------------------------------------------------------------------------- #

def test_sans_edge_le_sharpe_est_nul():
    """mu = r : le marché et la thèse disent la même chose, donc l'espérance
    nette est nulle et aucun strike ne se distingue.

    C'est le test de cohérence central du module. Il n'est vrai qu'en portant
    la mise à l'échéance (`r=R`, cf. carried_premium) : le payoff est reçu en T
    et la prime payée en 0. Sans cette mise à niveau, l'écart vaut
    prime·(e^(rT) - 1) > 0 sur TOUS les strikes -- un avantage qui n'est que la
    valeur temps du cash non investi, et non une thèse."""
    strike = 110.0
    premium = options_pricing.bs_price(SPOT, strike, T, SIGMA, "CALL", r=R, q=0.0)

    sharpe = ev.sharpe_contract(SPOT, strike, R, SIGMA, T, "CALL", premium, r=R)
    assert sharpe == pytest.approx(0.0, abs=1e-12)

    # `r` omis = prime déjà exprimée à la date du payoff (convention des
    # fonctions de maths pures) : l'écart redevient la seule valeur temps.
    sans_taux = ev.sharpe_contract(SPOT, strike, R, SIGMA, T, "CALL", premium)
    assert sans_taux == pytest.approx(
        (premium * math.exp(R * T) - premium)
        / math.sqrt(ev.variance_payoff_call(SPOT, strike, R, SIGMA, T)),
        rel=1e-9,
    )


def test_sans_edge_kelly_refuse_de_miser():
    """Corollaire du précédent, sur le critère réellement utilisé en
    production : sous mu = r, aucune mise log-optimale n'existe. C'est ce qui
    interdit d'ouvrir une position dont la thèse ne bat pas le taux sans
    risque."""
    strike = 110.0
    premium = options_pricing.bs_price(SPOT, strike, T, SIGMA, "CALL", r=R, q=0.0)
    assert ev.kelly_optimum(SPOT, strike, R, SIGMA, T, "CALL", premium, r=R) is None

    # Une thèse strictement meilleure que le cash, elle, doit miser.
    optimum = ev.kelly_optimum(SPOT, strike, R + 0.15, SIGMA, T, "CALL", premium, r=R)
    assert optimum is not None and 0 < optimum[0] < 1


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
    resultat = ev.optimal_strike(SPOT, mu, SIGMA, T, "CALL", grille, r=R, criterion="sharpe")

    assert resultat is not None
    assert resultat["edge"] > 0
    assert resultat["strike"] in grille
    # `r=R` comme optimal_strike : le score rendu porte la mise à l'échéance.
    assert resultat["score"] == pytest.approx(
        ev.sharpe_contract(
            SPOT, resultat["strike"], mu, SIGMA, T, "CALL", resultat["premium"], r=R),
        rel=1e-12,
    )


def test_le_strike_optimal_maximise_bien_le_sharpe():
    mu = 0.20
    grille = ev.strike_grid(SPOT, SIGMA, T)
    resultat = ev.optimal_strike(SPOT, mu, SIGMA, T, "CALL", grille, r=R, criterion="sharpe")

    for strike in grille:
        premium = options_pricing.bs_price(SPOT, strike, T, SIGMA, "CALL", r=R, q=0.0)
        # Même filtre d'admission que optimal_strike : le contrat doit battre le
        # cash, donc la mise est portée à l'échéance (cf. carried_premium).
        mise = ev.carried_premium(premium, R, T)
        if premium <= 0 or ev.expected_payoff_call(SPOT, strike, mu, SIGMA, T) - mise <= 0:
            continue
        sharpe = ev.sharpe_contract(SPOT, strike, mu, SIGMA, T, "CALL", premium, r=R)
        assert sharpe <= resultat["score"] + 1e-12


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
# Taux de croissance log-optimal (Kelly)
# --------------------------------------------------------------------------- #

def _prime(strike, option_type="CALL", sigma=SIGMA, t_years=T):
    return options_pricing.bs_price(SPOT, strike, t_years, sigma, option_type, r=R, q=0.0)


def test_la_quadrature_a_converge_bien_avant_le_reglage_par_defaut():
    """L'intégrande est analytique sur la région lucrative (l'atome de perte
    totale est traité en forme fermée), donc Gauss-Legendre y converge
    géométriquement. Mesuré : 32 nœuds suffisent déjà à 1e-12 près."""
    premium = _prime(120.0)
    reference = ev.kelly_optimum(SPOT, 120.0, 0.20, SIGMA, T, "CALL", premium, n_nodes=2048)[1]
    for n_nodes in (32, 64, 128, 512):
        valeur = ev.kelly_optimum(SPOT, 120.0, 0.20, SIGMA, T, "CALL", premium, n_nodes=n_nodes)[1]
        assert valeur == pytest.approx(reference, abs=1e-12)


def test_la_quadrature_est_reproductible():
    """Un Monte-Carlo rendrait une valeur différente à chaque appel, et ferait
    donc changer le strike retenu sans qu'aucune donnée n'ait bougé."""
    premium = _prime(115.0)
    appels = [ev.kelly_optimum(SPOT, 115.0, 0.18, SIGMA, T, "CALL", premium) for _ in range(5)]
    assert all(appel == appels[0] for appel in appels)


@pytest.mark.parametrize("option_type, strike, mu", [
    ("CALL", 120.0, 0.20), ("CALL", 80.0, 0.10), ("PUT", 90.0, -0.15),
])
def test_le_taux_de_croissance_est_coherent_avec_une_simulation(option_type, strike, mu):
    import numpy as np

    premium = _prime(strike, option_type)
    fraction, croissance = ev.kelly_optimum(SPOT, strike, mu, SIGMA, T, option_type, premium)

    rng = np.random.default_rng(3)
    s_t = SPOT * np.exp((mu - 0.5 * SIGMA ** 2) * T + SIGMA * math.sqrt(T) * rng.standard_normal(2_000_000))
    payoff = np.maximum(s_t - strike, 0.0) if option_type == "CALL" else np.maximum(strike - s_t, 0.0)
    assert croissance == pytest.approx(np.log1p(fraction * (payoff / premium - 1.0)).mean(), abs=2e-3)


def test_la_fraction_de_kelly_maximise_bien_la_croissance():
    """Contrôle direct de la bissection : g doit décroître de part et d'autre
    de f*."""
    import numpy as np

    premium = _prime(120.0)
    fraction, croissance = ev.kelly_optimum(SPOT, 120.0, 0.20, SIGMA, T, "CALL", premium)
    prob_perte, poids, rendements = ev._return_nodes(
        SPOT, 120.0, 0.20, SIGMA, T, "CALL", premium, 128)

    def g(f):
        return float(np.sum(poids * np.log1p(f * rendements))) + prob_perte * math.log1p(-f)

    assert 0.0 < fraction < 1.0
    assert g(fraction - 0.02) < croissance
    assert g(fraction + 0.02) < croissance


def test_sans_esperance_nette_positive_il_n_y_a_pas_de_mise_optimale():
    """La condition d'existence de Kelly (E[R] > 0) EST la règle « espérance
    positive obligatoire » : ce n'est pas un filtre ajouté par-dessus."""
    premium = _prime(120.0)
    assert ev.expected_payoff_call(SPOT, 120.0, -0.10, SIGMA, T) < premium
    assert ev.kelly_optimum(SPOT, 120.0, -0.10, SIGMA, T, "CALL", premium) is None


def test_sous_mu_egal_r_la_mise_reste_marginale():
    """Sans avantage informationnel réel, Kelly ne mise presque rien : le seul
    edge est la valeur temps du taux sans risque."""
    premium = _prime(100.0)
    fraction, _ = ev.kelly_optimum(SPOT, 100.0, R, SIGMA, T, "CALL", premium)
    assert fraction < 0.20


def test_la_fraction_de_kelly_croit_avec_la_conviction():
    fractions = [
        ev.kelly_optimum(SPOT, 110.0, mu, SIGMA, T, "CALL", _prime(110.0))[0]
        for mu in (0.08, 0.12, 0.18, 0.25)
    ]
    assert all(a < b for a, b in zip(fractions, fractions[1:]))


def test_kelly_refuse_un_contrat_sans_prime():
    assert ev.kelly_optimum(SPOT, 100.0, 0.20, SIGMA, T, "CALL", premium=0.0) is None


def test_kelly_refuse_un_type_inconnu():
    with pytest.raises(ValueError):
        ev._return_nodes(SPOT, 100.0, 0.20, SIGMA, T, "STRADDLE", 10.0, 64)


# --------------------------------------------------------------------------- #
# Sélection de strike : le critère de Kelly n'est pas dégénéré
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sigma, mu", [(0.20, 0.20), (0.20, 0.35), (0.30, 0.20), (0.30, 0.35)])
def test_l_optimum_de_kelly_est_interieur_quand_l_edge_est_reel(sigma, mu):
    """Le défaut central des deux critères en ratio est un optimum collé au
    bord de la grille. Dès que la thèse porte un avantage significatif, Kelly
    choisit un strike STRICTEMENT INTÉRIEUR."""
    grille = ev.strike_grid(SPOT, sigma, T)
    resultat = ev.optimal_strike(SPOT, mu, sigma, T, "CALL", grille, r=R)

    assert resultat is not None
    assert grille[0] < resultat["strike"] < grille[-1]


def test_le_strike_de_kelly_se_deplace_avec_la_conviction():
    """Propriété qui distingue Kelly des ratios : plus l'avantage est fort,
    plus le strike optimal s'éloigne vers le hors-la-monnaie, continûment."""
    strikes = [
        ev.optimal_strike(SPOT, mu, 0.20, T, "CALL", ev.strike_grid(SPOT, 0.20, T), r=R)["strike"]
        for mu in (0.10, 0.20, 0.28, 0.35)
    ]
    assert all(a <= b for a, b in zip(strikes, strikes[1:]))
    assert strikes[-1] > strikes[0]


def test_le_sharpe_lui_reste_colle_au_bord_dans_la_monnaie():
    """Témoin du problème documenté : conservé pour que la régression soit
    visible si quelqu'un rebranchait le Sharpe en production."""
    for mu in (0.10, 0.20, 0.35):
        grille = ev.strike_grid(SPOT, SIGMA, T)
        resultat = ev.optimal_strike(SPOT, mu, SIGMA, T, "CALL", grille, r=R, criterion="sharpe")
        assert resultat["strike"] == pytest.approx(grille[0])


def test_le_score_rendu_correspond_bien_au_critere_demande():
    grille = ev.strike_grid(SPOT, SIGMA, T)
    resultat = ev.optimal_strike(SPOT, 0.25, SIGMA, T, "CALL", grille, r=R)
    attendu = ev.kelly_optimum(
        SPOT, resultat["strike"], 0.25, SIGMA, T, "CALL", resultat["premium"], r=R)
    assert resultat["kelly_fraction"] == pytest.approx(attendu[0])
    assert resultat["score"] == pytest.approx(attendu[1])
    assert resultat["log_growth"] == pytest.approx(attendu[1])


def test_un_critere_inconnu_est_refuse():
    with pytest.raises(ValueError):
        ev.optimal_strike(SPOT, 0.20, SIGMA, T, "CALL", [100.0], r=R, criterion="sortino")


def test_le_resultat_porte_les_diagnostics_du_contrat_retenu():
    """Ils alimentent les compteurs agrégés du backtest, et évitent à
    l'appelant de refaire les mêmes calculs."""
    grille = ev.strike_grid(SPOT, SIGMA, T)
    resultat = ev.optimal_strike(SPOT, 0.25, SIGMA, T, "CALL", grille, r=R)
    for cle in ("strike", "criterion", "score", "kelly_fraction", "expected_payoff", "premium",
                "edge", "std_payoff", "downside_semideviation", "probability_of_profit",
                "n_candidates", "n_negative_edge"):
        assert cle in resultat
    assert 0.0 < resultat["probability_of_profit"] < 1.0
    # L'edge est net du CASH : E[payoff] - prime·e^(rT), pas E[payoff] - prime.
    assert resultat["edge"] == pytest.approx(
        resultat["expected_payoff"] - ev.carried_premium(resultat["premium"], R, T))
    assert resultat["edge"] < resultat["expected_payoff"] - resultat["premium"]


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
