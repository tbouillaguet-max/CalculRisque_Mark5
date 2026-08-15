"""
Espérance, variance et ratio de Sharpe du PAYOFF d'un contrat d'option sous
l'hypothèse lognormale, en forme fermée -- support mathématique de
backtest/strategies/valuation_gap_expected_value_options.py.

Module de MATHS PUR : aucun état, aucune dépendance au moteur de backtest, et
aucune lecture de données. Il ne connaît ni les positions, ni le portefeuille,
ni le calendrier -- il répond à une seule question, pour un jeu de paramètres
donné : à quel strike le contrat offre-t-il le meilleur rapport
espérance/risque ? C'est ce qui le rend testable ligne à ligne, ce que la
sélection de strike ad hoc des deux stratégies existantes (ATM, mi-chemin
cours/valeur théorique) n'est pas.


DEUX MESURES, ET C'EST TOUT L'INTÉRÊT
--------------------------------------
Le calcul fait délibérément coexister deux dérives différentes, et les
confondre viderait la démarche de son sens :

    - L'ESPÉRANCE de payoff est calculée sous la mesure PHYSIQUE, avec une
      dérive `mu` issue de la thèse de valorisation (convergence partielle du
      cours vers la valeur théorique -- voir la stratégie appelante). C'est la
      traduction en dollars de ce que le signal prétend savoir.

    - La PRIME payée est un prix de MARCHÉ, donc calculée sous la mesure
      risque-neutre avec le taux sans risque `r` (options_pricing.bs_price,
      jamais réécrit ici). Le marché ne facture pas la conviction de
      l'acheteur.

L'écart entre les deux -- `E[payoff] - prime` -- est exactement l'espérance de
gain net revendiquée par la thèse. Si `mu = r`, cet écart est nul à
l'actualisation près : c'est le test de cohérence le plus important du module
(voir tests/test_expected_value.py), et il vaut démonstration que la formule
d'espérance est correcte par construction.


POURQUOI UN SHARPE DE CONTRAT, ET PAS L'ESPÉRANCE SEULE
--------------------------------------------------------
Maximiser `E[payoff] - prime` seul pousse mécaniquement vers les strikes très
hors de la monnaie : leur prime tend vers zéro plus vite que leur espérance,
donc le ratio gain/mise explose alors que la probabilité de toucher quoi que
ce soit s'effondre. Rapporter l'espérance nette à l'ÉCART-TYPE du payoff
pénalise cette asymétrie : un contrat qui ne paie que dans 3 % des scénarios a
un écart-type énorme devant son espérance.

La variance est calculée en forme fermée elle aussi (second moment lognormal
tronqué), et non par Monte-Carlo : le bruit d'échantillonnage d'une simulation
se transmettrait directement au CHOIX du strike, avec un strike optimal qui
changerait d'un run à l'autre sans qu'aucune donnée n'ait bougé.


CONVENTIONS
-----------
S_T = S0 · exp((mu - sigma²/2)·T + sigma·√T·Z),  Z ~ N(0,1)

d'où E[S_T] = S0 · e^(mu·T) : `mu` est bien le rendement ESPÉRÉ annualisé en
composition continue, pas la dérive du logarithme. C'est la convention de
Black-Scholes avec `mu` à la place de `r`, ce qui rend les deux formules
directement superposables -- et c'est ce qui permet au test de cohérence
ci-dessus de porter.

    d1 = [ln(S0/K) + (mu + sigma²/2)·T] / (sigma·√T)
    d2 = d1 - sigma·√T

LIMITE FONDAMENTALE : tout ce que produit ce module est conditionnel à `mu` et
`sigma`, qui sont ESTIMÉS. L'espérance calculée n'est pas une prédiction, c'est
la traduction en dollars d'une hypothèse. Un `mu` faux donne une espérance
fausse avec la même précision apparente à la douzième décimale.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Iterable, List, Optional

import numpy as np

import config
from backtest import options_pricing

_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

CALL = "CALL"
PUT = "PUT"

# Bornes d'intégration en écarts-types de la loi normale sous-jacente. La
# densité au-delà de 10 sigma vaut ~1e-23 : la tronquer coûte moins que
# l'erreur d'arrondi du reste du calcul.
_Z_MAX = 10.0

# Fraction de Kelly maximale autorisée. Une option longue peut perdre 100 % de
# la mise, donc log(1 - f) diverge en f = 1 : le taux de croissance n'est fini
# que strictement sous 1. La borne est numérique, pas économique -- le
# dimensionnement réel reste celui du portefeuille.
_F_MAX = 0.999


def _norm_cdf(x: float) -> float:
    """Identique à options_pricing._norm_cdf (math.erfc plutôt que
    scipy.stats.norm : même valeur à la précision machine près, environ cent
    fois moins cher sur un scalaire). Redéfinie ici plutôt qu'importée d'un
    nom privé -- ce module doit rester utilisable seul."""
    return 0.5 * math.erfc(-x * _INV_SQRT_2)


def _d1_d2(spot: float, strike: float, mu: float, sigma: float, t_years: float) -> tuple:
    """(d1, d2) sous la mesure physique. L'appelant a déjà écarté les cas
    dégénérés (T <= 0, sigma <= 0, strike <= 0) : cette fonction suppose des
    paramètres strictement positifs."""
    sigma_sqrt_t = sigma * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (mu + 0.5 * sigma * sigma) * t_years) / sigma_sqrt_t
    return d1, d1 - sigma_sqrt_t


def _degenerate_spot(spot: float, mu: float, t_years: float) -> float:
    """Sous-jacent à l'échéance quand il n'y a plus d'aléa (sigma = 0, ou
    T = 0 auquel cas e^0 = 1 rend le spot lui-même)."""
    return spot * math.exp(mu * max(t_years, 0.0))


def truncated_moments_below(
    spot: float, level: float, mu: float, sigma: float, t_years: float,
) -> tuple:
    """Trois moments de S_T TRONQUÉS sous un niveau, sous la mesure physique :

        ( P(S_T <= L) , E[S_T·1{S_T<=L}] , E[S_T²·1{S_T<=L}] )
        = ( N(-d2) , S0·e^(mu·T)·N(-d1) , S0²·e^((2mu+sigma²)T)·N(-(d1+sigma√T)) )

    Ce sont les seules briques dont tout le module a besoin : espérances,
    seconds moments et mesures de risque asymétriques s'écrivent toutes comme
    des combinaisons de ces trois intégrales évaluées à un ou deux niveaux. Les
    exposer une fois évite de redériver (et de faire diverger) le même calcul
    dans cinq fonctions.

    (0, 0, 0) pour un niveau nul ou négatif : S_T est strictement positif, donc
    l'événement est vide. Cette convention est ce qui permet aux formules de
    risque ci-dessous de traiter sans cas particulier un seuil de rentabilité
    situé sous zéro."""
    if spot <= 0 or level <= 0:
        return 0.0, 0.0, 0.0

    if t_years <= 0 or sigma <= 0:
        # Sans aléa, S_T vaut son forward : la troncature est un simple test.
        forward = _degenerate_spot(spot, mu, t_years)
        if forward <= level:
            return 1.0, forward, forward * forward
        return 0.0, 0.0, 0.0

    sigma_sqrt_t = sigma * math.sqrt(t_years)
    d1, d2 = _d1_d2(spot, level, mu, sigma, t_years)
    return (
        _norm_cdf(-d2),
        spot * math.exp(mu * t_years) * _norm_cdf(-d1),
        spot * spot * math.exp((2 * mu + sigma * sigma) * t_years) * _norm_cdf(-(d1 + sigma_sqrt_t)),
    )


# ----------------------------------------------------------------------------
# Espérance du payoff (mesure physique)
# ----------------------------------------------------------------------------

def expected_payoff_call(spot: float, strike: float, mu: float, sigma: float, t_years: float) -> float:
    """E[max(S_T - K, 0)] sous la mesure physique.

        S0·e^(mu·T)·N(d1) - K·N(d2)

    Avec mu = r, cette valeur ACTUALISÉE (× e^(-rT)) est exactement le prix
    Black-Scholes : c'est la même intégrale, seule la mesure change."""
    if spot <= 0:
        return 0.0
    if strike <= 0:
        # Un call de strike nul ou négatif est toujours exercé : son payoff
        # est le sous-jacent lui-même. Le passage par ln(S0/K) serait indéfini.
        return _degenerate_spot(spot, mu, t_years)
    if t_years <= 0 or sigma <= 0:
        return max(_degenerate_spot(spot, mu, t_years) - strike, 0.0)

    d1, d2 = _d1_d2(spot, strike, mu, sigma, t_years)
    return spot * math.exp(mu * t_years) * _norm_cdf(d1) - strike * _norm_cdf(d2)


def expected_payoff_put(spot: float, strike: float, mu: float, sigma: float, t_years: float) -> float:
    """E[max(K - S_T, 0)] sous la mesure physique.

        K·N(-d2) - S0·e^(mu·T)·N(-d1)
    """
    if strike <= 0:
        return 0.0  # un put de strike nul ne paie jamais
    if spot <= 0:
        return strike
    if t_years <= 0 or sigma <= 0:
        return max(strike - _degenerate_spot(spot, mu, t_years), 0.0)

    d1, d2 = _d1_d2(spot, strike, mu, sigma, t_years)
    return strike * _norm_cdf(-d2) - spot * math.exp(mu * t_years) * _norm_cdf(-d1)


def expected_payoff(
    spot: float, strike: float, mu: float, sigma: float, t_years: float, option_type: str,
) -> float:
    if option_type == CALL:
        return expected_payoff_call(spot, strike, mu, sigma, t_years)
    if option_type == PUT:
        return expected_payoff_put(spot, strike, mu, sigma, t_years)
    raise ValueError(f"option_type attend 'CALL' ou 'PUT', reçu {option_type!r}.")


# ----------------------------------------------------------------------------
# Second moment et variance du payoff
# ----------------------------------------------------------------------------

def second_moment_payoff_call(spot: float, strike: float, mu: float, sigma: float, t_years: float) -> float:
    """E[max(S_T - K, 0)²].

    Développement de (S_T - K)² sur l'événement {S_T > K}, terme à terme :

        E[S_T²·1] = S0²·e^((2mu + sigma²)T)·N(d1 + sigma√T)
        E[S_T ·1] = S0 ·e^(mu·T)          ·N(d1)
        E[    1 ] =                        N(d2)

    d'où S0²·e^((2mu+sigma²)T)·N(d1+sigma√T) - 2K·S0·e^(mu·T)·N(d1) + K²·N(d2).
    Le premier exposant vient de 2m + 2s² avec m = (mu - sigma²/2)T et
    s² = sigma²T, et l'argument décalé de +sigma√T du changement de mesure
    induit par le carré."""
    if spot <= 0:
        return 0.0
    if strike <= 0:
        moment = spot * spot * math.exp((2 * mu + sigma * sigma) * max(t_years, 0.0))
        return moment
    if t_years <= 0 or sigma <= 0:
        return max(_degenerate_spot(spot, mu, t_years) - strike, 0.0) ** 2

    sigma_sqrt_t = sigma * math.sqrt(t_years)
    d1, d2 = _d1_d2(spot, strike, mu, sigma, t_years)
    return (
        spot * spot * math.exp((2 * mu + sigma * sigma) * t_years) * _norm_cdf(d1 + sigma_sqrt_t)
        - 2 * strike * spot * math.exp(mu * t_years) * _norm_cdf(d1)
        + strike * strike * _norm_cdf(d2)
    )


def second_moment_payoff_put(spot: float, strike: float, mu: float, sigma: float, t_years: float) -> float:
    """E[max(K - S_T, 0)²] -- symétrique du call, sur l'événement {S_T < K} :

        K²·N(-d2) - 2K·S0·e^(mu·T)·N(-d1) + S0²·e^((2mu+sigma²)T)·N(-(d1+sigma√T))
    """
    if strike <= 0:
        return 0.0
    if spot <= 0:
        return strike * strike
    if t_years <= 0 or sigma <= 0:
        return max(strike - _degenerate_spot(spot, mu, t_years), 0.0) ** 2

    sigma_sqrt_t = sigma * math.sqrt(t_years)
    d1, d2 = _d1_d2(spot, strike, mu, sigma, t_years)
    return (
        strike * strike * _norm_cdf(-d2)
        - 2 * strike * spot * math.exp(mu * t_years) * _norm_cdf(-d1)
        + spot * spot * math.exp((2 * mu + sigma * sigma) * t_years) * _norm_cdf(-(d1 + sigma_sqrt_t))
    )


def _variance_from_moments(second_moment: float, mean: float) -> float:
    """Var = E[X²] - E[X]², rabotée à zéro.

    La soustraction est une différence de deux grands nombres presque égaux
    dès que le payoff est quasi déterministe (strike très dans ou très hors de
    la monnaie) : elle rend régulièrement de petites valeurs négatives de
    l'ordre de 1e-16 en double précision. Les laisser passer ferait planter la
    racine carrée du Sharpe sur des cas parfaitement légitimes."""
    return max(second_moment - mean * mean, 0.0)


def variance_payoff_call(spot: float, strike: float, mu: float, sigma: float, t_years: float) -> float:
    return _variance_from_moments(
        second_moment_payoff_call(spot, strike, mu, sigma, t_years),
        expected_payoff_call(spot, strike, mu, sigma, t_years),
    )


def variance_payoff_put(spot: float, strike: float, mu: float, sigma: float, t_years: float) -> float:
    return _variance_from_moments(
        second_moment_payoff_put(spot, strike, mu, sigma, t_years),
        expected_payoff_put(spot, strike, mu, sigma, t_years),
    )


def variance_payoff(
    spot: float, strike: float, mu: float, sigma: float, t_years: float, option_type: str,
) -> float:
    if option_type == CALL:
        return variance_payoff_call(spot, strike, mu, sigma, t_years)
    if option_type == PUT:
        return variance_payoff_put(spot, strike, mu, sigma, t_years)
    raise ValueError(f"option_type attend 'CALL' ou 'PUT', reçu {option_type!r}.")


# ----------------------------------------------------------------------------
# Risque BAISSIER (semi-écart-type sous le seuil de rentabilité)
# ----------------------------------------------------------------------------
#
# POURQUOI PAS L'ÉCART-TYPE. Mesurer le risque d'un contrat par l'écart-type de
# son payoff conduit à un optimum DÉGÉNÉRÉ : le Sharpe ainsi défini est
# monotone en K et choisit toujours le strike le plus profondément dans la
# monnaie de la grille, quelle que soit la dérive. La raison est structurelle --
# très dans la monnaie, le payoff vaut S_T - K, donc son écart-type est celui du
# sous-jacent, CONSTANT en K, pendant que l'espérance nette, elle, croît quand K
# baisse. Mesuré sur S0=100, sigma=30%, T=2 ans, mu=20% : l'optimum reste collé
# à la borne basse pour toute grille de +/-2 à +/-8 sigma, et le Sharpe du
# contrat converge par en dessous vers celui de l'action seule (0,7366 contre
# 0,7424 pour un achat comptant). Autrement dit, le critère dit « n'achète pas
# d'option, achète l'action » -- réponse cohérente, mais qui ne sélectionne rien.
#
# La cause est que l'écart-type est SYMÉTRIQUE : il compte la dispersion à la
# hausse comme un risque, et surtout il ignore ce qui fait l'intérêt d'une
# option longue, à savoir que la perte est PLAFONNÉE à la prime. Le semi-écart-
# type sous le seuil de rentabilité ne compte que les scénarios où le contrat
# perd de l'argent, et voit donc ce plafond.
#
#     risque = racine( E[ max(prime - payoff, 0)² ] )
#
# soit le dénominateur d'un ratio de Sortino dont la cible est le seuil de
# rentabilité (et non zéro : le point mort d'une option longue est la prime
# payée, pas le payoff nul).

def downside_semivariance_call(
    spot: float, strike: float, mu: float, sigma: float, t_years: float, premium: float,
) -> float:
    """E[max(prime - payoff, 0)²] pour un CALL, en forme fermée.

    Le seuil de rentabilité est B = K + prime : au-dessus, le contrat gagne.
    Sous ce seuil, le manque à gagner prend deux formes, d'où deux morceaux :

        S_T <= K      : le contrat expire sans valeur, perte = prime (constante)
        K < S_T <= B  : perte = prime - (S_T - K) = B - S_T (linéaire)

    Le second morceau se développe en (B - S_T)² et s'exprime dans les moments
    tronqués évalués en B et en K -- même intégrale, deux bornes."""
    if premium <= 0:
        return 0.0
    if strike <= 0:
        strike = 0.0
    breakeven = strike + premium

    m0_k, m1_k, m2_k = truncated_moments_below(spot, strike, mu, sigma, t_years)
    m0_b, m1_b, m2_b = truncated_moments_below(spot, breakeven, mu, sigma, t_years)

    perte_totale = premium * premium * m0_k
    perte_partielle = (
        (m2_b - m2_k)
        - 2 * breakeven * (m1_b - m1_k)
        + breakeven * breakeven * (m0_b - m0_k)
    )
    return max(perte_totale + perte_partielle, 0.0)


def downside_semivariance_put(
    spot: float, strike: float, mu: float, sigma: float, t_years: float, premium: float,
) -> float:
    """E[max(prime - payoff, 0)²] pour un PUT -- symétrique du call.

    Seuil de rentabilité B = K - prime, atteint par le BAS :

        S_T >= K      : perte = prime (le contrat expire sans valeur)
        B <= S_T < K  : perte = prime - (K - S_T) = S_T - B

    Quand la prime dépasse le strike, B est négatif : le contrat ne peut
    structurellement jamais être rentable (le payoff maximal, K, est déjà
    inférieur à la mise). La convention (0,0,0) de `truncated_moments_below`
    sous un niveau négatif fait alors porter la formule sur tout le support,
    sans cas particulier."""
    if premium <= 0 or strike <= 0:
        return 0.0
    breakeven = strike - premium

    m0_k, m1_k, m2_k = truncated_moments_below(spot, strike, mu, sigma, t_years)
    m0_b, m1_b, m2_b = truncated_moments_below(spot, breakeven, mu, sigma, t_years)

    perte_totale = premium * premium * (1.0 - m0_k)
    perte_partielle = (
        (m2_k - m2_b)
        - 2 * breakeven * (m1_k - m1_b)
        + breakeven * breakeven * (m0_k - m0_b)
    )
    return max(perte_totale + perte_partielle, 0.0)


def downside_semivariance(
    spot: float, strike: float, mu: float, sigma: float, t_years: float, option_type: str,
    premium: float,
) -> float:
    if option_type == CALL:
        return downside_semivariance_call(spot, strike, mu, sigma, t_years, premium)
    if option_type == PUT:
        return downside_semivariance_put(spot, strike, mu, sigma, t_years, premium)
    raise ValueError(f"option_type attend 'CALL' ou 'PUT', reçu {option_type!r}.")


def probability_of_profit(
    spot: float, strike: float, mu: float, sigma: float, t_years: float, option_type: str,
    premium: float,
) -> float:
    """P(payoff > prime) : probabilité que le contrat soit rentable à
    l'échéance, sous la mesure physique.

    Sert de GARDE-FOU au critère de Sortino, pas de critère de sélection. Un
    ratio de Sortino peut être flatté par une queue de distribution : très hors
    de la monnaie, l'espérance nette et le risque baissier tendent tous deux
    vers zéro, et leur RAPPORT peut rester élevé alors que le contrat ne paie
    presque jamais. Or c'est précisément dans cette queue que l'hypothèse
    lognormale et l'estimation de `mu` sont les moins fiables. Exiger un
    minimum de probabilité de gain interdit d'acheter un ratio qui n'existe que
    dans la queue du modèle."""
    breakeven = strike + premium if option_type == CALL else strike - premium
    if option_type == CALL:
        return 1.0 - truncated_moments_below(spot, breakeven, mu, sigma, t_years)[0]
    if option_type == PUT:
        return truncated_moments_below(spot, breakeven, mu, sigma, t_years)[0]
    raise ValueError(f"option_type attend 'CALL' ou 'PUT', reçu {option_type!r}.")


# ----------------------------------------------------------------------------
# Taux de croissance log-optimal (Kelly)
# ----------------------------------------------------------------------------
#
# POURQUOI NI SHARPE NI SORTINO. Les deux ratios ont un optimum DÉGÉNÉRÉ, aux
# deux bouts opposés, et la cause est commune : un ratio invariant d'échelle
# est gouverné, dans la queue de distribution, par le rapport de vraisemblance
# entre la mesure physique et la mesure risque-neutre, qui est monotone en K.
#
#   - Écart-type : il croît plus vite que la prime quand on s'éloigne dans le
#     hors-la-monnaie, donc le Sharpe s'effondre et l'optimum FUIT VERS LE
#     DEDANS de la monnaie. Mesuré : optimum collé à la borne basse pour toute
#     grille de +/-2 à +/-8 sigma, quelle que soit la dérive.
#   - Semi-écart-type baissier : il est PLAFONNÉ PAR LA PRIME (la perte d'une
#     option longue est bornée par la mise), donc le Sortino se comporte comme
#     espérance/prime, qui diverge, et l'optimum FUIT VERS LE DEHORS. Mesuré :
#     ratio de 20,3 sur un contrat qui ne paie que dans 0,1 % des scénarios.
#
# Un plancher de probabilité de gain ne corrige pas cela, il ne fait que coller
# l'optimum à la contrainte -- et le fait basculer d'un extrême à l'autre :
# à mu = 20 %, un plancher de 30 % choisit K = 1,53·S0, un plancher de 40 %
# choisit K = 0,28·S0. Le seuil décide du strike ; ce n'est pas une sélection.
#
# LE CRITÈRE DE KELLY N'A PAS CE DÉFAUT, par construction et non par réglage.
# On maximise le taux de croissance logarithmique du capital :
#
#     g*(K) = max_f  E[ log(1 + f·R) ],   R = payoff/prime - 1
#
# La perte totale (R = -1) a une probabilité STRICTEMENT POSITIVE, et
# log(1 - f) diverge : le critère refuse donc de charger un contrat qui ne paie
# presque jamais, ce qui borne le côté hors-la-monnaie. Et le levier atteignable
# borne le côté dans-la-monnaie. Résultat mesuré : le strike optimal se déplace
# CONTINÛMENT avec la conviction (à sigma = 20 % : K*/S0 = 0,43 pour mu = 6 %,
# puis 0,99 pour mu = 20 %, puis 1,52 pour mu = 35 %), au lieu de sauter d'un
# bord à l'autre.
#
# ATTENTION : c'est l'optimisation CONJOINTE en (K, f) qui est bien posée. À f
# petit et FIXÉ, log(1 + f·R) ~ f·E[R] et le critère redevient « espérance par
# dollar de prime », donc redivergent vers le hors-la-monnaie. La fraction doit
# rester libre pendant la sélection ; le plafonnement éventuel (demi-Kelly, ou
# les poids du portefeuille) s'applique APRÈS, au dimensionnement.


@lru_cache(maxsize=8)
def _legendre_nodes(n_nodes: int) -> tuple:
    """Nœuds et poids de Gauss-Legendre sur [-1, 1], mis en cache.

    Quadrature DÉTERMINISTE, et c'est la raison de son emploi : le taux de
    croissance de Kelly n'a pas de primitive, mais l'estimer par Monte-Carlo
    transmettrait le bruit d'échantillonnage au CHOIX du strike, qui changerait
    d'un run à l'autre sans qu'aucune donnée n'ait bougé. Une quadrature rend
    exactement la même valeur à chaque appel."""
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    return x, w


def _return_nodes(
    spot: float, strike: float, mu: float, sigma: float, t_years: float, option_type: str,
    premium: float, n_nodes: int,
) -> Optional[tuple]:
    """Discrétisation du rendement R = payoff/prime - 1 en deux morceaux :

        (probabilité de perte TOTALE, poids de quadrature, valeurs de R)

    Le payoff a un POINT ANGULEUX au strike, et une quadrature gaussienne
    converge mal sur un intégrande non lisse. On découpe donc analytiquement :

      - Du côté où l'option expire sans valeur, R vaut exactement -1 : c'est un
        ATOME, dont la probabilité est connue en forme fermée (N(±z_K)). Aucune
        quadrature n'est nécessaire, et surtout aucune ne peut le lisser par
        erreur -- or c'est précisément cet atome qui borne le critère.
      - Du côté lucratif, R est analytique, et Gauss-Legendre y converge
        géométriquement.

    None si le contrat ne peut structurellement jamais payer (toute la masse
    est sur l'atome de perte totale)."""
    if premium <= 0 or spot <= 0 or strike <= 0 or sigma <= 0 or t_years <= 0:
        return None

    sigma_sqrt_t = sigma * math.sqrt(t_years)
    # Écart-type standardisé du seuil : S_T <= K  <=>  z <= z_k.
    z_k = (math.log(strike / spot) - (mu - 0.5 * sigma * sigma) * t_years) / sigma_sqrt_t

    if option_type == CALL:
        lo, hi = z_k, _Z_MAX          # l'option paie au-DESSUS du strike
        prob_perte = _norm_cdf(z_k)
    elif option_type == PUT:
        lo, hi = -_Z_MAX, z_k         # l'option paie au-DESSOUS du strike
        prob_perte = 1.0 - _norm_cdf(z_k)
    else:
        raise ValueError(f"option_type attend 'CALL' ou 'PUT', reçu {option_type!r}.")

    if hi <= lo:
        return None  # la région lucrative est hors du support numérique

    x, w = _legendre_nodes(n_nodes)
    demi = 0.5 * (hi - lo)
    z = demi * x + 0.5 * (hi + lo)
    # Poids = poids de Gauss-Legendre x densité normale x jacobien du changement
    # de variable. Leur somme vaut la probabilité de la région lucrative.
    poids = w * demi * _INV_SQRT_2PI * np.exp(-0.5 * z * z)

    s_t = spot * np.exp((mu - 0.5 * sigma * sigma) * t_years + sigma_sqrt_t * z)
    payoff = (s_t - strike) if option_type == CALL else (strike - s_t)
    return prob_perte, poids, payoff / premium - 1.0


def kelly_optimum(
    spot: float, strike: float, mu: float, sigma: float, t_years: float, option_type: str,
    premium: float, n_nodes: int = config.OPTIONS_EV_QUADRATURE_NODES,
) -> Optional[tuple]:
    """(fraction de Kelly f*, taux de croissance g*) du contrat.

    f* est la part du capital qu'il serait log-optimal d'engager sur ce
    contrat ; g* = E[log(1 + f*·R)] est le taux de croissance qui en résulte.
    C'est g* qui sert de score de sélection entre strikes -- une grandeur en
    unités de croissance, pas un ratio sans dimension, donc comparable d'un
    strike à l'autre sans être invariant d'échelle.

    La dérivée dg/df = E[R/(1 + f·R)] est strictement DÉCROISSANTE en f (g est
    concave), vaut E[R] en f = 0 et tend vers -infini quand f tend vers 1 à
    cause de l'atome de perte totale. Le maximum est donc unique et s'obtient
    par bissection sur la dérivée -- pas de recherche par section dorée, pas de
    dépendance à un point de départ.

    None si le contrat n'a pas d'espérance nette positive : c'est exactement la
    condition E[R] > 0, c'est-à-dire E[payoff] > prime. La règle « espérance
    positive obligatoire » n'est donc pas un filtre ajouté par-dessus Kelly,
    c'est la condition d'existence d'une mise log-optimale non nulle."""
    noeuds = _return_nodes(spot, strike, mu, sigma, t_years, option_type, premium, n_nodes)
    if noeuds is None:
        return None
    prob_perte, poids, rendements = noeuds

    def derivee(f: float) -> float:
        return float(np.sum(poids * rendements / (1.0 + f * rendements))) - prob_perte / (1.0 - f)

    if derivee(0.0) <= 0.0:
        return None  # espérance nette négative : aucune mise ne fait croître le capital

    lo, hi = 0.0, _F_MAX
    if derivee(hi) > 0.0:
        # Pas d'atome de perte détectable (très dans la monnaie) : le levier
        # log-optimal sature la borne numérique. Le contrat est alors une
        # position à effet de levier sur le sous-jacent, ce que g* traduira.
        f_star = hi
    else:
        for _ in range(80):  # bissection : 80 tours ramènent l'intervalle sous 1e-24
            milieu = 0.5 * (lo + hi)
            if derivee(milieu) > 0.0:
                lo = milieu
            else:
                hi = milieu
        f_star = 0.5 * (lo + hi)

    croissance = float(np.sum(poids * np.log1p(f_star * rendements)))
    if prob_perte > 0.0:
        croissance += prob_perte * math.log1p(-f_star)
    if not math.isfinite(croissance):
        return None
    return f_star, croissance


def log_growth_rate(
    spot: float, strike: float, mu: float, sigma: float, t_years: float, option_type: str,
    premium: float, n_nodes: int = config.OPTIONS_EV_QUADRATURE_NODES,
) -> Optional[float]:
    """Taux de croissance log-optimal g* seul (voir kelly_optimum)."""
    optimum = kelly_optimum(spot, strike, mu, sigma, t_years, option_type, premium, n_nodes)
    return None if optimum is None else optimum[1]


# ----------------------------------------------------------------------------
# Sharpe du contrat et choix du strike
# ----------------------------------------------------------------------------

def sharpe_contract(
    spot: float, strike: float, mu: float, sigma: float, t_years: float, option_type: str,
    premium: float,
) -> Optional[float]:
    """(E[payoff] - prime) / écart-type(payoff).

    `premium` est le prix de MARCHÉ du contrat (Black-Scholes au taux sans
    risque), pas une valeur recalculée sous `mu` : c'est la mise réellement
    engagée. Le numérateur est donc l'espérance de gain NET.

    None quand le ratio n'a pas de sens : payoff déterministe (écart-type nul,
    ce qui arrive à T = 0 ou sigma = 0) ou paramètres non finis. Un appelant
    ne doit pas classer des strikes sur un ratio indéfini -- retourner +inf
    ferait gagner ces cas dégénérés à tous les coups."""
    variance = variance_payoff(spot, strike, mu, sigma, t_years, option_type)
    if not math.isfinite(variance) or variance <= 0.0:
        return None
    esperance = expected_payoff(spot, strike, mu, sigma, t_years, option_type)
    if not math.isfinite(esperance) or not math.isfinite(premium):
        return None
    return (esperance - premium) / math.sqrt(variance)


def strike_grid(
    spot: float, sigma: float, t_years: float,
    n_sigma: float = config.OPTIONS_EV_STRIKE_GRID_N_SIGMA,
    step_sigma: float = config.OPTIONS_EV_STRIKE_GRID_STEP_SIGMA,
) -> List[float]:
    """Strikes candidats de S0·exp(-n·sigma√T) à S0·exp(+n·sigma√T), par pas de
    `step_sigma`·sigma·√T en LOG-prix.

    Adaptative par construction : la largeur de la grille suit la volatilité et
    la maturité, donc elle couvre toujours la même portion de la distribution
    du sous-jacent. Une grille en pourcentage fixe du spot serait trop large
    sur un titre calme à trois mois et trop étroite sur un titre nerveux à deux
    ans -- dans les deux cas, le strike optimal se retrouverait sur un bord de
    grille, c'est-à-dire non optimal.

    Grille vide si le sous-jacent ou la volatilité sont dégénérés : sans aléa,
    aucun strike ne se distingue d'un autre par son Sharpe."""
    if spot <= 0 or sigma <= 0 or t_years <= 0 or step_sigma <= 0 or n_sigma <= 0:
        return []
    largeur = sigma * math.sqrt(t_years)
    pas = step_sigma * largeur
    n_pas = int(round(n_sigma / step_sigma))
    return [spot * math.exp(k * pas) for k in range(-n_pas, n_pas + 1)]


def optimal_strike(
    spot: float, mu: float, sigma: float, t_years: float, option_type: str,
    strikes: Iterable[float],
    premium_fn: Optional[Callable[[float], float]] = None,
    r: Optional[float] = None,
    q: float = 0.0,
    criterion: str = "kelly",
    n_nodes: int = config.OPTIONS_EV_QUADRATURE_NODES,
) -> Optional[dict]:
    """Meilleur strike parmi `strikes`, sous contrainte d'espérance nette
    strictement positive.

    `criterion` vaut "kelly" (défaut) ou "sharpe" :

      - "kelly" maximise le taux de croissance log-optimal g*. C'est le seul
        des trois critères dont l'optimum ne fuit pas systématiquement vers un
        bord de grille -- voir le pavé de commentaires de la section Kelly.
      - "sharpe" maximise (E[payoff] - prime)/écart-type. CONSERVÉ POUR
        COMPARAISON UNIQUEMENT : son optimum est monotone en K et se colle
        toujours au strike le plus dans la monnaie. Il sert de témoin dans les
        tests et dans le script de comparaison, pas de critère de production.

    `premium_fn(strike) -> prime` permet à l'appelant d'injecter un prix coté
    réel ; par défaut, la prime est le prix Black-Scholes au taux sans risque
    `r` (config.RISK_FREE_RATE si non fourni) et au rendement de dividende `q`.
    Dans les deux cas, la prime est un prix de MARCHÉ : jamais recalculée sous
    `mu`, sinon l'espérance nette serait nulle par construction et plus rien ne
    classerait les strikes.

    None si aucun candidat n'a d'espérance nette positive -- la stratégie
    appelante doit alors ÉCARTER la ligne du portefeuille du jour plutôt que
    d'ouvrir au moins mauvais strike. Un pari dont on calcule soi-même qu'il
    perd en moyenne n'a aucune raison d'être pris.

    Le dictionnaire rendu porte, outre le strike retenu, tous les diagnostics
    du contrat choisi : ils servent aux compteurs agrégés du backtest et
    évitent à l'appelant de refaire les mêmes calculs."""
    if criterion not in ("kelly", "sharpe"):
        raise ValueError(f"criterion attend 'kelly' ou 'sharpe', reçu {criterion!r}.")
    if r is None:
        r = config.RISK_FREE_RATE

    def prime_de(strike: float) -> float:
        if premium_fn is not None:
            return premium_fn(strike)
        return options_pricing.bs_price(spot, strike, t_years, sigma, option_type, r=r, q=q)

    meilleur: Optional[dict] = None
    n_candidats = 0
    n_esperance_negative = 0

    for strike in strikes:
        if strike is None or not math.isfinite(strike) or strike <= 0:
            continue
        n_candidats += 1

        premium = prime_de(strike)
        if not math.isfinite(premium) or premium <= 0:
            # Sans mise, aucun rendement n'est définissable (et un prix nul est
            # le signe d'un strike hors de tout usage).
            continue

        esperance = expected_payoff(spot, strike, mu, sigma, t_years, option_type)
        edge = esperance - premium
        if edge <= 0:
            n_esperance_negative += 1
            continue

        if criterion == "kelly":
            optimum = kelly_optimum(spot, strike, mu, sigma, t_years, option_type, premium, n_nodes)
            if optimum is None:
                continue
            fraction, score = optimum
        else:
            score = sharpe_contract(spot, strike, mu, sigma, t_years, option_type, premium)
            if score is None:
                continue
            fraction = None

        if meilleur is None or score > meilleur["score"]:
            variance = variance_payoff(spot, strike, mu, sigma, t_years, option_type)
            semi_variance = downside_semivariance(
                spot, strike, mu, sigma, t_years, option_type, premium)
            meilleur = {
                "strike": strike,
                "criterion": criterion,
                "score": score,
                "kelly_fraction": fraction,
                "log_growth": score if criterion == "kelly" else None,
                "expected_payoff": esperance,
                "premium": premium,
                "edge": edge,
                "std_payoff": math.sqrt(variance),
                "downside_semideviation": math.sqrt(semi_variance),
                "probability_of_profit": probability_of_profit(
                    spot, strike, mu, sigma, t_years, option_type, premium),
            }

    if meilleur is None:
        return None
    meilleur["n_candidates"] = n_candidats
    meilleur["n_negative_edge"] = n_esperance_negative
    return meilleur


def convergence_drift(spot: float, target_value: float, t_years: float, fraction: float) -> Optional[float]:
    """Dérive annualisée `mu` correspondant à une convergence PARTIELLE du
    cours vers sa valeur théorique :

        mu = fraction · ln(V / S0) / T

    `fraction = 1` suppose que le cours atteint exactement la valeur théorique
    à l'échéance -- hypothèse que rien n'étaye et qui transformerait chaque
    écart de valorisation en gain certain. `fraction = 0.5` (le défaut de la
    stratégie appelante) ne suppose que la moitié du chemin, ce qui reste une
    hypothèse mais une hypothèse mesurée, et rend le paramètre optimisable au
    lieu de le laisser implicite.

    None si les entrées ne permettent pas de dérive définie (cours ou valeur
    non strictement positifs, maturité nulle)."""
    if spot <= 0 or target_value <= 0 or t_years <= 0:
        return None
    return fraction * math.log(target_value / spot) / t_years
