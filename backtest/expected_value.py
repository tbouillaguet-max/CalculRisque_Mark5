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
from typing import Callable, Iterable, List, Optional, Sequence

import config
from backtest import options_pricing

_INV_SQRT_2 = 1.0 / math.sqrt(2.0)

CALL = "CALL"
PUT = "PUT"


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
    n_sigma: float = 3.0, step_sigma: float = 0.25,
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
) -> Optional[dict]:
    """Strike qui maximise le Sharpe du contrat parmi `strikes`, sous
    contrainte d'espérance nette strictement positive.

    `premium_fn(strike) -> prime` permet à l'appelant d'injecter un prix
    coté réel ; par défaut, la prime est le prix Black-Scholes au taux sans
    risque `r` (config.RISK_FREE_RATE si non fourni) et au rendement de
    dividende `q`. Dans les deux cas, la prime est un prix de MARCHÉ : jamais
    recalculée sous `mu`, sinon l'espérance nette serait nulle par
    construction et le Sharpe ne classerait plus rien.

    None si aucun candidat n'a d'espérance nette positive -- la stratégie
    appelante doit alors ÉCARTER la ligne du portefeuille du jour plutôt que
    d'ouvrir au moins mauvais strike. Un pari dont on calcule soi-même qu'il
    perd en moyenne n'a aucune raison d'être pris."""
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
            # Sans mise, le ratio gain/risque n'est pas comparable aux autres
            # (et un prix nul est le signe d'un strike hors de tout usage).
            continue

        esperance = expected_payoff(spot, strike, mu, sigma, t_years, option_type)
        edge = esperance - premium
        if edge <= 0:
            n_esperance_negative += 1
            continue

        sharpe = sharpe_contract(spot, strike, mu, sigma, t_years, option_type, premium)
        if sharpe is None:
            continue

        if meilleur is None or sharpe > meilleur["sharpe"]:
            meilleur = {
                "strike": strike, "sharpe": sharpe, "expected_payoff": esperance,
                "premium": premium, "edge": edge,
                "std_payoff": math.sqrt(variance_payoff(spot, strike, mu, sigma, t_years, option_type)),
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
