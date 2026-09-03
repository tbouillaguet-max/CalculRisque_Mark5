"""
Pricer Black-Scholes-MERTON (dividende continu `q`, taux sans risque `r`
paramétrable par année) et volatilité réalisée, utilisés par
backtest/options_engine.py quand aucun snapshot réel n'est disponible (voir
data_loader.find_real_option_snapshot). Formules et
conventions de signe identiques à 08_recuperation_options.py (_bs_price /
_bs_greeks), dupliquées ici plutôt qu'importées : 08 charge ib_insync au
niveau module, une dépendance dont le backtest n'a pas besoin.

Les deux défauts (q=0, r=config.RISK_FREE_RATE) reproduisent EXACTEMENT
l'ancien comportement : c'est l'appelant qui décide de fournir mieux, et
options_engine le fait désormais à partir du secteur de l'entreprise et de
l'année simulée.

CE QUI N'EST TOUJOURS PAS MODÉLISÉ : le risque de VOLATILITÉ. Le repricing
quotidien se fait à volatilité figée (vol_mode="frozen") ou suivant la
volatilité RÉALISÉE (vol_mode="rolling"), jamais suivant l'implicite -- le
pipeline ne collecte aucune surface de volatilité historique. Une position
d'option longue est pourtant longue de vega : un effondrement de la volatilité
implicite lui fait perdre de l'argent même quand le sous-jacent va dans son
sens, et ce P&L-là n'apparaît nulle part dans les résultats. Voir
l'avertissement au démarrage de 10_backtest_options.py.

Implémentation SCALAIRE en `math` pur plutôt que scipy.stats.norm : le moteur
reprice chaque position ouverte chaque jour de bourse, soit des centaines de
milliers d'appels unitaires par run. `norm.cdf` passe par la machinerie
générique de scipy (validation d'arguments, broadcasting) qui coûte environ
cent fois le calcul lui-même sur un scalaire, alors que `math.erfc` donne la
même valeur à la précision machine près.
"""

from __future__ import annotations

import math

import numpy as np

import config

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x * _INV_SQRT_2)


def _norm_pdf(x: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def quoted_implied_vol(
    realized_vol: float, spot: float, strike: float, t_years: float,
    spread: float | None = None, skew_slope: float | None = None,
) -> float:
    """Volatilité IMPLICITE que le marché coterait pour ce contrat, à partir de
    la volatilité RÉALISÉE du titre.

        z  = ln(K / S) / (sigma_atm · racine(T))      (log-moneyness normalisée)
        IV = (réalisée + spread) + pente · max(-z, 0)

    C'est la grandeur à laquelle on ACHÈTE ; la réalisée est celle à laquelle le
    titre BOUGE. Les confondre revenait à s'offrir la prime de risque de
    variance et tout le skew (cf. config.OPTIONS_IMPLIED_VOL_SPREAD pour le
    raisonnement et les références).

    PAS D'ARGUMENT option_type, et c'est un point de correction, pas un oubli :
    par parité call-put, un call et un put de même strike et même échéance ont
    la MÊME volatilité implicite. Le skew est une propriété du sous-jacent
    indexée par le STRIKE. Un put hors de la monnaie coûte cher parce que son
    strike est bas, pas parce que c'est un put -- et c'est bien ce que cette
    formule reproduit.

    POURQUOI LE SKEW EST BRIDÉ D'UN SEUL CÔTÉ (le `max(-z, 0)`). Un skew
    actions réel est décroissant des deux côtés : les strikes hauts se paient
    MOINS que la monnaie. Le reproduire tel quel ici serait pourtant une faute,
    parce que le reste du module modélise S_T par une lognormale à volatilité
    UNIQUE. Une surface Q inclinée face à une loi P plate rend, par pure
    construction, les calls hors de la monnaie « bon marché » : on fabriquerait
    un edge au bord haut de la grille qui ne doit rien à la thèse de
    valorisation -- constaté en test, une thèse à +0,5 % ouvrait une position à
    22 fois le cours.

    Le vrai skew traduit d'ailleurs pour partie une asymétrie de la loi
    physique elle-même (risque de décrochage), et non une pure prime de risque.
    L'attribuer entièrement à la prime, c'est affirmer que les calls lointains
    sont réellement sous-payés -- affirmation que ce module n'a pas les moyens
    de soutenir, faute de modéliser une P asymétrique.

    On garde donc la seule moitié qui est un COÛT : au-dessus de la monnaie, la
    volatilité cotée reste au niveau ATM ; en dessous, elle monte. La correction
    ne peut ainsi que rendre les options PLUS chères, jamais moins -- ce qui est
    la bonne direction pour un correctif dont l'objet est de retirer un excès
    d'optimisme, et cela règle intégralement le problème mesuré, qui était la
    sous-facturation de la jambe put.

    La volatilité à la monnaie sert de normalisateur à z plutôt que la
    volatilité finale : la définition serait sinon implicite en elle-même. Sur
    la plage de strikes réellement traitée, l'écart entre les deux
    normalisations est du second ordre.

    Repli sur `realized_vol` telle quelle dès qu'un paramètre rend la formule
    indéfinie (maturité nulle, volatilité ou prix non strictement positifs) :
    le pricing dégénéré est déjà géré par bs_price, ce n'est pas ici qu'il faut
    l'intercepter."""
    if spread is None:
        spread = config.OPTIONS_IMPLIED_VOL_SPREAD
    if skew_slope is None:
        skew_slope = config.OPTIONS_VOL_SKEW_SLOPE

    atm = realized_vol + spread
    if realized_vol <= 0 or atm <= 0 or t_years <= 0 or spot <= 0 or strike <= 0:
        return realized_vol
    if not skew_slope:
        return min(max(atm, config.OPTIONS_QUOTED_VOL_MIN), config.OPTIONS_QUOTED_VOL_MAX)

    z = math.log(strike / spot) / (atm * math.sqrt(t_years))
    quoted = atm + skew_slope * max(-z, 0.0)
    return min(max(quoted, config.OPTIONS_QUOTED_VOL_MIN), config.OPTIONS_QUOTED_VOL_MAX)


def bs_price(
    spot: float, strike: float, t_years: float, vol: float, option_type: str,
    r: float = config.RISK_FREE_RATE, q: float = 0.0,
) -> float:
    """Prix Black-Scholes-Merton. t_years<=0 ou vol<=0 -> valeur intrinsèque
    (option expirée ou dégénérée).

    `q` : rendement du dividende continu du sous-jacent. Sans lui (q=0, le
    défaut, qui reproduit exactement l'ancien comportement), les CALLS sont
    systématiquement surévalués et les PUTS sous-évalués -- d'autant plus que
    le rendement est élevé, donc précisément sur les secteurs qu'un signal
    "value" sélectionne. Voir config.SECTOR_DIVIDEND_YIELD pour le défaut
    sectoriel appliqué par le moteur, faute de donnée par titre."""
    if t_years <= 0 or vol <= 0:
        intrinsic = (spot - strike) if option_type == "CALL" else (strike - spot)
        return max(0.0, intrinsic)
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    discount = math.exp(-r * t_years)
    # Le spot est actualisé du rendement du dividende : le détenteur d'une
    # option ne touche pas les dividendes versés d'ici l'échéance.
    spot_discounted = spot * math.exp(-q * t_years)
    if option_type == "CALL":
        return spot_discounted * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
    return strike * discount * _norm_cdf(-d2) - spot_discounted * _norm_cdf(-d1)


def bs_greeks(
    spot: float, strike: float, t_years: float, vol: float, option_type: str,
    r: float = config.RISK_FREE_RATE, q: float = 0.0,
) -> dict:
    """delta, gamma, vega (pour +1 point de vol, convention IBKR), theta
    (par jour). t_years<=0 -> greeks nuls (position expirée, plus de valeur
    temps ni de sensibilité). `q` : voir bs_price."""
    if t_years <= 0 or vol <= 0:
        intrinsic_delta = 1.0 if option_type == "CALL" else -1.0
        in_the_money = (spot > strike) if option_type == "CALL" else (spot < strike)
        return {"delta": intrinsic_delta if in_the_money else 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    pdf_d1 = _norm_pdf(d1)
    dividend_discount = math.exp(-q * t_years)
    gamma = dividend_discount * pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * dividend_discount * pdf_d1 * sqrt_t / 100
    discount = math.exp(-r * t_years)
    # Le terme en q est la part de theta due au dividende : elle change de
    # signe entre call et put, comme le terme en r.
    theta_common = -spot * dividend_discount * pdf_d1 * vol / (2 * sqrt_t)
    if option_type == "CALL":
        delta = dividend_discount * _norm_cdf(d1)
        theta = (
            theta_common
            - r * strike * discount * _norm_cdf(d2)
            + q * spot * dividend_discount * _norm_cdf(d1)
        ) / 365
    else:
        delta = dividend_discount * (_norm_cdf(d1) - 1)
        theta = (
            theta_common
            + r * strike * discount * _norm_cdf(-d2)
            - q * spot * dividend_discount * _norm_cdf(-d1)
        ) / 365
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def realized_volatility(
    close_history: np.ndarray,
    lookback_days: int = config.OPTIONS_REALIZED_VOL_LOOKBACK_DAYS,
) -> float | None:
    """Volatilité annualisée des rendements log quotidiens sur les
    lookback_days derniers jours de COTATION de `close_history`.

    close_history : cours de clôture jusqu'à la date d'évaluation INCLUSE et
    jamais au-delà -- le découpage point-in-time est fait par l'appelant (cf.
    data_loader.PricePanel.close_history). Un tableau numpy plutôt qu'une
    Series pandas indexée par date : extraire une colonne datée du panel à
    chaque entrée en position dominait le temps de run.

    None si l'historique disponible est trop court pour être significatif.
    """
    past = close_history[~np.isnan(close_history)]
    if len(past) < max(lookback_days // 2, 10):
        return None
    window = past[-lookback_days:]
    if len(window) < 6:
        return None
    log_returns = np.diff(np.log(window))
    return float(log_returns.std(ddof=1) * np.sqrt(252))
