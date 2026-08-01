"""
Pricer Black-Scholes (pas de dividende, taux constant) et volatilité
réalisée, utilisés par backtest/options_engine.py quand aucun snapshot réel
n'est disponible (voir data_loader.find_real_option_snapshot). Formules et
conventions de signe identiques à 08_recuperation_options.py (_bs_price /
_bs_greeks), dupliquées ici plutôt qu'importées : 08 charge ib_insync au
niveau module, une dépendance dont le backtest n'a pas besoin.

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


def bs_price(spot: float, strike: float, t_years: float, vol: float, option_type: str, r: float = config.RISK_FREE_RATE) -> float:
    """Prix Black-Scholes. t_years<=0 ou vol<=0 -> valeur intrinsèque (option
    expirée ou dégénérée)."""
    if t_years <= 0 or vol <= 0:
        intrinsic = (spot - strike) if option_type == "CALL" else (strike - spot)
        return max(0.0, intrinsic)
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    discount = math.exp(-r * t_years)
    if option_type == "CALL":
        return spot * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
    return strike * discount * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_greeks(spot: float, strike: float, t_years: float, vol: float, option_type: str, r: float = config.RISK_FREE_RATE) -> dict:
    """delta, gamma, vega (pour +1 point de vol, convention IBKR), theta
    (par jour). t_years<=0 -> greeks nuls (position expirée, plus de valeur
    temps ni de sensibilité)."""
    if t_years <= 0 or vol <= 0:
        intrinsic_delta = 1.0 if option_type == "CALL" else -1.0
        in_the_money = (spot > strike) if option_type == "CALL" else (spot < strike)
        return {"delta": intrinsic_delta if in_the_money else 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    pdf_d1 = _norm_pdf(d1)
    gamma = pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t / 100
    discount = math.exp(-r * t_years)
    if option_type == "CALL":
        delta = _norm_cdf(d1)
        theta = (-spot * pdf_d1 * vol / (2 * sqrt_t) - r * strike * discount * _norm_cdf(d2)) / 365
    else:
        delta = _norm_cdf(d1) - 1
        theta = (-spot * pdf_d1 * vol / (2 * sqrt_t) + r * strike * discount * _norm_cdf(-d2)) / 365
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
