"""
Pricer Black-Scholes (pas de dividende, taux constant) et volatilité
réalisée, utilisés par backtest/options_engine.py quand aucun snapshot réel
n'est disponible (voir data_loader.find_real_option_snapshot). Formules et
conventions de signe identiques à 08_recuperation_options.py (_bs_price /
_bs_greeks), dupliquées ici plutôt qu'importées : 08 charge ib_insync au
niveau module, une dépendance dont le backtest n'a pas besoin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

import config


def bs_price(spot: float, strike: float, t_years: float, vol: float, option_type: str, r: float = config.RISK_FREE_RATE) -> float:
    """Prix Black-Scholes. t_years<=0 ou vol<=0 -> valeur intrinsèque (option
    expirée ou dégénérée)."""
    if t_years <= 0 or vol <= 0:
        intrinsic = (spot - strike) if option_type == "CALL" else (strike - spot)
        return max(0.0, intrinsic)
    d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * t_years) / (vol * np.sqrt(t_years))
    d2 = d1 - vol * np.sqrt(t_years)
    if option_type == "CALL":
        return spot * norm.cdf(d1) - strike * np.exp(-r * t_years) * norm.cdf(d2)
    return strike * np.exp(-r * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def bs_greeks(spot: float, strike: float, t_years: float, vol: float, option_type: str, r: float = config.RISK_FREE_RATE) -> dict:
    """delta, gamma, vega (pour +1 point de vol, convention IBKR), theta
    (par jour). t_years<=0 -> greeks nuls (position expirée, plus de valeur
    temps ni de sensibilité)."""
    if t_years <= 0 or vol <= 0:
        intrinsic_delta = 1.0 if option_type == "CALL" else -1.0
        in_the_money = (spot > strike) if option_type == "CALL" else (spot < strike)
        return {"delta": intrinsic_delta if in_the_money else 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * t_years) / (vol * np.sqrt(t_years))
    d2 = d1 - vol * np.sqrt(t_years)
    pdf_d1 = norm.pdf(d1)
    gamma = pdf_d1 / (spot * vol * np.sqrt(t_years))
    vega = spot * pdf_d1 * np.sqrt(t_years) / 100
    if option_type == "CALL":
        delta = norm.cdf(d1)
        theta = (-spot * pdf_d1 * vol / (2 * np.sqrt(t_years)) - r * strike * np.exp(-r * t_years) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (-spot * pdf_d1 * vol / (2 * np.sqrt(t_years)) + r * strike * np.exp(-r * t_years) * norm.cdf(-d2)) / 365
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def realized_volatility(
    close_prices: pd.Series,
    as_of: pd.Timestamp,
    lookback_days: int = config.OPTIONS_REALIZED_VOL_LOOKBACK_DAYS,
) -> float | None:
    """Volatilité annualisée des rendements log quotidiens sur les
    lookback_days derniers jours de COTATION précédant (et incluant) as_of --
    jamais après, pour rester point-in-time. close_prices : série indexée par
    date (déjà triée), typiquement une colonne du panel de prix de
    data_loader.build_price_panel. None si l'historique disponible est trop
    court pour être significatif."""
    past = close_prices.loc[:as_of].dropna()
    if len(past) < max(lookback_days // 2, 10):
        return None
    window = past.iloc[-lookback_days:]
    log_returns = np.log(window / window.shift(1)).dropna()
    if len(log_returns) < 5:
        return None
    return float(log_returns.std() * np.sqrt(252))
