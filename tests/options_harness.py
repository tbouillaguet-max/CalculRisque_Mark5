"""Petit banc d'essai pour OptionsBacktestEngine : cours synthétiques, une
stratégie triviale, aucun snapshot réel (tout est simulé par Black-Scholes).

Assemblé une fois ici plutôt que dans chaque test : construire le moteur
demande six objets dont deux DataFrames au schéma précis, et c'est ce coût
d'installation qui expliquait qu'aucun des bugs du bloc C n'avait de test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.data_loader import PricePanel
from backtest.options_engine import OptionsBacktestEngine
from backtest.strategies.options_base import OptionsStrategy


class StrategieFixe(OptionsStrategy):
    """Vise toujours les mêmes symboles, dans le même sens, à poids égaux --
    pour que ce soit le MOTEUR qu'un test observe, pas la stratégie."""

    def __init__(self, directions: dict[str, str], **params):
        super().__init__(**params)
        self.directions = directions

    def generate_option_targets(self, signals, current_positions):
        symbols = [s for s in signals["symbol"] if s in self.directions]
        if not symbols:
            return {}
        return {
            s: {"option_type": self.directions[s], "weight": 1.0 / len(symbols)}
            for s in symbols
        }


def cours(closes: dict[str, list[float]], start: str = "2020-01-01") -> PricePanel:
    """Panel où l'ouverture égale la clôture : les tests raisonnent sur un
    seul prix par jour, sans avoir à suivre le décalage clôture J /
    ouverture J+1 du moteur."""
    dates = pd.bdate_range(start=start, periods=len(next(iter(closes.values()))))
    close = pd.DataFrame(closes, index=dates, dtype=float)
    last_valid = close.apply(lambda col: col.last_valid_index())
    return PricePanel(close, close.copy(), last_valid)


def signaux(symbols, published_date: str, gap_pct: float = 50.0, theoretical: float = 150.0) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "symbol": s, "published_date": pd.Timestamp(published_date), "fiscal_year": 2019,
            "sector": "Technologie", "close": 100.0,
            "valuation_theoretical_per_share": theoretical, "source": "multiples",
            "gap_pct": gap_pct, "period_type": "FY",
        }
        for s in symbols
    ])


def moteur(price_panel: PricePanel, signal_events: pd.DataFrame, directions: dict[str, str], **kwargs) -> OptionsBacktestEngine:
    """Moteur avec des réglages neutres : pas de filtre momentum (les cours
    synthétiques n'ont pas 12 mois d'historique), pas de frais minimum
    parasites, contrats entiers comme en production."""
    defaults = dict(
        universe_history=None,
        fallback_universe_symbols=set(price_panel.close.columns),
        option_snapshots=pd.DataFrame(),
        strategy=StrategieFixe(directions),
        initial_capital=1_000_000.0,
        commission_per_contract=0.65,
        slippage_pct_of_premium=0.0,
        stop_loss_pct=-50.0,
        take_profit_pct=100.0,
        max_positions=10,
        momentum_min_pct=None,
        max_fee_pct_of_trade=None,
        fee_bump_max_extra_pct=None,
        min_resize_relative_pct=None,
    )
    defaults.update(kwargs)
    return OptionsBacktestEngine(price_panel=price_panel, signal_events=signal_events, **defaults)


def serie_plate(valeur: float, n: int) -> list[float]:
    return [valeur] * n


def serie_bruitee(depart: float, n: int, sigma: float = 0.015, graine: int = 0) -> list[float]:
    """Un cours strictement plat donne une volatilité réalisée nulle, donc un
    repli à 30% dans le moteur : du bruit rend les tests représentatifs."""
    rng = np.random.default_rng(graine)
    return list(depart * np.cumprod(1 + rng.normal(0.0, sigma, n)))
