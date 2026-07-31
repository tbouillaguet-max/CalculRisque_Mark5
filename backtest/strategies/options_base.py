"""
Interface commune aux stratégies OPTIONS, + registre par nom séparé de celui
des stratégies actions (backtest/strategies/base.py) : les deux types
d'instruments ont des mécaniques trop différentes (expiration, greeks,
prime) pour partager une seule interface -- voir backtest/options_engine.py
pour ce que le moteur gère uniformément (dimensionnement par delta, stop-loss/
take-profit sur la prime, positions gelées, coûts).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

OPTIONS_STRATEGY_REGISTRY: dict[str, type["OptionsStrategy"]] = {}


def register_options_strategy(name: str):
    def decorator(cls: type["OptionsStrategy"]) -> type["OptionsStrategy"]:
        if name in OPTIONS_STRATEGY_REGISTRY:
            raise ValueError(f"Stratégie options '{name}' déjà enregistrée par {OPTIONS_STRATEGY_REGISTRY[name]}.")
        OPTIONS_STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


class OptionsStrategy(ABC):
    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate_option_targets(self, signals: pd.DataFrame, current_positions: dict[str, str]) -> dict[str, dict]:
        """
        signals : signaux CONNUS à la date courante (dernière valeur publiée
            par entreprise), déjà restreints aux entreprises actuellement
            membres du S&P 500. Colonnes : symbol, gap_pct (signé : positif
            = sous-évalué, négatif = survalué), sector,
            valuation_theoretical_per_share, close, fiscal_year, published_date, source.
        current_positions : {symbol: "CALL"|"PUT"} actuellement en portefeuille.

        Retourne {symbol: {"option_type": "CALL"|"PUT", "weight": float}}
        pour le sous-ensemble de candidats actifs. Poids relatifs entre eux
        (renormalisés par l'engine sur le capital disponible, cf.
        backtest/engine.py::_rebalance pour la logique des positions gelées,
        réutilisée telle quelle par options_engine.py).
        """
        raise NotImplementedError
