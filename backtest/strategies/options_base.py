"""
Interface commune aux stratégies OPTIONS, + registre par nom séparé de celui
des stratégies actions (backtest/strategies/base.py) : les deux types
d'instruments ont des mécaniques trop différentes (expiration, greeks,
prime) pour partager une seule interface -- voir backtest/options_engine.py
pour ce que le moteur gère uniformément (dimensionnement par delta, stop-loss/
take-profit, roulement, positions gelées, coûts).
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

        Deux clés OPTIONNELLES permettent de choisir un autre contrat que
        l'ATM à 2 ans retenu par défaut (voir options_engine._select_contract) :
            "strike_reference_price" : le strike visé est à mi-chemin entre ce
                prix et le spot d'exécution (une stratégie qui vise la
                convergence vers sa valeur théorique y met cette valeur).
            "tenor_days" : échéance visée, en jours.
        Les omettre reproduit exactement le comportement historique.
        """
        raise NotImplementedError

    def eligible_directions(self, signals: pd.DataFrame) -> dict[str, str]:
        """{symbol: "CALL"|"PUT"} pour TOUS les symboles dont le signal
        justifie encore une position, AVANT tout plafonnement du nombre de
        positions simultanées.

        Utilisé uniquement par les moteurs configurés en vente sur perte de
        signal ou en roulement (voir la docstring de
        backtest/options_engine.py) : une position évincée par le seul
        plafond ne doit pas être vendue comme si son signal avait disparu, ni
        empêchée de rouler. Par défaut, déduit des cibles -- à surcharger dès
        que la stratégie tronque sa liste de candidats."""
        return {s: t["option_type"] for s, t in self.generate_option_targets(signals, {}).items()}
