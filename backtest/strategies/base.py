"""
Interface commune à toutes les stratégies de backtest, + registre par nom
pour pouvoir en ajouter une nouvelle sans toucher à 09_backtest.py ni à
engine.py : il suffit de créer un fichier dans backtest/strategies/, y
définir une classe héritant de Strategy et décorée par @register_strategy,
puis de l'importer dans backtest/strategies/__init__.py.

Séparation des responsabilités (important pour comprendre pourquoi une
stratégie n'a PAS à gérer stop-loss/take-profit elle-même) :
    - La STRATÉGIE dit uniquement "parmi les entreprises actuellement
      éligibles, lesquelles acheter et avec quel poids relatif ?"
      (generate_target_weights).
    - Le MOTEUR (engine.py) gère tout le reste : dimensionnement réel du
      capital disponible, plafond de positions simultanées, stop-loss/
      take-profit, coûts de transaction, positions "gelées" (une position
      ouverte n'est jamais liquidée juste parce qu'elle sort du panier
      éligible -- seul un stop-loss/take-profit ou une disparition des
      données de prix ferme une position, cf. engine.py).
Ça permet à une nouvelle stratégie de ne se soucier que du signal, pas de la
gestion du risque ni de l'exécution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

STRATEGY_REGISTRY: dict[str, type["Strategy"]] = {}


def register_strategy(name: str):
    def decorator(cls: type["Strategy"]) -> type["Strategy"]:
        if name in STRATEGY_REGISTRY:
            raise ValueError(f"Stratégie '{name}' déjà enregistrée par {STRATEGY_REGISTRY[name]}.")
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


class Strategy(ABC):
    """params : hyperparamètres de la stratégie, exposés tels quels dans
    run_config.json (09_backtest.py) pour la reproductibilité d'un run."""

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate_target_weights(self, signals: pd.DataFrame, current_positions: set[str]) -> dict[str, float]:
        """
        signals : signaux CONNUS à la date courante (dernière valeur publiée
            par entreprise, pas seulement ceux publiés aujourd'hui),
            déjà restreints aux entreprises actuellement membres du S&P 500
            (voir data_loader.universe_asof). Colonnes : symbol, gap_pct,
            sector, valuation_dcf_per_share, close_at_filing, fiscal_year,
            published_date.
        current_positions : symboles actuellement en portefeuille (permet à
            une stratégie de favoriser la continuité si besoin -- non utilisé
            par ValuationGapDCFStrategy).

        Retourne {symbol: poids} pour le sous-ensemble de "candidats actifs"
        que la stratégie souhaite acheter/renforcer. Les poids sont relatifs
        entre eux (pas nécessairement normalisés à 1 : l'engine les
        renormalise si leur somme dépasse 1, et les alloue sur le capital
        RESTANT après les positions gelées -- voir docstring du module).
        Un symbole actuellement en position mais absent du résultat n'est
        PAS vendu : il devient une position "gelée" (voir engine.py).
        """
        raise NotImplementedError
