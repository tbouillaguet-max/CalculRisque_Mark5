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
    # True quand la stratégie vise une VALEUR THÉORIQUE (elle renseigne
    # "strike_reference_price" dans ses cibles) et non un simple mouvement du
    # sous-jacent. Le moteur prend alors ses gains à une fraction du chemin
    # parcouru vers cette valeur (OPTIONS_TAKE_PROFIT_CONVERGENCE_FRACTION)
    # plutôt qu'à un seuil fixe, ce qui rend `take_profit_pct` INERTE pour
    # elle. Déclaré ici pour que les outils d'optimisation sachent quel
    # paramètre a réellement un effet, au lieu de balayer le mauvais.
    targets_convergence = False

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
        que la stratégie tronque sa liste de candidats, ou qu'elle applique un
        seuil de SORTIE distinct du seuil d'entrée (hystérésis, cf.
        config.OPTIONS_EXIT_THRESHOLD_RATIO)."""
        return {s: t["option_type"] for s, t in self.generate_option_targets(signals, {}).items()}

    def evaluate(
        self, signals: pd.DataFrame, current_positions: dict[str, str],
    ) -> tuple[dict[str, dict], dict[str, str]]:
        """(cibles, sens encore justifiés) en UN SEUL passage.

        Le moteur a besoin des deux le même jour et sur la même photographie
        de marché (cf. options_engine._current_targets). Les demander par deux
        appels séparés faisait recalculer à l'identique tout le pipeline de
        sélection -- mesuré à 20% du temps de run total sur un backtest
        réaliste, pour un résultat rigoureusement identique.

        L'implémentation par défaut enchaîne simplement les deux méthodes
        publiques : une stratégie tierce n'a rien à changer. Une stratégie qui
        partage un calcul coûteux entre les deux (c'est le cas des deux
        stratégies du dépôt) surcharge cette méthode et ne le fait qu'une
        fois."""
        targets = self.generate_option_targets(signals, current_positions)
        return targets, self.eligible_directions(signals)
