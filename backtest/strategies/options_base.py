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
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

OPTIONS_STRATEGY_REGISTRY: dict[str, type["OptionsStrategy"]] = {}


@dataclass(frozen=True)
class MarketContext:
    """Vue en LECTURE SEULE de l'état de marché du jour, fournie par le moteur
    aux stratégies qui en ont besoin.

    POURQUOI CE DÉTOUR. La trame `signals` ne porte que le signal fondamental
    (valeur théorique, cours, secteur). Une stratégie qui pose son strike ad
    hoc (ATM, ou mi-chemin cours/valeur) n'a besoin de rien d'autre. Une
    stratégie qui OPTIMISE son strike a besoin, elle, de la volatilité et des
    strikes réellement cotés -- deux grandeurs que seul le moteur détient.

    Les recalculer dans la stratégie serait pire que ce détour : la volatilité
    servant à choisir le contrat doit être EXACTEMENT celle qui servira ensuite
    à le repricer (cf. options_engine._repricing_vol), sans quoi la position
    afficherait un saut de P&L dès le lendemain de son ouverture, sans qu'aucun
    prix n'ait bougé.

    Frozen, et exposé en callables plutôt qu'en données : la stratégie
    interroge le moteur pour les seuls symboles qu'elle retient (quelques
    dizaines), pas pour tout l'univers suivi (plusieurs centaines)."""

    today: pd.Timestamp
    # (symbole) -> volatilité réalisée annualisée, ou None si l'historique
    # disponible à cette date est trop court.
    realized_vol: Callable[[str], Optional[float]]
    # (symbole) -> (taux sans risque, rendement du dividende sectoriel).
    pricing_context: Callable[[str], tuple]
    # (symbole, type d'option, échéance visée en jours) -> volatilité IMPLICITE
    # réellement cotée, ou None si aucun snapshot exploitable.
    implied_vol: Callable[[str, str, float], Optional[float]]
    # (symbole, type d'option, échéance visée en jours) -> strikes cotés,
    # triés. Liste vide si aucun snapshot exploitable.
    quoted_strikes: Callable[[str, str, float], list]


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

    # True quand la stratégie a besoin de l'état de marché du jour (volatilité,
    # strikes cotés) pour construire ses cibles. Le moteur ne construit et ne
    # transmet le MarketContext que dans ce cas : les stratégies qui posent
    # leur strike ad hoc n'en ont aucun usage, et le leur fournir coûterait
    # une construction de contexte par jour de bourse pour rien.
    needs_market_context = False

    def __init__(self, **params):
        self.params = params
        self.market_context: Optional[MarketContext] = None

    def bind_market_context(self, context: MarketContext) -> None:
        """Appelé par le moteur avant chaque évaluation, si
        `needs_market_context`. Point d'ancrage volontairement passif : la
        stratégie stocke le contexte et le lit dans son propre calcul, plutôt
        que le moteur ne lui pousse des paramètres qu'il faudrait ajouter à la
        signature de generate_option_targets à chaque nouveau besoin."""
        self.market_context = context

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
