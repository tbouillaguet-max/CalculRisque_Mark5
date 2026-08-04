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

import config

STRATEGY_REGISTRY: dict[str, type["Strategy"]] = {}


def fill_to_min_positions(
    selected: pd.DataFrame,
    pool: pd.DataFrame,
    rank_column: str,
    min_positions: int | None = None,
    floor: float | None = None,
) -> pd.DataFrame:
    """Complète `selected` avec les meilleures candidates de `pool` restées
    sous le seuil d'entrée, jusqu'à atteindre `min_positions` entreprises.

    Un seuil d'entrée strict laisse le portefeuille très concentré (voire
    vide) dès que peu d'entreprises s'écartent nettement de leur valeur
    théorique : on préfère détenir les N meilleures convictions DISPONIBLES
    plutôt que trois lignes et du cash. Le classement reste fait sur
    `rank_column`, donc les lignes ajoutées sont bien celles dont le cours est
    le plus proche de la valeur théorique parmi les recalées -- pas des
    lignes prises au hasard.

    `floor` borne la relaxation (la stratégie actions, long-only, ne descend
    pas sous un écart positif : compléter avec une entreprise jugée
    survalorisée reviendrait à acheter contre son propre signal)."""
    if not min_positions or len(selected) >= min_positions:
        return selected

    extra = pool.drop(index=selected.index, errors="ignore")
    if floor is not None:
        extra = extra[extra[rank_column] > floor]
    if extra.empty:
        return selected

    needed = min_positions - len(selected)
    extra = extra.sort_values(rank_column, ascending=False).head(needed)
    return pd.concat([selected, extra])


def capped_weights(conviction: pd.Series, cap_pct: float | None = None, max_iter: int = 20) -> pd.Series:
    """Poids proportionnels à `conviction`, aucun ne dépassant cap_pct % du
    portefeuille (config.BACKTEST_MAX_WEIGHT_PER_POSITION_PCT par défaut).

    Pondérer au prorata de l'écart de valorisation SANS plafond laisse une
    seule ligne capter la quasi-totalité du capital dès qu'un écart est
    aberrant -- et l'historique en contient (écarts à plusieurs milliers de %
    produits par une valeur théorique proche de zéro).

    L'excédent des lignes plafonnées est redistribué aux autres au prorata,
    en répétant l'opération : une simple renormalisation après écrêtage
    remonterait mécaniquement certaines lignes au-dessus du plafond."""
    cap = config.BACKTEST_MAX_WEIGHT_PER_POSITION_PCT if cap_pct is None else cap_pct
    total = conviction.sum()
    if total <= 0:
        return conviction
    weights = conviction / total
    if not cap or cap <= 0:
        return weights

    cap = cap / 100
    # Plafond inatteignable (trop peu de candidats) : équipondération.
    if cap * len(weights) <= 1:
        return pd.Series(1 / len(weights), index=weights.index)

    for _ in range(max_iter):
        over = weights > cap
        if not over.any():
            break
        excess = float((weights[over] - cap).sum())
        weights = weights.where(~over, cap)
        under = ~over
        room = float(weights[under].sum())
        if room <= 0:
            break
        weights = weights.where(over, weights + excess * weights / room)
    return weights


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
            (voir data_loader.UniverseResolver). Colonnes : symbol, gap_pct,
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
