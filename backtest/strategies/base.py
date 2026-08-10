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
      capital disponible, stop-loss/take-profit, coûts de transaction,
      positions "gelées" (une position ouverte n'est jamais liquidée juste
      parce qu'elle sort du panier éligible -- seul un stop-loss/take-profit
      ou une disparition des données de prix ferme une position, cf.
      engine.py). Le nombre de positions simultanées n'est PAS plafonné :
      toutes les candidates retenues par la stratégie sont ouvertes.
Ça permet à une nouvelle stratégie de ne se soucier que du signal, pas de la
gestion du risque ni de l'exécution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

import config

STRATEGY_REGISTRY: dict[str, type["Strategy"]] = {}


def inflation_adjusted_gap(
    gap_pct: pd.Series,
    published_date: pd.Series,
    horizon_years: float,
    enabled: bool | None = None,
) -> pd.Series:
    """Écart de valorisation corrigé de l'inflation attendue sur l'horizon de
    convergence.

    RAISONNEMENT. L'écart g = (théorique - cours)/cours est un RATIO : à
    première vue l'inflation ne l'érode pas, et lui soustraire l'inflation
    serait faux. Mais la valeur théorique est une grandeur NOMINALE (chiffre
    d'affaires, résultats et flux futurs sont libellés en monnaie courante) :
    elle croît donc mécaniquement avec l'inflation. La convergence ne se fait
    pas vers V mais vers V x (1+pi)^T, et le mouvement NOMINAL attendu du
    titre devient :

        mouvement = (1 + g) x (1 + pi)^T - 1   ~=   g + pi x T

    L'inflation s'AJOUTE au mouvement attendu. C'est une asymétrie, pas un
    décalage uniforme, parce que les deux sens de position n'attendent pas le
    même mouvement :

        sous-évaluée (CALL) g=+20%, pi=5%, T=1 an  ->  +26%  (plus attractive)
        survalorisée (PUT)  g=-20%, pi=5%, T=1 an  ->  -16%  (moins attractive)

    Une entreprise survalorisée de 20% dans un régime à 5% d'inflation exige
    donc que le titre baisse de 20% alors que la dérive nominale le pousse à
    la hausse : sa thèse est plus fragile qu'un écart brut de -20% ne le
    laisse croire. C'est exactement le cas visé.

    NOTE pour une stratégie LONG-ONLY : n'ayant que des positions acheteuses,
    elle subit un décalage UNIFORME (+pi x T sur toutes ses candidates). Le
    classement est donc inchangé ; seul le franchissement du seuil d'entrée
    bouge. L'effet de re-classement n'existe que pour une stratégie
    directionnelle (call ET put).

    L'inflation retenue est celle CONNUE à la date de publication du signal
    (config.inflation_known_at), pas celle de l'année en cours : la moyenne
    annuelle n'est publiée qu'une fois l'année terminée."""
    if enabled is None:
        enabled = config.INFLATION_ADJUST_GAP
    if not enabled or horizon_years <= 0:
        return gap_pct

    inflation = published_date.map(config.inflation_known_at) / 100
    return ((1 + gap_pct / 100) * (1 + inflation) ** horizon_years - 1) * 100


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
