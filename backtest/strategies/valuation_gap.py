"""
Stratégie "écart de valorisation DCF" : achète (long-only) les entreprises
dont la valeur intrinsèque (DCF, 07_calcul_dcf.py) dépasse le cours de bourse
d'au moins entry_threshold_pct, pondérées par l'ampleur de l'écart -- plus
une entreprise est jugée sous-évaluée, plus sa part du capital est grande.

Ni sortie sur convergence de l'écart ni stop-loss/take-profit ici : ces
derniers sont gérés uniformément par l'engine pour toutes les stratégies
(voir backtest/strategies/base.py).
"""

from __future__ import annotations

import pandas as pd

import config
from backtest.strategies.base import Strategy, capped_weights, fill_to_min_positions, register_strategy


@register_strategy("valuation_gap_dcf")
class ValuationGapDCFStrategy(Strategy):
    def __init__(
        self,
        entry_threshold_pct: float = config.BACKTEST_ENTRY_THRESHOLD_PCT,
        max_positions: int = config.BACKTEST_MAX_POSITIONS,
        min_positions: int = config.BACKTEST_MIN_POSITIONS,
        **kwargs,
    ):
        super().__init__(
            entry_threshold_pct=entry_threshold_pct, max_positions=max_positions,
            min_positions=min_positions, **kwargs,
        )
        self.entry_threshold_pct = entry_threshold_pct
        self.max_positions = max_positions
        self.min_positions = min_positions

    def generate_target_weights(self, signals: pd.DataFrame, current_positions: set[str]) -> dict[str, float]:
        candidates = signals[signals["gap_pct"] >= self.entry_threshold_pct]
        # Trop peu d'entreprises au-dessus du seuil : on complète avec les
        # plus proches de leur valeur théorique, sans jamais descendre sous un
        # écart POSITIF (stratégie long-only : acheter une entreprise que le
        # modèle juge survalorisée serait acheter contre son propre signal).
        candidates = fill_to_min_positions(
            candidates, signals, "gap_pct", min_positions=self.min_positions, floor=0.0,
        )
        if candidates.empty:
            return {}

        # Plus de candidats que de places : on garde les plus sous-évalués.
        # L'engine réapplique de toute façon un plafond global (positions
        # gelées comprises), mais capper ici évite de diluer inutilement le
        # poids sur des candidats qui seraient de toute façon écartés.
        candidates = candidates.sort_values("gap_pct", ascending=False).head(self.max_positions)

        # Poids plafonnés (config.BACKTEST_MAX_WEIGHT_PER_POSITION_PCT) : le
        # classement reste fait sur l'écart brut, seul le DIMENSIONNEMENT est
        # borné -- un écart de plusieurs milliers de % est une conviction
        # légitime, pas une raison de mettre 90% du capital sur une ligne.
        weights = capped_weights(candidates["gap_pct"])
        return dict(zip(candidates["symbol"], weights))
