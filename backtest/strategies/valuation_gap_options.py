"""
Stratégie options "écart de valorisation" : DIRECTIONNELLE. Achète des CALL
sur les entreprises sous-évaluées de plus de entry_threshold_pct (écart
positif), des PUT sur les survalorisées de plus de entry_threshold_pct (écart
négatif) -- même seuil que 08_recuperation_options.py par défaut. Poids
proportionnel à l'ampleur de l'écart, plus grande conviction = plus grande
part du capital alloué (le nombre de contrats réel est dérivé du delta par
l'engine, voir backtest/options_engine.py). Aucune limite sur le nombre de
positions simultanées : toutes les candidates au-dessus du seuil sont
retenues.

Le signal (multiples sectoriels en priorité, DCF en repli) vient de
06b_calcul_valorisation_combinee.py, pas du DCF seul (backtest/strategies/valuation_gap.py).
"""

from __future__ import annotations

import pandas as pd

import config
from backtest.strategies.base import capped_weights, inflation_adjusted_gap
from backtest.strategies.options_base import OptionsStrategy, register_options_strategy


@register_options_strategy("valuation_gap_options")
class ValuationGapOptionsStrategy(OptionsStrategy):
    def __init__(
        self,
        entry_threshold_pct: float = config.OPTIONS_ENTRY_THRESHOLD_PCT,
        **kwargs,
    ):
        super().__init__(entry_threshold_pct=entry_threshold_pct, **kwargs)
        self.entry_threshold_pct = entry_threshold_pct

    def generate_option_targets(self, signals: pd.DataFrame, current_positions: dict[str, str]) -> dict[str, dict]:
        # Écart corrigé de l'inflation sur l'horizon du contrat : ici le
        # re-classement est RÉEL, la stratégie étant directionnelle (l'inflation
        # aide un call et pénalise un put -- cf. base.inflation_adjusted_gap).
        gap = inflation_adjusted_gap(
            signals["gap_pct"], signals["published_date"],
            config.OPTIONS_TARGET_TENOR_DAYS / 365.0,
        )
        pool = signals.assign(gap_pct=gap, _abs_gap=gap.abs())
        candidates = pool[pool["_abs_gap"] >= self.entry_threshold_pct].copy()
        if candidates.empty:
            return {}

        # Poids plafonnés (cf. base.capped_weights) : sans plafond, un écart
        # aberrant capte à lui seul l'essentiel du capital. Le classement,
        # lui, reste fait sur l'écart brut.
        candidates["_weight"] = capped_weights(candidates["_abs_gap"])
        return {
            row["symbol"]: {
                "option_type": "CALL" if row["gap_pct"] > 0 else "PUT",
                "weight": row["_weight"],
            }
            for _, row in candidates.iterrows()
        }
