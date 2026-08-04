"""
Stratégie options "écart de valorisation" : DIRECTIONNELLE. Achète des CALL
sur les entreprises sous-évaluées de plus de entry_threshold_pct (écart
positif), des PUT sur les survalorisées de plus de entry_threshold_pct (écart
négatif) -- même seuil que 08_recuperation_options.py par défaut. Poids
proportionnel à l'ampleur de l'écart, plus grande conviction = plus grande
part du capital alloué (le nombre de contrats réel est dérivé du delta par
l'engine, voir backtest/options_engine.py).

Le signal (multiples sectoriels en priorité, DCF en repli) vient de
06b_calcul_valorisation_combinee.py, pas du DCF seul (backtest/strategies/valuation_gap.py).
"""

from __future__ import annotations

import pandas as pd

import config
from backtest.strategies.base import capped_weights, fill_to_min_positions, inflation_adjusted_gap
from backtest.strategies.options_base import OptionsStrategy, register_options_strategy


@register_options_strategy("valuation_gap_options")
class ValuationGapOptionsStrategy(OptionsStrategy):
    def __init__(
        self,
        entry_threshold_pct: float = config.OPTIONS_ENTRY_THRESHOLD_PCT,
        max_positions: int = config.OPTIONS_MAX_POSITIONS,
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

    def generate_option_targets(self, signals: pd.DataFrame, current_positions: dict[str, str]) -> dict[str, dict]:
        # Écart corrigé de l'inflation sur l'horizon du contrat : ici le
        # re-classement est RÉEL, la stratégie étant directionnelle (l'inflation
        # aide un call et pénalise un put -- cf. base.inflation_adjusted_gap).
        gap = inflation_adjusted_gap(
            signals["gap_pct"], signals["published_date"],
            config.OPTIONS_TARGET_TENOR_DAYS / 365.0,
        )
        pool = signals.assign(gap_pct=gap, _abs_gap=gap.abs())
        candidates = pool[pool["_abs_gap"] >= self.entry_threshold_pct]
        # Trop peu d'entreprises au-dessus du seuil : on complète avec celles
        # dont le cours est le plus proche de la valeur théorique parmi les
        # recalées. Pas de plancher ici, contrairement à la stratégie actions :
        # le SENS de la position suit le signe de l'écart, donc une entreprise
        # à peine survalorisée reste jouable (en put).
        candidates = fill_to_min_positions(
            candidates, pool, "_abs_gap", min_positions=self.min_positions,
        ).copy()
        if candidates.empty:
            return {}

        candidates = candidates.sort_values("_abs_gap", ascending=False).head(self.max_positions)

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
