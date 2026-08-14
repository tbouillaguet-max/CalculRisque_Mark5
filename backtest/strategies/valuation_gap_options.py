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

Contrats achetés à 2 ans (config.OPTIONS_TARGET_TENOR_DAYS) et réexaminés à 9
mois de l'échéance (config.OPTIONS_ROLL_WHEN_DAYS_LEFT) : roulés sur une
nouvelle échéance pleine si l'écart passe encore le seuil d'entrée, clôturés
sinon. Stop-loss/take-profit mesurés sur le COURS DU SOUS-JACENT
(config.OPTIONS_STOP_BASIS) : sur une échéance aussi longue, les mêmes seuils
appliqués à la prime se déclencheraient sur la seule érosion de la valeur
temps.

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
        exit_threshold_ratio: float = config.OPTIONS_EXIT_THRESHOLD_RATIO,
        **kwargs,
    ):
        super().__init__(
            entry_threshold_pct=entry_threshold_pct,
            exit_threshold_ratio=exit_threshold_ratio, **kwargs,
        )
        self.entry_threshold_pct = entry_threshold_pct
        # Hystérésis (cf. config.OPTIONS_EXIT_THRESHOLD_RATIO). Cette
        # stratégie gèle ses positions par défaut, donc le seuil de sortie ne
        # sert qu'au réexamen de ROULEMENT -- où il a exactement le même sens :
        # renouveler un contrat dont la thèse a partiellement porté plutôt que
        # de le solder au premier retour sous le seuil d'entrée.
        self.exit_threshold_pct = entry_threshold_pct * float(exit_threshold_ratio)

    def _scored(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Écart corrigé de l'inflation, sans filtrage par seuil -- les seuils
        d'entrée et de sortie s'appliquent au même calcul (cf.
        OptionsStrategy.evaluate)."""
        # Ici le re-classement par l'inflation est RÉEL, la stratégie étant
        # directionnelle (l'inflation aide un call et pénalise un put -- cf.
        # base.inflation_adjusted_gap).
        gap = inflation_adjusted_gap(
            signals["gap_pct"], signals["published_date"],
            config.OPTIONS_TARGET_TENOR_DAYS / 365.0,
        )
        return signals.assign(gap_pct=gap, _abs_gap=gap.abs())

    @staticmethod
    def _directions_of(df: pd.DataFrame) -> dict[str, str]:
        return {
            symbol: ("CALL" if gap > 0 else "PUT")
            for symbol, gap in zip(df["symbol"].to_numpy(), df["gap_pct"].to_numpy())
        }

    def _targets_from(self, candidates: pd.DataFrame) -> dict[str, dict]:
        if candidates.empty:
            return {}
        # Poids plafonnés (cf. base.capped_weights) : sans plafond, un écart
        # aberrant capte à lui seul l'essentiel du capital. Le classement,
        # lui, reste fait sur l'écart brut.
        weights = capped_weights(candidates["_abs_gap"])
        # zip sur les colonnes numpy plutôt qu'iterrows, qui matérialise une
        # Series par ligne.
        return {
            symbol: {"option_type": "CALL" if gap > 0 else "PUT", "weight": weight}
            for symbol, gap, weight in zip(
                candidates["symbol"].to_numpy(),
                candidates["gap_pct"].to_numpy(),
                weights.to_numpy(),
            )
        }

    def generate_option_targets(self, signals: pd.DataFrame, current_positions: dict[str, str]) -> dict[str, dict]:
        scored = self._scored(signals)
        return self._targets_from(scored[scored["_abs_gap"] >= self.entry_threshold_pct])

    def eligible_directions(self, signals: pd.DataFrame) -> dict[str, str]:
        scored = self._scored(signals)
        return self._directions_of(scored[scored["_abs_gap"] >= self.exit_threshold_pct])

    def evaluate(
        self, signals: pd.DataFrame, current_positions: dict[str, str],
    ) -> tuple[dict[str, dict], dict[str, str]]:
        scored = self._scored(signals)
        if scored.empty:
            return {}, {}
        abs_gap = scored["_abs_gap"]
        return (
            self._targets_from(scored[abs_gap >= self.entry_threshold_pct]),
            self._directions_of(scored[abs_gap >= self.exit_threshold_pct]),
        )
