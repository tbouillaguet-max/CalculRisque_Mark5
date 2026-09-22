"""
Stratégie "écart de valorisation DCF" : achète (long-only) les entreprises
dont la valeur intrinsèque (DCF, 07_calcul_dcf.py) dépasse le cours de bourse
d'au moins entry_threshold_pct, pondérées par l'ampleur de l'écart -- plus
une entreprise est jugée sous-évaluée, plus sa part du capital est grande.
Aucune limite sur le nombre de positions simultanées : toutes les candidates
au-dessus du seuil sont retenues.

Ni sortie sur convergence de l'écart ni stop-loss/take-profit ici : ces
derniers sont gérés uniformément par l'engine pour toutes les stratégies
(voir backtest/strategies/base.py).
"""

from __future__ import annotations

import pandas as pd

import config
from backtest.strategies.base import Strategy, construire_poids, inflation_adjusted_gap, register_strategy


@register_strategy("valuation_gap_dcf")
class ValuationGapDCFStrategy(Strategy):
    def __init__(
        self,
        entry_threshold_pct: float = config.BACKTEST_ENTRY_THRESHOLD_PCT,
        max_weight_pct: float = config.BACKTEST_MAX_WEIGHT_PER_POSITION_PCT,
        vol_weight_exponent: float = config.BACKTEST_VOL_WEIGHT_EXPONENT,
        max_positions: int = 0,
        rank_weighting: bool = False,
        # 0 = PAS de plafond sectoriel, contrairement à
        # valuation_gap_sector_neutral qui en a un à 30%. C'est le
        # comportement d'origine de cette stratégie ; le réglage existe pour
        # être balayable, pas pour changer le défaut en douce.
        max_weight_per_sector_pct: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            entry_threshold_pct=entry_threshold_pct,
            max_weight_pct=max_weight_pct,
            vol_weight_exponent=vol_weight_exponent,
            max_positions=max_positions,
            rank_weighting=rank_weighting,
            max_weight_per_sector_pct=max_weight_per_sector_pct,
            **kwargs,
        )
        self.entry_threshold_pct = entry_threshold_pct
        self.vol_weight_exponent = vol_weight_exponent
        self.max_positions = max_positions
        self.rank_weighting = bool(rank_weighting)
        self.max_weight_per_sector_pct = max_weight_per_sector_pct
        # Plafond de concentration, jusqu'ici lu directement dans config par
        # capped_weights. L'exposer en paramètre de stratégie le rend
        # balayable par un grid-search sans toucher au module de config --
        # c'est un arbitrage de diversification, donc un réglage de Sharpe au
        # même titre que le seuil d'entrée. Le défaut ne change rien.
        self.max_weight_pct = max_weight_pct

    def generate_target_weights(self, signals: pd.DataFrame, current_positions: set[str]) -> dict[str, float]:
        # Écart corrigé de l'inflation attendue sur l'horizon de convergence
        # (cf. base.inflation_adjusted_gap). Stratégie long-only : le décalage
        # est uniforme, il ne change donc pas le CLASSEMENT, seulement le
        # franchissement du seuil d'entrée.
        signals = signals.assign(gap_pct=inflation_adjusted_gap(
            signals["gap_pct"], signals["published_date"],
            config.INFLATION_HORIZON_YEARS_STOCKS,
        ))
        candidates = signals[signals["gap_pct"] >= self.entry_threshold_pct]
        if candidates.empty:
            return {}

        # Poids plafonnés (config.BACKTEST_MAX_WEIGHT_PER_POSITION_PCT) : le
        # classement reste fait sur l'écart brut, seul le DIMENSIONNEMENT est
        # borné -- un écart de plusieurs milliers de % est une conviction
        # légitime, pas une raison de mettre 90% du capital sur une ligne.
        return construire_poids(
            candidates, candidates["gap_pct"],
            max_weight_pct=self.max_weight_pct,
            max_positions=self.max_positions,
            max_weight_per_sector_pct=self.max_weight_per_sector_pct,
            vol_weight_exponent=self.vol_weight_exponent,
            rank_weighting=self.rank_weighting,
        )
