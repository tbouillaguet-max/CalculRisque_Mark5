"""Stratégie actions « écart de valorisation COMBINÉE » : même logique que
`valuation_gap_dcf`, mais sur le signal de 06b au lieu du DCF seul.

CE QU'ELLE CORRIGE. Les deux moteurs du dépôt ne lisaient pas la même
valorisation, et rien ne le disait :

    09_backtest.py  (actions)  -> dcf_historique.parquet          (07, DCF seul)
    10_backtest_options.py     -> valorisation_combinee_...parquet (06b)

Or 06b n'est pas un raffinement mineur du DCF : c'est une valorisation d'une
autre nature. Elle prend en priorité les MULTIPLES SECTORIELS de l'année,
calculés en coupe sur les pairs déjà déposés à cette date, agrégés par moyenne
harmonique (Baker & Ruback 1999) et par hiérarchie de fiabilité entre
EV/EBITDA, P/E et EV/Sales (Liu, Nissim & Thomas 2002) ; le DCF ne sert que de
REPLI, quand le secteur a trop peu de pairs pour qu'une médiane ait un sens.

La différence est de fond. Un DCF fait dépendre la valeur d'un WACC et de deux
taux de croissance choisis à la main dans `SECTOR_DCF_PARAMS` -- une table
écrite aujourd'hui, en connaissant l'histoire boursière de 2010-2026, et dont
`valuation_gap_sector_neutral` existe précisément pour neutraliser l'effet. Un
multiple sectoriel point-in-time ne fait dépendre la valeur que de ce que le
marché payait ses pairs À CETTE DATE : aucune hypothèse rétrospective n'y
entre. Le côté actions se privait de ce signal sans qu'aucune décision ne
l'ait jamais tranché.

CE QU'ELLE NE CHANGE PAS. Le reste est identique à `valuation_gap_dcf` --
seuil d'entrée sur l'écart au cours, correction d'inflation, poids
proportionnels à l'écart et plafonnés, long-only, sorties laissées à l'engine.
C'est délibéré : la comparaison entre les deux ne doit mesurer QUE l'effet du
changement de signal.

```bash
python 09_backtest.py --strategy valuation_gap_combined --start-date 2015-01-01
```
"""

from __future__ import annotations

import pandas as pd

import config
from backtest.strategies.base import Strategy, capped_weights, inflation_adjusted_gap, register_strategy


@register_strategy("valuation_gap_combined")
class ValuationGapCombinedStrategy(Strategy):
    """`entry_threshold_pct` se lit comme celui de `valuation_gap_dcf` : un
    écart au COURS, en pourcentage. Les deux signaux produisent la même
    grandeur (100 x (théorique - cours) / cours), seule la façon d'établir la
    valeur théorique diffère -- ce qui rend le seuil directement comparable
    d'une stratégie à l'autre, contrairement à celui de
    `valuation_gap_sector_neutral`."""

    signal_source = "combinee"

    def __init__(
        self,
        entry_threshold_pct: float = config.BACKTEST_ENTRY_THRESHOLD_PCT,
        max_weight_pct: float = config.BACKTEST_MAX_WEIGHT_PER_POSITION_PCT,
        **kwargs,
    ):
        super().__init__(
            entry_threshold_pct=entry_threshold_pct,
            max_weight_pct=max_weight_pct,
            **kwargs,
        )
        self.entry_threshold_pct = entry_threshold_pct
        self.max_weight_pct = max_weight_pct

    def generate_target_weights(self, signals: pd.DataFrame, current_positions: set[str]) -> dict[str, float]:
        # Strictement la même mécanique que valuation_gap_dcf, pour que la
        # comparaison entre les deux ne mesure que le changement de signal.
        signals = signals.assign(gap_pct=inflation_adjusted_gap(
            signals["gap_pct"], signals["published_date"],
            config.INFLATION_HORIZON_YEARS_STOCKS,
        ))
        candidates = signals[signals["gap_pct"] >= self.entry_threshold_pct]
        if candidates.empty:
            return {}

        weights = capped_weights(candidates["gap_pct"], cap_pct=self.max_weight_pct)
        return dict(zip(candidates["symbol"], weights))
