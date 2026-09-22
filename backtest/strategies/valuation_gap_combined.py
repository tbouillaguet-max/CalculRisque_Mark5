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

from backtest.strategies.base import register_strategy
from backtest.strategies.valuation_gap import ValuationGapDCFStrategy


@register_strategy("valuation_gap_combined")
class ValuationGapCombinedStrategy(ValuationGapDCFStrategy):
    """Elle HÉRITE de `ValuationGapDCFStrategy` et ne redéfinit que la source
    du signal. C'est le point : la comparaison entre les deux ne doit mesurer
    QUE le changement de valorisation, et dupliquer la logique aurait laissé
    les deux implémentations diverger au premier réglage ajouté à l'une.

    `entry_threshold_pct` se lit donc exactement comme celui de
    `valuation_gap_dcf` : un écart au COURS, en pourcentage. Les deux signaux
    produisent la même grandeur -- 100 x (théorique - cours) / cours -- seule
    la façon d'établir la valeur théorique diffère, ce qui rend le seuil
    directement comparable d'une stratégie à l'autre (contrairement à celui de
    `valuation_gap_sector_neutral`, qui porte sur l'écart à la médiane du
    secteur)."""

    signal_source = "combinee"
