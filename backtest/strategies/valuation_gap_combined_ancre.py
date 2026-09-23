"""Stratégie actions « combinée ANCRÉE » : même signal que
`valuation_gap_combined`, mais une candidate neuve ne repèse plus tout le
portefeuille.

LE PROBLÈME QU'ELLE ATTAQUE. Mesuré sur 2015-2026, la stratégie combinée passe
39 044 ventes pour 2 568 thèses seulement : **93,4 % des ventes sont des
allègements de rebalancement**, pas des décisions. La zone de non-négociation
était censée les filtrer. Elle ne le fait pas, et le balayage le montre sans
ambiguïté :

    bande      Sharpe   ventes   dont rebalancement
        0       0,970   63 886           61 318
       15       0,977   39 044           36 476
       30       0,978   39 022           36 454
       50       0,978   39 022           36 454
   aucune       0,978   39 022           36 454

Élargir la bande de 15 à l'infini change **22 ventes sur 39 044**. Ce n'est pas
un réglage, c'est un plancher.

POURQUOI. `engine._drift_is_material` contient un coupe-circuit : toute
candidate encore absente du portefeuille dont la cible dépasse le trade minimum
renvoie `True` immédiatement, donc force le repesage intégral quelle que soit
la bande. Comme des dépôts SEC font apparaître des candidates neuves 2 624
séances sur 2 936, la bande n'est consultée que les jours sans nouveauté. Elle
supprime bien ~25 000 allègements -- ceux-là ne servaient à rien, les retirer
fait passer la rotation de 963 % à 892 % sans coûter de Sharpe -- mais elle ne
peut rien contre les 36 454 restants.

CE QUE CETTE STRATÉGIE CHANGE, ET RIEN D'AUTRE. Elle déclare
`entree_neuve_force_repesage = False`. Une candidate neuve compte alors dans la
dérive comme n'importe quel écart à la cible -- une position absente EST une
déviation -- mais elle ne décide plus à elle seule. Une grosse candidate
franchit encore le seuil toute seule ; une petite attend que la dérive
s'accumule.

LE DÉFAUT LATENT QU'IL FAUT ALORS CORRIGER AUTREMENT. Le coupe-circuit ne
servait pas qu'à repeser : il garantissait qu'une candidate SEULE finisse par
être achetée. Sans position en portefeuille, la dérive ne s'accumule pas --
elle vaut la cible de la candidate, indéfiniment -- et une candidate sous le
seuil ne serait donc JAMAIS achetée. La zone cesserait d'être un filtre de coût
pour devenir un filtre de signal, ce qu'elle n'a jamais eu vocation à être.
`_drift_is_material` force donc le repesage quand le portefeuille est VIDE,
uniquement dans ce mode. C'est un amorçage, pas un coupe-circuit : il ne peut
se déclencher qu'au démarrage ou après une liquidation totale.

CE QU'ELLE NE CHANGE PAS. Le signal, le seuil d'entrée, la pondération, les
plafonds, les sorties : tout est hérité de `valuation_gap_combined`. La
comparaison entre les deux ne mesure QUE l'effet du coupe-circuit.

CE QUI RESTE À FAIRE, ET POURQUOI CE N'EST PAS ICI. La cause première est la
renormalisation de `base.capped_weights` (`poids = conviction / somme`), qui
fait qu'un seul dépôt déplace réellement les 82 cibles. Tant qu'elle est là,
lever le coupe-circuit ne supprime que les repesages dont la dérive agrégée
reste sous le seuil -- utile, mais ce n'est pas la racine. Une pondération qui
ne se renormalise pas est l'étape suivante, et elle se mesurera séparément.

```bash
python 09_backtest.py --strategy valuation_gap_combined_ancre \\
    --start-date 2015-01-01 --vol-target-pct 0
```
"""

from __future__ import annotations

from backtest.strategies.base import register_strategy
from backtest.strategies.valuation_gap_combined import ValuationGapCombinedStrategy


@register_strategy("valuation_gap_combined_ancre")
class ValuationGapCombinedAncreeStrategy(ValuationGapCombinedStrategy):
    """Elle HÉRITE de `ValuationGapCombinedStrategy` et ne redéfinit qu'un
    attribut. C'est le point : la comparaison entre les deux ne doit mesurer
    QUE la levée du coupe-circuit, et dupliquer la logique aurait laissé les
    deux implémentations diverger au premier réglage ajouté à l'une.

    Le nom : les positions déjà ouvertes restent ANCRÉES à leur taille quand
    une candidate apparaît, au lieu d'être toutes redimensionnées pour lui
    faire de la place."""

    entree_neuve_force_repesage = False
