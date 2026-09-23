"""Hiérarchie de fiabilité des multiples : lequel tranche quand ils divergent ?

CE QUE CE MODULE RÈGLE. Trois multiples donnent trois valeurs théoriques pour
la même action, et elles divergent : mesuré sur les 13 240 lignes où P/E et
EV/EBITDA coexistent, l'écart médian entre les deux vaut 19,5 points de cours,
et 45,9 au troisième quartile. Il faut donc décider lequel tranche -- et ce
choix est un réglage de SIGNAL, le seul de ce dépôt qui ne soit pas un réglage
d'exécution.

POURQUOI UN MODULE PARTAGÉ. La combinaison était écrite dans 06b, qui produit
le parquet. Mais l'optimiseur a besoin de la rejouer SANS régénérer le parquet,
pour comparer les hiérarchies entre elles sur les mêmes données. Deux
implémentations divergeraient : celle-ci est la seule, et 06b comme
backtest.data_loader l'appellent (même convention que sector_history.py et
warranted_multiple.py).

CE QUE PERMET LA RECOMBINAISON À LA VOLÉE. 06b stocke les trois prix implicites
(`price_from_pe`, `price_from_ev_ebitda`, `price_from_ev_sales`) DÉJÀ filtrés
par l'applicabilité sectorielle. Toute l'information nécessaire est donc dans le
parquet, et `recombiner` reproduit exactement ce que 06b aurait écrit sous une
autre hiérarchie -- ce que `test_hierarchie_multiples` vérifie au dixième de
centime près contre le fichier réel.

CE QUE LA RECOMBINAISON A RÉVÉLÉ, et qui n'était pas cherché : le parquet en
production porte `flat`, alors que `config.MULTIPLE_COMBINATION` vaut `tiers`.
Les deux se contredisent depuis toujours, et tous les backtests `combinee` de
ce dépôt ont donc tourné sur la médiane à trois voix -- pas sur la hiérarchie
que la configuration documente. Voir le README.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Nom de colonne dans le parquet de 06b <-> nom du multiple dans les tables de
# config (SECTOR_MULTIPLES, MULTIPLE_RELIABILITY_TIERS).
COLONNES_IMPLICITES = {
    "EV/EBITDA": "price_from_ev_ebitda",
    "EV/Sales": "price_from_ev_sales",
    "P/E": "price_from_pe",
}

# LES RANGS, ET CE QUE CHACUN PARIE. 1 = le plus fiable ; un rang ne sert qu'aux
# lignes qu'aucun rang meilleur n'a servies, et à l'intérieur d'un rang c'est la
# médiane qui départage (pour deux valeurs, leur moyenne).
#
# Liu, Nissim & Thomas (Journal of Accounting Research, 2002) mesurent la
# précision relative des multiples : les multiples de RÉSULTATS dominent
# nettement, ceux de CHIFFRE D'AFFAIRES sont les moins précis de loin. Les trois
# hiérarchies ci-dessous sont d'accord là-dessus -- EV/Sales est toujours
# dernier -- et se distinguent uniquement sur l'arbitrage entre P/E et
# EV/EBITDA, que la littérature ne tranche pas nettement :
#
#   tiers        les deux multiples de résultats font jeu égal, on les moyenne.
#   pe_first     le P/E tranche ; l'EV/EBITDA n'est qu'un repli. Parie que le
#                résultat net, mesure la plus regardée, est aussi la mieux
#                arbitrée par le marché.
#   ebitda_first l'inverse. Parie que l'EBITDA, neutre à la structure de
#                capital et aux choix d'amortissement, compare mieux des pairs
#                aux niveaux d'endettement différents.
#
# `flat` n'est pas une hiérarchie mais son absence : la médiane des trois, donc
# EV/Sales -- le moins fiable -- qui départage dès qu'il tombe entre les deux
# autres. C'est le comportement historique, et c'est CELUI QUI TOURNE.
HIERARCHIES: dict[str, Optional[dict[str, int]]] = {
    "flat": None,
    "tiers": {"P/E": 1, "EV/EBITDA": 1, "EV/Sales": 2},
    "pe_first": {"P/E": 1, "EV/EBITDA": 2, "EV/Sales": 3},
    "ebitda_first": {"EV/EBITDA": 1, "P/E": 2, "EV/Sales": 3},
}

DEFAUT = "tiers"


def rangs(hierarchie: str | dict | None) -> Optional[dict[str, int]]:
    """Table de rangs d'une hiérarchie nommée, ou None pour la médiane à plat.

    Accepte aussi une table déjà construite, pour qu'un appelant puisse en
    essayer une qui n'est pas au catalogue sans modifier ce module."""
    if hierarchie is None or isinstance(hierarchie, dict):
        return hierarchie
    if hierarchie not in HIERARCHIES:
        raise ValueError(
            f"Hiérarchie de multiples inconnue : {hierarchie!r}. "
            f"Attendu l'une de {sorted(HIERARCHIES)}, ou une table de rangs."
        )
    return HIERARCHIES[hierarchie]


def combiner(implied: pd.DataFrame, hierarchie: str | dict | None = DEFAUT) -> pd.Series:
    """Valeur théorique par action à partir des prix implicites des multiples.

    `implied` porte un multiple par colonne, nommée comme dans les tables de
    config ("P/E", "EV/EBITDA", "EV/Sales").

    Chaque ligne n'utilise que le MEILLEUR rang disponible : les multiples de
    résultats quand au moins un est exploitable, EV/Sales seulement à défaut.
    Ce n'est pas une pondération mais une hiérarchie -- pondérer laisserait
    EV/Sales départager dès qu'il tombe entre les deux autres, ce qui est
    exactement le défaut qu'on corrige.

    Le calcul est vectorisé PAR RANG et non ligne à ligne : le nombre de rangs
    est fixe (deux ou trois) alors que le nombre de lignes se compte en dizaines
    de milliers.

    LA COUVERTURE NE DÉPEND PAS DE LA HIÉRARCHIE : une ligne qui a au moins un
    multiple exploitable en a un à un rang quelconque, donc reçoit une valeur
    quelle que soit la table. C'est ce qui fait de la hiérarchie un axe PROPRE
    -- il déplace la valeur, jamais le nombre de lignes valorisées."""
    table = rangs(hierarchie)
    if table is None:
        return implied.median(axis=1, skipna=True)

    pire = max(table.values()) + 1
    resultat = pd.Series(np.nan, index=implied.index, dtype=float)
    for rang in sorted({table.get(col, pire) for col in implied.columns}):
        colonnes = [c for c in implied.columns if table.get(c, pire) == rang]
        if not colonnes:
            continue
        # Un rang ne sert qu'aux lignes qu'aucun rang MEILLEUR n'a servies.
        resultat = resultat.where(
            resultat.notna(), implied[colonnes].median(axis=1, skipna=True))
    return resultat


def recombiner(df: pd.DataFrame, hierarchie: str | dict | None) -> pd.DataFrame:
    """Rejoue la combinaison d'un parquet 06b déjà écrit, sous une autre
    hiérarchie, sans repasser par le pipeline.

    Recalcule les quatre colonnes qui en dépendent -- valeur des multiples,
    valeur théorique, `source` et `gap_pct` -- en reproduisant exactement
    l'ordre de 06b : la valeur des multiples d'abord, le repli DCF ensuite,
    l'écart en dernier.

    `source` est recalculé bien qu'il ne doive pas bouger (la couverture est
    invariante, cf. `combiner`) : le recalculer plutôt que le supposer stable
    est ce qui permet de le VÉRIFIER, et la stratégie options filtre dessus.

    Rend le DataFrame inchangé si `hierarchie` est None."""
    if hierarchie is None:
        return df
    manquantes = [c for c in COLONNES_IMPLICITES.values() if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"Recombinaison impossible : colonnes {manquantes} absentes. Le parquet vient "
            "d'un 06b antérieur à leur écriture -- relance 06b_calcul_valorisation_combinee.py."
        )

    df = df.copy()
    implied = df[[COLONNES_IMPLICITES[m] for m in COLONNES_IMPLICITES]].rename(
        columns={v: k for k, v in COLONNES_IMPLICITES.items()})
    multiples = combiner(implied, hierarchie)

    a_des_multiples = multiples.notna()
    df["valuation_multiples_per_share"] = multiples
    df["n_multiples_used"] = implied.notna().sum(axis=1)
    df["valuation_theoretical_per_share"] = multiples.where(
        a_des_multiples, df["valuation_dcf_per_share"])
    df["source"] = np.select(
        [a_des_multiples, df["valuation_dcf_per_share"].notna()],
        ["multiples", "dcf_fallback"],
        default=None,
    )
    df["gap_pct"] = (
        (df["valuation_theoretical_per_share"] - df["close"]) / df["close"] * 100)
    return df
