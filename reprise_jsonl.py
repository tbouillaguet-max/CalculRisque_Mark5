"""
Lecture des fichiers de reprise JSONL (checkpoints, caches) -- partagée par
04c, 07b et 08.

DEUX DÉFAUTS QUE CES FICHIERS PORTAIENT, TOUS DEUX NÉS D'UNE INTERRUPTION
-----------------------------------------------------------------------
1. LES DOUBLONS APRÈS --resume. Chaque ligne est écrite dans le checkpoint dès
   qu'elle est produite, mais la liste des éléments déjà traités n'est
   sauvegardée que toutes les CHECKPOINT_EVERY unités (10). Un run interrompu
   puis repris refait donc jusqu'à neuf unités -- et les réécrit. Les trois
   scripts recopiaient ensuite le checkpoint tel quel dans leur fichier de
   sortie : un 8-K, une période ou un contrat en double, sans que rien ne le
   signale. Mesuré sur les fichiers du dépôt au 2026-09-25 : aucun doublon
   encore -- ils n'avaient jamais été repris.

2. LA DERNIÈRE LIGNE TRONQUÉE. Un run tué en pleine écriture laisse une ligne
   JSON incomplète. 07b et 08 la relisaient avec json.loads, qui lève : le
   --resume, fait pour sauver le travail, plantait à la toute fin, au moment
   d'écrire la sortie. 04c le tolérait déjà pour sa mémoire des 8-K.

LA RÈGLE : LA DERNIÈRE ÉCRITURE GAGNE -- c'est la plus récente, et c'est déjà
celle de la mémoire de 04c. Chaque ligne garde la place de sa PREMIÈRE
apparition : un fichier sans doublon ressort à l'identique.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Hashable, Iterable

import ecriture_atomique

logger = logging.getLogger("reprise_jsonl")


def lire_lignes(chemin: Path) -> tuple[list[dict], int]:
    """(lignes valides dans l'ordre du fichier, nombre de lignes illisibles)."""
    if not chemin.exists():
        return [], 0
    lignes: list[dict] = []
    illisibles = 0
    with chemin.open(encoding="utf-8") as fichier:
        for brut in fichier:
            brut = brut.strip()
            if not brut:
                continue
            try:
                ligne = json.loads(brut)
            except json.JSONDecodeError:
                illisibles += 1
                continue
            if isinstance(ligne, dict):
                lignes.append(ligne)
            else:
                illisibles += 1
    return lignes, illisibles


def dedoublonner(lignes: Iterable[dict], cle: Callable[[dict], Hashable]) -> tuple[list[dict], int]:
    """Une ligne par clé -- la dernière écrite -- et le nombre de doublons écartés."""
    lignes = list(lignes)
    par_cle: dict = {}
    for ligne in lignes:
        par_cle[cle(ligne)] = ligne   # un dict garde la place de la première insertion
    return list(par_cle.values()), len(lignes) - len(par_cle)


def lire_sans_doublons(chemin: Path, cle: Callable[[dict], Hashable],
                       journal: logging.Logger = logger) -> list[dict]:
    """Les lignes d'un checkpoint, sans ligne illisible ni doublon, et le dit."""
    lignes, illisibles = lire_lignes(chemin)
    uniques, doublons = dedoublonner(lignes, cle)
    if illisibles:
        journal.warning(
            "%s : %d ligne(s) illisible(s) ignorée(s) -- run interrompu en pleine écriture ?",
            chemin, illisibles)
    if doublons:
        journal.info(
            "%s : %d doublon(s) écarté(s), la dernière écriture est retenue -- unités refaites "
            "après une reprise (--resume).", chemin, doublons)
    return uniques


def reecrire(chemin: Path, lignes: Iterable[dict]) -> None:
    """Réécrit le fichier d'un bloc, sans jamais le laisser à moitié écrit."""
    ecriture_atomique.ecrire_texte(
        chemin, "".join(json.dumps(ligne, default=str, ensure_ascii=False) + "\n" for ligne in lignes))


# --------------------------------------------------------------------------- #
# Sortie d'un run PARTIEL
# --------------------------------------------------------------------------- #
def _valeur_de_cle(valeur) -> str:
    """Une même clé lue du JSON ou du parquet : None et NaN se valent, 2023 et
    2023.0 aussi -- sans quoi une ligne refaite ne reconnaîtrait pas son
    ancienne version, et la fusion la doublerait."""
    if valeur is None:
        return ""
    if isinstance(valeur, float):
        if valeur != valeur:
            return ""
        if valeur.is_integer():
            return str(int(valeur))
    return str(valeur)


def _cles(df, colonnes: list[str]) -> list[tuple]:
    return [tuple(_valeur_de_cle(v) for v in ligne)
            for ligne in df[colonnes].itertuples(index=False, name=None)]


def fusionner_run_partiel(nouveau, chemin: Path, cle: list[str]):
    """(sortie fusionnée, nombre de lignes anciennes conservées).

    POURQUOI. 04c et 07b écrivaient leur fichier de sortie avec les SEULES
    lignes du run. Un essai sur un ticker (`--ticker AAPL`) ou quelques-uns
    (`--limit 5`) réduisait donc les 99 147 8-K du fichier à ceux d'AAPL -- et
    le filtre d'événements du backtest comme du paper trading avec, sans rien
    signaler. Un run partiel remplace désormais les lignes qu'il a refaites
    (même clé) et garde toutes les autres."""
    import pandas as pd

    if not Path(chemin).exists():
        return nouveau, 0
    ancien = pd.read_parquet(chemin)
    manquantes = [c for c in cle if c not in ancien.columns or c not in nouveau.columns]
    if manquantes:
        raise ValueError(f"{chemin} : colonne(s) de clé absente(s) {manquantes}, fusion impossible.")
    refaites = set(_cles(nouveau, cle))
    anciennes = ancien[[k not in refaites for k in _cles(ancien, cle)]]
    return pd.concat([anciennes, nouveau], ignore_index=True), len(anciennes)
