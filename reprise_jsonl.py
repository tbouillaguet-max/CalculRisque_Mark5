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
