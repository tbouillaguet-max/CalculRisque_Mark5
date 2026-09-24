"""Remplacement atomique d'un fichier, qui survit aux verrous transitoires de Windows.

LE DÉFAUT. Tout le dépôt écrit ses fichiers d'état de la même façon : on écrit
`fichier.tmp`, puis `tmp.replace(fichier)`. Sur POSIX, ce renommage est
atomique et ne peut échouer que sur une vraie erreur de droits. Sur Windows,
il échoue avec `PermissionError: [WinError 5] Access is denied` dès qu'un
AUTRE processus tient la cible ouverte sans partage en suppression -- un
antivirus qui la scanne, l'indexeur de recherche, un client de synchronisation
(OneDrive synchronise le Bureau par défaut), un éditeur qui l'affiche.

Ces verrous durent de quelques millisecondes à une ou deux secondes. Un seul
suffisait à faire tomber `07b_validation_qualitative.py` au dernier
enregistrement de sa progression (run du 2026-09-05) -- et comme cet
enregistrement se trouve dans un `finally`, l'exception empêchait aussi
l'écriture finale du parquet de résultats.

LE CORRECTIF. Réessayer le remplacement, avec une attente qui double à chaque
tentative : 0,1 + 0,2 + 0,4 + 0,8 + 1,6 s, soit un peu plus de trois secondes
au total avant d'abandonner. Au-delà, le verrou n'est plus transitoire -- le
fichier est vraiment tenu ouvert --, et l'erreur remonte telle quelle : la
masquer ferait croire à une sauvegarde qui n'a pas eu lieu.

On réessaie sur toutes les plateformes, pas seulement Windows : sur POSIX,
une PermissionError n'est pas transitoire, et la réessayer retarde seulement
de trois secondes une erreur qui remontera de toute façon. En échange, le
comportement est le même partout, et il se teste sur n'importe quelle machine.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Union

logger = logging.getLogger("ecriture_atomique")

TENTATIVES = 6
PAUSE_INITIALE_SEC = 0.1

Chemin = Union[str, Path]


def remplacer(source: Chemin, cible: Chemin,
              tentatives: int = TENTATIVES, pause: float = PAUSE_INITIALE_SEC) -> None:
    """`os.replace(source, cible)`, réessayé sur PermissionError.

    Ne réessaie QUE sur PermissionError : un fichier source absent ou un
    disque plein ne se résolvent pas en attendant, et les réessayer
    retarderait une erreur qui doit remonter tout de suite."""
    for essai in range(tentatives):
        try:
            os.replace(source, cible)
            return
        except PermissionError:
            if essai == tentatives - 1:
                raise
            attente = pause * 2 ** essai
            logger.debug(
                "%s verrouillé par un autre processus : nouvel essai dans %.1f s (%d/%d).",
                cible, attente, essai + 2, tentatives)
            time.sleep(attente)


def ecrire_texte(cible: Chemin, texte: str, encoding: str = "utf-8") -> None:
    """Écrit `texte` dans `cible` sans jamais laisser un fichier à moitié écrit.

    Le fichier temporaire vit à côté de la cible (même volume, donc renommage
    atomique) et porte un suffixe `.tmp` ajouté au nom complet : `x.json`
    devient `x.json.tmp`, comme le faisaient déjà les scripts du dépôt."""
    cible = Path(cible)
    tmp = cible.with_name(cible.name + ".tmp")
    tmp.write_text(texte, encoding=encoding)
    remplacer(tmp, cible)
