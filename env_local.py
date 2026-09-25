"""
Clés et réglages lus dans le fichier `.env` à la racine du dépôt, en plus de
l'environnement. Chargé par config.py, donc par tous les scripts.

POURQUOI. Les scripts ne lisaient ces clés QUE dans l'environnement du
processus. Sous Windows, c'est un piège à trois étages : `setx` n'agit que sur
les terminaux ouverts APRÈS lui ; `$env:CLE = ...` ne vaut que pour la fenêtre
PowerShell où on le tape ; et `set CLE=...`, la syntaxe de cmd, crée en
PowerShell une variable qui n'est pas d'environnement. Un éditeur (VS Code...)
ou une tâche planifiée héritent en plus de l'environnement de LEUR lancement.
Résultat constaté : « Aucune clé LLM » alors que la clé avait bien été
« mise ». Le `.env`, lui, est lu par chaque script, d'où qu'il soit lancé --
et il existait déjà pour les identifiants d'IB Gateway (restart_gateway.py).

RÈGLES.
  - L'environnement GAGNE : une variable déjà définie n'est jamais écrasée.
  - Seules les variables de LISTE_BLANCHE sont chargées : le mot de passe IB
    du même fichier n'a rien à faire dans l'environnement de tous les scripts
    et de leurs sous-processus.
  - `.env` est ignoré par git (.gitignore) : une clé qui s'y trouve ne part
    jamais sur GitHub. Modèle : .env.example.
  - Syntaxe tolérante : `CLE=valeur`, guillemets autour de la valeur, `export `
    devant, commentaires `#`, BOM du Bloc-notes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

FICHIER = Path(__file__).resolve().parent / ".env"

LISTE_BLANCHE = frozenset({
    "SEC_CONTACT_EMAIL",
    "GEMINI_API_KEY", "GEMINI_MODEL",
    "MISTRAL_API_KEY", "MISTRAL_REQUESTS_PER_SECOND",
    "LLM_PROVIDER",
    "ALPHAVANTAGE_API_KEY",
})


def lire(chemin: Optional[Path] = None) -> dict[str, str]:
    """Paires CLE=valeur du fichier (FICHIER par défaut), toutes clés confondues."""
    chemin = chemin or FICHIER
    if not chemin.exists():
        return {}
    valeurs: dict[str, str] = {}
    for ligne in chemin.read_text(encoding="utf-8-sig").splitlines():
        ligne = ligne.strip()
        if not ligne or ligne.startswith("#") or "=" not in ligne:
            continue
        if ligne.startswith("export "):
            ligne = ligne[len("export "):].lstrip()
        cle, _, valeur = ligne.partition("=")
        valeur = valeur.strip()
        if len(valeur) >= 2 and valeur[0] == valeur[-1] and valeur[0] in "\"'":
            valeur = valeur[1:-1]
        elif " #" in valeur:
            valeur = valeur.split(" #", 1)[0].rstrip()   # commentaire en fin de ligne
        valeurs[cle.strip()] = valeur
    return valeurs


# Noms des variables que le dernier `charger` a posées -- jamais leurs valeurs.
CHARGEES: list[str] = []


def charger(chemin: Optional[Path] = None) -> list[str]:
    """Pose dans l'environnement les variables autorisées du `.env` qui n'y
    sont pas déjà ; rend leurs NOMS (jamais les valeurs, qui sont des clés)."""
    chargees = []
    for cle, valeur in lire(chemin).items():
        if cle in LISTE_BLANCHE and valeur and not os.environ.get(cle, "").strip():
            os.environ[cle] = valeur
            chargees.append(cle)
    CHARGEES[:] = chargees
    return chargees
