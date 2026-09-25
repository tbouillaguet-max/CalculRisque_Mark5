"""Rend la racine du dépôt importable depuis les tests, et protège `data/`.

Les scripts numérotés (04_recuperation_10k.py...) ne sont pas des identifiants
Python valides : les tests qui en ont besoin passent par importlib, exactement
comme le font déjà 04b/07 entre eux.

POURQUOI `data/` A BESOIN D'UN GARDE-FOU. `config.BASE_DIR` vaut `data` -- un
chemin RELATIF. Tous les chemins de production en découlent, donc tout ce qui
tourne depuis la racine du dépôt écrit dans les VRAIES données : il suffit
qu'un test appelle le `main()` d'un script de pipeline pour que
`multiples.parquet` ou `dcf_historique.parquet` soient réécrits. Rien ne
l'empêcherait, rien ne le signalerait, et le dégât ne se verrait qu'au prochain
`git status` -- sur des fichiers LFS de plusieurs mégaoctets, au milieu d'un
travail sans rapport.

Mesuré au moment où ce garde-fou est posé : aucun test n'écrit dans `data/`.
Il ne répare donc rien ; il empêche une régression facile à introduire et
coûteuse à diagnostiquer, pour deux parcours de répertoire par session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest


@pytest.fixture(autouse=True)
def _isoler_les_cles_llm(monkeypatch):
    """Le fournisseur de LLM se choisit sur les variables d'environnement
    (sec_filings_text.fournisseur_llm). Une GEMINI_API_KEY définie sur la
    machine de développement ferait basculer vers Gemini des tests écrits
    pour Mistral : chaque test repart donc d'un environnement sans clé, et
    pose lui-même celles dont il a besoin."""
    for nom in ("GEMINI_API_KEY", "GEMINI_MODEL", "MISTRAL_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(nom, raising=False)
