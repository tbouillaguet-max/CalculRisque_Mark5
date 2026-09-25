"""Rend la racine du dépôt importable depuis les tests.

Les scripts numérotés (04_recuperation_10k.py...) ne sont pas des identifiants
Python valides : les tests qui en ont besoin passent par importlib, exactement
comme le font déjà 04b/07 entre eux."""

from __future__ import annotations

import sys
from pathlib import Path

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
