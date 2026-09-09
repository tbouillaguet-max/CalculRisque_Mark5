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
