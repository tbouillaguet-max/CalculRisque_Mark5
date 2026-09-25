"""Aucun module ne lit un nom qu'il ne définit nulle part.

CE QUI S'EST PASSÉ. La fusion de `main` (passage du LLM à Gemini) dans une
branche qui avait, de son côté, fait évoluer 02 et 04c a gardé le code de
chaque côté mais perdu deux imports : `import sys` dans 02 (qui appelle
`sys.exit(1)` sur son chemin d'erreur) et `import os` dans 04c (qui lisait
`os.environ` dans son cache de classifications). Deux `NameError` en
production, sur des chemins que la suite ne parcourait pas toujours, et que
rien ne signalait à la relecture : le diff de la fusion montrait des
suppressions d'imports « inutiles » pour Gemini, justement.

LE CONTRÔLE. Analyse statique, sans dépendance : tout nom LU dans un module doit
y être lié quelque part (import, affectation, argument, définition) ou être un
builtin. Volontairement grossier -- il ne suit pas l'ordre d'exécution --, donc
sans faux positif sur ce dépôt, et il attrape exactement cette classe de casse.
Vérifié : sur les versions de 02 et 04c issues de la fusion, il signale `sys`
et `os`, et rien d'autre.
"""

from __future__ import annotations

import ast
import builtins
import pathlib

import pytest

RACINE = pathlib.Path(__file__).resolve().parent.parent
MODULES = sorted(
    p for motif in ("*.py", "backtest/*.py", "backtest/strategies/*.py",
                    "report/*.py", "report/pages/*.py", "tests/*.py")
    for p in RACINE.glob(motif)
)


def noms_non_lies(source: str) -> list[str]:
    arbre = ast.parse(source)
    lies = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__", "__builtins__"}
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Import):
            lies.update((a.asname or a.name).split(".")[0] for a in noeud.names)
        elif isinstance(noeud, ast.ImportFrom):
            lies.update(a.asname or a.name for a in noeud.names)
        elif isinstance(noeud, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lies.add(noeud.name)
        elif isinstance(noeud, ast.arg):
            lies.add(noeud.arg)
        elif isinstance(noeud, ast.Name) and not isinstance(noeud.ctx, ast.Load):
            lies.add(noeud.id)
        elif isinstance(noeud, ast.ExceptHandler) and noeud.name:
            lies.add(noeud.name)
        elif isinstance(noeud, (ast.Global, ast.Nonlocal)):
            lies.update(noeud.names)
        elif isinstance(noeud, ast.MatchAs) and noeud.name:
            lies.add(noeud.name)
    return sorted({
        f"{noeud.id} (ligne {noeud.lineno})" for noeud in ast.walk(arbre)
        if isinstance(noeud, ast.Name) and isinstance(noeud.ctx, ast.Load) and noeud.id not in lies
    })


@pytest.mark.parametrize("chemin", MODULES, ids=lambda p: str(p.relative_to(RACINE)))
def test_chaque_nom_lu_est_defini(chemin):
    fautes = noms_non_lies(chemin.read_text(encoding="utf-8-sig"))
    assert not fautes, f"{chemin.name} lit des noms jamais définis : {', '.join(fautes)}"


def test_le_controle_attrape_un_import_perdu():
    """Contrôle du contrôle, sur la forme exacte du défaut de la fusion."""
    source = "import logging\n\ndef main():\n    logging.error('x')\n    sys.exit(1)\n"
    assert noms_non_lies(source) == ["sys (ligne 5)"]
    assert noms_non_lies("import sys\n" + source) == []
