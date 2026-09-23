"""`--help` doit s'afficher, sur chaque script qui en a un.

CE QU'IL ATTAQUE. `python 09_backtest.py --help` plantait sur `ValueError:
unsupported format character ')'`. La cause : un `help=` interpolé À LA
CONSTRUCTION (`"... (%.0f%%) ..." % (...)`) puis RÉ-interpolé par argparse, qui
applique `help % params` au moment d'afficher l'aide. La première passe
produisait « (20%) », que la seconde lisait comme une directive de format
invalide.

POURQUOI CE TEST EXISTE. Une aide qui plante ne casse aucun backtest et
n'apparaît dans aucune métrique : elle ne se découvre qu'en tapant `--help`, ce
que personne ne fait sur un script qu'il utilise déjà. Le défaut est resté
invisible jusqu'à ce qu'on en ait besoin. Il est bon marché à empêcher.

CE N'EST PAS UN TEST DE CONTENU : on ne vérifie pas ce que l'aide dit, seulement
qu'elle s'affiche. C'est la seule propriété qui se dégrade en silence.
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import sys

import pytest

RACINE = pathlib.Path(__file__).resolve().parent.parent

# Les scripts numérotés du pipeline plus les modules de premier niveau qui
# exposent une ligne de commande. Découverts plutôt qu'énumérés : un script
# ajouté demain est couvert sans qu'on y pense.
SCRIPTS = sorted(
    p.name for p in RACINE.glob("*.py")
    if not p.name.startswith("test_") and "argparse" in p.read_text(errors="ignore")
)


def _aides(nom: str) -> list[tuple[tuple, str]]:
    """Chaînes `help=` que ce script déclare, sans lancer son traitement.

    On intercepte `parse_args` : le parseur est entièrement construit à ce
    moment-là, et rien de ce qui suit ne tourne."""
    collectees: list[tuple[tuple, str]] = []
    vrai_add = argparse.ArgumentParser.add_argument
    vrai_parse = argparse.ArgumentParser.parse_args
    vrai_parse_known = argparse.ArgumentParser.parse_known_args

    class _Stop(Exception):
        pass

    def espion(self, *a, **k):
        if isinstance(k.get("help"), str):
            collectees.append((a, k["help"]))
        return vrai_add(self, *a, **k)

    def stop(self, *a, **k):
        raise _Stop

    argparse.ArgumentParser.add_argument = espion
    argparse.ArgumentParser.parse_args = stop
    argparse.ArgumentParser.parse_known_args = stop
    argv = sys.argv
    try:
        sys.argv = [nom]
        module = importlib.import_module(nom[:-3])
        if hasattr(module, "main"):
            try:
                module.main()
            except _Stop:
                pass
    except _Stop:
        pass
    except BaseException as exc:                      # noqa: BLE001
        pytest.skip(f"{nom} n'est pas importable ici : {type(exc).__name__}: {exc}")
    finally:
        sys.argv = argv
        argparse.ArgumentParser.add_argument = vrai_add
        argparse.ArgumentParser.parse_args = vrai_parse
        argparse.ArgumentParser.parse_known_args = vrai_parse_known
    return collectees


@pytest.mark.parametrize("script", SCRIPTS)
def test_l_aide_s_affiche_sans_planter(script):
    """argparse applique `help % params` pour développer %(default)s. Tout `%`
    littéral doit donc être `%%` DANS LA CHAÎNE FINALE -- y compris quand cette
    chaîne est déjà le résultat d'un formatage."""
    # Les clés qu'argparse fournit réellement à l'interpolation (cf.
    # HelpFormatter._expand_help) : c'est ce jeu-là qui doit suffire.
    params = {"default": 0, "prog": script, "type": "float", "choices": "a, b",
              "metavar": "X", "dest": "x", "const": 1, "nargs": "+"}
    for option, aide in _aides(script):
        try:
            aide % params
        except (ValueError, KeyError, TypeError) as exc:
            pytest.fail(
                f"{script} {option} : --help plante ({type(exc).__name__}: {exc}).\n"
                f"Un % littéral doit s'écrire %% dans la chaîne FINALE. Si le help est "
                f"déjà formaté (\" ... \" % (...)), passe à une f-string et garde les %%.\n"
                f"Extrait : ...{aide[:180]}..."
            )


def test_le_script_qui_avait_le_defaut_est_bien_couvert():
    """Garde-fou de la découverte automatique : si `09_backtest.py` sortait de
    la liste, ce fichier passerait au vert sans rien vérifier."""
    assert "09_backtest.py" in SCRIPTS
    assert len(SCRIPTS) > 15, f"découverte suspecte : seulement {len(SCRIPTS)} scripts"
