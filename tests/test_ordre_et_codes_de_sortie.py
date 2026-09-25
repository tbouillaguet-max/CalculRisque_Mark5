"""Les étapes s'enchaînent dans l'ordre de leurs données, et une étape qui échoue le DIT.

CE QUI S'EST PASSÉ. Les trois listes d'étapes (quotidienne, trimestrielle,
replay) lançaient 06b AVANT 07, alors que 06b lit `dcf_historique.parquet`, que
07 écrit. En run quotidien, le repli DCF de la valorisation combinée valorisait
donc les comptes du run précédent. En replay, c'était pire : l'espace de travail
part sans aucun DCF, 06b sortait sur « Fichier manquant »... avec le code 0. Le
replay se déclarait réussi et ne produisait AUCUNE valorisation combinée --
vérifié : quatre étapes « success », fichier absent.

DEUX DÉFAUTS, DONC, et le second a caché le premier :
  - l'ordre -- `test_chaque_lecteur_passe_apres_son_ecrivain` ;
  - le `return` après un `logger.error` dans neuf `main()` de six scripts, qui
    faisait sortir le processus avec le code 0 -- `test_une_etape_privee_de_ses_entrees_echoue_bruyamment`.

L'ancien test d'ordre DISAIT dans sa docstring que « 06b a besoin du DCF (07) »,
mais ne l'assertait pas. D'où, ici, des dépendances LUES DANS LE CODE plutôt
que recopiées dans une liste.
"""

from __future__ import annotations

import ast
import pathlib
import re
import subprocess
import sys

import pytest

import run_pipeline_daily as daily
import run_pipeline_quarterly as quarterly

RACINE = pathlib.Path(__file__).resolve().parent.parent

ECRIT = re.compile(r"(?:to_parquet|ExcelWriter|to_csv|to_excel)\(\s*config\.([A-Z_]+_FILE)\b")
LIT = re.compile(r"(?:read_parquet|read_excel|read_csv)\(\s*config\.([A-Z_]+_FILE)\b")

ORCHESTRATEURS = {
    "quotidien": lambda: daily.daily_steps(7),
    "trimestriel": lambda: quarterly.LIVE_STEPS,
    "replay": lambda: quarterly.REPLAY_STEPS,
}


def _dependances(scripts: list[str]) -> list[tuple[str, str, str]]:
    """(écrivain, lecteur, fichier) déduits des appels `config.X_FILE` directs.

    UN PLANCHER, PAS UN GRAPHE COMPLET : une lecture qui passe par un helper
    ou une variable intermédiaire (la sortie de 06b, les cours de 03b) échappe
    à cette lecture. Mais chaque arête trouvée est une vraie dépendance -- pas
    de faux positif --, donc l'assertion qui suit ne peut pas mentir."""
    ecrit, lit = {}, {}
    for script in scripts:
        texte = (RACINE / script).read_text(encoding="utf-8")
        ecrit[script] = set(ECRIT.findall(texte))
        lit[script] = set(LIT.findall(texte)) - ecrit[script]
    return [(e, l, f) for l in scripts for f in lit[l] for e in scripts if f in ecrit[e]]


@pytest.mark.parametrize("nom", sorted(ORCHESTRATEURS))
def test_chaque_lecteur_passe_apres_son_ecrivain(nom):
    """LE garde-fou. Les étapes tournent en sous-processus : rien ne rattrape un
    ordre faux, chacune lit simplement le fichier du run précédent sans le
    dire."""
    etapes = [s.script for s in ORCHESTRATEURS[nom]()]
    rang = {s: i for i, s in enumerate(etapes)}
    fautes = [f"{l} lit {f} AVANT que {e} ne l'écrive"
              for e, l, f in _dependances(etapes) if e != l and rang[e] > rang[l]]
    assert not fautes, f"run {nom} :\n" + "\n".join(fautes)


@pytest.mark.parametrize("nom", sorted(ORCHESTRATEURS))
def test_le_cas_du_dcf_est_bien_couvert(nom):
    """Contrôle du plancher : la dépendance qui a causé le défaut doit être
    TROUVÉE par la lecture du code, sinon le test précédent passerait au vert
    sans rien vérifier."""
    etapes = [s.script for s in ORCHESTRATEURS[nom]()]
    assert ("07_calcul_dcf.py", "06b_calcul_valorisation_combinee.py", "DCF_HISTORY_FILE") \
        in _dependances(etapes)
    assert etapes.index("07_calcul_dcf.py") < etapes.index("06b_calcul_valorisation_combinee.py")


# --------------------------------------------------------------------------- #
# Codes de sortie
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", [
    "05_calcul_multiples.py", "06_calcul_multiples_moyens.py",
    "07_calcul_dcf.py", "06b_calcul_valorisation_combinee.py",
])
def test_une_etape_privee_de_ses_entrees_echoue_bruyamment(script, tmp_path):
    """C'EST CE QUI A CACHÉ LE DÉFAUT D'ORDRE EN REPLAY. Lancée dans un espace
    vide, chaque étape requise doit sortir en ERREUR -- l'orchestrateur ne
    juge une étape qu'à son code de sortie, et un « Fichier manquant » avec le
    code 0 devient un succès.

    Lancé comme l'orchestrateur le fait : en sous-processus, depuis un
    répertoire de travail où `data/` (chemin RELATIF de config.BASE_DIR) est
    vide."""
    (tmp_path / "data").mkdir()
    resultat = subprocess.run(
        [sys.executable, str(RACINE / script)], cwd=tmp_path,
        capture_output=True, text=True, timeout=120,
        env={**__import__("os").environ, "PYTHONPATH": str(RACINE)},
    )
    assert resultat.returncode != 0, (
        f"{script} privé de ses entrées sort avec le code 0 -- l'orchestrateur le "
        f"croira réussi.\n{resultat.stderr[-500:]}")


def test_aucun_main_ne_sort_en_silence_apres_une_erreur():
    """La forme générale du défaut, cherchée dans tous les scripts numérotés :
    un `logger.error(...)` suivi d'un `return` nu dans `main()`. Neuf
    occurrences dans six scripts avant ce correctif."""
    fautes = []
    for p in sorted(RACINE.glob("[0-9]*.py")):
        for f in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if not (isinstance(f, ast.FunctionDef) and f.name == "main"):
                continue
            for n in ast.walk(f):
                if not isinstance(n, ast.If):
                    continue
                erreur = any(isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                             and getattr(s.value.func, "attr", "") == "error" for s in n.body)
                retour_nu = any(isinstance(s, ast.Return) and s.value is None for s in n.body)
                if erreur and retour_nu:
                    fautes.append(f"{p.name}:{n.lineno}")
    assert not fautes, (
        "main() journalise une erreur puis sort avec le code 0 -- utilise sys.exit(1) :\n"
        + "\n".join(fautes))
