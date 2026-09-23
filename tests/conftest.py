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

DATA = ROOT / "data"


def pytest_addoption(parser):
    parser.addoption(
        "--data-guard-per-test", action="store_true", default=False,
        help="Vérifie data/ APRÈS CHAQUE TEST au lieu d'une fois par session, pour "
             "nommer le test fautif. Double environ la durée de la suite : à n'activer "
             "que pour un diagnostic.",
    )


def _empreinte() -> dict[Path, tuple[int, int]]:
    """État de `data/` : taille et date de modification par fichier.

    Pas de hachage : les parquet du dépôt pèsent une cinquantaine de
    mégaoctets, et une réécriture change toujours la mtime -- même à contenu
    identique, ce qui reste ce qu'on veut savoir."""
    if not DATA.is_dir():
        return {}
    return {p: (p.stat().st_mtime_ns, p.stat().st_size)
            for p in DATA.rglob("*") if p.is_file()}


def _touches(avant: dict, apres: dict) -> list[str]:
    return sorted(
        str(p.relative_to(DATA)) for p in set(avant) | set(apres)
        if avant.get(p) != apres.get(p)
    )


def _echoue(contexte: str, fichiers: list[str]) -> None:
    apercu = "\n".join(f"    {f}" for f in fichiers[:12])
    reste = f"\n    ... et {len(fichiers) - 12} autres" if len(fichiers) > 12 else ""
    pytest.fail(
        f"{contexte} a écrit dans data/ :\n{apercu}{reste}\n\n"
        "Les tests ne doivent pas toucher aux données de production. `config.BASE_DIR` "
        "est RELATIF (`data`), donc tout ce qui tourne depuis la racine du dépôt y écrit : "
        "un `main()` de script de pipeline suffit. Utilise tmp_path et monkeypatch les "
        "chemins de config, ou n'appelle pas la fonction qui écrit.\n"
        "Pour trouver le test fautif : pytest --data-guard-per-test",
        pytrace=False,
    )


@pytest.fixture(autouse=True)
def _garde_data_par_test(request):
    """Actif seulement sous --data-guard-per-test : nomme le test fautif.

    Par test, le parcours de `data/` coûte à peu près autant que la suite
    entière -- d'où le mode par session par défaut, et celui-ci à la demande."""
    if not request.config.getoption("--data-guard-per-test"):
        yield
        return
    avant = _empreinte()
    yield
    touches = _touches(avant, _empreinte())
    if touches:
        _echoue(f"Le test {request.node.nodeid}", touches)


def pytest_configure(config):
    config._empreinte_data = _empreinte()


def pytest_sessionfinish(session, exitstatus):
    """Deux parcours par session : le prix de l'invariant.

    On ne signale rien si la session s'est déjà mal passée -- un échec de test
    qui laisse un fichier derrière lui produirait un second message qui masque
    le premier."""
    avant = getattr(session.config, "_empreinte_data", None)
    if avant is None or exitstatus != 0:
        return
    touches = _touches(avant, _empreinte())
    if touches:
        session.exitstatus = 1
        apercu = "\n".join(f"    {f}" for f in touches[:12])
        reste = f"\n    ... et {len(touches) - 12} autres" if len(touches) > 12 else ""
        print(
            f"\n\nLA SUITE A ÉCRIT DANS data/ :\n{apercu}{reste}\n\n"
            "Les tests ne doivent pas toucher aux données de production "
            "(config.BASE_DIR est relatif : tout ce qui tourne depuis la racine y écrit).\n"
            "Pour trouver le test fautif : pytest --data-guard-per-test\n"
        )
