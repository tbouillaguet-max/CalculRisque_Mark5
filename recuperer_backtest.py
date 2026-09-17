#!/usr/bin/env python3
"""
Rapatrie sur ce PC les résultats d'un backtest lancé sur GitHub Actions.

POURQUOI CE SCRIPT. Les sorties de backtest ne sont pas versionnées (voir
.gitignore) : ce sont des résultats, pas des entrées, et elles pesaient la
moitié du dépôt. Un backtest lancé depuis l'onglet Actions publie donc ses
sorties sous forme d'ARTEFACT, et c'est ce script qui va les chercher pour les
déposer dans data/backtest_options/ (ou data/backtest/ pour la stratégie
actions) -- exactement là où `14_audit_backtest.py`, `make audit` et le
dashboard Streamlit vont les lire, comme si le run avait tourné en local.

UTILISATION

    python recuperer_backtest.py                # le dernier backtest en date
    python recuperer_backtest.py --liste        # ce qui est disponible
    python recuperer_backtest.py --run-id 123   # un run GitHub précis
    python recuperer_backtest.py --ecraser      # remplace les fichiers déjà là

AUTHENTIFICATION. L'API des artefacts exige un jeton, même sur un dépôt
public. Le script en cherche un dans cet ordre :

    1. la variable d'environnement GITHUB_TOKEN ou GH_TOKEN ;
    2. `gh auth token`, si le CLI GitHub est installé et connecté.

Sans jeton, il explique comment en créer un (droit `repo`, ou `actions:read`
seul pour un jeton à portée fine) plutôt que d'échouer sur un 401 obscur.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

import requests

API = "https://api.github.com"
PREFIXE_ARTEFACT = "backtest-"

# Dossiers attendus à la racine de l'archive, et où les déposer ici. La
# structure vient du dossier "artefact/" que rassemble le workflow : sans lui,
# la racine du zip dépendrait des fichiers trouvés et changerait d'un run à
# l'autre (voir .github/workflows/backtest.yml).
DESTINATIONS = {
    "backtest": Path("data") / "backtest",
    "backtest_options": Path("data") / "backtest_options",
}
FICHIER_META = "run_distant.json"


# ---------------------------------------------------------------------------
# Dépôt et authentification
# ---------------------------------------------------------------------------

_REMOTE = re.compile(
    r"(?:https://[^/]+/|git@[^:]+:)(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)


def depot_depuis_remote(url: str) -> Optional[tuple[str, str]]:
    """(propriétaire, dépôt) à partir d'une URL de remote, HTTPS ou SSH."""
    m = _REMOTE.search(url.strip())
    return (m.group("owner"), m.group("repo")) if m else None


def depot_courant() -> Optional[tuple[str, str]]:
    r = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True, text=True, check=False,
    )
    return depot_depuis_remote(r.stdout) if r.returncode == 0 else None


def jeton() -> Optional[str]:
    """Jeton GitHub, depuis l'environnement ou le CLI `gh`."""
    for variable in ("GITHUB_TOKEN", "GH_TOKEN"):
        valeur = os.environ.get(variable)
        if valeur:
            return valeur.strip()
    if shutil.which("gh"):
        r = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    return None


# ---------------------------------------------------------------------------
# API GitHub
# ---------------------------------------------------------------------------

def _entetes(cle: str) -> dict:
    return {
        "Authorization": f"Bearer {cle}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def lister_artefacts(owner: str, repo: str, cle: str) -> list[dict]:
    r = requests.get(
        f"{API}/repos/{owner}/{repo}/actions/artifacts",
        headers=_entetes(cle), params={"per_page": 100}, timeout=30,
    )
    r.raise_for_status()
    return r.json().get("artifacts", [])


def artefacts_de_backtest(artefacts: list[dict], run_id: Optional[int] = None) -> list[dict]:
    """Ceux que le workflow Backtest a produits, du plus récent au plus ancien.

    Les artefacts EXPIRÉS sont écartés : GitHub les garde listés après leur
    date de rétention alors que leur téléchargement renvoie 410, ce qui
    donnerait une erreur incompréhensible si on en choisissait un."""
    retenus = [
        a for a in artefacts
        if a.get("name", "").startswith(PREFIXE_ARTEFACT) and not a.get("expired")
    ]
    if run_id is not None:
        retenus = [
            a for a in retenus
            if (a.get("workflow_run") or {}).get("id") == run_id
        ]
    return sorted(retenus, key=lambda a: a.get("created_at", ""), reverse=True)


def telecharger(owner: str, repo: str, artefact: dict, cle: str) -> bytes:
    r = requests.get(
        f"{API}/repos/{owner}/{repo}/actions/artifacts/{artefact['id']}/zip",
        headers=_entetes(cle), timeout=300,
    )
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def destination_membre(nom: str, racine: Path = Path(".")) -> Optional[Path]:
    """Où déposer un membre de l'archive ici, None s'il n'est pas attendu.

    Le nom d'un membre de zip est une donnée comme une autre : un "../" ou un
    chemin absolu y écrirait hors du dépôt. On ne mappe donc que les préfixes
    connus, et on vérifie que le résultat reste sous sa destination."""
    if nom.endswith("/"):
        return None
    propre = nom.replace("\\", "/").lstrip("/")
    if propre == FICHIER_META:
        return racine / FICHIER_META

    tete, _, reste = propre.partition("/")
    base = DESTINATIONS.get(tete)
    if base is None or not reste:
        return None

    autorise = (racine / base).resolve()
    cible = (racine / base / reste).resolve()
    # is_relative_to plutôt qu'une comparaison de préfixes de chaînes, qui
    # accepterait un dossier voisin au nom plus long (".../backtest_evil"
    # commence bien par ".../backtest").
    return cible if cible.is_relative_to(autorise) else None


def extraire(archive: bytes, racine: Path = Path("."), ecraser: bool = False) -> dict:
    """Dépose les résultats. Ne touche à rien d'autre que les dossiers de
    DESTINATIONS, et ne remplace un fichier existant que sur demande."""
    compteurs = {"ecrits": 0, "ignores": 0, "hors_perimetre": 0}
    meta: Optional[dict] = None

    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        for membre in zf.infolist():
            if membre.is_dir():
                continue
            cible = destination_membre(membre.filename, racine)
            if cible is None:
                compteurs["hors_perimetre"] += 1
                continue
            if cible.name == FICHIER_META:
                try:
                    meta = json.loads(zf.read(membre).decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    meta = None
                continue
            if cible.exists() and not ecraser:
                compteurs["ignores"] += 1
                continue
            cible.parent.mkdir(parents=True, exist_ok=True)
            cible.write_bytes(zf.read(membre))
            compteurs["ecrits"] += 1

    compteurs["meta"] = meta
    return compteurs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _mo(octets: float) -> str:
    return f"{octets / (1024 * 1024):.1f} Mo"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--liste", action="store_true",
                   help="Affiche les backtests disponibles et s'arrête.")
    p.add_argument("--run-id", type=int, default=None,
                   help="Numéro du run GitHub à rapatrier (défaut : le plus récent).")
    p.add_argument("--ecraser", action="store_true",
                   help="Remplace les fichiers déjà présents au lieu de les garder.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    depot = depot_courant()
    if depot is None:
        print("Remote « origin » introuvable : lance ce script depuis le dépôt.",
              file=sys.stderr)
        return 1
    owner, repo = depot

    cle = jeton()
    if cle is None:
        print(
            "Aucun jeton GitHub trouvé, et l'API des artefacts en exige un même "
            "sur un dépôt public.\n"
            "  Le plus simple : installe le CLI GitHub (https://cli.github.com) "
            "puis `gh auth login`.\n"
            "  Sinon, crée un jeton sur https://github.com/settings/tokens "
            "(portée `repo`, ou `actions:read` pour un jeton à portée fine) et "
            "expose-le :\n"
            "      export GITHUB_TOKEN=ghp_...        # Git Bash / macOS / Linux\n"
            "      setx GITHUB_TOKEN ghp_...          # Windows, puis rouvre le terminal",
            file=sys.stderr,
        )
        return 1

    try:
        artefacts = artefacts_de_backtest(
            lister_artefacts(owner, repo, cle), run_id=args.run_id
        )
    except requests.HTTPError as exc:
        print(f"Appel à l'API GitHub refusé : {exc}\n"
              "  Vérifie que le jeton a bien le droit de lire les Actions de ce dépôt.",
              file=sys.stderr)
        return 1

    if not artefacts:
        quoi = f"pour le run {args.run_id}" if args.run_id else "dans ce dépôt"
        print(
            f"Aucun résultat de backtest {quoi}.\n"
            "  Lance-en un : onglet Actions > Backtest > Run workflow.\n"
            "  (Les artefacts sont conservés 30 jours ; au-delà il faut relancer.)",
            file=sys.stderr,
        )
        return 1

    if args.liste:
        print(f"Backtests disponibles sur {owner}/{repo} :\n")
        for a in artefacts:
            run = (a.get("workflow_run") or {}).get("id", "?")
            print(f"  run {run:<12} {a['created_at'][:16].replace('T', ' ')}  "
                  f"{_mo(a['size_in_bytes']):>10}  {a['name']}")
        print("\nRapatrie le dernier avec : python recuperer_backtest.py")
        return 0

    choisi = artefacts[0]
    run = (choisi.get("workflow_run") or {}).get("id", "?")
    print(f"Téléchargement de {choisi['name']} ({_mo(choisi['size_in_bytes'])}, run {run})...")
    archive = telecharger(owner, repo, choisi, cle)

    compteurs = extraire(archive, ecraser=args.ecraser)
    meta = compteurs.pop("meta")

    if meta:
        print(f"\n  stratégie : {meta.get('strategie')}")
        print(f"  depuis    : {meta.get('date_debut')}")
        if meta.get("args_supplementaires"):
            print(f"  options   : {meta['args_supplementaires']}")
        print(f"  commit    : {str(meta.get('commit'))[:12]}")

    print(f"\n{compteurs['ecrits']} fichiers déposés dans "
          f"{DESTINATIONS['backtest_options']}/ et {DESTINATIONS['backtest']}/.")
    if compteurs["ignores"]:
        print(f"{compteurs['ignores']} déjà présents, laissés en place "
              f"(--ecraser pour les remplacer).")
    if compteurs["ecrits"]:
        print("\nRelis-les avec : python 14_audit_backtest.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
