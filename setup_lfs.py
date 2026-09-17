#!/usr/bin/env python3
"""
Met data/ dans git en passant par Git LFS, et chiffre ce que ça coûte AVANT
le premier push.

POURQUOI LFS. data/ était exclu du dépôt (.gitignore) parce que git stocke
chaque version complète d'un fichier binaire dans son historique, pour
toujours : un daily_prices.parquet de 80 Mo réécrit à chaque run ajoute 80 Mo
au .git à chaque commit, et le dépôt devient impossible à cloner. Git LFS ne
met qu'un POINTEUR de 130 octets dans l'historique ; le contenu part sur le
stockage LFS du serveur et n'est rapatrié qu'au checkout.

CE QUE FAIT CE SCRIPT, dans l'ordre :

    1. vérifie que git-lfs est installé, et l'active pour ce dépôt ;
    2. demande à GIT LUI-MÊME (git check-attr) quels fichiers de data/ sont
       réellement couverts par les motifs de .gitattributes -- pas une
       ré-implémentation du filtrage, qui finirait par diverger ;
    3. chiffre le volume : ce qui part en LFS, ce qui reste en git brut, la
       répartition par sous-dossier, et les plus gros fichiers ;
    4. compare au palier gratuit GitHub (1 Go de stockage LFS, 1 Go de bande
       passante par mois) et dit franchement si ça passe ;
    5. avec --apply, indexe data/ et vérifie que les pointeurs LFS sont bien
       posés (git lfs ls-files).

UTILISATION

    python3 setup_lfs.py            # SIMULATION : diagnostic + chiffrage
    python3 setup_lfs.py --apply    # active LFS et indexe data/
    python3 setup_lfs.py --apply --commit "Ajout des donnees du pipeline"

Sans --apply, le script ne modifie rien : ni l'index git, ni la config du
dépôt. Il est relançable autant de fois que voulu.

SI LE VOLUME NE PASSE PAS. Le script liste le poids de chaque sous-dossier de
data/ : exclure le plus gros (`echo "data/xxx/" >> .gitignore`) est la seule
chose à faire, aucun script n'a besoin qu'il soit dans git pour tourner -- le
pipeline le régénère.

CÔTÉ MACHINE QUI CLONE. git-lfs doit y être installé aussi, sinon le checkout
ne récupère que les pointeurs et pandas échoue à ouvrir les parquet. Voir la
section LFS du README.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, NamedTuple, Optional

GO = 1024 ** 3
MO = 1024 ** 2

# Palier gratuit GitHub, à la date de rédaction : 1 Go de stockage LFS et 1 Go
# de bande passante par mois. Au-delà, "data pack" à 5 $/mois pour 50 Go de
# chaque. Chiffres à revérifier chez GitHub, ils servent ici de repère d'ordre
# de grandeur, pas de contrat.
QUOTA_LFS_GRATUIT = 1 * GO
DATA_PACK_GO = 50
DATA_PACK_USD = 5

DOSSIER_DONNEES = Path("data")


class Fichier(NamedTuple):
    chemin: Path
    relatif: str
    taille: int
    lfs: bool


# ---------------------------------------------------------------------------
# Dialogue avec git
# ---------------------------------------------------------------------------

def _git(*args: str, entree: Optional[str] = None) -> subprocess.CompletedProcess:
    """Appelle git et rend sa sortie décodée.

    Quand on ALIMENTE stdin, l'écriture se fait en binaire, jamais en mode
    texte. Sous Windows, `text=True` fait traduire par Python chaque "\\n" en
    "\\r\\n" ; or `git check-attr --stdin` et `git check-ignore --stdin` lisent
    leurs lignes avec strbuf_getline_lf, qui ne retire QUE le LF. Chaque chemin
    envoyé arrivait donc suffixé d'un CR, ne correspondait à aucun motif de
    .gitattributes, et le script concluait que rien n'allait en LFS -- sur un
    dépôt réel : 0 fichier sur 1116. Bug invisible sous Linux, où os.linesep
    vaut déjà "\\n"."""
    if entree is None:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False,
        )

    brut = subprocess.run(
        ["git", *args], input=entree.encode("utf-8"), capture_output=True, check=False,
    )
    return subprocess.CompletedProcess(
        brut.args, brut.returncode,
        brut.stdout.decode("utf-8", "replace"),
        brut.stderr.decode("utf-8", "replace"),
    )


def git_lfs_installe() -> Optional[str]:
    """Version de git-lfs, ou None s'il n'est pas installé."""
    if shutil.which("git-lfs") is None:
        return None
    r = _git("lfs", "version")
    return r.stdout.strip() if r.returncode == 0 else None


def racine_depot() -> Optional[Path]:
    r = _git("rev-parse", "--show-toplevel")
    return Path(r.stdout.strip()) if r.returncode == 0 else None


def activer_lfs() -> None:
    """`git lfs install --local` : installe les filtres LFS dans .git/config,
    pas dans le ~/.gitconfig de la machine. Le dépôt se configure lui-même
    sans toucher aux autres dépôts de l'utilisateur."""
    r = _git("lfs", "install", "--local")
    if r.returncode != 0:
        raise RuntimeError(f"git lfs install a échoué : {r.stderr.strip()}")


# Première ligne d'un fichier-pointeur LFS (spec v1). Un pointeur fait ~130
# octets et remplace le contenu tant qu'il n'a pas été rapatrié.
ENTETE_POINTEUR = b"version https://git-lfs.github.com/spec/v1"


def est_pointeur(chemin: Path) -> bool:
    """Ce fichier est-il un pointeur LFS non rapatrié plutôt que son contenu ?"""
    try:
        with chemin.open("rb") as f:
            return f.read(len(ENTETE_POINTEUR)) == ENTETE_POINTEUR
    except OSError:
        return False


def pointeurs_restants(racine: Path, ignorer: Iterable[str] = ()) -> list[Path]:
    """Fichiers de `racine` restés à l'état de pointeur, hors motifs ignorés.

    POURQUOI C'EST UTILE. Un pointeur non rapatrié ne dit pas son nom : pandas
    l'ouvre et échoue sur « Parquet magic bytes not found in footer », une
    erreur qui envoie chercher une corruption de données là où il ne manque
    qu'un `git lfs pull`. Le cas s'est produit en vrai -- un backtest lancé sur
    un dépôt dont un seul parquet n'avait pas été rapatrié."""
    motifs = [m.strip() for m in ignorer if m.strip()]
    restants = []
    for chemin in sorted(racine.rglob("*")):
        if not chemin.is_file():
            continue
        relatif = chemin.as_posix()
        if any(fnmatch.fnmatch(relatif, m) for m in motifs):
            continue
        if est_pointeur(chemin):
            restants.append(chemin)
    return restants


def chemins_suivis_par_lfs(relatifs: list[str]) -> set[str]:
    """Parmi ces chemins, ceux que git enverra réellement en LFS.

    On INTERROGE GIT (`git check-attr filter --stdin`) au lieu de refaire le
    filtrage de .gitattributes à la main : c'est la seule façon que le
    chiffrage affiché ici corresponde exactement à ce qui se passera au
    `git add`. Un motif mal écrit se voit alors tout de suite dans le rapport,
    au lieu d'être découvert après le push."""
    if not relatifs:
        return set()
    r = _git("check-attr", "filter", "--stdin", entree="\n".join(relatifs) + "\n")
    if r.returncode != 0:
        raise RuntimeError(f"git check-attr a échoué : {r.stderr.strip()}")

    suivis: set[str] = set()
    for ligne in r.stdout.splitlines():
        # Format : "<chemin>: filter: <valeur>" -- le chemin peut contenir
        # ": ", d'où le découpage par la DROITE sur les deux derniers champs.
        reste, _, valeur = ligne.rpartition(": ")
        chemin, _, _ = reste.rpartition(": ")
        if valeur.strip() == "lfs":
            suivis.add(chemin)
    return suivis


def fichiers_ignores(relatifs: list[str]) -> set[str]:
    """Ceux que .gitignore exclut : ils ne partiront ni en LFS ni en git, et
    ne doivent donc pas être comptés dans le chiffrage."""
    if not relatifs:
        return set()
    r = _git("check-ignore", "--stdin", entree="\n".join(relatifs) + "\n")
    # check-ignore sort en 1 quand AUCUN chemin n'est ignoré : ce n'est pas
    # une erreur.
    if r.returncode not in (0, 1):
        raise RuntimeError(f"git check-ignore a échoué : {r.stderr.strip()}")
    return {l for l in r.stdout.splitlines() if l}


# ---------------------------------------------------------------------------
# Inventaire
# ---------------------------------------------------------------------------

def inventorier(racine: Path) -> list[Fichier]:
    """Tous les fichiers de `racine` qui finiront dans git, avec leur taille et
    leur mode de stockage (LFS ou git brut)."""
    chemins = sorted(p for p in racine.rglob("*") if p.is_file())
    relatifs = [p.as_posix() for p in chemins]

    ignores = fichiers_ignores(relatifs)
    retenus = [(p, r) for p, r in zip(chemins, relatifs) if r not in ignores]
    lfs = chemins_suivis_par_lfs([r for _, r in retenus])

    return [
        Fichier(chemin=p, relatif=r, taille=p.stat().st_size, lfs=r in lfs)
        for p, r in retenus
    ]


def grouper_par_dossier(fichiers: Iterable[Fichier], profondeur: int = 2) -> dict:
    """{sous-dossier: (nb, octets LFS, octets git brut)}, regroupé sur les
    `profondeur` premiers segments du chemin (data/prices, data/options...)."""
    par_dossier: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for f in fichiers:
        segments = f.relatif.split("/")[:profondeur]
        cle = "/".join(segments) if len(segments) == profondeur else f.relatif
        par_dossier[cle][0] += 1
        par_dossier[cle][1 if f.lfs else 2] += f.taille
    return {k: tuple(v) for k, v in par_dossier.items()}


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

def _taille(octets: float) -> str:
    if octets >= GO:
        return f"{octets / GO:6.2f} Go"
    return f"{octets / MO:6.1f} Mo"


def afficher_rapport(fichiers: list[Fichier], racine: Path) -> None:
    total_lfs = sum(f.taille for f in fichiers if f.lfs)
    total_git = sum(f.taille for f in fichiers if not f.lfs)
    n_lfs = sum(1 for f in fichiers if f.lfs)

    print(f"\n{racine} : {len(fichiers)} fichiers, {_taille(total_lfs + total_git)}")
    print(f"  {_taille(total_lfs)}  {n_lfs:5d} fichiers  -> Git LFS (pointeurs dans git)")
    print(f"  {_taille(total_git)}  {len(fichiers) - n_lfs:5d} fichiers  -> git en direct")

    print("\nPar sous-dossier")
    print(f"  {'total':>9}  {'dont LFS':>9}  {'fichiers':>8}")
    groupes = grouper_par_dossier(fichiers)
    for dossier, (nb, octets_lfs, octets_git) in sorted(
        groupes.items(), key=lambda kv: -(kv[1][1] + kv[1][2])
    ):
        print(f"  {_taille(octets_lfs + octets_git)}  {_taille(octets_lfs)}  {nb:8d}  {dossier}")

    lourds = sorted(fichiers, key=lambda f: -f.taille)[:8]
    if lourds and lourds[0].taille > 5 * MO:
        print("\nLes plus gros fichiers")
        for f in lourds:
            if f.taille < MO:
                break
            marque = "LFS" if f.lfs else "git"
            print(f"  {_taille(f.taille)}  [{marque}]  {f.relatif}")

    _avertir_gros_hors_lfs(fichiers)
    _avertir_quota(total_lfs, total_git, fichiers)


# Au-delà de cette taille, un fichier laissé hors LFS pèse durablement sur le
# dépôt : git garde chaque version dans son historique, et l'historique ne se
# purge pas.
SEUIL_GROS_FICHIER = 5 * MO


def _avertir_gros_hors_lfs(fichiers: list[Fichier]) -> None:
    """Les fichiers volumineux qu'aucun motif de .gitattributes n'attrape.

    C'est l'angle mort du montage : ils s'ajoutent EN DUR à l'historique git,
    sans rien signaler, et les en retirer plus tard demande de réécrire
    l'historique. Typiquement les journaux de collecte de 08, qu'on imagine
    petits et qui atteignent la vingtaine de Mo sur un univers complet."""
    gros = [f for f in fichiers if not f.lfs and f.taille >= SEUIL_GROS_FICHIER]
    if not gros:
        return

    total = sum(f.taille for f in gros)
    print(
        f"\nGros fichiers HORS LFS : {len(gros)} fichiers, {_taille(total).strip()}"
    )
    print("  Ils entreront en dur dans l'historique git, qui ne se purge pas.")
    for f in sorted(gros, key=lambda f: -f.taille)[:5]:
        print(f"    {_taille(f.taille)}  {f.relatif}")
    if len(gros) > 5:
        print(f"    ... et {len(gros) - 5} autres")
    print(
        "  Deux issues : ajouter leur motif à .gitattributes pour les basculer "
        "en LFS, ou les exclure via .gitignore s'ils ne servent qu'au diagnostic."
    )


def _avertir_quota(total_lfs: int, total_git: int, fichiers: list[Fichier]) -> None:
    print("\nQuota")
    if total_lfs <= QUOTA_LFS_GRATUIT:
        reste = QUOTA_LFS_GRATUIT - total_lfs
        print(
            f"  Le premier push consomme {_taille(total_lfs).strip()} des "
            f"{_taille(QUOTA_LFS_GRATUIT).strip()} du palier gratuit GitHub "
            f"({_taille(reste).strip()} restants)."
        )
    else:
        packs = -(-total_lfs // (DATA_PACK_GO * GO))
        print(
            f"  ATTENTION : {_taille(total_lfs).strip()} dépassent le palier "
            f"gratuit ({_taille(QUOTA_LFS_GRATUIT).strip()}). Il faudra "
            f"{packs} data pack(s) GitHub (~{packs * DATA_PACK_USD} $/mois), "
            f"ou exclure le plus gros sous-dossier ci-dessus via .gitignore."
        )

    # Le vrai piège de LFS n'est pas le premier push mais l'accumulation :
    # chaque version d'un fichier est stockée ENTIÈREMENT et pour toujours.
    reecrits = sum(f.taille for f in fichiers if f.lfs and est_reecrit(f.relatif))
    if reecrits:
        print(
            f"  Chaque version compte : LFS garde une copie COMPLÈTE par commit. "
            f"Les fichiers réécrits à chaque run pèsent {_taille(reecrits).strip()}, "
            f"soit autant de quota consommé à chaque `make daily` qui les modifie."
        )
        runs = max(1, (QUOTA_LFS_GRATUIT - total_lfs) // reecrits) if total_lfs < QUOTA_LFS_GRATUIT else 0
        if runs:
            print(f"  À ce rythme, le palier gratuit tient environ {runs} run(s) après le premier push.")
    print("  Bande passante : chaque clone complet retélécharge tout le contenu LFS.")


# Horodatage AAAAMMJJ dans un nom de fichier ou de dossier de run
# (option_chains_20250115_120000.parquet, pipeline_runs/20250101_090000/...).
# On exige un millésime plausible plutôt que "contient un chiffre", sans quoi
# cache_8k_mistral.jsonl passerait pour un fichier daté à cause de son "8".
_HORODATAGE = re.compile(r"(?:19|20)\d{6}")

# Dossiers dont le contenu est AJOUTÉ run après run, jamais réécrit.
_DOSSIERS_DATES = ("/history/", "/backtest/", "/backtest_options/", "/pipeline_runs/")


def est_reecrit(relatif: str) -> bool:
    """Ce fichier sera-t-il ÉCRASÉ par les prochains runs (table consolidée),
    plutôt qu'AJOUTÉ à côté des précédents (snapshot ou résultat horodaté) ?

    La distinction décide du coût dans la durée : LFS garde une copie complète
    de CHAQUE version, donc un fichier réécrit repaie sa taille entière à
    chaque commit, là où un fichier horodaté ne se paie qu'une fois.

    Heuristique assumée, et qui ne sert qu'à l'avertissement de quota : un nom
    ou un dossier parent portant un horodatage plausible = ajouté ; tout le
    reste = réécrit. Elle penche volontairement vers "réécrit" dans le doute,
    pour prévenir large plutôt que de rassurer à tort."""
    if any(part in relatif for part in _DOSSIERS_DATES):
        return False
    return not _HORODATAGE.search(Path(relatif).name)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def indexer(racine: Path) -> int:
    """`git add` sur data/, puis compte les pointeurs LFS réellement posés."""
    r = _git("add", "--", str(racine))
    if r.returncode != 0:
        raise RuntimeError(f"git add a échoué : {r.stderr.strip()}")
    r = _git("lfs", "ls-files")
    return len([l for l in r.stdout.splitlines() if l])


def committer(message: str) -> None:
    r = _git("commit", "-m", message)
    if r.returncode != 0:
        raise RuntimeError(f"git commit a échoué : {r.stderr.strip() or r.stdout.strip()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir", type=Path, default=DOSSIER_DONNEES,
        help="Dossier de données à versionner (défaut: %(default)s).",
    )
    p.add_argument(
        "--apply", action="store_true",
        help="Active LFS pour ce dépôt et indexe le dossier. Sans cette "
             "option, le script ne fait que diagnostiquer et chiffrer.",
    )
    p.add_argument(
        "--commit", metavar="MESSAGE", default=None,
        help="Committe l'index avec ce message. Implique --apply.",
    )
    p.add_argument(
        "--verifier-pointeurs", action="store_true",
        help="Vérifie qu'aucun fichier n'est resté un pointeur LFS non "
             "rapatrié, et sort en erreur s'il en reste. À lancer après un "
             "clone ou un `git lfs pull` partiel : sans ça, pandas échouerait "
             "plus tard sur une erreur de parquet corrompu trompeuse.",
    )
    p.add_argument(
        "--ignorer", default="", metavar="MOTIFS",
        help="Motifs séparés par des virgules exemptés de --verifier-pointeurs "
             "(les mêmes que le --exclude du `git lfs pull`).",
    )
    args = p.parse_args(argv)
    # Sans ça, `--commit "..."` seul ne ferait rien du tout en silence : le
    # script s'arrêterait au diagnostic et l'utilisateur croirait avoir commité.
    args.apply = args.apply or args.commit is not None
    return args


def _verifier_pointeurs(racine: Path, ignorer: str) -> int:
    restants = pointeurs_restants(racine, ignorer.split(","))
    if not restants:
        print(f"Aucun pointeur LFS non rapatrié dans {racine}.")
        return 0

    print(
        f"\n{len(restants)} fichier(s) sont restés des POINTEURS LFS : leur "
        f"contenu n'a pas été rapatrié.\n"
        f"  Toute lecture échouera sur une erreur trompeuse du type "
        f"« Parquet magic bytes not found in footer ».\n"
        f"  Rapatrie-les : git lfs pull",
        file=sys.stderr,
    )
    for chemin in restants[:10]:
        print(f"    {chemin}", file=sys.stderr)
    if len(restants) > 10:
        print(f"    ... et {len(restants) - 10} autres", file=sys.stderr)
    return 1


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    version = git_lfs_installe()
    if version is None:
        print(
            "git-lfs n'est pas installé. C'est LUI qui remplace les gros "
            "fichiers par des pointeurs : sans lui, `git add data/` remettrait "
            "les parquet entiers dans l'historique.\n"
            "  Debian/Ubuntu : sudo apt install git-lfs\n"
            "  macOS         : brew install git-lfs\n"
            "  Windows       : inclus dans Git for Windows, sinon "
            "https://git-lfs.com",
            file=sys.stderr,
        )
        return 1
    print(f"git-lfs : {version}")

    if racine_depot() is None:
        print("Pas dans un dépôt git : lance ce script depuis la racine du dépôt.",
              file=sys.stderr)
        return 1

    if not Path(".gitattributes").exists():
        print(".gitattributes introuvable : les motifs LFS y sont définis, "
              "sans lui rien ne partira en LFS.", file=sys.stderr)
        return 1

    if not args.data_dir.exists():
        print(f"{args.data_dir} n'existe pas : lance d'abord le pipeline "
              f"(`make bootstrap`), il n'y a rien à versionner.", file=sys.stderr)
        return 1

    if args.verifier_pointeurs:
        return _verifier_pointeurs(args.data_dir, args.ignorer)

    fichiers = inventorier(args.data_dir)
    if not fichiers:
        print(f"{args.data_dir} est vide (ou entièrement ignoré par .gitignore).")
        return 0

    afficher_rapport(fichiers, args.data_dir)

    if not any(f.lfs for f in fichiers):
        print(
            "\nAucun fichier ne part en LFS : les motifs de .gitattributes ne "
            "correspondent à rien ici. Vérifie-les avant d'indexer, sinon les "
            "binaires entreront en dur dans l'historique git.",
            file=sys.stderr,
        )
        return 1

    if not args.apply:
        print("\nSimulation : rien n'a été modifié. Relance avec --apply pour "
              "activer LFS et indexer.")
        return 0

    # L'activation ne vient qu'ICI, après les vérifications : tant qu'on peut
    # encore refuser d'indexer, on ne touche pas au .git/config du dépôt.
    activer_lfs()
    print("\nFiltres LFS activés pour ce dépôt (.git/config).")

    print(f"Indexation de {args.data_dir}...")
    pointeurs = indexer(args.data_dir)
    print(f"  {pointeurs} fichiers suivis par LFS dans l'index.")

    if args.commit:
        committer(args.commit)
        print(f"  Commit créé : {args.commit}")
        print("\nPousse avec : git push -u origin <branche>")
    else:
        print("\nPrêt. `git commit` puis `git push` -- LFS téléversera le "
              "contenu au push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
