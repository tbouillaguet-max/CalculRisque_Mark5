"""Tests de la mise en place Git LFS (setup_lfs.py) et des motifs de
.gitattributes.

Les tests d'inventaire montent un VRAI dépôt git temporaire et y copient le
.gitattributes du dépôt : c'est git lui-même qui tranche quels fichiers
partent en LFS (`git check-attr`), donc un motif cassé se voit ici plutôt
qu'après un push. git-lfs n'a pas besoin d'être installé pour ça -- seules
l'activation et le comptage des pointeurs en dépendent, et ils ne sont pas
testés ici.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

import setup_lfs

RACINE = Path(__file__).resolve().parent.parent


@pytest.fixture
def depot(tmp_path, monkeypatch):
    """Dépôt git jetable, avec le .gitattributes et le .gitignore réels.

    git-lfs est simulé comme présent : les tests qui passent par main() se
    jouent des motifs et des garde-fous, pas de l'installation locale (que
    couvre test_git_lfs_absent_est_explique). Sans ça, la suite échouerait sur
    toute machine ou CI où git-lfs n'est pas installé."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for nom in (".gitattributes", ".gitignore"):
        shutil.copy(RACINE / nom, tmp_path / nom)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup_lfs, "git_lfs_installe", lambda: "git-lfs/3.4.1 (simulé)")
    return tmp_path


def ecrire(racine: Path, relatif: str, octets: int = 128) -> Path:
    chemin = racine / relatif
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_bytes(b"x" * octets)
    return chemin


# ---------------------------------------------------------------------------
# Motifs de .gitattributes : qui part en LFS, qui reste en git brut
# ---------------------------------------------------------------------------

EN_LFS = [
    "data/prices/daily_prices.parquet",
    "data/prices/year_end_prices.parquet",
    "data/financials/financials_ttm.parquet",
    "data/options/option_chains.parquet",
    "data/options/history/option_chains_20250115_120000.parquet",
    "data/dcf/resultats_dcf.xlsx",
    "data/multiples/multiples_moyens_par_secteur.xlsx",
    "data/financials/cache_8k_mistral.jsonl",
    "data/financials/checkpoint_8k.jsonl",
    "data/financials/sec_submissions/CIK0000320193.json",
    "data/backtest_options/20250101_120000/trades.parquet",
]

EN_GIT_DIRECT = [
    "data/universe/sp500_universe.csv",
    "data/universe/sp500_universe_full.csv",
    "data/prices/fetch_state_prices.json",
    "data/financials/progress_10q.json",
    "data/dcf/progress_qualitative.json",
    "data/pipeline_runs/20250101_090000/report.json",
    "data/prices/daily_prices_20250101_120000.log",
]


def test_les_fichiers_lourds_partent_en_lfs(depot):
    for relatif in EN_LFS + EN_GIT_DIRECT:
        ecrire(depot, relatif)

    suivis = setup_lfs.chemins_suivis_par_lfs(EN_LFS + EN_GIT_DIRECT)

    assert set(EN_LFS) <= suivis, f"manquent en LFS : {set(EN_LFS) - suivis}"


def test_les_petits_fichiers_texte_restent_en_git_direct(depot):
    """Les mettre en LFS consommerait du quota et ferait perdre les diffs,
    pour des fichiers que git compresse très bien tout seul."""
    for relatif in EN_LFS + EN_GIT_DIRECT:
        ecrire(depot, relatif)

    suivis = setup_lfs.chemins_suivis_par_lfs(EN_LFS + EN_GIT_DIRECT)

    assert not (set(EN_GIT_DIRECT) & suivis), \
        f"ne devraient pas être en LFS : {set(EN_GIT_DIRECT) & suivis}"


def test_les_motifs_lfs_ne_debordent_pas_hors_de_data(depot):
    """Les fixtures de tests et le cache de secteurs à la racine sont déjà
    versionnés en clair : les basculer en LFS casserait les clones sans
    git-lfs et polluerait le quota."""
    hors_data = [
        "secteur_cache.json",
        "tests/fixtures/companyfacts_AAPL.json",
        "report/requirements.txt",
    ]
    for relatif in hors_data:
        ecrire(depot, relatif)

    assert setup_lfs.chemins_suivis_par_lfs(hors_data) == set()


def test_un_chemin_ignore_n_est_pas_compte(depot):
    """.gitignore a le dernier mot : un dossier exclu ne doit apparaître ni
    dans le volume LFS ni dans le volume git du rapport."""
    (depot / ".gitignore").open("a", encoding="utf-8").write("\ndata/options/history/\n")
    ecrire(depot, "data/options/history/option_chains_20250115_120000.parquet", 4096)
    ecrire(depot, "data/options/option_chains.parquet", 2048)

    fichiers = setup_lfs.inventorier(Path("data"))

    assert [f.relatif for f in fichiers] == ["data/options/option_chains.parquet"]


# ---------------------------------------------------------------------------
# Inventaire et agrégation
# ---------------------------------------------------------------------------

def test_l_inventaire_classe_et_pese_chaque_fichier(depot):
    ecrire(depot, "data/prices/daily_prices.parquet", 5000)
    ecrire(depot, "data/universe/sp500_universe.csv", 300)

    fichiers = {f.relatif: f for f in setup_lfs.inventorier(Path("data"))}

    assert fichiers["data/prices/daily_prices.parquet"].lfs is True
    assert fichiers["data/prices/daily_prices.parquet"].taille == 5000
    assert fichiers["data/universe/sp500_universe.csv"].lfs is False
    assert fichiers["data/universe/sp500_universe.csv"].taille == 300


def test_le_regroupement_separe_lfs_et_git_par_sous_dossier(depot):
    ecrire(depot, "data/prices/daily_prices.parquet", 1000)
    ecrire(depot, "data/prices/year_end_prices.parquet", 500)
    ecrire(depot, "data/prices/fetch_state_prices.json", 40)
    ecrire(depot, "data/universe/sp500_universe.csv", 300)

    groupes = setup_lfs.grouper_par_dossier(setup_lfs.inventorier(Path("data")))

    assert groupes["data/prices"] == (3, 1500, 40)
    assert groupes["data/universe"] == (1, 0, 300)


# ---------------------------------------------------------------------------
# Heuristique de coût dans la durée
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("relatif", [
    "data/prices/daily_prices.parquet",
    "data/financials/financials_ttm.parquet",
    "data/options/option_chains.parquet",
    "data/dcf/resultats_dcf.xlsx",
    # Le "8" de 8-K ne fait pas un horodatage : ce fichier grossit et est
    # réécrit à chaque run, il repaie donc sa taille à chaque commit.
    "data/financials/cache_8k_mistral.jsonl",
])
def test_les_tables_consolidees_sont_reecrites(relatif):
    assert setup_lfs.est_reecrit(relatif) is True


@pytest.mark.parametrize("relatif", [
    "data/options/history/option_chains_20250115_120000.parquet",
    "data/options/history/option_chains_avhist_20240630.parquet",
    "data/backtest_options/20250101_120000/trades.parquet",
    "data/backtest/20250101_120000/metrics.json",
    "data/pipeline_runs/20250101_090000/report.json",
])
def test_les_fichiers_horodates_ne_sont_payes_qu_une_fois(relatif):
    assert setup_lfs.est_reecrit(relatif) is False


# ---------------------------------------------------------------------------
# Garde-fous du script
# ---------------------------------------------------------------------------

def test_le_script_refuse_d_indexer_si_aucun_motif_ne_prend(depot, capsys):
    """Un .gitattributes vide ou cassé ferait entrer les parquet EN DUR dans
    l'historique git -- exactement ce que LFS doit éviter, et irréversible
    sans réécrire l'historique. Le script s'arrête plutôt que d'indexer."""
    (depot / ".gitattributes").write_text("# plus aucun motif\n", encoding="utf-8")
    ecrire(depot, "data/prices/daily_prices.parquet", 4096)

    code = setup_lfs.main(["--apply"])

    assert code == 1
    assert "Aucun fichier ne part en LFS" in capsys.readouterr().err


def test_sans_apply_le_script_ne_touche_ni_a_l_index_ni_a_la_config(depot):
    ecrire(depot, "data/prices/daily_prices.parquet", 4096)

    assert setup_lfs.main([]) == 0

    indexes = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert indexes == ""
    assert "filter.lfs" not in (depot / ".git" / "config").read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("git-lfs") is None, reason="git-lfs non installé")
def test_l_apply_active_lfs_et_pose_les_pointeurs(depot):
    """Le chemin complet, sur un vrai dépôt : après --apply, le parquet est un
    pointeur dans l'index et le CSV garde son contenu. C'est le test qui
    vérifie que tout le montage tient réellement, pas seulement les motifs."""
    ecrire(depot, "data/prices/daily_prices.parquet", 4096)
    ecrire(depot, "data/universe/sp500_universe.csv", 300)

    assert setup_lfs.main(["--apply"]) == 0

    montre = subprocess.run(
        ["git", "show", ":data/prices/daily_prices.parquet"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert montre.startswith("version https://git-lfs.github.com/spec/v1")

    csv = subprocess.run(
        ["git", "show", ":data/universe/sp500_universe.csv"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert len(csv) == 300


def test_git_lfs_absent_est_explique(depot, monkeypatch, capsys):
    """Sans git-lfs, `git add data/` remettrait les parquet entiers dans
    l'historique : le script s'arrête et dit comment l'installer."""
    monkeypatch.setattr(setup_lfs, "git_lfs_installe", lambda: None)
    ecrire(depot, "data/prices/daily_prices.parquet", 4096)

    assert setup_lfs.main([]) == 1
    assert "apt install git-lfs" in capsys.readouterr().err


def test_un_dossier_de_donnees_absent_est_signale(depot, capsys):
    assert setup_lfs.main([]) == 1
    assert "n'existe pas" in capsys.readouterr().err


def test_les_tailles_sont_lisibles():
    assert setup_lfs._taille(500 * setup_lfs.MO).strip() == "500.0 Mo"
    assert setup_lfs._taille(2 * setup_lfs.GO).strip() == "2.00 Go"
