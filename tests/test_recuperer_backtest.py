"""Tests du rapatriement des backtests lancés sur GitHub Actions
(recuperer_backtest.py) et de la cohérence du workflow qui les produit.

Rien ici ne touche au réseau : les appels API sont remplacés, et l'extraction
travaille sur des archives construites en mémoire.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

import recuperer_backtest as rb

RACINE = Path(__file__).resolve().parent.parent
WORKFLOW = RACINE / ".github" / "workflows" / "backtest.yml"


def archive(membres: dict[str, bytes]) -> bytes:
    tampon = io.BytesIO()
    with zipfile.ZipFile(tampon, "w") as zf:
        for nom, contenu in membres.items():
            zf.writestr(nom, contenu)
    return tampon.getvalue()


# ---------------------------------------------------------------------------
# Identification du dépôt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://github.com/tbouillaguet-max/CalculRisque_Mark5.git",
    "https://github.com/tbouillaguet-max/CalculRisque_Mark5",
    "git@github.com:tbouillaguet-max/CalculRisque_Mark5.git",
    "  https://github.com/tbouillaguet-max/CalculRisque_Mark5.git\n",
])
def test_le_depot_est_lu_depuis_le_remote(url):
    assert rb.depot_depuis_remote(url) == ("tbouillaguet-max", "CalculRisque_Mark5")


def test_un_remote_incomprehensible_ne_fait_pas_deviner():
    assert rb.depot_depuis_remote("") is None


# ---------------------------------------------------------------------------
# Choix de l'artefact
# ---------------------------------------------------------------------------

def artefact(nom: str, cree: str, run: int, expire: bool = False) -> dict:
    return {
        "id": run, "name": nom, "created_at": cree, "expired": expire,
        "size_in_bytes": 1024, "workflow_run": {"id": run},
    }


def test_le_plus_recent_backtest_est_choisi():
    liste = [
        artefact("backtest-valuation_gap_options-1", "2026-09-01T10:00:00Z", 1),
        artefact("backtest-valuation_gap_options-3", "2026-09-12T10:00:00Z", 3),
        artefact("backtest-valuation_gap_options-2", "2026-09-05T10:00:00Z", 2),
    ]
    assert [a["id"] for a in rb.artefacts_de_backtest(liste)] == [3, 2, 1]


def test_les_artefacts_expires_sont_ecartes():
    """GitHub les laisse dans la liste après leur rétention, mais leur
    téléchargement renvoie 410 : en choisir un donnerait une erreur
    incompréhensible."""
    liste = [
        artefact("backtest-a-2", "2026-09-12T10:00:00Z", 2, expire=True),
        artefact("backtest-a-1", "2026-09-01T10:00:00Z", 1),
    ]
    assert [a["id"] for a in rb.artefacts_de_backtest(liste)] == [1]


def test_les_autres_artefacts_du_depot_sont_ignores():
    liste = [
        artefact("couverture-tests", "2026-09-12T10:00:00Z", 2),
        artefact("backtest-a-1", "2026-09-01T10:00:00Z", 1),
    ]
    assert [a["id"] for a in rb.artefacts_de_backtest(liste)] == [1]


def test_un_run_precis_peut_etre_demande():
    liste = [
        artefact("backtest-a-2", "2026-09-12T10:00:00Z", 2),
        artefact("backtest-a-1", "2026-09-01T10:00:00Z", 1),
    ]
    assert [a["id"] for a in rb.artefacts_de_backtest(liste, run_id=1)] == [1]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def test_les_resultats_atterrissent_la_ou_le_pipeline_les_lit(tmp_path):
    """C'est tout l'enjeu : dans data/backtest_options/<run_id>/, pour que
    `14_audit_backtest.py` et le dashboard les lisent sans rien changer."""
    zip_ = archive({
        "backtest_options/20260912_101500/trades.parquet": b"PAR1",
        "backtest_options/20260912_101500/metrics.json": b"{}",
        "backtest/20260912_101500/equity_curve.parquet": b"PAR1",
        "run_distant.json": json.dumps({"strategie": "valuation_gap_options"}).encode(),
    })

    compteurs = rb.extraire(zip_, racine=tmp_path)

    assert (tmp_path / "data/backtest_options/20260912_101500/trades.parquet").read_bytes() == b"PAR1"
    assert (tmp_path / "data/backtest_options/20260912_101500/metrics.json").exists()
    assert (tmp_path / "data/backtest/20260912_101500/equity_curve.parquet").exists()
    assert compteurs["ecrits"] == 3
    assert compteurs["meta"]["strategie"] == "valuation_gap_options"


def test_un_fichier_deja_present_est_garde_par_defaut(tmp_path):
    cible = tmp_path / "data/backtest_options/20260912_101500/trades.parquet"
    cible.parent.mkdir(parents=True)
    cible.write_bytes(b"resultat local")

    compteurs = rb.extraire(
        archive({"backtest_options/20260912_101500/trades.parquet": b"distant"}),
        racine=tmp_path,
    )

    assert cible.read_bytes() == b"resultat local"
    assert compteurs["ignores"] == 1
    assert compteurs["ecrits"] == 0


def test_ecraser_remplace_bien(tmp_path):
    cible = tmp_path / "data/backtest_options/20260912_101500/trades.parquet"
    cible.parent.mkdir(parents=True)
    cible.write_bytes(b"resultat local")

    rb.extraire(
        archive({"backtest_options/20260912_101500/trades.parquet": b"distant"}),
        racine=tmp_path, ecraser=True,
    )

    assert cible.read_bytes() == b"distant"


@pytest.mark.parametrize("nom", [
    "../../.ssh/authorized_keys",
    "backtest_options/../../../etc/passwd",
    "/etc/passwd",
    "backtest_options/../../config.py",
    # Dossier voisin au nom plus long : une comparaison de préfixes de chaînes
    # l'aurait accepté, is_relative_to non.
    "backtest_options/../backtest_options_evil/vole.txt",
])
def test_aucune_ecriture_hors_des_dossiers_de_resultats(nom, tmp_path):
    """Le nom d'un membre de zip est une donnée : un "../" y écrirait hors du
    dépôt. Seuls les préfixes attendus sont acceptés."""
    assert rb.destination_membre(nom, tmp_path) is None


def test_les_membres_inattendus_sont_comptes_sans_etre_ecrits(tmp_path):
    compteurs = rb.extraire(
        archive({"n_importe_quoi.txt": b"x", "backtest_options/r/t.parquet": b"y"}),
        racine=tmp_path,
    )

    assert compteurs["ecrits"] == 1
    assert compteurs["hors_perimetre"] == 1
    assert not (tmp_path / "n_importe_quoi.txt").exists()


def test_un_run_distant_illisible_n_empeche_pas_le_depot(tmp_path):
    """Les résultats valent plus que leur fiche descriptive : un JSON tronqué
    ne doit pas faire échouer le rapatriement."""
    compteurs = rb.extraire(
        archive({
            "run_distant.json": b"{ pas du json",
            "backtest_options/r/trades.parquet": b"PAR1",
        }),
        racine=tmp_path,
    )

    assert compteurs["meta"] is None
    assert compteurs["ecrits"] == 1


# ---------------------------------------------------------------------------
# Cohérence avec le workflow
# ---------------------------------------------------------------------------

def test_le_workflow_est_un_yaml_valide():
    yaml = pytest.importorskip("yaml")
    contenu = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # "on:" est interprété comme le booléen True par YAML 1.1, que PyYAML suit.
    assert "workflow_dispatch" in contenu[True]
    assert "backtest" in contenu["jobs"]


def test_les_strategies_du_workflow_existent_vraiment():
    """Une stratégie renommée dans le code sans l'être ici donnerait un run qui
    part, tourne, et échoue sur un nom inconnu."""
    yaml = pytest.importorskip("yaml")
    from backtest.strategies import OPTIONS_STRATEGY_REGISTRY, STRATEGY_REGISTRY

    contenu = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    proposees = contenu[True]["workflow_dispatch"]["inputs"]["strategie"]["options"]
    connues = set(OPTIONS_STRATEGY_REGISTRY) | set(STRATEGY_REGISTRY)

    assert set(proposees) <= connues, f"inconnues : {set(proposees) - connues}"


def test_le_workflow_publie_ce_que_le_script_attend():
    """Le dossier de rassemblement du workflow et les préfixes attendus à
    l'extraction doivent rester d'accord."""
    texte = WORKFLOW.read_text(encoding="utf-8")
    for prefixe in rb.DESTINATIONS:
        assert f"artefact/{prefixe}" in texte, f"{prefixe} absent du workflow"
    assert f"artefact/{rb.FICHIER_META}" in texte
    assert f"name: {rb.PREFIXE_ARTEFACT}" in texte
