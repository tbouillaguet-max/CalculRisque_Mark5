"""Orchestrateur quotidien : composition des étapes et mode dégradé.

Ce qu'on teste ici n'est PAS l'exécution des scripts (chacun a ses propres
tests) mais les deux décisions propres à l'orchestrateur, celles qui décident
si le signal du jour existe ou non :

  1. quelles étapes tournent dans quel mode (--prices-only, --skip-options) ;
  2. ce qui arrive quand IB Gateway ne répond pas -- sauter la récupération des
     cours laisserait le signal figé sur la veille, alors qu'une source de
     repli existe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import run_pipeline_daily as daily
from run_pipeline_quarterly import RunReport, resolve_gateway_args


@pytest.fixture
def rapport(tmp_path) -> RunReport:
    return RunReport(run_id="test", mode="daily", directory=tmp_path / "run")


# --------------------------------------------------------------------------- #
# Composition des étapes
# --------------------------------------------------------------------------- #

def test_les_etapes_de_calcul_du_signal_sont_requises():
    """05/06/06b/07 transforment les données brutes en écart de valorisation :
    si l'une échoue, le signal lu ensuite est incohérent avec les cours du
    jour -- ce n'est pas une dégradation acceptable, c'est un arrêt."""
    requises = {s.script for s in daily.daily_steps(7) if s.required}
    assert {"05_calcul_multiples.py", "06_calcul_multiples_moyens.py",
            "06b_calcul_valorisation_combinee.py", "07_calcul_dcf.py"} <= requises
    # Les cours du jour aussi : c'est la seule étape qui apporte réellement de
    # l'information nouvelle un jour ordinaire.
    assert "03b_recuperation_cours_quotidiens.py" in requises


def test_les_enrichissements_ne_bloquent_pas_le_run():
    """Une clé LLM expirée ou un Gateway fermé ne doit pas faire perdre une
    valorisation quotidienne par ailleurs correcte."""
    optionnelles = {s.script for s in daily.daily_steps(7) if not s.required}
    assert {"04_recuperation_10k.py", "04b_recuperation_10q.py", "04c_recuperation_8k.py",
            "07b_validation_qualitative.py", "08_recuperation_options.py"} <= optionnelles


def test_la_fenetre_de_fraicheur_est_transmise_aux_depots_sec():
    """Sans --refresh-days, 04/04b garderaient leur défaut de 30 jours : un
    10-K déposé aujourd'hui n'entrerait dans le signal qu'un mois plus tard."""
    par_script = {s.script: s.extra_args for s in daily.daily_steps(7)}
    assert par_script["04_recuperation_10k.py"] == ("--refresh-days", "7")
    assert par_script["04b_recuperation_10q.py"] == ("--refresh-days", "7")
    # Les étapes de calcul local n'ont rien à rafraîchir.
    assert par_script["05_calcul_multiples.py"] == ()


def test_prices_only_garde_de_quoi_recalculer_le_signal():
    """Le mode court doit rester UTILE : rafraîchir les cours sans recalculer
    l'écart ne mettrait aucun signal à jour."""
    assert "03b_recuperation_cours_quotidiens.py" in daily.PRICES_ONLY
    assert "06b_calcul_valorisation_combinee.py" in daily.PRICES_ONLY
    # ... et rester COURT : aucun appel SEC ni LLM.
    for reseau in ("04_recuperation_10k.py", "04b_recuperation_10q.py",
                   "04c_recuperation_8k.py", "07b_validation_qualitative.py"):
        assert reseau not in daily.PRICES_ONLY


# --------------------------------------------------------------------------- #
# Mode dégradé (IB Gateway indisponible)
# --------------------------------------------------------------------------- #

def test_les_cours_se_replient_sur_stooq_au_lieu_d_etre_sautes(monkeypatch, rapport):
    """LE point du mode dégradé. 03b sait travailler sans IBKR (--skip-ibkr,
    source Stooq) : la sauter parce que le Gateway est fermé laisserait le
    signal du jour calculé sur les cours de la veille, silencieusement."""
    monkeypatch.setattr("run_pipeline_quarterly.ensure_gateway_available", lambda: False)

    cours = next(s for s in daily.daily_steps(7)
                 if s.script == "03b_recuperation_cours_quotidiens.py")
    assert resolve_gateway_args(cours, rapport) == ["--skip-ibkr"]
    assert rapport.steps == []  # rien n'a été journalisé comme sauté


def test_une_etape_sans_repli_est_sautee(monkeypatch, rapport):
    """08 n'a aucune source alternative aux chaînes d'options d'IBKR : sans
    Gateway, il n'y a rien à collecter et l'étape est sautée, pas dégradée."""
    monkeypatch.setattr("run_pipeline_quarterly.ensure_gateway_available", lambda: False)

    options = next(s for s in daily.daily_steps(7)
                   if s.script == "08_recuperation_options.py")
    assert resolve_gateway_args(options, rapport) is None
    assert rapport.steps[0]["status"] == "skipped"


def test_gateway_disponible_n_ajoute_aucun_argument(monkeypatch, rapport):
    monkeypatch.setattr("run_pipeline_quarterly.ensure_gateway_available", lambda: True)
    for step in daily.daily_steps(7):
        assert resolve_gateway_args(step, rapport) == []


def test_une_etape_sans_gateway_n_est_jamais_interrogee(monkeypatch, rapport):
    """Tester le port pour 05/06/07 coûterait un timeout socket par étape et
    par run, pour une réponse dont aucune ne dépend."""
    def refuse() -> bool:
        raise AssertionError("le port ne doit pas être testé pour une étape locale")

    monkeypatch.setattr("run_pipeline_quarterly.ensure_gateway_available", refuse)
    locale = next(s for s in daily.daily_steps(7) if s.script == "07_calcul_dcf.py")
    assert resolve_gateway_args(locale, rapport) == []


# --------------------------------------------------------------------------- #
# Journal
# --------------------------------------------------------------------------- #

def test_le_run_quotidien_a_son_propre_mode(tmp_path, monkeypatch):
    """--resume ne doit jamais reprendre un run TRIMESTRIEL interrompu : les
    deux n'ont pas la même liste d'étapes, donc « déjà réussie » n'y désigne
    pas le même travail."""
    monkeypatch.setattr("config.DIR_PIPELINE_RUNS", tmp_path)

    trimestriel = RunReport(run_id="20200101_000000", mode="live", directory=tmp_path / "20200101_000000")
    trimestriel.status = "failed"
    trimestriel.steps = [{"script": "05_calcul_multiples.py", "status": "success"}]
    trimestriel.save()

    from run_pipeline_quarterly import find_interrupted_report
    assert find_interrupted_report("daily", "20260101_000000") is None
    assert find_interrupted_report("live", "20260101_000000") is not None


def test_les_scripts_declares_existent():
    """Un nom de script mal orthographié ne se verrait qu'au prochain cron,
    dans un log, après l'échec de l'étape -- et pour une étape optionnelle, pas
    du tout : le run finirait "partial" sans que personne ne sache pourquoi."""
    racine = Path(__file__).resolve().parent.parent
    for step in daily.daily_steps(7):
        assert (racine / step.script).exists(), f"{step.script} déclaré mais introuvable"


def test_l_ordre_respecte_les_dependances_de_donnees():
    """Les étapes s'enchaînent en sous-processus : rien ne rattrape un ordre
    faux, chacune lirait simplement le fichier de la veille sans le dire.
    06b a besoin des multiples (05/06) ET du DCF (07) pour choisir entre eux."""
    ordre = [s.script for s in daily.daily_steps(7)]
    rang = {script: i for i, script in enumerate(ordre)}

    assert rang["03b_recuperation_cours_quotidiens.py"] < rang["05_calcul_multiples.py"]
    assert rang["04b_recuperation_10q.py"] < rang["05_calcul_multiples.py"]
    assert rang["05_calcul_multiples.py"] < rang["06_calcul_multiples_moyens.py"]
    assert rang["06_calcul_multiples_moyens.py"] < rang["06b_calcul_valorisation_combinee.py"]
    assert rang["07_calcul_dcf.py"] < rang["07b_validation_qualitative.py"]
    # 08 filtre les entreprises à interroger sur l'écart de valorisation : il
    # lui faut le signal du jour, donc il passe après 06b/07.
    assert rang["06b_calcul_valorisation_combinee.py"] < rang["08_recuperation_options.py"]


def test_le_timeout_quotidien_est_plus_court_que_le_trimestriel():
    """Un run quotidien qui déborde sur la séance suivante n'a plus d'objet :
    le cron du lendemain le remplacera."""
    from run_pipeline_quarterly import STEP_TIMEOUT_DEFAULT as trimestriel
    assert daily.STEP_TIMEOUT_DEFAULT < trimestriel
