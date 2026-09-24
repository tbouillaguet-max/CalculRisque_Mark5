"""Un run qui tourne sans dépôts SEC frais doit le DIRE.

CE QUI S'EST PASSÉ. Run quotidien du 2026-09-05 (data/pipeline_runs/20260905_120704) :
SEC_CONTACT_EMAIL n'était pas définie. 04, 04b, 04c et 07b sont partis, ont
échoué, ont été relancés trois fois chacun à 30 s d'intervalle -- six minutes
perdues à réessayer une erreur de CONFIGURATION --, puis le run a fini
« partial ». Et 06b a recalculé le signal du jour sur les comptes de la veille,
sans que rien d'autre ne le signale : le tableau de bord le marquait même « à
jour », puisqu'il juge un fichier à sa date de modification.

LE TEST QUI COMPTE est `test_le_scenario_du_5_septembre_est_desormais_signale`.

ET UN PIÈGE QUE LE CORRECTIF OUVRAIT : une étape SEC désormais SAUTÉE n'est plus
« failed ». Sans règle supplémentaire, le même run aurait fini « success » --
pire qu'avant. `test_un_signal_sur_comptes_perimes_n_est_jamais_success` le ferme.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import run_pipeline_daily as daily
import run_pipeline_quarterly as quarterly

RACINE = pathlib.Path(__file__).resolve().parent.parent
SEC = {"04_recuperation_10k.py", "04b_recuperation_10q.py",
       "04c_recuperation_8k.py", "07b_validation_qualitative.py"}


@pytest.fixture
def sans_sec(monkeypatch):
    monkeypatch.delenv("SEC_CONTACT_EMAIL", raising=False)


@pytest.fixture
def avec_sec(monkeypatch):
    monkeypatch.setenv("SEC_CONTACT_EMAIL", "contact@exemple.fr")


@pytest.fixture
def lances(monkeypatch):
    """Remplace l'exécution réelle : chaque étape lancée « réussit » et est
    notée. Le Gateway répond, pour isoler la seule variable qui nous occupe."""
    journal: list[str] = []

    def faux(step, args, report, retries, timeout, cwd=None):
        journal.append(step.script)
        report.record({"script": step.script, "status": "success", "attempts": 1})
        return True

    monkeypatch.setattr(daily, "run_step", faux)
    monkeypatch.setattr(quarterly, "run_step", faux)
    monkeypatch.setattr(quarterly, "ensure_gateway_available", lambda: True)
    return journal


def _rapport(tmp_path, mode="daily"):
    return quarterly.RunReport(run_id="test", mode=mode, directory=tmp_path / "run")


def _run_quotidien(rapport, prices_only=False, deja=frozenset()):
    daily.run_daily(rapport, None, False, prices_only, 0, 60, set(deja), 7)
    avertissement = quarterly.avertissement_depots_sec(rapport, daily.daily_steps(7))
    if avertissement:
        rapport.avertissements.append(avertissement)


# --------------------------------------------------------------------------- #
# LE scénario
# --------------------------------------------------------------------------- #
def test_le_scenario_du_5_septembre_est_desormais_signale(tmp_path, sans_sec, lances):
    rapport = _rapport(tmp_path)
    _run_quotidien(rapport)

    # 1. Les étapes SEC ne sont plus tentées : zéro essai, zéro minute perdue.
    sautees = {s["script"]: s for s in rapport.steps if s["status"] == "skipped"}
    assert set(sautees) == SEC
    assert all(s["attempts"] == 0 for s in sautees.values())
    assert not SEC & set(lances)

    # 2. Le reste tourne : la valorisation du jour n'est pas perdue pour autant.
    assert "03b_recuperation_cours_quotidiens.py" in lances
    assert "06b_calcul_valorisation_combinee.py" in lances

    # 3. Et c'est DIT, en nommant ce qui manque plutôt que des numéros d'étape.
    assert len(rapport.avertissements) == 1
    texte = rapport.avertissements[0]
    for apport in ("comptes annuels", "comptes trimestriels", "8-K", "validation qualitative"):
        assert apport in texte, apport
    assert "SEC_CONTACT_EMAIL" in texte


def test_un_signal_sur_comptes_perimes_n_est_jamais_success(tmp_path, sans_sec, lances):
    """LE PIÈGE DU CORRECTIF. Sautée, une étape SEC n'est plus « failed » ; la
    règle d'avant (« partial » si une étape a échoué) laisserait donc passer ce
    run en « success ». C'est main() qui tranche -- on rejoue sa règle."""
    rapport = _rapport(tmp_path)
    _run_quotidien(rapport)
    echecs = [s for s in rapport.steps if s["status"] == "failed"]
    assert not echecs, "le scénario n'a plus d'échec : c'est précisément le piège"

    import inspect
    source = inspect.getsource(daily.main)
    assert 'report.status = "partial" if (failed_optional or report.avertissements)' in source
    assert 'report.status = "partial" if (failed_optional or report.avertissements)' in \
        inspect.getsource(quarterly.main)


def test_avec_l_adresse_rien_ne_change(tmp_path, avec_sec, lances):
    rapport = _rapport(tmp_path)
    _run_quotidien(rapport)
    assert SEC <= set(lances)
    assert rapport.avertissements == []


def test_prices_only_saute_la_sec_par_choix_et_ne_s_en_plaint_pas(tmp_path, sans_sec, lances):
    """--prices-only promet « aucun appel SEC » : ce n'est pas une panne, et
    l'annoncer comme telle chaque matin apprendrait à ignorer l'avertissement."""
    rapport = _rapport(tmp_path)
    _run_quotidien(rapport, prices_only=True)
    assert rapport.avertissements == []


def test_une_etape_sec_deja_reussie_a_la_reprise_ne_compte_pas(tmp_path, sans_sec, lances):
    """--resume : ce qui a réussi plus tôt a bien rafraîchi les dépôts."""
    rapport = _rapport(tmp_path)
    _run_quotidien(rapport, deja=SEC)
    assert rapport.avertissements == []


def test_un_echec_reel_d_une_etape_sec_est_signale_aussi(tmp_path, avec_sec, monkeypatch):
    """L'adresse peut être là et la SEC indisponible : le signal n'en est pas
    moins périmé. La cause change, l'avertissement demeure."""
    monkeypatch.setattr(quarterly, "ensure_gateway_available", lambda: True)

    def faux(step, args, report, retries, timeout, cwd=None):
        statut = "failed" if step.script == "04b_recuperation_10q.py" else "success"
        report.record({"script": step.script, "status": statut, "attempts": 1})
        return statut == "success"

    monkeypatch.setattr(daily, "run_step", faux)
    rapport = _rapport(tmp_path)
    _run_quotidien(rapport)

    assert len(rapport.avertissements) == 1
    assert "comptes trimestriels" in rapport.avertissements[0]
    assert "comptes annuels" not in rapport.avertissements[0]
    assert "échec de l'étape" in rapport.avertissements[0]


# --------------------------------------------------------------------------- #
# Run trimestriel : la SEC y est la raison d'être
# --------------------------------------------------------------------------- #
def test_le_run_trimestriel_s_arrete_au_lieu_de_calculer_sur_des_comptes_perimes(
        tmp_path, sans_sec, lances):
    """04b y est REQUISE : c'est la récupération des comptes que ce run a pour
    objet. Recalculer tout le reste sans elle, ce serait produire le signal
    trimestriel sur les comptes du trimestre précédent. Il s'arrête, et le
    message dit quoi faire."""
    rapport = _rapport(tmp_path, mode="live")
    with pytest.raises(RuntimeError, match="SEC_CONTACT_EMAIL"):
        quarterly.run_live(rapport, None, False, 0, 60, set())
    assert lances == []


# --------------------------------------------------------------------------- #
# La vérification elle-même
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("valeur,attendu", [
    ("contact@exemple.fr", True),
    ("   ", False),          # présente mais vide : sec_http la refuse, l'orchestrateur aussi
    ("", False),
])
def test_l_orchestrateur_et_sec_http_sont_d_accord(monkeypatch, valeur, attendu):
    """La règle n'existe qu'à un endroit (sec_http.contact_email) : un
    orchestrateur qui jugerait « configurée » une adresse que les scripts
    refusent lancerait les étapes pour les voir échouer."""
    monkeypatch.setenv("SEC_CONTACT_EMAIL", valeur)
    assert quarterly.sec_contact_configure() is attendu


def test_toute_etape_qui_interroge_la_sec_est_declaree_comme_telle():
    """LE GARDE-FOU DURABLE. Une étape qui importe sec_http (directement ou via
    sec_filings_text) sans être marquée needs_sec retomberait exactement dans
    le scénario du 5 septembre. On le vérifie depuis le CODE, pas depuis une
    liste recopiée qui vieillirait avec lui."""
    def interroge_la_sec(script: str) -> bool:
        texte = (RACINE / script).read_text(encoding="utf-8")
        return bool(re.search(r"^\s*(import|from)\s+(sec_http|sec_filings_text)\b", texte, re.M))

    for etapes, nom in ((daily.daily_steps(7), "quotidien"), (quarterly.LIVE_STEPS, "trimestriel")):
        for step in etapes:
            if interroge_la_sec(step.script):
                assert step.needs_sec, f"run {nom} : {step.script} interroge la SEC sans needs_sec"


def test_l_avertissement_est_ecrit_dans_le_rapport(tmp_path, sans_sec, lances):
    """C'est le rapport que lit le tableau de bord : un avertissement qui ne
    vivrait que dans le journal texte ne serait vu par personne."""
    import json
    rapport = _rapport(tmp_path)
    _run_quotidien(rapport)
    rapport.save()
    lu = json.loads((tmp_path / "run" / "report.json").read_text(encoding="utf-8"))
    assert lu["avertissements"] and "SEC" in lu["avertissements"][0]


def test_le_tableau_de_bord_montre_celui_du_dernier_run_seulement():
    """Un avertissement d'hier est caduc dès qu'un run plus récent a rafraîchi
    les dépôts -- le montrer encore apprendrait à l'ignorer."""
    import sys
    sys.path.insert(0, str(RACINE / "report"))
    import utils
    runs = [{"run_id": "20260906", "avertissements": []},
            {"run_id": "20260905", "avertissements": ["périmé"]}]
    assert utils.avertissements_du_dernier_run(runs) == ("20260906", [])
    assert utils.avertissements_du_dernier_run(runs[1:]) == ("20260905", ["périmé"])
    assert utils.avertissements_du_dernier_run([]) == (None, [])
