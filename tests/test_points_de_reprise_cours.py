"""03b : ne pas perdre des heures de collecte à la moindre interruption.

CE QUI A ÉTÉ CONSTATÉ EN RUN RÉEL. 03b s'impose 11 s entre deux requêtes
historiques (pacing IBKR, ~55 requêtes / 10 min) : un univers de 504 tickers
demande donc ~94 minutes AU MIEUX. Le parquet n'étant écrit qu'à la toute fin,
une interruption à 60 minutes -- le délai que l'orchestrateur imposait --
jetait les 322 tickers déjà récupérés, et la tentative suivante repartait du
PREMIER ticker parce que le cache n'avait pas bougé. Trois tentatives, trois
heures, aucun progrès.

Deux défauts distincts, donc deux séries de tests :

    A  le délai de l'orchestrateur doit tenir compte de la cadence IMPOSÉE des
       étapes qui interrogent IBKR ticker par ticker ;
    B  une collecte longue doit poser des points de reprise, pour qu'une
       interruption ne coûte que quelques tickers -- et pour que la relance
       saute ce qui est déjà en cache.
"""

from __future__ import annotations

import importlib
from datetime import date

import pandas as pd
import pytest

import run_pipeline_daily as daily
import run_pipeline_quarterly as quarterly

m03b = importlib.import_module("03b_recuperation_cours_quotidiens")


# --------------------------------------------------------------------------- #
# A -- le délai doit être compatible avec la cadence imposée
# --------------------------------------------------------------------------- #

def test_le_delai_de_03b_couvre_un_univers_complet_a_la_cadence_ibkr():
    """LE défaut à l'origine de ces tests. Le délai doit dépasser ce que la
    cadence du courtier impose, sinon l'étape est structurellement incapable
    d'aboutir -- ce n'est pas de la lenteur, c'est une impossibilité
    arithmétique."""
    cours = next(s for s in daily.daily_steps(7)
                 if s.script == "03b_recuperation_cours_quotidiens.py")

    duree_sp500 = 504 * m03b.HIST_REQUEST_PAUSE_SEC
    assert duree_sp500 > daily.STEP_TIMEOUT_DEFAULT, (
        "le défaut global suffisait : ce test ne protège plus rien")
    assert cours.timeout is not None
    assert cours.timeout > duree_sp500

    # Et de quoi absorber l'univers COMPLET (radiées comprises), sur lequel
    # tourne `make bootstrap`.
    assert cours.timeout > 1500 * m03b.HIST_REQUEST_PAUSE_SEC


def test_les_etapes_ibkr_ont_un_delai_propre_les_autres_le_defaut():
    """Un délai unique est forcément faux pour l'une ou pour l'autre : les
    étapes de calcul travaillent sur le cache local en quelques secondes,
    celles qui interrogent IBKR sont bornées par le courtier."""
    par_script = {s.script: s.timeout for s in daily.daily_steps(7)}

    assert par_script["03b_recuperation_cours_quotidiens.py"] == daily.STEP_TIMEOUT_IBKR
    assert par_script["08_recuperation_options.py"] == daily.STEP_TIMEOUT_IBKR
    for calcul in ("05_calcul_multiples.py", "06_calcul_multiples_moyens.py",
                   "06b_calcul_valorisation_combinee.py", "07_calcul_dcf.py"):
        assert par_script[calcul] is None, calcul


def test_le_delai_de_l_etape_prime_sur_le_delai_global(tmp_path, monkeypatch):
    """C'est ce qui rend Step.timeout effectif : sans cette priorité, la
    déclaration ne changerait rien."""
    vus: list[int] = []

    def espion(cmd, cwd, log_path, timeout):
        vus.append(timeout)
        return 0

    monkeypatch.setattr(quarterly, "_stream_subprocess", espion)
    # run_step journalise le chemin du log RELATIVEMENT à config.BASE_DIR.
    monkeypatch.setattr(quarterly.config, "BASE_DIR", tmp_path)
    rapport = quarterly.RunReport(run_id="t", mode="daily", directory=tmp_path / "run")

    quarterly.run_step(quarterly.Step("x.py", timeout=99_999), [], rapport, 0, 3600)
    quarterly.run_step(quarterly.Step("y.py"), [], rapport, 0, 3600)

    assert vus == [99_999, 3600]
    # Et le délai retenu est journalisé, pour qu'un run interrompu soit
    # diagnosticable sans relire le code.
    assert rapport.steps[0]["timeout_seconds"] == 99_999


# --------------------------------------------------------------------------- #
# B -- points de reprise
# --------------------------------------------------------------------------- #

def _lignes(symbol: str, jours: int, depart: str = "2026-01-05") -> list[dict]:
    dates = pd.bdate_range(depart, periods=jours)
    return [
        {"symbol": symbol, "date": d.date(), "open": 10.0, "high": 11.0,
         "low": 9.0, "close": 10.5, "volume": 1000.0}
        for d in dates
    ]


def test_l_ecriture_est_idempotente(tmp_path):
    """Un point de reprise réécrit le fichier avec un `all_rows` qui s'allonge :
    dix appels intermédiaires doivent donner exactement ce qu'aurait donné un
    unique appel final."""
    sortie = tmp_path / "daily_prices.parquet"
    vide = pd.DataFrame(columns=m03b.OUTPUT_COLUMNS)
    lignes = _lignes("AAA", 10)

    for coupe in (3, 6, 10):
        m03b.ecrire_cours(vide, lignes[:coupe], sortie)

    en_une_fois = m03b.ecrire_cours(vide, lignes, sortie)
    assert len(en_une_fois) == 10
    assert len(pd.read_parquet(sortie)) == 10


def test_l_ecriture_fusionne_sans_dupliquer(tmp_path):
    sortie = tmp_path / "daily_prices.parquet"
    existant = pd.DataFrame(_lignes("AAA", 5))
    # Chevauchement volontaire : les mêmes dates, plus des nouvelles.
    combine = m03b.ecrire_cours(existant, _lignes("AAA", 8), sortie)

    assert len(combine) == 8
    assert not combine.duplicated(subset=["symbol", "date"]).any()


def test_l_ecriture_est_atomique(tmp_path):
    """Le cache représente des heures de collecte. Une écriture directe le
    laisserait tronqué si le process est tué en plein milieu -- ce qui est
    précisément le moment où un point de reprise s'exécute."""
    sortie = tmp_path / "daily_prices.parquet"
    m03b.ecrire_cours(pd.DataFrame(columns=m03b.OUTPUT_COLUMNS), _lignes("AAA", 4), sortie)

    assert sortie.exists()
    # Aucun fichier temporaire ne subsiste après un cycle réussi.
    assert list(tmp_path.glob("*.tmp")) == []


def test_ce_qui_est_sauvegarde_est_saute_a_la_relance():
    """LA propriété qui rend le point de reprise utile : sans elle, on
    écrirait sans jamais reprendre, et le rattrapage n'avancerait pas.

    determine_fetch_from lit le cache pour décider quoi retélécharger : ce
    qu'un point de reprise a écrit devient donc du travail déjà fait."""
    aujourdhui = date(2026, 9, 5)
    cache = pd.DataFrame(_lignes("AAA", 5, depart="2026-09-01"))

    # Ticker présent dans le cache et frais : rien à refaire.
    assert m03b.determine_fetch_from(
        cache, "AAA", date(2010, 1, 1), aujourdhui, refresh_days=7) is None
    # Ticker absent du cache : à collecter depuis le début.
    assert m03b.determine_fetch_from(
        cache, "ZZZ", date(2010, 1, 1), aujourdhui, refresh_days=7) == date(2010, 1, 1)


def test_le_pas_de_reprise_borne_la_perte_a_quelques_tickers():
    """Le réglage doit rester petit devant l'univers, sinon il ne borne rien --
    et assez grand pour que la réécriture du parquet ne domine pas la
    collecte."""
    assert 0 < m03b.CHECKPOINT_EVERY_TICKERS <= 50
    # En minutes de collecte perdues au pire, à la cadence IBKR.
    perte_max_minutes = m03b.CHECKPOINT_EVERY_TICKERS * m03b.HIST_REQUEST_PAUSE_SEC / 60
    assert perte_max_minutes < 10
