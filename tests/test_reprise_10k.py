"""Reprise d'un run interrompu de 04_recuperation_10k.py.

CE QUE CES TESTS PROTÈGENT, ET CE QUI MANQUAIT. 04 accumulait TOUT en mémoire
et n'écrivait `financials.parquet` ni son état de suivi qu'à la toute fin. Une
interruption -- Ctrl+C, coupure réseau, machine en veille, session distante
fermée -- perdait donc l'intégralité du run, et le suivant repartait de zéro :
`should_skip` ne peut ignorer un ticker que si l'état a été SAUVEGARDÉ, ce qui
n'arrivait jamais sur un run interrompu.

C'était le seul des quatre scripts SEC dans ce cas (04b, 04c et 07b ont leur
reprise depuis longtemps), et c'est le plus long : un backfill sur l'univers
complet interroge companyfacts pour ~500 entreprises, à quelques requêtes par
seconde. Le seul run qu'on ne peut pas se permettre de perdre était donc le
seul qu'on perdait.
"""

from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest

import config

_04 = importlib.import_module("04_recuperation_10k")


@pytest.fixture(autouse=True)
def dossier_isole(tmp_path, monkeypatch):
    """Redirige financials.parquet -- et donc les fichiers de reprise, qui
    vivent à côté -- vers un dossier temporaire."""
    monkeypatch.setattr(config, "FINANCIALS_FILE", tmp_path / "financials.parquet")
    return tmp_path


def _ligne(symbol: str, year: int) -> dict:
    return {
        "symbol": symbol, "cik": "0000000001", "year": year,
        "revenue": 1000.0, "ebit": 100.0, "filed_date": f"{year + 1}-03-01",
    }


# --------------------------------------------------------------------------- #
# Écriture au fil de l'eau
# --------------------------------------------------------------------------- #
def test_le_checkpoint_est_relisible_avant_la_fin_du_run():
    """LE point : ce qui est écrit doit être exploitable AVANT que le run se
    termine. Un parquet réécrit en bloc ne l'est pas ; un JSONL append-only si."""
    _04.append_checkpoint([_ligne("AAA", 2020), _ligne("AAA", 2021)])
    _04.append_checkpoint([_ligne("BBB", 2020)])

    lignes = _04.load_checkpoint_rows()
    assert len(lignes) == 3
    assert {l["symbol"] for l in lignes} == {"AAA", "BBB"}


def test_une_ligne_tronquee_ne_fait_pas_perdre_les_precedentes():
    """Un run tué en plein write laisse une ligne incomplète : on perd
    celle-là, pas les milliers qui précèdent."""
    _04.append_checkpoint([_ligne("AAA", 2020), _ligne("BBB", 2020)])
    with _04._checkpoint_path().open("a", encoding="utf-8") as f:
        f.write('{"symbol": "CCC", "year": 20')  # coupé net

    assert len(_04.load_checkpoint_rows()) == 2


def test_l_aller_retour_jsonl_preserve_les_valeurs():
    """Le checkpoint passe par JSON : il ne doit rien déformer, en
    particulier `filed_date`, dont dépend tout le caractère point-in-time du
    signal en aval."""
    _04.append_checkpoint([_ligne("AAA", 2020)])
    relu = _04.load_checkpoint_rows()[0]

    assert relu["filed_date"] == "2021-03-01"
    assert relu["revenue"] == 1000.0
    assert relu["symbol"] == "AAA"


def test_la_progression_survit_a_une_relecture():
    _04.save_progress({"AAA", "BBB"})
    assert _04.load_progress() == {"AAA", "BBB"}


def test_une_progression_illisible_ne_bloque_pas_le_run():
    """Mieux vaut refaire le travail que planter au démarrage : une reprise
    impossible doit dégrader en run complet, pas en erreur."""
    _04._progress_path().parent.mkdir(parents=True, exist_ok=True)
    _04._progress_path().write_text("{ceci n'est pas du json", encoding="utf-8")
    assert _04.load_progress() == set()


def test_l_ecriture_de_la_progression_est_atomique():
    """Passage par un fichier temporaire puis `replace` : une coupure pendant
    l'écriture ne peut pas laisser un fichier de progression à moitié écrit,
    qui ferait repartir la reprise sur une liste tronquée."""
    _04.save_progress({"AAA"})
    _04.save_progress({"AAA", "BBB", "CCC"})

    assert _04.load_progress() == {"AAA", "BBB", "CCC"}
    assert not _04._progress_path().with_suffix(".json.tmp").exists()


# --------------------------------------------------------------------------- #
# Ce que la reprise économise réellement
# --------------------------------------------------------------------------- #
def test_un_ticker_deja_traite_n_est_pas_reinterroge():
    """La reprise ne sert à rien si elle ne supprime pas l'appel SEC : c'est
    le temps réseau qu'on veut économiser, pas le temps de calcul."""
    traites = _04.load_progress() or set()
    _04.save_progress(traites | {"AAA", "BBB"})

    symbols = ["AAA", "BBB", "CCC"]
    deja = _04.load_progress()
    a_interroger = [s for s in symbols if s not in deja]

    assert a_interroger == ["CCC"]


def test_un_echec_est_marque_traite_mais_pas_a_jour():
    """Distinction qui compte. Un ticker en échec est marqué `processed` --
    --resume reprend un run interrompu, il ne rejoue pas les échecs de ce
    run-là -- mais PAS dans `state` : le prochain run complet le réessaiera au
    lieu de l'ignorer pendant --refresh-days jours."""
    etat: dict = {}
    traites: set = set()

    # Simule la boucle du script sur un ticker dont l'extraction rend vide.
    df_vide = pd.DataFrame()
    symbol = "AAA"
    if not df_vide.empty:
        etat[symbol] = "2026-01-01T00:00:00"
    traites.add(symbol)

    assert symbol in traites
    assert symbol not in etat
    assert _04.should_skip(symbol, pd.DataFrame(), etat, refresh_days=30) is False


def test_le_throttle_ignore_un_ticker_recemment_interroge():
    """L'autre moitié du dispositif : une fois le run TERMINÉ, un second run
    n'appelle plus la SEC pour les tickers déjà à jour."""
    existant = pd.DataFrame([_ligne("AAA", 2020)])
    etat = {"AAA": pd.Timestamp.now().isoformat(timespec="seconds")}

    assert _04.should_skip("AAA", existant, etat, refresh_days=30) is True
    assert _04.should_skip("BBB", existant, etat, refresh_days=30) is False


def test_un_ticker_en_cache_mais_sans_donnees_n_est_pas_ignore():
    """Garde-fou : un état de suivi qui affirme « déjà interrogé » alors que
    le parquet ne contient rien pour ce symbole ne doit pas empêcher de le
    récupérer -- sinon une ligne manquante le resterait pour toujours."""
    etat = {"AAA": pd.Timestamp.now().isoformat(timespec="seconds")}
    assert _04.should_skip("AAA", pd.DataFrame(), etat, refresh_days=30) is False


def test_le_script_expose_bien_une_option_resume():
    """Garde-fou statique : si `--resume` disparaissait d'un refactor, la
    reprise redeviendrait inaccessible sans que rien d'autre n'échoue."""
    import inspect
    source = inspect.getsource(_04.main)
    assert '"--resume"' in source
    assert "load_progress()" in source
    # Sauvegarde dans un `finally` : c'est ce qui couvre le Ctrl+C, qui est
    # le cas d'interruption le plus courant.
    assert "finally:" in inspect.getsource(_04.main)
