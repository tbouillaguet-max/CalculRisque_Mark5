"""Le journal dit ce que l'univers point-in-time contient VRAIMENT.

04, 04b et 04c annonçaient « les entreprises RADIÉES sont incluses » dès que le
fichier de 01b existait, sans l'ouvrir. Le fichier du dépôt ne porte pourtant
que les membres actuels de l'indice (0 radiée au 2026-09-25) : le journal
promettait une correction du biais de survivance qui n'avait pas lieu.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

import config


@pytest.fixture
def historique(tmp_path, monkeypatch):
    def ecrire(membres_actuels: list[bool]) -> None:
        chemin = tmp_path / "historique.parquet"
        pd.DataFrame({"ric": [f"T{i}" for i in range(len(membres_actuels))],
                      "is_current_member": membres_actuels}).to_parquet(chemin)
        monkeypatch.setattr(config, "UNIVERSE_HISTORY_FILE", chemin)
    return ecrire


def _journal(caplog) -> list[logging.LogRecord]:
    with caplog.at_level(logging.INFO):
        config.journaliser_univers_retenu(logging.getLogger("test"), config.UNIVERSE_FULL_FILE)
    return caplog.records


def test_un_univers_sans_radiee_est_signale_comme_tel(historique, caplog):
    historique([True, True, True])
    [message] = _journal(caplog)
    assert message.levelno == logging.WARNING
    assert "AUCUNE entreprise radiée" in message.getMessage()
    assert "01b_historique_univers_sp500.py" in message.getMessage()


def test_les_radiees_sont_comptees(historique, caplog):
    historique([True, False, False])
    [message] = _journal(caplog)
    assert message.levelno == logging.INFO and "2 entreprises radiées incluses" in message.getMessage()


def test_sans_historique_le_journal_ne_pretend_rien(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(config, "UNIVERSE_HISTORY_FILE", tmp_path / "absent.parquet")
    [message] = _journal(caplog)
    assert "des entreprises radiées incluses" in message.getMessage()


def test_l_univers_actuel_ne_declenche_aucun_message(caplog):
    with caplog.at_level(logging.INFO):
        config.journaliser_univers_retenu(logging.getLogger("test"), config.UNIVERSE_FILE)
    assert caplog.records == []


def test_les_trois_collecteurs_passent_par_le_meme_message():
    import pathlib
    racine = pathlib.Path(__file__).resolve().parent.parent
    for script in ("04_recuperation_10k.py", "04b_recuperation_10q.py", "04c_recuperation_8k.py"):
        texte = (racine / script).read_text(encoding="utf-8")
        assert "config.journaliser_univers_retenu(logger, tickers_file)" in texte, script
        assert "les entreprises RADIÉES sont incluses" not in texte, script
