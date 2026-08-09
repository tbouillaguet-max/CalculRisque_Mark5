"""Erreur 200 d'IBKR : contrats bâtis sur SMART, et bruit de sondage filtré.

L'erreur 200 ("No security definition has been found for the request") est la
réponse NORMALE d'IBKR à une combinaison (strike, échéance) non cotée, et
qualify_contracts_in_batches sonde délibérément une grille dont une partie
n'existe pas. Elle était pourtant journalisée en ERROR, et surtout bien plus
fréquente que nécessaire."""

from __future__ import annotations

import importlib.util
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent


def _charger_08():
    spec = importlib.util.spec_from_file_location("module_08", RACINE / "08_recuperation_options.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module_08 = pytest.importorskip("ib_insync") and _charger_08()


# Échéances calculées à partir d'aujourd'hui : des dates en dur seraient
# filtrées comme échues dès que le temps passe, et le test se mettrait à
# vérifier autre chose que ce qu'il annonce.
PROCHE = (datetime.now() + timedelta(days=200)).strftime("%Y%m%d")
LOINTAINE = (datetime.now() + timedelta(days=650)).strftime("%Y%m%d")


class _Chaine:
    """Entrée de reqSecDefOptParams : ses listes de strikes et d'échéances
    sont l'UNION de ce qui est coté toutes places confondues, alors que son
    champ `exchange` ne désigne qu'UNE place."""

    exchange = "PSE"
    tradingClass = "ICE"
    multiplier = "100"
    expirations = [PROCHE, LOINTAINE]
    strikes = [130.0, 134.0, 135.0]


class _Sousjacent:
    symbol = "ICE"
    currency = "USD"


def _contrats(min_days=1, band_pct=0.30, spot=133.0):
    return module_08.build_candidate_contracts(_Sousjacent(), _Chaine(), spot, min_days, band_pct)


# --------------------------------------------------------------------------- #
# Construction des contrats
# --------------------------------------------------------------------------- #

def test_les_contrats_sont_batis_sur_smart():
    """chain.exchange (PSE) ne liste pas forcément tout ce que chain.strikes
    annonce : bâtir la grille dessus demandait des contrats qui existent
    ailleurs -- d'où le flot d'erreurs 200, et des contrats bien réels perdus."""
    contrats = _contrats()
    assert contrats, "aucun contrat candidat construit"
    assert {c.exchange for c in contrats} == {"SMART"}


def test_la_classe_et_le_multiplicateur_viennent_toujours_de_la_chaine():
    """Seule la PLACE change : le reste de l'identité du contrat reste celui
    que reqSecDefOptParams a renvoyé."""
    for contrat in _contrats():
        assert contrat.tradingClass == "ICE"
        assert contrat.multiplier == "100"
        assert contrat.currency == "USD"


def test_la_grille_couvre_strikes_echeances_et_sens():
    contrats = _contrats()
    assert {c.right for c in contrats} == {"C", "P"}
    assert {c.strike for c in contrats} == {130.0, 134.0, 135.0}
    assert {c.lastTradeDateOrContractMonth for c in contrats} == {PROCHE, LOINTAINE}


def test_la_bande_de_strikes_est_respectee():
    contrats = _contrats(band_pct=0.01, spot=134.0)
    assert {c.strike for c in contrats} == {134.0, 135.0}


def test_les_echeances_trop_proches_sont_ecartees():
    assert _contrats(min_days=100_000) == []


# --------------------------------------------------------------------------- #
# Filtrage du bruit
# --------------------------------------------------------------------------- #

def _filtre(message: str) -> bool:
    """True = le message est conservé dans les logs."""
    record = logging.LogRecord("ib_insync", logging.ERROR, "", 0, message, None, None)
    return module_08.NoisyIBFilter().filter(record)


def test_l_erreur_200_est_filtree():
    assert not _filtre(
        "Error 200, reqId 8856: No security definition has been found for the request, "
        "contract: Option(symbol='ICE', lastTradeDateOrContractMonth='20270617', strike=134.0)"
    )


@pytest.mark.parametrize("message", [
    "Error 162, reqId 42: Historical Market Data Service error",
    "Error 322, reqId 1: Duplicate ticker id",
    "Error 10197, reqId 7: No market data during competing live session",
])
def test_les_erreurs_qui_comptent_restent_visibles(message):
    """Le filtre doit rester chirurgical : 200 est un résultat de sondage,
    162/322/10197 sont de vrais incidents que le script traite."""
    assert _filtre(message)


def test_200_ne_declenche_ni_blacklist_ni_recuperation_de_session():
    """Une combinaison non cotée ne dit RIEN sur la place ni sur la session."""
    assert 200 not in module_08.PERMISSION_ERROR_CODES
    assert 200 not in module_08.SESSION_ERROR_CODES

    module_08.BLACKLISTED_EXCHANGES.clear()
    module_08._session_poisoned.clear()
    module_08._on_ib_error(8856, 200, "No security definition has been found for the request")
    assert not module_08.BLACKLISTED_EXCHANGES
    assert not module_08._session_poisoned.is_set()
