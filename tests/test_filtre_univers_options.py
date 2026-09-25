"""08 collecte les options de ce que les stratégies options peuvent ouvrir -- pas d'autre chose.

CE QUI N'ALLAIT PAS. `filter_universe_by_valuation_gap` lisait l'écart du DCF
(07, `resultats_dcf.xlsx`), alors que toutes les stratégies options tradent la
valorisation COMBINÉE (06b). Une banque, un assureur ou une foncière n'a pas de
DCF : 111 entreprises valorisées par 06b en sont dépourvues. Elles étaient
écartées quel que soit leur écart, donc leurs options n'étaient JAMAIS
collectées -- et une position ouverte dessus serait restée simulée par
Black-Scholes pour toujours.

UN SECOND ÉCART, TROUVÉ EN CORRIGEANT LE PREMIER. La stratégie multiples entre
sur un écart en LOG, symétrique : ratio théorique/cours >= 1,20 pour un call,
<= 1/1,20 = 0,833 pour un put. Le filtre coupait à ±20 % en écart SIMPLE, soit
un ratio <= 0,80 côté put : la bande 0,80-0,833 était tradée, jamais collectée.

CE QUI NE CHANGE PAS, ET QU'IL FALLAIT VÉRIFIER : le « tout simulé » des
backtests options. Sur trois runs 2015-2026, toutes les positions sauf une ou
deux s'ouvrent avant le 2026-07-29, date du premier snapshot réel (2 732 sur
2 734 pour la stratégie multiples) : le moteur ne regarde qu'en arrière, et
l'historique ne couvre que cinq semaines. Ce correctif compte pour la collecte
À VENIR, pas pour le passé. Des rares positions ouvertes après, un PUT COIN a
été simulé faute de chaîne parce que COIN n'a pas de DCF ; un PUT IP aussi,
alors qu'IP en a un -- le filtre n'explique pas tout.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pandas as pd
import pytest

import config

_08 = importlib.import_module("08_recuperation_options")


@pytest.fixture
def valorisation(tmp_path, monkeypatch):
    """Écrit une valorisation combinée minimale et y fait pointer config."""
    def ecrire(lignes: list[dict]) -> None:
        chemin = tmp_path / "valorisation_combinee_historique.parquet"
        pd.DataFrame(lignes).to_parquet(chemin, index=False)
        monkeypatch.setattr(config, "VALORISATION_COMBINEE_FILE", chemin)
    return ecrire


def _univers(*symboles) -> pd.DataFrame:
    return pd.DataFrame({"ib_symbol": list(symboles)})


def _ligne(symbole, gap_pct, filed="2026-08-01") -> dict:
    return {"symbol": symbole, "filed_date": pd.Timestamp(filed), "gap_pct": gap_pct}


def _retenus(univers, seuil=20.0) -> set:
    return set(_08.filter_universe_by_valuation_gap(univers, threshold_pct=seuil)["ib_symbol"])


# --------------------------------------------------------------------------- #
# LA correction
# --------------------------------------------------------------------------- #
def test_une_banque_sans_dcf_est_desormais_collectee(valorisation, monkeypatch, tmp_path):
    """LE point. Une entreprise valorisée par ses multiples, sans aucun DCF, doit
    être retenue si son écart combiné dépasse le seuil -- et le filtre ne doit
    même plus avoir besoin du fichier DCF pour le décider."""
    valorisation([_ligne("JPM", 45.0)])
    monkeypatch.setattr(config, "DCF_FILE", tmp_path / "absent.xlsx")
    assert _retenus(_univers("JPM")) == {"JPM"}


def test_un_ecart_combine_sous_le_seuil_n_est_plus_collecte(valorisation):
    """L'autre sens : un DCF à +40 % ne dit rien si la valeur COMBINÉE, celle
    que les stratégies lisent, ne s'écarte que de 5 %. 56 entreprises étaient
    collectées pour rien de cette façon."""
    valorisation([_ligne("AAA", 5.0)])
    assert _retenus(_univers("AAA")) == set()


# --------------------------------------------------------------------------- #
# Le seuil en log, symétrique
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gap_pct,retenu", [
    (20.0, True),       # ratio 1,20 : le seuil exact côté call
    (19.0, False),
    (-16.67, True),     # ratio 0,833 : le seuil exact côté put
    (-18.0, True),      # dans la bande 0,80-0,833, que l'écart simple ratait
    (-16.0, False),     # ratio 0,84 : sous le seuil
    (-20.0, True),
])
def test_le_seuil_est_symetrique_en_log(valorisation, gap_pct, retenu):
    """Ratio >= 1,20 ou <= 1/1,20 : exactement la règle d'entrée de
    valuation_gap_multiples_options (100 x ln(théorique/cours) >= 100 x ln 1,2)."""
    valorisation([_ligne("AAA", gap_pct)])
    assert (_retenus(_univers("AAA")) == {"AAA"}) is retenu


def test_le_seuil_du_filtre_est_celui_de_la_strategie_multiples():
    """Si l'un des deux bougeait sans l'autre, le filtre recommencerait à
    collecter autre chose que ce qui est tradé."""
    assert 100 * math.log(1 + config.VALUATION_GAP_THRESHOLD_PCT / 100) == pytest.approx(
        config.OPTIONS_MULTIPLES_ENTRY_THRESHOLD_PCT)


def test_une_valeur_theorique_negative_n_est_retenue_par_aucun_seuil(valorisation):
    """Pas de logarithme, donc pas de signal -- la stratégie multiples, qui
    calcule le même logarithme, n'entrerait pas non plus."""
    valorisation([_ligne("AAA", -150.0)])
    assert _retenus(_univers("AAA")) == set()
    assert np.isnan(_08.ecart_log_pct(pd.Series([-150.0])).iloc[0])


# --------------------------------------------------------------------------- #
# Le signal retenu
# --------------------------------------------------------------------------- #
def test_seul_le_dernier_depot_compte(valorisation):
    """Un écart de 40 % il y a deux ans, démenti par le dernier dépôt, ne
    déclenche plus rien : c'est le dernier signal qu'une stratégie lirait."""
    valorisation([_ligne("AAA", 40.0, "2024-08-01"), _ligne("AAA", 3.0, "2026-08-01")])
    assert _retenus(_univers("AAA")) == set()


def test_seules_les_entreprises_de_l_univers_fourni_sont_rendues(valorisation):
    valorisation([_ligne("AAA", 40.0), _ligne("BBB", 40.0)])
    assert _retenus(_univers("AAA")) == {"AAA"}


def test_sans_valorisation_combinee_le_message_dit_quoi_lancer(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VALORISATION_COMBINEE_FILE", tmp_path / "absent.parquet")
    with pytest.raises(FileNotFoundError, match="06b_calcul_valorisation_combinee"):
        _08.filter_universe_by_valuation_gap(_univers("AAA"), threshold_pct=20.0)


# --------------------------------------------------------------------------- #
# Sur les vraies données
# --------------------------------------------------------------------------- #
def test_sur_les_donnees_du_depot_les_entreprises_sans_dcf_sont_rattrapees():
    """Mesure plutôt que promesse : parmi les entreprises retenues, il doit y
    en avoir sans aucun DCF -- sinon le correctif n'aurait rien changé."""
    try:
        dcf = pd.read_excel(config.DCF_FILE, sheet_name="DCF", engine="openpyxl")
        # À travers le VRAI chargeur de 08 : c'est lui qui dérive ib_symbol du
        # RIC. Lire le CSV à la main, c'est tester une colonne qui n'existe pas.
        univers = _08.load_universe(config.UNIVERSE_FILE)
    except (FileNotFoundError, OSError, ValueError) as exc:
        pytest.skip(f"données indisponibles : {exc}")

    retenus = set(_08.filter_universe_by_valuation_gap(
        univers, threshold_pct=config.VALUATION_GAP_THRESHOLD_PCT)["ib_symbol"])
    avec_dcf = set(dcf.dropna(subset=["Écart_DCF_vs_Cours_%"])["Ticker"].astype(str))
    assert len(retenus - avec_dcf) > 20, (
        f"seulement {len(retenus - avec_dcf)} entreprises sans DCF retenues")
