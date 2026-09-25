"""HORIZON DE CONVERGENCE : au bout de combien de temps une thèse n'en est plus une ?

CE QU'IL ATTAQUE. Le moteur actions ne ferme une position que sur stop-loss,
prise de gain, stop suiveur, perte de signal ou repesage. Une thèse de
convergence qui ne se réalise jamais n'est donc fermée par RIEN : elle occupe du
capital indéfiniment. `max_holding_days` est la règle qui y met un terme, et
c'est le premier axe de la grille qui porte sur le SIGNAL et non sur
l'exécution -- les cinq autres disent comment on négocie, celui-ci dit combien
de temps on croit.

CE QUI A MOTIVÉ LA MESURE. Sur les 13 141 sorties de la configuration de
référence, le rendement ANNUALISÉ décroît de +79 %/an sous 30 jours à +4 %/an
au-delà de 545, pendant que le rendement ABSOLU reste plat (3,1 % à 9,5 %) : le
gain s'accumule dans les premières semaines puis s'arrête. La décroissance
survit au retrait des sorties déclenchées par le prix, donc ce n'est pas un
artefact du take-profit qui tronque les positions rapides.

LE TEST QUI COMPTE est `test_l_horizon_ferme_vraiment_les_positions_qui_trainent` :
sans lui, un axe qui ne change rien passerait pour un axe mesuré.
"""

from __future__ import annotations

import argparse
import importlib

import pandas as pd
import pytest

import config

_opt = importlib.import_module("16_optimize_strategie_actions")


def _args(**surcharges) -> argparse.Namespace:
    base = dict(entry_threshold_grid=None, momentum_grid=None, stop_loss_grid=None,
                take_profit_grid=None, rebalance_band_grid=None, max_weight_grid=None,
                max_holding_grid=None, multiple_hierarchy_grid=None, quick=False)
    return argparse.Namespace(**{**base, **surcharges})


# --------------------------------------------------------------------------- #
# L'axe est branché, et il mord
# --------------------------------------------------------------------------- #
def test_l_horizon_est_un_axe_de_la_grille():
    """L'axe existe toujours ; son DÉFAUT a été réduit à la valeur de production
    après mesure (cf. tests/test_grille_hierarchie). Il se balaie à la demande,
    et c'est ce que fait ce test."""
    grille = _opt._build_grid(_args(), "valuation_gap_combined")
    assert "max_holding_days" in grille[0]
    assert {c["max_holding_days"] for c in grille} == {None}

    balaye = _opt._build_grid(
        _args(max_holding_grid=[-1.0, 90.0, 180.0, 365.0]), "valuation_gap_combined")
    assert {c["max_holding_days"] for c in balaye} == {None, 90, 180, 365}


def test_l_horizon_arrive_bien_au_moteur():
    """Un axe balayé mais jamais transmis produirait 432 runs identiques, et un
    rapport parfaitement calculé sur du vide."""
    import inspect
    source = inspect.getsource(_opt._run_one)
    assert 'max_holding_days=combo["max_holding_days"]' in source


def test_l_ordre_des_axes_fait_foi():
    """_build_grid zippe AXES sur ses axes : décaler l'un des deux renommerait
    silencieusement toutes les colonnes de la grille. Ce test est le seul
    endroit où cette correspondance est vérifiée."""
    grille = _opt._build_grid(_args(), "valuation_gap_combined")
    assert tuple(grille[0].keys()) == _opt.AXES
    # Et les valeurs atterrissent sous le bon nom, pas seulement dans le bon ordre.
    for combo in grille:
        assert combo["stop_loss_pct"] in _opt.DEFAULT_STOP_LOSS_GRID
        assert combo["take_profit_pct"] in _opt.DEFAULT_TAKE_PROFIT_GRID
        assert combo["max_holding_days"] in _opt.DEFAULT_MAX_HOLDING_GRID


def test_l_horizon_ferme_vraiment_les_positions_qui_trainent():
    """LE point de tout le mécanisme : sans cette propriété, l'axe serait une
    colonne de plus dans un CSV. Mesuré sur l'historique complet, un horizon de
    365 jours produit 155 sorties `max_holding` et ramène la détention maximale
    de 928 à 369 jours."""
    from backtest.engine import BacktestEngine
    from backtest.strategies import STRATEGY_REGISTRY

    cls = STRATEGY_REGISTRY["valuation_gap_combined"]
    pytest.importorskip("pyarrow")
    try:
        data = _opt._load_data(config.BENCHMARK_SYMBOL, cls.signal_source)
    except (FileNotFoundError, OSError) as exc:      # données LFS non matérialisées
        pytest.skip(f"données indisponibles : {exc}")

    durees = {}
    for horizon in (None, 180):
        engine = BacktestEngine(
            price_panel=data["price_panel"], signal_events=data["signal_events"],
            universe_history=data["universe_history"],
            fallback_universe_symbols=data["fallback_symbols"],
            material_events_8k=data["material_events"], strategy=cls(),
            start_date=pd.Timestamp("2015-01-01"),
            initial_capital=config.BACKTEST_INITIAL_CAPITAL,
            cost_bps=config.BACKTEST_COMMISSION_BPS + config.BACKTEST_SLIPPAGE_BPS,
            stop_loss_pct=config.BACKTEST_STOP_LOSS_PCT,
            take_profit_pct=config.BACKTEST_TAKE_PROFIT_PCT,
            rebalance_band_pct=config.BACKTEST_REBALANCE_BAND_PCT,
            momentum_min_pct=config.BACKTEST_STOCKS_MOMENTUM_MIN_PCT,
            max_holding_days=horizon,
        )
        _, _, trades, _ = engine.run()
        durees[horizon] = trades

    assert (durees[None].exit_reason == "max_holding").sum() == 0
    assert (durees[180].exit_reason == "max_holding").sum() > 100
    assert durees[None].holding_days.max() > 900
    # La borne est franchie de quelques jours, et c'est NORMAL : la décision est
    # prise à la clôture de J et exécutée à l'ouverture de J+1, week-ends et
    # jours fériés compris. Exiger <= 180 ferait échouer un moteur correct.
    assert 180 <= durees[180].holding_days.max() <= 180 + 10


# --------------------------------------------------------------------------- #
# La configuration en production reste comparable
# --------------------------------------------------------------------------- #
def test_la_production_n_a_aucun_horizon():
    """Le défaut mesuré : c'est la valeur CONTRE laquelle la grille compare."""
    assert config.BACKTEST_MAX_HOLDING_DAYS is None
    assert _opt._reference_combo("valuation_gap_combined")["max_holding_days"] is None


def test_la_grille_contient_toujours_la_configuration_en_production():
    """Condition du test apparié : sans la production DANS la grille, il n'y a
    rien à comparer. Ajouter un axe est exactement le moment où on la perd."""
    reference = _opt._reference_combo("valuation_gap_combined")
    grille = [{**c, "_nav": [1.0], "_dates": pd.DatetimeIndex([pd.Timestamp("2015-01-01")])}
              for c in _opt._build_grid(_args(), "valuation_gap_combined")]
    assert _opt._index_de_reference(grille, reference) is not None


@pytest.mark.parametrize("strategie", ["valuation_gap_dcf", "valuation_gap_combined",
                                       "valuation_gap_sector_neutral"])
def test_chaque_strategie_garde_sa_production_dans_la_grille(strategie):
    reference = _opt._reference_combo(strategie)
    grille = [{**c, "_nav": [1.0], "_dates": pd.DatetimeIndex([pd.Timestamp("2015-01-01")])}
              for c in _opt._build_grid(_args(), strategie)]
    assert _opt._index_de_reference(grille, reference) is not None, strategie


# --------------------------------------------------------------------------- #
# Lecture et ligne de commande
# --------------------------------------------------------------------------- #
def test_aucun_horizon_se_lit_en_clair():
    """Un NaN dans la colonne d'un axe est une VALEUR (« pas d'horizon »), pas
    une absence : l'afficher brut laisserait croire à une donnée manquante."""
    assert "aucun horizon" in _opt._lisible("max_holding_days", None)
    assert "aucun horizon" in _opt._lisible("max_holding_days", float("nan"))
    assert _opt._lisible("max_holding_days", 365) == "365 j"
    # Relu depuis un CSV, un entier revient en flottant.
    assert _opt._lisible("max_holding_days", 365.0) == "365 j"


def test_une_valeur_negative_en_ligne_de_commande_vaut_desactive():
    """Même convention que --momentum-grid : argparse ne sait pas lire None."""
    grille = _opt._build_grid(_args(max_holding_grid=[-1.0, 200.0]), "valuation_gap_combined")
    assert {c["max_holding_days"] for c in grille} == {None, 200}


def test_la_grille_reste_entiere_quand_on_balaie_l_horizon():
    """4 take-profit x 3 momentum x 3 seuils x 3 bandes = 108 par horizon."""
    balaye = _opt._build_grid(
        _args(max_holding_grid=[-1.0, 90.0, 180.0, 365.0],
              multiple_hierarchy_grid=["flat"]), "valuation_gap_combined")
    assert len(balaye) == 432


def test_le_mode_quick_garde_les_bornes_de_l_axe():
    """--quick sert à valider le montage : s'il perdait l'axe, il validerait un
    montage qui n'est pas celui qui tourne."""
    grille = _opt._build_grid(
        _args(quick=True, max_holding_grid=[-1.0, 90.0, 180.0, 365.0]),
        "valuation_gap_combined")
    assert {c["max_holding_days"] for c in grille} == {None, 365}


# --------------------------------------------------------------------------- #
# Le rapport suit l'axe tout seul
# --------------------------------------------------------------------------- #
def test_le_tableau_du_rapport_suit_AXES():
    """Les colonnes affichées étaient une liste recopiée : un axe ajouté y
    manquait en silence, et on lisait un classement dont une colonne variait
    sans apparaître."""
    import inspect
    source = inspect.getsource(_opt._report)
    assert "colonnes = [*AXES, rank_key]" in source
    assert "for key in AXES:" in source
