"""Fonctions pures de 11d_optimize_entry_threshold.py.

Le script rejoue des backtests complets : il n'est pas testable ici. Ce qui
l'est -- et ce qui décide de la lecture qu'on fera de la grille -- c'est la
conversion d'unités entre points de log et écart en pourcentage, et le fait
qu'une ligne en erreur reste identifiable dans le CSV.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import config

_CHEMIN = Path(__file__).resolve().parent.parent / "11d_optimize_entry_threshold.py"
_spec = importlib.util.spec_from_file_location("optimize_entry_threshold", _CHEMIN)
opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(opt)


def test_le_seuil_par_defaut_vaut_bien_un_ecart_de_vingt_pourcent():
    """Le nombre 18,23 n'est pas un pourcentage : c'est log(1,20) x 100. Toute
    la lecture de la grille repose sur cette distinction."""
    assert opt._log_points_to_gap_pct(config.OPTIONS_MULTIPLES_ENTRY_THRESHOLD_PCT) == pytest.approx(20.0)


@pytest.mark.parametrize("ecart_pct", [5.0, 10.0, 20.0, 35.0, 50.0, 100.0])
def test_les_deux_conversions_sont_reciproques(ecart_pct):
    aller = opt._gap_pct_to_log_points(ecart_pct)
    assert opt._log_points_to_gap_pct(aller) == pytest.approx(ecart_pct)


def test_un_seuil_nul_laisse_tout_passer():
    assert opt._log_points_to_gap_pct(0.0) == pytest.approx(0.0)


def test_la_grille_par_defaut_contient_le_seuil_en_production():
    """Une grille qui ne contient pas la valeur courante ne peut pas dire si le
    seuil proposé est un gain : il manquerait la ligne de référence."""
    defaut = round(config.OPTIONS_MULTIPLES_ENTRY_THRESHOLD_PCT, 2)
    assert defaut in opt.DEFAULT_ENTRY_THRESHOLD_GRID


def test_la_grille_par_defaut_est_croissante_et_positive():
    grille = opt.DEFAULT_ENTRY_THRESHOLD_GRID
    assert grille == sorted(grille)
    assert all(seuil > 0 for seuil in grille)


def test_le_script_vise_la_strategie_esperance_de_gain_par_defaut():
    from backtest.strategies import OPTIONS_STRATEGY_REGISTRY

    assert opt.DEFAULT_STRATEGY in OPTIONS_STRATEGY_REGISTRY


def test_une_ligne_en_erreur_reste_identifiable():
    """Un seuil qui plante ne doit pas tuer la grille, et sa ligne doit rester
    lisible : sans le seuil ni son écart équivalent, le CSV porterait un trou
    qu'on ne saurait pas rattacher à un point de la grille."""
    ligne = opt._run_one(
        entry_threshold=22.31,
        strategy_name="valuation_gap_expected_value_options",
        strategy_params={"exit_threshold_ratio": 0.70},
        baseline_settings={},          # provoque le KeyError : _DATA est vide de toute façon
        engine_kwargs={},
        start_date=None,
        end_date=None,
    )

    assert ligne["error"]
    assert ligne["entry_threshold_pct"] == 22.31
    assert ligne["ecart_equivalent_pct"] == pytest.approx(25.0, abs=0.01)


def test_le_seuil_de_sortie_suit_le_seuil_d_entree():
    """Le point le plus facile à manquer en lisant la grille : balayer l'entrée
    déplace AUSSI la sortie, parce que la stratégie la dérive du seuil d'entrée
    (exit = entree x ratio). Le CSV doit porter les deux."""
    from backtest.strategies import OPTIONS_STRATEGY_REGISTRY

    cls = OPTIONS_STRATEGY_REGISTRY[opt.DEFAULT_STRATEGY]
    strategie = cls(entry_threshold_pct=22.31, exit_threshold_ratio=0.70)
    assert strategie.exit_threshold_pct == pytest.approx(22.31 * 0.70)


def test_les_trois_strategies_acceptent_le_parametre_balaye():
    """Le script propose les trois stratégies dans --strategy : chacune doit
    réellement accepter entry_threshold_pct, y compris « espérance de gain »
    qui ne l'expose que via **kwargs."""
    from backtest.strategies import OPTIONS_STRATEGY_REGISTRY

    for nom, cls in OPTIONS_STRATEGY_REGISTRY.items():
        strategie = cls(entry_threshold_pct=26.24, exit_threshold_ratio=0.70)
        assert strategie.entry_threshold_pct == pytest.approx(26.24), nom
