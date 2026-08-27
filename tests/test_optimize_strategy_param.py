"""Balayage générique d'un paramètre de stratégie (11e_optimize_strategy_param.py).

Ce script existe pour qu'un réglage optimisable n'ait pas besoin de son propre
fichier de 400 lignes pour être balayable -- et surtout pour fermer un piège :

    Strategy.__init__(**params) stocke TOUT ce qu'on lui passe sans rien
    valider (backtest/strategies/base.py). Un nom mal orthographié est donc
    accepté, rangé dans self.params (jusque dans run_config.json), et n'a
    aucun effet. Un balayage lancé sur ce nom-là produirait N runs
    RIGOUREUSEMENT IDENTIQUES, puis recommanderait une valeur.

Ces tests vérifient d'abord que le piège est réel (sans quoi le garde-fou
n'aurait pas d'objet), puis que le script le refuse.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

from backtest.strategies.valuation_gap_expected_value_options import (
    ValuationGapExpectedValueOptionsStrategy,
)
from backtest.strategies.valuation_gap_multiples_options import (
    ValuationGapMultiplesOptionsStrategy,
)

_opt = importlib.import_module("11e_optimize_strategy_param")


# --------------------------------------------------------------------------- #
# Le piège que le script ferme
# --------------------------------------------------------------------------- #

def test_une_faute_de_frappe_est_bien_avalee_par_la_strategie():
    """La prémisse du garde-fou. Si ce test cessait de passer -- parce que le
    constructeur s'est mis à valider ses kwargs -- le garde-fou deviendrait
    redondant, et il faudrait le dire plutôt que de le garder par habitude."""
    strategie = ValuationGapExpectedValueOptionsStrategy(min_kely_fraction=0.42)
    assert strategie.params["min_kely_fraction"] == 0.42     # accepté...
    assert strategie.min_kelly_fraction == 0.05              # ...et sans aucun effet


def test_les_parametres_acceptes_excluent_kwargs():
    """`**kwargs` ne prouve rien : c'est exactement ce qui rend la faute de
    frappe silencieuse. Seul un nom DÉCLARÉ atteste que le réglage est
    branché."""
    acceptes = _opt._parametres_acceptes(ValuationGapExpectedValueOptionsStrategy)
    assert "min_kelly_fraction" in acceptes          # déclaré par la classe
    assert "convergence_fraction" in acceptes
    assert "entry_threshold_pct" in acceptes         # déclaré par un ancêtre
    assert "kwargs" not in acceptes and "args" not in acceptes
    assert "self" not in acceptes
    assert "min_kely_fraction" not in acceptes       # la faute de frappe


def test_les_parametres_acceptes_suivent_la_strategie():
    """Chaque stratégie a les siens : le script ne doit pas offrir à l'une les
    réglages d'une autre."""
    multiples = _opt._parametres_acceptes(ValuationGapMultiplesOptionsStrategy)
    assert "entry_threshold_pct" in multiples
    assert "min_kelly_fraction" not in multiples


# --------------------------------------------------------------------------- #
# Grilles connues
# --------------------------------------------------------------------------- #

def test_les_grilles_connues_visent_des_parametres_reels():
    """Une grille par défaut pour un paramètre qui n'existe ni sur une
    stratégie ni sur le moteur serait le même piège, déplacé dans ce script."""
    from backtest.strategies import OPTIONS_STRATEGY_REGISTRY

    tous = set(_opt._parametres_moteur())
    for cls in OPTIONS_STRATEGY_REGISTRY.values():
        tous.update(_opt._parametres_acceptes(cls))
    inconnus = [p for p in _opt.KNOWN_PARAMS if p not in tous]
    assert not inconnus, f"grilles par défaut sans paramètre correspondant : {inconnus}"


# --------------------------------------------------------------------------- #
# Paramètres du moteur
# --------------------------------------------------------------------------- #

def test_les_parametres_moteur_sont_reconnus():
    """Les réglages du moteur sont tout aussi optimisables que ceux de la
    stratégie, et aucun n'avait de script avant celui-ci (11 couvre le couple
    stop/take, 11b l'epsilon de rebalancement, et c'est tout)."""
    moteur = _opt._parametres_moteur()
    for nom in ("min_holding_days", "max_delta_notional_pct", "roll_when_days_left",
                "target_tenor_days", "take_profit_convergence_fraction", "min_deployment_pct"):
        assert nom in moteur, nom


def test_les_donnees_et_la_strategie_ne_sont_pas_balayables():
    """Ce que ce script fournit lui-même n'a aucun sens comme axe de balayage :
    les proposer inviterait à une erreur qui ne planterait qu'au bout d'un
    chargement de données complet."""
    moteur = _opt._parametres_moteur()
    for reserve in ("price_panel", "signal_events", "strategy", "start_date",
                    "end_date", "universe_history", "self"):
        assert reserve not in moteur, reserve


def test_run_one_injecte_la_valeur_du_bon_cote():
    """Garde-fou statique sur le point le plus facile à casser en refactor :
    un paramètre moteur injecté dans la stratégie (ou l'inverse) serait
    silencieusement ignoré, et toute la grille rendrait le même run -- le
    piège même que ce script est censé fermer."""
    import inspect
    source = inspect.getsource(_opt._run_one)
    assert 'if cible == "strategy":' in source
    assert 'if cible == "engine":' in source
    # Et les réglages moteur passent par UN SEUL dictionnaire : les séparer en
    # deux ferait qu'une moitié des paramètres serait ignorée du balayage.
    assert "**engine_kwargs," in source
    assert "kwargs[param] = value" in source


@pytest.mark.parametrize("param,valeurs_hors_bornes", [
    ("min_kelly_fraction", [1.0, 1.5, -0.1]),     # f* est borné STRICTEMENT sous 1
    ("strike_grid_n_sigma", [0.0, -1.0]),
    ("exit_threshold_ratio", [-0.1, 1.5]),
])
def test_les_bornes_des_grilles_connues_refusent_l_absurde(param, valeurs_hors_bornes):
    test, _libelle = _opt.KNOWN_PARAMS[param]["bounds"]
    for v in valeurs_hors_bornes:
        assert not test(v), f"{param}={v} devrait être refusé"
    for v in _opt.KNOWN_PARAMS[param]["grid"]:
        assert test(v), f"la grille par défaut de {param} contient {v}, hors de ses propres bornes"


def test_la_grille_par_defaut_du_plancher_encadre_la_valeur_de_config():
    """Une grille qui ne contient pas la valeur en vigueur ne peut pas dire si
    le réglage actuel est bon -- elle ne compare qu'à des alternatives."""
    import config
    grille = _opt.KNOWN_PARAMS["min_kelly_fraction"]["grid"]
    assert config.OPTIONS_EV_MIN_KELLY_FRACTION in grille
    assert min(grille) == 0.0, "la grille doit inclure 0 (plancher désactivé) comme témoin"


# --------------------------------------------------------------------------- #
# Collecte des résultats
# --------------------------------------------------------------------------- #

def test_les_metriques_collectees_sont_celles_du_moteur():
    """Garde-fou statique : METRIC_KEYS ne doit pas dériver vers des noms que
    metrics.compute_metrics ne produit pas -- le CSV se remplirait de colonnes
    vides sans que rien ne le signale."""
    from backtest import metrics as metrics_mod

    equity = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=30),
        "nav": [1_000_000.0 * (1 + 0.001 * i) for i in range(30)],
        "cash": [0.0] * 30, "invested_value": [0.0] * 30, "num_positions": [1] * 30,
    })
    trades = pd.DataFrame([{
        "symbol": "AAA", "entry_date": pd.Timestamp("2020-01-02"),
        "exit_date": pd.Timestamp("2020-01-20"), "shares": 100.0,
        "entry_price": 10.0, "exit_price": 12.0, "pnl": 200.0,
        "return_pct": 20.0, "holding_days": 18, "exit_reason": "take_profit",
    }])
    produites = set(metrics_mod.compute_metrics(equity, trades, 1_000_000.0))
    manquantes = [k for k in _opt.METRIC_KEYS if k not in produites]
    # total_commission/slippage/friction viennent de `extra` (execution_diagnostics),
    # pas de compute_metrics seul : elles sont attendues absentes ici.
    manquantes = [k for k in manquantes if not k.startswith("total_")]
    assert not manquantes, f"METRIC_KEYS demande des métriques que le moteur ne produit pas : {manquantes}"


def test_les_compteurs_de_la_strategie_sont_ramasses_sans_liste_en_dur():
    """C'est ce qui fait qu'un nouveau compteur (comme
    dropped_kelly_below_floor_count) apparaît dans le CSV sans toucher au
    script."""
    import inspect
    source = inspect.getsource(_opt._run_one)
    assert 'getattr(strategy, "diagnostics"' in source
    # Et les compteurs de la stratégie sont bien tous scalaires, donc
    # sérialisables en colonnes CSV.
    diagnostics = ValuationGapExpectedValueOptionsStrategy().diagnostics()
    assert all(
        isinstance(v, (int, float, type(None))) for v in diagnostics.values()
    ), diagnostics
    assert "dropped_kelly_below_floor_count" in diagnostics
