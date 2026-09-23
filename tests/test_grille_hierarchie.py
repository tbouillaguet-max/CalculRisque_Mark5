"""L'axe de hiérarchie, branché sur la grille.

CE QU'IL A DE PARTICULIER, et pourquoi il mérite ses propres tests : c'est le
PREMIER axe non numérique de cette grille, et le premier qui change les
ÉVÉNEMENTS DE SIGNAL plutôt qu'un réglage de moteur. Les deux comparaisons de
référence (`_index_de_reference`, `_compare_a_la_reference`) convertissaient
toute valeur d'axe en flottant : `float("flat")` lève ValueError, et la grille
entière serait tombée au premier test apparié.

LE TEST QUI COMPTE est `test_la_reference_se_retrouve_avec_un_axe_en_chaine`.
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np
import pandas as pd
import pytest

import hierarchie_multiples as hm

_opt = importlib.import_module("16_optimize_strategie_actions")


def _args(**surcharges) -> argparse.Namespace:
    base = dict(entry_threshold_grid=None, momentum_grid=None, stop_loss_grid=None,
                take_profit_grid=None, rebalance_band_grid=None, max_weight_grid=None,
                max_holding_grid=None, multiple_hierarchy_grid=None, quick=False)
    return argparse.Namespace(**{**base, **surcharges})


def _ligne(reference: dict, **surcharges) -> dict:
    rng = np.random.default_rng(0)
    return {
        **reference, **surcharges, "sharpe_ratio": 1.0,
        "_nav": 100_000.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.011, 400)),
        "_dates": pd.bdate_range("2015-01-01", periods=400),
    }


# --------------------------------------------------------------------------- #
# Le piège du premier axe non numérique
# --------------------------------------------------------------------------- #
def test_la_reference_se_retrouve_avec_un_axe_en_chaine():
    """LE point. Avant `_meme_valeur_d_axe`, la comparaison faisait
    `float(valeur)` sur chaque axe : la ligne de référence était introuvable et
    le test apparié ne tournait plus du tout."""
    reference = _opt._reference_combo("valuation_gap_combined")
    assert isinstance(reference["multiple_hierarchy"], str)

    lignes = [
        _ligne(reference, multiple_hierarchy="tiers"),
        _ligne(reference, multiple_hierarchy="pe_first"),
        _ligne(reference),
    ]
    assert _opt._index_de_reference(lignes, reference) == 2


def test_une_chaine_differente_n_est_pas_la_reference():
    reference = _opt._reference_combo("valuation_gap_combined")
    assert _opt._index_de_reference([_ligne(reference, multiple_hierarchy="tiers")],
                                    reference) is None


@pytest.mark.parametrize("valeur,attendue,identiques", [
    ("flat", "flat", True),
    ("flat", "tiers", False),
    (None, None, True),
    (float("nan"), None, True),
    (None, "flat", False),
    ("flat", None, False),
    (15.0, 15.0, True),
    (15.0 + 1e-12, 15.0, True),      # le CSV arrondit
    (15.01, 15.0, False),
    (float("nan"), 15.0, False),
])
def test_la_comparaison_d_axe_couvre_les_trois_cas(valeur, attendue, identiques):
    assert _opt._meme_valeur_d_axe(valeur, attendue) is identiques


def test_une_chaine_n_est_jamais_prise_pour_une_valeur_absente():
    """`pd.isna` sur une chaîne rend False, mais l'appeler sur certains objets
    lève : d'où `_est_vide`, qui tranche avant."""
    assert _opt._est_vide("flat") is False
    assert _opt._est_vide(None) is True
    assert _opt._est_vide(float("nan")) is True


# --------------------------------------------------------------------------- #
# L'axe dans la grille
# --------------------------------------------------------------------------- #
def test_l_axe_balaie_les_quatre_hierarchies_sur_la_strategie_combinee():
    grille = _opt._build_grid(_args(), "valuation_gap_combined")
    assert {c["multiple_hierarchy"] for c in grille} == set(hm.HIERARCHIES)
    assert len(grille) == 432


def test_l_axe_est_sans_objet_sur_une_strategie_dcf():
    """Le DCF ne combine aucun multiple : balayer l'axe produirait quatre
    copies identiques de chaque combinaison, présentées comme distinctes."""
    grille = _opt._build_grid(_args(), "valuation_gap_dcf")
    assert {c["multiple_hierarchy"] for c in grille} == {None}
    assert len(grille) == 108


def test_la_reference_est_la_hierarchie_REELLEMENT_utilisee():
    """`config.MULTIPLE_COMBINATION` dit `tiers`, le parquet porte `flat` (cf.
    tests/test_hierarchie_multiples). La référence du test apparié doit être ce
    qui TOURNE, sinon la grille se compare à une configuration qui n'a jamais
    existé."""
    assert _opt._reference_combo("valuation_gap_combined")["multiple_hierarchy"] == "flat"
    assert _opt._reference_combo("valuation_gap_dcf")["multiple_hierarchy"] is None


def test_chaque_strategie_garde_sa_production_dans_la_grille():
    for strategie in ("valuation_gap_dcf", "valuation_gap_combined",
                      "valuation_gap_sector_neutral"):
        reference = _opt._reference_combo(strategie)
        grille = [_ligne(c) for c in _opt._build_grid(_args(), strategie)]
        assert _opt._index_de_reference(grille, reference) is not None, strategie


def test_la_ligne_de_commande_restreint_l_axe():
    grille = _opt._build_grid(
        _args(multiple_hierarchy_grid=["flat", "tiers"]), "valuation_gap_combined")
    assert {c["multiple_hierarchy"] for c in grille} == {"flat", "tiers"}


# --------------------------------------------------------------------------- #
# Le signal suit l'axe
# --------------------------------------------------------------------------- #
def test_le_signal_est_choisi_par_la_combinaison_et_non_fige():
    """Un axe balayé mais jamais transmis produirait 432 runs sur le MÊME
    signal, et un rapport parfaitement calculé sur du vide."""
    import inspect
    source = inspect.getsource(_opt._run_one)
    assert 'combo.get("multiple_hierarchy")' in source
    assert '_DATA["signal_events_par_hierarchie"][hierarchie]' in source


def test_seules_les_hierarchies_de_la_grille_sont_construites():
    """Recombiner 27 000 lignes quatre fois quand une seule est balayée serait
    du calcul pur perdu."""
    import inspect
    source = inspect.getsource(_opt.main)
    assert 'c["multiple_hierarchy"] for c in grid' in source


def test_l_axe_apparait_dans_le_rapport():
    assert "multiple_hierarchy" in _opt.AXES


def test_la_hierarchie_se_lit_en_clair():
    assert "médiane des trois" in _opt._lisible("multiple_hierarchy", "flat")
    assert "sans objet" in _opt._lisible("multiple_hierarchy", None)
    assert "sans objet" in _opt._lisible("multiple_hierarchy", float("nan"))
    assert _opt._lisible("multiple_hierarchy", "pe_first") == "pe_first"


# --------------------------------------------------------------------------- #
# L'horizon, mesuré puis réduit
# --------------------------------------------------------------------------- #
def test_l_horizon_est_reduit_a_sa_valeur_de_production():
    """Mesuré négatif sur toute sa plage (365 j +0,002, 180 j -0,025, 90 j
    -0,073) : même traitement que stop_loss_pct et max_weight_pct, et pour la
    même raison -- un axe dont la réponse est connue coûte un facteur quatre sur
    la grille et relève le plancher de bruit du Sharpe déflaté."""
    import config
    assert _opt.DEFAULT_MAX_HOLDING_GRID == [config.BACKTEST_MAX_HOLDING_DAYS]
    assert config.BACKTEST_MAX_HOLDING_DAYS is None


def test_l_horizon_reste_balayable_a_la_demande():
    grille = _opt._build_grid(_args(max_holding_grid=[-1.0, 365.0]), "valuation_gap_combined")
    assert {c["max_holding_days"] for c in grille} == {None, 365}
