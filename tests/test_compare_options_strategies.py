"""Assemblage des tableaux de compare_options_strategies.py.

Le script lui-même rejoue trois backtests complets : il n'est pas testable
ici. Ses deux fonctions d'assemblage, elles, sont pures -- et ce sont elles qui
décident de ce que l'utilisateur lit.
"""

from __future__ import annotations

import pandas as pd
import pytest

import compare_options_strategies as cmp_mod


def _trades(option_types, exit_reasons, pnls) -> pd.DataFrame:
    return pd.DataFrame({
        "option_type": option_types,
        "exit_reason": exit_reasons,
        "pnl": pnls,
        "return_pct": [p / 100.0 for p in pnls],
        "holding_days": [180] * len(pnls),
    })


def test_le_tableau_met_une_colonne_par_strategie():
    tableau = cmp_mod._comparison_table({
        "strategie_a": {"cagr_pct": 12.0, "sharpe_ratio": 0.8, "num_trades": 40},
        "strategie_b": {"cagr_pct": 9.5, "sharpe_ratio": 1.1, "num_trades": 55},
    })

    assert list(tableau.columns) == ["strategie_a", "strategie_b"]
    assert tableau.loc["CAGR %", "strategie_a"] == 12.0
    assert tableau.loc["Sharpe", "strategie_b"] == 1.1


def test_une_metrique_que_personne_ne_renseigne_est_omise():
    """Les compteurs propres à une stratégie ne doivent pas remplir le tableau
    de lignes vides quand cette stratégie n'est pas dans la comparaison."""
    tableau = cmp_mod._comparison_table({
        "strategie_a": {"cagr_pct": 12.0},
        "strategie_b": {"cagr_pct": 9.5},
    })
    assert "Lignes écartées (espérance <= 0)" not in tableau.index
    assert "CAGR %" in tableau.index


def test_une_metrique_renseignee_par_une_seule_strategie_est_conservee():
    """Ne rien mesurer n'est pas mesurer zéro : la ligne reste, avec un trou
    en face des stratégies qui n'exposent pas ce compteur."""
    tableau = cmp_mod._comparison_table({
        "multiples": {"cagr_pct": 9.5},
        "expected_value": {"cagr_pct": 11.0, "dropped_expected_value_negative_count": 317},
    })

    ligne = tableau.loc["Lignes écartées (espérance <= 0)"]
    assert ligne["expected_value"] == 317
    # pandas convertit le None en NaN dans une colonne numérique ; c'est
    # l'affichage (na_rep="-") qui le rend comme un trou, jamais comme un zéro.
    assert pd.isna(ligne["multiples"])


def test_le_pnl_par_motif_est_empile_avec_la_strategie_en_colonne():
    par_motif = cmp_mod._exit_reason_table({
        "multiples": _trades(["CALL", "PUT"], ["expiry", "stop_loss"], [1000.0, -400.0]),
        "expected_value": _trades(["CALL"], ["take_profit"], [2500.0]),
    })

    assert set(par_motif["strategy"]) == {"multiples", "expected_value"}
    assert list(par_motif.columns)[:3] == ["strategy", "option_type", "exit_reason"]
    total = par_motif.groupby("strategy")["pnl_dollar"].sum()
    assert total["multiples"] == pytest.approx(600.0)
    assert total["expected_value"] == pytest.approx(2500.0)


def test_une_strategie_sans_trade_ne_casse_pas_la_decomposition():
    par_motif = cmp_mod._exit_reason_table({
        "vide": pd.DataFrame(),
        "pleine": _trades(["CALL"], ["expiry"], [100.0]),
    })
    assert set(par_motif["strategy"]) == {"pleine"}


def test_aucun_trade_du_tout_rend_un_tableau_vide():
    assert cmp_mod._exit_reason_table({"a": pd.DataFrame(), "b": pd.DataFrame()}).empty


def test_les_trois_strategies_du_depot_sont_bien_celles_comparees_par_defaut():
    from backtest.strategies import OPTIONS_STRATEGY_REGISTRY

    assert set(cmp_mod.DEFAULT_STRATEGIES) == set(OPTIONS_STRATEGY_REGISTRY)
