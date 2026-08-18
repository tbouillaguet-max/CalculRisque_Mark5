"""14_audit_backtest.py : relecture d'un run existant."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
import pytest

audit = importlib.import_module("14_audit_backtest")


@pytest.fixture
def run_dir(tmp_path):
    """Un run synthétique complet, avec la structure exacte que 09 écrit."""
    dates = pd.bdate_range("2015-01-01", periods=760)  # ~3 ans
    nav = pd.Series([1_000_000.0 * (1.0005 ** i) for i in range(len(dates))])

    equity_curve = pd.DataFrame({
        "date": dates, "nav": nav, "cash": 0.0,
        "invested_value": nav, "num_positions": 5,
    })
    trades = pd.DataFrame([
        # Une thèse allégée trois fois pendant sa hausse, puis soldée au stop :
        # 4 exécutions, 3 "gagnantes", pour UN pari perdant.
        {"symbol": "AAA", "entry_date": dates[0], "exit_date": dates[10 * i],
         "shares": 10.0, "entry_price": 100.0, "exit_price": 110.0,
         "pnl": 1_000.0, "return_pct": 10.0, "holding_days": 10 * i,
         "exit_reason": "rebalance"}
        for i in range(1, 4)
    ] + [
        {"symbol": "AAA", "entry_date": dates[0], "exit_date": dates[100],
         "shares": 40.0, "entry_price": 100.0, "exit_price": 85.0,
         "pnl": -6_000.0, "return_pct": -15.0, "holding_days": 140,
         "exit_reason": "stop_loss"},
        {"symbol": "BBB", "entry_date": dates[50], "exit_date": dates[200],
         "shares": 20.0, "entry_price": 50.0, "exit_price": 65.0,
         "pnl": 300.0, "return_pct": 30.0, "holding_days": 210,
         "exit_reason": "take_profit"},
    ])
    signals_history = pd.DataFrame([
        {"date": dates[5], "symbol": "AAA", "sector": "Technologie",
         "fiscal_year": 2014, "gap_pct": 40.0, "valuation_dcf_per_share": 140.0},
        {"date": dates[5], "symbol": "BBB", "sector": "Technologie",
         "fiscal_year": 2014, "gap_pct": 60.0, "valuation_dcf_per_share": 80.0},
    ])

    equity_curve.to_parquet(tmp_path / "equity_curve.parquet", index=False)
    trades.to_parquet(tmp_path / "trades.parquet", index=False)
    signals_history.to_parquet(tmp_path / "signals_history.parquet", index=False)
    (tmp_path / "metrics.json").write_text(json.dumps({
        "cagr_pct": 13.4, "alpha_pct": 4.2, "num_trades": 5,
        "win_rate_pct": 80.0, "profit_factor": 1.4,
    }), encoding="utf-8")
    (tmp_path / "run_config.json").write_text(json.dumps({
        "commission_bps": 5.0, "slippage_bps": 5.0,
    }), encoding="utf-8")
    return tmp_path


def test_le_run_est_relu_sans_recalcul(run_dir):
    run = audit.load_run(run_dir)
    assert len(run["equity_curve"]) == 760
    assert run["run_config"]["slippage_bps"] == 5.0


def test_les_theses_sont_distinguees_des_executions(run_dir):
    from backtest import metrics as metrics_mod

    par_these = metrics_mod._position_level_metrics(audit.load_run(run_dir)["trades"])
    assert par_these["num_positions_closed"] == 2  # AAA et BBB, pas 5 exécutions
    assert par_these["win_rate_positions_pct"] == 50.0


def test_le_decoupage_par_annee_couvre_tout_le_run(run_dir):
    equity_curve = audit.load_run(run_dir)["equity_curve"]
    par_annee = audit.audit_par_annee(equity_curve, benchmark=None)
    assert list(par_annee["annee"]) == [2015, 2016, 2017]
    assert (par_annee["strategie_pct"] > 0).all()


def test_les_sous_periodes_glissantes_avancent_d_un_an(run_dir):
    equity_curve = audit.load_run(run_dir)["equity_curve"]
    fenetres = audit.audit_sous_periodes(equity_curve, benchmark=None, fenetre_annees=2)
    assert len(fenetres) >= 1
    assert (fenetres["strategie_cagr_pct"] > 0).all()


def test_decaler_le_depart_change_le_cagr(run_dir):
    equity_curve = audit.load_run(run_dir)["equity_curve"]
    decalages = audit.audit_date_de_depart(equity_curve, benchmark=None)
    assert list(decalages["annees"]) == sorted(decalages["annees"], reverse=True)
    assert decalages["strategie_cagr_pct"].notna().all()


def test_la_sensibilite_aux_couts_degrade_le_cagr(run_dir):
    run = audit.load_run(run_dir)
    couts = audit.audit_couts(run)
    assert couts["cout_aller_simple_bps"].iloc[0] == 10.0
    assert couts["surcout_total_dollar"].iloc[0] == 0.0  # le coût du run lui-même
    assert couts["surcout_total_dollar"].is_monotonic_increasing
    assert couts["cagr_approx_pct"].is_monotonic_decreasing


def test_la_couverture_mesure_actuels_et_anciens_membres(run_dir, monkeypatch):
    """AAA est encore membre (end_date absente), CCC est sortie en 2016 et n'a
    jamais produit de signal : c'est la signature du biais de survivance."""
    history = pd.DataFrame([
        {"ric": "AAA", "start_date": pd.Timestamp("2010-01-01"), "end_date": pd.NaT},
        {"ric": "BBB", "start_date": pd.Timestamp("2010-01-01"), "end_date": pd.NaT},
        {"ric": "CCC", "start_date": pd.Timestamp("2010-01-01"), "end_date": pd.Timestamp("2016-06-01")},
    ])
    monkeypatch.setattr(audit.data_loader, "load_universe_history", lambda *a, **k: history)

    run = audit.load_run(run_dir)
    couverture = audit.audit_signal_coverage(run["signals_history"], run["equity_curve"])

    ligne_2015 = couverture[couverture["annee"] == 2015].iloc[0]
    assert ligne_2015["membres"] == 3
    assert ligne_2015["avec_signal"] == 2
    assert ligne_2015["couverture_membres_actuels"] == 1.0
    assert ligne_2015["couverture_anciens_membres"] == 0.0

    # CCC sortie mi-2016 : elle ne compte plus dans l'univers de fin 2016.
    ligne_2016 = couverture[couverture["annee"] == 2016].iloc[0]
    assert ligne_2016["membres"] == 2
    assert ligne_2016["couverture"] == 1.0
    assert np.isnan(ligne_2016["couverture_anciens_membres"])


def test_sans_historique_d_univers_la_couverture_n_est_pas_inventee(run_dir, monkeypatch):
    monkeypatch.setattr(audit.data_loader, "load_universe_history", lambda *a, **k: None)
    run = audit.load_run(run_dir)
    assert audit.audit_signal_coverage(run["signals_history"], run["equity_curve"]) is None


def test_le_run_le_plus_recent_est_choisi_par_defaut(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.config, "DIR_BACKTEST", tmp_path)
    for nom in ("20200101_000000", "20260816_000429"):
        (tmp_path / nom).mkdir()
        pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "nav": [1.0]}).to_parquet(
            tmp_path / nom / "equity_curve.parquet", index=False,
        )
    assert audit.resolve_run_dir(None, None).name == "20260816_000429"
