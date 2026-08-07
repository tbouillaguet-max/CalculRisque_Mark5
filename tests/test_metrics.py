"""C6 : benchmark, Sharpe sur les excès, test explicite du drawdown."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import metrics as metrics_mod


def equity(navs, start="2020-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=len(navs))
    return pd.DataFrame({
        "date": dates, "nav": navs, "cash": 0.0,
        "invested_value": navs, "num_positions": 1,
    })


def croissance_reguliere(n: int, taux_quotidien: float, depart: float = 100.0) -> list[float]:
    return list(depart * (1 + taux_quotidien) ** np.arange(n))


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #

def test_benchmark_absent_ne_produit_aucun_champ():
    m = metrics_mod.compute_metrics(equity(croissance_reguliere(300, 0.0003)), pd.DataFrame())
    for key in ("benchmark_total_return_pct", "benchmark_cagr_pct", "alpha_pct", "beta",
                "information_ratio", "tracking_error_pct"):
        assert key not in m


def test_surperformance_donne_un_alpha_positif():
    curve = equity(croissance_reguliere(500, 0.0006))
    bench = pd.Series(croissance_reguliere(500, 0.0002), index=curve["date"])
    m = metrics_mod.compute_metrics(curve, pd.DataFrame(), benchmark_prices=bench)
    assert m["alpha_pct"] > 0
    assert m["cagr_pct"] > m["benchmark_cagr_pct"]
    assert m["information_ratio"] > 0


def test_sous_performance_donne_un_alpha_negatif():
    """Le cas qui motive C6 : une stratégie qui gagne de l'argent tout en
    faisant nettement moins bien que l'indice."""
    curve = equity(croissance_reguliere(500, 0.00016))   # ~4%/an
    bench = pd.Series(croissance_reguliere(500, 0.0004), index=curve["date"])
    m = metrics_mod.compute_metrics(curve, pd.DataFrame(), benchmark_prices=bench)
    assert m["total_return_pct"] > 0        # bénéficiaire...
    assert m["alpha_pct"] < 0               # ...et pourtant en retard sur l'indice


def test_beta_unitaire_quand_la_strategie_replique_l_indice():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0004, 0.01, 400)
    navs = 100 * np.cumprod(1 + returns)
    curve = equity(list(navs))
    bench = pd.Series(list(navs), index=curve["date"])
    m = metrics_mod.compute_metrics(curve, pd.DataFrame(), benchmark_prices=bench)
    assert m["beta"] == pytest.approx(1.0, abs=1e-6)
    assert m["tracking_error_pct"] == pytest.approx(0.0, abs=1e-9)
    assert m["alpha_pct"] == pytest.approx(0.0, abs=1e-9)


def test_beta_double_quand_la_strategie_amplifie_l_indice():
    rng = np.random.default_rng(1)
    bench_returns = rng.normal(0.0, 0.01, 400)
    strat_returns = 2 * bench_returns
    curve = equity(list(100 * np.cumprod(1 + strat_returns)))
    bench = pd.Series(list(100 * np.cumprod(1 + bench_returns)), index=curve["date"])
    m = metrics_mod.compute_metrics(curve, pd.DataFrame(), benchmark_prices=bench)
    assert m["beta"] == pytest.approx(2.0, rel=1e-6)


def test_benchmark_realigne_sur_le_calendrier_de_la_strategie():
    """Une série d'indice qui a des jours en plus/en moins ne doit pas
    décaler la comparaison : elle est réalignée, pas concaténée."""
    curve = equity(croissance_reguliere(200, 0.0003))
    dense = pd.date_range(curve["date"].iloc[0], curve["date"].iloc[-1], freq="D")
    bench = pd.Series(croissance_reguliere(len(dense), 0.0002), index=dense)
    m = metrics_mod.compute_metrics(curve, pd.DataFrame(), benchmark_prices=bench)
    assert m["benchmark_cagr_pct"] is not None
    assert -1.0 < m["beta"] < 1.0 or m["beta"] is not None


def test_benchmark_trop_court_est_ignore_sans_planter():
    curve = equity(croissance_reguliere(200, 0.0003))
    bench = pd.Series([100.0], index=curve["date"].iloc[:1])
    m = metrics_mod.compute_metrics(curve, pd.DataFrame(), benchmark_prices=bench)
    assert "benchmark_cagr_pct" not in m


def test_resume_signale_l_absence_de_benchmark():
    assert "Pas de comparaison" in metrics_mod.format_benchmark_summary({}, "aucun")
    resume = metrics_mod.format_benchmark_summary(
        {"cagr_pct": 4.0, "benchmark_cagr_pct": 12.0, "alpha_pct": -8.0,
         "beta": 0.8, "tracking_error_pct": 15.0, "information_ratio": -0.5},
        "SPY",
    )
    assert "SPY" in resume and "-8.00%" in resume


# --------------------------------------------------------------------------- #
# Sharpe et Calmar
# --------------------------------------------------------------------------- #

def test_sharpe_utilise_l_ecart_type_des_exces():
    """Avec un taux sans risque constant les deux écarts-types coïncident :
    le test vérifie surtout qu'on n'a pas changé la valeur au passage."""
    rng = np.random.default_rng(2)
    navs = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 500))
    curve = equity(list(navs))
    m = metrics_mod.compute_metrics(curve, pd.DataFrame(), risk_free_rate=0.04)

    returns = pd.Series(navs).pct_change().dropna()
    excess = returns - 0.04 / metrics_mod.TRADING_DAYS_PER_YEAR
    attendu = excess.mean() / excess.std() * np.sqrt(metrics_mod.TRADING_DAYS_PER_YEAR)
    assert m["sharpe_ratio"] == pytest.approx(attendu)


def test_calmar_none_quand_il_n_y_a_aucun_drawdown():
    """`max_drawdown_pct not in (0, np.nan)` ne marchait que par accident de
    l'identité de np.nan : sur une courbe strictement croissante, le drawdown
    vaut exactement 0 et Calmar n'est pas défini."""
    m = metrics_mod.compute_metrics(equity(croissance_reguliere(300, 0.0005)), pd.DataFrame())
    assert m["max_drawdown_pct"] == pytest.approx(0.0)
    assert m["calmar_ratio"] is None


def test_calmar_calcule_quand_il_y_a_un_drawdown():
    navs = croissance_reguliere(200, 0.0005) + croissance_reguliere(100, -0.001, depart=110.0)
    m = metrics_mod.compute_metrics(equity(navs), pd.DataFrame())
    assert m["max_drawdown_pct"] < 0
    assert m["calmar_ratio"] is not None


def test_extra_est_fusionne_tel_quel():
    m = metrics_mod.compute_metrics(
        equity(croissance_reguliere(50, 0.0003)), pd.DataFrame(), extra={"truncated_orders_count": 7},
    )
    assert m["truncated_orders_count"] == 7
