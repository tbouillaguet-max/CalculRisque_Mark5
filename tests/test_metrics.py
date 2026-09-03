"""C6 : benchmark, Sharpe sur les excès, test explicite du drawdown."""

from __future__ import annotations

import math

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
    """Sharpe = moyenne des EXCÈS / écart-type des EXCÈS, le taux sans risque
    étant pris année par année (config.RISK_FREE_RATE_BY_YEAR)."""
    import config

    rng = np.random.default_rng(2)
    navs = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 500))
    curve = equity(list(navs))
    m = metrics_mod.compute_metrics(curve, pd.DataFrame(), risk_free_rate=0.04)

    returns = pd.Series(navs).pct_change().dropna()
    rf = pd.Series(
        [config.risk_free_rate_for(d.year) / metrics_mod.TRADING_DAYS_PER_YEAR
         for d in curve["date"].iloc[1:]],
        index=returns.index,
    )
    excess = returns - rf
    attendu = excess.mean() / excess.std() * np.sqrt(metrics_mod.TRADING_DAYS_PER_YEAR)
    assert m["sharpe_ratio"] == pytest.approx(attendu)


def test_sharpe_repli_sur_la_constante_hors_table(monkeypatch):
    """Sans courbe annuelle disponible, le comportement d'avant est conservé :
    taux constant appliqué à toute la période."""
    import config

    monkeypatch.setattr(config, "RISK_FREE_RATE_BY_YEAR", {}, raising=False)
    rng = np.random.default_rng(3)
    navs = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300))
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


# --------------------------------------------------------------------------- #
# Sharpe déflaté du nombre d'essais
# --------------------------------------------------------------------------- #

def _rendements(
    n: int, sharpe_annualise: float, ecart_type: float = 0.012, graine: int = 0,
) -> pd.Series:
    """Série dont le Sharpe annualisé vaut EXACTEMENT la cible.

    Standardisée plutôt que simplement tirée : un tirage de 2 520 points autour
    d'une moyenne donnée rend un Sharpe d'échantillon qui s'en écarte largement
    (mesuré : 1,10 pour une cible de 1,59), et un test qui asserte un seuil
    devient alors une loterie de graine plutôt qu'une vérification."""
    brut = pd.Series(np.random.default_rng(graine).normal(0.0, 1.0, n))
    brut = (brut - brut.mean()) / brut.std()
    return brut * ecart_type + sharpe_annualise / math.sqrt(252) * ecart_type


def test_le_quantile_normal_est_exact():
    """_norm_ppf est une approximation rationnelle : on vérifie qu'elle vaut
    bien la fonction qu'elle approche, plutôt que de la croire sur parole."""
    from scipy.stats import norm
    for p in (0.001, 0.025, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999):
        assert metrics_mod._norm_ppf(p) == pytest.approx(norm.ppf(p), abs=1e-8)


def test_un_seul_essai_n_a_pas_de_plancher():
    """Sans sélection, il n'y a pas de biais de sélection à retirer."""
    assert metrics_mod.expected_maximum_sharpe(1, 0.05) == 0.0
    assert metrics_mod.expected_maximum_sharpe(0, 0.05) == 0.0


def test_le_plancher_de_bruit_croit_avec_le_nombre_d_essais():
    """Le cœur du problème : le maximum de N tirages n'est pas nul même quand
    la vraie performance l'est, et il croît avec N."""
    ecart_type = 1 / math.sqrt(252 * 10 - 1)
    planchers = [metrics_mod.expected_maximum_sharpe(n, ecart_type) for n in (8, 16, 64, 200)]
    assert planchers == sorted(planchers)
    assert all(p > 0 for p in planchers)

    # Annualisé, pour se lire à côté d'un sharpe_ratio : sur 10 ans, une
    # grille de 64 combinaisons (8 stops x 8 take-profits, cf.
    # 11_optimize_options_stops.py) offre déjà 0,75 de Sharpe gratuit.
    assert metrics_mod.expected_maximum_sharpe(64, ecart_type) * math.sqrt(252) == pytest.approx(0.75, abs=0.02)


def test_une_strategie_sans_edge_ne_survit_pas_a_la_deflation():
    """Rendements purement aléatoires : le Sharpe obtenu est du bruit, et le
    Sharpe déflaté doit le dire dès que plusieurs essais ont eu lieu."""
    bruit = _rendements(252 * 10, sharpe_annualise=0.05)

    assert metrics_mod.deflated_sharpe_ratio(bruit, n_trials=64) < 0.05
    assert metrics_mod.deflated_sharpe_ratio(bruit, n_trials=200) < 0.05


def test_la_deflation_est_monotone_en_nombre_d_essais():
    """Plus on a essayé de configurations, moins le même résultat est probant.
    C'est toute la propriété qu'on attend du chiffre."""
    serie = _rendements(252 * 10, sharpe_annualise=1.2)
    scores = [metrics_mod.deflated_sharpe_ratio(serie, n_trials=n) for n in (1, 8, 64, 200)]
    assert scores == sorted(scores, reverse=True)


def test_une_vraie_performance_survit_a_la_deflation():
    """Contre-épreuve indispensable : le critère doit pouvoir dire OUI, sinon
    il ne discrimine rien. Sharpe annualisé de 1,6 sur 10 ans, très au-dessus
    du plancher de bruit de 0,75 que valent 64 essais sur cette durée."""
    solide = _rendements(252 * 10, sharpe_annualise=1.6)
    assert metrics_mod.deflated_sharpe_ratio(solide, n_trials=64) > 0.95


def test_l_asymetrie_et_les_queues_penalisent_le_sharpe_probabiliste():
    """La correction de Mertens : à Sharpe égal, une distribution à queue
    gauche épaisse -- ce qu'est un portefeuille d'options longues -- mérite
    moins de confiance qu'une gaussienne."""
    rng = np.random.default_rng(3)
    n = 252 * 8
    gaussien = pd.Series(rng.normal(0.0008, 0.012, n))
    # Même moyenne et même écart-type, mais asymétrie négative marquée.
    asymetrique = pd.Series(-rng.gumbel(0.0, 1.0, n))
    asymetrique = (asymetrique - asymetrique.mean()) / asymetrique.std()
    asymetrique = asymetrique * gaussien.std() + gaussien.mean()

    assert asymetrique.skew() < -0.5
    assert metrics_mod.probabilistic_sharpe_ratio(asymetrique) < metrics_mod.probabilistic_sharpe_ratio(gaussien)


def test_les_metriques_portent_le_nombre_d_essais():
    """metrics.json doit garder la trace du nombre d'essais : sans lui, un
    Sharpe issu d'un grid-search est indiscernable d'un Sharpe de run isolé."""
    courbe = equity(croissance_reguliere(252 * 6, 0.0004))
    resultat = metrics_mod.compute_metrics(courbe, pd.DataFrame(), n_trials=64)
    assert resultat["n_trials"] == 64
    assert resultat["sharpe_noise_floor"] > 0

    isole = metrics_mod.compute_metrics(courbe, pd.DataFrame())
    assert isole["n_trials"] == 1
    assert isole["sharpe_noise_floor"] == pytest.approx(0.0)
    # À performance identique, plus d'essais = moins de confiance.
    assert resultat["deflated_sharpe_ratio"] <= isole["deflated_sharpe_ratio"]
