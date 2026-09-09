"""Décomposition CALL / PUT : l'identité de réconciliation, et le fait que
chaque tableau isole bien la jambe qu'il prétend isoler."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest import put_call_analysis as pca
from tests.options_harness import cours, moteur, serie_bruitee, signaux

N_JOURS = 250


@pytest.fixture
def run_reel():
    """Un vrai run du moteur, avec les deux jambes ouvertes : c'est le seul
    moyen de tester la réconciliation contre les conventions de comptabilité
    RÉELLES du moteur (commission incorporée à entry_premium, ventes non
    passées, expirations) plutôt que contre l'idée qu'on s'en fait."""
    panel = cours({
        "AAA": serie_bruitee(100.0, N_JOURS, graine=1),
        "BBB": serie_bruitee(80.0, N_JOURS, graine=2),
        "CCC": serie_bruitee(120.0, N_JOURS, graine=3),
    })
    evenements = signaux(["AAA", "BBB", "CCC"], "2020-01-06")
    engine = moteur(
        panel, evenements, {"AAA": "CALL", "BBB": "PUT", "CCC": "CALL"},
        min_deployment_pct=60.0, target_tenor_days=180,
    )
    equity_curve, positions_history, trades, _ = engine.run()
    return equity_curve, positions_history, trades


# --------------------------------------------------------------------------- #
# Réconciliation : NAV = capital initial + réalisé + latent
# --------------------------------------------------------------------------- #

def test_les_deux_jambes_reconstituent_le_nav_du_moteur(run_reel):
    """Toute la décomposition repose sur cette identité. Si le moteur changeait
    de convention (commission comptée hors entry_premium, par exemple), les
    tableaux CALL/PUT deviendraient faux en silence -- d'où ce test."""
    equity_curve, positions_history, trades = run_reel
    residual = pca.reconciliation_residual(equity_curve, positions_history, trades)
    initial = equity_curve["nav"].iloc[0]
    assert residual["max_abs_residual_dollar"] < initial * 1e-6, residual


def test_les_pnl_des_deux_jambes_s_additionnent_au_total(run_reel):
    equity_curve, positions_history, trades = run_reel
    per_side = pca.side_metrics(equity_curve, positions_history, trades)
    somme = per_side["CALL"]["leg_pnl_dollar"] + per_side["PUT"]["leg_pnl_dollar"]
    assert somme == pytest.approx(per_side["ALL"]["leg_pnl_dollar"], rel=1e-9, abs=1e-3)


def test_le_run_de_reference_ouvre_bien_les_deux_jambes(run_reel):
    """Sans ça, les tests ci-dessus passeraient sur un run à une seule jambe
    et ne prouveraient rien de la séparation."""
    _, positions_history, trades = run_reel
    assert set(positions_history["option_type"]) == {"CALL", "PUT"}
    assert set(trades["option_type"]) == {"CALL", "PUT"}


# --------------------------------------------------------------------------- #
# Isolation des jambes
# --------------------------------------------------------------------------- #

def _curve(navs, start="2020-01-01"):
    return pd.DataFrame({
        "date": pd.bdate_range(start=start, periods=len(navs)),
        "nav": navs, "cash": 0.0, "invested_value": 0.0, "num_positions": 0,
    })


def _trade(option_type, exit_date, pnl, entry_price=5.0, shares=100.0, reason="expiry"):
    return {
        "symbol": "AAA", "entry_date": pd.Timestamp("2020-01-01"), "exit_date": pd.Timestamp(exit_date),
        "shares": shares, "entry_price": entry_price, "exit_price": entry_price + pnl / shares,
        "pnl": pnl, "return_pct": pnl / (entry_price * shares) * 100,
        "holding_days": 30, "exit_reason": reason, "option_type": option_type,
        "strike": 100.0, "contracts": shares / 100.0, "source": "simulated",
    }


def test_une_jambe_perdante_n_est_pas_masquee_par_l_autre():
    """Le cas qui motive tout le module : un total positif qui cache une jambe
    profondément négative."""
    equity_curve = _curve([1000.0] * 5 + [1100.0] * 5)
    trades = pd.DataFrame([
        _trade("CALL", "2020-01-08", +400.0),
        _trade("PUT", "2020-01-08", -300.0),
    ])
    per_side = pca.side_metrics(equity_curve, pd.DataFrame(), trades)

    assert per_side["ALL"]["leg_pnl_dollar"] == pytest.approx(100.0)
    assert per_side["CALL"]["leg_pnl_dollar"] == pytest.approx(400.0)
    assert per_side["PUT"]["leg_pnl_dollar"] == pytest.approx(-300.0)
    assert per_side["PUT"]["total_return_pct"] < 0 < per_side["CALL"]["total_return_pct"]


def test_une_jambe_absente_ne_fait_pas_planter_l_analyse():
    equity_curve = _curve([1000.0] * 5 + [1400.0] * 5)
    trades = pd.DataFrame([_trade("CALL", "2020-01-08", +400.0)])
    per_side = pca.side_metrics(equity_curve, pd.DataFrame(), trades)

    assert per_side["PUT"]["num_trades"] == 0
    # Jambe vide = courbe strictement plate au capital initial, donc aucun
    # rendement : elle ne doit ni disparaître du tableau ni inventer un chiffre.
    assert per_side["PUT"]["total_return_pct"] == pytest.approx(0.0)
    assert pca.metrics_table(per_side).loc["leg_pnl_dollar", "PUT"] == pytest.approx(0.0)


def test_une_sortie_hors_calendrier_ne_decale_pas_le_pnl_realise():
    """Le reindex naïf d'une série de P&L réalisé sur le calendrier de
    l'equity_curve écrase silencieusement les dates absentes -- ici un samedi,
    qu'aucun bdate_range ne contient."""
    dates = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=10))
    trades = pd.DataFrame([_trade("CALL", "2020-01-04", +500.0)])   # un samedi
    curve = pca.side_equity_curve(dates, trades, pd.DataFrame(), initial_capital=1000.0)
    assert curve["nav"].iloc[-1] == pytest.approx(1500.0)


# --------------------------------------------------------------------------- #
# Tableaux de répartition
# --------------------------------------------------------------------------- #

def test_la_repartition_des_volumes_somme_a_cent_pourcent(run_reel):
    _, _, trades = run_reel
    volumes = pca.volume_breakdown(trades)
    assert volumes.loc["num_trades", "ALL"] == len(trades)
    assert volumes.loc["num_trades_pct", "CALL"] + volumes.loc["num_trades_pct", "PUT"] == pytest.approx(100.0)
    assert volumes.loc["turnover_dollar_pct", "CALL"] + volumes.loc["turnover_dollar_pct", "PUT"] == pytest.approx(100.0)
    # La colonne de référence porte 100% et non un trou (cf. volume_breakdown).
    assert volumes.loc["num_trades_pct", "ALL"] == pytest.approx(100.0)


def test_le_pnl_des_volumes_est_realise_seulement(run_reel):
    """`realized_pnl_dollar` (positions clôturées) et `leg_pnl_dollar`
    (réalisé + latent) ne mesurent pas la même chose : le test fixe la
    différence pour qu'un renommage ne les confonde pas."""
    equity_curve, positions_history, trades = run_reel
    volumes = pca.volume_breakdown(trades)
    assert volumes.loc["realized_pnl_dollar", "ALL"] == pytest.approx(trades["pnl"].sum())

    latent = positions_history[positions_history["date"] == positions_history["date"].max()]
    attendu = trades["pnl"].sum() + latent["unrealized_pnl"].fillna(0.0).sum()
    per_side = pca.side_metrics(equity_curve, positions_history, trades)
    assert per_side["ALL"]["leg_pnl_dollar"] == pytest.approx(attendu, abs=1e-3)


def test_la_repartition_des_valorisations_moyenne_sur_tous_les_jours():
    """Une jambe ouverte un jour sur deux à 100$ pèse 50$ en moyenne, pas
    100$ : la moyenne porte sur le calendrier, pas sur les jours d'exposition."""
    positions = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "symbol": "AAA", "option_type": "CALL",
         "market_value": 100.0, "unrealized_pnl": 0.0},
        {"date": pd.Timestamp("2020-01-02"), "symbol": "BBB", "option_type": "PUT",
         "market_value": 100.0, "unrealized_pnl": 0.0},
    ])
    breakdown = pca.valuation_breakdown(positions)
    assert breakdown.loc["avg_market_value_dollar", "CALL"] == pytest.approx(50.0)
    assert breakdown.loc["max_market_value_dollar", "CALL"] == pytest.approx(100.0)
    assert breakdown.loc["pct_of_days_with_exposure", "CALL"] == pytest.approx(50.0)
    assert breakdown.loc["pct_of_avg_invested", "CALL"] == pytest.approx(50.0)


def test_le_pnl_par_motif_de_sortie_separe_les_jambes():
    trades = pd.DataFrame([
        _trade("CALL", "2020-01-08", +400.0, reason="take_profit"),
        _trade("PUT", "2020-01-08", -300.0, reason="stop_loss"),
        _trade("PUT", "2020-01-09", -200.0, reason="stop_loss"),
    ])
    table = pca.pnl_by_exit_reason(trades).set_index(["option_type", "exit_reason"])
    assert table.loc[("PUT", "stop_loss"), "pnl_dollar"] == pytest.approx(-500.0)
    assert table.loc[("PUT", "stop_loss"), "num_trades"] == 2
    assert table.loc[("CALL", "take_profit"), "pnl_dollar"] == pytest.approx(400.0)


def test_le_pnl_annuel_ventile_par_annee_de_sortie():
    trades = pd.DataFrame([
        _trade("CALL", "2020-06-01", +100.0),
        _trade("PUT", "2020-06-01", -40.0),
        _trade("PUT", "2021-06-01", -60.0),
    ])
    table = pca.annual_pnl_by_side(trades)
    assert table.loc[2020, "CALL"] == pytest.approx(100.0)
    assert table.loc[2020, "PUT"] == pytest.approx(-40.0)
    assert table.loc[2020, "TOTAL"] == pytest.approx(60.0)
    assert table.loc[2021, "TOTAL"] == pytest.approx(-60.0)


def test_la_contribution_quotidienne_finit_sur_le_pnl_total(run_reel):
    equity_curve, positions_history, trades = run_reel
    serie = pca.pnl_contribution_over_time(equity_curve, positions_history, trades)
    derniere = serie.iloc[-1]
    assert derniere["CALL_pnl_cumul"] + derniere["PUT_pnl_cumul"] == pytest.approx(
        derniere["total_pnl_cumul"], abs=1e-3,
    )
