"""Métriques de performance et de risque calculées à partir de la sortie de
BacktestEngine.run() (equity_curve, trades). Objectif de 09_backtest.py :
répondre à "cette stratégie est-elle bénéficiaire ?", pas seulement via le
rendement total mais avec de quoi juger la robustesse (drawdown, Sharpe,
Sortino, VaR/CVaR, profit factor...).

"Bénéficiaire" ne suffit pas : un CAGR de 4% sur 2015-2026 est une
SOUS-PERFORMANCE massive, et aucune des métriques ci-dessus ne le dit. D'où
`benchmark_prices` (voir compute_metrics) : tant qu'une stratégie n'est pas
comparée à ce qu'aurait rapporté le fait de ne rien décider, ses chiffres ne
se lisent pas."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _max_drawdown_duration_days(equity_curve: pd.DataFrame) -> int:
    running_max = equity_curve["nav"].cummax()
    in_drawdown = equity_curve["nav"] < running_max
    if not in_drawdown.any():
        return 0
    # Longueur (en jours calendaires) de la plus longue séquence continue en drawdown.
    group_id = (~in_drawdown).cumsum()
    durations = equity_curve.loc[in_drawdown].groupby(group_id[in_drawdown])["date"].agg(lambda d: (d.max() - d.min()).days + 1)
    return int(durations.max()) if not durations.empty else 0


def _benchmark_metrics(equity_curve: pd.DataFrame, benchmark_prices: pd.Series, n_years: float) -> dict:
    """Comparaison à un indice de référence, sur EXACTEMENT le même
    calendrier que la courbe de NAV : la série de l'indice est réalignée sur
    les dates de l'equity_curve (report de la dernière clôture connue), sinon
    un jour férié d'un côté seulement décalerait toute la série de rendements
    et le beta comme le tracking error deviendraient du bruit.

    Retourne {} si l'indice ne couvre pas réellement la période -- mieux vaut
    pas de benchmark qu'un benchmark faux. Le décompte porte sur les
    observations RÉELLES tombant dans la fenêtre, pas sur la série remplie :
    une série qui s'arrête au premier jour serait reportée en ligne plate sur
    tout le run et afficherait un rendement de 0%, ce qui se lirait comme une
    surperformance imaginaire au lieu d'une absence de donnée."""
    dates = pd.DatetimeIndex(equity_curve["date"])
    observed = pd.to_numeric(benchmark_prices, errors="coerce").dropna()
    in_window = observed[(observed.index >= dates.min()) & (observed.index <= dates.max())]
    if len(in_window) < 2:
        return {}

    aligned = pd.to_numeric(
        observed.reindex(observed.index.union(dates)).ffill().reindex(dates), errors="coerce",
    )
    if aligned.notna().sum() < 2:
        return {}

    bench_start, bench_end = aligned.dropna().iloc[0], aligned.dropna().iloc[-1]
    if not bench_start or bench_start <= 0:
        return {}

    bench_total_return_pct = (bench_end / bench_start - 1) * 100
    bench_cagr_pct = (
        ((bench_end / bench_start) ** (1 / n_years) - 1) * 100
        if n_years and n_years > 0 else np.nan
    )

    # Rendements quotidiens appariés position par position : les deux séries
    # sont déjà sur le même calendrier (réalignement ci-dessus), on ne garde
    # que les jours où les DEUX sont définis (premier jour, trous de
    # couverture de l'indice).
    paired = pd.DataFrame({
        "strategy": equity_curve["nav"].pct_change().to_numpy(),
        "benchmark": aligned.pct_change().to_numpy(),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    metrics = {
        "benchmark_total_return_pct": float(bench_total_return_pct),
        "benchmark_cagr_pct": float(bench_cagr_pct) if pd.notna(bench_cagr_pct) else None,
    }

    strategy_cagr_pct = (
        ((equity_curve["nav"].iloc[-1] / equity_curve["nav"].iloc[0]) ** (1 / n_years) - 1) * 100
        if n_years and n_years > 0 else np.nan
    )
    # alpha = excès de CAGR, pas l'alpha de Jensen : c'est la question posée
    # ("combien ai-je gagné de plus qu'en achetant l'indice ?"), et elle se lit
    # sans avoir à supposer un modèle de marché.
    metrics["alpha_pct"] = (
        float(strategy_cagr_pct - bench_cagr_pct)
        if pd.notna(strategy_cagr_pct) and pd.notna(bench_cagr_pct) else None
    )

    if len(paired) > 1:
        bench_var = paired["benchmark"].var()
        beta = paired["strategy"].cov(paired["benchmark"]) / bench_var if bench_var > 0 else np.nan
        active = paired["strategy"] - paired["benchmark"]
        tracking_error = active.std()
        metrics.update({
            "beta": float(beta) if pd.notna(beta) else None,
            "tracking_error_pct": float(tracking_error * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)
            if pd.notna(tracking_error) else None,
            "information_ratio": float(active.mean() / tracking_error * np.sqrt(TRADING_DAYS_PER_YEAR))
            if pd.notna(tracking_error) and tracking_error > 0 else None,
        })
    return metrics


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.0,
    benchmark_prices: Optional[pd.Series] = None,
    extra: Optional[dict] = None,
) -> dict:
    """benchmark_prices : clôtures quotidiennes de l'indice de référence
    (Series indexée par date). Absent -> les champs benchmark_*, alpha_pct,
    beta, information_ratio et tracking_error_pct ne sont simplement pas
    produits.

    `extra` : métriques calculées par le moteur lui-même (ordres tronqués,
    exposition delta...), fusionnées telles quelles dans le dictionnaire de
    sortie -- elles n'ont pas leur place ici, mais elles ont leur place dans
    metrics.json."""
    if equity_curve.empty:
        return {}
    equity_curve = equity_curve.sort_values("date").reset_index(drop=True)
    daily_returns = equity_curve["nav"].pct_change().dropna()

    nav_start, nav_end = equity_curve["nav"].iloc[0], equity_curve["nav"].iloc[-1]
    n_days = (equity_curve["date"].iloc[-1] - equity_curve["date"].iloc[0]).days
    n_years = n_days / 365.25 if n_days > 0 else np.nan

    total_return_pct = (nav_end / nav_start - 1) * 100
    cagr_pct = ((nav_end / nav_start) ** (1 / n_years) - 1) * 100 if n_years and n_years > 0 else np.nan

    ann_vol_pct = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100 if len(daily_returns) > 1 else np.nan
    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_returns - rf_daily
    # Sharpe = moyenne des EXCÈS / écart-type des EXCÈS. L'ancienne version
    # divisait par l'écart-type des rendements BRUTS : le taux sans risque
    # étant une constante, les deux écarts-types coïncident tant que le taux
    # est constant, mais plus dès que RISK_FREE_RATE_BY_YEAR le fait varier
    # (cf. C7). Autant écrire la définition.
    excess_std = excess.std()
    sharpe = (excess.mean() / excess_std) * np.sqrt(TRADING_DAYS_PER_YEAR) if excess_std > 0 else np.nan

    downside = daily_returns[daily_returns < 0]
    sortino = (excess.mean() / downside.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside) > 1 and downside.std() > 0 else np.nan

    running_max = equity_curve["nav"].cummax()
    drawdown = equity_curve["nav"] / running_max - 1
    max_drawdown_pct = drawdown.min() * 100
    max_drawdown_duration_days = _max_drawdown_duration_days(equity_curve)
    # `max_drawdown_pct not in (0, np.nan)` ne fonctionnait que par accident :
    # `in` teste d'abord l'IDENTITÉ, or np.nan est un singleton -- le test
    # "n'est pas NaN" passait donc pour le np.nan du module numpy et pour lui
    # seul, pas pour un NaN calculé. Test explicite.
    calmar = (
        cagr_pct / abs(max_drawdown_pct)
        if pd.notna(max_drawdown_pct) and max_drawdown_pct != 0 and pd.notna(cagr_pct)
        else np.nan
    )

    var_95_pct = daily_returns.quantile(0.05) * 100 if len(daily_returns) > 1 else np.nan
    cvar_95_pct = daily_returns[daily_returns <= daily_returns.quantile(0.05)].mean() * 100 if len(daily_returns) > 1 else np.nan

    metrics = {
        "start_date": str(equity_curve["date"].iloc[0].date()),
        "end_date": str(equity_curve["date"].iloc[-1].date()),
        "initial_nav": float(nav_start),
        "final_nav": float(nav_end),
        "total_return_pct": float(total_return_pct),
        "cagr_pct": float(cagr_pct) if pd.notna(cagr_pct) else None,
        "annualized_volatility_pct": float(ann_vol_pct) if pd.notna(ann_vol_pct) else None,
        "sharpe_ratio": float(sharpe) if pd.notna(sharpe) else None,
        "sortino_ratio": float(sortino) if pd.notna(sortino) else None,
        "max_drawdown_pct": float(max_drawdown_pct) if pd.notna(max_drawdown_pct) else None,
        "max_drawdown_duration_days": max_drawdown_duration_days,
        "calmar_ratio": float(calmar) if pd.notna(calmar) else None,
        "var_95_daily_pct": float(var_95_pct) if pd.notna(var_95_pct) else None,
        "cvar_95_daily_pct": float(cvar_95_pct) if pd.notna(cvar_95_pct) else None,
        "skewness_daily_returns": float(daily_returns.skew()) if len(daily_returns) > 2 else None,
        "kurtosis_daily_returns": float(daily_returns.kurt()) if len(daily_returns) > 3 else None,
        "best_day_pct": float(daily_returns.max() * 100) if len(daily_returns) > 0 else None,
        "worst_day_pct": float(daily_returns.min() * 100) if len(daily_returns) > 0 else None,
        "avg_num_positions": float(equity_curve["num_positions"].mean()),
        "max_num_positions": int(equity_curve["num_positions"].max()),
        "avg_exposure_pct": float((equity_curve["invested_value"] / equity_curve["nav"]).mean() * 100),
    }

    if trades is not None and not trades.empty:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]
        gross_profit = wins["pnl"].sum()
        gross_loss = -losses["pnl"].sum()
        turnover_dollar = (trades["shares"] * trades["exit_price"]).sum() + (trades["shares"] * trades["entry_price"]).sum()

        metrics.update({
            "num_trades": int(len(trades)),
            "win_rate_pct": float(len(wins) / len(trades) * 100),
            "avg_win_pct": float(wins["return_pct"].mean()) if not wins.empty else None,
            "avg_loss_pct": float(losses["return_pct"].mean()) if not losses.empty else None,
            "avg_trade_return_pct": float(trades["return_pct"].mean()),
            "best_trade_pct": float(trades["return_pct"].max()),
            "worst_trade_pct": float(trades["return_pct"].min()),
            "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else None,
            "avg_holding_days": float(trades["holding_days"].mean()),
            "exits_by_reason": trades["exit_reason"].value_counts().to_dict(),
            "annualized_turnover_pct": float(turnover_dollar / equity_curve["nav"].mean() / n_years * 100) if n_years and n_years > 0 else None,
        })
    else:
        metrics.update({"num_trades": 0})

    if benchmark_prices is not None and not benchmark_prices.empty:
        metrics.update(_benchmark_metrics(equity_curve, benchmark_prices, n_years))

    if extra:
        metrics.update(extra)

    return metrics


def format_benchmark_summary(metrics: dict, benchmark_label: str) -> str:
    """Ligne de résumé de fin de run. Sortie explicite quand l'indice n'a pas
    pu être construit : l'absence de comparaison doit se voir, pas se
    deviner à l'absence de la ligne."""
    if "benchmark_cagr_pct" not in metrics:
        return (
            "Pas de comparaison à un indice de référence sur ce run "
            "(ni SPY ni univers point-in-time exploitable dans les cours quotidiens)."
        )

    def pct(key: str) -> str:
        value = metrics.get(key)
        return "n/a" if value is None else f"{value:+.2f}%"

    def ratio(key: str) -> str:
        value = metrics.get(key)
        return "n/a" if value is None else f"{value:.2f}"

    return (
        f"Comparaison à {benchmark_label} : CAGR stratégie {pct('cagr_pct')} vs "
        f"{pct('benchmark_cagr_pct')} -> alpha {pct('alpha_pct')} "
        f"(beta {ratio('beta')}, tracking error {pct('tracking_error_pct')}, "
        f"information ratio {ratio('information_ratio')})."
    )
