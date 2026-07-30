"""Métriques de performance et de risque calculées à partir de la sortie de
BacktestEngine.run() (equity_curve, trades). Objectif de 09_backtest.py :
répondre à "cette stratégie est-elle bénéficiaire ?", pas seulement via le
rendement total mais avec de quoi juger la robustesse (drawdown, Sharpe,
Sortino, VaR/CVaR, profit factor...)."""

from __future__ import annotations

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


def compute_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
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
    sharpe = (excess.mean() / daily_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if daily_returns.std() > 0 else np.nan

    downside = daily_returns[daily_returns < 0]
    sortino = (excess.mean() / downside.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside) > 1 and downside.std() > 0 else np.nan

    running_max = equity_curve["nav"].cummax()
    drawdown = equity_curve["nav"] / running_max - 1
    max_drawdown_pct = drawdown.min() * 100
    max_drawdown_duration_days = _max_drawdown_duration_days(equity_curve)
    calmar = (cagr_pct / abs(max_drawdown_pct)) if max_drawdown_pct not in (0, np.nan) and pd.notna(cagr_pct) else np.nan

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

    return metrics
