"""
Décomposition CALL / PUT des résultats d'un backtest options.

Le moteur (backtest/options_engine.py) publie un NAV agrégé : il dit combien
la stratégie a gagné, jamais lequel des deux paris y a contribué. Or les deux
jambes n'ont ni le même profil ni la même raison d'exister -- un call parie
sur une convergence à la hausse, un put sur une baisse d'un titre survalorisé,
et sur un marché haussier de long terme la seconde peut saigner en silence
pendant que la première la masque. Ce module sépare les deux.

    contribution d'une jambe = P&L RÉALISÉ cumulé + P&L LATENT du jour

Cette identité est exacte, pas approchée. Le moteur débite à l'entrée
exactement `contracts x multiplier x entry_premium` (la commission d'ordre et
le slippage sont incorporés à `entry_premium`, cf. options_engine
`_open_or_reinforce_position`) et crédite à la sortie exactement
`contracts x multiplier x exit_price` (commission déduite, cf.
`_reduce_position`). En sommant sur les deux jambes on retrouve donc le NAV
du moteur au centime près :

    NAV = capital initial + P&L réalisé total + P&L latent total

`reconciliation_residual` vérifie cette égalité sur toute la période plutôt
que de la supposer : un écart matériel signalerait que le moteur a changé de
convention et que la décomposition ci-dessous ne se lit plus.

COURBE SYNTHÉTIQUE PAR JAMBE. Pour calculer un Sharpe ou un drawdown par
jambe il faut une courbe de NAV, pas une courbe de P&L. Chaque jambe est donc
rejouée comme un portefeuille partant du capital initial COMPLET et ne
recevant que le P&L de cette jambe. Ce n'est PAS "ce qu'aurait donné une
stratégie qui n'achète que des calls" -- le dimensionnement aurait été tout
autre, le capital disponible aussi. C'est la CONTRIBUTION de la jambe,
exprimée dans une unité qui se compare à celle des autres jambes et à
l'agrégat.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from backtest import metrics as metrics_mod

logger = logging.getLogger("backtest.put_call_analysis")

SIDES = ("CALL", "PUT")

# Colonnes attendues par metrics_mod.compute_metrics dans une equity_curve.
_CURVE_COLUMNS = ["date", "nav", "invested_value", "num_positions"]


def _as_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def side_equity_curve(
    dates: pd.DatetimeIndex,
    trades_side: pd.DataFrame,
    positions_side: pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:
    """Courbe de NAV synthétique d'une jambe (voir la docstring du module).

    `dates` est le calendrier de l'equity_curve du moteur : la courbe produite
    porte exactement les mêmes dates, sans quoi aucune métrique ne serait
    comparable à l'agrégat."""
    realized = pd.Series(0.0, index=dates)
    if not trades_side.empty:
        by_exit = trades_side.groupby(_as_datetime(trades_side["exit_date"]))["pnl"].sum()
        # Union avant cumsum : une date de sortie hors calendrier serait sinon
        # silencieusement écrasée par le reindex, et tout le P&L réalisé
        # postérieur s'en trouverait décalé.
        index = dates.union(pd.DatetimeIndex(by_exit.index))
        realized = by_exit.reindex(index, fill_value=0.0).cumsum().reindex(dates)

    if positions_side.empty:
        unrealized = pd.Series(0.0, index=dates)
        market_value = pd.Series(0.0, index=dates)
        num_positions = pd.Series(0, index=dates)
    else:
        grouped = positions_side.assign(date=_as_datetime(positions_side["date"])).groupby("date")
        # unrealized_pnl est None quand aucune prime n'a pu être calculée ce
        # jour-là : le moteur marque alors la position à son prix de revient,
        # ce qui vaut bien un latent nul.
        unrealized = grouped["unrealized_pnl"].sum(min_count=0).reindex(dates, fill_value=0.0).fillna(0.0)
        market_value = grouped["market_value"].sum(min_count=0).reindex(dates, fill_value=0.0).fillna(0.0)
        num_positions = grouped["symbol"].count().reindex(dates, fill_value=0)

    return pd.DataFrame({
        "date": dates,
        "nav": initial_capital + realized.to_numpy() + unrealized.to_numpy(),
        "invested_value": market_value.to_numpy(),
        "num_positions": num_positions.to_numpy(),
        "realized_pnl_cumul": realized.to_numpy(),
        "unrealized_pnl": unrealized.to_numpy(),
    }).reset_index(drop=True)


def reconciliation_residual(
    equity_curve: pd.DataFrame, positions_history: pd.DataFrame, trades: pd.DataFrame,
) -> dict:
    """Écart entre le NAV du moteur et `capital initial + réalisé + latent`,
    reconstruit toutes jambes confondues (voir la docstring du module).

    Retourné plutôt que journalisé : un rapport qui affiche une décomposition
    doit pouvoir dire à quel point elle boucle."""
    dates = pd.DatetimeIndex(_as_datetime(equity_curve["date"]))
    initial_capital = float(equity_curve["nav"].iloc[0])
    reconstructed = side_equity_curve(dates, trades, positions_history, initial_capital)
    residual = equity_curve["nav"].to_numpy() - reconstructed["nav"].to_numpy()
    max_abs = float(np.nanmax(np.abs(residual))) if len(residual) else 0.0
    return {
        "max_abs_residual_dollar": max_abs,
        "max_abs_residual_pct_of_initial": float(max_abs / initial_capital * 100) if initial_capital else None,
        "final_residual_dollar": float(residual[-1]) if len(residual) else 0.0,
    }


def _keep_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if df.empty or "option_type" not in df.columns:
        return df.iloc[0:0]
    return df[df["option_type"] == side]


def side_metrics(
    equity_curve: pd.DataFrame,
    positions_history: pd.DataFrame,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.0,
    benchmark_prices: Optional[pd.Series] = None,
) -> dict[str, dict]:
    """{"ALL"|"CALL"|"PUT": métriques}. Les métriques d'une jambe sont celles
    de metrics_mod.compute_metrics appliquées à sa courbe synthétique et à ses
    seuls trades -- mêmes définitions que l'agrégat, donc directement
    comparables ligne à ligne."""
    dates = pd.DatetimeIndex(_as_datetime(equity_curve["date"]))
    initial_capital = float(equity_curve["nav"].iloc[0])

    out = {
        "ALL": metrics_mod.compute_metrics(
            equity_curve, trades, risk_free_rate=risk_free_rate, benchmark_prices=benchmark_prices,
        ),
    }
    for side in SIDES:
        curve = side_equity_curve(
            dates, _keep_side(trades, side), _keep_side(positions_history, side), initial_capital,
        )
        out[side] = metrics_mod.compute_metrics(
            curve[_CURVE_COLUMNS], _keep_side(trades, side),
            risk_free_rate=risk_free_rate, benchmark_prices=benchmark_prices,
        )
        # Le P&L de la jambe en dollars : c'est la seule grandeur qui s'ADDITIONNE
        # entre jambes (les %, eux, sont rapportés au capital initial complet).
        out[side]["leg_pnl_dollar"] = float(curve["nav"].iloc[-1] - initial_capital)
    out["ALL"]["leg_pnl_dollar"] = float(equity_curve["nav"].iloc[-1] - initial_capital)
    return out


def metrics_table(per_side: dict[str, dict], keys: Optional[list[str]] = None) -> pd.DataFrame:
    """Tableau lisible : une ligne par métrique, une colonne par jambe."""
    keys = keys or [
        "leg_pnl_dollar", "total_return_pct", "cagr_pct", "annualized_volatility_pct",
        "sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "max_drawdown_duration_days",
        "calmar_ratio", "num_trades", "win_rate_pct", "profit_factor",
        "avg_win_pct", "avg_loss_pct", "avg_trade_return_pct", "best_trade_pct", "worst_trade_pct",
        "avg_holding_days", "avg_num_positions", "avg_exposure_pct",
    ]
    columns = [c for c in ("ALL", *SIDES) if c in per_side]
    return pd.DataFrame(
        {c: [per_side[c].get(k) for k in keys] for c in columns},
        index=keys,
    )


def valuation_breakdown(positions_history: pd.DataFrame) -> pd.DataFrame:
    """Répartition des VALORISATIONS entre calls et puts : combien de prime
    chaque jambe immobilise, en moyenne et au pic, et quelle part du
    portefeuille elle représente.

    Moyenne prise sur TOUS les jours de bourse (une jambe absente compte zéro),
    pas seulement sur les jours où la jambe est ouverte : c'est la part du
    capital réellement consacrée à ce pari sur la durée du backtest."""
    if positions_history.empty:
        return pd.DataFrame()

    df = positions_history.assign(date=_as_datetime(positions_history["date"]))
    daily = df.pivot_table(
        index="date", columns="option_type", values="market_value", aggfunc="sum", fill_value=0.0,
    )
    counts = df.pivot_table(
        index="date", columns="option_type", values="symbol", aggfunc="count", fill_value=0,
    )
    for side in SIDES:
        if side not in daily.columns:
            daily[side] = 0.0
        if side not in counts.columns:
            counts[side] = 0

    total_mean = daily[list(SIDES)].sum(axis=1).mean()
    rows = {}
    for side in SIDES:
        rows[side] = {
            "avg_market_value_dollar": float(daily[side].mean()),
            "pct_of_avg_invested": float(daily[side].mean() / total_mean * 100) if total_mean else None,
            "max_market_value_dollar": float(daily[side].max()),
            "avg_num_positions": float(counts[side].mean()),
            "max_num_positions": int(counts[side].max()),
            # Jours où la jambe porte au moins une position : distingue "petite
            # en taille" de "rarement ouverte", que la seule moyenne confond.
            "days_with_exposure": int((counts[side] > 0).sum()),
            "pct_of_days_with_exposure": float((counts[side] > 0).mean() * 100),
        }
    return pd.DataFrame(rows)


def volume_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """Répartition des VOLUMES DE TRADE entre calls et puts : nombre de
    trades, contrats négociés et montants effectivement décaissés/encaissés.

    `entry_notional` et `exit_notional` sont des flux de trésorerie réels
    (prime tout compris x contrats x multiplicateur), pas des notionnels
    delta-équivalents : ce sont eux qui portent les frais.

    `realized_pnl_dollar` ne compte que les positions CLÔTURÉES -- il diffère
    donc du `leg_pnl_dollar` de side_metrics, qui ajoute le latent des
    positions encore ouvertes à la fin du backtest."""
    if trades.empty:
        return pd.DataFrame()

    df = trades.copy()
    df["entry_notional"] = df["shares"] * df["entry_price"]
    df["exit_notional"] = df["shares"] * df["exit_price"]

    def aggregate(sub: pd.DataFrame) -> dict:
        return {
            "num_trades": len(sub),
            "contracts_traded": float(sub["contracts"].sum()) if "contracts" in sub.columns else None,
            "entry_notional_dollar": float(sub["entry_notional"].sum()),
            "exit_notional_dollar": float(sub["exit_notional"].sum()),
            "turnover_dollar": float(sub["entry_notional"].sum() + sub["exit_notional"].sum()),
            "realized_pnl_dollar": float(sub["pnl"].sum()),
        }

    totals = aggregate(df)
    rows = {}
    for side in SIDES:
        values = aggregate(_keep_side(df, side))
        for key in list(values):
            total = totals[key]
            values[f"{key}_pct"] = (
                float(values[key] / total * 100) if total and values[key] is not None else None
            )
        rows[side] = values
    # La colonne ALL porte les mêmes lignes que les jambes (100% partout)
    # plutôt que des trous : un NaN en face de "num_trades_pct" se lirait
    # comme une donnée manquante alors que c'est le dénominateur lui-même.
    rows["ALL"] = {**totals, **{f"{key}_pct": 100.0 for key in totals}}
    return pd.DataFrame(rows)[["ALL", *SIDES]]


def cost_breakdown(executions: pd.DataFrame) -> pd.DataFrame:
    """Coûts d'exécution (commission + frais tiers, slippage) par jambe.

    Lit le JOURNAL DES EXÉCUTIONS et non trades.parquet : celui-ci n'a que les
    ventes, donc il manquerait toute la friction payée à l'ACHAT -- la moitié
    des fills, et en pratique bien plus puisque le redéploiement du cash oisif
    ne fait qu'acheter (cf. options_engine._record_fill).

    `commission_dollar` inclut les frais tiers (ORF, CAT, OCC, TAF, SEC) en
    plus de la commission du courtier : c'est le coût total facturé sur
    l'ordre, cf. options_engine._order_commission."""
    if executions is None or executions.empty:
        return pd.DataFrame()

    def aggregate(sub: pd.DataFrame) -> dict:
        traded = (sub["contracts"] * sub["multiplier"] * sub["price"]).sum()
        friction = float(sub["commission"].sum() + sub["slippage"].sum())
        return {
            "num_fills": len(sub),
            "num_buys": int((sub["side"] == "buy").sum()),
            "num_sells": int((sub["side"] == "sell").sum()),
            "contracts_traded": float(sub["contracts"].sum()),
            "commission_dollar": float(sub["commission"].sum()),
            "slippage_dollar": float(sub["slippage"].sum()),
            "friction_dollar": friction,
            # Rapportée au volume BRUT échangé : dit combien coûte un dollar
            # de contrat mis en place ou dénoué, indépendamment du P&L.
            "friction_pct_of_turnover": float(friction / traded * 100) if traded else None,
        }

    rows = {"ALL": aggregate(executions)}
    for side in SIDES:
        rows[side] = aggregate(_keep_side(executions, side))
    return pd.DataFrame(rows)[["ALL", *SIDES]]


def cost_by_reason(executions: pd.DataFrame) -> pd.DataFrame:
    """Coûts d'exécution par MOTIF d'ordre : dit quel mécanisme du moteur
    consomme la friction.

    C'est le tableau qui rend visible un poste comme `deploy_idle_cash` (le
    redéploiement du cash oisif, appelé chaque jour de bourse), dont la
    friction se noyait auparavant dans un total agrégé."""
    if executions is None or executions.empty:
        return pd.DataFrame()

    grouped = executions.groupby("reason").agg(
        num_fills=("contracts", "size"),
        contracts_traded=("contracts", "sum"),
        commission_dollar=("commission", "sum"),
        slippage_dollar=("slippage", "sum"),
    )
    grouped["friction_dollar"] = grouped["commission_dollar"] + grouped["slippage_dollar"]
    total = grouped["friction_dollar"].sum()
    grouped["pct_of_friction"] = grouped["friction_dollar"] / total * 100 if total else None
    return grouped.sort_values("friction_dollar", ascending=False)


def quantity_reconciliation(executions: pd.DataFrame, positions_history: pd.DataFrame) -> pd.DataFrame:
    """Contrats ACHETÉS vs VENDUS vs ENCORE DÉTENUS, par jambe -- l'invariant
    que trades.parquet seul ne permet pas de vérifier.

    `ecart` doit valoir zéro : un contrat vendu a forcément été acheté. Un
    écart non nul signalerait une vraie erreur de comptabilité du moteur (et
    non l'illusion d'optique de trades.parquet, où une position construite en
    plusieurs achats solde plus de contrats qu'il n'en a été acheté à son
    `entry_date` -- cf. options_engine._record_fill)."""
    if executions is None or executions.empty:
        return pd.DataFrame()

    detenus = {}
    if positions_history is not None and not positions_history.empty:
        df = positions_history.assign(date=_as_datetime(positions_history["date"]))
        dernier_jour = df[df["date"] == df["date"].max()]
        detenus = dernier_jour.groupby("option_type")["contracts"].sum().to_dict()

    rows = {}
    for side in SIDES:
        sub = _keep_side(executions, side)
        achetes = float(sub[sub["side"] == "buy"]["contracts"].sum())
        vendus = float(sub[sub["side"] == "sell"]["contracts"].sum())
        restants = float(detenus.get(side, 0.0))
        rows[side] = {
            "contracts_achetes": achetes,
            "contracts_vendus": vendus,
            "contracts_encore_detenus": restants,
            "ecart": achetes - vendus - restants,
        }
    return pd.DataFrame(rows)


def pnl_by_exit_reason(trades: pd.DataFrame) -> pd.DataFrame:
    """P&L par (jambe, motif de sortie) -- le tableau qui répond réellement à
    "où est-ce que je perds de l'argent ?".

    Un stop-loss massivement négatif sur la jambe put ne se lit pas de la même
    façon qu'une expiration massivement négative : le premier dit que les
    seuils coupent au mauvais moment, le second que le pari lui-même ne se
    réalise pas dans le temps imparti."""
    if trades.empty or "exit_reason" not in trades.columns:
        return pd.DataFrame()

    grouped = trades.groupby(["option_type", "exit_reason"]).agg(
        num_trades=("pnl", "size"),
        pnl_dollar=("pnl", "sum"),
        avg_return_pct=("return_pct", "mean"),
        median_return_pct=("return_pct", "median"),
        win_rate_pct=("pnl", lambda s: (s > 0).mean() * 100),
        avg_holding_days=("holding_days", "mean"),
    )
    return grouped.sort_values(["option_type", "pnl_dollar"]).reset_index()


def pnl_contribution_over_time(
    equity_curve: pd.DataFrame, positions_history: pd.DataFrame, trades: pd.DataFrame,
) -> pd.DataFrame:
    """Contribution cumulée de chaque jambe, jour par jour (réalisé + latent).

    Sert à voir QUAND une jambe décroche, ce qu'aucun agrégat de fin de
    période ne montre : une jambe put peut être à l'équilibre sur toute la
    période et n'avoir perdu que sur un seul trimestre de rebond."""
    dates = pd.DatetimeIndex(_as_datetime(equity_curve["date"]))
    initial_capital = float(equity_curve["nav"].iloc[0])
    out = pd.DataFrame({"date": dates})
    for side in SIDES:
        curve = side_equity_curve(
            dates, _keep_side(trades, side), _keep_side(positions_history, side), initial_capital,
        )
        out[f"{side}_pnl_cumul"] = curve["nav"].to_numpy() - initial_capital
        out[f"{side}_market_value"] = curve["invested_value"].to_numpy()
    out["total_pnl_cumul"] = equity_curve["nav"].to_numpy() - initial_capital
    return out


def annual_pnl_by_side(trades: pd.DataFrame) -> pd.DataFrame:
    """P&L RÉALISÉ par année de sortie et par jambe. Complémentaire de
    pnl_contribution_over_time (qui inclut le latent) : ici chaque dollar est
    définitivement acquis ou perdu, ce qui rend les années comparables."""
    if trades.empty:
        return pd.DataFrame()
    df = trades.assign(exit_year=_as_datetime(trades["exit_date"]).dt.year)
    table = df.pivot_table(
        index="exit_year", columns="option_type", values="pnl", aggfunc="sum", fill_value=0.0,
    )
    for side in SIDES:
        if side not in table.columns:
            table[side] = 0.0
    table = table[list(SIDES)]
    table["TOTAL"] = table.sum(axis=1)
    return table
