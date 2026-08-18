"""
Page Stratégies & Backtests : évolution du portefeuille (valorisation +
composition), et journal des achats/ventes (quand, pourquoi) pour un run de
backtest choisi (09_backtest.py, stratégie actions, ou 10_backtest_options.py,
stratégie options).

Ce rapport ne fait QUE lire les fichiers déjà écrits par 09/10 dans
data/backtest*/<run_id>/ -- il ne relance jamais de backtest lui-même.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Même correctif que Home.py / les autres pages : utils.py vit dans report/,
# un niveau au-dessus de ce dossier report/pages/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import build_trade_log, list_backtest_runs, load_backtest_run

st.set_page_config(page_title="Stratégies & Backtests — Pipeline options US", page_icon="🧪", layout="wide")
st.title("🧪 Stratégies & Backtests")

# ============================================================================
# Sélection du run
# ============================================================================
kind_label = st.sidebar.radio("Stratégie", ["Actions (09)", "Options (10)"], horizontal=False)
kind = "actions" if kind_label.startswith("Actions") else "options"

runs = list_backtest_runs(kind)
if not runs:
    script = "09_backtest.py" if kind == "actions" else "10_backtest_options.py"
    st.warning(f"Aucun run de backtest trouvé pour cette stratégie. Lance d'abord `{script}`.")
    st.stop()

run_id = st.sidebar.selectbox("Run (le plus récent en premier)", runs)
data = load_backtest_run(kind, run_id)

equity_curve = data["equity_curve"]
positions_history = data["positions_history"]
trades = data["trades"]
signals_history = data["signals_history"]
metrics = data["metrics"]
run_config = data["run_config"]

if equity_curve.empty:
    st.warning(f"Le run {run_id} n'a pas de courbe d'equity exploitable (fichier vide ou manquant).")
    st.stop()

equity_curve = equity_curve.copy()
equity_curve["date"] = pd.to_datetime(equity_curve["date"])

st.caption(
    f"Run **{run_id}** — stratégie `{run_config.get('strategy', '?')}`, "
    f"du {run_config.get('start_date', '?')} au {run_config.get('end_date', '?')}."
)

# ============================================================================
# KPI
# ============================================================================
def _fmt_pct(v) -> str:
    return f"{v:.1f}%" if v is not None else "n/a"


def _fmt_num(v, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if v is not None else "n/a"


c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Rendement total", _fmt_pct(metrics.get("total_return_pct")))
c2.metric("CAGR", _fmt_pct(metrics.get("cagr_pct")))
c3.metric("Sharpe", _fmt_num(metrics.get("sharpe_ratio")))
c4.metric("Max drawdown", _fmt_pct(metrics.get("max_drawdown_pct")))
# Par THÈSE et non par exécution : trades.parquet logue une ligne par vente,
# allègements de rebalancement compris, si bien qu'une ligne qui monte inscrit
# une série de petits gains puis UNE perte au stop. Mesuré, une thèse unique
# perdante s'affichait à 91% de réussite sur 32 "trades" (cf.
# metrics._position_level_metrics). Repli sur les anciens champs pour les runs
# antérieurs à ce calcul, avec un libellé qui dit lequel est affiché.
_par_these = metrics.get("win_rate_positions_pct") is not None
c5.metric(
    "Taux de réussite" + ("" if _par_these else " (par exécution)"),
    _fmt_pct(metrics.get("win_rate_positions_pct", metrics.get("win_rate_pct"))),
)
c6.metric(
    "Positions closes" if _par_these else "Exécutions",
    metrics.get("num_positions_closed", metrics.get("num_trades", 0)),
)

# ============================================================================
# Évolution du portefeuille : valorisation (NAV) et composition
# ============================================================================
st.markdown("### Évolution du portefeuille")

tab_nav, tab_composition = st.tabs(["Valorisation (NAV)", "Composition"])

with tab_nav:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve["date"], y=equity_curve["nav"], mode="lines", name="NAV", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=equity_curve["date"], y=equity_curve["cash"], mode="lines", name="Cash", line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=equity_curve["date"], y=equity_curve["invested_value"], mode="lines", name="Valeur investie", line=dict(width=1, dash="dot")))
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=20, b=0), yaxis_title="$", xaxis_title=None, legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

with tab_composition:
    if positions_history.empty:
        st.info("Aucune position enregistrée sur ce run.")
    else:
        ph = positions_history.copy()
        ph["date"] = pd.to_datetime(ph["date"])
        pivot = ph.pivot_table(index="date", columns="symbol", values="market_value", aggfunc="sum").fillna(0.0)
        fig2 = px.area(pivot, x=pivot.index, y=pivot.columns)
        fig2.update_layout(height=450, margin=dict(l=0, r=0, t=20, b=0), yaxis_title="Valeur de marché ($)", xaxis_title=None, legend_title="Symbole")
        st.plotly_chart(fig2, width="stretch")
        st.caption(f"{ph['symbol'].nunique()} symbole(s) détenu(s) au cours du run.")

# ============================================================================
# Journal achats / ventes : quand et pourquoi
# ============================================================================
st.markdown("### Journal des achats / ventes")
st.caption(
    "Les ventes reprennent la raison d'exécution du moteur (stop-loss, take-profit, "
    "rebalancement, expiration, disparition des données). Les achats sont reconstruits "
    "à partir des positions détenues (le moteur ne loggue explicitement que les ventes) "
    "et rattachés au dernier signal de valorisation connu à cette date."
)

log = build_trade_log(positions_history, trades, signals_history)
if log.empty:
    st.info("Aucun achat/vente enregistré sur ce run.")
else:
    symbols = sorted(log["symbol"].dropna().unique())
    col_a, col_b = st.columns([2, 1])
    with col_a:
        symbols_sel = st.multiselect("Filtrer par symbole", symbols, default=[])
    with col_b:
        action_sel = st.radio("Action", ["Toutes", "Achat", "Vente"], horizontal=True)

    view = log.copy()
    if symbols_sel:
        view = view[view["symbol"].isin(symbols_sel)]
    if action_sel != "Toutes":
        view = view[view["action"] == action_sel]

    st.dataframe(
        view,
        width="stretch",
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Date"),
            "symbol": st.column_config.TextColumn("Symbole"),
            "action": st.column_config.TextColumn("Action"),
            "quantite": st.column_config.NumberColumn("Quantité", format="%.2f"),
            "prix": st.column_config.NumberColumn("Prix", format="%.2f"),
            "raison": st.column_config.TextColumn("Pourquoi", width="large"),
            "pnl": st.column_config.NumberColumn("P&L", format="%.2f"),
            "return_pct": st.column_config.NumberColumn("Rendement %", format="%.1f"),
        },
    )
    st.caption(f"{len(view)} ligne(s) affichée(s) sur {len(log)} au total.")
