"""
Page Analyse : nappe de volatilité, greeks et indicateurs de liquidité pour
une entreprise sélectionnée, plus un clustering des multiples de valorisation
sur l'ensemble du portefeuille suivi.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Même correctif que Home.py / pages/1_Data.py : utils.py vit dans report/,
# un niveau au-dessus de ce dossier report/pages/ (Streamlit n'ajoute pas
# systématiquement le dossier du script à sys.path selon la version).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    add_liquidity_metrics,
    cluster_multiples,
    compute_days_to_expiry,
    fit_vol_surface,
    flag_liquidity_anomalies,
    load_dcf,
    load_multiples,
    load_options_history,
    load_valorisation_combinee,
)

st.set_page_config(page_title="Analyse — Pipeline options US", page_icon="📈", layout="wide")
st.title("📈 Analyse options & valorisation")

history = load_options_history()
if history.empty:
    st.warning("Aucune donnée d'options trouvée. Lance d'abord 08_recuperation_options.py.")
    st.stop()

# ============================================================================
# Sélection entreprise + date de snapshot
# ============================================================================
tickers = sorted(history["symbol"].dropna().unique())
symbol = st.sidebar.selectbox("Entreprise", tickers)

df_symbol_all = history[history["symbol"] == symbol].copy()

date_col = "snapshot_date" if "snapshot_date" in df_symbol_all.columns else None
if date_col and df_symbol_all[date_col].notna().any():
    dates = sorted(df_symbol_all[date_col].dropna().unique())
    if len(dates) > 1:
        snap_date = st.sidebar.select_slider("Date du snapshot", options=dates, value=dates[-1])
    else:
        snap_date = dates[0]
        st.sidebar.caption(f"Un seul snapshot disponible pour {symbol} ({snap_date}).")
    df_symbol = df_symbol_all[df_symbol_all[date_col] == snap_date].copy()
else:
    snap_date = None
    df_symbol = df_symbol_all
    st.sidebar.caption("Pas d'horodatage de snapshot sur ces données (fichier antérieur au correctif d'archivage).")

st.subheader(symbol + (f" — snapshot du {snap_date}" if snap_date else ""))

if df_symbol.empty:
    st.info("Aucun contrat pour cette combinaison entreprise / date.")
    st.stop()

df_symbol["days_to_expiry"] = compute_days_to_expiry(df_symbol)
df_symbol["log_moneyness"] = np.log(df_symbol["strike"] / df_symbol["underlying_spot"])
df_symbol = add_liquidity_metrics(df_symbol)

option_type = st.sidebar.radio("Type de contrat", ["Les deux", "CALL", "PUT"], horizontal=True)
df_view = df_symbol if option_type == "Les deux" else df_symbol[df_symbol["option_type"] == option_type]

# ============================================================================
# Nappe de volatilité
# ============================================================================
st.markdown("### Nappe de volatilité implicite")
st.caption(
    "Surface lissée par processus gaussien (`GaussianProcessRegressor`, scikit-learn) "
    "sur les points (log-moneyness, jours avant échéance) → volatilité implicite. "
    "Les points bruts observés sont superposés en rouge."
)

if "iv_source" in df_view.columns:
    counts = df_view["iv_source"].value_counts()
    st.caption(
        f"Provenance de l'IV sur cette sélection — IBKR : {counts.get('ibkr', 0)}, "
        f"estimée localement (Black-Scholes, IBKR ne fournit pas de greeks en "
        f"données différées) : {counts.get('estime_local', 0)}, "
        f"indisponible : {counts.get('indisponible', 0)}."
    )

surface = fit_vol_surface(df_view)
if surface is None:
    n_total = len(df_view)
    n_iv_ok = df_view["implied_vol"].notna().sum() if "implied_vol" in df_view.columns else 0
    st.info(
        f"Pas assez de contrats avec IV valide pour ajuster une nappe (minimum 5 points) : "
        f"{n_iv_ok} contrat(s) avec IV sur {n_total} au total pour cette sélection. "
        "Si ce nombre est à 0 alors que tu utilises un fichier généré avant le correctif "
        "de 08_recuperation_options.py (repli Black-Scholes), relance la collecte avec le "
        "script mis à jour."
    )
else:
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=surface["LM"], y=surface["DTE"], z=surface["IV"],
            colorscale="Viridis", opacity=0.85, name="Nappe ajustée (GP)",
            colorbar=dict(title="IV"),
        )
    )
    pts = surface["points"]
    fig.add_trace(
        go.Scatter3d(
            x=pts["log_moneyness"], y=pts["days_to_expiry"], z=pts["implied_vol"],
            mode="markers", marker=dict(size=3, color="red"), name="Contrats observés",
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Log-moneyness (ln(strike/spot))",
            yaxis_title="Jours avant échéance",
            zaxis_title="Volatilité implicite",
        ),
        height=650, margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, width="stretch")

# ============================================================================
# Skew & structure par terme
# ============================================================================
st.markdown("### Skew et structure par terme")
col1, col2 = st.columns(2)

with col1:
    atm = df_view.dropna(subset=["moneyness_pct", "implied_vol"]).copy()
    atm["abs_moneyness"] = atm["moneyness_pct"].abs()
    atm_iv = (
        atm.sort_values("abs_moneyness")
        .groupby("expiry", as_index=False)
        .first()[["expiry", "days_to_expiry", "implied_vol"]]
        .sort_values("days_to_expiry")
    )
    if atm_iv.empty:
        st.info("Pas assez de données pour la structure par terme.")
    else:
        fig_term = px.line(
            atm_iv, x="days_to_expiry", y="implied_vol", markers=True,
            title="Structure par terme (IV la plus proche de ATM par échéance)",
        )
        fig_term.update_layout(xaxis_title="Jours avant échéance", yaxis_title="IV ATM (approx.)")
        st.plotly_chart(fig_term, width="stretch")

with col2:
    skew_rows = []
    for expiry, g in df_symbol.groupby("expiry"):
        calls = g[(g["option_type"] == "CALL") & (g["moneyness_pct"].between(15, 30))]
        puts = g[(g["option_type"] == "PUT") & (g["moneyness_pct"].between(-30, -15))]
        if calls.empty or puts.empty:
            continue
        skew_rows.append(
            {
                "expiry": expiry,
                "days_to_expiry": g["days_to_expiry"].iloc[0],
                "skew_put_call": puts["implied_vol"].mean() - calls["implied_vol"].mean(),
            }
        )
    if skew_rows:
        skew_df = pd.DataFrame(skew_rows).sort_values("days_to_expiry")
        fig_skew = px.bar(
            skew_df, x="expiry", y="skew_put_call",
            title="Skew (IV puts OTM 15-30% − IV calls OTM 15-30%), par échéance",
        )
        fig_skew.update_layout(xaxis_title="Échéance", yaxis_title="Écart d'IV")
        st.plotly_chart(fig_skew, width="stretch")
    else:
        st.info("Pas assez de contrats OTM des deux côtés (±15-30%) pour estimer un skew.")

# ============================================================================
# Greeks
# ============================================================================
st.markdown("### Greeks")
greek = st.selectbox("Greek à afficher", ["delta", "gamma", "vega", "theta"])
df_greek = df_view.dropna(subset=[greek])
if df_greek.empty:
    st.info("Pas de valeurs disponibles pour ce greek sur cette sélection.")
else:
    fig_greek = px.scatter(
        df_greek, x="strike", y=greek, color="expiry", symbol="option_type",
        hover_data=["implied_vol", "volume", "open_interest"],
        title=f"{greek.capitalize()} par strike",
    )
    fig_greek.add_vline(
        x=df_symbol["underlying_spot"].iloc[0], line_dash="dash",
        annotation_text="Spot", annotation_position="top",
    )
    st.plotly_chart(fig_greek, width="stretch")

# ============================================================================
# Liquidité
# ============================================================================
st.markdown("### Liquidité du marché")

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Spread moyen (%)",
    f"{df_view['spread_pct'].mean() * 100:.2f}%" if df_view["spread_pct"].notna().any() else "n/a",
)
c2.metric("Volume total", int(df_view["volume"].fillna(0).sum()))
c3.metric("Open interest total", int(df_view["open_interest"].fillna(0).sum()))
vol_calls = df_symbol.loc[df_symbol["option_type"] == "CALL", "volume"].fillna(0).sum()
vol_puts = df_symbol.loc[df_symbol["option_type"] == "PUT", "volume"].fillna(0).sum()
c4.metric("Ratio put/call (volume)", f"{(vol_puts / vol_calls):.2f}" if vol_calls else "n/a")

anomalies = flag_liquidity_anomalies(df_view)
n_anom = int(anomalies["is_anomaly"].sum())
st.caption(
    f"Détection d'anomalies (`IsolationForest`, scikit-learn) sur spread/volume/OI/moneyness : "
    f"{n_anom} contrat(s) signalé(s) comme atypique(s) sur {len(anomalies)}. "
    "« Atypique » ne veut pas dire « erreur » — à vérifier au cas par cas "
    "(quote figée, contrat peu traité...)."
)
if n_anom:
    cols = [
        c for c in
        ["expiry", "strike", "option_type", "implied_vol", "spread_pct", "volume", "open_interest", "moneyness_pct", "anomaly_score"]
        if c in anomalies.columns
    ]
    st.dataframe(
        anomalies[anomalies["is_anomaly"]][cols].sort_values("anomaly_score", ascending=False),
        width="stretch", hide_index=True,
    )

# ============================================================================
# Repères de valorisation
# ============================================================================
st.markdown("### Repères de valorisation")
multiples = load_multiples()
dcf = load_dcf()

col1, col2 = st.columns(2)
with col1:
    if not multiples.empty:
        row = multiples[multiples["symbol"] == symbol].sort_values("year").tail(1)
    else:
        row = pd.DataFrame()
    if not row.empty:
        st.write("**Multiples (dernier exercice disponible)**")
        cols = [c for c in ["year", "EV/Sales", "EV/EBITDA", "P/E"] if c in row.columns]
        st.dataframe(row[cols], hide_index=True, width="stretch")
    else:
        st.info("Pas de multiples calculés pour cette entreprise (lance 05_calcul_multiples.py).")

with col2:
    if not dcf.empty and "Ticker" in dcf.columns:
        row_dcf = dcf[dcf["Ticker"] == symbol]
    else:
        row_dcf = pd.DataFrame()
    if not row_dcf.empty:
        st.write("**Valorisation DCF**")
        cols = [c for c in ["Année", "Valeur_par_action_DCF", "Cours_actuel", "Écart_DCF_vs_Cours_%"] if c in row_dcf.columns]
        st.dataframe(row_dcf[cols], hide_index=True, width="stretch")
    else:
        st.info("Pas de DCF calculé pour cette entreprise (lance 07_calcul_dcf.py).")

st.markdown("#### Valorisation combinée (utilisée par la stratégie options)")
st.caption(
    "Multiples sectoriels moyens PAR ANNÉE en priorité (comparaison aux seuls pairs du même "
    "secteur cette année-là), DCF en repli quand le secteur a trop peu de pairs ou que les "
    "multiples ne sont pas calculables. Voir 06b_calcul_valorisation_combinee.py."
)
valorisation_combinee = load_valorisation_combinee()
if not valorisation_combinee.empty:
    row_vc = valorisation_combinee[valorisation_combinee["symbol"] == symbol].sort_values("year")
else:
    row_vc = pd.DataFrame()
if not row_vc.empty:
    cols = [
        "year", "close", "valuation_multiples_per_share", "valuation_dcf_per_share",
        "valuation_theoretical_per_share", "source", "gap_pct", "n_multiples_used",
    ]
    cols = [c for c in cols if c in row_vc.columns]
    st.dataframe(row_vc[cols], hide_index=True, width="stretch")
else:
    st.info("Pas de valorisation combinée pour cette entreprise (lance 06b_calcul_valorisation_combinee.py).")

# ============================================================================
# Clustering des multiples (portefeuille entier)
# ============================================================================
st.divider()
st.markdown("## Clustering des multiples de valorisation (tout le portefeuille)")
st.caption(
    "`KMeans` (scikit-learn) sur EV/EBITDA, EV/Sales et P/E standardisés, "
    "projeté en 2D par PCA pour la visualisation. Permet de voir si les "
    "regroupements statistiques recoupent le secteur GICS assigné en "
    "02_categoriser_secteurs.py, ou révèlent d'autres familles de comparables."
)

if multiples.empty:
    st.info("Pas de fichier de multiples trouvé (lance 05_calcul_multiples.py).")
else:
    n_clusters = st.slider("Nombre de clusters (k)", min_value=2, max_value=10, value=5)
    result = cluster_multiples(multiples, n_clusters=n_clusters)
    if result is None:
        st.info("Pas assez d'entreprises avec des multiples complets pour ce nombre de clusters.")
    else:
        clustered, explained_var = result
        hover_cols = [c for c in ["symbol", "sector", "EV/EBITDA", "EV/Sales", "P/E"] if c in clustered.columns]
        fig_cluster = px.scatter(
            clustered, x="pca_1", y="pca_2", color="cluster", hover_data=hover_cols,
            title=f"Clusters de valorisation (PCA, {explained_var.sum() * 100:.0f}% de variance expliquée)",
        )
        st.plotly_chart(fig_cluster, width="stretch")

        if "sector" in clustered.columns:
            st.write("**Recoupement cluster ↔ secteur GICS déclaré**")
            st.dataframe(pd.crosstab(clustered["cluster"], clustered["sector"]), width="stretch")
