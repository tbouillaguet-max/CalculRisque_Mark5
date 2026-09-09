"""
Page Pipeline : état du pipeline lui-même, sans ouvrir un seul fichier de log
à la main.

Lit data/pipeline_runs/<run_id>/report.json (écrit par
run_pipeline_quarterly.py) et la date de modification des fichiers de sortie.
Comme les autres pages, elle ne relance jamais rien : elle montre ce qui a
tourné, ce qui a échoué, et ce qui a vieilli.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Même correctif que Home.py : utils.py vit dans report/, un niveau au-dessus
# de ce dossier report/pages/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    STATUS_COLORS, build_freshness_table, load_pipeline_runs, read_step_log, status_badge,
)

st.set_page_config(page_title="Pipeline — Pipeline options US", page_icon="🩺", layout="wide")
st.title("🩺 État du pipeline")

runs = load_pipeline_runs()

# ============================================================================
# Fraîcheur des données : vraie même sans orchestrateur (runs manuels)
# ============================================================================
st.markdown("### Fraîcheur des données")

stale_after = st.slider(
    "Considérer une sortie comme périmée au-delà de", min_value=30, max_value=400,
    value=100, step=10, format="%d jours",
    help="Le pipeline est trimestriel (~92 jours) : 100 jours laisse une marge avant d'alerter.",
)

freshness = build_freshness_table(stale_after_days=stale_after)
missing = int((freshness["_status"] == "failed").sum())
stale = int((freshness["_status"] == "partial").sum())

c1, c2, c3 = st.columns(3)
c1.metric("Sorties à jour", int((freshness["_status"] == "success").sum()), delta=None)
c2.metric("Sorties périmées", stale)
c3.metric("Sorties absentes", missing)

st.dataframe(
    freshness.drop(columns=["_status"]),
    width="stretch", hide_index=True,
    column_config={
        "etat": st.column_config.TextColumn("État", width="small"),
        "sortie": st.column_config.TextColumn("Sortie"),
        "fichier": st.column_config.TextColumn("Fichier", width="medium"),
        "produit_par": st.column_config.TextColumn("Produit par"),
        "derniere_maj": st.column_config.DatetimeColumn("Dernière mise à jour", format="YYYY-MM-DD HH:mm"),
        "age_jours": st.column_config.NumberColumn("Âge (jours)", format="%d"),
        "taille_mo": st.column_config.NumberColumn("Taille (Mo)", format="%.2f"),
    },
)

if missing or stale:
    a_relancer = freshness.loc[freshness["_status"] != "success", "produit_par"].tolist()
    st.caption("Scripts à relancer pour combler ces manques : " + ", ".join(f"`{s}`" for s in a_relancer))

st.divider()

# ============================================================================
# Journal des runs de l'orchestrateur
# ============================================================================
st.markdown("### Runs de `run_pipeline_quarterly.py`")

if not runs:
    st.info(
        "Aucun run enregistré pour l'instant. `run_pipeline_quarterly.py` écrit un "
        "rapport dans `data/pipeline_runs/` à chaque exécution — le tableau de bord "
        "des étapes apparaîtra ici dès le premier run."
    )
    st.stop()

# --- Vue d'ensemble : les 20 derniers runs ---------------------------------
history = pd.DataFrame([
    {
        "run_id": r["run_id"], "mode": r.get("mode", "?"), "statut": r.get("status", "?"),
        "debut": pd.to_datetime(r.get("started_at"), errors="coerce"),
        "duree_min": (r.get("duration_seconds") or 0) / 60,
        "etapes_ok": sum(1 for s in r.get("steps", []) if s.get("status") == "success"),
        "etapes_total": len(r.get("steps", [])),
    }
    for r in runs
]).head(20)

fig_history = go.Figure()
for status in ("success", "partial", "failed", "running"):
    subset = history[history["statut"] == status]
    if subset.empty:
        continue
    fig_history.add_trace(go.Bar(
        x=subset["run_id"], y=subset["duree_min"], name=status_badge(status),
        marker=dict(color=STATUS_COLORS[status], line=dict(width=0)),
        customdata=subset[["debut", "mode", "etapes_ok", "etapes_total"]],
        hovertemplate=(
            "<b>%{x}</b> (%{customdata[1]})<br>"
            "%{customdata[0]|%Y-%m-%d %H:%M}<br>Durée : %{y:.1f} min<br>"
            "Étapes réussies : %{customdata[2]}/%{customdata[3]}<extra></extra>"
        ),
    ))
# Axe catégoriel (un run = une barre, du plus ancien au plus récent) plutôt
# que temporel : les runs sont trimestriels et irréguliers, les espacer dans
# le temps n'apporte rien et écrase les barres à quelques pixels.
fig_history.update_layout(
    height=280, margin=dict(l=0, r=0, t=10, b=0), bargap=0.35,
    yaxis_title="Durée (min)",
    xaxis=dict(title=None, type="category", categoryorder="array",
               categoryarray=list(reversed(history["run_id"].tolist()))),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_history, width="stretch")
st.caption("Durée des 20 derniers runs. La couleur reprend le statut, toujours doublé de son libellé dans la légende.")

# --- Run sélectionné --------------------------------------------------------
labels = {f"{r['run_id']} — {r.get('mode', '?')} — {status_badge(r.get('status', '?'))}": r for r in runs}
choice = st.selectbox("Run à inspecter (le plus récent en premier)", list(labels))
run = labels[choice]
steps = run.get("steps", [])

k1, k2, k3, k4 = st.columns(4)
k1.metric("Statut", status_badge(run.get("status", "?")))
k2.metric("Durée totale", f"{(run.get('duration_seconds') or 0) / 60:.1f} min")
k3.metric("Étapes réussies", f"{sum(1 for s in steps if s.get('status') == 'success')}/{len(steps)}")
k4.metric("Réessais cumulés", sum(max(s.get("attempts", 1) - 1, 0) for s in steps))

if run.get("status") == "failed":
    st.error(
        "Ce run s'est arrêté sur l'échec d'une étape requise. "
        "`python run_pipeline_quarterly.py --resume` repart des étapes restantes."
    )
elif run.get("status") == "partial":
    st.warning(
        "Ce run est allé au bout, mais une étape optionnelle (8-K, validation "
        "qualitative ou options) a échoué : la valorisation est à jour, son "
        "enrichissement ne l'est pas."
    )

if not steps:
    st.info("Ce run n'a encore enregistré aucune étape.")
    st.stop()

# --- Durée par étape --------------------------------------------------------
steps_df = pd.DataFrame([
    {
        "script": s["script"], "statut": s.get("status", "?"),
        "duree_s": s.get("duration_seconds") or 0.0,
        "tentatives": s.get("attempts", 0),
        "code_retour": s.get("returncode"),
        "raison": s.get("reason", ""),
        "requise": s.get("required", True),
    }
    for s in steps
])

order = list(reversed(steps_df["script"].tolist()))
fig_steps = go.Figure()
for status in ("success", "partial", "failed", "skipped", "running"):
    subset = steps_df[steps_df["statut"] == status]
    if subset.empty:
        continue
    fig_steps.add_trace(go.Bar(
        y=subset["script"], x=subset["duree_s"], orientation="h", name=status_badge(status),
        marker=dict(color=STATUS_COLORS[status], line=dict(width=0)),
        customdata=subset[["tentatives", "statut"]],
        hovertemplate="<b>%{y}</b><br>%{x:.1f} s<br>Tentatives : %{customdata[0]}<extra></extra>",
    ))
fig_steps.update_layout(
    height=max(260, 42 * len(steps_df)), margin=dict(l=0, r=0, t=10, b=0),
    xaxis_title="Durée (s)", yaxis=dict(categoryorder="array", categoryarray=order, title=None),
    bargap=0.35, legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_steps, width="stretch")

st.dataframe(
    steps_df.assign(etat=steps_df["statut"].map(status_badge)).drop(columns=["statut"])[
        ["etat", "script", "duree_s", "tentatives", "code_retour", "requise", "raison"]
    ],
    width="stretch", hide_index=True,
    column_config={
        "etat": st.column_config.TextColumn("État", width="small"),
        "script": st.column_config.TextColumn("Étape"),
        "duree_s": st.column_config.NumberColumn("Durée (s)", format="%.1f"),
        "tentatives": st.column_config.NumberColumn("Tentatives"),
        "code_retour": st.column_config.NumberColumn("Code retour"),
        "requise": st.column_config.CheckboxColumn("Bloquante"),
        "raison": st.column_config.TextColumn("Raison (si sautée)", width="medium"),
    },
)

# --- Logs -------------------------------------------------------------------
st.markdown("#### Log d'une étape")
with_logs = [s["script"] for s in steps if s.get("status") != "skipped"]
if not with_logs:
    st.caption("Aucune étape n'a produit de log sur ce run (toutes sautées).")
else:
    default = next((s["script"] for s in steps if s.get("status") == "failed"), with_logs[-1])
    script = st.selectbox("Étape", with_logs, index=with_logs.index(default))
    st.code(read_step_log(run["run_id"], script), language="text")
