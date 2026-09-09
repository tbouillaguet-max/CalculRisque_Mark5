"""
Page Data : liste des entreprises collectées (options, cours, 10-K) et
niveau de couverture par source.
"""

import sys
from pathlib import Path

import streamlit as st

# Même correctif que Home.py : utils.py vit dans report/, un niveau
# au-dessus de ce dossier report/pages/ (voir Home.py pour le détail du bug
# Streamlit constaté).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import build_coverage_table

st.set_page_config(page_title="Data — Pipeline options US", page_icon="📊", layout="wide")
st.title("📊 Data collectées")

df = build_coverage_table()

if df.empty:
    st.warning("Aucune donnée trouvée. Lance d'abord 01_build_universe.py (et idéalement toute la chaîne 01→08).")
    st.stop()

# --- Filtres ----------------------------------------------------------------
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    secteurs = sorted(df["sector"].dropna().unique()) if "sector" in df.columns else []
    secteurs_sel = st.multiselect("Secteur", secteurs, default=[])
with col2:
    recherche = st.text_input("Recherche (ticker ou nom)", "")
with col3:
    complet_seul = st.checkbox("Couverture complète uniquement", value=False)

filtered = df.copy()
if secteurs_sel:
    filtered = filtered[filtered["sector"].isin(secteurs_sel)]
if recherche:
    q = recherche.strip().lower()
    filtered = filtered[
        filtered["symbol"].str.lower().str.contains(q) | filtered["company_name"].str.lower().str.contains(q)
    ]
if complet_seul:
    filtered = filtered[filtered["couverture_complete"]]

# --- Résumé -------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Entreprises affichées", len(filtered))
c2.metric("Avec cours", int((filtered["annees_cours"] > 0).sum()))
c3.metric("Avec 10-K", int((filtered["exercices_10k"] > 0).sum()))
c4.metric("Avec options", int((filtered["contrats_options"] > 0).sum()))

# --- Tableau ------------------------------------------------------------
st.dataframe(
    filtered,
    width="stretch",
    hide_index=True,
    column_config={
        "symbol": "Ticker",
        "company_name": "Entreprise",
        "sector": "Secteur",
        "annees_cours": st.column_config.NumberColumn("Années de cours"),
        "premiere_annee_cours": st.column_config.NumberColumn("Cours depuis", format="%d"),
        "derniere_annee_cours": st.column_config.NumberColumn("Cours jusqu'à", format="%d"),
        "dernier_cours": st.column_config.NumberColumn("Dernier cours", format="%.2f $"),
        "exercices_10k": st.column_config.NumberColumn("Exercices 10-K"),
        "dernier_exercice_10k": st.column_config.NumberColumn("Dernier 10-K", format="%d"),
        "contrats_options": st.column_config.NumberColumn("Contrats options"),
        "derniere_maj_options": "Options collectées le",
        "couverture_complete": st.column_config.CheckboxColumn("Couverture complète"),
    },
)

st.caption(
    "« Couverture complète » = au moins un cours, un exercice 10-K et un contrat "
    "d'options collectés pour l'entreprise. Une case décochée pointe vers le "
    "script à relancer (03, 04 ou 08) pour ce ticker."
)
