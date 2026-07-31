"""
Point d'entrée du rapport. Lance avec :

    streamlit run report/Home.py

Nécessite que le pipeline (01 à 08) ait déjà tourné au moins une fois : ce
rapport ne fait QUE lire les fichiers dans data/, il ne relance jamais de
collecte lui-même.
"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit n'ajoute pas systématiquement le dossier du script à sys.path
# selon la version (constaté : `from utils import ...` échoue avec
# ModuleNotFoundError sur Streamlit >= 1.6x, même lancé via
# `streamlit run report/Home.py`) : on l'ajoute nous-mêmes pour trouver
# utils.py, qui vit dans ce même dossier report/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import config, load_financials, load_options_current, load_prices, load_universe

st.set_page_config(page_title="Pipeline options US — Rapport", page_icon="📈", layout="wide")

st.title("📈 Rapport pipeline options — entreprises américaines")

st.markdown(
    """
Ce rapport lit les fichiers produits par le pipeline (`01_build_universe.py`
à `08_recuperation_options.py`) dans `data/` et ne relance jamais lui-même de
collecte.

Utilise la barre latérale pour naviguer :

- **📊 Data** — quelles entreprises ont été collectées, et à quel niveau
  (cours, 10-K, options).
- **📈 Analyse** — nappe de volatilité, greeks, indicateurs de liquidité, et
  clustering des multiples de valorisation.
- **🧪 Stratégies** — pour un run de backtest (`09_backtest.py` ou
  `10_backtest_options.py`) : évolution du portefeuille (valorisation et
  composition), et journal des achats/ventes (quand, pourquoi).
"""
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Entreprises dans l'univers", len(load_universe()))
col2.metric("Lignes de cours", len(load_prices()))
col3.metric("Lignes financières (10-K)", len(load_financials()))
col4.metric("Contrats d'options (dernier snapshot)", len(load_options_current()))

history_files = list(config.DIR_OPTIONS_HISTORY.glob("*.parquet")) if config.DIR_OPTIONS_HISTORY.exists() else []
if not history_files:
    st.info(
        "Aucun snapshot d'options archivé pour l'instant (`data/options/history/`). "
        "La nappe de volatilité dans le temps (page Analyse) ne montrera qu'un seul "
        "instant tant que `08_recuperation_options.py` n'aura tourné qu'une fois "
        "depuis la mise à jour du script (voir le correctif d'archivage)."
    )
else:
    st.caption(f"{len(history_files)} snapshot(s) d'options archivé(s) dans data/options/history/.")
