"""
Fonctions de chargement de données et d'analyse partagées par les pages du
rapport Streamlit. Centralise la lecture des fichiers produits par le
pipeline (01 à 08) : ce rapport ne fait QUE lire ce qui existe déjà dans
data/, il ne relance jamais lui-même une collecte.

config.py vit à la racine du dépôt, un niveau au-dessus de ce dossier
report/ : on ajoute donc ce dossier parent à sys.path avant de l'importer,
pour que `streamlit run report/Home.py` fonctionne depuis n'importe où.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


# ============================================================================
# Chargement des fichiers du pipeline
# ============================================================================

@st.cache_data(ttl=300)
def load_universe() -> pd.DataFrame:
    if not config.UNIVERSE_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(config.UNIVERSE_FILE, encoding="utf-8-sig")


@st.cache_data(ttl=300)
def load_prices() -> pd.DataFrame:
    if not config.PRICES_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(config.PRICES_FILE)


@st.cache_data(ttl=300)
def load_financials() -> pd.DataFrame:
    if not config.FINANCIALS_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(config.FINANCIALS_FILE)


@st.cache_data(ttl=300)
def load_options_current() -> pd.DataFrame:
    """Dernier snapshot uniquement (config.OPTIONS_FILE, toujours écrasé par
    le run suivant de 04_recuperation_options.py)."""
    if not config.OPTIONS_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(config.OPTIONS_FILE)


@st.cache_data(ttl=300)
def load_options_history() -> pd.DataFrame:
    """Concatène tous les snapshots archivés dans data/options/history/ (un
    fichier par run de 04, jamais écrasé). Retombe sur le seul snapshot
    courant si aucune archive n'existe encore (ex: juste après la mise à
    jour du script, avant le premier nouveau run)."""
    files = sorted(config.DIR_OPTIONS_HISTORY.glob("option_chains_*.parquet"))
    if not files:
        return load_options_current()
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


@st.cache_data(ttl=300)
def load_multiples() -> pd.DataFrame:
    if not config.MULTIPLES_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(config.MULTIPLES_FILE)


@st.cache_data(ttl=300)
def load_dcf() -> pd.DataFrame:
    if not config.DCF_FILE.exists():
        return pd.DataFrame()
    return pd.read_excel(config.DCF_FILE, sheet_name="DCF", engine="openpyxl")


# ============================================================================
# Page Data : table de couverture par entreprise
# ============================================================================

def build_coverage_table() -> pd.DataFrame:
    """Une ligne par entreprise de l'univers : combien d'années de cours,
    d'exercices 10-K, de contrats d'options collectés, et à quand remonte la
    dernière mise à jour de chaque source."""
    universe = load_universe()
    if universe.empty:
        return pd.DataFrame()

    base = universe[["RIC", "Instrument_Name"]].rename(
        columns={"RIC": "symbol_ric", "Instrument_Name": "company_name"}
    )
    if "sector" in universe.columns:
        base["sector"] = universe["sector"]
    base["symbol"] = base["symbol_ric"].apply(config.to_ib_symbol)

    prices = load_prices()
    if not prices.empty:
        agg_prices = (
            prices.sort_values(["symbol", "year"])
            .groupby("symbol")
            .agg(
                annees_cours=("year", "nunique"),
                premiere_annee_cours=("year", "min"),
                derniere_annee_cours=("year", "max"),
                dernier_cours=("close", "last"),
            )
            .reset_index()
        )
    else:
        agg_prices = pd.DataFrame(
            columns=["symbol", "annees_cours", "premiere_annee_cours", "derniere_annee_cours", "dernier_cours"]
        )

    financials = load_financials()
    if not financials.empty:
        agg_fin = (
            financials.groupby("symbol")
            .agg(exercices_10k=("year", "nunique"), dernier_exercice_10k=("year", "max"))
            .reset_index()
        )
    else:
        agg_fin = pd.DataFrame(columns=["symbol", "exercices_10k", "dernier_exercice_10k"])

    options = load_options_current()
    if not options.empty:
        agg_kwargs = {"contrats_options": ("symbol", "size")}
        if "snapshot_datetime" in options.columns:
            agg_kwargs["derniere_maj_options"] = ("snapshot_datetime", "max")
        agg_opt = options.groupby("symbol").agg(**agg_kwargs).reset_index()
        if "derniere_maj_options" not in agg_opt.columns:
            agg_opt["derniere_maj_options"] = None
    else:
        agg_opt = pd.DataFrame(columns=["symbol", "contrats_options", "derniere_maj_options"])

    df = base.merge(agg_prices, on="symbol", how="left")
    df = df.merge(agg_fin, on="symbol", how="left")
    df = df.merge(agg_opt, on="symbol", how="left")

    for col in ("annees_cours", "exercices_10k", "contrats_options"):
        df[col] = df[col].fillna(0).astype(int)

    df["couverture_complete"] = (
        (df["annees_cours"] > 0) & (df["exercices_10k"] > 0) & (df["contrats_options"] > 0)
    )

    return df.drop(columns=["symbol_ric"]).sort_values("company_name").reset_index(drop=True)


# ============================================================================
# Page Analyse : dérivés options, nappe de vol, liquidité, clustering
# ============================================================================

def compute_days_to_expiry(df: pd.DataFrame) -> pd.Series:
    """expiry est au format IBKR 'YYYYMMDD' (str). La date de référence est
    le fetch_timestamp du contrat si présent (le plus précis), sinon
    snapshot_datetime (horodatage du run), sinon la date du jour."""
    expiry_dt = pd.to_datetime(df["expiry"], format="%Y%m%d", errors="coerce")
    if "fetch_timestamp" in df.columns:
        as_of = pd.to_datetime(df["fetch_timestamp"], errors="coerce")
    elif "snapshot_datetime" in df.columns:
        as_of = pd.to_datetime(df["snapshot_datetime"], errors="coerce")
    else:
        as_of = pd.Timestamp.now()
    return (expiry_dt - as_of).dt.days


def add_liquidity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mid = (df["bid"] + df["ask"]) / 2
    df["spread_abs"] = df["ask"] - df["bid"]
    df["spread_pct"] = df["spread_abs"] / mid.replace({0: np.nan})
    df["oi_volume_ratio"] = df["open_interest"] / df["volume"].replace({0: np.nan})
    return df


def fit_vol_surface(df: pd.DataFrame, n_grid: int = 40):
    """
    Ajuste une surface de volatilité implicite lissée par processus gaussien
    (sklearn.gaussian_process.GaussianProcessRegressor) sur les points
    (log-moneyness, jours avant échéance) -> implied_vol observés, et
    retourne une grille régulière prête pour un plotly Surface, plus les
    points bruts utilisés (pour les superposer en scatter).

    Un GP est préféré à une simple interpolation (scipy.griddata) car la
    grille strike x échéance d'une chaîne d'options réelle est irrégulière
    et creuse : le GP interpole ET extrapole proprement, avec une incertitude
    en prime (utile pour distinguer une zone bien couverte d'une zone
    quasiment devinée).
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    pts = df.dropna(subset=["log_moneyness", "days_to_expiry", "implied_vol"])
    pts = pts[(pts["implied_vol"] > 0) & (pts["days_to_expiry"] > 0)]
    if len(pts) < 5:
        return None

    X = pts[["log_moneyness", "days_to_expiry"]].to_numpy()
    y = pts["implied_vol"].to_numpy()

    # Normalisation des échelles (log-moneyness ~ [-0.3;0.3] contre jours
    # ~ [0;1000]) pour que le kernel Matern voie des distances comparables
    # sur les deux axes.
    x_mean, x_std = X.mean(axis=0), X.std(axis=0) + 1e-9
    X_norm = (X - x_mean) / x_std

    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)
    gp.fit(X_norm, y)

    lm_grid = np.linspace(pts["log_moneyness"].min(), pts["log_moneyness"].max(), n_grid)
    dte_grid = np.linspace(pts["days_to_expiry"].min(), pts["days_to_expiry"].max(), n_grid)
    LM, DTE = np.meshgrid(lm_grid, dte_grid)
    grid = np.column_stack([LM.ravel(), DTE.ravel()])
    grid_norm = (grid - x_mean) / x_std
    Z, Z_std = gp.predict(grid_norm, return_std=True)

    return {
        "LM": LM, "DTE": DTE,
        "IV": Z.reshape(LM.shape), "IV_STD": Z_std.reshape(LM.shape),
        "points": pts,
    }


def flag_liquidity_anomalies(
    df: pd.DataFrame,
    features=("spread_pct", "volume", "open_interest", "moneyness_pct"),
    contamination: float = 0.05,
) -> pd.DataFrame:
    """IsolationForest (scikit-learn) sur des indicateurs de liquidité/prix
    pour repérer les contrats atypiques (spread anormalement large,
    volume/OI incohérents avec la position en moneyness...). is_anomaly=True
    ne veut pas dire "erreur de données", juste "statistiquement atypique
    dans cette chaîne d'options" : à vérifier manuellement avant d'agir
    dessus (quote figée, contrat très peu traité, etc.)."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    features = [f for f in features if f in df.columns]
    data = df.dropna(subset=features).copy()
    if len(data) < 10 or not features:
        data["is_anomaly"] = False
        data["anomaly_score"] = np.nan
        return data

    X_scaled = StandardScaler().fit_transform(data[features].to_numpy())
    iso = IsolationForest(contamination=contamination, random_state=0)
    data["is_anomaly"] = iso.fit_predict(X_scaled) == -1
    data["anomaly_score"] = -iso.score_samples(X_scaled)  # plus haut = plus atypique
    return data


def cluster_multiples(
    df: pd.DataFrame,
    n_clusters: int = 5,
    features=("EV/EBITDA", "EV/Sales", "P/E"),
):
    """KMeans (scikit-learn) sur les multiples de valorisation standardisés,
    projeté en 2D par PCA pour la visualisation. Sert à voir si les
    regroupements statistiques recoupent le secteur GICS assigné en
    02_categoriser_secteurs.py, ou révèlent d'autres familles de
    comparables. Retourne (DataFrame avec colonnes cluster/pca_1/pca_2,
    variance expliquée par les 2 composantes) ou None si pas assez de
    données."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    features = list(features)
    data = df.dropna(subset=features).copy()
    # Ne garde qu'un exercice par entreprise (le plus récent) pour ne pas
    # sur-pondérer les entreprises avec un long historique dans le clustering.
    if "year" in data.columns and "symbol" in data.columns:
        data = data.sort_values("year").groupby("symbol", as_index=False).tail(1)

    if len(data) < n_clusters:
        return None

    X_scaled = StandardScaler().fit_transform(data[features].to_numpy())

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    data["cluster"] = km.fit_predict(X_scaled).astype(str)

    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X_scaled)
    data["pca_1"] = coords[:, 0]
    data["pca_2"] = coords[:, 1]

    return data, pca.explained_variance_ratio_
