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

import json
import sys
from pathlib import Path
from typing import Optional

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
    le run suivant de 08_recuperation_options.py)."""
    if not config.OPTIONS_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(config.OPTIONS_FILE)


@st.cache_data(ttl=300)
def load_options_history() -> pd.DataFrame:
    """Concatène tous les snapshots archivés dans data/options/history/ (un
    fichier par run de 08, jamais écrasé). Retombe sur le seul snapshot
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


@st.cache_data(ttl=300)
def load_valorisation_combinee() -> pd.DataFrame:
    """Valorisation théorique combinée (06b) : multiples sectoriels PAR ANNÉE
    en priorité, DCF en repli. Signal utilisé par la stratégie options
    (backtest/strategies/valuation_gap_options.py)."""
    if not config.VALORISATION_COMBINEE_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(config.VALORISATION_COMBINEE_FILE)


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


# ============================================================================
# Page Stratégies : runs de backtest (09_backtest.py / 10_backtest_options.py)
# ============================================================================

BACKTEST_RUN_FILES = ("equity_curve", "positions_history", "trades", "signals_history", "executions")


def _backtest_base_dir(kind: str) -> Path:
    return config.DIR_BACKTEST if kind == "actions" else config.DIR_BACKTEST_OPTIONS


@st.cache_data(ttl=60)
def list_backtest_runs(kind: str) -> list[str]:
    """kind: 'actions' (09_backtest.py) ou 'options' (10_backtest_options.py).
    Sous-dossiers de run (un par exécution, cf. --run-id) triés du plus
    récent au plus ancien (date de modification)."""
    base = _backtest_base_dir(kind)
    if not base.exists():
        return []
    runs = [p for p in base.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in runs]


@st.cache_data(ttl=60)
def load_backtest_run(kind: str, run_id: str) -> dict:
    """Charge les tables Parquet + les 2 JSON écrits par 09/10 en fin de run.
    Tables absentes (run antérieur à un ajout de fichier) -> DataFrame vide,
    pas une erreur.

    `executions` n'est écrit que par 10_backtest_options.py : le moteur
    actions (backtest/engine.py) n'a pas de journal de fills. Un run actions,
    comme un run options antérieur à ce journal, renvoie donc un DataFrame
    vide ici -- ce que build_trade_log traite comme le signal de retomber sur
    la reconstruction depuis positions_history."""
    run_dir = _backtest_base_dir(kind) / run_id
    result: dict = {}
    for name in BACKTEST_RUN_FILES:
        path = run_dir / f"{name}.parquet"
        result[name] = pd.read_parquet(path) if path.exists() else pd.DataFrame()

    metrics_path = run_dir / "metrics.json"
    result["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    run_config_path = run_dir / "run_config.json"
    result["run_config"] = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
    return result


# ============================================================================
# Page Pipeline : journal des runs (run_pipeline_quarterly.py) et fraîcheur
# ============================================================================

# Couleurs d'état, réservées : jamais réutilisées pour une série de données,
# et toujours accompagnées d'une icône + d'un libellé (voir STATUS_ICONS) pour
# qu'un statut ne repose jamais sur la seule couleur.
STATUS_COLORS = {
    "success": "#0ca30c",   # good
    "partial": "#fab219",   # warning
    "skipped": "#ec835a",   # serious
    "failed": "#d03b3b",    # critical
    "running": "#ec835a",
}
STATUS_ICONS = {
    "success": "✅", "partial": "⚠️", "skipped": "⏭️", "failed": "❌", "running": "⏳",
}
STATUS_LABELS = {
    "success": "Réussi", "partial": "Partiel", "skipped": "Sautée",
    "failed": "Échec", "running": "En cours",
}


def status_badge(status: str) -> str:
    return f"{STATUS_ICONS.get(status, '❔')} {STATUS_LABELS.get(status, status)}"


@st.cache_data(ttl=30)
def load_pipeline_runs() -> list[dict]:
    """Rapports de run écrits par run_pipeline_quarterly.py
    (data/pipeline_runs/<run_id>/report.json), du plus récent au plus ancien.
    Liste vide si l'orchestrateur n'a jamais tourné."""
    if not config.DIR_PIPELINE_RUNS.exists():
        return []
    reports = []
    for run_dir in sorted(config.DIR_PIPELINE_RUNS.iterdir(), reverse=True):
        path = run_dir / config.PIPELINE_RUN_REPORT_NAME
        if not path.exists():
            continue
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return reports


def read_step_log(run_id: str, script: str, max_lines: int = 400) -> str:
    """Fin du log d'une étape (les erreurs sont en fin de fichier)."""
    path = config.DIR_PIPELINE_RUNS / run_id / f"{script}.log"
    if not path.exists():
        return "(aucun log pour cette étape)"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return f"[... {len(lines) - max_lines} ligne(s) tronquée(s) ...]\n" + "\n".join(lines[-max_lines:])


# Fichiers de sortie du pipeline suivis pour leur fraîcheur, avec le script
# qui les produit (affiché tel quel quand un fichier manque ou vieillit).
PIPELINE_OUTPUTS: list[tuple[str, Path, str]] = [
    ("Univers S&P 500", config.UNIVERSE_FILE, "01_build_universe.py"),
    ("Univers point-in-time", config.UNIVERSE_FULL_FILE, "01b_historique_univers_sp500.py"),
    ("Cours annuels", config.PRICES_FILE, "03_recuperation_cours.py"),
    ("Cours quotidiens", config.DAILY_PRICES_FILE, "03b_recuperation_cours_quotidiens.py"),
    ("Financiers annuels (10-K)", config.FINANCIALS_FILE, "04_recuperation_10k.py"),
    ("Financiers TTM (10-Q)", config.FINANCIALS_TTM_FILE, "04b_recuperation_10q.py"),
    ("Événements 8-K", config.MATERIAL_EVENTS_8K_FILE, "04c_recuperation_8k.py"),
    ("Multiples", config.MULTIPLES_FILE, "05_calcul_multiples.py"),
    ("Valorisation combinée", config.VALORISATION_COMBINEE_FILE, "06b_calcul_valorisation_combinee.py"),
    ("DCF (historique)", config.DCF_HISTORY_FILE, "07_calcul_dcf.py"),
    ("Validation qualitative", config.QUALITATIVE_VALIDATION_FILE, "07b_validation_qualitative.py"),
    ("Chaînes d'options", config.OPTIONS_FILE, "08_recuperation_options.py"),
]


def build_freshness_table(stale_after_days: int = 100) -> pd.DataFrame:
    """Une ligne par sortie du pipeline : présence, taille, âge en jours.

    Le seuil par défaut (100 jours) correspond au rythme trimestriel du
    pipeline (~92 jours) plus une marge : au-delà, la donnée n'a pas été
    rafraîchie au dernier trimestre attendu."""
    freshness_badge = {"success": "✅ À jour", "partial": "⚠️ Périmé", "failed": "❌ Absent"}
    now = pd.Timestamp.now()
    rows = []
    for label, path, producer in PIPELINE_OUTPUTS:
        exists = path.exists()
        modified = pd.Timestamp(path.stat().st_mtime, unit="s") if exists else pd.NaT
        age_days = (now - modified).days if exists else None
        if not exists:
            status = "failed"
        elif age_days > stale_after_days:
            status = "partial"
        else:
            status = "success"
        rows.append({
            "etat": freshness_badge[status],
            "_status": status,
            "sortie": label,
            "fichier": str(path),
            "produit_par": producer,
            "derniere_maj": modified,
            "age_jours": age_days,
            "taille_mo": round(path.stat().st_size / 1e6, 2) if exists else None,
        })
    return pd.DataFrame(rows)


def _first_present(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


TRADE_LOG_COLUMNS = ["date", "symbol", "contrat", "action", "quantite", "prix", "raison", "pnl", "return_pct"]

# Motifs d'exécution du moteur options (options_engine : `reason` d'un fill),
# en clair. Un motif inconnu est affiché tel quel plutôt que masqué.
EXECUTION_REASON_LABELS = {
    "rebalance": "rebalancement",
    "roll": "roulement",
    "deploy_idle_cash": "redéploiement du cash oisif",
    "delever": "dé-levier",
    "stop_loss": "stop-loss",
    "take_profit": "take-profit",
    "signal_lost": "signal perdu",
    "direction_flip": "retournement du signal",
    "expiry": "expiration",
    "data_gap": "cours du sous-jacent indisponible",
}


def _contract_label(row) -> str:
    """« CALL 31.75 éch. 2014-01-17 » -- ce qui identifie le CONTRAT, et pas
    seulement le sous-jacent. Chaîne vide sur un run actions.

    Colonne à part entière (et non fondue dans le motif) : c'est la maille à
    laquelle un achat répond à une vente, donc celle sur laquelle on veut
    pouvoir trier et filtrer. Le strike est arrondi au centime -- le moteur
    calcule un strike théorique en flottant, dont les douze décimales ne
    disent rien de plus que les deux premières."""
    option_type = row.get("option_type")
    if option_type is None or not pd.notna(option_type):
        return ""
    parts = [str(option_type)]
    strike = row.get("strike")
    if strike is not None and pd.notna(strike):
        parts.append(f"{float(strike):.2f}")
    expiry = row.get("expiry")
    if expiry is not None and pd.notna(expiry):
        parts.append(f"éch. {pd.Timestamp(expiry):%Y-%m-%d}")
    return " ".join(parts)


def _drop_empty_contract_column(log: pd.DataFrame) -> pd.DataFrame:
    """Un run actions n'a pas de contrat à montrer : plutôt qu'une colonne
    vide, pas de colonne."""
    if "contrat" in log.columns and not log["contrat"].astype(bool).any():
        return log.drop(columns=["contrat"])
    return log


def _contract_key(df: pd.DataFrame) -> list[str]:
    """Colonnes qui identifient un contrat dans le journal des exécutions.

    Le symbole NE SUFFIT PAS : un roulement clôture un contrat et en rouvre un
    autre sur le même sous-jacent le même jour. Strike et échéance ne sont
    présents que sur les runs postérieurs à leur ajout au journal -- sur un run
    plus ancien, on retombe sur (symbole, sens), ce qui laisse le solde
    fusionner les contrats successifs."""
    return [c for c in ("symbol", "option_type", "strike", "expiry") if c in df.columns]


def _last_known_gap(sig: pd.DataFrame, symbol: str, date: pd.Timestamp) -> Optional[float]:
    if sig.empty:
        return None
    candidates = sig[(sig["symbol"] == symbol) & (sig["date"] <= date)]
    if candidates.empty:
        return None
    gap = candidates.iloc[-1].get("gap_pct")
    return float(gap) if pd.notna(gap) else None


def _sorted_signals(signals_history: pd.DataFrame) -> pd.DataFrame:
    if signals_history is None or signals_history.empty:
        return pd.DataFrame()
    sig = signals_history.copy()
    sig["date"] = pd.to_datetime(sig["date"])
    return sig.sort_values(["symbol", "date"])


def build_trade_log(
    positions_history: pd.DataFrame,
    trades: pd.DataFrame,
    signals_history: pd.DataFrame,
    executions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Journal achats/ventes, pour affichage ("quand, quoi, pourquoi").

    DEUX SOURCES POSSIBLES, très inégales.

    1. executions.parquet (runs options) : une ligne par fill RÉELLEMENT
       EXÉCUTÉ, achat comme vente, avec sa quantité en CONTRATS, son contrat
       (sens/strike/échéance) et son motif. C'est la comptabilité du moteur
       elle-même ; le journal en est alors une simple mise en forme, et la
       colonne `solde` (contrats détenus après le fill, contrat par contrat)
       se vérifie ligne à ligne.

    2. À défaut (runs actions, ou runs options antérieurs au journal) :
       reconstruction depuis positions_history, puisque trades.parquet ne
       logue que les VENTES (cf. engine._execute_trade / options_engine.
       _reduce_position). Les achats y sont déduits des hausses quotidiennes
       de quantité détenue. Approximation acceptable pour les actions -- un
       symbole = un instrument -- mais FAUSSE sur les options, où un
       roulement remplace un contrat par un autre le même jour sans variation
       nette de quantité : l'achat du contrat neuf disparaît, et sa vente
       ultérieure semble alors porter sur des contrats jamais achetés.

    C'est ce qu'un rapport d'incohérence a mis au jour (33 contrats achetés,
    3 300 « vendus »), doublé d'une confusion d'unités : trades.parquet porte
    à la fois `contracts` et `shares` = contracts x 100, et le journal lisait
    `shares` côté vente contre des contrats côté achat."""
    if executions is not None and not executions.empty:
        return _trade_log_from_executions(executions, trades, signals_history)
    return _trade_log_from_positions(positions_history, trades, signals_history)


def _trade_log_from_executions(
    executions: pd.DataFrame, trades: pd.DataFrame, signals_history: pd.DataFrame,
) -> pd.DataFrame:
    ex = executions.copy()
    ex["date"] = pd.to_datetime(ex["date"])
    if "expiry" in ex.columns:
        ex["expiry"] = pd.to_datetime(ex["expiry"])

    # L'ORDRE DES LIGNES EST L'ORDRE D'EXÉCUTION du moteur, et il porte de
    # l'information : plusieurs fills tombent le même jour (les ventes sont
    # exécutées avant les achats qu'elles financent, cf.
    # options_engine._execution_order), et un roulement y est une vente suivie
    # d'un achat à la même date. Le tri d'affichage est par date : on garde
    # donc le rang d'origine pour départager, et on calcule le solde
    # AVANT de trier.
    ex = ex.reset_index(drop=True)
    ex["_seq"] = np.arange(len(ex))

    ex["_signed"] = ex["contracts"] * ex["side"].map({"buy": 1.0, "sell": -1.0})
    # dropna=False : strike/échéance sont nuls sur un run antérieur à leur
    # ajout au journal -- sans cela, pandas écarterait ces lignes du
    # regroupement et le solde y serait vide plutôt qu'approché.
    ex["solde"] = ex.groupby(_contract_key(ex), dropna=False)["_signed"].cumsum()
    ex["solde_avant"] = ex["solde"] - ex["_signed"]

    pnl, return_pct = _attach_sell_pnl(ex, trades)
    sig = _sorted_signals(signals_history)

    rows: list[dict] = []
    for i, e in ex.iterrows():
        label = EXECUTION_REASON_LABELS.get(e["reason"], str(e["reason"]))
        if e["side"] == "buy":
            if e["solde_avant"] > 1e-9:
                raison = f"Renforcement — {label}"
            elif e["reason"] == "roll":
                # Un roulement n'exprime AUCUN signal neuf : c'est la même
                # thèse reconduite sur une échéance pleine. Lui accoler le
                # dernier écart connu (vieux de plusieurs mois) laisserait
                # croire à une décision qui n'a pas été prise.
                raison = "Ouverture du contrat de remplacement — roulement"
            else:
                raison = f"Ouverture — {label}"
                gap = _last_known_gap(sig, e["symbol"], e["date"])
                if gap is not None:
                    sens = "sous-évaluation" if gap > 0 else "survalorisation"
                    raison = f"Ouverture — signal {sens} (écart {gap:.1f}%), {label}"
        else:
            raison = f"Clôture — {label}" if e["solde"] <= 1e-9 else f"Réduction — {label}"
        rows.append({
            "date": e["date"], "symbol": e["symbol"], "contrat": _contract_label(e),
            "action": "Achat" if e["side"] == "buy" else "Vente",
            "quantite": e["contracts"], "prix": e["price"],
            "raison": raison, "solde": e["solde"],
            "pnl": pnl[i], "return_pct": return_pct[i],
        })

    log = pd.DataFrame(rows)
    # Tri d'affichage : le plus récent d'abord, l'ordre d'exécution du moteur
    # départageant les fills d'une même journée (mergesort = tri stable).
    log = log.sort_values("date", ascending=False, kind="mergesort").reset_index(drop=True)
    return _drop_empty_contract_column(log)


def _attach_sell_pnl(ex: pd.DataFrame, trades: pd.DataFrame) -> tuple[list, list]:
    """P&L et rendement d'une vente, repris de trades.parquet.

    APPARIEMENT POSITIONNEL, et non par jointure sur (symbole, date) : une
    même position peut être réduite deux fois le même jour, ce qu'aucune clé
    disponible ne distingue. options_engine._reduce_position écrit ses deux
    lignes -- le fill puis le trade -- dans la foulée, sans autre chemin de
    code alimentant `trades` : la n-ième vente du journal est donc le n-ième
    trade. L'hypothèse est VÉRIFIÉE (nombre de lignes et symboles) avant
    d'être utilisée ; à défaut, les colonnes restent vides plutôt que
    fausses."""
    n = len(ex)
    pnl: list = [None] * n
    return_pct: list = [None] * n
    sell_positions = ex.index[ex["side"] == "sell"].tolist()
    if trades is None or trades.empty or len(sell_positions) != len(trades):
        return pnl, return_pct

    tr = trades.reset_index(drop=True)
    if not (ex.loc[sell_positions, "symbol"].to_numpy() == tr["symbol"].to_numpy()).all():
        return pnl, return_pct
    if "contracts" in tr.columns:
        ecart = np.abs(ex.loc[sell_positions, "contracts"].to_numpy() - tr["contracts"].to_numpy())
        if ecart.max() > 1e-6:
            return pnl, return_pct

    for k, i in enumerate(sell_positions):
        pnl[i] = tr.at[k, "pnl"] if "pnl" in tr.columns else None
        return_pct[i] = tr.at[k, "return_pct"] if "return_pct" in tr.columns else None
    return pnl, return_pct


def _trade_log_from_positions(
    positions_history: pd.DataFrame, trades: pd.DataFrame, signals_history: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruction de repli (cf. build_trade_log) : sans journal de fills,
    les achats se déduisent des hausses de quantité détenue."""
    rows: list[dict] = []

    if trades is not None and not trades.empty:
        # MÊME UNITÉ DES DEUX CÔTÉS. Un trade options porte `contracts` ET
        # `shares` = contracts x multiplicateur (100) ; les achats
        # reconstruits plus bas sont, eux, en contrats. Lire `shares` ici
        # affichait donc des ventes 100 fois plus grosses que les achats
        # correspondants.
        qty_col = "contracts" if "contracts" in trades.columns else "shares"
        for _, t in trades.iterrows():
            motif = t.get("exit_reason", "?")
            rows.append({
                "date": t["exit_date"], "symbol": t["symbol"], "contrat": _contract_label(t),
                "action": "Vente",
                "quantite": t.get(qty_col), "prix": t.get("exit_price"),
                "raison": EXECUTION_REASON_LABELS.get(motif, motif),
                "pnl": t.get("pnl"), "return_pct": t.get("return_pct"),
            })

    if positions_history is not None and not positions_history.empty:
        ph = positions_history.sort_values(["symbol", "date"]).copy()
        qty_col = _first_present(ph, ("shares", "contracts"))
        price_col = _first_present(ph, ("price", "premium"))

        ph["prev_qty"] = ph.groupby("symbol")[qty_col].shift(1).fillna(0.0)
        buys = ph[ph[qty_col] > ph["prev_qty"] + 1e-9].copy()
        buys["qty_bought"] = buys[qty_col] - buys["prev_qty"]
        buys["is_new_entry"] = buys["prev_qty"] <= 1e-9

        sig = _sorted_signals(signals_history)

        for _, b in buys.iterrows():
            if b["is_new_entry"]:
                reason = "Nouvelle position (rebalancement)"
                gap = _last_known_gap(sig, b["symbol"], pd.Timestamp(b["date"]))
                if gap is not None:
                    sens = "sous-évaluation" if gap > 0 else "survalorisation"
                    reason = f"Signal {sens} (écart {gap:.1f}%)"
            else:
                reason = "Renforcement (rebalancement)"
            rows.append({
                "date": b["date"], "symbol": b["symbol"], "contrat": _contract_label(b),
                "action": "Achat",
                "quantite": b["qty_bought"], "prix": b.get(price_col),
                "raison": reason, "pnl": None, "return_pct": None,
            })

    if not rows:
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)

    log = pd.DataFrame(rows)
    log["date"] = pd.to_datetime(log["date"])
    log = log.sort_values("date", ascending=False).reset_index(drop=True)
    return _drop_empty_contract_column(log)
