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

# `executions` n'existe que pour la stratégie OPTIONS, et seulement depuis que
# 10_backtest_options.py l'écrit : c'est le journal fill par fill (achats ET
# ventes) du moteur, la seule sortie qui permette de vérifier la conservation
# des quantités -- trades.parquet, lui, ne logue que les ventes. Les tables
# absentes deviennent un DataFrame vide (cf. load_backtest_run), donc l'ajouter
# ici ne casse ni les runs actions ni les runs options antérieurs.
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
    """Charge les tables Parquet (cf. BACKTEST_RUN_FILES) + les 2 JSON écrits
    par 09/10 en fin de run (même schéma de fichiers dans les deux cas, cf.
    leurs main()). Tables absentes (run antérieur à un ajout de fichier) ->
    DataFrame vide, pas une erreur."""
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


# Colonnes de quantité des sorties de backtest, PAR ORDRE DE PRIORITÉ.
#
# Le moteur options écrit LES DEUX dans trades.parquet : `contracts` (le nombre
# de contrats réellement négociés) et `shares` (contracts x multiplicateur,
# c'est-à-dire 100 fois plus pour un contrat standard). Le moteur actions
# n'écrit que `shares`, qui y est bien un nombre d'actions. Lire `shares` en
# priorité revenait donc, côté options, à afficher des VENTES exprimées en
# actions sous-jacentes face à des ACHATS comptés en contrats : à unités
# mélangées, la somme cumulée des achats moins les ventes plongeait aussitôt
# dans le négatif alors que le moteur n'a jamais vendu un contrat qu'il ne
# détenait pas (invariant vérifié par tests/test_journal_executions.py).
_QTY_COLUMNS = ("contracts", "shares")
_PRICE_COLUMNS = ("premium", "price")

# Motifs d'ACHAT du moteur (colonne `reason` du journal des exécutions), rendus
# lisibles. Les motifs de VENTE restent affichés tels quels (stop_loss,
# take_profit, expiry, data_gap...), comme avant : ce sont ceux que les scripts
# 10/12/13 emploient déjà dans leurs propres sorties.
_BUY_REASON_LABELS = {
    "rebalance": "Rebalancement (dépôt SEC)",
    "rebalance_daily": "Rebalancement journalier",
    "roll": "Roulement d'échéance",
    "deploy_idle_cash": "Redéploiement du cash oisif",
}

TRADE_LOG_COLUMNS = ["date", "symbol", "action", "quantite", "prix", "raison", "solde", "pnl", "return_pct"]


def _contract_detail(option_type, strike) -> str:
    """Suffixe ' (CALL 123.4)' pour une ligne d'options, vide pour une ligne
    d'actions."""
    if option_type is None or not pd.notna(option_type):
        return ""
    if strike is None or not pd.notna(strike):
        return f" ({option_type})"
    return f" ({option_type} {strike:g})"


def _signal_lookup(signals_history: pd.DataFrame) -> dict:
    """{symbole: (dates triées, gap_pct)} pour retrouver en O(log n) le dernier
    signal de valorisation connu à une date donnée.

    Le journal des exécutions compte une ligne par fill (le redéploiement du
    cash oisif en produit un par jour de bourse et par position) : un filtrage
    du DataFrame de signaux par achat, comme le faisait la version précédente,
    y devient quadratique."""
    if signals_history is None or signals_history.empty or "gap_pct" not in signals_history.columns:
        return {}
    sig = signals_history.dropna(subset=["gap_pct"]).copy()
    if sig.empty:
        return {}
    sig["date"] = pd.to_datetime(sig["date"])
    sig = sig.sort_values("date")
    return {
        symbol: (grp["date"].to_numpy(), grp["gap_pct"].to_numpy())
        for symbol, grp in sig.groupby("symbol")
    }


def _entry_reason(lookup: dict, symbol, date, detail: str) -> str:
    """Raison d'une OUVERTURE : le dernier écart de valorisation connu à cette
    date quand il y en a un, sinon un libellé neutre."""
    entry = lookup.get(symbol)
    if entry is not None:
        dates, gaps = entry
        i = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(date)), side="right"))
        if i > 0:
            gap = gaps[i - 1]
            sens = "sous-évaluation" if gap > 0 else "survalorisation"
            return f"Signal {sens} (écart {gap:.1f}%){detail}"
    return f"Nouvelle position{detail}"


def _sell_rows(trades: pd.DataFrame) -> list[dict]:
    """Une ligne par VENTE, depuis trades.parquet (qui n'enregistre que
    celles-là) : la raison de sortie, le P&L et le rendement n'existent que
    là."""
    if trades is None or trades.empty:
        return []
    qty_col = _first_present(trades, _QTY_COLUMNS)
    if qty_col is None:
        return []
    has_options = "option_type" in trades.columns
    return [
        {
            "date": t["exit_date"], "symbol": t["symbol"], "action": "Vente",
            "quantite": float(t[qty_col]), "prix": t.get("exit_price"),
            "raison": f"{t.get('exit_reason', '?')}"
                      f"{_contract_detail(t.get('option_type'), t.get('strike')) if has_options else ''}",
            "pnl": t.get("pnl"), "return_pct": t.get("return_pct"),
        }
        for _, t in trades.iterrows()
    ]


def _buy_rows_from_executions(
    executions: pd.DataFrame, positions_history: pd.DataFrame, lookup: dict,
) -> Optional[list[dict]]:
    """Une ligne par ACHAT réellement exécuté, lue au journal des exécutions du
    moteur options (executions.parquet : une ligne par fill, achat comme
    vente, quantité en CONTRATS). None si le run n'a pas ce journal.

    C'est la source exacte, et la raison pour laquelle le moteur l'écrit :
    aucun achat n'a besoin d'être deviné. Les motifs y sont ceux du moteur
    (rebalance, roll, deploy_idle_cash...), donc un renforcement, un roulement
    et un redéploiement de cash cessent d'être confondus."""
    if executions is None or executions.empty or "side" not in executions.columns:
        return None
    if not (executions["side"] == "buy").any():
        return []

    # Strike du contrat acheté : absent du journal, mais présent dans
    # positions_history à la date de l'achat (la position y est enregistrée en
    # fin de journée, donc APRÈS le fill -- y compris le jour d'un roulement,
    # où c'est bien le NOUVEAU contrat qui figure).
    strikes: dict = {}
    if positions_history is not None and not positions_history.empty and "strike" in positions_history.columns:
        ph = positions_history.copy()
        ph["date"] = pd.to_datetime(ph["date"])
        strikes = ph.set_index(["symbol", "date"])["strike"].to_dict()

    # Le journal est parcouru EN ENTIER, ventes comprises, pour savoir si un
    # achat ouvre une position ou en renforce une : ne suivre que les achats
    # laisserait le solde à zéro seulement au tout premier, et présenterait
    # comme un renforcement toute ré-entrée après une sortie complète.
    held: dict = {}
    rows = []
    for _, e in executions.iterrows():
        symbol, qty = e["symbol"], float(e["contracts"])
        if e["side"] != "buy":
            held[symbol] = held.get(symbol, 0.0) - qty
            continue
        detail = _contract_detail(e.get("option_type"), strikes.get((symbol, pd.Timestamp(e["date"]))))
        reason = e.get("reason")
        if reason == "roll":
            # Un roulement solde le contrat arrivé à son point de décision et
            # en rouvre un immédiatement : le solde passe bien par zéro entre
            # les deux, mais l'annoncer comme une ouverture masquerait le fait
            # que c'est la MÊME thèse qui continue, à exposition inchangée.
            raison = f"{_BUY_REASON_LABELS['roll']}{detail}"
        elif held.get(symbol, 0.0) <= 1e-9:
            raison = _entry_reason(lookup, symbol, e["date"], detail)
        else:
            raison = f"Renforcement — {_BUY_REASON_LABELS.get(reason, reason)}{detail}"
        held[symbol] = held.get(symbol, 0.0) + qty
        rows.append({
            "date": e["date"], "symbol": symbol, "action": "Achat",
            "quantite": qty, "prix": e.get("price"),
            "raison": raison, "pnl": None, "return_pct": None,
        })
    return rows


def _buy_rows_from_positions_history(
    positions_history: pd.DataFrame, trades: pd.DataFrame, lookup: dict,
) -> list[dict]:
    """Achats RECONSTRUITS, pour les runs sans journal des exécutions (toute la
    stratégie actions, et les runs options antérieurs au journal).

    positions_history n'enregistre que les positions ENCORE OUVERTES en fin de
    journée : une position soldée n'y laisse aucune ligne. Comparer la quantité
    du jour à celle de la ligne précédente du même symbole -- ce que faisait la
    version précédente -- enjambe donc les périodes où la position n'existait
    pas, et rate tout achat qui ne fait pas monter le solde de fin de journée :

        - réouverture après une sortie complète (stop, expiration, perte de
          signal) : la ligne précédente est celle d'AVANT la sortie, souvent
          plus grosse, donc la ré-entrée passe pour un allègement ;
        - roulement d'échéance : le moteur solde N contrats et en rouvre M le
          même jour ; si M <= N, l'achat de M contrats disparaît ;
        - toute vente partielle suivie d'un renfort le même jour.

    Ces achats manquants étaient bien vendus plus tard, eux : d'où des soldes
    cumulés négatifs, qui ne reflétaient aucune vente à découvert.

    La reconstruction repose donc sur la conservation des quantités, qui
    n'enjambe rien puisqu'elle réintègre les ventes du jour :

        achats(J) = détenu(J) - détenu(J-1) + ventes(J)

    où détenu(J) vaut 0 pour toute date sans ligne dans positions_history."""
    if positions_history is None or positions_history.empty:
        return []
    qty_col = _first_present(positions_history, _QTY_COLUMNS)
    if qty_col is None:
        return []
    price_col = _first_present(positions_history, _PRICE_COLUMNS)
    has_options = "option_type" in positions_history.columns

    ph = positions_history.copy()
    ph["date"] = pd.to_datetime(ph["date"])
    ph = ph.sort_values(["symbol", "date"])

    # Ventes agrégées par (symbole, date), dans la MÊME unité que qty_col : les
    # deux tables viennent du même moteur, donc `contracts` face à `contracts`
    # (options) ou `shares` face à `shares` (actions).
    aucune_vente = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    sold: dict[str, pd.Series] = {}
    if trades is not None and not trades.empty:
        trade_qty_col = _first_present(trades, _QTY_COLUMNS)
        if trade_qty_col is not None:
            sells = trades[["symbol", "exit_date", trade_qty_col]].copy()
            sells["exit_date"] = pd.to_datetime(sells["exit_date"])
            sold = {
                symbol: grp.groupby("exit_date")[trade_qty_col].sum()
                for symbol, grp in sells.groupby("symbol", sort=False)
            }

    rows = []
    for symbol, grp in ph.groupby("symbol", sort=False):
        held = grp.groupby("date")[qty_col].sum()
        sold_sym = sold.get(symbol, aucune_vente)
        # Les dates de VENTE comptent même sans ligne de position : c'est
        # précisément là que la position a pu être soldée en entier.
        dates = held.index.union(sold_sym.index)
        held = held.reindex(dates).fillna(0.0)
        previous = held.shift(1).fillna(0.0)
        bought = held - previous + sold_sym.reindex(dates).fillna(0.0)

        detail_by_date = (
            grp.set_index("date")[["option_type", "strike"]].to_dict("index") if has_options else {}
        )
        price_by_date = grp.set_index("date")[price_col].to_dict() if price_col else {}

        for date, qty in bought[bought > 1e-9].items():
            contract = detail_by_date.get(date, {})
            detail = _contract_detail(contract.get("option_type"), contract.get("strike"))
            if previous[date] <= 1e-9:
                raison = _entry_reason(lookup, symbol, date, detail)
            else:
                raison = f"Renforcement (rebalancement){detail}"
            rows.append({
                "date": date, "symbol": symbol, "action": "Achat",
                "quantite": float(qty), "prix": price_by_date.get(date),
                "raison": raison, "pnl": None, "return_pct": None,
            })
    return rows


def build_trade_log(
    positions_history: pd.DataFrame,
    trades: pd.DataFrame,
    signals_history: pd.DataFrame,
    executions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Journal achats/ventes unifié, pour affichage ("quand et pourquoi"), avec
    la quantité détenue après chaque exécution (colonne `solde`).

    Les VENTES viennent de trades.parquet, seule sortie qui porte leur raison
    (exit_reason : stop_loss/take_profit/roll/signal_lost/expiry/data_gap), leur
    P&L et leur rendement.

    Les ACHATS n'y sont pas : le moteur ne loggue explicitement que les ventes
    (cf. BacktestEngine._execute_trade, OptionsBacktestEngine._reduce_position).
    Ils sont donc lus au journal des exécutions du moteur options quand le run
    en a un (executions.parquet, exact et fill par fill), et RECONSTRUITS depuis
    positions_history sinon -- voir _buy_rows_from_positions_history pour les
    pièges de cette reconstruction.

    Les deux schémas (actions : shares/price ; options : contracts/premium,
    option_type/strike en plus) sont gérés sans être fusionnés en un schéma
    commun : seules les colonnes nécessaires à l'affichage sont lues, dans
    l'unité du contrat négocié (contrats pour les options, actions pour les
    actions -- jamais les deux dans le même journal, cf. _QTY_COLUMNS)."""
    lookup = _signal_lookup(signals_history)
    buys = _buy_rows_from_executions(executions, positions_history, lookup)
    if buys is None:
        buys = _buy_rows_from_positions_history(positions_history, trades, lookup)

    rows = _sell_rows(trades) + buys
    if not rows:
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)

    log = pd.DataFrame(rows)
    log["date"] = pd.to_datetime(log["date"])

    # ORDRE CHRONOLOGIQUE, établi une seule fois : à date égale, les ventes sont
    # comptées AVANT les achats, comme le moteur les exécute (cf.
    # OptionsBacktestEngine._execution_order : le produit d'une vente finance
    # les achats du même jour) -- c'est aussi l'ordre d'un roulement, qui solde
    # le contrat arrivé à son point de décision avant de rouvrir le suivant. Le
    # tri est STABLE, donc à date et sens égaux les lignes gardent l'ordre
    # d'exécution du moteur.
    log["_ordre"] = log["action"].map({"Vente": 0, "Achat": 1})
    log = log.sort_values(["date", "_ordre"], kind="stable")

    # SOLDE : quantité détenue par symbole après chaque exécution. C'est le
    # cumul que ce journal doit rendre vérifiable d'un coup d'oeil -- il ne peut
    # pas devenir négatif, le backtest étant non margé (aucune vente à
    # découvert).
    signed = log["quantite"] * log["action"].map({"Achat": 1.0, "Vente": -1.0})
    log["solde"] = signed.groupby(log["symbol"], sort=False).cumsum()

    # Affichage du plus récent au plus ancien : l'EXACT inverse de l'ordre qui
    # vient de servir au cumul. Un second sort_values ne le garantirait pas --
    # un tri stable descendant laisse les ex aequo (même jour, même sens) dans
    # leur ordre croissant, et la colonne `solde` se lirait alors à rebours.
    return log[TRADE_LOG_COLUMNS].iloc[::-1].reset_index(drop=True)
