"""
Assemble les données produites par le pipeline (01b, 03b, 07) en trois objets
consommés par backtest/engine.py :

    - prices  : cours quotidiens OHLC, pivotés (date x symbol), avec
                forward-fill borné (voir FORWARD_FILL_MAX_DAYS) pour absorber
                les jours fériés/gaps mineurs sans masquer une vraie sortie
                de cotation.
    - signals : événements de publication de l'écart DCF (une ligne par
                (symbol, date de publication) = filed_date du 10-K, PAS la
                date de clôture d'exercice) -- c'est cette date qui pilote le
                moteur (aucune info n'est utilisée avant sa date réelle de
                publication).
    - universe: historique d'appartenance au S&P 500 (01b), pour ne
                considérer comme candidates à l'ENTRÉE que les entreprises
                réellement membres de l'indice à la date courante (voir
                universe_asof). Les positions déjà ouvertes ne sont PAS
                clôturées de force si l'entreprise sort de l'indice --
                seuls stop-loss/take-profit et une disparition des données
                de prix ferment une position (cf. engine.py).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

import config

logger = logging.getLogger("backtest.data_loader")

# Un symbole sans nouvelle donnée de prix pendant plus de FORWARD_FILL_MAX_DAYS
# jours de calendrier est considéré comme ayant disparu des données (radiation,
# fusion...) plutôt que comme un simple jour férié/gap de collecte : toute
# position ouverte dessus est alors forcée à la clôture par l'engine (voir
# BacktestEngine._handle_stale_symbols), au lieu d'être silencieusement
# marked-to-market sur un prix de plus en plus périmé.
FORWARD_FILL_MAX_DAYS = 10

# Un 10-K est en pratique déposé dans les ~60-90 jours suivant la clôture de
# l'exercice (délai SEC selon la catégorie de filer). Utilisé UNIQUEMENT en
# repli quand filed_date est manquant (ex: financials.parquet mis en cache
# avant l'ajout de cette colonne à 04_recuperation_10k.py) -- approximation
# documentée, pas une vérité SEC.
DEFAULT_FILING_LAG_DAYS = 75


def load_daily_prices(path=None) -> pd.DataFrame:
    path = path or config.DAILY_PRICES_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Lance d'abord 03b_recuperation_cours_quotidiens.py."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def build_price_panel(daily_prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pivote les cours quotidiens en deux tables larges (index=date,
    colonnes=symbol) : 'close' (forward-fillée, bornée à
    FORWARD_FILL_MAX_DAYS) et 'open' (NON forward-fillée : un ordre ne doit
    jamais s'exécuter sur un prix d'ouverture périmé -- si l'open du jour est
    manquant pour un symbole, l'exécution de ses ordres est simplement
    reportée au jour suivant par l'engine).

    Retourne aussi 'last_valid_date' : dernière date avec un cours RÉEL
    (non comblé) par symbole, utilisée par l'engine pour détecter les
    symboles dont les données se sont arrêtées (cf. FORWARD_FILL_MAX_DAYS)."""
    close_raw = daily_prices.pivot(index="date", columns="symbol", values="close").sort_index()
    open_raw = daily_prices.pivot(index="date", columns="symbol", values="open").sort_index()

    last_valid_date = close_raw.apply(lambda col: col.last_valid_index())

    close_ffill = close_raw.ffill(limit=FORWARD_FILL_MAX_DAYS)
    return {"close": close_ffill, "open": open_raw, "last_valid_date": last_valid_date}


def _fill_missing_filed_dates(df: pd.DataFrame, path) -> pd.DataFrame:
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
    missing_filed = df["filed_date"].isna()
    if missing_filed.any():
        df.loc[missing_filed, "filed_date"] = pd.to_datetime(
            df.loc[missing_filed, "year"].astype(int).map(lambda y: date(y + 1, 4, 1))
        )
        logger.warning(
            "%d/%d lignes de %s sans filed_date (cache antérieur à l'ajout de cette "
            "colonne dans 04) : approximé à year+1-04-01 (délai de dépôt SEC par défaut, "
            "~%d jours après la clôture). Relance 04_recuperation_10k.py --force-refresh "
            "pour une date réelle.",
            missing_filed.sum(), len(df), path, DEFAULT_FILING_LAG_DAYS,
        )
    return df


def load_dcf_history(path=None) -> pd.DataFrame:
    path = path or config.DCF_HISTORY_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Lance d'abord 07_calcul_dcf.py (avec 04 déjà à jour)."
        )
    df = _fill_missing_filed_dates(pd.read_parquet(path), path)
    return df.dropna(subset=["gap_pct", "symbol"]).sort_values(["symbol", "filed_date"]).reset_index(drop=True)


def build_signal_events(dcf_history: pd.DataFrame) -> pd.DataFrame:
    """Un événement par (symbol, filed_date) -- la date à laquelle ce
    signal devient publiquement connu. Colonnes : symbol, published_date,
    fiscal_year, sector, close_at_filing, valuation_dcf_per_share, gap_pct."""
    return dcf_history.rename(columns={
        "filed_date": "published_date", "year": "fiscal_year", "close": "close_at_filing",
    })[["symbol", "published_date", "fiscal_year", "sector", "close_at_filing", "valuation_dcf_per_share", "gap_pct"]]


def load_valorisation_combinee_history(path=None) -> pd.DataFrame:
    """Signal de la stratégie OPTIONS (multiples sectoriels par année en
    priorité, DCF en repli -- voir 06b_calcul_valorisation_combinee.py),
    distinct de load_dcf_history (stratégie actions, DCF seul)."""
    path = path or config.VALORISATION_COMBINEE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Lance d'abord 06b_calcul_valorisation_combinee.py "
            "(après 05_calcul_multiples.py et 07_calcul_dcf.py)."
        )
    df = _fill_missing_filed_dates(pd.read_parquet(path), path)
    return df.dropna(subset=["gap_pct", "symbol"]).sort_values(["symbol", "filed_date"]).reset_index(drop=True)


def build_options_signal_events(valorisation_combinee: pd.DataFrame) -> pd.DataFrame:
    """Un événement par (symbol, filed_date). gap_pct garde son signe (positif
    = sous-évalué -> call, négatif = survalué -> put) : c'est la stratégie
    (backtest/strategies/valuation_gap_options.py) qui décide du sens."""
    return valorisation_combinee.rename(columns={"filed_date": "published_date", "year": "fiscal_year"})[[
        "symbol", "published_date", "fiscal_year", "sector", "close",
        "valuation_theoretical_per_share", "source", "gap_pct",
    ]]


def load_universe_history(path=None) -> Optional[pd.DataFrame]:
    """None si 01b_historique_univers_sp500.py n'a jamais été lancé -- pas
    une erreur bloquante : l'engine retombe alors sur l'univers ACTUEL
    (config.UNIVERSE_FILE) appliqué à toutes les dates, en le signalant une
    fois comme biais de survivance connu plutôt que de planter."""
    path = path or config.UNIVERSE_HISTORY_FILE
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    return df


def universe_asof(history: Optional[pd.DataFrame], as_of: pd.Timestamp, fallback_symbols: set[str]) -> set[str]:
    if history is None or history.empty:
        return fallback_symbols
    start_ok = history["start_date"].isna() | (history["start_date"] <= as_of)
    end_ok = history["end_date"].isna() | (history["end_date"] > as_of)
    return set(history.loc[start_ok & end_ok, "ric"].map(config.to_ib_symbol))


def load_current_universe_symbols() -> set[str]:
    universe = pd.read_csv(config.UNIVERSE_FILE, encoding="utf-8-sig")
    return set(universe["RIC"].dropna().map(config.to_ib_symbol))


def load_option_snapshots_history(directory=None) -> pd.DataFrame:
    """Concatène TOUS les snapshots archivés par 08_recuperation_options.py
    (data/options/history/option_chains_*.parquet, un fichier par run,
    jamais écrasé). DataFrame vide si 08 n'a encore jamais tourné -- pas une
    erreur : backtest/options_engine.py retombe alors entièrement sur le
    pricing Black-Scholes simulé (voir find_real_option_snapshot)."""
    directory = directory or config.DIR_OPTIONS_HISTORY
    files = sorted(directory.glob("option_chains_*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["expiry"] = pd.to_datetime(df["expiry"], format="%Y%m%d", errors="coerce")
    return df


def find_real_option_snapshot(
    option_snapshots: pd.DataFrame,
    symbol: str,
    option_type: str,
    as_of: pd.Timestamp,
    tolerance_days: int = config.OPTIONS_REAL_SNAPSHOT_TOLERANCE_DAYS,
) -> Optional[dict]:
    """Contrat réel le plus représentatif pour (symbol, option_type) autour
    de as_of, ou None si aucun snapshot 08 n'existe dans la fenêtre de
    tolérance (l'appelant doit alors simuler par Black-Scholes).

    Sélection en deux temps : (1) le snapshot_date le plus proche de as_of,
    dans la fenêtre de tolérance ; (2) parmi les contrats de cette date pour
    ce symbole/type, celui le plus proche de la monnaie (min |moneyness_pct|),
    départagé par l'échéance la plus proche de OPTIONS_TARGET_TENOR_DAYS --
    cohérent avec la sélection déjà faite par 08 (ATM, >9 mois)."""
    if option_snapshots.empty:
        return None
    candidates = option_snapshots[
        (option_snapshots["symbol"] == symbol) & (option_snapshots["option_type"] == option_type)
    ]
    if candidates.empty:
        return None

    date_diff = (candidates["snapshot_date"] - as_of).abs()
    within_tolerance = candidates[date_diff <= pd.Timedelta(days=tolerance_days)]
    if within_tolerance.empty:
        return None

    closest_date = within_tolerance.loc[date_diff[within_tolerance.index].idxmin(), "snapshot_date"]
    same_date = within_tolerance[within_tolerance["snapshot_date"] == closest_date].copy()

    same_date["_moneyness_abs"] = same_date["moneyness_pct"].abs()
    same_date["_tenor_diff"] = (same_date["expiry"] - closest_date).dt.days.sub(config.OPTIONS_TARGET_TENOR_DAYS).abs()
    best = same_date.sort_values(["_moneyness_abs", "_tenor_diff"]).iloc[0]

    bid, ask = best.get("bid"), best.get("ask")
    if pd.notna(bid) and pd.notna(ask):
        premium = (bid + ask) / 2
    else:
        premium = best.get("last_price") if pd.notna(best.get("last_price")) else best.get("close")

    return {
        "strike": best["strike"], "expiry": best["expiry"], "premium": premium,
        "implied_vol": best.get("implied_vol"), "delta": best.get("delta"), "gamma": best.get("gamma"),
        "vega": best.get("vega"), "theta": best.get("theta"), "underlying_spot": best.get("underlying_spot"),
        "multiplier": float(best.get("multiplier") or config.OPTIONS_CONTRACT_MULTIPLIER),
        "snapshot_date": closest_date, "source": "real",
    }
