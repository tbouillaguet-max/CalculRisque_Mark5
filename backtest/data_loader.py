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
                UniverseResolver). Les positions déjà ouvertes ne sont PAS
                clôturées de force si l'entreprise sort de l'indice --
                seuls stop-loss/take-profit et une disparition des données
                de prix ferment une position (cf. engine.py).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
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


class PricePanel:
    """Cours quotidiens pivotés (date x symbole), avec un accès unitaire en
    temps constant.

    Les moteurs de backtest interrogent un cours (symbole, date) des
    centaines de milliers de fois par run. Passer par pandas à chaque appel
    (`close.loc[date]` ou `close[symbol].get(date)`) reconstruit une Series
    intermédiaire à chaque fois et domine largement le temps de run : les
    valeurs sont donc aussi conservées en tableaux numpy bruts, adressés par
    deux dictionnaires date->ligne et symbole->colonne.

    Les tables pandas (`close`, `open`, `last_valid_date`) restent exposées
    telles quelles pour les usages non unitaires (calendrier, colonnes).
    """

    def __init__(self, close: pd.DataFrame, open_: pd.DataFrame, last_valid_date: pd.Series):
        self.close = close
        self.open = open_
        self.last_valid_date = last_valid_date

        self._close_values = close.to_numpy(dtype=float)
        self._open_values = open_.to_numpy(dtype=float)
        self._row_of_date = {ts: i for i, ts in enumerate(close.index)}
        self._close_col = {sym: j for j, sym in enumerate(close.columns)}
        self._open_col = {sym: j for j, sym in enumerate(open_.columns)}

    def row_of(self, date: pd.Timestamp) -> Optional[int]:
        return self._row_of_date.get(date)

    def close_at(self, symbol: str, date: pd.Timestamp) -> Optional[float]:
        """Clôture forward-fillée, ou None si le symbole/la date est absent
        du panel ou si la valeur est manquante."""
        row = self._row_of_date.get(date)
        col = self._close_col.get(symbol)
        if row is None or col is None:
            return None
        value = self._close_values[row, col]
        return None if value != value else float(value)

    def open_at(self, symbol: str, date: pd.Timestamp) -> Optional[float]:
        """Ouverture NON forward-fillée (cf. build_price_panel) : None dès
        que le symbole n'a pas coté ce jour-là."""
        row = self._row_of_date.get(date)
        col = self._open_col.get(symbol)
        if row is None or col is None:
            return None
        value = self._open_values[row, col]
        return None if value != value else float(value)

    def close_history(self, symbol: str, date: pd.Timestamp) -> np.ndarray:
        """Clôtures du symbole jusqu'à `date` INCLUSE (jamais au-delà : c'est
        ce découpage qui garantit le caractère point-in-time des indicateurs
        calculés dessus, cf. options_pricing.realized_volatility). Tableau
        vide si le symbole ou la date est inconnu."""
        row = self._row_of_date.get(date)
        col = self._close_col.get(symbol)
        if row is None or col is None:
            return np.empty(0)
        return self._close_values[: row + 1, col]


def build_price_panel(daily_prices: pd.DataFrame) -> PricePanel:
    """Pivote les cours quotidiens en deux tables larges (index=date,
    colonnes=symbol) : 'close' (forward-fillée, bornée à
    FORWARD_FILL_MAX_DAYS) et 'open' (NON forward-fillée : un ordre ne doit
    jamais s'exécuter sur un prix d'ouverture périmé -- si l'open du jour est
    manquant pour un symbole, l'exécution de ses ordres est simplement
    reportée au jour suivant par l'engine).

    Expose aussi 'last_valid_date' : dernière date avec un cours RÉEL
    (non comblé) par symbole, utilisée par l'engine pour détecter les
    symboles dont les données se sont arrêtées (cf. FORWARD_FILL_MAX_DAYS)."""
    close_raw = daily_prices.pivot(index="date", columns="symbol", values="close").sort_index()
    open_raw = daily_prices.pivot(index="date", columns="symbol", values="open").sort_index()

    last_valid_date = close_raw.apply(lambda col: col.last_valid_index())

    close_ffill = close_raw.ffill(limit=FORWARD_FILL_MAX_DAYS)
    return PricePanel(close_ffill, open_raw, last_valid_date)


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


class UniverseResolver:
    """Composition de l'indice à une date donnée, à partir des spans
    d'appartenance de 01b_historique_univers_sp500.py.

    Les bornes et les symboles sont convertis une seule fois (le mapping
    RIC -> symbole IBKR était refait à chaque appel), et le résultat est mis
    en cache par date : l'engine réinterroge la même date autant de fois
    qu'il y a de rebalancements ce jour-là.

    `history=None` (01b jamais lancé) -> l'univers ACTUEL est renvoyé pour
    toutes les dates : biais de survivance connu, signalé une fois par
    l'engine plutôt que bloquant."""

    def __init__(self, history: Optional[pd.DataFrame], fallback_symbols: set[str]):
        self.fallback_symbols = fallback_symbols
        self._has_history = history is not None and not history.empty
        if self._has_history:
            self._starts = history["start_date"].to_numpy(dtype="datetime64[ns]")
            self._ends = history["end_date"].to_numpy(dtype="datetime64[ns]")
            self._symbols = history["ric"].map(config.to_ib_symbol).to_numpy()
        self._cache: dict[pd.Timestamp, set[str]] = {}

    def asof(self, as_of: pd.Timestamp) -> set[str]:
        if not self._has_history:
            return self.fallback_symbols
        cached = self._cache.get(as_of)
        if cached is not None:
            return cached
        moment = np.datetime64(as_of.to_datetime64())
        start_ok = np.isnat(self._starts) | (self._starts <= moment)
        end_ok = np.isnat(self._ends) | (self._ends > moment)
        members = set(self._symbols[start_ok & end_ok])
        self._cache[as_of] = members
        return members


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


def _premium_of(row: pd.Series):
    bid, ask = row.get("bid"), row.get("ask")
    if pd.notna(bid) and pd.notna(ask):
        return (bid + ask) / 2
    return row.get("last_price") if pd.notna(row.get("last_price")) else row.get("close")


class OptionSnapshotIndex:
    """Index des snapshots réels archivés par 08_recuperation_options.py,
    interrogeable par (symbole, type d'option, date).

    Le contrat retenu pour un (symbole, type, snapshot_date) ne dépend pas de
    la date d'interrogation : il est donc choisi UNE FOIS ici, à la
    construction, plutôt qu'à chaque entrée en position -- le moteur filtrait
    et triait l'intégralité des snapshots à chaque candidat.

    Sélection, inchangée : le contrat le plus proche de la monnaie
    (min |moneyness_pct|), départagé par l'échéance la plus proche de
    OPTIONS_TARGET_TENOR_DAYS -- cohérent avec ce que 08 collecte déjà
    (ATM, >9 mois)."""

    def __init__(self, option_snapshots: pd.DataFrame):
        self._by_key: dict[tuple[str, str], tuple[np.ndarray, list[dict]]] = {}
        if option_snapshots.empty:
            return

        df = option_snapshots.copy()
        df["_moneyness_abs"] = df["moneyness_pct"].abs()
        df["_tenor_diff"] = (df["expiry"] - df["snapshot_date"]).dt.days.sub(config.OPTIONS_TARGET_TENOR_DAYS).abs()
        best_per_date = (
            df.sort_values(["symbol", "option_type", "snapshot_date", "_moneyness_abs", "_tenor_diff"])
            .groupby(["symbol", "option_type", "snapshot_date"], as_index=False, sort=False)
            .head(1)
        )

        for (symbol, option_type), group in best_per_date.groupby(["symbol", "option_type"], sort=False):
            group = group.sort_values("snapshot_date")
            contracts = [
                {
                    "strike": row["strike"], "expiry": row["expiry"], "premium": _premium_of(row),
                    "implied_vol": row.get("implied_vol"), "delta": row.get("delta"), "gamma": row.get("gamma"),
                    "vega": row.get("vega"), "theta": row.get("theta"), "underlying_spot": row.get("underlying_spot"),
                    "multiplier": float(row.get("multiplier") or config.OPTIONS_CONTRACT_MULTIPLIER),
                    "snapshot_date": row["snapshot_date"], "source": "real",
                }
                for _, row in group.iterrows()
            ]
            self._by_key[(symbol, option_type)] = (
                group["snapshot_date"].to_numpy(dtype="datetime64[ns]"), contracts,
            )

    def find(
        self,
        symbol: str,
        option_type: str,
        as_of: pd.Timestamp,
        tolerance_days: int = config.OPTIONS_REAL_SNAPSHOT_TOLERANCE_DAYS,
    ) -> Optional[dict]:
        """Contrat réel le plus représentatif autour de as_of, ou None si
        aucun snapshot n'existe dans la fenêtre de tolérance (l'appelant doit
        alors simuler par Black-Scholes)."""
        entry = self._by_key.get((symbol, option_type))
        if entry is None:
            return None
        dates, contracts = entry

        moment = np.datetime64(as_of.to_datetime64())
        insert_at = int(np.searchsorted(dates, moment))
        # La date la plus proche est forcément l'une des deux qui encadrent
        # as_of dans la liste triée.
        candidates = [i for i in (insert_at - 1, insert_at) if 0 <= i < len(dates)]
        if not candidates:
            return None
        best_index = min(candidates, key=lambda i: abs(dates[i] - moment))

        if abs(dates[best_index] - moment) > np.timedelta64(tolerance_days, "D"):
            return None
        return contracts[best_index]
