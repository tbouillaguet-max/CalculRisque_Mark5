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
        self._realized_vol_cache: dict[int, np.ndarray] = {}

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

    def momentum_12_1(self, symbol: str, date: pd.Timestamp, lag_days: int = 21, lookback_days: int = 252) -> Optional[float]:
        """Momentum "12-1" à `date` : rendement du titre sur les 12 derniers
        mois EN EXCLUANT le dernier (convention académique -- le dernier mois
        porte un effet de retournement à court terme, de signe opposé).

        Point-in-time par construction : s'appuie sur close_history, qui
        s'arrête à `date` incluse. None si l'historique est trop court (titre
        récemment entré dans l'univers)."""
        hist = self.close_history(symbol, date)
        if len(hist) < lookback_days + 1:
            return None
        start, end = hist[-(lookback_days + 1)], hist[-(lag_days + 1)]
        if not np.isfinite(start) or not np.isfinite(end) or start <= 0:
            return None
        return float(end / start - 1)

    def realized_vol_panel(self, lookback_days: int) -> np.ndarray:
        """Volatilité réalisée glissante pour CHAQUE (date, symbole), calculée
        une seule fois puis mise en cache -- nécessaire dès qu'une position est
        reprise à la volatilité du jour plutôt qu'à celle figée à son entrée
        (cf. options_engine, vol_mode="rolling") : la recalculer position par
        position et jour par jour dominerait le temps de run.

        Reproduit EXACTEMENT options_pricing.realized_volatility, qui reste
        l'estimateur utilisé à l'entrée -- sans quoi entrée et repricing
        mesureraient deux choses différentes et la prime sauterait dès le
        premier jour :
            - fenêtre des `lookback_days` dernières clôtures NON MANQUANTES
              (donc lookback_days - 1 rendements log au plus) ;
            - écart-type ddof=1, annualisé par sqrt(252) ;
            - None (NaN ici) tant que l'historique est trop court.
        """
        cached = self._realized_vol_cache.get(lookback_days)
        if cached is not None:
            return cached

        min_history = max(lookback_days // 2, 10)  # garde-fou "historique trop court"
        panel = np.full(self._close_values.shape, np.nan)
        for col in range(self._close_values.shape[1]):
            values = self._close_values[:, col]
            valid_rows = np.flatnonzero(~np.isnan(values))
            if len(valid_rows) < min_history:
                continue
            series = pd.Series(values[valid_rows])
            log_returns = np.log(series).diff()
            # rolling sur lookback_days - 1 rendements : une fenêtre de N
            # clôtures ne porte que N-1 rendements. min_periods=5 reproduit le
            # garde-fou "moins de 6 points dans la fenêtre".
            vol = log_returns.rolling(lookback_days - 1, min_periods=5).std(ddof=1) * np.sqrt(252)
            vol.iloc[: min_history - 1] = np.nan  # historique encore trop court
            panel[valid_rows, col] = vol.to_numpy()

        self._realized_vol_cache[lookback_days] = panel
        return panel

    def realized_vol_at(self, symbol: str, date: pd.Timestamp, lookback_days: int) -> Optional[float]:
        """Volatilité réalisée du symbole à `date`, ou None si l'historique
        disponible à cette date est trop court (mêmes règles qu'à l'entrée)."""
        row = self._row_of_date.get(date)
        col = self._close_col.get(symbol)
        if row is None or col is None:
            return None
        value = self.realized_vol_panel(lookback_days)[row, col]
        return None if value != value else float(value)


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


def _period_key(df: pd.DataFrame) -> pd.Series:
    """Clé de période (symbol|period_type|fiscal_year|fiscal_quarter) sous
    forme de chaîne : un merge direct sur ces quatre colonnes échouerait sur
    fiscal_quarter, vide (None) pour les lignes annuelles et de dtype variable
    d'un fichier à l'autre."""
    def norm(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series([""] * len(df), index=df.index)
        return df[col].astype("object").where(df[col].notna(), "").astype(str)

    return norm("symbol") + "|" + norm("period_type") + "|" + norm("fiscal_year") + "|" + norm("fiscal_quarter")


def apply_qualitative_gate(df: pd.DataFrame, path=None) -> pd.DataFrame:
    """Écarte les périodes dont 07b_validation_qualitative.py a jugé le texte
    du 10-K/10-Q CONTRADICTOIRE avec l'écart de valorisation quantitatif
    (config.QUALITATIVE_GATE_EXCLUDED_VERDICTS) -- les value traps que le
    chiffre seul ne voit pas (dépréciation d'actif, guidance abaissée).

    Sans effet si 07b n'a jamais tourné, ou tant que MISTRAL_API_KEY n'est pas
    définie : tous les verdicts valent alors "non_evalue", et une période non
    évaluée est CONSERVÉE (on ne filtre que sur un jugement explicite)."""
    excluded = set(config.QUALITATIVE_GATE_EXCLUDED_VERDICTS or ())
    path = path or config.QUALITATIVE_VALIDATION_FILE
    if not excluded or df.empty or not path.exists():
        return df

    qual = pd.read_parquet(path)
    if "verdict" not in qual.columns:
        return df
    rejected = qual[qual["verdict"].isin(excluded)]
    if rejected.empty:
        return df

    # Jointure par clé de période plutôt que merge : aucune colonne ajoutée,
    # donc aucun risque de dupliquer une ligne de signal si 07b contient
    # plusieurs verdicts pour la même période.
    mask = ~_period_key(df).isin(set(_period_key(rejected)))
    if not mask.all():
        logger.info(
            "Filtre qualitatif (07b) : %d/%d signaux écartés (verdict %s).",
            (~mask).sum(), len(df), "/".join(sorted(excluded)),
        )
    return df[mask].reset_index(drop=True)


def load_dcf_history(path=None) -> pd.DataFrame:
    path = path or config.DCF_HISTORY_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Lance d'abord 07_calcul_dcf.py (avec 04 déjà à jour)."
        )
    df = _fill_missing_filed_dates(pd.read_parquet(path), path)
    df = apply_qualitative_gate(df)
    return df.dropna(subset=["gap_pct", "symbol"]).sort_values(["symbol", "filed_date"]).reset_index(drop=True)


def build_signal_events(dcf_history: pd.DataFrame) -> pd.DataFrame:
    """Un événement par (symbol, filed_date) -- la date à laquelle ce
    signal devient publiquement connu. Colonnes : symbol, published_date,
    fiscal_year, sector, close_at_filing, valuation_dcf_per_share, gap_pct."""
    # Ne renomme PAS "year" -> "fiscal_year" : dcf_history a les deux colonnes
    # depuis l'empilement FY+TTM (07_calcul_dcf.py), toujours égales par
    # construction (05_calcul_multiples.load_financials_with_periods pose déjà
    # year=fiscal_year pour les lignes TTM). Les renommer produirait deux
    # colonnes "fiscal_year" -- pandas droppe alors l'une des deux SANS
    # erreur au premier .to_dict("records") (engine.py), juste un UserWarning
    # "columns are not unique" facile à manquer dans les logs.
    renamed = dcf_history.rename(columns={"filed_date": "published_date", "close": "close_at_filing"})
    cols = ["symbol", "published_date", "fiscal_year", "sector", "close_at_filing", "valuation_dcf_per_share", "gap_pct"]
    # period_type : porté jusqu'à l'engine pour dater la péremption du signal
    # (un TTM trimestriel se périme plus vite qu'un exercice annuel, cf.
    # signal_max_age_for). Absent des caches antérieurs au TTM (04b).
    if "period_type" in renamed.columns:
        cols.append("period_type")
    return renamed[cols]


class MaterialEventResolver:
    """Dates de dépôt des 8-K jugés MATÉRIELS par 04c_recuperation_8k.py,
    indexées par symbole pour une interrogation en O(log n) par jour simulé.

    Sert à périmer un signal : entre le dépôt du 10-K/10-Q qui l'a produit et
    aujourd'hui, un événement matériel (dépréciation, litige, changement de
    direction, guidance revue) a pu invalider les fondamentaux sur lesquels il
    repose. On ne fait AUCUNE hypothèse sur le SENS de l'événement -- 04c ne
    la capture pas : un événement matériel rend le signal périmé, il ne le
    retourne pas."""

    def __init__(self, events: Optional[pd.DataFrame]):
        self._dates_by_symbol: dict[str, np.ndarray] = {}
        if events is None or events.empty:
            return
        for symbol, group in events.groupby("symbol"):
            dates = np.sort(group["filed_date"].to_numpy(dtype="datetime64[ns]"))
            if len(dates):
                self._dates_by_symbol[symbol] = dates

    def has_event_between(self, symbol: str, after: pd.Timestamp, upto: pd.Timestamp) -> bool:
        """Un événement matériel a-t-il été déposé dans ]after, upto] ?"""
        dates = self._dates_by_symbol.get(symbol)
        if dates is None:
            return False
        lo = np.searchsorted(dates, np.datetime64(pd.Timestamp(after).to_datetime64()), side="right")
        hi = np.searchsorted(dates, np.datetime64(pd.Timestamp(upto).to_datetime64()), side="right")
        return bool(hi > lo)  # bool natif, pas un numpy.bool_


def load_material_events_8k(path=None) -> Optional[pd.DataFrame]:
    """8-K jugés matériels (04c). None si 04c n'a jamais tourné, ou si aucune
    ligne n'est marquée matérielle -- notamment quand MISTRAL_API_KEY n'est pas
    définie : 04c écrit alors materiality=None partout (aucune classification),
    et le filtre reste sans effet plutôt que d'écarter tous les signaux."""
    path = path or config.MATERIAL_EVENTS_8K_FILE
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "materiality" not in df.columns or "filed_date" not in df.columns:
        return None
    df = df[df["materiality"].fillna(False).astype(bool)].copy()
    if df.empty:
        logger.info(
            "%s ne contient aucun 8-K classé matériel (MISTRAL_API_KEY non définie lors du "
            "run de 04c ?) : le filtre d'événements matériels reste sans effet.", path,
        )
        return None
    df["filed_date"] = config.to_naive_day(df["filed_date"])
    return df.dropna(subset=["symbol", "filed_date"])


def signal_max_age_for(signal: dict, default: int) -> int:
    """Âge maximal (en jours) au-delà duquel ce signal n'est plus actionnable,
    selon le type de période qui l'a produit (config.BACKTEST_SIGNAL_MAX_AGE_DAYS_BY_PERIOD).
    Repli sur `default` si le period_type est inconnu ou absent du cache."""
    by_period = getattr(config, "BACKTEST_SIGNAL_MAX_AGE_DAYS_BY_PERIOD", None) or {}
    return by_period.get(signal.get("period_type"), default)


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
    df = apply_qualitative_gate(df)
    return df.dropna(subset=["gap_pct", "symbol"]).sort_values(["symbol", "filed_date"]).reset_index(drop=True)


def build_options_signal_events(valorisation_combinee: pd.DataFrame) -> pd.DataFrame:
    """Un événement par (symbol, filed_date). gap_pct garde son signe (positif
    = sous-évalué -> call, négatif = survalué -> put) : c'est la stratégie
    (backtest/strategies/valuation_gap_options.py) qui décide du sens."""
    # "year" -> "fiscal_year" SEULEMENT si fiscal_year manque, comme le fait
    # déjà build_signal_events pour la même raison. 06b propage les colonnes
    # de multiples.parquet, où 05_calcul_multiples.load_financials_with_periods
    # pose DÉJÀ fiscal_year (=year pour les lignes annuelles, et year=
    # fiscal_year pour les lignes TTM) : renommer inconditionnellement
    # produisait DEUX colonnes "fiscal_year", que pandas droppe ensuite sans
    # erreur au premier .to_dict("records") (options_engine.py) -- juste un
    # UserWarning "columns are not unique" noyé dans les logs de chaque run
    # de 10. Les deux colonnes sont égales par construction aujourd'hui, donc
    # rien de faux n'en sortait ; le jour où elles divergeraient, la perte
    # serait silencieuse. Le renommage reste actif pour les caches produits
    # par une version de 05 antérieure au TTM (04b), qui n'ont que "year".
    to_rename = {"filed_date": "published_date"}
    if "fiscal_year" not in valorisation_combinee.columns:
        to_rename["year"] = "fiscal_year"
    renamed = valorisation_combinee.rename(columns=to_rename)
    cols = [
        "symbol", "published_date", "fiscal_year", "sector", "close",
        "valuation_theoretical_per_share", "source", "gap_pct",
    ]
    if "period_type" in renamed.columns:
        cols.append("period_type")
    return renamed[cols]


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


def build_benchmark_series(
    price_panel: PricePanel, universe: UniverseResolver, symbol: Optional[str] = None,
) -> tuple[Optional[pd.Series], str]:
    """Indice de référence auquel comparer la stratégie (cf. metrics.py), et
    son libellé.

    1. `symbol` (config.BENCHMARK_SYMBOL, SPY par défaut) s'il est présent
       dans les cours quotidiens : c'est le meilleur repère, celui que
       l'utilisateur aurait effectivement pu acheter.
    2. Sinon, indice ÉQUIPONDÉRÉ de l'univers point-in-time : moyenne des
       rendements quotidiens des seules entreprises membres de l'indice ce
       jour-là, capitalisée. Équipondéré et non pondéré par la
       capitalisation, faute de flottant historique -- c'est donc un repère
       proche d'un S&P 500 Equal Weight, pas du S&P 500 lui-même : il
       surpondère structurellement les petites capitalisations de l'indice.
       Le libellé retourné le dit, pour qu'aucun alpha ne soit lu contre le
       mauvais repère.
    3. (None, libellé) si même ça n'est pas calculable.

    Point-in-time par construction dans les deux cas : le repli n'utilise à
    chaque date que la composition de l'indice À cette date (UniverseResolver),
    jamais la composition d'aujourd'hui appliquée au passé."""
    symbol = symbol or config.BENCHMARK_SYMBOL
    close = price_panel.close

    if symbol in close.columns:
        series = close[symbol].dropna()
        if len(series) > 1:
            return series, symbol
        logger.warning(
            "%s est présent dans les cours quotidiens mais sans historique exploitable : "
            "repli sur un indice équipondéré de l'univers.", symbol,
        )
    else:
        logger.info(
            "%s absent de %s : l'indice de référence est reconstruit en équipondéré à partir "
            "de l'univers point-in-time. Ajoute %s à 03b_recuperation_cours_quotidiens.py pour "
            "une comparaison au S&P 500 lui-même.", symbol, config.DAILY_PRICES_FILE, symbol,
        )

    returns = close.pct_change()
    columns = np.asarray(close.columns)
    daily_mean = np.full(len(close.index), np.nan)
    returns_values = returns.to_numpy(dtype=float)
    for row, date in enumerate(close.index):
        if row == 0:
            continue
        members = universe.asof(date)
        if not members:
            continue
        selected = returns_values[row, np.isin(columns, list(members))]
        selected = selected[~np.isnan(selected)]
        if len(selected):
            daily_mean[row] = selected.mean()

    valid = ~np.isnan(daily_mean)
    if valid.sum() < 2:
        logger.warning(
            "Indice de référence équipondéré non calculable (aucun rendement exploitable) : "
            "le run n'aura pas de point de comparaison."
        )
        return None, "aucun"

    daily_mean[~valid] = 0.0
    index = pd.Series(np.cumprod(1.0 + daily_mean), index=close.index)
    return index, "univers S&P 500 équipondéré (point-in-time)"


def load_option_snapshots_history(directory=None) -> pd.DataFrame:
    """Concatène TOUS les snapshots archivés par 08_recuperation_options.py
    (data/options/history/option_chains_*.parquet, un fichier par run,
    jamais écrasé). DataFrame vide si 08 n'a encore jamais tourné -- pas une
    erreur : backtest/options_engine.py retombe alors entièrement sur le
    pricing Black-Scholes simulé (voir OptionSnapshotIndex)."""
    directory = directory or config.DIR_OPTIONS_HISTORY
    files = sorted(directory.glob("option_chains_*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["expiry"] = pd.to_datetime(df["expiry"], format="%Y%m%d", errors="coerce")
    return df


def _premium_series(df: pd.DataFrame) -> pd.Series:
    """Prime retenue pour un contrat : milieu bid/ask quand les deux côtés
    sont cotés, sinon dernier prix traité, sinon clôture."""
    def col(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=float)

    bid, ask = col("bid"), col("ask")
    mid = ((bid + ask) / 2).where(bid.notna() & ask.notna())
    return mid.fillna(col("last_price")).fillna(col("close"))


def _optional_float(value) -> Optional[float]:
    """NaN -> None : les appelants testent la présence d'un champ par sa
    valeur de vérité (`if real["implied_vol"]`), et un NaN est "vrai" en
    Python -- il se propagerait silencieusement dans le pricing au lieu de
    déclencher le repli Black-Scholes."""
    return None if value is None or pd.isna(value) else float(value)


class OptionSnapshotIndex:
    """Index des snapshots réels archivés par 08_recuperation_options.py,
    interrogeable par (symbole, type d'option, date).

    Deux modes de sélection du contrat, selon ce que demande l'appelant :

    - SANS strike/échéance cibles (stratégie valuation_gap_options) : le
      contrat le plus proche de la monnaie (min |moneyness_pct|), départagé
      par l'échéance la plus proche de OPTIONS_TARGET_TENOR_DAYS -- cohérent
      avec ce que 08 collecte déjà (ATM, >9 mois). Ce contrat ne dépend pas de
      la date d'interrogation : son indice est donc calculé UNE FOIS ici, à la
      construction, et le `find` correspondant reste en O(1).

    - AVEC strike et/ou échéance cibles (stratégie
      valuation_gap_multiples_options, qui vise un strike à mi-chemin et une
      échéance 2 ans) : le contrat le plus proche du strike demandé, départagé
      par l'échéance la plus proche de celle demandée. La sélection dépend
      alors de la demande, donc se fait à l'interrogation -- sur le seul
      sous-ensemble des contrats de la date retenue, pas sur tout l'historique.

    Les colonnes sont stockées à plat en numpy (une seule fois par
    symbole/type) plutôt qu'en dictionnaires par contrat : seul le contrat
    finalement retenu est matérialisé en dict."""

    _ARRAY_COLUMNS = (
        "strike", "premium", "implied_vol", "delta", "gamma", "vega", "theta",
        "underlying_spot", "multiplier",
    )

    def __init__(self, option_snapshots: pd.DataFrame):
        self._by_key: dict[tuple[str, str], dict] = {}
        if option_snapshots.empty:
            return

        df = option_snapshots.copy()
        df["premium"] = _premium_series(df)
        df["_moneyness_abs"] = df["moneyness_pct"].abs()
        df["_tenor_days"] = (df["expiry"] - df["snapshot_date"]).dt.days
        df["_tenor_diff"] = df["_tenor_days"].sub(config.OPTIONS_TARGET_TENOR_DAYS).abs()

        if "multiplier" in df.columns:
            df["multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce")
        else:
            df["multiplier"] = np.nan
        # Un multiplicateur manquant vaut la convention US (1 contrat = 100
        # actions) plutôt que NaN, qui contaminerait le dimensionnement.
        df["multiplier"] = df["multiplier"].replace(0, np.nan).fillna(float(config.OPTIONS_CONTRACT_MULTIPLIER))

        for optional in ("implied_vol", "delta", "gamma", "vega", "theta", "underlying_spot"):
            if optional not in df.columns:
                df[optional] = np.nan

        # Tri unique : (symbole, type, date) puis monnaie, puis échéance. La
        # PREMIÈRE ligne de chaque date est donc le contrat par défaut --
        # exactement ce que renvoyait l'ancien groupby(...).head(1).
        df = df.sort_values(["symbol", "option_type", "snapshot_date", "_moneyness_abs", "_tenor_diff"])

        for (symbol, option_type), group in df.groupby(["symbol", "option_type"], sort=False):
            dates = group["snapshot_date"].to_numpy(dtype="datetime64[ns]")
            unique_dates, starts = np.unique(dates, return_index=True)
            cols = {name: group[name].to_numpy(dtype=float) for name in self._ARRAY_COLUMNS}
            cols["expiry"] = group["expiry"].to_numpy(dtype="datetime64[ns]")
            cols["snapshot_date"] = dates
            cols["tenor_days"] = group["_tenor_days"].to_numpy(dtype=float)
            cols["moneyness_abs"] = group["_moneyness_abs"].to_numpy(dtype=float)
            self._by_key[(symbol, option_type)] = {
                "dates": unique_dates,
                "starts": starts,
                "ends": np.append(starts[1:], len(dates)),
                "cols": cols,
            }

    def _contract_at(self, entry: dict, i: int) -> dict:
        cols = entry["cols"]
        return {
            "strike": float(cols["strike"][i]),
            "expiry": pd.Timestamp(cols["expiry"][i]),
            "premium": _optional_float(cols["premium"][i]),
            "implied_vol": _optional_float(cols["implied_vol"][i]),
            "delta": _optional_float(cols["delta"][i]),
            "gamma": _optional_float(cols["gamma"][i]),
            "vega": _optional_float(cols["vega"][i]),
            "theta": _optional_float(cols["theta"][i]),
            "underlying_spot": _optional_float(cols["underlying_spot"][i]),
            "multiplier": float(cols["multiplier"][i]),
            "snapshot_date": pd.Timestamp(cols["snapshot_date"][i]),
            "source": "real",
        }

    def find(
        self,
        symbol: str,
        option_type: str,
        as_of: pd.Timestamp,
        tolerance_days: int = config.OPTIONS_REAL_SNAPSHOT_TOLERANCE_DAYS,
        target_strike: Optional[float] = None,
        target_tenor_days: Optional[float] = None,
    ) -> Optional[dict]:
        """Contrat réel le plus représentatif autour de as_of, ou None si
        aucun snapshot n'existe dans la fenêtre de tolérance (l'appelant doit
        alors simuler par Black-Scholes).

        target_strike / target_tenor_days : contrat visé quand la stratégie
        impose autre chose que l'ATM à ~9 mois (voir docstring de la classe).
        Aucun des deux n'est contraignant -- le contrat réellement coté le
        plus proche est retenu, même s'il est loin du strike demandé : c'est à
        l'appelant de juger si l'écart reste acceptable."""
        entry = self._by_key.get((symbol, option_type))
        if entry is None:
            return None
        dates = entry["dates"]

        moment = np.datetime64(as_of.to_datetime64())
        insert_at = int(np.searchsorted(dates, moment))
        # La date la plus proche est forcément l'une des deux qui encadrent
        # as_of dans la liste triée.
        candidates = [i for i in (insert_at - 1, insert_at) if 0 <= i < len(dates)]
        if not candidates:
            return None
        best_date = min(candidates, key=lambda i: abs(dates[i] - moment))

        if abs(dates[best_date] - moment) > np.timedelta64(tolerance_days, "D"):
            return None

        if target_strike is None and target_tenor_days is None:
            return self._contract_at(entry, int(entry["starts"][best_date]))

        lo, hi = int(entry["starts"][best_date]), int(entry["ends"][best_date])
        # Sans strike demandé, le critère prioritaire reste la monnaie -- pour
        # qu'une cible portant seulement sur l'échéance ne retienne pas un
        # strike arbitraire.
        strike_dist = (
            np.abs(entry["cols"]["strike"][lo:hi] - target_strike)
            if target_strike is not None else entry["cols"]["moneyness_abs"][lo:hi]
        )
        # np.zeros(hi - lo) et non np.zeros(width) : `width` n'a jamais existé
        # dans cette portée -- le NameError se déclenchait dès qu'un appelant
        # demandait un strike SANS échéance cible et qu'un snapshot réel
        # existait pour ce (symbole, type).
        tenor_dist = (
            np.abs(entry["cols"]["tenor_days"][lo:hi] - target_tenor_days)
            if target_tenor_days is not None else np.zeros(hi - lo)
        )
        # Strike prioritaire, échéance en départage : même hiérarchie que la
        # sélection par défaut (monnaie d'abord, échéance ensuite).
        best = int(np.lexsort((tenor_dist, strike_dist))[0])
        return self._contract_at(entry, lo + best)
