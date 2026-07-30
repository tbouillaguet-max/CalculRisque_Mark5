"""
Récupération des cours de bourse QUOTIDIENS (OHLCV), contrairement à
03_recuperation_cours.py qui ne garde que la clôture de fin d'année. Requis
pour backtester une stratégie au jour le jour (voir 09_backtest.py) : le
signal de valorisation (DCF, recalculé une fois par an) doit être confronté à
un cours qui bouge tous les jours pour simuler stop-loss/take-profit et
calculer un P&L quotidien.

Deux sources, dans cet ordre :
    1. IBKR (comme 03), pour tout ticker dont le contrat se résout encore
       (entreprises actuellement cotées).
    2. Repli Stooq (https://stooq.com, gratuit, sans clé API) pour les
       tickers qu'IBKR ne résout plus -- typiquement les entreprises sorties
       du S&P 500 (rachat, faillite, radiation) produites par
       01b_historique_univers_sp500.py (config.UNIVERSE_FULL_FILE). Stooq
       conserve l'historique de nombreux tickers US même après leur
       radiation, ce qu'IBKR ne fait pas (IBKR ne sert que des instruments
       actuellement négociables).

Un ticker introuvable sur LES DEUX sources est loggé comme trou de
couverture (jamais silencieusement ignoré) : le rapport de couverture
(report/) et le data_loader du backtest doivent pouvoir s'appuyer dessus.

Incrémental, comme 03 : le fichier existant (config.DAILY_PRICES_FILE) est
chargé en début de run, seuls les jours manquants sont (re)téléchargés.

Prérequis (IBKR seulement -- Stooq ne demande rien) :
    1. TWS ou IB Gateway lancé, API activée.
    2. pip install ib_insync pandas pyarrow requests

Usage :
    python 03b_recuperation_cours_quotidiens.py
    python 03b_recuperation_cours_quotidiens.py --tickers data/universe/sp500_universe_full.csv
    python 03b_recuperation_cours_quotidiens.py --skip-ibkr   # Stooq uniquement, pas besoin de Gateway
    python 03b_recuperation_cours_quotidiens.py --limit 10
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

import config

IB_HOST = "127.0.0.1"
IB_PORT_DEFAULT = 4002
IB_CLIENT_ID = 4  # différent de 03 (clientId=3) : les deux scripts peuvent tourner côte à côte

RETRY_PER_TICKER = 3
HIST_REQUEST_PAUSE_SEC = 11  # même pacing IBKR que 03 (~55 req/10min sur reqHistoricalData)
STOOQ_REQUEST_PAUSE_SEC = 1.0

STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"
STOOQ_HEADERS = {"User-Agent": "options-pipeline/1.0 (contact: voir SEC_CONTACT_EMAIL dans 04_recuperation_10k.py)"}

REQUIRED_COLUMNS = {"RIC", "Instrument_Name", "Country", "Currency", "Exchange"}
OUTPUT_COLUMNS = [
    "symbol", "ric", "company_name", "country", "currency", "exchange",
    "date", "open", "high", "low", "close", "volume", "source",
]

logger = logging.getLogger("daily_prices")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"daily_prices_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logger.info("Log file: %s", log_path)


def default_tickers_file() -> Path:
    """Univers PIT (actuels + radiés, sortie de 01b) si disponible, sinon
    retombe sur l'univers actuel (sortie de 01) -- ce script reste utilisable
    même si 01b n'a jamais été lancé."""
    if config.UNIVERSE_FULL_FILE.exists():
        return config.UNIVERSE_FULL_FILE
    return config.UNIVERSE_FILE


def load_universe(csv_path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {csv_path}: {missing}")

    df = df.dropna(subset=["RIC"]).drop_duplicates(subset=["RIC"]).reset_index(drop=True)
    df["ib_symbol"] = df["RIC"].apply(config.to_ib_symbol)
    df["ib_exchange"] = df["Exchange"].map(config.EXCHANGE_MAP).fillna(config.DEFAULT_EXCHANGE)

    if limit:
        df = df.head(limit)

    logger.info("Univers chargé : %d tickers.", len(df))
    return df


def to_stooq_symbol(ric: str) -> str:
    """Convention Stooq pour les actions US : minuscules, tiret au lieu du
    point pour les classes d'actions, suffixe ".us" (ex: "BRK.B" -> "brk-b.us").
    Contrairement à config.to_ib_symbol (IBKR utilise un espace), Stooq
    utilise un tiret -- deux conventions différentes, d'où une fonction
    séparée plutôt que de réutiliser to_ib_symbol."""
    return ric.strip().lower().replace(".", "-") + ".us"


def determine_fetch_from(existing: pd.DataFrame, symbol: str, start_date: date, today: date, refresh_days: int) -> Optional[date]:
    """Détermine depuis quelle date (re)récupérer ce symbole, ou None si le
    cache est déjà à jour (dernier point en cache à moins de refresh_days
    jours). Contrairement à 03 (années civiles définitives une fois
    passées), ici TOUT l'historique quotidien peut en théorie être ajusté
    (splits/dividendes révisés) -- mais on ne re-télécharge que le delta
    récent pour rester raisonnable en nombre de requêtes ; --force-refresh
    retélécharge tout au besoin."""
    cached = existing[existing["symbol"] == symbol] if not existing.empty else existing
    if cached.empty:
        return start_date
    last_date = pd.to_datetime(cached["date"]).max().date()
    age_days = (today - last_date).days
    if age_days <= refresh_days:
        return None
    return last_date  # re-fetch en incluant le dernier jour en cache (chevauchement voulu, dédupliqué ensuite)


def connect_ib(port: int):
    from ib_insync import IB
    ib = IB()
    ib.connect(IB_HOST, port, clientId=IB_CLIENT_ID, timeout=20)
    logger.info("Connecté à IBKR sur %s:%s (clientId=%s)", IB_HOST, port, IB_CLIENT_ID)
    return ib


def resolve_stock_contract(ib, symbol: str, exchange: str, currency: str):
    from ib_insync import Stock
    if exchange == "SMART":
        candidates = [Stock(symbol, "SMART", currency)]
    else:
        candidates = [Stock(symbol, exchange, currency), Stock(symbol, "SMART", currency, primaryExchange=exchange)]
    for contract in candidates:
        try:
            details = ib.reqContractDetails(contract)
        except Exception as exc:  # noqa: BLE001
            logger.debug("reqContractDetails a échoué pour %s: %s", contract, exc)
            details = []
        if details:
            return details[0].contract
    return None


def _duration_str(duration_days: int) -> str:
    """IBKR rejette (erreur 321) une durée en jours ("D") au-delà de 365
    jours pour reqHistoricalData : au-delà, il faut exprimer la durée en
    années ("Y"). Le premier backfill d'un ticker (plusieurs années d'un
    coup, fetch_from=start_date) dépasse systématiquement ce seuil ; les
    rafraîchissements incrémentaux (quelques jours) restent en "D"."""
    if duration_days <= 365:
        return f"{duration_days} D"
    years_needed = math.ceil(duration_days / 365)
    return f"{years_needed} Y"


def fetch_ibkr_daily(ib, contract, fetch_from: date, today: date) -> pd.DataFrame:
    from ib_insync import util
    duration_days = max((today - fetch_from).days + 3, 2)  # +3 : marge week-ends/jours fériés
    bars = ib.reqHistoricalData(
        contract, endDateTime="", durationStr=_duration_str(duration_days),
        barSizeSetting="1 day", whatToShow="TRADES", useRTH=True, formatDate=1,
    )
    time.sleep(HIST_REQUEST_PAUSE_SEC)
    if not bars:
        return pd.DataFrame()
    history = util.df(bars)
    history["date"] = pd.to_datetime(history["date"]).dt.date
    history = history[history["date"] >= fetch_from]
    return history.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})[
        ["date", "open", "high", "low", "close", "volume"]
    ].reset_index(drop=True)


def fetch_stooq_daily(ric: str, fetch_from: date, today: date) -> pd.DataFrame:
    symbol = to_stooq_symbol(ric)
    url = STOOQ_URL.format(symbol=symbol, d1=fetch_from.strftime("%Y%m%d"), d2=today.strftime("%Y%m%d"))
    try:
        resp = requests.get(url, headers=STOOQ_HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.debug("Requête Stooq échouée pour %s: %s", symbol, exc)
        return pd.DataFrame()
    finally:
        time.sleep(STOOQ_REQUEST_PAUSE_SEC)

    text = resp.text
    if not text or text.startswith("<") or "Exceeded" in text[:200]:
        # Stooq renvoie du HTML (ticker inconnu) ou un message de quota
        # dépassé au lieu d'un CSV : traité comme "pas de données", pas
        # comme une exception (repli normal, pas une erreur bloquante).
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:  # noqa: BLE001
        logger.debug("CSV Stooq illisible pour %s: %s", symbol, exc)
        return pd.DataFrame()

    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def process_ticker(ib, row: pd.Series, fetch_from: date, today: date, skip_ibkr: bool) -> tuple[list[dict], str]:
    """Retourne (lignes, source utilisée). Essaie IBKR d'abord (sauf
    --skip-ibkr), puis Stooq. Lève RuntimeError si aucune des deux sources
    ne renvoie de données (le ticker est alors loggé comme échec par
    l'appelant, pas silencieusement perdu)."""
    symbol = row["ib_symbol"]
    history, source = pd.DataFrame(), None

    if not skip_ibkr and ib is not None:
        contract = resolve_stock_contract(ib, symbol, row["ib_exchange"], row["Currency"])
        if contract is not None:
            for attempt in range(RETRY_PER_TICKER + 1):
                try:
                    history = fetch_ibkr_daily(ib, contract, fetch_from, today)
                    source = "ibkr"
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt < RETRY_PER_TICKER:
                        logger.debug("  tentative IBKR %d échouée pour %s (%s), nouvel essai...", attempt + 1, symbol, exc)
                        ib.sleep(2)

    if history.empty:
        history = fetch_stooq_daily(row["RIC"], fetch_from, today)
        source = "stooq" if not history.empty else source

    if history.empty:
        raise RuntimeError(f"Aucune donnée (ni IBKR ni Stooq) pour {symbol}")

    rows = [{
        "symbol": symbol, "ric": row["RIC"], "company_name": row["Instrument_Name"],
        "country": row["Country"], "currency": row["Currency"], "exchange": row["Exchange"],
        "date": r["date"], "open": r["open"], "high": r["high"], "low": r["low"],
        "close": r["close"], "volume": r["volume"], "source": source,
    } for _, r in history.iterrows()]
    return rows, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", type=Path, default=None, help="CSV d'univers (défaut: univers PIT de 01b si présent, sinon univers actuel de 01)")
    parser.add_argument("--output-dir", default=config.DIR_PRICES, type=Path)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--port", type=int, default=IB_PORT_DEFAULT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh-days", type=int, default=1, help="Rafraîchit un symbole si son dernier point en cache a plus de N jours (défaut: %(default)s).")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore le cache et retélécharge tout l'historique pour tous les tickers.")
    parser.add_argument("--skip-ibkr", action="store_true", help="Ne pas se connecter à IBKR, utiliser uniquement Stooq (pas besoin de TWS/Gateway).")
    args = parser.parse_args()

    today = datetime.now().date()
    start_date = date(args.start_year, 1, 1)
    setup_logging(args.output_dir)

    tickers_path = args.tickers or default_tickers_file()
    universe = load_universe(tickers_path, limit=args.limit)

    output_file = args.output_dir / config.DAILY_PRICES_FILE.name
    existing = pd.read_parquet(output_file) if output_file.exists() else pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not existing.empty:
        logger.info("Cache existant chargé : %s (%d lignes, %d symboles).", output_file, len(existing), existing["symbol"].nunique())

    plan = []
    for _, row in universe.iterrows():
        fetch_from = start_date if args.force_refresh else determine_fetch_from(existing, row["ib_symbol"], start_date, today, args.refresh_days)
        plan.append((row, fetch_from))

    skip_count = sum(1 for _, fetch_from in plan if fetch_from is None)
    to_process = [(row, fetch_from) for row, fetch_from in plan if fetch_from is not None]

    if not to_process:
        logger.info("Les %d tickers sont déjà à jour (cache < %d jour(s)) : rien à faire.", len(universe), args.refresh_days)
        return

    logger.info("%d/%d tickers déjà à jour (ignorés), %d à récupérer.", skip_count, len(universe), len(to_process))

    ib = None
    if not args.skip_ibkr:
        try:
            ib = connect_ib(args.port)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Connexion IBKR impossible (%s) : repli Stooq pour TOUS les tickers.", exc)
            ib = None

    all_rows: list[dict] = []
    ok_count, fail_count = 0, 0
    source_counts: dict[str, int] = {}

    try:
        for idx, (row, fetch_from) in enumerate(to_process):
            symbol = row["ib_symbol"]
            logger.info("[%d/%d] %s (%s), depuis %s...", idx + 1, len(to_process), symbol, row["Instrument_Name"], fetch_from)
            try:
                rows, source = process_ticker(ib, row, fetch_from, today, args.skip_ibkr or ib is None)
                all_rows.extend(rows)
                ok_count += 1
                source_counts[source] = source_counts.get(source, 0) + 1
                logger.info("  -> %d jours récupérés pour %s (source: %s)", len(rows), symbol, source)
            except Exception as exc:  # noqa: BLE001
                fail_count += 1
                logger.warning("  -> ECHEC (ni IBKR ni Stooq) pour %s : %s -- TROU DE COUVERTURE.", symbol, exc)
    finally:
        if ib is not None:
            ib.disconnect()
            logger.info("Déconnecté d'IBKR.")

    logger.info(
        "Terminé. OK: %d | Échecs (trous de couverture): %d | Déjà à jour (ignorés): %d | Nouvelles lignes: %d | Sources: %s",
        ok_count, fail_count, skip_count, len(all_rows), source_counts,
    )

    if not all_rows:
        if existing.empty:
            logger.warning("Aucune donnée récupérée, pas de fichier de sortie généré.")
        else:
            logger.info("Rien de nouveau à écrire : fichier existant conservé tel quel (%s).", output_file)
        return

    df_new = pd.DataFrame(all_rows)
    combined = pd.concat([existing, df_new], ignore_index=True) if not existing.empty else df_new
    combined = (
        combined.sort_values(["symbol", "date"])
        .drop_duplicates(subset=["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_file, index=False, engine="pyarrow")
    logger.info("Fichier écrit : %s (%d lignes au total, %d nouvelles/mises à jour ce run).", output_file, len(combined), len(df_new))


if __name__ == "__main__":
    main()
