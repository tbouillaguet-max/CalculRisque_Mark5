"""
Récupération des cours de bourse au 31 décembre (ou dernier jour de cotation
précédent) depuis START_YEAR jusqu'à aujourd'hui, pour l'univers américain
(config.UNIVERSE_FILE), via l'API IBKR (ib_insync).

Corrections / changements par rapport à RecuperationCourDeBourseMark1 :
    - Exchange map réduite aux places US (config.EXCHANGE_MAP) ; les valeurs
      non mappées ne sont plus ignorées mais routées en SMART par défaut.
    - Lit toujours config.UNIVERSE_FILE (--tickers reste possible pour
      surcharger avec un autre fichier au même format).
    - Bug corrigé : le fichier de sortie était écrit en dur dans /cours/...
      (ignorant --output-dir) ET nommé "options_*.parquet" alors qu'il s'agit
      de cours. Écrit maintenant dans config.PRICES_FILE.

Prérequis :
    1. TWS ou IB Gateway lancé, API activée (Configuration > API > Settings).
       Ports par défaut : TWS Paper=7497, TWS Live=7496, Gateway Paper=4002, Gateway Live=4001
    2. pip install ib_insync pandas pyarrow

Usage :
    python 03_recuperation_cours.py
    python 03_recuperation_cours.py --start-year 2015 --port 4002
    python 03_recuperation_cours.py --limit 10          # test rapide
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd

from ib_insync import IB, Stock, Contract, util

import config

IB_HOST = "127.0.0.1"
IB_PORT_DEFAULT = 4002
IB_CLIENT_ID = 3

RETRY_PER_TICKER = 3
HIST_REQUEST_PAUSE_SEC = 11  # pacing IBKR pour reqHistoricalData (~55 req/10min)

REQUIRED_COLUMNS = {"RIC", "Instrument_Name", "Country", "Currency", "Exchange"}

logger = logging.getLogger("year_end_prices")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"year_end_prices_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logger.info("Log file: %s", log_path)


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


def connect_ib(port: int) -> IB:
    ib = IB()
    ib.connect(IB_HOST, port, clientId=IB_CLIENT_ID, timeout=20)
    logger.info("Connecté à IBKR sur %s:%s (clientId=%s)", IB_HOST, port, IB_CLIENT_ID)
    return ib


def resolve_stock_contract(ib: IB, symbol: str, exchange: str, currency: str) -> Optional[Contract]:
    """Essaie l'exchange direct, puis SMART avec primaryExchange en fallback."""
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


def fetch_daily_history(ib: IB, contract: Contract, start_year: int, end_year: int) -> pd.DataFrame:
    """Un SEUL appel reqHistoricalData couvrant toute la période
    [start_year, end_year] (endDateTime="" = jusqu'à maintenant côté IBKR) :
    extract_year_end_close() se charge ensuite de ne garder que le dernier
    jour de cotation de chaque année civile. Corrigé : la version précédente
    recréait bien end_dt par année dans la boucle mais l'écrasait aussitôt
    avec la date du jour, et interrogeait "5 D" (5 derniers jours depuis
    maintenant) à CHAQUE itération -- donc jamais l'historique par année, et
    la fonction ne retournait rien (pas de return). Un seul appel par ticker
    est aussi cohérent avec HIST_REQUEST_PAUSE_SEC (pacing ~55 req/10min sur
    reqHistoricalData) : une requête par année et par ticker dépasserait
    largement ce budget sur l'ensemble de l'univers."""
    years_needed = end_year - start_year + 1
    bars = ib.reqHistoricalData(
        contract, endDateTime="", durationStr=f"{years_needed} Y",
        barSizeSetting="1 day", whatToShow="TRADES", useRTH=True, formatDate=1,
    )
    time.sleep(HIST_REQUEST_PAUSE_SEC)
    if not bars:
        return pd.DataFrame()

    history = util.df(bars)
    history["date"] = pd.to_datetime(history["date"])
    return history[history["date"].dt.year >= start_year].reset_index(drop=True)


def extract_year_end_close(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])
    results = []
    for year, group in history.groupby(history["date"].dt.year):
        cutoff = pd.Timestamp(date(int(year), 12, 31))
        eligible = group[group["date"] <= cutoff]
        if eligible.empty:
            continue
        last_row = eligible.sort_values("date").iloc[-1]
        results.append({"year": int(year), "date": last_row["date"].date(), "close": last_row["close"]})
    return pd.DataFrame(results)


def process_ticker(ib: IB, row: pd.Series, start_year: int, end_year: int) -> list[dict]:
    symbol = row["ib_symbol"]
    exchange = row["ib_exchange"]
    currency = row["Currency"]

    contract = resolve_stock_contract(ib, symbol, exchange, currency)
    if contract is None:
        raise RuntimeError(f"Impossible de résoudre le contrat pour {symbol} ({exchange})")

    history = fetch_daily_history(ib, contract, start_year, end_year)
    year_end = extract_year_end_close(history)
    if year_end.empty:
        raise RuntimeError(f"Aucune donnée historique récupérée pour {symbol}")

    rows = []
    for _, r in year_end.iterrows():
        rows.append({
            "symbol": symbol,
            "ric": row["RIC"],
            "company_name": row["Instrument_Name"],
            "country": row["Country"],
            "currency": currency,
            "exchange": exchange,
            "year": r["year"],
            "date": r["date"],
            "close": r["close"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", type=Path, default=config.UNIVERSE_FILE, help="CSV d'univers (par défaut: univers US construit par 01)")
    parser.add_argument("--output-dir", default=config.DIR_PRICES, type=Path)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--port", type=int, default=IB_PORT_DEFAULT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    end_year = datetime.now().year
    setup_logging(args.output_dir)
    universe = load_universe(args.tickers, limit=args.limit)

    ib = connect_ib(args.port)
    all_rows: list[dict] = []
    ok_count, fail_count = 0, 0

    try:
        for idx, row in universe.iterrows():
            symbol = row["ib_symbol"]
            logger.info("[%d/%d] %s (%s)...", idx + 1, len(universe), symbol, row["Instrument_Name"])
            last_exc, rows = None, None
            for attempt in range(RETRY_PER_TICKER + 1):
                try:
                    rows = process_ticker(ib, row, args.start_year, end_year)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt < RETRY_PER_TICKER:
                        logger.debug("  -> tentative %d échouée pour %s (%s), nouvel essai...", attempt + 1, symbol, exc)
                        ib.sleep(2)
            if rows is not None:
                all_rows.extend(rows)
                ok_count += 1
                logger.info("  -> %d années récupérées pour %s", len(rows), symbol)
            else:
                fail_count += 1
                logger.warning("  -> ECHEC pour %s : %s (ticker ignoré, on continue)", symbol, last_exc)
    finally:
        ib.disconnect()
        logger.info("Déconnecté d'IBKR.")

    logger.info("Terminé. Tickers OK: %d | Tickers en échec: %d | Total lignes: %d", ok_count, fail_count, len(all_rows))

    if not all_rows:
        logger.warning("Aucune donnée récupérée, pas de fichier de sortie généré.")
        return

    df = pd.DataFrame(all_rows).sort_values(["symbol", "year"]).reset_index(drop=True)
    # --output-dir pilote désormais réellement l'emplacement du fichier de
    # sortie (avant : seuls les logs en tenaient compte, le fichier de
    # données allait toujours dans config.PRICES_FILE quel que soit l'argument).
    output_file = args.output_dir / config.PRICES_FILE.name
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False, engine="pyarrow")
    logger.info("Fichier écrit : %s (%d lignes)", output_file, len(df))


if __name__ == "__main__":
    main()
