"""
Récupération des cours de bourse au 31 décembre (ou dernier jour de cotation
précédent) depuis START_YEAR jusqu'à aujourd'hui, pour l'univers américain
(config.UNIVERSE_FILE), via l'API IBKR (ib_insync).

Incrémental : le fichier existant (config.PRICES_FILE) est chargé en début
de run et fusionné avec les nouvelles données (jamais écrasé en totalité).
Une année < année en cours est une clôture définitive : une fois en cache,
elle n'est plus jamais re-téléchargée. Seule l'année en cours peut avoir
besoin d'un rafraîchissement (nouveaux jours de bourse depuis le dernier
run), et seulement si le dernier point en cache date de plus de
--refresh-days jours (voir determine_fetch_start_year). Un ticker déjà
entièrement à jour ne déclenche AUCUN appel IBKR ; si tout l'univers est à
jour, le script ne se connecte même pas à IBKR. --force-refresh retrouve
l'ancien comportement (retélécharge tout, pour tous les tickers).

Deux corrections récentes, sans lesquelles cet incrémental ne tenait pas ses
promesses :

    - UNE ANNÉE ABSENTE N'EST PAS UNE ANNÉE À RETÉLÉCHARGER. Toute année de
      [start_year, année en cours] absente du cache était considérée comme
      manquante, et le script repartait de la plus ancienne. Une entreprise
      entrée en bourse en 2014 n'ayant jamais de cours 2010-2013, ces années
      restaient éternellement "manquantes" : chaque run retéléchargeait
      l'historique complet de tous les tickers concernés, soit plus d'une
      heure de requêtes pour ne rien apprendre. Un état de suivi
      (data/prices/fetch_state_prices.json) mémorise l'année la plus ancienne
      déjà RÉCLAMÉE par symbole : une année réclamée et toujours absente est
      un fait établi, pas un manque à combler.

    - LE FICHIER EST ÉCRIT EN COURS DE ROUTE. Le parquet n'était écrit qu'une
      fois la boucle terminée : une interruption (Ctrl+C, coupure de session
      IBKR, plantage) faisait perdre l'intégralité du run. Il est désormais
      réécrit tous les CHECKPOINT_EVERY_ROWS points, et systématiquement en
      fin de run même sur erreur.

Corrections / changements par rapport à RecuperationCourDeBourseMark1 :
    - Exchange map réduite aux places US (config.EXCHANGE_MAP) ; les valeurs
      non mappées ne sont plus ignorées mais routées en SMART par défaut.
    - Lit toujours config.UNIVERSE_FILE (--tickers reste possible pour
      surcharger avec un autre fichier au même format).
    - Bug corrigé : le fichier de sortie était écrit en dur dans /cours/...
      (ignorant --output-dir) ET nommé "options_*.parquet" alors qu'il s'agit
      de cours. Écrit maintenant dans config.PRICES_FILE.
    - NOUVEAU : incrémental (voir plus haut) au lieu de retélécharger tout
      l'historique de tous les tickers à chaque run.

Prérequis :
    1. TWS ou IB Gateway lancé, API activée (Configuration > API > Settings).
       Ports par défaut : TWS Paper=7497, TWS Live=7496, Gateway Paper=4002, Gateway Live=4001
    2. pip install ib_insync pandas pyarrow

Usage :
    python 03_recuperation_cours.py
    python 03_recuperation_cours.py --start-year 2015 --port 4002
    python 03_recuperation_cours.py --limit 10          # test rapide
    python 03_recuperation_cours.py --refresh-days 0    # rafraîchit l'année en cours à chaque run
    python 03_recuperation_cours.py --force-refresh      # retélécharge tout, pour tous les tickers
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd

from ib_insync import IB, Stock, Contract, util

import config
import ib_connect

IB_HOST = "127.0.0.1"
IB_PORT_DEFAULT = 4002
IB_CLIENT_ID = 3

RETRY_PER_TICKER = 3
# Lignes accumulées avant réécriture du parquet. À ~11s de pacing IBKR par
# ticker, un univers complet dépasse l'heure : sans sauvegarde intermédiaire,
# une interruption faisait perdre tout le run.
CHECKPOINT_EVERY_ROWS = 200
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
    """Connexion via ib_connect, qui se rabat sur une connexion API seule si la
    synchronisation de compte d'ib_insync bloque -- ce script ne lit que des
    barres historiques (voir la docstring de ib_connect.py)."""
    return ib_connect.connect(port=port, client_id=IB_CLIENT_ID, host=IB_HOST)


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
    # whatToShow="TRADES" : cours de TRANSACTION, ajustés des splits mais PAS
    # des dividendes -- le P&L actions calculé dessus ignore donc les
    # dividendes réinvestis, soit environ 2%/an de rendement manquant sur le
    # S&P 500 (davantage sur les secteurs à haut rendement, précisément ceux
    # qu'un signal "value" sélectionne). Biais connu et documenté dans la
    # section "Biais et limites connus" du README, non corrigé ici : IBKR
    # n'expose pas de série totale-return sur cet endpoint, et changer de
    # source de cours dépasse le cadre de ce correctif.
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


def determine_fetch_start_year(
    existing: pd.DataFrame, symbol: str, start_year: int, end_year: int, refresh_days: int,
    requested_from: Optional[int] = None,
) -> Optional[int]:
    """Détermine à partir de quelle année (re)récupérer les cours de ce
    symbole, ou None si le cache couvre déjà tout ce dont on a besoin.

    Les années < end_year sont des clôtures définitives : une fois dans
    `existing`, elles ne sont jamais recomptées dans le besoin. Seule
    end_year (année en cours, pas encore clôturée) peut nécessiter un
    rafraîchissement, et seulement si son dernier point en cache a plus de
    refresh_days jours.

    CORRIGÉ -- UNE ANNÉE ABSENTE N'EST PAS UNE ANNÉE À RETÉLÉCHARGER.
    L'ancienne version traitait toute année de [start_year, end_year] absente
    du cache comme "manquante", et repartait de la PLUS ANCIENNE d'entre
    elles. Or une entreprise entrée en bourse en 2014 n'aura JAMAIS de cours
    2010-2013 : ces années restaient éternellement "manquantes", et chaque run
    retéléchargeait les 17 années complètes de tous les tickers concernés.
    D'où le "0/503 tickers déjà à jour" malgré un cache de 7 882 lignes -- et
    une heure et demie de requêtes IBKR gaspillées à chaque lancement.

    `requested_from` (état de suivi, voir load_fetch_state) est l'année la plus
    ancienne DÉJÀ DEMANDÉE pour ce symbole. Une année >= requested_from et
    toujours absente du cache a donc été réclamée à IBKR qui n'avait rien :
    c'est un fait établi, pas un manque à combler. Seules restent à récupérer
    les années jamais demandées (l'utilisateur a abaissé --start-year) et
    celles postérieures au dernier millésime en cache."""
    cached = existing[existing["symbol"] == symbol] if not existing.empty else existing
    cached_years = {int(y) for y in cached["year"].dropna()} if not cached.empty else set()

    if not cached_years:
        return start_year

    # Années jamais réclamées à IBKR : --start-year a été abaissé depuis le
    # dernier run, il y a peut-être là de l'historique à récupérer.
    if requested_from is not None and start_year < requested_from:
        return start_year
    if requested_from is None and min(cached_years) > start_year:
        # Pas d'état de suivi (cache antérieur à son introduction) : on ne peut
        # pas savoir si l'historique manquant a déjà été réclamé. Prudence --
        # on ne redemande PAS, au risque de manquer quelques années anciennes,
        # plutôt que de retélécharger tout l'univers à chaque run. Un
        # --force-refresh ponctuel les récupérera.
        logger.debug(
            "%s : cache démarrant en %d alors que --start-year vaut %d, sans état de suivi. "
            "Années antérieures supposées déjà réclamées.", symbol, min(cached_years), start_year,
        )

    # Millésimes postérieurs au dernier connu : les seuls réellement nouveaux.
    derniere_connue = max(cached_years)
    if derniere_connue < end_year:
        return derniere_connue + 1

    # Tout est là jusqu'à l'année en cours : reste à savoir si sa clôture
    # provisoire est assez fraîche.
    points_annee_courante = cached.loc[cached["year"] == end_year, "date"]
    if points_annee_courante.empty:
        return end_year
    last_date = pd.to_datetime(points_annee_courante.iloc[0])
    age_days = (pd.Timestamp.now().normalize() - last_date.normalize()).days
    return end_year if age_days > refresh_days else None


def fetch_state_path(output_dir: Path) -> Path:
    return output_dir / "fetch_state_prices.json"


def load_fetch_state(output_dir: Path) -> dict:
    """{symbole: {"requested_from": année, "last_attempt": iso}}.

    Mémorise l'année la plus ancienne DÉJÀ RÉCLAMÉE à IBKR pour chaque
    symbole. Sans cette trace, impossible de distinguer "cette année n'a
    jamais été demandée" de "elle a été demandée et IBKR n'avait rien" -- et
    c'est cette confusion qui faisait retélécharger tout l'historique à chaque
    run (voir determine_fetch_start_year)."""
    path = fetch_state_path(output_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("État de suivi illisible (%s), on repart de zéro.", exc)
        return {}


def save_fetch_state(output_dir: Path, state: dict) -> None:
    path = fetch_state_path(output_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.warning("État de suivi non écrit (%s) : le prochain run redemandera plus large.", exc)


def merge_and_save(existing: pd.DataFrame, new_rows: list[dict], output_file: Path) -> pd.DataFrame:
    """Fusionne les nouvelles lignes au cache et écrit le parquet.

    Appelée périodiquement pendant le run et pas seulement à la fin : à ~11
    secondes de pacing IBKR par ticker, un univers complet demande plus d'une
    heure, et une interruption (Ctrl+C, coupure de session, plantage) faisait
    jusqu'ici perdre TOUT le travail du run -- le fichier n'était écrit qu'une
    fois la boucle terminée."""
    if not new_rows:
        return existing
    df_new = pd.DataFrame(new_rows)
    combined = pd.concat([existing, df_new], ignore_index=True) if not existing.empty else df_new
    combined = (
        combined.sort_values(["symbol", "year"])
        .drop_duplicates(subset=["symbol", "year"], keep="last")  # les nouvelles valeurs l'emportent
        .reset_index(drop=True)
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_file, index=False, engine="pyarrow")
    return combined


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
    parser.add_argument(
        "--refresh-days", type=int, default=1,
        help="Rafraîchit l'année en cours si le dernier point en cache a plus de N jours "
             "(défaut: %(default)s). Les années passées, déjà complètes, ne sont jamais "
             "retéléchargées quel que soit cet argument.",
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Ignore le cache et retélécharge tout l'historique pour tous les tickers.",
    )
    args = parser.parse_args()

    end_year = datetime.now().year
    setup_logging(args.output_dir)
    universe = load_universe(args.tickers, limit=args.limit)

    # --output-dir pilote réellement l'emplacement du fichier de sortie
    # (avant : seuls les logs en tenaient compte, le fichier de données
    # allait toujours dans config.PRICES_FILE quel que soit l'argument).
    output_file = args.output_dir / config.PRICES_FILE.name
    existing = pd.read_parquet(output_file) if output_file.exists() else pd.DataFrame(
        columns=["symbol", "ric", "company_name", "country", "currency", "exchange", "year", "date", "close"]
    )
    if not existing.empty:
        logger.info("Cache existant chargé : %s (%d lignes, %d symboles).", output_file, len(existing), existing["symbol"].nunique())

    state = load_fetch_state(args.output_dir)

    plan = []  # (row, fetch_start_year) ; fetch_start_year=None => rien à faire pour ce ticker
    for _, row in universe.iterrows():
        symbol = row["ib_symbol"]
        fetch_start_year = args.start_year if args.force_refresh else determine_fetch_start_year(
            existing, symbol, args.start_year, end_year, args.refresh_days,
            requested_from=state.get(symbol, {}).get("requested_from"),
        )
        plan.append((row, fetch_start_year))

    skip_count = sum(1 for _, fsy in plan if fsy is None)
    to_process = [(row, fsy) for row, fsy in plan if fsy is not None]

    if not to_process:
        logger.info(
            "Les %d tickers de l'univers sont déjà à jour (cache < %d jour(s)) : "
            "aucun appel IBKR nécessaire.", len(universe), args.refresh_days,
        )
        return

    logger.info("%d/%d tickers déjà à jour (ignorés), %d à récupérer.", skip_count, len(universe), len(to_process))

    ib = connect_ib(args.port)
    all_rows: list[dict] = []
    non_sauvegardees: list[dict] = []
    ok_count, fail_count = 0, 0
    now_iso = datetime.now().isoformat(timespec="seconds")

    try:
        for idx, (row, fetch_start_year) in enumerate(to_process):
            symbol = row["ib_symbol"]
            logger.info("[%d/%d] %s (%s), depuis %d...", idx + 1, len(to_process), symbol, row["Instrument_Name"], fetch_start_year)
            last_exc, rows = None, None
            for attempt in range(RETRY_PER_TICKER + 1):
                try:
                    rows = process_ticker(ib, row, fetch_start_year, end_year)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt < RETRY_PER_TICKER:
                        logger.debug("  -> tentative %d échouée pour %s (%s), nouvel essai...", attempt + 1, symbol, exc)
                        ib.sleep(2)

            if rows is not None:
                all_rows.extend(rows)
                non_sauvegardees.extend(rows)
                ok_count += 1
                logger.info("  -> %d années récupérées pour %s", len(rows), symbol)
                # L'année réclamée est notée même quand IBKR ne renvoie rien
                # pour une partie de la plage : c'est précisément ce "demandé,
                # rien reçu" qu'il faut mémoriser pour ne pas le redemander à
                # chaque run (cf. determine_fetch_start_year).
                ancienne = state.get(symbol, {}).get("requested_from")
                state[symbol] = {
                    "requested_from": min(fetch_start_year, ancienne) if ancienne else fetch_start_year,
                    "last_attempt": now_iso,
                }
            else:
                fail_count += 1
                logger.warning("  -> ECHEC pour %s : %s (ticker ignoré, on continue)", symbol, last_exc)

            if len(non_sauvegardees) >= CHECKPOINT_EVERY_ROWS:
                existing = merge_and_save(existing, non_sauvegardees, output_file)
                save_fetch_state(args.output_dir, state)
                logger.info("  (point de sauvegarde : %d lignes au total dans %s)", len(existing), output_file)
                non_sauvegardees = []
    finally:
        # Toujours exécuté, y compris sur Ctrl+C ou coupure de session : le
        # travail déjà fait ne doit jamais être perdu.
        if non_sauvegardees:
            existing = merge_and_save(existing, non_sauvegardees, output_file)
        save_fetch_state(args.output_dir, state)
        try:
            ib.disconnect()
            logger.info("Déconnecté d'IBKR.")
        except Exception:  # noqa: BLE001
            pass

    logger.info(
        "Terminé. OK: %d | Échecs: %d | Déjà à jour (ignorés): %d | Nouvelles lignes: %d",
        ok_count, fail_count, skip_count, len(all_rows),
    )

    if not all_rows:
        if existing.empty:
            logger.warning("Aucune donnée récupérée, pas de fichier de sortie généré.")
        else:
            logger.info("Rien de nouveau à écrire : fichier existant conservé tel quel (%s).", output_file)
        return

    combined = existing
    logger.info(
        "Fichier écrit : %s (%d lignes au total, %d nouvelles/mises à jour ce run).",
        output_file, len(combined), len(all_rows),
    )


if __name__ == "__main__":
    main()
