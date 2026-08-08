"""
Construit l'univers POINT-IN-TIME du S&P 500 (composants actuels + composants
historiquement radiés, avec leurs dates d'entrée/sortie de l'indice), pour
permettre un backtest sans biais de survivance.

01_build_universe.py ne capture que la photo ACTUELLE de l'indice : une
entreprise rachetée, en faillite ou simplement sortie de l'indice avant
aujourd'hui n'y apparaît jamais, alors que "01" tourne à chaque fois -- un
backtest qui n'utiliserait que cet univers actuel appliqué rétroactivement
surestimerait systématiquement la performance passée (biais de survivance :
on ne regarde que les entreprises qui ont "survécu" jusqu'à aujourd'hui).

Source : la même page Wikipedia que 01, mais sa DEUXIÈME table ("Selected
changes to the components of the S&P 500"), qui liste chaque ajout/retrait
avec sa date. Limite connue : ce tableau ne remonte pas à la création de
l'indice (1957) mais seulement à ~1996-2000 selon les révisions -- une
entreprise radiée avant le début du suivi Wikipedia n'apparaîtra pas ici. Best
effort, cohérent avec le reste du pipeline (cf. 04 : historique SEC limité
aux entreprises qui ont déjà déposé du XBRL).

Sorties :
    - config.UNIVERSE_HISTORY_FILE (parquet) : un span d'appartenance par
      ligne (ric, company_name, start_date, end_date [NaT = toujours membre],
      is_current_member). Une entreprise qui est sortie de l'indice PUIS
      revenue plus tard aura plusieurs lignes (plusieurs spans non contigus).
    - config.UNIVERSE_FULL_FILE (csv) : superset RIC/Instrument_Name/Country/
      Currency/Exchange (même schéma que config.UNIVERSE_FILE, sortie de 01),
      actuels + radiés, dédupliqué par RIC -- à passer en --tickers à 03b et
      04 pour backfiller aussi les cours et financials des entreprises
      radiées.

Usage :
    python 01b_historique_univers_sp500.py
"""

from __future__ import annotations

import io
import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "options-pipeline/1.0 (contact: variable d'environnement SEC_CONTACT_EMAIL)"}


_tables_cache: Optional[List[pd.DataFrame]] = None


def _fetch_tables() -> List[pd.DataFrame]:
    """Tables de la page Wikipedia, téléchargées UNE SEULE FOIS par run.

    build_universe_history appelle load_current_constituents puis
    load_changes_log, qui appelaient chacune cette fonction : la page (plus
    d'un mégaoctet) était téléchargée et re-parsée trois fois par run, pour un
    contenu strictement identique."""
    global _tables_cache
    if _tables_cache is None:
        resp = requests.get(WIKI_URL, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        _tables_cache = pd.read_html(io.StringIO(resp.text))
    return _tables_cache


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """La table des changements a des colonnes MultiIndex (ex: ("Added",
    "Ticker")). On les aplatit en "Added_Ticker", "Removed_Ticker", etc. Une
    colonne à un seul niveau utile (ex: "Date", pas de sous-colonne) devient
    simplement "Date"."""
    df = df.copy()
    flat_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = [str(p).strip() for p in col if str(p).strip() and "Unnamed" not in str(p)]
            flat_cols.append("_".join(parts) if parts else "?")
        else:
            flat_cols.append(str(col).strip())
    df.columns = flat_cols
    return df


def load_current_constituents() -> pd.DataFrame:
    """Réutilise la même logique que 01_build_universe.build_universe(), mais
    conserve en plus 'Date added' (nécessaire ici pour dater le début du span
    des membres actuels)."""
    tables = _fetch_tables()
    raw = tables[0]
    df = pd.DataFrame({
        "ric": raw["Symbol"].str.strip(),
        "company_name": raw["Security"].str.strip(),
        "date_added_raw": raw["Date added"] if "Date added" in raw.columns else None,
    })
    df = df.dropna(subset=["ric"]).drop_duplicates(subset=["ric"]).reset_index(drop=True)
    df["date_added"] = pd.to_datetime(df["date_added_raw"], errors="coerce").dt.date
    return df[["ric", "company_name", "date_added"]]


def load_changes_log() -> pd.DataFrame:
    """Table des changements ("Selected changes to the components of the
    S&P 500"), avec colonnes Date / Added_Ticker / Added_Security /
    Removed_Ticker / Removed_Security. Retourne un DataFrame vide (avec les
    bonnes colonnes) si la table est absente ou dans un format inattendu,
    plutôt que de planter tout le script -- l'univers PIT retombe alors sur
    les seuls membres actuels (comme 01), ce qui est signalé via un warning."""
    empty = pd.DataFrame(columns=["date", "added_ticker", "added_security", "removed_ticker", "removed_security"])
    try:
        tables = _fetch_tables()
        if len(tables) < 2:
            logger.warning("Pas de deuxième table sur la page Wikipedia : historique des changements indisponible.")
            return empty
        raw = _flatten_columns(tables[1])
    except Exception as exc:  # noqa: BLE001
        logger.error("Échec de récupération de la table des changements : %s", exc)
        return empty

    rename_map = {
        "Date": "date",
        "Added_Ticker": "added_ticker",
        "Added_Security": "added_security",
        "Removed_Ticker": "removed_ticker",
        "Removed_Security": "removed_security",
    }
    missing = [c for c in ["Date", "Added_Ticker", "Removed_Ticker"] if c not in raw.columns]
    if missing:
        logger.error(
            "Colonnes attendues absentes de la table des changements (%s) -- "
            "structure Wikipedia probablement modifiée, historique ignoré.", missing,
        )
        return empty

    df = raw.rename(columns=rename_map)
    for col in ["added_security", "removed_security"]:
        if col not in df.columns:
            df[col] = None
    df = df[["date", "added_ticker", "added_security", "removed_ticker", "removed_security"]]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    return df


def _build_events_by_ticker(changes: pd.DataFrame) -> Dict[str, List[Tuple[date, str]]]:
    """{ticker: [(date, 'added'|'removed'), ...]} trié par date croissante."""
    events: Dict[str, List[Tuple[date, str]]] = {}
    for _, row in changes.iterrows():
        d = row["date"]
        added = row.get("added_ticker")
        removed = row.get("removed_ticker")
        if isinstance(added, str) and added.strip():
            events.setdefault(added.strip(), []).append((d, "added"))
        if isinstance(removed, str) and removed.strip():
            events.setdefault(removed.strip(), []).append((d, "removed"))
    for ticker in events:
        events[ticker].sort(key=lambda t: t[0])
    return events


def _build_spans_for_ticker(
    ticker_events: List[Tuple[date, str]], is_current_member: bool, current_date_added: Optional[date],
) -> List[Tuple[Optional[date], Optional[date]]]:
    """Rejoue chronologiquement les événements 'added'/'removed' d'un ticker
    et retourne la liste de ses spans d'appartenance (start, end). end=None
    signifie "toujours membre aujourd'hui". start=None signifie "membre
    depuis avant le début du suivi Wikipedia des changements" (span tronqué).

    Une entreprise peut sortir puis revenir dans l'indice : plusieurs spans
    non contigus sont alors retournés."""
    spans: List[Tuple[Optional[date], Optional[date]]] = []
    open_start: Optional[date] = None
    for d, action in ticker_events:
        if action == "added":
            if open_start is not None:
                # Deux 'added' consécutifs sans 'removed' entre les deux :
                # anomalie de la table Wikipedia, on ferme le span précédent
                # à cette date plutôt que de la perdre silencieusement.
                spans.append((open_start, d))
            open_start = d
        elif action == "removed":
            if open_start is None:
                # Retrait logué sans ajout logué avant lui : l'entreprise
                # était déjà membre avant le début du suivi des changements.
                spans.append((None, d))
            else:
                spans.append((open_start, d))
                open_start = None

    if open_start is not None:
        spans.append((open_start, None))
    elif is_current_member:
        # Membre actuel dont le dernier événement loggé est un 'removed' (ou
        # qui n'a aucun événement du tout) : span ouvert, daté par "Date
        # added" de la table des membres actuels si disponible.
        spans.append((current_date_added, None))

    return spans


def build_universe_history() -> pd.DataFrame:
    current = load_current_constituents()
    changes = load_changes_log()

    current_by_ticker = current.set_index("ric")[["company_name", "date_added"]].to_dict("index")
    events_by_ticker = _build_events_by_ticker(changes)

    all_tickers = set(current_by_ticker) | set(events_by_ticker)
    logger.info(
        "%d membres actuels, %d tickers supplémentaires vus dans l'historique des changements (%d au total).",
        len(current_by_ticker), len(all_tickers - set(current_by_ticker)), len(all_tickers),
    )

    rows = []
    for ticker in sorted(all_tickers):
        is_current = ticker in current_by_ticker
        current_added = current_by_ticker.get(ticker, {}).get("date_added")
        company_name = current_by_ticker.get(ticker, {}).get("company_name")
        ticker_events = events_by_ticker.get(ticker, [])

        if not company_name:
            # Ticker radié : le nom vient de la table des changements (celle
            # du retrait, à défaut celle de l'ajout).
            names = [
                row["removed_security"] for row in changes.to_dict("records")
                if row.get("removed_ticker") == ticker and isinstance(row.get("removed_security"), str)
            ] or [
                row["added_security"] for row in changes.to_dict("records")
                if row.get("added_ticker") == ticker and isinstance(row.get("added_security"), str)
            ]
            company_name = names[0] if names else ticker

        spans = _build_spans_for_ticker(ticker_events, is_current, current_added)
        if not spans:
            # Membre actuel sans aucun événement loggé ET sans "Date added"
            # exploitable : span entièrement ouvert (début inconnu).
            spans = [(current_added, None)]

        for start, end in spans:
            rows.append({
                "ric": ticker,
                "company_name": company_name,
                "start_date": start,
                "end_date": end,
                "is_current_member": is_current,
            })

    df = pd.DataFrame(rows).sort_values(["ric", "start_date"], na_position="first").reset_index(drop=True)
    return df


def universe_asof(history: pd.DataFrame, as_of: date) -> List[str]:
    """Liste des RIC membres du S&P 500 à une date donnée. Utilisée par le
    data_loader du backtest (un span est actif si start_date <= as_of et
    (end_date is None ou end_date > as_of))."""
    if history.empty:
        return []
    start_ok = history["start_date"].isna() | (history["start_date"] <= as_of)
    end_ok = history["end_date"].isna() | (history["end_date"] > as_of)
    return sorted(history.loc[start_ok & end_ok, "ric"].unique().tolist())


def build_full_universe_csv(history: pd.DataFrame) -> pd.DataFrame:
    """Superset RIC/Instrument_Name/Country/Currency/Exchange, même schéma
    que config.UNIVERSE_FILE (sortie de 01), pour que 03b/04 puissent le
    consommer sans transformation. Exchange="SMART" pour tout le monde : les
    tickers radiés depuis longtemps n'ont de toute façon plus de place de
    cotation fiable à renseigner, et SMART reste le routage par défaut du
    reste du pipeline (cf. config.DEFAULT_EXCHANGE)."""
    dedup = history.sort_values("is_current_member", ascending=False).drop_duplicates(subset=["ric"], keep="first")
    return pd.DataFrame({
        "RIC": dedup["ric"],
        "Instrument_Name": dedup["company_name"],
        "Country": config.COUNTRY,
        "Currency": config.CURRENCY,
        "Exchange": "SMART",
    }).sort_values("RIC").reset_index(drop=True)


def main() -> None:
    history = build_universe_history()
    config.UNIVERSE_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(config.UNIVERSE_HISTORY_FILE, index=False, engine="pyarrow")
    logger.info(
        "Historique d'univers sauvegardé : %s (%d spans, %d tickers distincts, %d radiés).",
        config.UNIVERSE_HISTORY_FILE, len(history), history["ric"].nunique(),
        history.loc[~history["is_current_member"], "ric"].nunique(),
    )

    full = build_full_universe_csv(history)
    full.to_csv(config.UNIVERSE_FULL_FILE, index=False, encoding="utf-8-sig")
    logger.info(
        "Univers complet (actuels + radiés) sauvegardé : %s (%d entreprises). "
        "Passe ce fichier en --tickers à 03b_recuperation_cours_quotidiens.py et "
        "04_recuperation_10k.py pour backfiller aussi les entreprises radiées.",
        config.UNIVERSE_FULL_FILE, len(full),
    )


if __name__ == "__main__":
    main()
