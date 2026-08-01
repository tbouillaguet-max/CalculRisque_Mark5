"""
Extrait les données financières annuelles (10-K) de toutes les entreprises
de l'univers US, via l'API XBRL "companyfacts" de la SEC.

Corrections majeures par rapport à Recuperation10KMark2 :
    - `import json` manquant dans l'original -> save_financials plantait
      systématiquement (NameError) dès le premier appel.
    - L'URL de téléchargement d'origine
      (sec.gov/Archives/edgar/data/{cik}/{accession}/R{accession}.xml)
      ne correspond à AUCUN endpoint EDGAR réel : "R*.xml" désigne les
      fichiers de rendu interne d'un filing (R1.htm, R2.htm...), pas un
      export XBRL consolidé, et le nom de fichier ne suit pas ce format.
      Le script ne pouvait donc jamais télécharger de données valides.
    - Remplacé par l'API officielle "companyfacts", qui renvoie en UN SEUL
      appel JSON l'intégralité des faits XBRL taggés d'une entreprise sur
      toute son historique (au lieu de télécharger et parser le XBRL de
      chaque filing un par un) : plus simple, plus robuste, et beaucoup
      moins de requêtes (donc plus respectueux des limites de la SEC).
    - User-Agent conforme à la politique d'accès équitable de la SEC
      (https://www.sec.gov/os/webmaster-faq#developers) : ils demandent un
      User-Agent identifiant un contact réel ("NomOutil contact@email.com"),
      pas un User-Agent de navigateur. À renseigner dans SEC_CONTACT_EMAIL
      ci-dessous avant utilisation, sous peine de blocage (429/403).
    - Écrit un fichier consolidé unique (config.FINANCIALS_FILE, Parquet)
      pour tous les tickers/années au lieu d'un JSON par ticker/année, afin
      que 05_calcul_multiples.py puisse le lire directement.
    - Colonnes renommées dans le schéma canonique (revenue, ebitda, ebit,
      net_income, capex, da, working_capital, net_debt, cash, tax_rate,
      interest_expense, shares_outstanding) au lieu du mélange FR/EN
      d'origine ("CA", "Dette net", "Nombre d'action"...) qui empêchait
      06/07/08 de relire correctement les données produites par ce script.
    - NOUVEAU : throttle par ticker (data/financials/fetch_state.json) et
      fusion avec le fichier existant au lieu de tout réinterroger/écraser
      à chaque run. Un 10-K annuel ne sort qu'une fois par an : inutile
      d'interroger companyfacts pour un ticker déjà interrogé il y a moins
      de --refresh-days jours (défaut 30). --force-refresh retrouve
      l'ancien comportement (interroge tous les tickers demandés).

Débit et robustesse réseau : les tickers sont interrogés en PARALLÈLE
(--workers, 8 par défaut) sur une connexion HTTP persistante partagée, sous
un limiteur de débit global calé sur la limite d'accès équitable de la SEC
(MAX_REQUESTS_PER_SECOND). Le nombre de requêtes envoyées à la SEC est
identique à avant -- seule leur mise en attente change : le script passait
l'essentiel de son temps à attendre une réponse, une poignée à la fois.
Les erreurs transitoires (429, 5xx) sont réessayées automatiquement avec un
backoff exponentiel, et un ticker en échec n'est PAS marqué comme récupéré :
il sera réinterrogé au run suivant au lieu d'être ignoré pendant
--refresh-days jours.

Usage :
    python 04_recuperation_10k.py                 # tout l'univers US
    python 04_recuperation_10k.py --limit 10       # test rapide
    python 04_recuperation_10k.py --ticker AAPL    # une seule entreprise
    python 04_recuperation_10k.py --refresh-days 7  # throttle plus court
    python 04_recuperation_10k.py --force-refresh   # réinterroge tous les tickers demandés
    python 04_recuperation_10k.py --workers 4       # moins de parallélisme (réseau lent/instable)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

import config

FETCH_STATE_FILE = config.DIR_FINANCIALS / "fetch_state.json"
REFRESH_DAYS_DEFAULT = 30

# ⚠️ À COMPLÉTER : la SEC exige un User-Agent identifiant un contact réel.
# Surchargeable par la variable d'environnement du même nom, pour déployer le
# pipeline (cron, conteneur) sans éditer le script.
SEC_CONTACT_EMAIL = os.getenv("SEC_CONTACT_EMAIL", "jeanboubou1er@gmail.com")

HEADERS = {
    "User-Agent": f"OptionsPipeline/1.0 ({SEC_CONTACT_EMAIL})",
    "Accept": "application/json",
}

TICKERS_JSON_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Politique d'accès équitable de la SEC : ~10 requêtes/seconde avec un
# User-Agent identifiant un contact réel. On reste juste en dessous, tous
# threads confondus (voir _RateLimiter).
MAX_REQUESTS_PER_SECOND = 8.0
DEFAULT_WORKERS = 8

# Tags candidats par métrique, dans l'ordre de préférence (certaines
# entreprises taguent différemment selon les années/schémas). Un tag préfixé
# "namespace:" cherche dans cette taxonomie XBRL au lieu de "us-gaap" par
# défaut (voir extract_annual_values) -- utilisé pour shares_outstanding,
# dont beaucoup d'entreprises (ex: V, VLO) ne taguent jamais la version
# us-gaap et ne renseignent que la page de garde du 10-K (taxonomie "dei",
# quasi universelle chez les émetteurs SEC).
XBRL_TAGS: Dict[str, List[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "ebit": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "short_term_debt": ["DebtCurrent", "ShortTermBorrowings"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "CapitalExpenditures"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet"],
    "income_tax_expense": ["IncomeTaxExpenseBenefit"],
    "interest_expense": ["InterestExpense"],
    "shares_outstanding": [
        "CommonStockSharesOutstanding", "dei:EntityCommonStockSharesOutstanding", "CommonStockSharesIssued",
    ],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class _RateLimiter:
    """Espace les requêtes d'AU MOINS 1/rate seconde, tous threads confondus.

    L'attente a lieu à l'intérieur du verrou : les threads se mettent
    naturellement en file et le débit global reste borné, quel que soit leur
    nombre -- c'est cette garantie qui permet de paralléliser sans dépasser
    la limite d'accès équitable de la SEC."""

    def __init__(self, rate_per_second: float):
        self._min_interval = 1.0 / rate_per_second
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait > 0:
                time.sleep(wait)
                now = self._next_slot
            self._next_slot = now + self._min_interval


_rate_limiter = _RateLimiter(MAX_REQUESTS_PER_SECOND)
_session_local = threading.local()


def _build_session(pool_size: int) -> requests.Session:
    """Session HTTP persistante : la connexion TLS vers data.sec.gov est
    négociée une fois puis réutilisée pour tous les tickers (elle l'était
    pour chacun d'eux auparavant).

    Les réessais sont gérés par _get et NON par l'adaptateur (max_retries=0) :
    un réessai interne à urllib3 ne repasserait pas par le limiteur de débit,
    et rejouerait donc une requête hors quota -- précisément au moment où la
    SEC vient de nous signaler qu'on en envoie trop (429)."""
    session = requests.Session()
    session.headers.update(HEADERS)
    adapter = HTTPAdapter(max_retries=0, pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("https://", adapter)
    return session


RETRYABLE_STATUS = frozenset((429, 500, 502, 503, 504))
MAX_ATTEMPTS = 4


def _retry_delay(response: Optional[requests.Response], attempt: int) -> float:
    """Backoff exponentiel (2s, 4s, 8s), sauf si la SEC indique elle-même
    combien de temps attendre via Retry-After."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return float(retry_after)
    return 2.0 ** (attempt + 1)


def _get(url: str, session: Optional[requests.Session] = None) -> Optional[dict]:
    if session is None:
        session = getattr(_session_local, "session", None)
        if session is None:
            session = _session_local.session = _build_session(DEFAULT_WORKERS)

    for attempt in range(MAX_ATTEMPTS):
        _rate_limiter.acquire()
        response = None
        try:
            response = session.get(url, timeout=30)
            if response.status_code in RETRYABLE_STATUS:
                raise requests.exceptions.HTTPError(f"HTTP {response.status_code}", response=response)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            if attempt == MAX_ATTEMPTS - 1:
                logger.error("Échec de la requête %s (%d tentatives) : %s", url, MAX_ATTEMPTS, exc)
                return None
            delay = _retry_delay(response, attempt)
            logger.warning("Requête %s en échec (%s), nouvelle tentative dans %.0fs.", url, exc, delay)
            time.sleep(delay)
    return None


def get_cik_map(session: Optional[requests.Session] = None) -> Dict[str, str]:
    """Ticker -> CIK (10 chiffres), depuis le fichier officiel SEC."""
    data = _get(TICKERS_JSON_URL, session=session)
    if not data:
        return {}
    return {entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in data.values()}


def extract_annual_values(facts: dict, tags: List[str]) -> Dict[int, Tuple[float, str]]:
    """
    Pour une métrique donnée (plusieurs tags candidats, dans l'ordre de
    préférence), retourne {année_fiscale: (valeur, date_de_dépôt)} en ne
    gardant que les faits associés à un 10-K annuel (form == "10-K",
    fp == "FY"). La date de dépôt ("filed", format "YYYY-MM-DD") est
    retournée en plus de la valeur (et non plus jetée après sélection) : le
    backtest (09_backtest.py) en a besoin pour savoir à partir de quand cette
    valeur était réellement publique -- l'utiliser dès le 31/12 de l'exercice
    serait du look-ahead bias (un 10-K est déposé ~2-3 mois après la clôture).

    Un tag "namespace:tag" (ex: "dei:EntityCommonStockSharesOutstanding")
    cherche dans cette taxonomie XBRL au lieu de "us-gaap" par défaut --
    nécessaire pour shares_outstanding, que beaucoup d'entreprises ne taguent
    qu'en "dei" (page de garde du 10-K) et jamais en "us-gaap".

    Tous les tags de la liste sont consultés (pas seulement le premier qui a
    des données) : le premier tag reste prioritaire pour une année donnée,
    mais si une année n'est couverte par aucun fait de ce tag, on va chercher
    la valeur dans les tags suivants. Nécessaire car de nombreuses
    entreprises ont changé de tag XBRL en cours d'historique (ex: passage de
    "Revenues" à "RevenueFromContractWithCustomerExcludingAssessedTax" lors
    du changement de norme comptable ASC 606 en 2018) : s'arrêter au premier
    tag non vide faisait perdre silencieusement toutes les années couvertes
    uniquement par un tag de repli.
    """
    all_facts = facts.get("facts", {})
    by_year: Dict[int, tuple] = {}  # year -> (tag_priority, filed_date, value)

    for priority, tag_spec in enumerate(tags):
        namespace, _, tag = tag_spec.rpartition(":") if ":" in tag_spec else ("us-gaap", "", tag_spec)
        entry = all_facts.get(namespace, {}).get(tag)
        if not entry:
            continue
        for unit_values in entry.get("units", {}).values():
            for v in unit_values:
                if v.get("form") != "10-K" or v.get("fp") != "FY":
                    continue
                fy = v.get("fy")
                filed = v.get("filed", "")
                val = v.get("val")
                if fy is None or val is None:
                    continue
                current = by_year.get(fy)
                if current is None:
                    # Aucune valeur encore trouvée pour cette année (ni par un
                    # tag mieux classé, ni par celui-ci) -> on la prend.
                    by_year[fy] = (priority, filed, val)
                elif priority == current[0] and filed > current[1]:
                    # Même tag que la valeur déjà retenue pour cette année :
                    # on garde le dépôt le plus récent (ex: restatement).
                    by_year[fy] = (priority, filed, val)
                # Si un tag mieux classé (priority plus petite) a déjà fourni
                # une valeur pour cette année, on ne l'écrase pas avec un tag
                # de repli moins prioritaire.

    return {year: (val, filed) for year, (_, filed, val) in by_year.items()}


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Métriques dérivées, calculées sur le schéma canonique déjà en place.

    Vectorisé sur tout le DataFrame plutôt qu'appliqué ligne par ligne, à
    comportement strictement identique -- y compris la subtilité du `x or 0`
    d'origine : un NaN étant "vrai" en Python, il n'était PAS remplacé par 0
    et se propageait aux métriques dérivées, contrairement à un None (métrique
    qu'aucun exercice ne tague, colonne restée vide)."""
    def zeroed(column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series(0, index=df.index)
        series = df[column]
        if series.dtype == object:
            series = series.map(lambda value: value or 0)
        return pd.to_numeric(series, errors="coerce")

    def raw(column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[column], errors="coerce")

    long_term_debt, short_term_debt = zeroed("long_term_debt"), zeroed("short_term_debt")
    net_income, da, capex = zeroed("net_income"), zeroed("da"), zeroed("capex")
    income_tax_expense = zeroed("income_tax_expense")
    total_debt = long_term_debt + short_term_debt
    ebit, current_assets, current_liabilities = raw("ebit"), raw("current_assets"), raw("current_liabilities")

    df["net_debt"] = total_debt - zeroed("cash")
    # EBIT réellement manquant (None/NaN) -> EBITDA laissé à None plutôt que
    # d'être silencieusement égal à la D&A seule (cf. rapport, erreur 2.3).
    df["ebitda"] = (ebit + da).where(ebit.notna())
    df["fcf"] = net_income + da - capex
    df["working_capital"] = (current_assets - current_liabilities).where(
        current_assets.notna() & current_liabilities.notna()
    )
    df["cost_of_debt"] = (zeroed("interest_expense") / (total_debt + 1e-6)).where(total_debt != 0)
    pretax_income = net_income + income_tax_expense
    df["tax_rate"] = (income_tax_expense / pretax_income).where(pretax_income != 0)
    return df


def extract_financials_for_ticker(ric: str, cik: str, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """ric : ticker tel qu'utilisé par la SEC pour retrouver le CIK (peut
    contenir un point, ex "BRK.B"). La colonne 'symbol' du DataFrame de
    sortie utilise en revanche config.to_ib_symbol(ric), pour être
    identique au 'symbol' produit par 03/04 (cf. erreur 2.1 du rapport)."""
    facts = _get(COMPANYFACTS_URL.format(cik=cik), session=session)
    if not facts:
        logger.warning("Aucune donnée companyfacts pour %s (CIK %s)", ric, cik)
        return pd.DataFrame()

    per_metric = {metric: extract_annual_values(facts, tags) for metric, tags in XBRL_TAGS.items()}
    all_years = sorted({year for values in per_metric.values() for year in values})
    if not all_years:
        logger.warning("Aucun exercice 10-K annuel trouvé pour %s", ric)
        return pd.DataFrame()

    symbol = config.to_ib_symbol(ric)
    rows = []
    for year in all_years:
        row = {"symbol": symbol, "cik": cik, "year": year}
        filed_dates_this_year = []
        for metric, values in per_metric.items():
            entry = values.get(year)
            if entry is None:
                row[metric] = None
                continue
            val, filed = entry
            row[metric] = val
            if filed:
                filed_dates_this_year.append(filed)
        # Date de dépôt retenue pour l'exercice : la PLUS TARDIVE parmi les
        # métriques utilisées (pas la première) -- tant que toutes les
        # métriques nécessaires au DCF n'ont pas été déposées, l'exercice
        # n'est pas exploitable pour un backtest point-in-time.
        row["filed_date"] = max(filed_dates_this_year) if filed_dates_this_year else None
        rows.append(row)

    return compute_derived(pd.DataFrame(rows))


def load_fetch_state(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("État de suivi illisible (%s), on repart de zéro.", e)
        return {}


def save_fetch_state(path: Path, state: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def should_skip(symbol: str, existing: pd.DataFrame, state: Dict[str, str], refresh_days: int) -> bool:
    """Un ticker est ignoré (aucun appel SEC) s'il a déjà des données en
    cache ET a été interrogé il y a moins de refresh_days jours. Un 10-K
    n'est déposé qu'une fois par an : pas besoin de rappeler companyfacts
    à chaque run pour un ticker déjà à jour récemment."""
    if symbol not in state:
        return False
    if existing.empty or symbol not in set(existing["symbol"]):
        return False
    last_fetched = datetime.fromisoformat(state[symbol])
    return (datetime.now() - last_fetched).days <= refresh_days


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", type=Path, default=config.UNIVERSE_FILE)
    parser.add_argument("--ticker", type=str, default=None, help="Une seule entreprise (ex: AAPL)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--refresh-days", type=int, default=REFRESH_DAYS_DEFAULT,
        help="Ignore un ticker déjà interrogé il y a moins de N jours (défaut: %(default)s).",
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Ignore le throttle et interroge la SEC pour tous les tickers demandés.",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="Tickers interrogés en parallèle (défaut: %(default)s). Le débit vers la "
             f"SEC reste plafonné à {MAX_REQUESTS_PER_SECOND:.0f} requêtes/seconde quel que soit ce nombre.",
    )
    args = parser.parse_args()

    if SEC_CONTACT_EMAIL == "ton_email@example.com":
        logger.warning(
            "SEC_CONTACT_EMAIL n'a pas été renseigné en haut du script : la SEC "
            "peut bloquer les requêtes avec un User-Agent générique. "
            "Modifie SEC_CONTACT_EMAIL avant un run complet."
        )

    if args.ticker:
        symbols = [args.ticker.upper()]
    else:
        universe = pd.read_csv(args.tickers, encoding="utf-8-sig")
        symbols = universe["RIC"].dropna().unique().tolist()
        if args.limit:
            symbols = symbols[: args.limit]

    existing = pd.read_parquet(config.FINANCIALS_FILE) if config.FINANCIALS_FILE.exists() else pd.DataFrame()
    if not existing.empty:
        logger.info("Cache existant chargé : %s (%d lignes, %d symboles).", config.FINANCIALS_FILE, len(existing), existing["symbol"].nunique())

    state = load_fetch_state(FETCH_STATE_FILE)
    to_query = [
        t for t in symbols
        if args.force_refresh or not should_skip(config.to_ib_symbol(t), existing, state, args.refresh_days)
    ]
    skip_count = len(symbols) - len(to_query)

    if not to_query:
        logger.info(
            "Les %d tickers demandés sont déjà à jour (interrogés il y a moins de %d jours) : "
            "aucun appel SEC nécessaire.", len(symbols), args.refresh_days,
        )
        return

    workers = max(1, args.workers)
    logger.info("%d/%d tickers déjà à jour (ignorés), %d à interroger.", skip_count, len(symbols), len(to_query))
    session = _build_session(workers)
    logger.info("Récupération des CIK SEC pour %d tickers...", len(to_query))
    cik_map = get_cik_map(session=session)

    with_cik = [(t, cik_map[t.upper()]) for t in to_query if t.upper() in cik_map]
    for ticker in to_query:
        if ticker.upper() not in cik_map:
            logger.warning("CIK introuvable pour %s, ignoré.", ticker)

    all_frames = []
    ok_count, fail_count = 0, len(to_query) - len(with_cik)
    now_iso = datetime.now().isoformat(timespec="seconds")

    logger.info("Interrogation de companyfacts sur %d threads...", workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(extract_financials_for_ticker, ticker, cik, session): ticker
            for ticker, cik in with_cik
        }
        for done, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            df_ticker = future.result()
            if df_ticker.empty:
                # Pas de marquage dans `state` : un échec (réseau, CIK sans
                # 10-K exploitable) doit être réessayé au prochain run, pas
                # ignoré pendant --refresh-days jours.
                fail_count += 1
                continue
            state[config.to_ib_symbol(ticker)] = now_iso
            all_frames.append(df_ticker)
            ok_count += 1
            if done % 25 == 0 or done == len(futures):
                logger.info("[%d/%d] tickers traités...", done, len(futures))

    logger.info("Terminé. OK: %d | Échecs: %d | Déjà à jour (ignorés): %d", ok_count, fail_count, skip_count)
    save_fetch_state(FETCH_STATE_FILE, state)

    if not all_frames:
        if existing.empty:
            logger.warning("Aucune donnée financière récupérée, pas de fichier de sortie généré.")
        else:
            logger.info("Rien de nouveau à écrire : fichier existant conservé tel quel (%s).", config.FINANCIALS_FILE)
        return

    df_new = pd.concat(all_frames, ignore_index=True)
    combined = pd.concat([existing, df_new], ignore_index=True) if not existing.empty else df_new
    combined = (
        combined.sort_values(["symbol", "year"])
        .drop_duplicates(subset=["symbol", "year"], keep="last")  # les nouvelles valeurs l'emportent (ex: restatement)
        .reset_index(drop=True)
    )
    config.FINANCIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(config.FINANCIALS_FILE, index=False, engine="pyarrow")
    logger.info(
        "Fichier écrit : %s (%d lignes au total, %d nouvelles/mises à jour ce run).",
        config.FINANCIALS_FILE, len(combined), len(df_new),
    )


if __name__ == "__main__":
    main()
