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
    - CORRIGÉ : extract_annual_values classait les faits par `v["fy"]`, qui
      désigne l'exercice du RAPPORT et non la période décrite par le fait --
      les trois exercices comparatifs d'un même 10-K atterrissaient sous la
      même clé et c'est un fait arbitraire qui l'emportait. Le rattachement
      se fait désormais sur `end`/`start` (voir sa docstring et le module
      partagé sec_xbrl.py). CE CORRECTIF INVALIDE financials.parquet : il
      faut relancer `04 --force-refresh`, puis 05, 07 et 06b avant tout
      backtest.
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
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import config
import sec_http
import sec_xbrl

FETCH_STATE_FILE = config.DIR_FINANCIALS / "fetch_state.json"
REFRESH_DAYS_DEFAULT = 30

TICKERS_JSON_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Débit et réessais : voir sec_http (module partagé avec 04b/04c/sec_filings_text).
MAX_REQUESTS_PER_SECOND = sec_http.MAX_REQUESTS_PER_SECOND
DEFAULT_WORKERS = sec_http.DEFAULT_WORKERS

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

# Chaque métrique doit être classée flux XOR stock : c'est cette
# classification qui décide si un fait XBRL doit passer le filtre de durée
# annuelle (voir sec_xbrl.collect_annual_entries). Une métrique ajoutée à
# XBRL_TAGS sans être classée ferait échouer l'assertion au chargement du
# module plutôt que d'être silencieusement traitée comme un poste de bilan.
assert sec_xbrl.FLOW_METRICS | sec_xbrl.STOCK_METRICS == set(XBRL_TAGS), (
    "Chaque tag XBRL doit être classé flux XOR stock dans sec_xbrl."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Le limiteur de débit, la session persistante et la politique de réessai
# vivent dans sec_http : ils étaient écrits ici, mais 04b, 04c et
# sec_filings_text n'en avaient aucun équivalent (voir la docstring de ce
# module). Les alias ci-dessous gardent les noms locaux utilisés plus bas.
_build_session = sec_http.build_session
RETRYABLE_STATUS = sec_http.RETRYABLE_STATUS
MAX_ATTEMPTS = sec_http.MAX_ATTEMPTS


def _get(url: str, session: Optional[requests.Session] = None) -> Optional[dict]:
    """None sur échec, quel qu'en soit le motif -- comportement historique de
    ce script, conservé : extract_financials_for_ticker traite déjà l'absence
    de données comme un échec à réessayer au run suivant (le ticker n'est pas
    marqué dans fetch_state)."""
    return sec_http.get_json_or_none(url, session=session)


def get_cik_map(session: Optional[requests.Session] = None) -> Dict[str, str]:
    """Ticker -> CIK (10 chiffres), depuis le fichier officiel SEC."""
    data = _get(TICKERS_JSON_URL, session=session)
    if not data:
        return {}
    return {entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in data.values()}


def extract_annual_values(
    facts: dict, tags: List[str], metric_is_flow: bool, period_ends: Optional[List[str]] = None,
) -> Dict[int, Tuple[float, str]]:
    """
    Pour une métrique donnée (plusieurs tags candidats, dans l'ordre de
    préférence), retourne {exercice: (valeur, date_de_dépôt)}. La date de
    dépôt ("filed", format "YYYY-MM-DD") est retournée en plus de la valeur :
    le backtest (09_backtest.py) en a besoin pour savoir à partir de quand
    cette valeur était réellement publique -- l'utiliser dès le 31/12 de
    l'exercice serait du look-ahead bias (un 10-K est déposé ~2-3 mois après
    la clôture).

    CORRIGÉ : la période d'un fait se lit sur `start`/`end`, pas sur `fy`
    ------------------------------------------------------------------------
    L'ancienne version filtrait sur (form == "10-K", fp == "FY") puis classait
    par `v["fy"]`. Or `fy`/`fp` décrivent LE RAPPORT dans lequel le fait
    apparaît, pas la période qu'il mesure : les comparatifs FY2022 et FY2021
    publiés dans un 10-K FY2023 sortent tous du JSON avec fy=2023, fp="FY",
    form="10-K" et la MÊME date `filed`. Trois exercices atterrissaient donc
    sous la même clé, et le départage `elif priority == current[0] and
    filed > current[1]` ne pouvait pas les séparer (égalité stricte des
    `filed`) : c'est le premier fait rencontré dans l'ordre du JSON qui
    l'emportait, arbitrairement -- souvent la période la plus ancienne.

    Le rattachement se fait désormais sur l'année de `end`, après un filtre de
    durée pour les postes de flux (voir sec_xbrl, module partagé avec
    04b_recuperation_10q.py qui applique la même mécanique à sa fenêtre
    trimestrielle). `metric_is_flow` dit lequel des deux régimes appliquer :
        - flux (revenue, ebit, net_income, capex, da, income_tax_expense,
          interest_expense) : seuls les faits de 330 à 400 jours sont retenus,
          ce qui écarte les cumuls 6/9 mois et les trimestres isolés ;
        - bilan (total_assets, cash, dettes, current_assets,
          current_liabilities, shares_outstanding) : pas de `start`, donc pas
          de durée à vérifier.

    La priorité entre tags candidats et la règle "dépôt le plus récent gagne"
    (restatements) sont conservées, appliquées APRÈS ce filtre.

    Un tag "namespace:tag" (ex: "dei:EntityCommonStockSharesOutstanding")
    cherche dans cette taxonomie XBRL au lieu de "us-gaap" par défaut --
    nécessaire pour shares_outstanding, que beaucoup d'entreprises ne taguent
    qu'en "dei" (page de garde du 10-K) et jamais en "us-gaap". Ces tags de
    page de garde sont datés du jour où l'émetteur arrête le compte (quelques
    semaines avant le dépôt, donc APRÈS la clôture de l'exercice décrit) :
    `period_ends` sert alors à les recaler sur la dernière fin d'exercice
    réellement observée, sans quoi le passage au rattachement par `end` les
    aurait tous décalés d'un exercice.

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
    by_year: Dict[int, tuple] = {}  # year -> (tag_priority, filed_date, value)

    for priority, tag_spec in enumerate(tags):
        if tag_spec in sec_xbrl.COVER_PAGE_TAGS and period_ends:
            per_year = sec_xbrl.collect_cover_page_entries(facts, tag_spec, period_ends)
        else:
            per_year = sec_xbrl.collect_annual_entries(facts, tag_spec, metric_is_flow)
        for year, entries in per_year.items():
            # Départage INTRA-tag : dépôt le plus récent (restatement gagnant).
            # Il opère enfin, puisque les entrées regroupées ici décrivent
            # toutes la même période -- ce sont bien des republications de la
            # même valeur, pas trois exercices différents.
            best = sec_xbrl.best_entry(entries)
            if best is None:
                continue
            current = by_year.get(year)
            # Un tag mieux classé (priority plus petite) qui a déjà fourni une
            # valeur pour cette année n'est jamais écrasé par un tag de repli.
            if current is None or priority < current[0]:
                by_year[year] = (priority, best["filed"], best["val"])

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


def extract_financials_for_ticker(
    ric: str, cik: str, session: Optional[requests.Session] = None, facts: Optional[dict] = None,
) -> pd.DataFrame:
    """ric : ticker tel qu'utilisé par la SEC pour retrouver le CIK (peut
    contenir un point, ex "BRK.B"). La colonne 'symbol' du DataFrame de
    sortie utilise en revanche config.to_ib_symbol(ric), pour être
    identique au 'symbol' produit par 03/04 (cf. erreur 2.1 du rapport).

    `facts` déjà chargé court-circuite l'appel réseau -- utilisé par les tests
    pour rejouer un companyfacts figé (tests/fixtures/), sans quoi la
    vérification du rattachement des exercices dépendrait de data.sec.gov."""
    if facts is None:
        facts = _get(COMPANYFACTS_URL.format(cik=cik), session=session)
    if not facts:
        logger.warning("Aucune donnée companyfacts pour %s (CIK %s)", ric, cik)
        return pd.DataFrame()

    # Premier passage : les fins d'exercice réellement observées, nécessaires
    # pour recaler les tags de page de garde (cf. extract_annual_values).
    period_ends = sec_xbrl.collect_period_end_dates(facts, XBRL_TAGS)
    per_metric = {
        metric: extract_annual_values(facts, tags, metric in sec_xbrl.FLOW_METRICS, period_ends)
        for metric, tags in XBRL_TAGS.items()
    }
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

    if sec_http.require_contact_email(logger) is None:
        sys.exit(1)

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
