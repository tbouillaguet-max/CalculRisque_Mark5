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

Usage :
    python 04_recuperation_10k.py                 # tout l'univers US
    python 04_recuperation_10k.py --limit 10       # test rapide
    python 04_recuperation_10k.py --ticker AAPL    # une seule entreprise
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

import config

# ⚠️ À COMPLÉTER : la SEC exige un User-Agent identifiant un contact réel.
SEC_CONTACT_EMAIL = "jeanboubou1er@gmail.com"

HEADERS = {
    "User-Agent": f"OptionsPipeline/1.0 ({SEC_CONTACT_EMAIL})",
    "Accept": "application/json",
}

TICKERS_JSON_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Faible délai : companyfacts fait 1 requête par entreprise (contre N par
# entreprise dans l'original), la SEC tolère ~10 req/s avec un bon User-Agent.
REQUEST_DELAY = 0.15

# Tags US-GAAP candidats par métrique, dans l'ordre de préférence (certaines
# entreprises taguent différemment selon les années/schémas).
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
    "shares_outstanding": ["CommonStockSharesOutstanding"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _get(url: str) -> Optional[dict]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.error("Échec de la requête %s: %s", url, e)
        return None
    finally:
        time.sleep(REQUEST_DELAY)


def get_cik_map() -> Dict[str, str]:
    """Ticker -> CIK (10 chiffres), depuis le fichier officiel SEC."""
    data = _get(TICKERS_JSON_URL)
    if not data:
        return {}
    return {entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in data.values()}


def extract_annual_values(facts: dict, tags: List[str]) -> Dict[int, float]:
    """
    Pour une métrique donnée (plusieurs tags candidats US-GAAP, dans l'ordre
    de préférence), retourne {année_fiscale: valeur} en ne gardant que les
    faits associés à un 10-K annuel (form == "10-K", fp == "FY").

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
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    by_year: Dict[int, tuple] = {}  # year -> (tag_priority, filed_date, value)

    for priority, tag in enumerate(tags):
        entry = us_gaap.get(tag)
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

    return {year: val for year, (_, _, val) in by_year.items()}


def compute_derived(row: pd.Series) -> pd.Series:
    """Métriques dérivées, calculées sur le schéma canonique déjà en place."""
    long_term_debt = row.get("long_term_debt") or 0
    short_term_debt = row.get("short_term_debt") or 0
    cash = row.get("cash") or 0
    net_income = row.get("net_income") or 0
    da = row.get("da") or 0
    capex = row.get("capex") or 0
    income_tax_expense = row.get("income_tax_expense") or 0
    current_assets = row.get("current_assets")
    current_liabilities = row.get("current_liabilities")
    ebit = row.get("ebit")

    row["net_debt"] = long_term_debt + short_term_debt - cash
    # EBIT réellement manquant (None/NaN) -> EBITDA laissé à None plutôt que
    # d'être silencieusement égal à la D&A seule (cf. rapport, erreur 2.3).
    row["ebitda"] = (ebit + da) if pd.notna(ebit) else None
    row["fcf"] = net_income + da - capex
    row["working_capital"] = (
        current_assets - current_liabilities
        if pd.notna(current_assets) and pd.notna(current_liabilities)
        else None
    )
    total_debt = long_term_debt + short_term_debt
    row["cost_of_debt"] = (row.get("interest_expense") or 0) / (total_debt + 1e-6) if total_debt else None
    pretax_income = net_income + income_tax_expense
    row["tax_rate"] = income_tax_expense / pretax_income if pretax_income else None
    return row


def extract_financials_for_ticker(ric: str, cik: str) -> pd.DataFrame:
    """ric : ticker tel qu'utilisé par la SEC pour retrouver le CIK (peut
    contenir un point, ex "BRK.B"). La colonne 'symbol' du DataFrame de
    sortie utilise en revanche config.to_ib_symbol(ric), pour être
    identique au 'symbol' produit par 03/04 (cf. erreur 2.1 du rapport)."""
    facts = _get(COMPANYFACTS_URL.format(cik=cik))
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
        for metric, values in per_metric.items():
            row[metric] = values.get(year)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.apply(compute_derived, axis=1)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", type=Path, default=config.UNIVERSE_FILE)
    parser.add_argument("--ticker", type=str, default=None, help="Une seule entreprise (ex: AAPL)")
    parser.add_argument("--limit", type=int, default=None)
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

    logger.info("Récupération des CIK SEC pour %d tickers...", len(symbols))
    cik_map = get_cik_map()

    all_frames = []
    ok_count, fail_count = 0, 0
    for i, ticker in enumerate(symbols, start=1):
        cik = cik_map.get(ticker.upper())
        if not cik:
            logger.warning("[%d/%d] CIK introuvable pour %s, ignoré.", i, len(symbols), ticker)
            fail_count += 1
            continue

        logger.info("[%d/%d] %s (CIK %s)...", i, len(symbols), ticker, cik)
        df_ticker = extract_financials_for_ticker(ticker, cik)
        if df_ticker.empty:
            fail_count += 1
            continue
        all_frames.append(df_ticker)
        ok_count += 1

    logger.info("Terminé. OK: %d | Échecs: %d", ok_count, fail_count)

    if not all_frames:
        logger.warning("Aucune donnée financière récupérée, pas de fichier de sortie généré.")
        return

    df_all = pd.concat(all_frames, ignore_index=True)
    config.FINANCIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(config.FINANCIALS_FILE, index=False, engine="pyarrow")
    logger.info("Fichier écrit : %s (%d lignes)", config.FINANCIALS_FILE, len(df_all))


if __name__ == "__main__":
    main()
