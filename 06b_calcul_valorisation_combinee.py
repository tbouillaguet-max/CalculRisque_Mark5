"""
Valorisation théorique COMBINÉE (multiples sectoriels en priorité, DCF en
repli), pour CHAQUE exercice historique -- signal utilisé par la stratégie
options (backtest/strategies/valuation_gap_options.py). La stratégie actions
(backtest/strategies/valuation_gap.py) continue elle d'utiliser le DCF seul
(dcf_historique.parquet, 07) et n'est pas affectée par ce script.

Pourquoi pas 06_calcul_multiples_moyens.py : ce script blend TOUT
l'historique en un seul jeu de moyennes/médianes par secteur, ce qui
introduirait un biais de look-ahead si on l'utilisait pour valoriser une
entreprise à une date passée (les multiples sectoriels évoluent dans le
temps -- le P/E moyen de la tech en 2010 n'a rien à voir avec 2021). Ici, les
multiples sectoriels moyens sont recalculés PAR ANNÉE (comparaison
cross-sectionnelle des seuls pairs de cette année-là).

Méthode, pour chaque (symbol, year) :
    1. Multiples sectoriels médians de cette année-là (EV/EBITDA, EV/Sales,
       P/E), à partir des seules entreprises du même secteur cette année
       (ignoré si moins de MIN_PEERS_PER_SECTOR_YEAR pairs disponibles).
    2. Valeur implicite par action selon chacun des 3 multiples, appliqués
       aux fondamentaux propres de l'entreprise (EBITDA, revenue, net_income).
    3. valuation_multiples_per_share = médiane des valeurs implicites
       disponibles (plus robuste qu'une moyenne à un multiple aberrant).
    4. Repli sur valuation_dcf_per_share (07) si aucun multiple sectoriel
       n'est exploitable cette année-là (secteur trop restreint, fondamentaux
       manquants...).

Bénéfice notable du repli multiples-d'abord : une banque ou une foncière
(EBIT non tagué en XBRL, cf. 04) peut avoir une valorisation par les
multiples même quand son DCF est impossible à calculer -- les multiples ne
nécessitent pas l'EBIT.

Usage :
    python 06b_calcul_valorisation_combinee.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MULTIPLE_COLUMNS = ["EV/EBITDA", "EV/Sales", "P/E"]
MIN_PEERS_PER_SECTOR_YEAR = 3  # en dessous, la médiane sectorielle n'est pas jugée assez robuste


def compute_sector_year_multiples(df: pd.DataFrame) -> pd.DataFrame:
    """Médiane de chaque multiple par (secteur, année), cross-sectionnel."""
    valid = df.dropna(subset=["sector"])
    rows = []
    for (sector, year), group in valid.groupby(["sector", "year"]):
        row = {"sector": sector, "year": year}
        for col in MULTIPLE_COLUMNS:
            vals = group[col].replace([np.inf, -np.inf], np.nan).dropna()
            vals = vals[vals > 0]
            row[f"{col}_median"] = vals.median() if len(vals) >= MIN_PEERS_PER_SECTOR_YEAR else None
            row[f"{col}_n_peers"] = len(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_implied_valuations(df: pd.DataFrame, sector_year_multiples: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(sector_year_multiples, on=["sector", "year"], how="left")

    ev_ebitda_med, ev_sales_med, pe_med = df["EV/EBITDA_median"], df["EV/Sales_median"], df["P/E_median"]
    ebitda, revenue, net_income = df["ebitda"], df["revenue"], df["net_income"]
    shares = df["shares_outstanding"]
    # net_debt (04) est DÉJÀ net de cash (dette brute - cash) : contrairement
    # à 07_calcul_dcf.py qui fait "EV - net_debt + cash" (double compte le
    # cash, bug préexistant non corrigé ici pour ne pas modifier le
    # comportement de 07), la conversion EV -> equity correcte est
    # simplement "EV - net_debt".
    net_debt = df["net_debt"]

    def implied_price(implied_ev: pd.Series) -> pd.Series:
        equity = implied_ev - net_debt
        price = equity / shares
        return price.where((shares > 0) & np.isfinite(price))

    price_from_ebitda = implied_price(ev_ebitda_med * ebitda).where(ebitda > 0)
    price_from_sales = implied_price(ev_sales_med * revenue).where(revenue > 0)
    price_from_pe = (pe_med * (net_income / shares)).where((net_income > 0) & (shares > 0))

    implied = pd.concat([price_from_ebitda, price_from_sales, price_from_pe], axis=1)
    df["valuation_multiples_per_share"] = implied.median(axis=1, skipna=True)
    df["n_multiples_used"] = implied.notna().sum(axis=1)
    df["price_from_ev_ebitda"] = price_from_ebitda
    df["price_from_ev_sales"] = price_from_sales
    df["price_from_pe"] = price_from_pe
    return df


def build_combined_valuation() -> pd.DataFrame:
    multiples = pd.read_parquet(config.MULTIPLES_FILE)
    if not config.DCF_HISTORY_FILE.exists():
        raise FileNotFoundError(f"{config.DCF_HISTORY_FILE} introuvable. Lance d'abord 07_calcul_dcf.py.")
    dcf_history = pd.read_parquet(config.DCF_HISTORY_FILE)

    sector_year_multiples = compute_sector_year_multiples(multiples)
    df = compute_implied_valuations(multiples, sector_year_multiples)

    df = df.merge(
        dcf_history[["symbol", "year", "valuation_dcf_per_share"]],
        on=["symbol", "year"], how="left",
    )

    has_multiples = df["valuation_multiples_per_share"].notna()
    df["valuation_theoretical_per_share"] = df["valuation_multiples_per_share"].where(
        has_multiples, df["valuation_dcf_per_share"]
    )
    df["source"] = np.select(
        [has_multiples, df["valuation_dcf_per_share"].notna()],
        ["multiples", "dcf_fallback"],
        default=None,
    )

    df = df.dropna(subset=["valuation_theoretical_per_share", "close"]).copy()
    df["gap_pct"] = (df["valuation_theoretical_per_share"] - df["close"]) / df["close"] * 100

    return df[[
        "symbol", "sector", "year", "filed_date", "close",
        "valuation_multiples_per_share", "valuation_dcf_per_share", "valuation_theoretical_per_share",
        "source", "gap_pct", "n_multiples_used", "price_from_ev_ebitda", "price_from_ev_sales", "price_from_pe",
    ]].sort_values(["symbol", "year"]).reset_index(drop=True)


def main() -> None:
    if not config.MULTIPLES_FILE.exists():
        logger.error("Fichier manquant: %s. Lance d'abord 05_calcul_multiples.py.", config.MULTIPLES_FILE)
        return
    if not config.DCF_HISTORY_FILE.exists():
        logger.error("Fichier manquant: %s. Lance d'abord 07_calcul_dcf.py.", config.DCF_HISTORY_FILE)
        return

    df = build_combined_valuation()
    if df.empty:
        logger.error("Aucune valorisation combinée calculée. Vérifie les données d'entrée.")
        return

    config.VALORISATION_COMBINEE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.VALORISATION_COMBINEE_FILE, index=False, engine="pyarrow")

    n_multiples = (df["source"] == "multiples").sum()
    n_fallback = (df["source"] == "dcf_fallback").sum()
    logger.info(
        "Valorisation combinée sauvegardée : %s (%d lignes, %d entreprises ; "
        "%d via multiples sectoriels, %d en repli DCF).",
        config.VALORISATION_COMBINEE_FILE, len(df), df["symbol"].nunique(), n_multiples, n_fallback,
    )


if __name__ == "__main__":
    main()
