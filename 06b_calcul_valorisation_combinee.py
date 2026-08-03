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

Méthode, pour chaque ligne (symbol, period_type, fiscal_year, fiscal_quarter)
-- period_type="FY" (annuel, 04) ou "TTM" (trimestriel glissant, 04b, si
config.FINANCIALS_TTM_FILE existe ; sinon annuel seul, comportement inchangé) :
    1. Multiples sectoriels médians du même "millésime de publication"
       (période exacte : même period_type + fiscal_year + fiscal_quarter),
       à partir des seules entreprises du même secteur pour cette période
       (ignoré si moins de MIN_PEERS_PER_SECTOR_YEAR pairs disponibles --
       plus fréquent en TTM tant que peu d'entreprises ont un historique
       04b, repli DCF alors plus sollicité, cf. point 4).
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
# En dessous, la médiane sectorielle n'est pas jugée assez robuste. À 3 pairs,
# la "médiane" est la valeur du milieu d'un trio : aucune robustesse à un
# outlier, alors que c'est précisément ce qu'on lui demande.
MIN_PEERS_PER_SECTOR_YEAR = 5

# Délai de dépôt SEC par défaut (~2-3 mois après la clôture d'exercice),
# utilisé UNIQUEMENT en repli quand filed_date est introuvable (cache
# financials.parquet/dcf_historique.parquet antérieur à l'ajout de cette
# colonne à 04_recuperation_10k.py, ou jamais régénéré depuis -- un run
# throttlé de 04 qui ne retélécharge rien laisse l'ancien schéma en place).
DEFAULT_FILING_LAG_DAYS = 75


GROUP_COLUMNS = ["sector", "period_type", "fiscal_year", "fiscal_quarter"]


def _normalize_fiscal_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """Force fiscal_quarter en dtype object (défensif : un aller-retour
    Parquet, ou un DataFrame construit différemment selon la provenance,
    peut typer une colonne 100% None en float64 d'un côté et object de
    l'autre -- merge/groupby échouent alors sur cette clé)."""
    if "fiscal_quarter" in df.columns:
        df = df.copy()
        df["fiscal_quarter"] = df["fiscal_quarter"].astype(object)
    return df


def _ensure_period_columns(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """multiples.parquet (05) porte period_type/fiscal_year/fiscal_quarter
    depuis l'ajout du TTM (04b) -- absent seulement si régénéré par une
    version de 05 antérieure à ce changement. Repli en "tout annuel" (comme
    avant) plutôt que de planter, avec le même style d'avertissement que le
    repli filed_date déjà en place plus bas."""
    missing = [c for c in ("period_type", "fiscal_year", "fiscal_quarter") if c not in df.columns]
    if not missing:
        return df
    logger.warning(
        "%s n'a pas de colonne(s) %s : relance 05_calcul_multiples.py pour bénéficier du TTM "
        "(04b_recuperation_10q.py). Repli sur period_type='FY' uniquement en attendant.",
        source_label, missing,
    )
    df = df.copy()
    df["period_type"] = "FY"
    df["fiscal_year"] = df["year"]
    # dtype "object" explicite -- voir le même commentaire dans
    # 05_calcul_multiples.py::load_financials_with_periods (sinon merge/
    # groupby ultérieur sur fiscal_quarter échoue si l'autre côté a des
    # valeurs "Q1".."Q4", object).
    df["fiscal_quarter"] = pd.Series([None] * len(df), index=df.index, dtype=object)
    return df


def compute_sector_year_multiples(df: pd.DataFrame) -> pd.DataFrame:
    """Médiane de chaque multiple par (secteur, period_type, fiscal_year,
    fiscal_quarter) -- comparaison cross-sectionnelle au sein du même
    "millésime de publication" (une ligne TTM-Q3-2025 comparée aux pairs
    eux-mêmes en TTM-Q3-2025, pas mélangée avec des exercices FY d'une autre
    période). Avec period_type="FY" uniquement (pas de 04b), équivaut à
    l'ancien regroupement par (secteur, année)."""
    df = _normalize_fiscal_quarter(df)
    valid = df.dropna(subset=["sector"])
    rows = []
    for keys, group in valid.groupby(GROUP_COLUMNS, dropna=False):
        row = dict(zip(GROUP_COLUMNS, keys))
        for col in MULTIPLE_COLUMNS:
            vals = group[col].replace([np.inf, -np.inf], np.nan).dropna()
            vals = vals[vals > 0]
            # Valeurs aberrantes écartées AVANT la médiane (cf.
            # config.MULTIPLE_PLAUSIBLE_RANGE) : un P/E à 800x (sortie de
            # perte) déplace la médiane d'un secteur peu peuplé.
            lo, hi = config.MULTIPLE_PLAUSIBLE_RANGE.get(col, (0.0, float("inf")))
            vals = vals[(vals >= lo) & (vals <= hi)]
            row[f"{col}_median"] = vals.median() if len(vals) >= MIN_PEERS_PER_SECTOR_YEAR else None
            row[f"{col}_n_peers"] = len(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_implied_valuations(df: pd.DataFrame, sector_year_multiples: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_fiscal_quarter(df)
    sector_year_multiples = _normalize_fiscal_quarter(sector_year_multiples)
    df = df.merge(sector_year_multiples, on=GROUP_COLUMNS, how="left")

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

    # Chaque multiple n'est retenu que s'il a un sens pour le secteur de la
    # ligne (config.SECTOR_MULTIPLES) : sinon la médiane des trois valeurs
    # implicites mélange, pour une banque, un P/E pertinent et un EV/EBITDA
    # qui ne veut rien dire.
    for col, series in (("EV/EBITDA", price_from_ebitda), ("EV/Sales", price_from_sales), ("P/E", price_from_pe)):
        applicable = df["sector"].map(
            lambda s: col in config.SECTOR_MULTIPLES.get(
                s if isinstance(s, str) else "", config.SECTOR_MULTIPLES["_default"],
            )
        )
        series.where(applicable, inplace=True)

    implied = pd.concat([price_from_ebitda, price_from_sales, price_from_pe], axis=1)
    df["valuation_multiples_per_share"] = implied.median(axis=1, skipna=True)
    df["n_multiples_used"] = implied.notna().sum(axis=1)
    df["price_from_ev_ebitda"] = price_from_ebitda
    df["price_from_ev_sales"] = price_from_sales
    df["price_from_pe"] = price_from_pe
    return df


def build_combined_valuation() -> pd.DataFrame:
    multiples = pd.read_parquet(config.MULTIPLES_FILE)
    if "filed_date" not in multiples.columns:
        # multiples.parquet régénéré par 05 à partir d'un financials.parquet
        # antérieur à l'ajout de filed_date (04_recuperation_10k.py), ou
        # jamais régénéré depuis (un run throttlé de 04 qui ne retélécharge
        # rien laisse l'ancien schéma en place) : colonne absente plutôt que
        # simplement vide. Créée à NaT ici pour ne pas planter -- complétée
        # par repli plus bas, mais régénère financials.parquet
        # (04 --force-refresh, puis 05) pour une date réelle.
        logger.warning(
            "%s n'a pas de colonne filed_date : relance 04_recuperation_10k.py "
            "--force-refresh puis 05_calcul_multiples.py pour une date réelle. "
            "Repli sur une approximation en attendant.", config.MULTIPLES_FILE,
        )
        multiples = multiples.copy()
        multiples["filed_date"] = pd.NaT

    multiples = _ensure_period_columns(multiples, str(config.MULTIPLES_FILE))

    if not config.DCF_HISTORY_FILE.exists():
        raise FileNotFoundError(f"{config.DCF_HISTORY_FILE} introuvable. Lance d'abord 07_calcul_dcf.py.")
    dcf_history = pd.read_parquet(config.DCF_HISTORY_FILE)
    dcf_history = _ensure_period_columns(dcf_history, str(config.DCF_HISTORY_FILE))
    dcf_history = _normalize_fiscal_quarter(dcf_history)

    sector_year_multiples = compute_sector_year_multiples(multiples)
    df = compute_implied_valuations(multiples, sector_year_multiples)
    df = _normalize_fiscal_quarter(df)

    df = df.merge(
        dcf_history[["symbol", *GROUP_COLUMNS[1:], "valuation_dcf_per_share", "filed_date"]].rename(
            columns={"filed_date": "filed_date_dcf"}
        ),
        on=["symbol", *GROUP_COLUMNS[1:]], how="left",
    )
    # filed_date de 05 (financials) en priorité, repli sur celle de 07 (DCF) si
    # absente -- les deux dérivent normalement de la même ligne financials,
    # donc identiques en pratique ; l'une peut manquer si un cache est plus
    # ancien que l'autre.
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
    df["filed_date_dcf"] = pd.to_datetime(df["filed_date_dcf"], errors="coerce")
    df["filed_date"] = df["filed_date"].fillna(df["filed_date_dcf"])
    df = df.drop(columns=["filed_date_dcf"])

    missing_filed = df["filed_date"].isna()
    if missing_filed.any():
        df.loc[missing_filed, "filed_date"] = pd.to_datetime(
            df.loc[missing_filed, "year"].astype(int).map(lambda y: pd.Timestamp(year=y + 1, month=4, day=1))
        )
        logger.warning(
            "%d/%d lignes sans filed_date exploitable : approximé à year+1-04-01 "
            "(délai de dépôt SEC par défaut, ~%d jours après la clôture).",
            missing_filed.sum(), len(df), DEFAULT_FILING_LAG_DAYS,
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
        "symbol", "sector", "period_type", "year", "fiscal_quarter", "filed_date", "close",
        "valuation_multiples_per_share", "valuation_dcf_per_share", "valuation_theoretical_per_share",
        "source", "gap_pct", "n_multiples_used", "price_from_ev_ebitda", "price_from_ev_sales", "price_from_pe",
    ]].sort_values(["symbol", "filed_date"]).reset_index(drop=True)


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
