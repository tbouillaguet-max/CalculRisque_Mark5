"""
Calcule les multiples (EV/EBITDA, EV/Sales, P/E) à partir des cours (03/03b)
et des données financières annuelles (04) + trimestrielles TTM (04b, si
disponibles).

Corrections par rapport à CalculMultipleMark2 :
    - df_financials était lu avec pd.read_json(..., lines=True) (format
      JSONL) alors que 05 d'origine sauvegardait un .json par ticker/année
      (pas du JSONL) : le chargement plantait ou renvoyait un DataFrame
      vide. Lit maintenant config.FINANCIALS_FILE (Parquet consolidé, écrit
      par le nouveau 05).
    - calculate_ev utilisait row["shares_outstanding"] (colonne inexistante
      côté cours) alors que le nombre d'actions vient des données
      financières ("SharesOutstanding") : keyError silencieux évité par
      .get, mais le calcul de market cap donnait alors toujours 0. Corrigé :
      market_cap calculé une seule fois, à partir de shares_outstanding côté
      financials.
    - Noms de colonnes harmonisés avec le schéma canonique de config.py
      ("CA" -> "revenue", etc.).
    - NOUVEAU : empile l'annuel (04, period_type="FY") et le TTM trimestriel
      (04b, period_type="TTM") si config.FINANCIALS_TTM_FILE existe -- sinon
      comportement STRICTEMENT identique à avant (annuel seul). Le cours
      utilisé n'est plus systématiquement la clôture de fin d'année : associé
      au cours quotidien connu à filed_date (DAILY_PRICES_FILE, 03b) quand
      disponible, pour que le multiple d'une ligne TTM reflète le cours au
      moment où cette donnée est devenue publique, pas le 31/12. Repli sur la
      clôture annuelle (PRICES_FILE, 03) sinon -- comportement d'avant.

Usage :
    python 05_calcul_multiples.py
"""

from __future__ import annotations

import logging

import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_financials_with_periods() -> pd.DataFrame:
    """Empile l'annuel (04, une ligne par exercice 10-K, period_type="FY")
    et le TTM trimestriel (04b, period_type="TTM") si disponible. Ajoute
    fiscal_year/fiscal_quarter aux lignes annuelles (fiscal_quarter=None)
    pour un schéma commun avec le TTM. Si 04b n'a jamais tourné
    (FINANCIALS_TTM_FILE absent), renvoie l'annuel seul -- comportement
    identique à avant l'ajout du TTM."""
    annual = pd.read_parquet(config.FINANCIALS_FILE).copy()
    annual["period_type"] = "FY"
    # dtype "object" explicite (pas le float64 qu'un simple "= None" produit
    # sur une colonne 100% vide) : sinon la fusion avec les lignes TTM
    # (fiscal_quarter="Q1".."Q4", object) ou un merge/groupby ultérieur sur
    # cette colonne échoue ("merge sur des colonnes object et float64").
    annual["fiscal_quarter"] = pd.Series([None] * len(annual), index=annual.index, dtype=object)
    annual["fiscal_year"] = annual["year"]

    if not config.FINANCIALS_TTM_FILE.exists():
        return annual

    ttm = pd.read_parquet(config.FINANCIALS_TTM_FILE).copy()
    ttm["year"] = ttm["fiscal_year"]
    return pd.concat([annual, ttm], ignore_index=True, sort=False)


def match_price_asof(df: pd.DataFrame) -> pd.DataFrame:
    """Associe à chaque ligne le dernier cours de clôture QUOTIDIEN connu à
    filed_date (jamais un cours futur -- même principe point-in-time que le
    reste du pipeline), via pandas.merge_asof sur DAILY_PRICES_FILE (03b).
    Repli sur le cours de clôture ANNUEL (PRICES_FILE, 03, jointure sur
    year) pour les lignes sans correspondance quotidienne (03b jamais lancé,
    ou trou de couverture au-delà de DAILY_PRICE_ASOF_TOLERANCE_DAYS) --
    comportement d'avant pour ces lignes-là."""
    df = df.dropna(subset=["filed_date"]).copy()
    # config.to_naive_day des deux côtés (et pas un simple pd.to_datetime) :
    # merge_asof refuse deux clés de résolutions différentes, or filed_date
    # (chaîne SEC -> datetime64[us]) et la date des cours quotidiens (date32
    # dans le Parquet de 03b -> datetime64[s]) n'ont pas la même.
    df["filed_date"] = config.to_naive_day(df["filed_date"])
    df = df.dropna(subset=["filed_date"])

    if config.DAILY_PRICES_FILE.exists():
        daily = pd.read_parquet(config.DAILY_PRICES_FILE)[["symbol", "date", "close"]].copy()
        daily["date"] = config.to_naive_day(daily["date"])
        # Tri sur la SEULE clé temporelle des deux côtés : merge_asof exige que
        # 'on' soit globalement croissant, y compris avec by="symbol" (un tri
        # ["symbol", "date"] rend la colonne de dates non monotone d'un symbole
        # au suivant -> ValueError: right keys must be sorted).
        daily = daily.dropna(subset=["date", "close"]).sort_values("date")

        df = df.sort_values("filed_date")
        df = pd.merge_asof(
            df, daily.rename(columns={"date": "price_date"}),
            left_on="filed_date", right_on="price_date", by="symbol", direction="backward",
            tolerance=pd.Timedelta(days=config.DAILY_PRICE_ASOF_TOLERANCE_DAYS),
        ).drop(columns=["price_date"])
    else:
        df["close"] = None

    missing = df["close"].isna()
    if missing.any() and config.PRICES_FILE.exists():
        annual_prices = pd.read_parquet(config.PRICES_FILE)[["symbol", "year", "close"]]
        fallback = df.loc[missing, ["symbol", "year"]].merge(annual_prices, on=["symbol", "year"], how="left")
        df.loc[missing, "close"] = fallback["close"].values

    return df


def calculate_multiples() -> pd.DataFrame:
    df = load_financials_with_periods()
    df = match_price_asof(df)

    if df.empty:
        logger.warning(
            "Aucune ligne après association des cours : vérifie que les "
            "dates couvertes se recoupent entre %s (et éventuellement %s) et %s/%s.",
            config.FINANCIALS_FILE, config.FINANCIALS_TTM_FILE, config.DAILY_PRICES_FILE, config.PRICES_FILE,
        )
        return df

    # 👇 Supprime les lignes où shares_outstanding ou le cours est manquant
    df = df.dropna(subset=["shares_outstanding", "close"])

    # Calcul de la market cap (close * shares_outstanding)
    df["market_cap"] = df["close"] * df["shares_outstanding"]
    df["EV"] = df["market_cap"] + df["net_debt"].fillna(0)

    # Calcul des multiples (évite division par zéro)
    df["EV/Sales"] = df["EV"] / df["revenue"].replace({0: pd.NA})
    df["EV/EBITDA"] = df["EV"] / df["ebitda"].replace({0: pd.NA})
    df["P/E"] = df["close"] / (df["net_income"] / df["shares_outstanding"]).replace({0: pd.NA})

    return df


def main() -> None:
    if not (config.FINANCIALS_FILE.exists() and config.PRICES_FILE.exists()):
        logger.error("Fichiers manquants. Lance d'abord 03_recuperation_cours.py et 04_recuperation_10k.py.")
        return

    df = calculate_multiples()
    if df.empty:
        return

    # Rattache le secteur depuis l'univers (ajouté par 02) pour permettre le
    # regroupement par secteur dans 06. On convertit le RIC au même format
    # de symbole que le reste du pipeline (config.to_ib_symbol) : sinon les
    # tickers à classes d'actions (BRK.B, BF.B) se retrouvent sans secteur.
    universe = pd.read_csv(config.UNIVERSE_FILE, encoding="utf-8-sig")
    if "sector" in universe.columns:
        universe = universe.copy()
        universe["symbol"] = universe["RIC"].apply(config.to_ib_symbol)
        df = df.merge(universe[["symbol", "sector"]], on="symbol", how="left")

    config.MULTIPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.MULTIPLES_FILE, index=False)
    logger.info("Multiples sauvegardés : %s (%d lignes)", config.MULTIPLES_FILE, len(df))


if __name__ == "__main__":
    main()
