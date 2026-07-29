"""
Calcule les multiples (EV/EBITDA, EV/Sales, P/E) à partir des cours (03) et
des données financières (04).

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

Usage :
    python 05_calcul_multiples.py
"""

from __future__ import annotations

import logging

import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_multiples() -> pd.DataFrame:
    df_financials = pd.read_parquet(config.FINANCIALS_FILE)
    df_prices = pd.read_parquet(config.PRICES_FILE)

    # Fusion sur symbol + year
    df = pd.merge(df_financials, df_prices[["symbol", "year", "close"]], on=["symbol", "year"], how="inner")

    if df.empty:
        logger.warning(
            "Aucune ligne après fusion financials/prices : vérifie que les "
            "années couvertes se recoupent entre %s et %s.",
            config.FINANCIALS_FILE, config.PRICES_FILE,
        )
        return df

    # 👇 Supprime les lignes où shares_outstanding est manquant
    df = df.dropna(subset=["shares_outstanding"])

    # Calcul de la market cap (close * shares_outstanding)
    df["market_cap"] = df["close"] * df["shares_outstanding"]
    df["EV"] = df["market_cap"] + df["net_debt"].fillna(0)

    # Calcul des multiples (évite division par zéro)
    df["EV/Sales"] = df["EV"] / df["revenue"].replace({0: pd.NA})
    df["EV/EBITDA"] = df["EV"] / df["ebitda"].replace({0: pd.NA})
    df["P/E"] = df["close"] / (df["net_income"] / df["shares_outstanding"]).replace({0: pd.NA})

    return df


def main() -> None:
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
