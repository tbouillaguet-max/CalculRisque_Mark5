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

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Au-delà de cette part de lignes tombées sur le repli annuel, le signal est
# majoritairement calculé contre des cours vieux de plusieurs mois : on le
# signale bruyamment plutôt que de le laisser passer pour un run normal.
FALLBACK_WARNING_RATIO = 0.20


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

    Repli sur les clôtures ANNUELLES (PRICES_FILE, 03) pour les lignes sans
    correspondance quotidienne (03b jamais lancé, ou trou de couverture
    au-delà de DAILY_PRICE_ASOF_TOLERANCE_DAYS).

    CORRIGÉ : ce repli joignait sur `year`, donc récupérait la clôture du
    31/12 de l'exercice décrit -- alors que filed_date tombe deux à trois mois
    PLUS TARD. L'écart de valorisation était donc calculé contre un cours que
    le marché n'avait pas encore formé au moment du dépôt, pendant que le
    backtest, lui, entre au cours du jour. Le repli prend maintenant la
    clôture annuelle la plus proche ANTÉRIEURE OU ÉGALE à filed_date : pour un
    10-K déposé en mars, c'est le 31/12 de l'année PRÉCÉDENTE -- un cours plus
    ancien, mais point-in-time correct, ce qui est la seule propriété qui
    compte ici.

    La proportion de lignes passées par ce repli est journalisée, et
    au-delà de FALLBACK_WARNING_RATIO un avertissement invite à lancer 03b :
    un repli massif signifie que la quasi-totalité du signal est calculée
    contre des cours vieux de plusieurs mois."""
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

    df["price_source"] = np.where(df["close"].notna(), "quotidien", None)

    missing = df["close"].isna()
    if missing.any() and config.PRICES_FILE.exists():
        annual = _annual_prices_asof()
        if annual is not None:
            filled = pd.merge_asof(
                df.loc[missing].sort_values("filed_date"),
                annual, left_on="filed_date", right_on="price_date", by="symbol", direction="backward",
            )
            # merge_asof réordonne : on réaligne sur l'index d'origine plutôt
            # que de supposer que l'ordre a été conservé.
            filled = filled.set_index(df.loc[missing].sort_values("filed_date").index)
            df.loc[filled.index, "close"] = filled["close_annual"]
            df.loc[filled.index[filled["close_annual"].notna()], "price_source"] = "annuel_anterieur"

    _log_price_source_summary(df)
    return df


def _annual_prices_asof() -> "pd.DataFrame | None":
    """Clôtures annuelles (03) indexées par leur DATE de cotation réelle, pour
    un merge_asof point-in-time.

    PRICES_FILE porte une colonne `date` (dernier jour coté de l'exercice)
    depuis 03_recuperation_cours.py. Un cache plus ancien qui ne l'aurait pas
    est reconstitué au 31/12 de l'année -- approximation, mais qui reste du
    bon côté : elle ne peut que rendre le cours retenu plus ancien, jamais
    futur."""
    annual = pd.read_parquet(config.PRICES_FILE)
    if "close" not in annual.columns or "symbol" not in annual.columns:
        logger.warning("%s n'a ni 'symbol' ni 'close' : repli annuel désactivé.", config.PRICES_FILE)
        return None

    if "date" in annual.columns:
        annual["price_date"] = config.to_naive_day(annual["date"])
    elif "year" in annual.columns:
        logger.warning(
            "%s n'a pas de colonne 'date' (cache antérieur) : les clôtures annuelles sont "
            "datées au 31/12 par défaut. Relance 03_recuperation_cours.py pour une date réelle.",
            config.PRICES_FILE,
        )
        annual["price_date"] = pd.to_datetime(
            annual["year"].astype("Int64").map(lambda y: f"{y}-12-31" if pd.notna(y) else None),
            errors="coerce",
        )
    else:
        return None

    annual = (
        annual.dropna(subset=["price_date", "close", "symbol"])
        .rename(columns={"close": "close_annual"})[["symbol", "price_date", "close_annual"]]
        .sort_values("price_date")
    )
    return annual if not annual.empty else None


def _log_price_source_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return
    counts = df["price_source"].value_counts(dropna=False)
    total = len(df)
    fallback = int(counts.get("annuel_anterieur", 0))
    logger.info(
        "Association des cours : %d lignes au total -- %d au cours quotidien (03b), "
        "%d en repli sur la clôture annuelle antérieure (03), %d sans cours.",
        total, int(counts.get("quotidien", 0)), fallback, int(df["close"].isna().sum()),
    )
    if total and fallback / total > FALLBACK_WARNING_RATIO:
        logger.warning(
            "%.0f%% des lignes utilisent le repli annuel : leur écart de valorisation est "
            "calculé contre une clôture vieille de plusieurs mois, alors que le backtest entre "
            "au cours du jour. Lance 03b_recuperation_cours_quotidiens.py pour des cours "
            "quotidiens point-in-time.",
            fallback / total * 100,
        )


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


def _sectors_from(path) -> pd.DataFrame:
    """{symbol -> sector} lu depuis un fichier d'univers (sortie de 02). Vide
    si le fichier n'existe pas ou n'a jamais été catégorisé. Le RIC est
    converti au format de symbole du reste du pipeline (config.to_ib_symbol) :
    sinon les tickers à classes d'actions (BRK.B, BF.B) se retrouvent sans
    secteur."""
    vide = pd.DataFrame(columns=["symbol", "sector"])
    if path is None or not path.exists():
        return vide
    try:
        universe = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, pd.errors.ParserError) as exc:
        logger.warning("%s illisible (%s) : secteurs non repris de ce fichier.", path, exc)
        return vide
    if "sector" not in universe.columns or "RIC" not in universe.columns:
        return vide
    universe = universe.copy()
    universe["symbol"] = universe["RIC"].apply(config.to_ib_symbol)
    return universe[["symbol", "sector"]].dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])


def merge_sectors(df: pd.DataFrame) -> pd.DataFrame:
    """Rattache le secteur : univers COURANT d'abord (01+02), univers COMPLET
    (01b+02) en complément pour les entreprises RADIÉES.

    Sans ce second niveau, une entreprise radiée backfillée dans financials.
    parquet arrive ici sans secteur, et 06b l'écarte de ses médianes
    (dropna(subset=["sector"])). Le backfill aurait alors coûté des milliers
    de requêtes SEC pour rien : le biais de survivance serait resté entier,
    et silencieusement."""
    sectors = _sectors_from(config.UNIVERSE_FILE)
    if sectors.empty:
        return df

    df = df.merge(sectors, on="symbol", how="left")

    complet = _sectors_from(getattr(config, "UNIVERSE_FULL_FILE", None))
    if not complet.empty:
        manquants = df["sector"].isna()
        if manquants.any():
            repli = df.loc[manquants, "symbol"].map(complet.set_index("symbol")["sector"])
            df.loc[manquants, "sector"] = repli
            logger.info(
                "%d lignes sans secteur dans l'univers courant : %d complétées depuis %s "
                "(entreprises radiées).",
                int(manquants.sum()), int(repli.notna().sum()), config.UNIVERSE_FULL_FILE,
            )

    orphelines = df["sector"].isna()
    if orphelines.any():
        symboles = sorted(df.loc[orphelines, "symbol"].dropna().unique())
        logger.warning(
            "%d lignes (%d entreprises) restent SANS SECTEUR : elles seront exclues des "
            "médianes sectorielles de 06b, ce qui rouvre le biais de survivance pour ces "
            "titres. Lance 02_categoriser_secteurs.py --universe %s. Concernées (extrait) : %s%s",
            int(orphelines.sum()), len(symboles), getattr(config, "UNIVERSE_FULL_FILE", "?"),
            ", ".join(symboles[:15]), "..." if len(symboles) > 15 else "",
        )
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
    #
    # Ce secteur est la classification d'AUJOURD'HUI. Elle n'est plus
    # appliquée rétroactivement en aval : 06b la ramène au secteur d'époque
    # avant de composer les groupes de pairs (sector_history.sector_asof,
    # qui rejoue à l'envers les remaniements GICS -- notamment la création de
    # "Communication Services" en 2018, qui a sorti l'internet grand public de
    # la technologie). C'est donc bien le secteur COURANT qu'on veut ici.
    df = merge_sectors(df)

    config.MULTIPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.MULTIPLES_FILE, index=False)
    logger.info("Multiples sauvegardés : %s (%d lignes)", config.MULTIPLES_FILE, len(df))


if __name__ == "__main__":
    main()
