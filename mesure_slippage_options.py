"""
Mesure le slippage RÉEL à partir des snapshots d'options archivés
(data/options/history/, écrits par 08_recuperation_options.py), pour remplacer
l'hypothèse config.OPTIONS_SLIPPAGE_PCT_OF_PREMIUM (5% par défaut) par une
valeur observée sur ton univers.

Proxy retenu : le DEMI-SPREAD en % du milieu bid/ask. C'est ce que coûte un
aller simple exécuté au marché quand on part du mid -- exactement la
convention du backtest, qui applique la prime x (1 +/- slippage_rate) selon le
sens (voir backtest/options_engine.py::_open_or_resize et _reduce_position).

Deux précautions, qui font l'essentiel de l'écart avec une moyenne naïve :

  1. NE MESURER QUE CE QUE LA STRATÉGIE TRADE RÉELLEMENT. Les snapshots
     couvrent une bande large (OPTIONS_STRIKE_BAND_PCT = +/-30% autour du
     spot) alors que le moteur retient un contrat proche de la monnaie, à
     échéance longue. Les contrats très hors de la monnaie ont un spread
     relatif énorme (prime de quelques centimes) : les inclure dans la moyenne
     surestime largement le slippage applicable. On filtre donc par moneyness
     et par échéance, et on rapporte les deux chiffres pour que l'écart soit
     visible.

  2. VÉRIFIER QUE bid/ask SONT RÉELLEMENT COTÉS. Avec un compte IBKR sans
     abonnement données options (MARKET_DATA_TYPE=4, delayed-frozen), une
     partie des tickers revient sans bid/ask. Une moyenne calculée sur les
     rares lignes cotées ne serait pas représentative : le taux de
     remplissage est donc affiché AVANT toute statistique.

Usage :
    python mesure_slippage_options.py
    python mesure_slippage_options.py --moneyness-pct 5 --min-tenor-days 180
    python mesure_slippage_options.py --by-symbol
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Bande de moneyness considérée comme "ce que la stratégie trade" par défaut.
# valuation_gap_options vise l'ATM ; valuation_gap_multiples_options vise un
# strike à mi-chemin théorique/cours, donc plus loin -- d'où la ventilation
# par tranche de moneyness affichée en plus de l'agrégat.
DEFAULT_MONEYNESS_PCT = 10.0
MONEYNESS_BANDS = [(0, 2.5), (2.5, 5), (5, 10), (10, 20), (20, 100)]


def load_snapshots() -> pd.DataFrame:
    files = sorted(config.DIR_OPTIONS_HISTORY.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"Aucun snapshot dans {config.DIR_OPTIONS_HISTORY}. Lance d'abord "
            "08_recuperation_options.py (TWS/IB Gateway ouvert), ou "
            "08_recuperation_options.py --av-backfill-dates YYYY-MM-DD pour "
            "un historique Alpha Vantage."
        )
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    logger.info("%d contrats chargés depuis %d fichier(s) de snapshot.", len(df), len(files))
    return df


def compute_half_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute mid et half_spread_pct, en n'gardant que les lignes réellement
    cotées des DEUX côtés (un bid ou un ask manquant ne donne pas de spread)."""
    for col in ("bid", "ask"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    quoted = df["bid"].notna() & df["ask"].notna() & (df["bid"] > 0) & (df["ask"] > 0)
    # ask < bid = cotation croisée (donnée figée/incohérente) : écartée plutôt
    # que de produire un spread négatif.
    coherent = quoted & (df["ask"] >= df["bid"])
    logger.info(
        "bid/ask exploitables : %d/%d lignes (%.1f%%)%s",
        int(coherent.sum()), len(df), coherent.mean() * 100,
        "" if coherent.mean() > 0.5 else "  <-- FAIBLE : mesure peu représentative, cf. docstring",
    )
    if not coherent.any():
        raise ValueError(
            "Aucune ligne avec bid ET ask cotés : impossible de mesurer un spread. "
            "Avec un compte sans abonnement données options, relance 08 en séance "
            "(marché US ouvert), ou utilise --av-backfill-dates (Alpha Vantage "
            "renvoie bid/ask sur l'historique)."
        )

    out = df[coherent].copy()
    out["mid"] = (out["bid"] + out["ask"]) / 2
    out["half_spread_pct"] = (out["ask"] - out["bid"]) / (2 * out["mid"]) * 100
    return out


def add_tenor_days(df: pd.DataFrame) -> pd.DataFrame:
    expiry = pd.to_datetime(df["expiry"], format="%Y%m%d", errors="coerce")
    as_of = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df["tenor_days"] = (expiry - as_of).dt.days
    return df


def describe(label: str, values: pd.Series) -> None:
    if values.empty:
        print(f"{label:38} (aucune ligne)")
        return
    print(
        f"{label:38} n={len(values):6d}  médiane={values.median():6.2f}%  "
        f"moyenne={values.mean():6.2f}%  p75={values.quantile(0.75):6.2f}%  "
        f"p90={values.quantile(0.90):6.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--moneyness-pct", type=float, default=DEFAULT_MONEYNESS_PCT,
        help="Bande de moneyness (en %% autour du spot) retenue comme représentative "
             f"de ce que la stratégie trade. Défaut: {DEFAULT_MONEYNESS_PCT}.",
    )
    parser.add_argument(
        "--min-tenor-days", type=int, default=180,
        help="Échéance minimale retenue (le moteur vise %d à %d jours). Défaut: 180."
             % (config.OPTIONS_TARGET_TENOR_DAYS, config.OPTIONS_MULTIPLES_TENOR_DAYS),
    )
    parser.add_argument("--by-symbol", action="store_true", help="Ventile aussi par symbole (les 15 plus/moins liquides).")
    args = parser.parse_args()

    df = add_tenor_days(compute_half_spread(load_snapshots()))

    if "iv_source" in df.columns:
        print("\nSources des snapshots :", df["iv_source"].value_counts(dropna=False).to_dict())
    if "snapshot_date" in df.columns:
        dates = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.date
        print(f"Dates couvertes : {dates.min()} -> {dates.max()} ({dates.nunique()} date(s) distincte(s))")

    print("\n--- Demi-spread, TOUS contrats (mesure naïve : surestime le slippage applicable) ---")
    describe("tous contrats", df["half_spread_pct"])

    print("\n--- Par tranche de moneyness (|strike/spot - 1|) ---")
    moneyness = df["moneyness_pct"].abs()
    for lo, hi in MONEYNESS_BANDS:
        describe(f"  {lo:g}-{hi:g}% du spot", df.loc[(moneyness >= lo) & (moneyness < hi), "half_spread_pct"])

    print("\n--- Par échéance ---")
    for lo, hi, label in [(0, 90, "< 3 mois"), (90, 270, "3-9 mois"), (270, 550, "9-18 mois"), (550, 10_000, "> 18 mois")]:
        describe(f"  {label}", df.loc[(df["tenor_days"] >= lo) & (df["tenor_days"] < hi), "half_spread_pct"])

    if args.by_symbol:
        print("\n--- Par symbole (médiane du demi-spread) ---")
        per_symbol = df.groupby("symbol")["half_spread_pct"].agg(["median", "count"]).sort_values("median")
        print(per_symbol.head(15).to_string())
        print("  ...")
        print(per_symbol.tail(15).to_string())

    # Périmètre effectivement tradé : c'est CE chiffre qui doit alimenter config.
    tradable = df[(moneyness <= args.moneyness_pct) & (df["tenor_days"] >= args.min_tenor_days)]
    print(
        f"\n=== PÉRIMÈTRE TRADÉ (|moneyness| <= {args.moneyness_pct:g}%, "
        f"échéance >= {args.min_tenor_days} j) ==="
    )
    describe("périmètre tradé", tradable["half_spread_pct"])

    if tradable.empty:
        print(
            "\nAucun contrat dans ce périmètre : élargis --moneyness-pct / abaisse "
            "--min-tenor-days, ou collecte plus de snapshots."
        )
        return

    # La médiane plutôt que la moyenne : la distribution des spreads est très
    # asymétrique (quelques contrats illiquides tirent la moyenne vers le haut),
    # et le backtest applique un taux unique à toutes les transactions.
    recommended = float(tradable["half_spread_pct"].median())
    print(
        f"\nValeur suggérée  ->  OPTIONS_SLIPPAGE_PCT_OF_PREMIUM = {recommended:.2f}"
        f"   (actuellement {config.OPTIONS_SLIPPAGE_PCT_OF_PREMIUM})"
    )
    print(
        "Prudence : c'est une médiane sur les snapshots disponibles. Si tous datent "
        "de la même semaine, elle ne dit rien des régimes de stress (mars 2020), où "
        "les spreads s'écartent fortement. Le p90 ci-dessus donne une borne haute "
        "plus conservatrice ; backteste les deux."
    )


if __name__ == "__main__":
    main()
