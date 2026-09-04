"""
Le multiple MÉRITÉ prédit-il mieux que le multiple SECTORIEL ?

C'est la question qui décide s'il faut brancher warranted_multiple.py dans
06b_calcul_valorisation_combinee.py, et elle se tranche SANS backtest. Si le
multiple mérité ne prédit pas mieux le multiple réellement observé, il n'y a
aucune raison d'espérer qu'il produise un meilleur signal de valorisation, et
le brancher serait une complication gratuite. C'est le test de Bhojraj & Lee
eux-mêmes (Journal of Accounting Research, 2002), et il coûte quelques minutes
là où un A/B de backtest coûte des heures.

CE QUI EST COMPARÉ
------------------
Pour chaque ligne, deux prédicteurs de son multiple observé, construits sur les
MÊMES pairs point-in-time :

    sectoriel : l'agrégat du groupe de pairs (06b, moyenne harmonique ou
                médiane selon config.SECTOR_MULTIPLE_AGGREGATOR)
    mérité    : exp(a + b·x) d'une régression en coupe sur les fondamentaux
                (warranted_multiple.py)

HORS ÉCHANTILLON PAR CONSTRUCTION. La ligne évaluée est exclue des pairs qui
servent à ajuster -- exactement comme 06b l'exclut déjà de sa propre médiane.
Sans cette exclusion, la régression gagnerait par construction : elle a plus de
degrés de liberté, donc un meilleur ajustement DANS l'échantillon, ce qui ne
dit rien de sa valeur prédictive.

LA MESURE
---------
Erreur absolue en LOG, |ln(observé) - ln(prédit)|, résumée par sa MÉDIANE. En
log parce qu'un multiple est un ratio ; par la médiane parce que la
distribution des erreurs de valorisation a des queues épaisses -- c'est la
convention de Liu, Nissim & Thomas (JAR, 2002), qui rapportent l'erreur
absolue médiane en pourcentage.

Le script rapporte aussi la part de lignes où le mérité fait mieux, qui dit si
un gain médian vient d'une amélioration générale ou de quelques cas extrêmes.

LE REGROUPEMENT EST LA VRAIE DÉCISION
--------------------------------------
Une régression à cinq régresseurs demande des dizaines d'observations ; un
secteur x millésime en compte rarement plus de vingt. D'où deux modes :

    --grouping millesime  (défaut) ajuste sur la coupe COMPLÈTE de l'indice au
                          même millésime, avec indicatrices sectorielles : les
                          indicatrices récupèrent le niveau sectoriel, la pente
                          des fondamentaux obtient ses centaines d'observations.
    --grouping secteur    ajuste secteur par secteur, sans indicatrices. Plus
                          fidèle à l'esprit du multiple sectoriel, mais souvent
                          trop peu peuplé -- le script journalise alors combien
                          de groupes ont dû être abandonnés.

Le comparatif ne porte QUE sur les lignes où les deux prédicteurs existent :
comparer une méthode sur les cas faciles à l'autre sur tous les cas ne mesure
rien.

Usage :
    python 15_test_multiple_merite.py
    python 15_test_multiple_merite.py --multiple "P/E" --grouping secteur
    python 15_test_multiple_merite.py --min-peers 30 --output-csv resultats.csv
"""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
import sector_history
import warranted_multiple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_multiple_merite")

m06b = importlib.import_module("06b_calcul_valorisation_combinee")

GROUP_COLUMNS = m06b.GROUP_COLUMNS
MILLESIME_COLUMNS = ["period_type", "fiscal_year", "fiscal_quarter"]


def _filed_dates(df: pd.DataFrame) -> pd.Series:
    """Dates de dépôt, en repli sur le délai SEC usuel quand la colonne manque
    -- même convention que 06b, dont ce script doit reproduire exactement le
    groupe de pairs visible."""
    if "filed_date" in df.columns:
        dates = pd.to_datetime(df["filed_date"], errors="coerce")
        if dates.notna().any():
            return dates
    fin = pd.to_datetime(df.get("fiscal_year", pd.Series(index=df.index)), format="%Y", errors="coerce")
    return fin + pd.Timedelta(days=m06b.DEFAULT_FILING_LAG_DAYS)


def comparer(
    df: pd.DataFrame,
    multiple_col: str,
    grouping: str,
    min_peers: int,
    membership: Optional[sector_history.MembershipIndex] = None,
) -> pd.DataFrame:
    """Une ligne par observation évaluée : multiple observé, les deux
    prédictions, et leurs erreurs absolues en log."""
    df = m06b._normalize_fiscal_quarter(df).reset_index(drop=True)
    df = df.assign(_filed=_filed_dates(df))
    features = warranted_multiple.build_features(df)
    # Les régresseurs sont TOUJOURS recalculés depuis les fondamentaux, jamais
    # repris d'une colonne homonyme déjà présente : un concat laisserait deux
    # colonnes du même nom, et toute sélection ultérieure rendrait un DataFrame
    # à deux colonnes là où le code attend une Series.
    df = pd.concat([df.drop(columns=[c for c in features.columns if c in df.columns]),
                    features], axis=1)

    cles = MILLESIME_COLUMNS if grouping == "millesime" else GROUP_COLUMNS
    secteur_pour_regression = "sector" if grouping == "millesime" else None

    lignes: list[dict] = []
    groupes_abandonnes = 0
    for _, groupe in df.groupby(cles, dropna=False):
        groupe = groupe.sort_values("_filed", kind="mergesort")
        multiples_propres = m06b._clean_multiple(groupe[multiple_col], multiple_col)
        if multiples_propres.empty:
            continue

        for position, (idx, ligne) in enumerate(groupe.iterrows()):
            observe = groupe.at[idx, multiple_col]
            if idx not in multiples_propres.index or not np.isfinite(observe):
                continue

            # PAIRS VISIBLES : déposés avant (ou en même temps), membres de
            # l'indice à cette date, et jamais la ligne elle-même.
            anterieurs = groupe.iloc[: position + 1].index
            pairs = groupe.loc[groupe.index.isin(anterieurs)].drop(index=idx, errors="ignore")
            if membership is not None and pd.notna(ligne["_filed"]):
                pairs = pairs[[
                    membership.is_member(s, ligne["_filed"]) for s in pairs["symbol"]
                ]]
            pairs_propres = pairs.loc[pairs.index.isin(multiples_propres.index)]
            if len(pairs_propres) < min_peers:
                continue

            sectoriel = m06b.aggregate_multiple(pairs_propres[multiple_col])
            merite_serie = warranted_multiple.fit_predict(
                pairs_propres, groupe.loc[[idx]], multiple_col,
                sector_column=secteur_pour_regression,
            )
            if merite_serie is None or merite_serie.empty:
                groupes_abandonnes += 1
                continue
            merite = float(merite_serie.iloc[0])
            if not (np.isfinite(merite) and merite > 0) or not sectoriel:
                continue

            lignes.append({
                "symbol": ligne.get("symbol"),
                "sector": ligne.get("sector"),
                "fiscal_year": ligne.get("fiscal_year"),
                "period_type": ligne.get("period_type"),
                "n_peers": len(pairs_propres),
                "observe": float(observe),
                "sectoriel": float(sectoriel),
                "merite": merite,
                "err_sectoriel": abs(np.log(observe / sectoriel)),
                "err_merite": abs(np.log(observe / merite)),
            })

    if groupes_abandonnes:
        logger.info(
            "%d ligne(s) écartée(s) faute d'un échantillon suffisant pour la régression "
            "(cf. warranted_multiple.MIN_OBS_PER_FEATURE).", groupes_abandonnes)
    return pd.DataFrame(lignes)


def rapporter(resultats: pd.DataFrame, multiple_col: str) -> None:
    if resultats.empty:
        logger.error(
            "Aucune observation comparable. Causes usuelles : --min-peers trop élevé, "
            "regroupement trop fin (essaie --grouping millesime), ou fondamentaux "
            "manquants pour les régresseurs.")
        return

    err_s = resultats["err_sectoriel"]
    err_m = resultats["err_merite"]
    part_gagnante = float((err_m < err_s).mean() * 100)
    gain = float((err_s.median() - err_m.median()) / err_s.median() * 100)

    print(f"\n=== Multiple mérité vs multiple sectoriel -- {multiple_col} ===")
    print(f"observations comparées : {len(resultats):,}")
    print(f"pairs par régression   : médiane {resultats['n_peers'].median():.0f} "
          f"(min {resultats['n_peers'].min():.0f}, max {resultats['n_peers'].max():.0f})")
    print()
    print(f"{'erreur absolue en log':<28} {'sectoriel':>12} {'mérité':>12}")
    for label, quantile in (("médiane", 0.5), ("moyenne", None), ("3e quartile", 0.75)):
        s = err_s.mean() if quantile is None else err_s.quantile(quantile)
        m = err_m.mean() if quantile is None else err_m.quantile(quantile)
        print(f"{label:<28} {s:>12.4f} {m:>12.4f}")
    print()
    print(f"lignes où le mérité fait mieux : {part_gagnante:.1f}%")
    print(f"gain sur l'erreur médiane      : {gain:+.1f}%")

    print()
    if gain > 5 and part_gagnante > 55:
        print(">>> Le multiple mérité prédit MIEUX. Le brancher dans 06b se justifie ;")
        print("    l'étape suivante est l'A/B du signal sur le backtest corrigé.")
    elif gain < 0:
        print(">>> Le multiple mérité prédit MOINS BIEN que la médiane sectorielle.")
        print("    Ne pas le brancher. La régression absorbe dans sa valeur ajustée")
        print("    ce qu'on cherchait à isoler, ou l'échantillon est trop court.")
    else:
        print(">>> Gain marginal, non concluant. Le brancher ajouterait de la complexité")
        print("    pour un bénéfice que ce test ne démontre pas. Regarde d'abord si un")
        print("    autre regroupement (--grouping) ou un autre multiple change le verdict.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--multiple", default="EV/EBITDA", choices=m06b.MULTIPLE_COLUMNS,
        help="Multiple dont on compare les deux prédicteurs (défaut: %(default)s).")
    parser.add_argument(
        "--grouping", default="millesime", choices=("millesime", "secteur"),
        help="Population d'ajustement de la régression (défaut: %(default)s -- "
             "voir la docstring du module).")
    parser.add_argument(
        "--min-peers", type=int, default=m06b.MIN_PEERS_PER_SECTOR_YEAR,
        help="Pairs minimaux pour évaluer une ligne (défaut: %(default)s).")
    parser.add_argument(
        "--no-point-in-time-peers", action="store_true",
        help="Désactive le filtre d'appartenance à l'indice (diagnostic seulement).")
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    if not config.MULTIPLES_FILE.exists():
        logger.error("%s introuvable -- lance d'abord 05_calcul_multiples.py.",
                     config.MULTIPLES_FILE)
        raise SystemExit(1)

    df = pd.read_parquet(config.MULTIPLES_FILE)
    df = m06b._ensure_period_columns(df, str(config.MULTIPLES_FILE))
    df = df.dropna(subset=["sector"])
    # Tri par (entreprise, date de dépôt) : la croissance du chiffre d'affaires
    # est une variation d'une période à la précédente, elle n'a de sens que sur
    # un historique ordonné.
    df = df.sort_values(["symbol", "filed_date"] if "filed_date" in df.columns else ["symbol"])

    membership = None
    if not args.no_point_in_time_peers:
        historique = sector_history.load_universe_history()
        membership = sector_history.MembershipIndex(historique) if historique is not None else None
        if membership is None:
            logger.warning(
                "Pas d'historique d'univers (01b) : les pairs ne sont pas restreints aux "
                "membres de l'indice d'alors. Le comparatif reste valide -- les DEUX "
                "prédicteurs voient les mêmes pairs -- mais il ne reproduit pas 06b.")

    logger.info("Comparaison sur %s, regroupement '%s'...", args.multiple, args.grouping)
    resultats = comparer(df, args.multiple, args.grouping, args.min_peers, membership)
    rapporter(resultats, args.multiple)

    if args.output_csv is not None and not resultats.empty:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        resultats.to_csv(args.output_csv, index=False)
        logger.info("Détail par observation : %s", args.output_csv)


if __name__ == "__main__":
    main()
