"""
Calcule la valorisation DCF (Discounted Cash Flow) pour chaque entreprise à
partir des données financières (05) et des cours (03), et compare la valeur
intrinsèque par action au cours actuel.

Corrections par rapport à CalculDCFMark1 :
    - Le script lisait un fichier fixe "donnees_entreprises.xlsx" que
      personne dans le pipeline ne produisait. Construit maintenant lui-même
      son entrée en fusionnant 05 (financials) + 03 (cours) + 02 (secteur).
    - Colonnes attendues en entrée ("EBIT"/"CapEx"/"BFR"/"Dette net"/
      "Cash et Cash Equivalents"/"Nombre d'action"/"Taux d'imposition")
      étaient un mélange incohérent avec les noms produits par les autres
      scripts d'origine (ex: "SharesOutstanding" vs "Nombre d'action") :
      aucune valeur ne matchait jamais, tout tombait sur les valeurs par
      défaut. Utilise maintenant les colonnes canoniques
      (ebit, capex, working_capital, net_debt, cash, shares_outstanding,
      tax_rate) produites par 05.
    - On utilise l'exercice financier le plus récent disponible par
      entreprise, mis en correspondance avec le cours de la même année (pas
      systématiquement la dernière année de cours si les 10-K ont du retard).

Usage :
    python 08_calcul_dcf.py
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HYPOTHESES_DEFAUT = {
    "periode_prevision": 5,
    "taux_croissance_fcf": 0.05,
    "taux_croissance_terminal": 0.02,
    "taux_actualisation": 0.10,
    "taux_imposition_defaut": 0.21,  # taux fédéral US ; utilisé seulement si tax_rate manquant
}


def calculer_fcf(ebit: float, taux_imposition: float, capex: float, variation_bfr: float, interets: float = 0) -> float:
    """FCF = EBIT * (1 - t) - CapEx - ΔBFR - Intérêts * (1 - t)."""
    return (ebit * (1 - taux_imposition)) - capex - variation_bfr - (interets * (1 - taux_imposition))


def calculer_fcf_futurs(fcf_actuel: float, taux_croissance: float, periode: int) -> np.ndarray:
    return fcf_actuel * (1 + taux_croissance) ** np.arange(1, periode + 1)


def calculer_terminal_value(fcf_final: float, taux_croissance_terminal: float, taux_actualisation: float) -> float:
    if taux_actualisation <= taux_croissance_terminal:
        raise ValueError("Le taux d'actualisation doit être > au taux de croissance terminal.")
    return fcf_final * (1 + taux_croissance_terminal) / (taux_actualisation - taux_croissance_terminal)


def calculer_dcf(
    fcf_actuel: float, taux_croissance_fcf: float, taux_croissance_terminal: float,
    taux_actualisation: float, periode_prevision: int, dette_nette: float = 0, cash: float = 0,
) -> Tuple[float, Dict]:
    fcf_futurs = calculer_fcf_futurs(fcf_actuel, taux_croissance_fcf, periode_prevision)
    tv = calculer_terminal_value(fcf_futurs[-1], taux_croissance_terminal, taux_actualisation)
    valeur_actualisee_fcf = np.sum(fcf_futurs / (1 + taux_actualisation) ** np.arange(1, periode_prevision + 1))
    valeur_actualisee_tv = tv / (1 + taux_actualisation) ** periode_prevision
    ev = valeur_actualisee_fcf + valeur_actualisee_tv
    equity_value = ev - dette_nette + cash

    details = {
        "FCF_futurs": fcf_futurs.tolist(), "Terminal_Value": tv,
        "Valeur_actualisee_FCF": valeur_actualisee_fcf, "Valeur_actualisee_TV": valeur_actualisee_tv,
        "Enterprise_Value": ev, "Equity_Value": equity_value,
    }
    return equity_value, details


def build_input_table() -> pd.DataFrame:
    """Fusionne financials (05) + cours (03) + secteur (02) sur (symbol, year),
    calcule la variation du BFR (BFR de l'exercice N moins BFR de
    l'exercice N-1, à partir de l'historique complet de financials.parquet),
    puis ne garde que le dernier exercice disponible par entreprise."""
    financials = pd.read_parquet(config.FINANCIALS_FILE)
    prices = pd.read_parquet(config.PRICES_FILE)
    universe = pd.read_csv(config.UNIVERSE_FILE, encoding="utf-8-sig")

    # 'working_capital' (produit par 05) est un NIVEAU (current_assets -
    # current_liabilities) pour une année donnée, pas une variation. On
    # calcule ici la vraie variation année sur année (ΔBFR), en se basant
    # sur l'historique multi-exercices déjà présent dans financials.parquet
    # -- AVANT de filtrer sur le dernier exercice uniquement, sinon
    # l'exercice précédent nécessaire au calcul du delta serait déjà perdu.
    financials = financials.sort_values(["symbol", "year"])
    financials["working_capital_change"] = financials.groupby("symbol")["working_capital"].diff()

    df = pd.merge(financials, prices[["symbol", "year", "close"]], on=["symbol", "year"], how="inner")
    if "sector" in universe.columns:
        universe = universe.copy()
        universe["symbol"] = universe["RIC"].apply(config.to_ib_symbol)
        df = df.merge(universe[["symbol", "sector"]], on="symbol", how="left")

    df = df.sort_values(["symbol", "year"]).groupby("symbol", as_index=False).tail(1)
    return df


def calculer_dcf_par_entreprise(df: pd.DataFrame, hypotheses: Dict = HYPOTHESES_DEFAUT) -> pd.DataFrame:
    results = []
    for _, row in df.iterrows():
        symbol = row.get("symbol", "?")
        try:
            ebit = row.get("ebit")
            if pd.isna(ebit) or ebit is None or ebit <= 0:
                logger.warning("EBIT manquant/invalide pour %s. DCF non calculé.", symbol)
                continue

            capex = row.get("capex") or 0
            # ΔBFR réel (variation N vs N-1), pas le niveau du BFR (cf.
            # erreur 1.3 du rapport). Si l'historique ne contient qu'un seul
            # exercice pour cette entreprise, la variation est inconnue :
            # on la traite comme 0 tout en le signalant, plutôt que de
            # soustraire par erreur le niveau absolu du BFR au FCF.
            bfr = row.get("working_capital_change")
            if pd.isna(bfr):
                logger.warning(
                    "Variation de BFR indisponible pour %s (un seul exercice "
                    "dans l'historique) : traitée comme 0 dans le calcul du FCF.",
                    symbol,
                )
                bfr = 0
            dette_nette = row.get("net_debt") or 0
            cash = row.get("cash") or 0
            shares_outstanding = row.get("shares_outstanding")
            taux_imposition = row.get("tax_rate")
            if pd.isna(taux_imposition) or taux_imposition is None or not (0 <= taux_imposition < 1):
                taux_imposition = hypotheses["taux_imposition_defaut"]

            if pd.isna(shares_outstanding) or not shares_outstanding or shares_outstanding <= 0:
                logger.warning("Nombre d'actions manquant pour %s. DCF non calculé.", symbol)
                continue

            fcf_actuel = calculer_fcf(ebit=ebit, taux_imposition=taux_imposition, capex=capex, variation_bfr=bfr)
            if fcf_actuel <= 0:
                logger.warning("FCF actuel <= 0 pour %s. DCF non calculé.", symbol)
                continue

            equity_value, details = calculer_dcf(
                fcf_actuel=fcf_actuel,
                taux_croissance_fcf=hypotheses["taux_croissance_fcf"],
                taux_croissance_terminal=hypotheses["taux_croissance_terminal"],
                taux_actualisation=hypotheses["taux_actualisation"],
                periode_prevision=hypotheses["periode_prevision"],
                dette_nette=dette_nette, cash=cash,
            )

            valeur_par_action = equity_value / shares_outstanding
            cours_actuel = row.get("close")
            ecart_pct = (
                (valeur_par_action - cours_actuel) / cours_actuel * 100
                if pd.notna(cours_actuel) and cours_actuel else None
            )

            results.append({
                "Ticker": symbol, "Secteur": row.get("sector"), "Année": row.get("year"),
                "FCF_actuel": fcf_actuel, "Enterprise_Value": details["Enterprise_Value"],
                "Equity_Value": equity_value, "Valeur_par_action_DCF": valeur_par_action,
                "Cours_actuel": cours_actuel, "Écart_DCF_vs_Cours_%": ecart_pct,
                **{k: v for k, v in details.items() if k != "FCF_futurs"},
            })
        except Exception as e:  # noqa: BLE001
            logger.error("Erreur pour %s: %s", symbol, e)
            continue

    return pd.DataFrame(results)


def main() -> None:
    if not (config.FINANCIALS_FILE.exists() and config.PRICES_FILE.exists()):
        logger.error("Fichiers manquants. Lance d'abord 03_recuperation_cours.py et 05_recuperation_10k.py.")
        return

    df_input = build_input_table()
    df_dcf = calculer_dcf_par_entreprise(df_input, HYPOTHESES_DEFAUT)

    if df_dcf.empty:
        logger.error("Aucun résultat DCF calculé. Vérifiez les données d'entrée.")
        return

    config.DCF_FILE.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(config.DCF_FILE, engine="openpyxl") as writer:
        df_dcf.to_excel(writer, sheet_name="DCF", index=False)
        df_input.to_excel(writer, sheet_name="Données d'entrée", index=False)
    logger.info("Résultats DCF sauvegardés : %s", config.DCF_FILE)

    logger.info("Résumé des résultats DCF:")
    for _, row in df_dcf.iterrows():
        cours = row["Cours_actuel"]
        cours_str = f"{cours:.2f}$" if pd.notna(cours) else "n/a"
        ecart = row["Écart_DCF_vs_Cours_%"]
        ecart_str = f"{ecart:.1f}%" if pd.notna(ecart) else "n/a"
        logger.info(
            "%s (%s): Valeur DCF/action = %.2f$, Cours = %s, Écart = %s",
            row["Ticker"], row["Secteur"], row["Valeur_par_action_DCF"], cours_str, ecart_str,
        )


if __name__ == "__main__":
    main()