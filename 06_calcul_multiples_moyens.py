"""
Calcule les multiples moyens/médians par secteur à partir de la sortie de
05_calcul_multiples.py, et sauvegarde le résultat en Excel.

Corrections par rapport à CalculMultipleMoyenMark1 :
    - Lisait un fichier Excel d'entrée fixe ("donnees_entreprises.xlsx") qui
      n'était produit par aucun autre script du pipeline. Lit maintenant
      directement config.MULTIPLES_FILE (sortie de 06).
    - Colonnes "Ticker"/"Secteur" renommées en "symbol"/"sector" pour
      correspondre au schéma canonique produit par 02 et 06.

Usage :
    python 06_calcul_multiples_moyens.py
"""

from __future__ import annotations

import logging

import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLONNES_MULTIPLES = ["EV/EBITDA", "EV/Sales", "P/E"]


def calculer_multiples_moyens_par_secteur(df: pd.DataFrame, colonnes_multiples: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    colonnes_requises = ["symbol", "sector"] + colonnes_multiples
    for colonne in colonnes_requises:
        if colonne not in df.columns:
            raise ValueError(f"La colonne '{colonne}' est manquante dans le fichier d'entrée.")

    df = df.dropna(subset=["sector"]).copy()

    rows = []
    for secteur in df["sector"].unique():
        df_secteur = df[df["sector"] == secteur]
        for multiple in colonnes_multiples:
            valeurs = df_secteur[multiple].dropna()
            valeurs = valeurs[valeurs > 0]
            rows.append({
                "Secteur": secteur,
                "Multiple": multiple,
                "Moyenne": valeurs.mean() if not valeurs.empty else None,
                "Médiane": valeurs.median() if not valeurs.empty else None,
                "Min": valeurs.min() if not valeurs.empty else None,
                "Max": valeurs.max() if not valeurs.empty else None,
                "Écart-type": valeurs.std() if not valeurs.empty else None,
                "Nombre d'entreprises": len(valeurs),
            })

    df_resultats = pd.DataFrame(rows)
    df_pivot = df_resultats.pivot_table(
        index="Secteur", columns="Multiple",
        values=["Moyenne", "Médiane", "Min", "Max", "Écart-type", "Nombre d'entreprises"],
        aggfunc="first",
    )
    df_pivot.columns = [f"{stat}_{multiple}" for stat, multiple in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    return df_resultats, df_pivot


def main() -> None:
    if not config.MULTIPLES_FILE.exists():
        logger.error("Fichier introuvable: %s. Lance d'abord 05_calcul_multiples.py.", config.MULTIPLES_FILE)
        return

    df = pd.read_parquet(config.MULTIPLES_FILE)
    df_resultats, df_pivot = calculer_multiples_moyens_par_secteur(df, COLONNES_MULTIPLES)

    config.MULTIPLES_MOYENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(config.MULTIPLES_MOYENS_FILE, engine="openpyxl") as writer:
        df_resultats.to_excel(writer, sheet_name="Détails", index=False)
        df_pivot.to_excel(writer, sheet_name="Résumé", index=False)
        df.to_excel(writer, sheet_name="Données brutes", index=False)

    logger.info("Fichier sauvegardé : %s", config.MULTIPLES_MOYENS_FILE)


if __name__ == "__main__":
    main()
