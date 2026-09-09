"""
Catégorise chaque entreprise de l'univers dans un secteur (liste fixe),
et écrit la colonne "sector" dans config.UNIVERSE_FILE.

Corrections / changements par rapport à CateEntMark2 :
    - Lit/écrit directement config.UNIVERSE_FILE (au lieu d'un chemin en dur
      "slpublic_sxxp_20260701.csv"), pour s'enchaîner avec 01/03/04.
    - Utilise le secteur GICS déjà présent dans l'univers (ajouté par
      01_build_universe.py) via un mapping direct GICS -> ta liste de
      secteurs : la quasi-totalité des entreprises US n'ont donc plus besoin
      d'appel API. Mistral n'est appelé qu'en dernier recours (GICS absent
      ou mapping ambigu), ce qui réduit fortement le coût/temps par rapport
      au script d'origine qui appelait l'API pour les 600 entreprises.
    - API_KEY se lit maintenant depuis la variable d'environnement
      MISTRAL_API_KEY (au lieu d'être en dur dans le fichier) : évite de
      committer une clé par erreur. Le script tourne sans clé si le mapping
      GICS suffit (cas le plus fréquent) et sans fichier secteurs_manuels.json.

Usage :
    python 02_categoriser_secteurs.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Config API Mistral (fallback uniquement) --------------------------------
# ⚠️ La clé était précédemment codée en dur dans ce fichier (committée dans un
# dépôt public) : si tu utilises encore cette clé, RÉVOQUE-LA côté Mistral et
# régénères-en une nouvelle avant toute chose. Elle se lit maintenant
# uniquement depuis la variable d'environnement MISTRAL_API_KEY, ex:
#   export MISTRAL_API_KEY="ta_nouvelle_cle"
API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

SECTEURS = [
    "Agro-alimentaire et boissons", "Assurance", "Automobiles et équipementiers",
    "Banques", "Bâtiment et matériaux de construction", "Biens et services industriels",
    "Chimie", "Distribution", "Immobilier", "Matières premières", "Medias",
    "Pétrole et gaz", "Produits ménagers et de soin personnel", "Santé",
    "Services aux collectivités", "Services financiers", "Technologie",
    "Télécommunications", "Voyage et loisirs",
]

# Mapping direct GICS Sector (US, tel que fourni par Wikipedia) -> ta liste
# de secteurs. Couvre les 11 secteurs GICS ; à ajuster si tu veux un
# découpage plus fin (ex: séparer Banques / Services financiers / Assurance
# qui sont tous dans "Financials" en GICS -> ici mis par défaut sur
# "Services financiers", à corriger manuellement via secteurs_manuels.json
# pour les banques et assureurs si tu veux la distinction).
GICS_TO_SECTEUR = {
    "Information Technology": "Technologie",
    "Health Care": "Santé",
    "Financials": "Services financiers",
    "Consumer Discretionary": "Distribution",
    "Communication Services": "Medias",
    "Industrials": "Biens et services industriels",
    "Consumer Staples": "Agro-alimentaire et boissons",
    "Energy": "Pétrole et gaz",
    "Utilities": "Services aux collectivités",
    "Real Estate": "Immobilier",
    "Materials": "Matières premières",
}

CACHE_FILE = Path("secteur_cache.json")
MANUAL_SECTORS_FILE = Path("secteurs_manuels.json")

BATCH_SIZE = 5
MAX_RETRIES = 3
RETRY_DELAY = 2
TEMPERATURE = 0.1
MAX_TOKENS = 100


def charger_json(path: Path) -> Dict[str, str]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Erreur de chargement de %s: %s", path, e)
    return {}


def sauvegarder_cache(cache: Dict[str, str]) -> None:
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.error("Erreur de sauvegarde du cache: %s", e)


def appeler_mistral(entreprises: List[str]) -> Dict[str, Optional[str]]:
    if not entreprises:
        return {}
    if not API_KEY:
        logger.warning(
            "MISTRAL_API_KEY non définie : %d entreprises sans secteur GICS "
            "exploitable resteront 'indetermine' (renseigne secteurs_manuels.json "
            "ou exporte MISTRAL_API_KEY pour les résoudre via l'API).",
            len(entreprises),
        )
        return {}

    prompt = f"""
    Tu es un expert en analyse financière. Catégorise les entreprises suivantes dans UN SEUL des secteurs ci-dessous.
    Si aucune correspondance, réponds "indetermine" pour cette entreprise.
    Secteurs possibles: {', '.join(SECTEURS)}.

    Entreprises à catégoriser:
    {chr(10).join(f"- {e}" for e in entreprises)}

    Réponds UNIQUEMENT avec un JSON valide au format:
    {{"{entreprises[0]}": "secteur ou indetermine"}}
    """
    data = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS * max(1, len(entreprises)),
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(MISTRAL_URL, headers=HEADERS, json=data, timeout=30)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            if not content.startswith("{") or not content.endswith("}"):
                logger.warning("Réponse invalide de Mistral: %s", content)
                return {}
            result = json.loads(content)
            for entreprise in entreprises:
                if entreprise not in result:
                    logger.warning("Entreprise manquante dans la réponse: %s", entreprise)
                    return {}
            return result
        except requests.exceptions.RequestException as e:
            delay = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
            logger.warning("Tentative %d échouée pour %s: %s. Nouvel essai dans %.1fs...",
                            attempt + 1, entreprises, e, delay)
            time.sleep(delay)
        except (KeyError, json.JSONDecodeError) as e:
            logger.error("Erreur de parsing pour %s: %s", entreprises, e)
            return {}

    logger.error("Échec après %d tentatives pour %s", MAX_RETRIES, entreprises)
    return {}


# Bucket retenu pour une financière dont la sous-industrie est absente ou
# inconnue de la table. "Banques" et non "Services financiers", ce qui revient
# à l'EXCLURE du DCF (config.SECTORS_SANS_DCF) : les deux erreurs possibles
# n'ont pas le même coût. Classer à tort un encaisseur de commissions en
# prêteur fait perdre un signal ; classer à tort un prêteur en encaisseur de
# commissions fabrique un DCF sur une banque -- une valorisation qui ne veut
# rien dire, et sur laquelle la stratégie prendrait position. En cas de doute,
# on renonce au signal.
SECTEUR_FINANCIER_PAR_DEFAUT = "Banques"


def secteur_fin(gics_sector, gics_sub_industry) -> Optional[str]:
    """Secteur déduit de la SOUS-INDUSTRIE GICS, ou None si le découpage fin
    ne s'applique pas (entreprise non financière).

    Restreint à `Financials` À DESSEIN. C'est le seul secteur GICS que le
    pipeline traite comme un bloc alors qu'il réunit trois métiers dont un
    seul se valorise par un FCFF (cf. config.GICS_SUB_INDUSTRY_TO_SECTEUR).
    Le garde-fou sur le secteur n'est pas cosmétique : "Data Processing &
    Outsourced Services" était une sous-industrie de la TECHNOLOGIE avant le
    remaniement de 2023, et sans cette restriction une SSII y serait classée
    en services financiers."""
    if not isinstance(gics_sector, str) or gics_sector.strip() != "Financials":
        return None
    if not isinstance(gics_sub_industry, str) or not gics_sub_industry.strip():
        return SECTEUR_FINANCIER_PAR_DEFAUT
    return config.GICS_SUB_INDUSTRY_TO_SECTEUR.get(
        gics_sub_industry.strip(), SECTEUR_FINANCIER_PAR_DEFAUT,
    )


def categoriser_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cache = charger_json(CACHE_FILE)
    secteurs_manuels = charger_json(MANUAL_SECTORS_FILE)

    secteurs: Dict[str, str] = {}
    a_appeler: List[str] = []

    fines = 0
    financieres_sans_sous_industrie: List[str] = []

    for _, row in df.drop_duplicates(subset=["Instrument_Name"]).iterrows():
        nom = row["Instrument_Name"]
        gics = row.get("GICS_Sector")
        sous_industrie = row.get("GICS_Sub_Industry")
        fine = secteur_fin(gics, sous_industrie)
        if fine is not None and fine == SECTEUR_FINANCIER_PAR_DEFAUT and (
            not isinstance(sous_industrie, str)
            or sous_industrie.strip() not in config.GICS_SUB_INDUSTRY_TO_SECTEUR
        ):
            financieres_sans_sous_industrie.append(nom)

        if nom in secteurs_manuels:
            secteurs[nom] = secteurs_manuels[nom]
        elif fine is not None:
            # AVANT le cache, délibérément : la sous-industrie est une donnée
            # officielle lue dans une table, le cache n'existe que pour éviter
            # des appels LLM. Laisser le cache gagner aurait figé les
            # "Services financiers" en bloc déjà écrits par les runs
            # précédents, et le découpage fin n'aurait jamais pris effet.
            secteurs[nom] = fine
            fines += 1
        elif nom in cache:
            secteurs[nom] = cache[nom]
        elif pd.notna(gics) and gics in GICS_TO_SECTEUR:
            secteurs[nom] = GICS_TO_SECTEUR[gics]
        else:
            a_appeler.append(nom)

    logger.info(
        "%d entreprises résolues via secteurs manuels/cache/GICS (dont %d financières "
        "découpées par sous-industrie), %d via API Mistral (fallback).",
        len(secteurs), fines, len(a_appeler),
    )
    if financieres_sans_sous_industrie:
        logger.warning(
            "%d financières sans sous-industrie GICS exploitable : rabattues sur '%s', donc "
            "EXCLUES du DCF par prudence (un FCFF sur un prêteur ne veut rien dire, alors "
            "qu'une exclusion à tort ne coûte qu'un signal). Attendu pour les entreprises "
            "RADIÉES, absentes de la table Wikipedia des membres actuels ; pour un membre "
            "actuel, relance 01_build_universe.py afin de récupérer la colonne "
            "GICS Sub-Industry. Concernées : %s",
            len(financieres_sans_sous_industrie), SECTEUR_FINANCIER_PAR_DEFAUT,
            ", ".join(sorted(financieres_sans_sous_industrie)[:20]),
        )

    for i in range(0, len(a_appeler), BATCH_SIZE):
        batch = a_appeler[i:i + BATCH_SIZE]
        result = appeler_mistral(batch)
        for nom in batch:
            secteurs[nom] = result.get(nom, "indetermine")

    cache.update({k: v for k, v in secteurs.items() if v != "indetermine"})
    sauvegarder_cache(cache)

    df["sector"] = df["Instrument_Name"].map(secteurs)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--universe", type=Path, default=config.UNIVERSE_FILE,
        help="Fichier d'univers à catégoriser, mis à jour SUR PLACE (défaut: %(default)s). "
             "Passe config.UNIVERSE_FULL_FILE (sortie de 01b) pour catégoriser aussi les "
             "entreprises RADIÉES : sans secteur, elles sont exclues des médianes "
             "sectorielles de 06b et le biais de survivance subsiste malgré le backfill.",
    )
    args = parser.parse_args()

    if not args.universe.exists():
        logger.error(
            "Univers introuvable: %s. Lance d'abord 01_build_universe.py (ou "
            "01b_historique_univers_sp500.py pour l'univers complet).", args.universe,
        )
        return

    df = pd.read_csv(args.universe, encoding="utf-8-sig")
    if "GICS_Sector" not in df.columns:
        # UNIVERSE_FULL_FILE (01b) ne porte pas le secteur GICS : les radiées
        # ne figurent plus dans la table Wikipedia des membres actuels. Tout
        # passe donc par le cache, les secteurs manuels et Mistral.
        logger.info(
            "%s n'a pas de colonne GICS_Sector : catégorisation via le cache, %s et "
            "l'API Mistral uniquement (attendu pour l'univers complet de 01b).",
            args.universe, MANUAL_SECTORS_FILE,
        )
    df_categorise = categoriser_df(df)
    df_categorise.to_csv(args.universe, index=False, encoding="utf-8-sig")
    logger.info("Univers mis à jour avec les secteurs : %s", args.universe)

    indetermines = df_categorise[df_categorise["sector"] == "indetermine"]
    if not indetermines.empty:
        logger.warning("%d entreprises à catégoriser manuellement (ajoute-les dans %s):",
                        len(indetermines), MANUAL_SECTORS_FILE)
        for _, row in indetermines.iterrows():
            logger.warning("  - %s (%s)", row["Instrument_Name"], row["RIC"])

    stats = df_categorise["sector"].value_counts()
    logger.info("Statistiques de catégorisation:\n%s", stats.to_string())


if __name__ == "__main__":
    main()
