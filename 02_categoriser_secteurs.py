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


def categoriser_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cache = charger_json(CACHE_FILE)
    secteurs_manuels = charger_json(MANUAL_SECTORS_FILE)

    secteurs: Dict[str, str] = {}
    a_appeler: List[str] = []

    for _, row in df.drop_duplicates(subset=["Instrument_Name"]).iterrows():
        nom = row["Instrument_Name"]
        gics = row.get("GICS_Sector")
        if nom in secteurs_manuels:
            secteurs[nom] = secteurs_manuels[nom]
        elif nom in cache:
            secteurs[nom] = cache[nom]
        elif pd.notna(gics) and gics in GICS_TO_SECTEUR:
            secteurs[nom] = GICS_TO_SECTEUR[gics]
        else:
            a_appeler.append(nom)

    logger.info(
        "%d entreprises résolues via secteurs manuels/cache/GICS, %d via API Mistral (fallback).",
        len(secteurs), len(a_appeler),
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
    if not config.UNIVERSE_FILE.exists():
        logger.error("Univers introuvable: %s. Lance d'abord 01_build_universe.py.", config.UNIVERSE_FILE)
        return

    df = pd.read_csv(config.UNIVERSE_FILE, encoding="utf-8-sig")
    df_categorise = categoriser_df(df)
    df_categorise.to_csv(config.UNIVERSE_FILE, index=False, encoding="utf-8-sig")
    logger.info("Univers mis à jour avec les secteurs : %s", config.UNIVERSE_FILE)

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
