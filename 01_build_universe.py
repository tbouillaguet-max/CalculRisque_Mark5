"""
NOUVEAU (n'existait pas dans tes scripts d'origine) : construit le fichier
d'univers pour les entreprises américaines, au même format que celui que tu
utilisais pour le STOXX 600 (RIC, Instrument_Name, Country, Currency,
Exchange), pour que 02/03/04 puissent le consommer sans rien changer.

Source : liste des composants du S&P 500 sur Wikipedia (table "Symbol").
Si tu as déjà un fichier CSV avec un autre univers US (Russell 1000, ta
propre liste...), tu peux l'ignorer : il suffit que ton CSV ait les 5
colonnes RIC, Instrument_Name, Country, Currency, Exchange (Exchange doit
être une valeur du EXCHANGE_MAP de config.py, sinon SMART sera utilisé par
défaut).

Usage :
    python 01_build_universe.py
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia renvoie une erreur 403 aux requêtes sans User-Agent identifiable
# (c'est ce que fait pandas.read_html(url) par défaut, via urllib). On passe
# donc par requests avec un User-Agent explicite, puis on donne le HTML déjà
# téléchargé à read_html.
HEADERS = {"User-Agent": "options-pipeline/1.0 (contact: voir SEC_CONTACT_EMAIL en tête de 05_recuperation_10k.py)"}


def build_universe() -> pd.DataFrame:
    resp = requests.get(WIKI_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    raw = tables[0]  # première table = composants actuels

    df = pd.DataFrame({
        "RIC": raw["Symbol"].str.strip(),
        "Instrument_Name": raw["Security"].str.strip(),
        "Country": config.COUNTRY,
        "Currency": config.CURRENCY,
        # La table Wikipedia actuelle n'a PAS de colonne "Exchange" (elle a
        # Symbol/Security/GICS Sector/GICS Sub-Industry/Headquarters
        # Location/Date added/CIK/Founded, vérifié en juillet 2026) : on met
        # directement "SMART", qui route correctement vers NYSE/NASDAQ/AMEX
        # sans avoir besoin de connaître la place précise de cotation.
        "Exchange": "SMART",
        # Bonus : Wikipedia fournit déjà le secteur GICS officiel. On le
        # récupère ici pour que 02_categoriser_secteurs.py puisse s'en servir
        # et n'appeler l'API Mistral que pour les cas ambigus (économie d'appels).
        "GICS_Sector": raw["GICS Sector"].str.strip() if "GICS Sector" in raw.columns else None,
    })

    # Certains tickers Wikipedia utilisent un point pour les classes d'actions
    # (BRK.B, BF.B) : le mapping SYMBOL_OVERRIDES de config.py les corrige en
    # aval (03/04), pas besoin d'y toucher ici.
    df = df.dropna(subset=["RIC"]).drop_duplicates(subset=["RIC"]).reset_index(drop=True)

    return df


def main() -> None:
    df = build_universe()
    df.to_csv(config.UNIVERSE_FILE, index=False, encoding="utf-8-sig")
    logger.info("Univers S&P 500 sauvegardé : %s (%d entreprises)", config.UNIVERSE_FILE, len(df))


if __name__ == "__main__":
    main()