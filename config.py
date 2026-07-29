"""
Configuration partagée par tous les scripts du pipeline.

Objectif du pipeline (entreprises américaines uniquement) :
    01_build_universe.py        -> univers S&P 500 (RIC, Instrument_Name, Country, Currency, Exchange)
    02_categoriser_secteurs.py  -> ajoute une colonne "sector" à l'univers
    03_recuperation_cours.py    -> cours de clôture de fin d'année via IBKR
    04_recuperation_options.py  -> chaînes d'options (ITM/ATM/OTM) via IBKR + greeks
    05_recuperation_10k.py      -> données financières via l'API XBRL companyfacts de la SEC
    06_calcul_multiples.py      -> EV/EBITDA, EV/Sales, P/E à partir de 03 + 05
    07_calcul_multiples_moyens.py -> moyennes/médianes des multiples par secteur
    08_calcul_dcf.py            -> valorisation DCF à partir de 05 (+ 02 + 03 pour comparaison)

Tous les scripts lisent/écrivent dans des sous-dossiers de BASE_DIR, avec un
schéma de colonnes commun défini ci-dessous, pour que les scripts puissent
s'enchaîner sans transformation manuelle entre les deux.
"""

from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Arborescence de sortie (unique, partagée par tous les scripts)
# ----------------------------------------------------------------------------
# Change BASE_DIR si besoin (ex: vers un disque dédié). Tout le reste est
# calculé automatiquement à partir de cette seule variable.
BASE_DIR = Path("./data")

DIR_UNIVERSE = BASE_DIR / "universe"
DIR_PRICES = BASE_DIR / "prices"
DIR_OPTIONS = BASE_DIR / "options"
DIR_OPTIONS_HISTORY = DIR_OPTIONS / "history"          # archive d'un snapshot par run de 04 (voir plus bas)
DIR_FINANCIALS = BASE_DIR / "financials"
DIR_MULTIPLES = BASE_DIR / "multiples"
DIR_DCF = BASE_DIR / "dcf"

UNIVERSE_FILE = DIR_UNIVERSE / "sp500_universe.csv"          # sortie de 01, entrée de 02/03/04
PRICES_FILE = DIR_PRICES / "year_end_prices.parquet"          # sortie de 03
OPTIONS_FILE = DIR_OPTIONS / "option_chains.parquet"          # sortie de 04 : DERNIER snapshot uniquement
FINANCIALS_FILE = DIR_FINANCIALS / "financials.parquet"       # sortie consolidée de 05
MULTIPLES_FILE = DIR_MULTIPLES / "multiples.parquet"          # sortie de 06
MULTIPLES_MOYENS_FILE = DIR_MULTIPLES / "multiples_moyens_par_secteur.xlsx"  # sortie de 07
DCF_FILE = DIR_DCF / "resultats_dcf.xlsx"                     # sortie de 08

for d in (DIR_UNIVERSE, DIR_PRICES, DIR_OPTIONS, DIR_OPTIONS_HISTORY, DIR_FINANCIALS, DIR_MULTIPLES, DIR_DCF):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Schéma de colonnes canonique (utilisé par TOUS les scripts en aval)
# ----------------------------------------------------------------------------
# symbol           : ticker IBKR / SEC, ex "AAPL"
# company_name     : nom de l'entreprise
# sector           : secteur (catégorisé par 02)
# year             : année (int)
# close            : cours de clôture de fin d'année (03)
# shares_outstanding, revenue, ebitda, ebit, net_income, capex, da,
# working_capital, net_debt, cash, tax_rate, interest_expense : (05)
#
# C'est ce schéma qui remplace les anciens noms mixtes FR/EN
# ("CA", "Nombre d'action", "Dette net", "Cash et Cash Equivalents", ...)
# qui empêchaient les scripts de se lire les uns les autres.

# ----------------------------------------------------------------------------
# Univers : entreprises américaines uniquement
# ----------------------------------------------------------------------------
# Mapping "Exchange" (tel que présent dans le fichier d'univers) -> exchange
# IBKR pour le contrat ACTION sous-jacent. On ne garde QUE les places US.
EXCHANGE_MAP: dict[str, str] = {
    "NYSE": "NYSE",
    "New York Stock Exchange": "NYSE",
    "Nasdaq": "NASDAQ",
    "NASDAQ": "NASDAQ",
    "Nasdaq Global Select": "NASDAQ",
    "Nasdaq NMS - Global Market": "NASDAQ",
    "Nasdaq NMS - Global Select Market": "NASDAQ",
    "NYSE American": "AMEX",
    "NYSE MKT LLC": "AMEX",
    "NYSE Arca": "ARCA",
    "Cboe BZX": "BATS",
}
# Toute valeur d'Exchange absente de ce mapping est routée en direct via
# SMART (voir resolve_stock_contract dans les scripts 03/04) plutôt que
# d'être ignorée : contrairement à l'univers Europe multi-place d'origine,
# l'univers US est homogène et SMART route correctement vers NYSE/NASDAQ/AMEX.
DEFAULT_EXCHANGE = "SMART"

# Overrides ticker Wikipedia/RIC -> symbole IBKR, pour les classes d'actions
# où IBKR utilise un espace au lieu d'un point (ex: BRK.B -> "BRK B").
# Complétez au besoin si un ticker se résout mal.
SYMBOL_OVERRIDES: dict[str, str] = {
    "BRK.B": "BRK B",
    "BF.B": "BF B",
}

CURRENCY = "USD"
COUNTRY = "United States"


def to_ib_symbol(ric: str) -> str:
    """Convertit un RIC de l'univers vers le symbole IBKR utilisé comme
    colonne 'symbol' canonique dans TOUS les fichiers du pipeline (cours,
    options, financials). Centralisé ici (au lieu d'être dupliqué dans
    03/04/05) pour garantir que les trois scripts produisent exactement le
    même format de symbole et que les jointures sur 'symbol' fonctionnent
    pour les tickers à classes d'actions (ex: "BRK.B" -> "BRK B").
    """
    return SYMBOL_OVERRIDES.get(ric, ric.split(".")[0])
