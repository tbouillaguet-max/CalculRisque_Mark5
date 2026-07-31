"""
Configuration partagée par tous les scripts du pipeline.

Objectif du pipeline (entreprises américaines uniquement) : on récupère
d'abord les données de marché "de base" (cours + 10-K), on en dérive une
valorisation théorique, et on ne va chercher les chaînes d'options (lent,
rate-limité côté IBKR) que pour les entreprises où cette valorisation
théorique s'écarte significativement du cours de bourse.

    01_build_universe.py        -> univers S&P 500 (RIC, Instrument_Name, Country, Currency, Exchange)
    02_categoriser_secteurs.py  -> ajoute une colonne "sector" à l'univers
    03_recuperation_cours.py    -> cours de clôture de fin d'année via IBKR
    04_recuperation_10k.py      -> données financières ANNUELLES (10-K) via l'API XBRL
                                    companyfacts de la SEC
    04b_recuperation_10q.py     -> données TRIMESTRIELLES (10-Q + 10-K), reconstruit un TTM
                                    (somme glissante des 4 derniers trimestres) point-in-time,
                                    voir sa docstring pour la distinction TTM vs trimestre brut
    04c_recuperation_8k.py      -> événements matériels (8-K) entre deux trimestres TTM connus,
                                    classifiés par LLM (Mistral) via sec_filings_text.py
    05_calcul_multiples.py      -> EV/EBITDA, EV/Sales, P/E à partir de 03(b) + 04 + 04b
    07b_validation_qualitative.py -> verdict LLM de cohérence qualitative (texte du 10-K/10-Q à sa
                                    date de dépôt) vs l'écart de valorisation quantitatif (07/06b)
    06_calcul_multiples_moyens.py -> moyennes/médianes des multiples par secteur
    07_calcul_dcf.py            -> valorisation théorique (DCF) à partir de 04 (+ 02 + 03),
                                    calcule l'écart en % entre cours de bourse et valeur théorique
    08_recuperation_options.py  -> chaînes d'options (ITM/ATM/OTM) via IBKR + greeks, UNIQUEMENT
                                    pour les entreprises dont l'écart calculé en 07 dépasse
                                    VALUATION_GAP_THRESHOLD_PCT (en valeur absolue). IV/greeks en
                                    priorité via Alpha Vantage (gratuit, ALPHAVANTAGE_API_KEY,
                                    voir 08), Black-Scholes local en dernier repli. --av-backfill-dates
                                    permet aussi de reconstituer un VRAI historique d'options passées.

Backtest (voir README.md pour le détail) :
    01b_historique_univers_sp500.py     -> univers point-in-time (actuels + radiés)
    03b_recuperation_cours_quotidiens.py -> cours quotidiens (IBKR + repli Stooq)
    06b_calcul_valorisation_combinee.py -> valorisation combinée (multiples par année
                                    en priorité, DCF en repli), signal de la stratégie options
    09_backtest.py              -> backtest actions (écart DCF)
    10_backtest_options.py      -> backtest options (call/put, dimensionné par delta)

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
DIR_BACKTEST = BASE_DIR / "backtest"

UNIVERSE_FILE = DIR_UNIVERSE / "sp500_universe.csv"          # sortie de 01, entrée de 02/03/04
PRICES_FILE = DIR_PRICES / "year_end_prices.parquet"          # sortie de 03
OPTIONS_FILE = DIR_OPTIONS / "option_chains.parquet"          # sortie de 04 : DERNIER snapshot uniquement
FINANCIALS_FILE = DIR_FINANCIALS / "financials.parquet"       # sortie consolidée de 05

# Sortie de 04b_recuperation_10q.py (10-Q + reconstruction TTM, voir sa
# docstring) : FINANCIALS_QUARTERLY_FILE garde CHAQUE trimestre discret
# (utile pour audit/debug de la discrétisation) ; FINANCIALS_TTM_FILE garde
# les lignes TTM glissantes (somme des flux sur 4 trimestres, dernière valeur
# connue pour les postes de bilan) réellement consommées par 05/06b/07 en
# plus de FINANCIALS_FILE (annuel, 10-K seul, inchangé).
FINANCIALS_QUARTERLY_FILE = DIR_FINANCIALS / "financials_quarterly.parquet"
FINANCIALS_TTM_FILE = DIR_FINANCIALS / "financials_ttm.parquet"

# Fenêtre de tolérance (jours) pour associer à une ligne financière (annuelle
# ou TTM) le dernier cours de clôture QUOTIDIEN connu à sa filed_date (05/07,
# via DAILY_PRICES_FILE -- 03b_recuperation_cours_quotidiens.py). Au-delà,
# repli sur le cours de clôture ANNUEL (PRICES_FILE, 03) le plus proche.
DAILY_PRICE_ASOF_TOLERANCE_DAYS = 10
MULTIPLES_FILE = DIR_MULTIPLES / "multiples.parquet"          # sortie de 06
MULTIPLES_MOYENS_FILE = DIR_MULTIPLES / "multiples_moyens_par_secteur.xlsx"  # sortie de 07
DCF_FILE = DIR_DCF / "resultats_dcf.xlsx"                     # sortie de 08

# Sortie de 07b_validation_qualitative.py : verdict LLM (Mistral) de
# cohérence qualitative entre le signal quantitatif (DCF_HISTORY_FILE /
# VALORISATION_COMBINEE_FILE) et le texte du 10-K/10-Q DE CE DÉPÔT PRÉCIS
# (jamais un filing plus récent -- contrainte anti-anticipation, voir sa
# docstring), une ligne par (symbol, period_type, fiscal_year, fiscal_quarter).
QUALITATIVE_VALIDATION_FILE = DIR_DCF / "validation_qualitative.parquet"

# Sortie de 04c_recuperation_8k.py : événements matériels détectés dans les
# 8-K déposés entre deux trimestres TTM connus (rachats d'actions, guidance,
# départ de dirigeant, procédure judiciaire, M&A...), classifiés par le même
# module LLM point-in-time que 07b (sec_filings_text.py).
MATERIAL_EVENTS_8K_FILE = DIR_FINANCIALS / "material_events_8k.parquet"

# ----------------------------------------------------------------------------
# Backtest (voir 01b/03b/09 et le package backtest/)
# ----------------------------------------------------------------------------
# Univers point-in-time : composants ACTUELS + historiquement radiés, avec
# leurs dates d'entrée/sortie de l'indice, pour permettre un backtest sans
# biais de survivance (01b_historique_univers_sp500.py).
UNIVERSE_HISTORY_FILE = DIR_UNIVERSE / "sp500_universe_history.parquet"  # sortie de 01b : spans d'appartenance
UNIVERSE_FULL_FILE = DIR_UNIVERSE / "sp500_universe_full.csv"            # sortie de 01b : superset actuels+radiés, entrée de 03b/04

# Cours quotidiens (contrairement à PRICES_FILE qui ne garde que la clôture
# de fin d'année) : nécessaires pour un backtest à granularité journalière.
DAILY_PRICES_FILE = DIR_PRICES / "daily_prices.parquet"       # sortie de 03b

# Écart de valorisation DCF pour CHAQUE exercice historique (pas seulement le
# dernier comme DCF_FILE), avec la date de dépôt SEC (filed_date) de chaque
# exercice pour savoir à partir de quand un signal est réellement disponible
# (le 10-K de l'exercice N est déposé ~2-3 mois après sa clôture : l'utiliser
# dès le 31/12 de l'exercice N serait du look-ahead bias).
DCF_HISTORY_FILE = DIR_DCF / "dcf_historique.parquet"         # sortie de 07 (en plus de DCF_FILE)

# Valorisation théorique combinée (multiples sectoriels PAR ANNÉE en priorité,
# DCF en repli quand les multiples sont indisponibles), pour CHAQUE exercice
# historique -- signal utilisé par la stratégie options (voir
# backtest/strategies/valuation_gap_options.py), distincte de DCF_HISTORY_FILE
# (utilisée par la stratégie actions, DCF seul, inchangée).
VALORISATION_COMBINEE_FILE = DIR_MULTIPLES / "valorisation_combinee_historique.parquet"  # sortie de 06b

DIR_BACKTEST_OPTIONS = BASE_DIR / "backtest_options"  # sortie de 10_backtest_options.py

# ----------------------------------------------------------------------------
# Journal d'exécution du pipeline (run_pipeline_quarterly.py)
# ----------------------------------------------------------------------------
# Un sous-dossier par run, contenant report.json (statut/durée/tentatives de
# chaque étape) et un fichier de log par étape. Lu par la page Streamlit
# "Pipeline" pour montrer l'état du pipeline sans ouvrir les logs à la main,
# et par --resume pour repartir des étapes qui restent à faire.
DIR_PIPELINE_RUNS = BASE_DIR / "pipeline_runs"
PIPELINE_RUN_REPORT_NAME = "report.json"

for d in (
    DIR_UNIVERSE, DIR_PRICES, DIR_OPTIONS, DIR_OPTIONS_HISTORY, DIR_FINANCIALS,
    DIR_MULTIPLES, DIR_DCF, DIR_BACKTEST, DIR_BACKTEST_OPTIONS, DIR_PIPELINE_RUNS,
):
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

# Taux sans risque constant utilisé par 08_recuperation_options.py pour le
# pricing Black-Scholes de repli (estimate_iv_and_greeks) quand IBKR ne
# renvoie pas de modelGreeks. Référencé mais jamais défini auparavant (le
# script plantait à l'import dès qu'un greek devait être estimé
# localement) ; approximation du taux 3 mois US, à ajuster si besoin.
RISK_FREE_RATE = 0.04

# ----------------------------------------------------------------------------
# Filtre de valorisation (déclenche la récupération des options en 08)
# ----------------------------------------------------------------------------
# 07_calcul_dcf.py calcule pour chaque entreprise l'écart en % entre son
# cours de bourse et sa valeur théorique (DCF). 08_recuperation_options.py ne
# récupère les chaînes d'options que pour les entreprises dont cet écart
# dépasse ce seuil, en valeur absolue : la récupération d'options via IBKR
# est lente et rate-limitée, inutile de la lancer sur tout l'univers si seule
# une fraction des entreprises montre un écart de valorisation significatif.
VALUATION_GAP_THRESHOLD_PCT = 20.0

# ----------------------------------------------------------------------------
# Paramètres par défaut du backtest (voir backtest/strategies/valuation_gap.py)
# ----------------------------------------------------------------------------
# Réutilise VALUATION_GAP_THRESHOLD_PCT comme seuil d'entrée par défaut (même
# logique que 08 : une sous-évaluation de moins de 20% n'est pas jugée
# significative). Ajustable via --entry-threshold-pct sur 09_backtest.py.
BACKTEST_ENTRY_THRESHOLD_PCT = VALUATION_GAP_THRESHOLD_PCT
BACKTEST_STOP_LOSS_PCT = -15.0     # clôture la position si le cours baisse de 15% depuis l'entrée
BACKTEST_TAKE_PROFIT_PCT = 30.0    # clôture la position si le cours monte de 30% depuis l'entrée
BACKTEST_MAX_POSITIONS = 20        # nombre de lignes simultanées max dans le portefeuille
BACKTEST_INITIAL_CAPITAL = 100_000.0
BACKTEST_COMMISSION_BPS = 5.0      # coût de transaction (aller simple), en points de base du notionnel
BACKTEST_SLIPPAGE_BPS = 5.0        # glissement d'exécution estimé (aller simple), en points de base

# Un signal DCF (10-K annuel) n'est considéré comme une base valable pour une
# NOUVELLE entrée que s'il a été publié il y a moins de ce nombre de jours ;
# au-delà, il est traité comme périmé (pas de nouveau 10-K depuis plus d'un
# an = donnée trop ancienne pour justifier un achat aujourd'hui). > 365 pour
# tolérer le glissement habituel de quelques semaines de la date de dépôt
# d'une année sur l'autre. N'affecte PAS les positions déjà ouvertes (elles
# restent gelées jusqu'à stop-loss/take-profit, voir backtest/engine.py).
BACKTEST_SIGNAL_MAX_AGE_DAYS = 400

# ----------------------------------------------------------------------------
# Paramètres par défaut de la stratégie OPTIONS (backtest/options_engine.py)
# ----------------------------------------------------------------------------
# Reprend le même seuil d'entrée que 08 (écart significatif = ±20%), mais ici
# directionnel : call si sous-évalué, put si survalué (voir
# backtest/strategies/valuation_gap_options.py).
OPTIONS_ENTRY_THRESHOLD_PCT = VALUATION_GAP_THRESHOLD_PCT

# Contrat visé : mêmes valeurs que MIN_DAYS_TO_EXPIRY / STRIKE_BAND_PCT dans
# 08_recuperation_options.py (dupliquées ici plutôt qu'importées : 08 charge
# ib_insync au niveau module, une dépendance dont le backtest n'a pas besoin
# pour son mode simulé). Garde ces deux jeux de constantes synchronisés si tu
# changes l'un des deux.
OPTIONS_TARGET_TENOR_DAYS = 270     # ~9 mois, échéance cible à l'entrée
OPTIONS_STRIKE_BAND_PCT = 0.30      # bande de strikes considérée autour du spot (ATM y compris)
OPTIONS_CONTRACT_MULTIPLIER = 100   # 1 contrat = 100 actions sous-jacentes (convention US)

# Stop-loss/take-profit exprimés en % de variation de la PRIME (pas du cours
# du sous-jacent comme pour la stratégie actions) : une option étant à effet
# de levier, ses seuils naturels sont beaucoup plus larges.
OPTIONS_STOP_LOSS_PCT = -50.0
OPTIONS_TAKE_PROFIT_PCT = 100.0
OPTIONS_MAX_POSITIONS = 20
OPTIONS_INITIAL_CAPITAL = 100_000.0

# Coûts par contrat (pas en bps du notionnel comme les actions : une option a
# un notionnel qui ne reflète pas son coût de transaction réel). ~0.65$/contrat
# est l'ordre de grandeur usuel des courtiers US ; le slippage est exprimé en %
# de la prime (les spreads bid/ask sur options sont larges, surtout hors ATM).
OPTIONS_COMMISSION_PER_CONTRACT = 0.65
OPTIONS_SLIPPAGE_PCT_OF_PREMIUM = 5.0

# Fenêtre de tolérance (en jours) pour rattacher un signal à un VRAI snapshot
# archivé par 08_recuperation_options.py (data/options/history/) plutôt que
# de simuler par Black-Scholes -- au-delà, le snapshot est jugé trop éloigné
# de la date du signal pour être représentatif.
OPTIONS_REAL_SNAPSHOT_TOLERANCE_DAYS = 14

# Fenêtre (en jours de cotation) de la volatilité réalisée utilisée comme
# proxy de la volatilité implicite pour le pricing Black-Scholes simulé,
# quand aucun snapshot réel n'est disponible (voir backtest/options_pricing.py).
OPTIONS_REALIZED_VOL_LOOKBACK_DAYS = 60


def to_ib_symbol(ric: str) -> str:
    """Convertit un RIC de l'univers vers le symbole IBKR utilisé comme
    colonne 'symbol' canonique dans TOUS les fichiers du pipeline (cours,
    options, financials). Centralisé ici (au lieu d'être dupliqué dans
    03/04/05) pour garantir que les trois scripts produisent exactement le
    même format de symbole et que les jointures sur 'symbol' fonctionnent
    pour les tickers à classes d'actions (ex: "BRK.B" -> "BRK B").
    """
    return SYMBOL_OVERRIDES.get(ric, ric.split(".")[0])
