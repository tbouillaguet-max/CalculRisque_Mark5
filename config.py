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

# Plafond de concentration : part maximale du portefeuille pour UNE ligne,
# quel que soit son écart de valorisation. Les stratégies pondèrent au prorata
# de l'écart ; sans plafond, un écart aberrant (valeur théorique proche de
# zéro -> plusieurs milliers de %) capte à lui seul l'essentiel du capital.
# 20% = 4x le poids équipondéré à 20 positions. 0 ou None désactive le plafond.
BACKTEST_MAX_WEIGHT_PER_POSITION_PCT = 20.0

# Filtre momentum : une entreprise dont le cours a chuté de plus de X% sur les
# 12 derniers mois (dernier mois exclu, cf. PricePanel.momentum_12_1) n'est
# plus éligible à une NOUVELLE entrée, même si son écart de valorisation est
# large. C'est le garde-fou classique de la "value trap" : un écart qui
# s'élargit parce que le marché intègre une dégradation que les derniers
# états financiers publiés ne montrent pas encore.
# None désactive le filtre (0.0 est un seuil valide : "aucune baisse tolérée").
BACKTEST_MOMENTUM_MIN_PCT = -10.0
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

# Durée de vie d'un signal selon le type de période qui l'a produit : un TTM
# trimestriel (04b) est remplacé par un nouveau ~90 jours plus tard, alors
# qu'un exercice annuel (10-K) reste la dernière information publiée pendant
# ~12 mois. Garder 400 jours pour les deux laissait un signal trimestriel
# actif bien après avoir été démenti par le trimestre suivant.
# BACKTEST_SIGNAL_MAX_AGE_DAYS reste le repli quand le period_type est inconnu.
BACKTEST_SIGNAL_MAX_AGE_DAYS_BY_PERIOD = {"FY": 270, "TTM": 120}

# Verdicts de 07b_validation_qualitative.py qui DISQUALIFIENT un signal dans
# les backtests (voir backtest/data_loader.apply_qualitative_gate). Vide ->
# filtre désactivé. "non_evalue" ne doit PAS y figurer : c'est la valeur prise
# par toutes les périodes quand MISTRAL_API_KEY n'est pas définie.
QUALITATIVE_GATE_EXCLUDED_VERDICTS = ("contradictoire",)

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

# Chaque dépôt de filing (10-K/10-Q) d'UNE entreprise déclenche un rebalancement
# qui recalcule les poids de TOUTES les positions détenues (renormalisation à
# somme=1 après plafonnement, cf. options_engine._rebalance). Sur 503
# entreprises et des filings trimestriels, ça met en file un micro-ajustement
# de resize sur chaque position ouverte à chaque événement -- des centaines
# par an -- alors que MIN_TRADE_DOLLAR (1$) ne bloque que les montants
# absolument négligeables, pas les resizes proportionnellement mineurs sur
# des positions de plusieurs milliers de dollars.
# Une position déjà ouverte n'est resize QUE si le changement dépasse ce
# pourcentage de sa valeur actuelle ; en dessous, elle reste gelée à sa taille
# actuelle (comme si le rebalancement n'avait pas eu lieu pour elle -- une
# NOUVELLE position n'est jamais concernée). None ou 0 désactive le filtre
# (comportement historique : tout changement, même infime, déclenche un ordre).
OPTIONS_MIN_RESIZE_RELATIVE_PCT = 15.0

# ----------------------------------------------------------------------------
# Stratégie options "multiples" (backtest/strategies/valuation_gap_multiples_options.py)
# ----------------------------------------------------------------------------
# Variante LONG TERME de la stratégie options, distincte de
# valuation_gap_options (ci-dessus) sur quatre points, tous voulus :
#   1. signal = multiples sectoriels SEULS (les lignes en repli DCF de 06b sont
#      écartées) ;
#   2. écart rapporté à la valeur THÉORIQUE, pas au cours (voir plus bas) ;
#   3. strike à mi-chemin entre valeur théorique et cours, échéance 2 ans
#      (pari sur une convergence progressive, pas sur un mouvement immédiat) ;
#   4. stop-loss/take-profit sur le COURS DU SOUS-JACENT, pas sur la prime.
#
# Le point 4 n'est pas un détail : sur une option 2 ans hors de la monnaie, le
# levier est d'environ 3,5x et la seule érosion du temps fait perdre ~29% de la
# prime en 9 mois à cours inchangé. Des seuils -25%/+30% appliqués à la prime
# clôtureraient donc la position avant que la convergence visée ait le temps de
# se produire ; appliqués au cours du sous-jacent, ils décrivent bien le
# scénario voulu ("le titre a baissé de 25%" / "le titre a monté de 30%").
OPTIONS_MULTIPLES_ENTRY_THRESHOLD_PCT = 20.0

# Base de calcul de l'écart : "theoretical" -> (théorique - cours)/théorique,
# "close" -> (théorique - cours)/cours (convention historique de gap_pct en
# 06b, utilisée par valuation_gap_options). Les deux ne sélectionnent pas les
# mêmes entreprises : à 20%, théo=120/cours=100 donne +16,7% en base théorique
# (écarté) contre +20,0% en base cours (retenu).
OPTIONS_MULTIPLES_GAP_BASIS = "theoretical"

# Appliqués au COURS DU SOUS-JACENT (voir plus haut), et orientés dans le sens
# de la position : pour un PUT, "le titre monte de 25%" est la perte et "le
# titre baisse de 30%" le gain (voir options_engine._check_stop_loss_take_profit).
OPTIONS_MULTIPLES_STOP_LOSS_PCT = -25.0
OPTIONS_MULTIPLES_TAKE_PROFIT_PCT = 30.0

# Échéance visée à l'entrée, et seuil de roulement : à 9 mois de l'échéance, la
# position est clôturée et rouverte sur une nouvelle échéance 2 ans (au strike
# recalculé avec la valorisation théorique la plus récente), tant que l'écart
# reste au-dessus du seuil d'entrée. Évite de subir l'accélération de la perte
# de valeur temps sur la dernière année de vie du contrat.
OPTIONS_MULTIPLES_TENOR_DAYS = 730
OPTIONS_MULTIPLES_ROLL_WHEN_DAYS_LEFT = 270

# Plafond appliqué à l'écart QUAND IL SERT À DIMENSIONNER une position (pas
# quand il sert à sélectionner : le classement reste fait sur l'écart brut).
# En base "theoretical", l'écart est borné à +100% du côté sous-évalué mais
# NON borné du côté survalorisé -- une valeur théorique proche de zéro donne
# un écart de plusieurs milliers de %, et une seule ligne capterait alors
# l'essentiel du capital. None ou 0 désactive le plafond.
OPTIONS_MULTIPLES_WEIGHT_CAP_PCT = 100.0


# ----------------------------------------------------------------------------
# Hypothèses DCF par SECTEUR (07_calcul_dcf.py)
# ----------------------------------------------------------------------------
# Un WACC unique pour toutes les entreprises est un biais sectoriel, pas un
# signal : 10% surestime le coût du capital d'une utility régulée (dette bon
# marché, flux stables) et le sous-estime pour une techno. Toutes les valeurs
# stables ressortaient donc "sous-évaluées" en permanence.
#
# Les clés sont les secteurs tels que 02_categoriser_secteurs.py les écrit
# (SECTEURS, en français) -- PAS les libellés GICS anglais : une clé qui ne
# correspond à rien retomberait silencieusement sur "_default" et le
# paramétrage sectoriel n'aurait aucun effet. Ordres de grandeur usuels
# (Damodaran, secteur US), à ajuster si tu as mieux.
SECTOR_DCF_PARAMS: dict[str, dict] = {
    "Technologie":                          {"wacc": 0.100, "fcf_growth": 0.07, "terminal_growth": 0.030},
    "Santé":                                {"wacc": 0.090, "fcf_growth": 0.06, "terminal_growth": 0.025},
    "Agro-alimentaire et boissons":         {"wacc": 0.070, "fcf_growth": 0.03, "terminal_growth": 0.020},
    "Produits ménagers et de soin personnel": {"wacc": 0.070, "fcf_growth": 0.03, "terminal_growth": 0.020},
    "Services aux collectivités":           {"wacc": 0.065, "fcf_growth": 0.02, "terminal_growth": 0.015},
    "Banques":                              {"wacc": 0.090, "fcf_growth": 0.04, "terminal_growth": 0.020},
    "Assurance":                            {"wacc": 0.090, "fcf_growth": 0.04, "terminal_growth": 0.020},
    "Services financiers":                  {"wacc": 0.090, "fcf_growth": 0.04, "terminal_growth": 0.020},
    "Immobilier":                           {"wacc": 0.070, "fcf_growth": 0.03, "terminal_growth": 0.020},
    "Pétrole et gaz":                       {"wacc": 0.100, "fcf_growth": 0.03, "terminal_growth": 0.015},
    "Biens et services industriels":        {"wacc": 0.090, "fcf_growth": 0.04, "terminal_growth": 0.020},
    "Bâtiment et matériaux de construction": {"wacc": 0.090, "fcf_growth": 0.04, "terminal_growth": 0.020},
    "Matières premières":                   {"wacc": 0.090, "fcf_growth": 0.04, "terminal_growth": 0.020},
    "Chimie":                               {"wacc": 0.090, "fcf_growth": 0.04, "terminal_growth": 0.020},
    "Medias":                               {"wacc": 0.090, "fcf_growth": 0.05, "terminal_growth": 0.025},
    "Télécommunications":                   {"wacc": 0.090, "fcf_growth": 0.05, "terminal_growth": 0.025},
    "Distribution":                         {"wacc": 0.095, "fcf_growth": 0.05, "terminal_growth": 0.025},
    "Automobiles et équipementiers":        {"wacc": 0.095, "fcf_growth": 0.05, "terminal_growth": 0.025},
    "Voyage et loisirs":                    {"wacc": 0.095, "fcf_growth": 0.05, "terminal_growth": 0.025},
    # Secteur inconnu, "indetermine" (02) ou absent de l'univers.
    "_default":                             {"wacc": 0.100, "fcf_growth": 0.05, "terminal_growth": 0.020},
}


# ----------------------------------------------------------------------------
# Multiples pertinents par SECTEUR (06b_calcul_valorisation_combinee.py)
# ----------------------------------------------------------------------------
# Appliquer les trois mêmes multiples à tous les secteurs mélange des mesures
# qui n'ont pas de sens partout : une banque n'a pas d'EBITDA ni de chiffre
# d'affaires comparable à celui d'un industriel (structure de bilan
# différente), et le P/E d'une foncière est écrasé par les amortissements.
# Mêmes clés que SECTOR_DCF_PARAMS (libellés de 02, en français).
SECTOR_MULTIPLES: dict[str, list] = {
    "Banques": ["P/E"],
    "Assurance": ["P/E"],
    "Services financiers": ["P/E"],
    "Immobilier": ["EV/EBITDA"],
    "Services aux collectivités": ["EV/EBITDA", "EV/Sales"],
    "_default": ["EV/EBITDA", "EV/Sales", "P/E"],
}

# Bornes de plausibilité appliquées AVANT le calcul de la médiane sectorielle :
# une entreprise sortant d'une perte affiche un P/E à plusieurs centaines de x
# et déforme la médiane du secteur, surtout sur un groupe de quelques pairs.
MULTIPLE_PLAUSIBLE_RANGE: dict[str, tuple] = {
    "EV/EBITDA": (0.0, 50.0),
    "EV/Sales": (0.0, 20.0),
    "P/E": (0.0, 60.0),
}


def to_ib_symbol(ric: str) -> str:
    """Convertit un RIC de l'univers vers le symbole IBKR utilisé comme
    colonne 'symbol' canonique dans TOUS les fichiers du pipeline (cours,
    options, financials). Centralisé ici (au lieu d'être dupliqué dans
    03/04/05) pour garantir que les trois scripts produisent exactement le
    même format de symbole et que les jointures sur 'symbol' fonctionnent
    pour les tickers à classes d'actions (ex: "BRK.B" -> "BRK B").
    """
    return SYMBOL_OVERRIDES.get(ric, ric.split(".")[0])


def to_naive_day(values):
    """Normalise une colonne de dates en datetime64[us] naïf, tronqué au jour.

    À utiliser des DEUX côtés de toute jointure sur une date, pour la même
    raison que to_ib_symbol pour les jointures sur 'symbol' : les fichiers du
    pipeline ne stockent pas tous leurs dates de la même façon. 03b écrit des
    datetime.date (stockés en date32 dans le Parquet, relus en
    datetime64[s]), alors que 'filed_date' est une chaîne "YYYY-MM-DD" venant
    de la SEC (04/04b), parsée en datetime64[us] par pandas >= 3.

    pandas.merge_asof exige des clés STRICTEMENT du même dtype et refuse ce
    mélange de résolutions ("MergeError: incompatible merge keys ... must be
    the same type"), là où un merge classique le tolère silencieusement.
    """
    import pandas as pd  # local : garde config.py importable sans pandas

    series = pd.to_datetime(pd.Series(values), errors="coerce")
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        series = series.dt.tz_convert("UTC").dt.tz_localize(None)
    return series.dt.normalize().astype("datetime64[us]")
