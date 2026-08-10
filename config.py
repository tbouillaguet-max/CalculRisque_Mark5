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
from typing import Optional

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

# Taux sans risque PAR ANNÉE (rendement du Treasury 3 mois, moyenne annuelle).
#
# Un taux constant à 4% sur 2010-2026 est faux des deux côtés : le taux réel
# est allé de ~0,05% (2011-2015, après la crise) à ~5,3% (2023-2024). Sur une
# option à 9 mois, l'écart déplace la prime de plusieurs pour cent, et dans un
# sens systématique par période -- il surévaluait les calls du début de
# l'historique et sous-évaluait ceux de la fin. La même courbe sert de taux
# sans risque aux métriques (Sharpe, Sortino), où un taux de 4% appliqué à
# 2012 fabriquait une prime de risque négative sur une année pourtant positive.
#
# SOURCES : moyennes annuelles du 3-Month Treasury Bill (séries FRED DTB3 /
# TB3MS), arrondies au dixième de point. Elles n'ont PAS pu être re-vérifiées
# automatiquement dans l'environnement de rédaction (accès sortant bloqué) --
# recoupe-les si un chiffre te paraît douteux : https://fred.stlouisfed.org/series/TB3MS
# Une année absente de la table retombe sur RISK_FREE_RATE ci-dessus.
RISK_FREE_RATE_BY_YEAR: dict[int, float] = {
    2005: 0.0322, 2006: 0.0482, 2007: 0.0444, 2008: 0.0137, 2009: 0.0015,
    2010: 0.0014, 2011: 0.0005, 2012: 0.0009, 2013: 0.0006, 2014: 0.0003,
    2015: 0.0005, 2016: 0.0032, 2017: 0.0093, 2018: 0.0194, 2019: 0.0206,
    2020: 0.0037, 2021: 0.0004, 2022: 0.0202, 2023: 0.0515, 2024: 0.0521,
    2025: 0.0430, 2026: 0.0400,
}


def risk_free_rate_for(year: Optional[int] = None) -> float:
    """Taux sans risque de l'année demandée, avec repli sur la constante
    RISK_FREE_RATE (années hors table, ou appelant qui n'a pas de date sous
    la main -- le pricing de 08_recuperation_options.py, par exemple, ne
    valorise que des contrats du jour)."""
    if year is None:
        return RISK_FREE_RATE
    return RISK_FREE_RATE_BY_YEAR.get(int(year), RISK_FREE_RATE)


# Rendement du dividende par SECTEUR, utilisé par le pricing Black-Scholes du
# backtest (backtest/options_pricing.py, paramètre `q`).
#
# Black-Scholes sans dividende surévalue les CALLS et sous-évalue les PUTS,
# systématiquement -- et d'autant plus que le rendement est élevé, donc
# précisément sur les secteurs qu'un signal "value" sélectionne (utilities,
# télécoms, pétrole, banques). Le biais n'est donc pas aléatoire : il pousse la
# stratégie à surpayer ses calls sur exactement les titres qu'elle achète.
#
# Un rendement PAR TITRE et PAR DATE serait meilleur, mais le pipeline ne
# collecte aucune donnée de dividende (03/03b demandent whatToShow="TRADES" à
# IBKR, cf. la section "Biais et limites connus" du README) : ces moyennes
# sectorielles sont une approximation assumée, pas une mesure. Ordres de
# grandeur usuels du marché US.
SECTOR_DIVIDEND_YIELD: dict[str, float] = {
    "Technologie": 0.008,
    "Santé": 0.017,
    "Agro-alimentaire et boissons": 0.028,
    "Produits ménagers et de soin personnel": 0.025,
    "Services aux collectivités": 0.035,
    "Banques": 0.030,
    "Assurance": 0.022,
    "Services financiers": 0.020,
    "Immobilier": 0.040,
    "Pétrole et gaz": 0.035,
    "Biens et services industriels": 0.018,
    "Bâtiment et matériaux de construction": 0.015,
    "Matières premières": 0.020,
    "Chimie": 0.022,
    "Medias": 0.012,
    "Télécommunications": 0.045,
    "Distribution": 0.015,
    "Automobiles et équipementiers": 0.020,
    "Voyage et loisirs": 0.012,
    "_default": 0.018,
}


def dividend_yield_for(sector) -> float:
    """Rendement du dividende retenu pour un secteur (repli "_default")."""
    return SECTOR_DIVIDEND_YIELD.get(
        sector if isinstance(sector, str) else "", SECTOR_DIVIDEND_YIELD["_default"],
    )

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

# Plafond de concentration : part maximale du portefeuille pour UNE ligne,
# quel que soit son écart de valorisation. Les stratégies pondèrent au prorata
# de l'écart ; sans plafond, un écart aberrant (valeur théorique proche de
# zéro -> plusieurs milliers de %) capte à lui seul l'essentiel du capital.
# Le NOMBRE de positions n'étant pas plafonné, c'est le seul garde-fou de
# concentration du portefeuille. 0 ou None le désactive.
BACKTEST_MAX_WEIGHT_PER_POSITION_PCT = 20.0

# Filtre momentum : une entreprise dont le cours a chuté de plus de X% sur les
# 12 derniers mois (dernier mois exclu, cf. PricePanel.momentum_12_1) n'est
# plus éligible à une NOUVELLE entrée, même si son écart de valorisation est
# large. C'est le garde-fou classique de la "value trap" : un écart qui
# s'élargit parce que le marché intègre une dégradation que les derniers
# états financiers publiés ne montrent pas encore.
# None désactive le filtre (0.0 est un seuil valide : "aucune baisse tolérée").
BACKTEST_MOMENTUM_MIN_PCT = -10.0
# Capital simulé au départ des backtests (actions et options : voir
# OPTIONS_INITIAL_CAPITAL, tenu à la même valeur -- c'est le même
# portefeuille selon qu'on l'investit en actions ou en options).
BACKTEST_INITIAL_CAPITAL = 1_000_000.0
BACKTEST_COMMISSION_BPS = 5.0      # coût de transaction (aller simple), en points de base du notionnel
BACKTEST_SLIPPAGE_BPS = 5.0        # glissement d'exécution estimé (aller simple), en points de base

# Un signal DCF (10-K annuel) n'est considéré comme une base valable pour une
# NOUVELLE entrée que s'il a été publié il y a moins de ce nombre de jours ;
# au-delà, il est traité comme périmé (pas de nouveau 10-K depuis plus d'un
# an = donnée trop ancienne pour justifier un achat aujourd'hui). > 365 pour
# tolérer le glissement habituel de quelques semaines de la date de dépôt
# d'une année sur l'autre. N'affecte PAS les positions déjà ouvertes (elles
# restent gelées jusqu'à stop-loss/take-profit, voir backtest/engine.py).
# Indice de référence auquel 09/10 comparent la stratégie (metrics.py). Doit
# être un symbole présent dans DAILY_PRICES_FILE (03b) : à défaut, un indice
# ÉQUIPONDÉRÉ de l'univers point-in-time est reconstruit à la volée, repère
# utile mais différent du S&P 500 pondéré (voir
# backtest/data_loader.build_benchmark_series).
BENCHMARK_SYMBOL = "SPY"

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

# Contrat visé. STRIKE_BAND_PCT reprend la valeur de 08_recuperation_options.py
# (dupliquée ici plutôt qu'importée : 08 charge ib_insync au niveau module, une
# dépendance dont le backtest n'a pas besoin pour son mode simulé) ; garde les
# deux synchronisées si tu changes l'une des deux. MIN_DAYS_TO_EXPIRY côté 08
# est un PLANCHER de collecte (au moins 9 mois), pas une cible : les échéances
# 2 ans visées ici sont donc bien dans ce qu'il archive.
#
# Échéance cible à l'entrée : 2 ans, comme valuation_gap_multiples_options. Une
# convergence vers la valeur théorique demande des trimestres, pas des
# semaines ; un contrat 9 mois obligeait à avoir raison ET vite, et faisait
# subir l'accélération de la perte de valeur temps en fin de vie.
OPTIONS_TARGET_TENOR_DAYS = 730
# Point de décision, à 9 mois de l'échéance : la position est réexaminée à
# l'aune du signal courant. Écart toujours au-dessus du seuil d'entrée -> le
# contrat est roulé sur une nouvelle échéance pleine, à exposition inchangée ;
# écart repassé sous le seuil (ou retourné de sens) -> la position est
# clôturée. On ne détient donc jamais un contrat sur sa dernière année de vie,
# là où la valeur temps s'érode le plus vite.
OPTIONS_ROLL_WHEN_DAYS_LEFT = 270
OPTIONS_STRIKE_BAND_PCT = 0.30      # bande de strikes considérée autour du spot (ATM y compris)
OPTIONS_CONTRACT_MULTIPLIER = 100   # 1 contrat = 100 actions sous-jacentes (convention US)

# Base de mesure des stop-loss/take-profit : "underlying" -> variation du COURS
# DU SOUS-JACENT (comme la stratégie actions), "premium" -> variation de la
# prime de l'option.
#
# Le sous-jacent est retenu par défaut depuis le passage des entrées à 2 ans.
# Adossés à la prime, les seuils étaient atteints par la seule érosion de la
# valeur temps : une option ATM à 2 ans perd de l'ordre de 20% de sa prime en
# ~15 mois à cours STRICTEMENT INCHANGÉ -- un stop à -20% sortait donc des
# positions dont la thèse n'avait pas bougé, avant que la convergence visée ait
# eu le temps de se produire. L'effet de levier jouait dans le même sens : sur
# la prime, -20% correspond à une baisse du titre de quelques points seulement.
# Mesurés sur le cours, les seuils décrivent bien le scénario voulu ("le titre
# a baissé de 20%" / "le titre a monté de 80%"). --stop-basis premium rétablit
# l'ancienne base.
OPTIONS_STOP_BASIS = "underlying"

# Seuils, appliqués à la base ci-dessus et ORIENTÉS dans le sens de la position :
# pour un PUT, "le titre monte de 20%" est la perte et "le titre baisse de 80%"
# le gain (voir options_engine._position_move_pct). Encadrement asymétrique
# assumé : on coupe vite une thèse qui se dégrade, on laisse courir celle qui
# marche -- un take-profit à +80% du sous-jacent est un mouvement rare sur 2
# ans, c'est bien l'intention (la sortie normale d'une position gagnante est le
# réexamen à 9 mois, pas le take-profit).
OPTIONS_STOP_LOSS_PCT = -20.0
OPTIONS_TAKE_PROFIT_PCT = 80.0
# Même capital que le backtest actions. Cette valeur n'est PAS neutre pour la
# stratégie options depuis le passage aux contrats entiers
# (OPTIONS_WHOLE_CONTRACTS) : un contrat vaut 100 x la prime, soit ~1 000$ pour
# une option à 10$, et une position visée plus petite que cela n'est tout
# simplement pas prenable. Mesuré sur 20 positions simultanées : à 100 000$,
# 41% des positions visées tombaient sous le contrat unique et étaient
# abandonnées, et celles qui passaient étaient déformées de ~80% par l'arrondi
# -- le backtest mesurait alors la stratégie bridée par la taille minimale,
# pas la stratégie. À 1 000 000$, plus aucune position n'est perdue et l'écart
# d'arrondi médian tombe à ~4%. Le nombre de positions n'étant plus plafonné,
# ce seuil se déplace avec le nombre de candidates retenues : plus elles sont
# nombreuses, plus chacune est petite, et plus le capital doit être élevé pour
# que l'arrondi au contrat entier reste négligeable.
OPTIONS_INITIAL_CAPITAL = 1_000_000.0

# Coûts par contrat (pas en bps du notionnel comme les actions : une option a
# un notionnel qui ne reflète pas son coût de transaction réel). ~0.65$/contrat
# est l'ordre de grandeur usuel des courtiers US ; le slippage est exprimé en %
# de la prime (les spreads bid/ask sur options sont larges, surtout hors ATM).
OPTIONS_COMMISSION_PER_CONTRACT = 0.65
OPTIONS_SLIPPAGE_PCT_OF_PREMIUM = 5.0

# Minimum FACTURÉ PAR ORDRE (pas par contrat) : IBKR applique un taux par
# contrat mais jamais moins de 1,00$ par ordre. Un ordre d'un seul contrat
# coûte donc 1,00$ -- et une stratégie qui multiplie les petits ordres paie
# nettement plus que "nb_contrats x taux". Modéliser la commission comme un
# simple taux par contrat sous-estimait donc structurellement les frais.
# Ce minimum porte sur la COMMISSION seule ; les frais tiers ci-dessous
# s'ajoutent par-dessus.
OPTIONS_COMMISSION_MIN_PER_ORDER = 1.0

# Grille dégressive IBKR (options US), relevée sur la tarification officielle.
# Le taux dépend du VOLUME MENSUEL cumulé en contrats ET du niveau de PRIME.
#   (volume_mensuel_max, ((prime_max, taux), ...))
# volume_mensuel_max=None -> dernier palier ; prime_max=None -> "toutes les
# primes au-dessus". Une prime STRICTEMENT inférieure à prime_max prend ce taux.
#
# Note : contrairement aux actions, cette grille n'a PAS de plafond en % de la
# valeur négociée -- vérifié sur les exemples officiels (5 contrats à 0,03$ de
# prime = 15$ de valeur, facturés 1,25$, soit 8,3% de la valeur).
OPTIONS_COMMISSION_TIERS = (
    (10_000,  ((0.05, 0.25), (0.10, 0.50), (None, 0.65))),
    (50_000,  ((0.05, 0.25), (None, 0.50))),
    (100_000, ((None, 0.25),)),
    (None,    ((None, 0.15),)),
)

# Frais tiers, facturés EN PLUS de la commission (non soumis au minimum par
# ordre). Par contrat, des deux côtés (achat comme vente).
OPTIONS_FEE_ORF_PER_CONTRACT = 0.02295   # Options Regulatory Fee
OPTIONS_FEE_CAT_PER_CONTRACT = 0.0003    # FINRA Consolidated Audit Trail
OPTIONS_FEE_OCC_PER_CONTRACT = 0.025     # compensation OCC
# À la VENTE uniquement (frais réglementaires sur les cessions).
OPTIONS_FEE_FINRA_TAF_PER_CONTRACT = 0.00329   # FINRA Trading Activity Fee
OPTIONS_FEE_SEC_PCT_OF_SALE = 0.0000206        # x valeur de la vente

# Part MINIMALE du portefeuille investie en primes, en % du NAV. En dessous,
# le moteur renforce les positions déjà ouvertes (au prorata de leur taille)
# pour remettre le capital au travail plutôt que de le laisser dormir.
#
# CE RÉGLAGE PILOTE UN ARBITRAGE À TROIS TERMES -- cash, theta, levier :
#   - trop BAS : le capital dort, et le rendement du portefeuille est celui
#     d'une petite poche investie noyée dans du cash non rémunéré ;
#   - trop HAUT : c'est la totalité du capital qui paie la perte de valeur
#     temps (theta) tous les jours, et surtout le LEVIER explose. Le moteur
#     dimensionne en exposition NOTIONNELLE delta-équivalente (nb_contrats =
#     budget / (|delta| x spot x multiplicateur)), ce qui vise une exposition
#     delta d'environ 1x le NAV. Or la prime d'une option ATM à 9 mois ne vaut
#     que ~8 à 12% du spot : forcer 90% du NAV en primes revenait donc à
#     porter une exposition delta de l'ordre de 8 à 10x le NAV -- cause
#     structurelle des drawdowns extrêmes observés, et annulation pure et
#     simple du dimensionnement par delta.
#
# La valeur était 90 ; ramenée à 25, complétée par le plafond explicite
# d'exposition delta ci-dessous qui borne le levier quoi qu'il arrive.
# 0 ou None désactive le redéploiement.
OPTIONS_MIN_DEPLOYMENT_PCT = 25.0

# Exposition NOTIONNELLE delta-équivalente maximale du portefeuille d'options,
# en % du NAV : somme sur les positions de |delta| x spot x contrats x
# multiplicateur. C'est la mesure honnête du levier -- combien de dollars de
# sous-jacent le portefeuille suit réellement, par opposition au montant de
# primes décaissé, qui n'en est qu'une fraction.
#
# 100% = le portefeuille bouge comme s'il détenait son NAV en actions. Au-delà
# de ce plafond, le redéploiement du cash oisif s'arrête, même si la part
# investie en primes reste sous OPTIONS_MIN_DEPLOYMENT_PCT : c'est le plafond
# qui prime, le plancher n'étant qu'une préférence. 0 ou None le désactive
# (comportement d'avant : levier non borné).
#
# PORTÉE : ce plafond contraint l'ORDRE au moment où il est passé, pas la
# position dans la durée. Entre deux renforcements, le delta du contrat dérive
# avec le sous-jacent (gamma) et le moteur ne vend JAMAIS pour se désendetter
# -- le levier réalisé peut donc dépasser le plafond de quelques dizaines de
# points de NAV. Suivre la colonne delta_notional_pct de l'equity_curve pour
# le constater sur un run donné.
OPTIONS_MAX_DELTA_NOTIONAL_PCT = 100.0

# Optimisation de taille au regard des frais.
#
# Le minimum PAR ORDRE fait qu'un tout petit ordre paie un tarif par contrat
# bien supérieur au tarif affiché : à 0,65$/contrat, 1 contrat coûte 1,00$
# (soit 1,00$/contrat) alors que 2 contrats coûtent 1,30$ (0,65$/contrat).
# Monter à la taille où le minimum cesse de mordre améliore donc le coût
# unitaire -- mais au prix d'une exposition en plus, et l'arithmétique est
# brutale : +100% d'exposition (1 -> 2 contrats) pour économiser 0,35$, jusqu'à
# +600% au palier 0,15$/contrat. Cette remontée n'est donc appliquée que si
# l'écart à l'exposition VISÉE reste sous la tolérance ci-dessous. 0 la
# désactive complètement.
OPTIONS_FEE_BUMP_MAX_EXTRA_PCT = 20.0

# Garde-fou inverse, et de loin le plus rentable : un ordre d'ENTRÉE ou de
# renforcement dont les frais dépassent ce pourcentage de sa propre valeur
# est purement abandonné -- il détruit plus de valeur qu'il n'en apporte.
# Ne s'applique JAMAIS aux sorties (stop-loss, take-profit, expiration,
# roulement) : une position doit pouvoir être fermée quel qu'en soit le coût.
# 0 ou None désactive le garde-fou.
OPTIONS_MAX_FEE_PCT_OF_TRADE = 1.0

# Les options se négocient par contrats ENTIERS. Le moteur dimensionnait en
# contrats fractionnaires (0,931 contrat...), ce qui n'existe pas et fausse
# doublement les frais : la commission par contrat était appliquée au prorata,
# et le minimum par ordre n'existait pas. False rétablit l'ancien
# comportement fractionnaire (utile seulement pour comparer).
OPTIONS_WHOLE_CONTRACTS = True

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
# Variante LONG TERME de la stratégie options. L'échéance 2 ans, le roulement à
# 9 mois et les stops mesurés sur le cours du sous-jacent sont désormais le
# comportement par défaut du moteur (OPTIONS_TARGET_TENOR_DAYS,
# OPTIONS_ROLL_WHEN_DAYS_LEFT, OPTIONS_STOP_BASIS) : elle ne s'en distingue
# plus. Ce qui la sépare de valuation_gap_options (ci-dessus) tient à quatre
# points, tous voulus :
#   1. signal = multiples sectoriels SEULS (les lignes en repli DCF de 06b sont
#      écartées) ;
#   2. écart rapporté à la valeur THÉORIQUE, pas au cours (voir plus bas) ;
#   3. strike à mi-chemin entre valeur théorique et cours, pas ATM (pari sur une
#      convergence progressive, pas sur un mouvement immédiat) ;
#   4. seuils de stop plus resserrés (-25%/+30% contre -20%/+80%), et écart
#      refermé vendu au rebalancement suivant au lieu d'attendre le réexamen
#      de roulement.
OPTIONS_MULTIPLES_ENTRY_THRESHOLD_PCT = 20.0

# Base de calcul de l'écart : "theoretical" -> (théorique - cours)/théorique,
# "close" -> (théorique - cours)/cours (convention historique de gap_pct en
# 06b, utilisée par valuation_gap_options). Les deux ne sélectionnent pas les
# mêmes entreprises : à 20%, théo=120/cours=100 donne +16,7% en base théorique
# (écarté) contre +20,0% en base cours (retenu).
OPTIONS_MULTIPLES_GAP_BASIS = "theoretical"

# Appliqués au COURS DU SOUS-JACENT (comme OPTIONS_STOP_LOSS_PCT depuis le
# passage aux entrées 2 ans, cf. OPTIONS_STOP_BASIS), et orientés dans le sens
# de la position : pour un PUT, "le titre monte de 25%" est la perte et "le
# titre baisse de 30%" le gain (voir options_engine._position_move_pct). Plus
# resserrés que ceux de valuation_gap_options : le strike hors de la monnaie
# rend la thèse plus fragile à un mouvement adverse.
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


# Secteurs pour lesquels un DCF de type FCFF n'a PAS de sens, et que
# 07_calcul_dcf.py écarte donc explicitement.
#
# Le FCFF part de l'EBIT et traite la dette comme un financement à retrancher
# en fin de calcul. Pour une banque ou un assureur, la dette est un INTRANT DU
# MÉTIER (les dépôts et les provisions techniques financent l'actif) et l'EBIT
# n'est pas une mesure opérationnelle pertinente : les valoriser ainsi produit
# un chiffre qui a l'air d'un DCF sans en être un. Pour une foncière, l'essentiel
# du résultat est absorbé par des amortissements sans contrepartie de trésorerie
# et le capex se confond avec l'acquisition d'actifs -- le FCFF n'y décrit rien
# non plus.
#
# Jusqu'ici, il suffisait qu'OperatingIncomeLoss soit tagué pour qu'un chiffre
# sorte quand même. Ces entreprises ne perdent rien au change :
# 06b_calcul_valorisation_combinee.py les valorise déjà par les multiples
# sectoriels que SECTOR_MULTIPLES juge pertinents pour elles (P/E pour les
# financières, EV/EBITDA pour l'immobilier) -- et les multiples, eux, n'ont pas
# besoin de l'EBIT.
#
# Libellés de 02_categoriser_secteurs.py (français), comme SECTOR_DCF_PARAMS.
SECTORS_SANS_DCF: tuple = ("Banques", "Assurance", "Services financiers", "Immobilier")

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


# ----------------------------------------------------------------------------
# Inflation (ajustement des écarts de valorisation)
# ----------------------------------------------------------------------------
# Inflation annuelle US (CPI-U, moyenne annuelle), en %.
#
# SOURCES : 2020-2025 recoupés en ligne (macrotrends, usinflationcalculator) ;
# les années antérieures viennent des moyennes annuelles BLS bien établies mais
# n'ont PAS pu être re-vérifiées automatiquement (bls.gov et les agrégateurs
# renvoient 403 aux requêtes automatisées). Recoupe-les si un chiffre te paraît
# douteux : https://www.bls.gov/cpi/ -> "Historical CPI-U".
INFLATION_BY_YEAR: dict[int, float] = {
    2005: 3.39, 2006: 3.23, 2007: 2.85, 2008: 3.84, 2009: -0.36,
    2010: 1.64, 2011: 3.16, 2012: 2.07, 2013: 1.46, 2014: 1.62,
    2015: 0.12, 2016: 1.26, 2017: 2.13, 2018: 2.44, 2019: 1.81,
    2020: 1.23, 2021: 4.70, 2022: 8.00, 2023: 4.12, 2024: 2.95,
    2025: 2.70,
}
# Année absente de la table (avant 2005, ou plus récente que la dernière mise
# à jour) : valeur de repli, proche de la cible de la Fed.
INFLATION_DEFAULT_PCT = 2.0

# Ajuste l'écart de valorisation de l'inflation ANTICIPÉE sur l'horizon de
# convergence de la stratégie. Voir backtest/strategies/base.inflation_adjusted_gap
# pour le raisonnement : la valeur théorique est une grandeur NOMINALE, elle
# inflate donc avec le temps, et la convergence se fait vers cette valeur
# inflatée -- ce qui aide une position acheteuse et pénalise une vendeuse.
INFLATION_ADJUST_GAP = True

# Horizon de convergence retenu pour la stratégie ACTIONS (les stratégies
# options utilisent leur propre échéance de contrat, qui est leur horizon réel).
INFLATION_HORIZON_YEARS_STOCKS = 1.0


def inflation_known_at(date) -> float:
    """Inflation annuelle (en %) CONNUE à cette date, donc celle de l'année
    civile PRÉCÉDENTE : la moyenne annuelle d'une année n'est publiée qu'une
    fois l'année terminée. Utiliser l'inflation de l'année en cours
    introduirait un look-ahead -- exactement le biais que tout le reste du
    pipeline s'attache à éviter."""
    import pandas as pd  # local : garde config.py importable sans pandas

    timestamp = pd.Timestamp(date)
    if pd.isna(timestamp):
        return INFLATION_DEFAULT_PCT
    return INFLATION_BY_YEAR.get(timestamp.year - 1, INFLATION_DEFAULT_PCT)


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
