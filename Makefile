# Raccourcis du pipeline. `make` seul affiche cette aide.
#
# Le pipeline reste pilotable script par script (voir README.md) : ce fichier
# ne fait qu'assembler les invocations courantes, il n'ajoute aucune logique.
# Chaque cible affiche la commande qu'elle lance, pour pouvoir la reprendre à
# la main avec d'autres options.

PYTHON  ?= python3
LIMIT   ?=
STRATEGY?= valuation_gap_multiples_options
MULTIPLE ?= EV/EBITDA
GROUPING ?= millesime
START   ?= 2015-01-01

# --limit n'est transmis que s'il est renseigné (make daily LIMIT=10).
LIMIT_ARG := $(if $(LIMIT),--limit $(LIMIT),)

.DEFAULT_GOAL := help
.PHONY: help daily daily-fast quarterly replay backtest backtest-actions audit \
        compare report test install universe bootstrap slippage merite

help:
	@echo "CalculRisque -- raccourcis disponibles"
	@echo
	@echo "  MISE A JOUR"
	@echo "    make daily         Mise a jour QUOTIDIENNE complete (cours + depots recents + signal)"
	@echo "    make daily-fast    Cours + recalcul du signal seulement (aucun appel SEC ni LLM)"
	@echo "    make quarterly     Rafraichissement TRIMESTRIEL (10-Q, 8-K, valorisation)"
	@echo "    make replay AS_OF=2024-06-30   Reconstitution point-in-time, hors ligne"
	@echo
	@echo "  BACKTEST ET ANALYSE"
	@echo "    make backtest      Backtest options (STRATEGY=$(STRATEGY), START=$(START))"
	@echo "    make backtest-actions   Backtest de la strategie actions (DCF)"
	@echo "    make audit         Relit le dernier run de backtest sans le relancer"
	@echo "    make compare       Compare les strategies options entre elles"
	@echo "    make slippage      Mesure le slippage reel sur les snapshots archives"
	@echo "    make merite        Le multiple merite predit-il mieux que le sectoriel ?"
	@echo "    make report        Dashboard Streamlit"
	@echo
	@echo "  DEVELOPPEMENT"
	@echo "    make test          Suite de tests"
	@echo "    make install       Dependances"
	@echo "    make bootstrap     Premier remplissage complet de data/ (long : plusieurs heures)"
	@echo
	@echo "  Variables : LIMIT=10  STRATEGY=...  START=AAAA-MM-JJ  MULTIPLE=P/E  GROUPING=secteur"

# ---------------------------------------------------------------------------
# Mise a jour
# ---------------------------------------------------------------------------

# La cible du cron quotidien. Voir la docstring de run_pipeline_daily.py pour
# ce qui est requis et ce qui degrade sans arreter le run.
daily:
	$(PYTHON) run_pipeline_daily.py $(LIMIT_ARG)

# Sans appel SEC ni LLM : rafraichit les cours et recalcule l'ecart de
# valorisation avec la valeur theorique deja connue. C'est le run le plus court
# qui met encore le signal du jour a jour.
daily-fast:
	$(PYTHON) run_pipeline_daily.py --prices-only $(LIMIT_ARG)

quarterly:
	$(PYTHON) run_pipeline_quarterly.py $(LIMIT_ARG)

# make replay AS_OF=2024-06-30
replay:
	@test -n "$(AS_OF)" || (echo "Renseigne AS_OF, ex: make replay AS_OF=2024-06-30" && false)
	$(PYTHON) run_pipeline_quarterly.py --as-of-date $(AS_OF)

# ---------------------------------------------------------------------------
# Backtest et analyse
# ---------------------------------------------------------------------------

backtest:
	$(PYTHON) 10_backtest_options.py --strategy $(STRATEGY) --start-date $(START)

backtest-actions:
	$(PYTHON) 09_backtest.py --strategy valuation_gap_dcf --start-date $(START)

# Relit les sorties du dernier run (quelques secondes) : couverture des
# signaux, theses reelles derriere le win-rate, sous-periodes glissantes.
audit:
	$(PYTHON) 14_audit_backtest.py

compare:
	$(PYTHON) compare_options_strategies.py

slippage:
	$(PYTHON) mesure_slippage_options.py

# Le multiple MERITE (regression sur fondamentaux) predit-il mieux que le
# multiple sectoriel ? Se tranche sans backtest, et decide s'il faut le brancher
# dans 06b. MULTIPLE=... et GROUPING=... pour explorer.
merite:
	$(PYTHON) 15_test_multiple_merite.py --multiple "$(MULTIPLE)" --grouping $(GROUPING)

report:
	streamlit run report/Home.py

# ---------------------------------------------------------------------------
# Developpement
# ---------------------------------------------------------------------------

test:
	$(PYTHON) -m pytest tests/ -q

install:
	$(PYTHON) -m pip install -r requirements.txt

universe:
	$(PYTHON) 01_build_universe.py
	$(PYTHON) 01b_historique_univers_sp500.py
	$(PYTHON) 02_categoriser_secteurs.py

# Premier remplissage de data/, dans l'ordre de dependance. L'univers COMPLET
# (sortie de 01b) est passe a 03b et 04 pour backfiller aussi les entreprises
# sorties de l'indice -- sans quoi le backtest retombe sur l'univers actuel
# applique retroactivement (biais de survivance, cf. README).
bootstrap: universe
	$(PYTHON) 03b_recuperation_cours_quotidiens.py --tickers data/universe/sp500_universe_full.csv
	$(PYTHON) 04_recuperation_10k.py --tickers data/universe/sp500_universe_full.csv
	$(PYTHON) run_pipeline_daily.py --skip-options
