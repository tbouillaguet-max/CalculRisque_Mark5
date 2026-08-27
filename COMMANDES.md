# Toutes les commandes du dépôt

Référence exhaustive : chaque script exécutable, ce qu'il fait, ce qu'il lit, ce
qu'il écrit, et ses options. Le *pourquoi* des choix de conception est dans
[README.md](README.md) ; ce fichier-ci est l'aide-mémoire opérationnel.

> **Toutes les commandes se lancent depuis la RACINE du dépôt**, jamais depuis un
> sous-dossier. `config.py` résout ses chemins relativement au répertoire
> d'exécution (`./data/...`) : lancé d'ailleurs, un script écrira ses sorties
> dans un `data/` parallèle sans prévenir.

---

## Sommaire

| | |
|---|---|
| [Installation](#installation) | dépendances, clés API |
| [Démarrage rapide](#démarrage-rapide) | le chemin le plus court vers un premier backtest |
| [1. Univers et secteurs](#1-univers-et-secteurs) | `01` `01b` `02` |
| [2. Cours de bourse](#2-cours-de-bourse) | `03` `03b` |
| [3. Données financières SEC](#3-données-financières-sec) | `04` `04b` `04c` |
| [4. Valorisation](#4-valorisation) | `05` `06` `06b` `07` `07b` |
| [5. Chaînes d'options](#5-chaînes-doptions) | `08` |
| [6. Backtests](#6-backtests) | `09` `10` |
| [7. Optimisation](#7-optimisation) | `11` `11b` `11c` `11d` `11e` `optimize_options_multiples` |
| [8. Analyse d'un run](#8-analyse-dun-run) | `12` `13` `14` `compare` `mesure_slippage` |
| [9. Rapport Streamlit](#9-rapport-streamlit) | dashboard |
| [10. Orchestrateur](#10-orchestrateur-trimestriel) | `run_pipeline_quarterly` |
| [11. Outils IBKR](#11-outils-ibkr) | `restart_gateway` |
| [12. Tests](#12-tests) | pytest |
| [Tableau récapitulatif](#tableau-récapitulatif) | tout sur une page |

**Convention :** tout script listé ici avec des options accepte `--help`.
Cinq n'en ont aucune et se lancent nus : `01`, `01b`, `05`, `06`, `07`.

---

## Installation

```bash
pip install -r requirements.txt          # pipeline complet
pip install -r report/requirements.txt   # dashboard seul (sous-ensemble)
```

### Variables d'environnement

```bash
export SEC_CONTACT_EMAIL="ton.adresse@exemple.fr"   # OBLIGATOIRE
export MISTRAL_API_KEY="ta_cle"                     # optionnel
export ALPHAVANTAGE_API_KEY="ta_cle"                # optionnel
```

| Variable | Utilisée par | Obligatoire ? |
|---|---|---|
| `SEC_CONTACT_EMAIL` | `01`, `01b`, `03b`, `04`, `04b`, `04c`, `07b` | **oui**, dès qu'on interroge la SEC |
| `MISTRAL_API_KEY` | `02`, `04c`, `07b` | non — ces étapes se dégradent ou se sautent sans elle |
| `ALPHAVANTAGE_API_KEY` | `08 --av-backfill-dates` | non — sans elle, pas de backfill d'options passées |

`SEC_CONTACT_EMAIL` n'a **aucune valeur par défaut**, volontairement : la SEC
exige un User-Agent identifiant un contact réel, et un User-Agent générique se
fait bloquer (403/429). Les scripts concernés **échouent au démarrage** avec un
message explicite plutôt que de dégrader silencieusement — c'est un message
d'erreur immédiat au lieu de trous invisibles dans les parquets de sortie.

Aucune clé n'est nécessaire pour IBKR (`03`, `03b`, `08`), qui passe par IB
Gateway/TWS en local.

---

## Démarrage rapide

Premier run complet, du dépôt vide à un backtest options exploitable :

```bash
# 0. la seule variable obligatoire
export SEC_CONTACT_EMAIL="ton.adresse@exemple.fr"

# 1. l'univers et ses secteurs
python 01_build_universe.py
python 01b_historique_univers_sp500.py
python 02_categoriser_secteurs.py

# 2. les cours (IB Gateway doit tourner)
python 03_recuperation_cours.py
python 03b_recuperation_cours_quotidiens.py

# 3. les données financières SEC
python 04_recuperation_10k.py
python 04b_recuperation_10q.py

# 4. la valorisation théorique
python 05_calcul_multiples.py
python 06_calcul_multiples_moyens.py
python 07_calcul_dcf.py
python 06b_calcul_valorisation_combinee.py

# 5. le backtest
python 10_backtest_options.py --strategy valuation_gap_multiples_options --start-date 2015-01-01

# 6. le rapport
streamlit run report/Home.py
```

`08_recuperation_options.py` est **facultatif** : sans snapshots réels, le moteur
simule toutes les entrées par Black-Scholes (il le dit au démarrage). Lance-le
régulièrement pour accumuler des chaînes réelles au fil du temps.

**Pour tout enchaîner d'un coup**, voir [l'orchestrateur](#10-orchestrateur-trimestriel).

---

## 1. Univers et secteurs

### `01_build_universe.py` — la liste des entreprises suivies

Construit l'univers S&P 500 **actuel** depuis Wikipédia.

```bash
python 01_build_universe.py
```

*Aucune option.* Écrit `data/universe/sp500_universe.csv`.

### `01b_historique_univers_sp500.py` — l'univers point-in-time

Reconstruit **qui était dans l'indice à chaque date passée** (composants actuels
+ entrées/sorties historiques). Sans ce fichier, les backtests appliquent
l'univers d'aujourd'hui à tout le passé : **biais de survivance**, résultats
optimistes. Les moteurs préviennent quand il manque.

```bash
python 01b_historique_univers_sp500.py
```

*Aucune option.* Écrit `data/universe/sp500_universe_full.csv`.

### `02_categoriser_secteurs.py` — le secteur de chaque entreprise

Classe chaque entreprise dans un secteur (liste fixe). Le secteur sert aux
multiples comparables (`06`) et au rendement du dividende du pricing d'options.

```bash
python 02_categoriser_secteurs.py
python 02_categoriser_secteurs.py --universe autre_univers.csv
```

| Option | Effet |
|---|---|
| `--universe` | Fichier d'univers à catégoriser (défaut : celui de `01`) |

Met à jour le CSV d'univers et le cache `secteur_cache.json`.

---

## 2. Cours de bourse

**Les deux exigent IB Gateway ou TWS en cours d'exécution** (port 4001 par
défaut). Voir [Outils IBKR](#11-outils-ibkr) si la passerelle décroche.

### `03_recuperation_cours.py` — cours de fin d'année

Un cours par entreprise et par année (dernier jour coté de décembre). Sert aux
multiples historiques.

```bash
python 03_recuperation_cours.py
python 03_recuperation_cours.py --limit 10                 # test rapide
python 03_recuperation_cours.py --tickers AAPL MSFT
python 03_recuperation_cours.py --start-year 2010 --force-refresh
```

| Option | Effet |
|---|---|
| `--tickers` | Ne traite que ces symboles |
| `--limit` | S'arrête après N entreprises (test) |
| `--start-year` | Première année à récupérer |
| `--refresh-days` | Âge au-delà duquel une donnée est rafraîchie |
| `--force-refresh` | Ignore le cache, tout re-télécharger |
| `--port` | Port IB Gateway (défaut 4001) |
| `--output-dir` | Répertoire de sortie |

Écrit `data/prices/year_end_prices.parquet`.

### `03b_recuperation_cours_quotidiens.py` — cours quotidiens OHLCV

**Indispensable aux backtests** : `09` et `10` s'exécutent jour de bourse par
jour de bourse. Récupère aussi l'indice de référence.

```bash
python 03b_recuperation_cours_quotidiens.py
python 03b_recuperation_cours_quotidiens.py --start-year 2015 --limit 20
python 03b_recuperation_cours_quotidiens.py --skip-ibkr      # cache seul, hors ligne
```

| Option | Effet |
|---|---|
| `--tickers` `--limit` `--start-year` | comme `03` |
| `--refresh-days` `--force-refresh` `--port` `--output-dir` | comme `03` |
| `--benchmark-symbol` | Indice de référence (défaut : celui de `config.py`) |
| `--no-benchmark` | Ne récupère pas l'indice |
| `--skip-ibkr` | N'appelle pas IBKR : n'utilise que ce qui est déjà en cache |

Écrit `data/prices/daily_prices.parquet`.

---

## 3. Données financières SEC

Source : SEC EDGAR (gratuit, mais `SEC_CONTACT_EMAIL` est **obligatoire** : sans
elle, ces scripts s'arrêtent au démarrage). `04` et `04b` sont les deux seuls
indispensables ; `04c` est optionnel.

### `04_recuperation_10k.py` — comptes annuels

```bash
python 04_recuperation_10k.py
python 04_recuperation_10k.py --ticker AAPL          # une seule entreprise
python 04_recuperation_10k.py --limit 20 --workers 4
```

| Option | Effet |
|---|---|
| `--ticker` | Une seule entreprise |
| `--tickers` | Plusieurs symboles |
| `--limit` | S'arrête après N entreprises |
| `--workers` | Téléchargements en parallèle |
| `--refresh-days` `--force-refresh` | Gestion du cache |

Écrit `data/financials/financials.parquet`.

### `04b_recuperation_10q.py` — comptes trimestriels (TTM)

Douze mois glissants, reconstitués trimestre par trimestre. C'est ce qui permet
au backtest de réagir à une publication au lieu d'attendre l'exercice suivant.

```bash
python 04b_recuperation_10q.py
python 04b_recuperation_10q.py --resume               # reprend où ça s'est arrêté
python 04b_recuperation_10q.py --ticker AAPL
```

| Option | Effet |
|---|---|
| `--ticker` `--tickers` `--limit` | Périmètre |
| `--resume` | Reprend un run interrompu |
| `--refresh-days` `--force-refresh` `--output-dir` | Cache et sortie |

Écrit `data/financials/financials_ttm.parquet`.

### `04c_recuperation_8k.py` — événements matériels *(optionnel)*

Détecte les annonces significatives (8-K) entre deux trimestres.

```bash
python 04c_recuperation_8k.py
python 04c_recuperation_8k.py --limit 50 --resume
```

| Option | Effet |
|---|---|
| `--ticker` `--tickers` `--limit` `--resume` `--output-dir` | comme `04b` |
| `--max-failure-ratio` | Part d'échecs tolérée avant abandon |
| `--no-llm-cache` | Ignore le cache des appels LLM |

Écrit `data/financials/material_events_8k.parquet`.

---

## 4. Valorisation

### `05_calcul_multiples.py` — multiples par entreprise

EV/EBITDA, EV/Sales, P/E, à partir des cours (`03`/`03b`) et des comptes (`04`).

```bash
python 05_calcul_multiples.py
```

*Aucune option.* Écrit `data/multiples/multiples.parquet`.

### `06_calcul_multiples_moyens.py` — multiples sectoriels

Moyennes et médianes par secteur : les comparables auxquels chaque entreprise
est confrontée.

```bash
python 06_calcul_multiples_moyens.py
```

*Aucune option.*

### `07_calcul_dcf.py` — valorisation DCF

```bash
python 07_calcul_dcf.py
```

*Aucune option.* Écrit `data/dcf/resultats_dcf.xlsx` et
`data/dcf/dcf_historique.parquet`.

### `06b_calcul_valorisation_combinee.py` — **le signal des stratégies options**

Valorisation théorique combinée : multiples sectoriels **par année** en
priorité, DCF en repli quand le secteur a trop peu de pairs. C'est le fichier
que lisent `10` et tous les optimiseurs.

```bash
python 06b_calcul_valorisation_combinee.py
python 06b_calcul_valorisation_combinee.py --no-point-in-time-peers
```

| Option | Effet |
|---|---|
| `--no-point-in-time-peers` | Utilise les pairs sectoriels ACTUELS pour toutes les dates passées (plus rapide, mais réintroduit un biais de survivance dans les comparables) |

Écrit `data/multiples/valorisation_combinee_historique.parquet`.

> **À lancer après `07`**, pas avant : le repli DCF a besoin de son fichier.

### `07b_validation_qualitative.py` — relecture LLM *(optionnel)*

Confronte le signal quantitatif au texte des filings, via Mistral. Nécessite
`MISTRAL_API_KEY`.

```bash
python 07b_validation_qualitative.py
python 07b_validation_qualitative.py --limit 30 --resume
```

| Option | Effet |
|---|---|
| `--limit` `--resume` `--output-dir` | Périmètre, reprise, sortie |

---

## 5. Chaînes d'options

### `08_recuperation_options.py` — snapshots d'options réels

Récupère les chaînes ITM/ATM/OTM via IBKR. **Chaque run archive un snapshot
horodaté** dans `data/options/history/` (jamais écrasé), en plus du fichier
courant. Plus tu le lances souvent, plus les backtests entrent sur des contrats
réels au lieu de contrats simulés.

```bash
python 08_recuperation_options.py
python 08_recuperation_options.py --limit 20 --resume
python 08_recuperation_options.py --auto-restart-gateway
python 08_recuperation_options.py --av-backfill-dates 2023-06-15 2023-12-15
```

| Option | Effet |
|---|---|
| `--tickers` `--limit` `--resume` `--output-dir` | Périmètre, reprise, sortie |
| `--valuation-threshold` | Ne collecte que les entreprises dont l'écart de valorisation dépasse ce seuil |
| `--skip-valuation-filter` | Collecte tout l'univers, sans filtre de valorisation |
| `--auto-restart-gateway` | Relance IB Gateway automatiquement s'il décroche |
| `--av-backfill-dates` | Peuple l'historique avec de **vraies dates passées** via Alpha Vantage (nécessite `ALPHAVANTAGE_API_KEY`) — sans attendre l'accumulation de runs futurs |
| `--no-alpha-vantage` | Désactive Alpha Vantage |

Écrit `data/options/option_chains.parquet` (écrasé) **et**
`data/options/history/option_chains_<horodatage>.parquet` (conservé).

---

## 6. Backtests

Les deux écrivent un sous-dossier par run, contenant `equity_curve.parquet`,
`positions_history.parquet`, `trades.parquet`, `signals_history.parquet`,
`metrics.json` et `run_config.json`. `10` y ajoute `executions.parquet` (une
ligne par fill, achat **et** vente).

### `09_backtest.py` — stratégie ACTIONS

```bash
python 09_backtest.py --list-strategies
python 09_backtest.py --strategy valuation_gap_dcf --start-date 2015-01-01
python 09_backtest.py --entry-threshold-pct 25 --stop-loss-pct -10 --take-profit-pct 40
```

**Stratégies disponibles :** `valuation_gap_dcf`, `valuation_gap_sector_neutral`

| Option | Effet |
|---|---|
| `--list-strategies` | Liste les stratégies puis sort |
| `--strategy` | Stratégie à jouer |
| `--start-date` `--end-date` | Période (`YYYY-MM-DD`) |
| `--initial-capital` | Capital de départ |
| `--entry-threshold-pct` | Seuil d'entrée. **Non précisé, chaque stratégie garde le sien** — ils ne se lisent pas pareil (`valuation_gap_dcf` : écart au cours ; `valuation_gap_sector_neutral` : écart à la médiane du secteur) |
| `--stop-loss-pct` | Négatif, ex. `-15` |
| `--take-profit-pct` | Positif |
| `--momentum-min-pct` / `--no-momentum-filter` | Filtre momentum |
| `--commission-bps` `--slippage-bps` | Coûts de transaction |
| `--risk-free-rate` `--benchmark-symbol` | Taux sans risque, indice de référence |
| `--strategy-param KEY=VALUE` | Paramètre supplémentaire de la stratégie (répétable) |
| `--run-id` | Nom du sous-dossier de sortie (défaut : horodatage) |

Écrit `data/backtest/<run_id>/`.

### `10_backtest_options.py` — stratégie OPTIONS

```bash
python 10_backtest_options.py --list-strategies
python 10_backtest_options.py --strategy valuation_gap_multiples_options --start-date 2015-01-01
python 10_backtest_options.py --strategy valuation_gap_expected_value_options \
    --strategy-param min_kelly_fraction=0.10
```

**Stratégies disponibles :** `valuation_gap_options`,
`valuation_gap_multiples_options`, `valuation_gap_expected_value_options`

Les options se rangent en cinq familles :

**Périmètre et capital**
`--strategy` `--list-strategies` `--start-date` `--end-date` `--initial-capital`
`--benchmark-symbol` `--run-id` `--strategy-param KEY=VALUE`

**Signal et entrée**
`--entry-threshold-pct` `--exit-threshold-pct` `--momentum-min-pct`
`--no-momentum-filter`

**Contrat et sorties**

| Option | Effet |
|---|---|
| `--target-tenor-days` | Échéance visée à l'entrée (défaut : 2 ans) |
| `--roll-when-days-left` | Point de décision du roulement, en jours avant échéance |
| `--stop-loss-pct` `--take-profit-pct` | Seuils de sortie |
| `--stop-basis` | `underlying` (défaut) ou `premium` : sur quoi les stops se mesurent |
| `--take-profit-convergence-fraction` | Prise de gain à X % du chemin vers la valeur théorique, au lieu d'un seuil fixe |
| `--min-holding-days` | Durée minimale de détention avant qu'un stop puisse sortir |
| `--exit-when-signal-lost` / `--no-exit-when-signal-lost` | Vendre quand l'écart se referme |
| `--vol-mode` | `frozen` ou `rolling` : la volatilité de repricing suit-elle le marché |

**Coûts et frottements**
`--commission-per-contract` `--commission-min-per-order` `--slippage-pct-of-premium`
`--max-fee-pct-of-trade` `--fee-bump-max-extra-pct` `--no-cash-interest`
`--fractional-contracts`

**Dimensionnement et risque**

| Option | Effet |
|---|---|
| `--max-delta-notional-pct` | **Plafond de levier** : exposition delta-équivalente maximale, en % du NAV |
| `--delever-tolerance-pct` | Bande de tolérance avant réduction au prorata |
| `--min-delta-for-sizing` | Delta plancher pour dimensionner (évite d'empiler des contrats sur des options mortes) |
| `--max-trade-dollar` `--max-trade-pct-of-nav` | Plafond par ordre d'achat |
| `--min-deployment-pct` | Plancher de primes investies (0 = désactivé) |
| `--min-resize-relative-pct` | Redimensionnement minimal pour qu'un ordre soit passé |
| `--rebalance-log-gap-threshold` | ε : dérive de l'écart nécessaire pour retoucher une position existante |
| `--daily-rebalance` / `--no-daily-rebalance` | Réévaluer chaque jour, ou seulement aux publications |
| `--real-snapshot-tolerance-days` | Fenêtre de tolérance pour utiliser un snapshot réel |

Écrit `data/backtest_options/<run_id>/`.

---

## 7. Optimisation

Tous rejouent le backtest pour chaque valeur d'une grille, sur des **données
chargées une seule fois**, tous les autres réglages figés. Tous acceptent
`--workers N` pour paralléliser, et écrivent un CSV dans
`data/backtest_options/`.

> **Aucun n'est une validation.** `11`, `11b` et `11d` séparent apprentissage et
> test (`--train-fraction`, `--no-walk-forward`) ; `11c` et `11e` sont
> *in-sample* : la valeur retenue est choisie sur les données qui servent
> ensuite à la juger. C'est un point de départ, pas une preuve.

### `11_optimize_options_stops.py` — le couple stop-loss / take-profit

```bash
python 11_optimize_options_stops.py
python 11_optimize_options_stops.py --objective calmar_ratio --workers 4
python 11_optimize_options_stops.py \
    --stop-loss-grid -10 -15 -20 -25 -30 --take-profit-grid 20 40 60 100
```

| Option | Effet |
|---|---|
| `--stop-loss-grid` `--take-profit-grid` | Les deux grilles balayées |
| `--take-profit-mode` | Comment interpréter la grille de prise de gain |
| `--objective` | Métrique de classement (défaut `sharpe_ratio`) |
| `--min-trades` | Sous ce nombre, une combinaison n'est pas recommandable |
| `--train-fraction` / `--no-walk-forward` | Séparation apprentissage / test |
| `--top-n` `--output-csv` `--workers` | Affichage, sortie, parallélisme |

### `11b_optimize_rebalance_threshold.py` — ε, le seuil anti-churn

Combien l'écart de valorisation doit dériver avant qu'on retouche une position
déjà engagée.

```bash
python 11b_optimize_rebalance_threshold.py --epsilon-grid 0 0.1 0.15 0.2 0.3
```

### `11c_optimize_convergence_fraction.py` — l'hypothèse de convergence

Quelle **part** du chemin vers la valeur théorique on suppose parcourue à
l'échéance (stratégie « espérance de gain » uniquement).

```bash
python 11c_optimize_convergence_fraction.py --fraction-grid 0.3 0.5 0.7 1.0 --workers 4
```

### `11d_optimize_entry_threshold.py` — le seuil d'entrée

```bash
python 11d_optimize_entry_threshold.py --entry-threshold-grid 10 15 20 25 30
python 11d_optimize_entry_threshold.py --gap-grid ... --exit-threshold-ratio 0.5
```

### `11e_optimize_strategy_param.py` — **n'importe quel autre paramètre**

Balaye tout paramètre scalaire, de la stratégie **ou** du moteur, en trouvant
tout seul auquel des deux il appartient. Évite d'écrire un `11f`, `11g`… par
réglage.

```bash
python 11e_optimize_strategy_param.py --list-params          # tout ce qui est balayable
python 11e_optimize_strategy_param.py                        # min_kelly_fraction (défaut)
python 11e_optimize_strategy_param.py --param min_holding_days
python 11e_optimize_strategy_param.py --param min_kelly_fraction --grid 0 0.05 0.1 0.2
```

| Option | Effet |
|---|---|
| `--param` | Paramètre à balayer (défaut `min_kelly_fraction`) |
| `--grid` | Valeurs à tester. Obligatoire hors des paramètres à grille connue |
| `--list-params` | Liste ce que la stratégie et le moteur acceptent, puis sort |
| `--target strategy\|engine` | Seulement en cas d'homonymie entre les deux |
| `--objective` `--min-trades` `--top-n` `--output-csv` `--workers` | comme les autres |

**Grilles par défaut fournies pour :** `min_kelly_fraction`,
`strike_grid_n_sigma`, `strike_grid_step_sigma`, `exit_threshold_ratio`,
`min_holding_days`, `take_profit_convergence_fraction`,
`max_delta_notional_pct`, `delever_tolerance_pct`, `min_deployment_pct`,
`roll_when_days_left`, `target_tenor_days`.

Le script **refuse un nom mal orthographié** et propose le plus proche : sans
ça, `Strategy.__init__(**params)` l'accepterait sans erreur et toute la grille
rendrait le même run.

### `optimize_options_multiples.py` — recherche aléatoire multi-paramètres

Contrairement aux `11*` (une seule dimension balayée), celui-ci tire
aléatoirement **quatre paramètres à la fois** de la stratégie multiples.

```bash
python optimize_options_multiples.py --trials 300 --objectif calmar_ratio
python optimize_options_multiples.py --trials 50 --seed 42 --top 20
```

| Option | Effet |
|---|---|
| `--trials` | Nombre de tirages (défaut 300) |
| `--objectif` | Métrique de classement (défaut `calmar_ratio`) |
| `--min-trades` | Plancher de crédibilité (défaut 15) |
| `--seed` | Graine, pour un tirage reproductible |
| `--top` `--output` `--checkpoint-every` | Affichage, sortie, sauvegardes intermédiaires |

### `demo_kelly_optimization_workflow.py` — l'enchaînement guidé

Déroule le workflow d'optimisation complet en phases.

```bash
python demo_kelly_optimization_workflow.py --phase all
python demo_kelly_optimization_workflow.py --phase 2 --workers 4
```

`--phase` accepte `1`, `2`, `3`, `3bis`, `4` ou `all` (les enchaîne dans l'ordre).

---

## 8. Analyse d'un run

### `12_analyse_put_call.py` — décomposition CALL / PUT

Le résumé d'un backtest agrège les deux paris en un seul NAV : il dit combien la
stratégie gagne, jamais **laquelle des deux jambes** le gagne.

```bash
python 12_analyse_put_call.py                        # dernier run de la stratégie
python 12_analyse_put_call.py --run-id 20260810_143000 --export
```

| Option | Effet |
|---|---|
| `--run-id` | Run à analyser (défaut : le plus récent) |
| `--strategy` | Stratégie dont on prend le dernier run |
| `--export` | Écrit les tableaux en CSV dans le dossier du run |

### `13_diagnostic_friction.py` — thèse contre friction contre churn

Rejoue la stratégie sur un plan 2×2 (slippage réel/nul × rebalancement
quotidien activé/désactivé) et sépare ce qui vient de la thèse, du coût de
l'exécuter, et du fait de trop trader.

```bash
python 13_diagnostic_friction.py --strategy valuation_gap_multiples_options --start-date 2015-01-01
```

| Option | Effet |
|---|---|
| `--strategy` `--start-date` `--end-date` | Périmètre |
| `--slippage-pct-of-premium` `--max-trade-dollar` `--initial-capital` | Réglages du plan |
| `--entry-threshold-pct` `--benchmark-symbol` | Signal et référence |
| `--export` | Écrit les tableaux en CSV |

### `14_audit_backtest.py` — audit d'un run actions

Relit les sorties de `09` et répond aux questions qu'un résultat trop beau doit
soulever.

```bash
python 14_audit_backtest.py --run-id 20260810_143000
python 14_audit_backtest.py --run-dir data/backtest/mon_run --fenetre-annees 3
```

| Option | Effet |
|---|---|
| `--run-id` | Run à auditer |
| `--run-dir` | Chemin explicite, au lieu de `--run-id` |
| `--fenetre-annees` | Largeur de la fenêtre glissante d'analyse |

### `compare_options_strategies.py` — les trois stratégies côte à côte

Même période, mêmes données, mêmes coûts : la seule différence est la stratégie.

```bash
python compare_options_strategies.py --start-date 2015-01-01
python compare_options_strategies.py --strategies valuation_gap_options valuation_gap_multiples_options
python compare_options_strategies.py --reuse-existing
```

| Option | Effet |
|---|---|
| `--strategies` | Sous-ensemble à comparer (défaut : les trois) |
| `--start-date` `--end-date` `--initial-capital` `--benchmark-symbol` | Périmètre commun |
| `--reuse-existing` | Réutilise les runs déjà joués au lieu de tout rejouer |
| `--extra-arg` | Argument supplémentaire passé à chaque backtest |
| `--output-csv` | Fichier de sortie |

### `mesure_slippage_options.py` — le slippage réel

Mesure l'écart bid/ask réellement observé dans les snapshots archivés, pour
calibrer le `--slippage-pct-of-premium` de `10_backtest_options.py` sur des
chiffres plutôt que sur une hypothèse.

```bash
python mesure_slippage_options.py
python mesure_slippage_options.py --by-symbol --moneyness-pct 5 --min-tenor-days 180
```

| Option | Effet |
|---|---|
| `--by-symbol` | Détail par entreprise au lieu de l'agrégat |
| `--moneyness-pct` | Ne retient que les contrats dans cette bande autour de la monnaie |
| `--min-tenor-days` | Échéance minimale considérée |

---

## 9. Rapport Streamlit

Dashboard en lecture seule : il n'affiche que des fichiers déjà produits, il ne
relance **jamais** un calcul.

```bash
streamlit run report/Home.py
```

Quatre pages :

| Page | Contenu |
|---|---|
| **📊 Data** | Couverture par entreprise : années de cours, exercices 10-K, contrats d'options, fraîcheur de chaque source |
| **📈 Analyse** | Nappe de volatilité implicite, structure par terme, skew, greeks, liquidité, repères de valorisation, clustering des multiples |
| **🧪 Stratégies** | Pour un run de `09` ou `10` : KPI, NAV, composition, et **journal des achats/ventes** avec le solde détenu après chaque exécution |
| **⚙️ Pipeline** | Journal des runs de l'orchestrateur et fraîcheur des sorties |

> À lancer **depuis la racine**, pas depuis `report/`.

---

## 10. Orchestrateur trimestriel

### `run_pipeline_quarterly.py` — tout enchaîner

Exécute dans l'ordre `04b` → `04c` → `05` → `06` → `06b` → `07` → `07b` → `08`,
avec journalisation par étape, réessais et reprise.

```bash
python run_pipeline_quarterly.py                        # run live complet
python run_pipeline_quarterly.py --limit 20             # test rapide
python run_pipeline_quarterly.py --skip-options         # sans IB Gateway
python run_pipeline_quarterly.py --resume               # reprend le dernier run
python run_pipeline_quarterly.py --as-of-date 2024-06-30   # replay, aucun appel réseau
```

| Option | Effet |
|---|---|
| `--as-of-date YYYY-MM-DD` | **Mode replay** point-in-time, sans aucun appel réseau (rejoue `05`/`06`/`06b`/`07` sur les données déjà collectées) |
| `--limit` | Transmis à `04b`/`04c`/`07b` — mode live seulement |
| `--skip-options` | Saute `08` au lieu de tenter de relancer IB Gateway |
| `--retries` | Réessais par étape, backoff exponentiel (défaut 2) |
| `--step-timeout` | Durée maximale d'une étape, en secondes (défaut 7200) |
| `--resume` | Reprend le dernier run du même mode en sautant les étapes réussies |
| `--run-id` | Nom du sous-dossier de journal (défaut : horodatage) |

Écrit `data/pipeline_runs/<run_id>/` (un `report.json` + un `.log` par étape),
que la page **⚙️ Pipeline** du dashboard affiche.

---

## 11. Outils IBKR

### `restart_gateway.py` — relancer IB Gateway

Redémarre IB Gateway via [IBC](https://github.com/IbcAlpha/IBC), quand la
passerelle décroche au milieu d'une collecte.

```bash
python restart_gateway.py
```

*Aucune option.* Aussi appelé automatiquement par
`08_recuperation_options.py --auto-restart-gateway`.

### `ib_connect.py` — module, pas un script

Connexion partagée par `03`, `03b` et `08`. **Ne se lance pas directement.**

---

## 12. Tests

```bash
pytest                                   # toute la suite (~700 tests, ~2 min)
pytest tests/test_options_engine.py      # un fichier
pytest -k kelly                          # par mot-clé
pytest -x -q                             # s'arrête au premier échec
```

Quelques fichiers utiles à connaître :

| Fichier | Ce qu'il protège |
|---|---|
| `tests/test_journal_executions.py` | Conservation des quantités : aucun contrat vendu sans avoir été acheté |
| `tests/test_journal_rapport.py` | Le journal achats/ventes du dashboard, et son solde jamais négatif |
| `tests/test_options_engine.py` | Le moteur options (sorties, roulements, dimensionnement) |
| `tests/test_expected_value*.py` | Le critère de Kelly et la sélection de strike |
| `tests/test_optimizers_multiprocessing.py` | Les optimiseurs sous `--workers`, y compris sur Windows |
| `tests/test_look_ahead_valorisation.py` | Absence de fuite d'information future |

---

## Tableau récapitulatif

| Commande | Rôle | Prérequis |
|---|---|---|
| `python 01_build_universe.py` | Univers S&P 500 actuel | `SEC_CONTACT_EMAIL` |
| `python 01b_historique_univers_sp500.py` | Univers point-in-time (anti-biais de survivance) | `SEC_CONTACT_EMAIL` |
| `python 02_categoriser_secteurs.py` | Secteur de chaque entreprise | `01` |
| `python 03_recuperation_cours.py` | Cours de fin d'année | `01`, IB Gateway |
| `python 03b_recuperation_cours_quotidiens.py` | Cours quotidiens **(requis par les backtests)** | `01`, IB Gateway, `SEC_CONTACT_EMAIL` |
| `python 04_recuperation_10k.py` | Comptes annuels SEC | `01`, `SEC_CONTACT_EMAIL` |
| `python 04b_recuperation_10q.py` | Comptes trimestriels TTM | `01`, `SEC_CONTACT_EMAIL` |
| `python 04c_recuperation_8k.py` | Événements matériels *(optionnel)* | `01`, `SEC_CONTACT_EMAIL` |
| `python 05_calcul_multiples.py` | Multiples par entreprise | `03`, `04` |
| `python 06_calcul_multiples_moyens.py` | Multiples sectoriels | `05` |
| `python 07_calcul_dcf.py` | Valorisation DCF | `04` |
| `python 06b_calcul_valorisation_combinee.py` | **Signal des stratégies options** | `05`, `06`, `07` |
| `python 07b_validation_qualitative.py` | Relecture LLM *(optionnel)* | `06b`, `SEC_CONTACT_EMAIL`, `MISTRAL_API_KEY` |
| `python 08_recuperation_options.py` | Chaînes d'options réelles *(optionnel)* | `06b`, IB Gateway |
| `python 09_backtest.py` | Backtest **actions** | `03b`, `07` |
| `python 10_backtest_options.py` | Backtest **options** | `03b`, `06b` |
| `python 11_optimize_options_stops.py` | Optimise stop-loss / take-profit | comme `10` |
| `python 11b_optimize_rebalance_threshold.py` | Optimise ε (anti-churn) | comme `10` |
| `python 11c_optimize_convergence_fraction.py` | Optimise l'hypothèse de convergence | comme `10` |
| `python 11d_optimize_entry_threshold.py` | Optimise le seuil d'entrée | comme `10` |
| `python 11e_optimize_strategy_param.py` | Optimise **tout autre paramètre** | comme `10` |
| `python optimize_options_multiples.py` | Recherche aléatoire à 4 paramètres | comme `10` |
| `python demo_kelly_optimization_workflow.py` | Workflow d'optimisation guidé | comme `10` |
| `python 12_analyse_put_call.py` | Décomposition CALL / PUT | un run de `10` |
| `python 13_diagnostic_friction.py` | Thèse / friction / churn | comme `10` |
| `python 14_audit_backtest.py` | Audit d'un run actions | un run de `09` |
| `python compare_options_strategies.py` | Les 3 stratégies côte à côte | comme `10` |
| `python mesure_slippage_options.py` | Slippage réel observé | `data/options/history/` |
| `python run_pipeline_quarterly.py` | Enchaîne `04b` → `08` | `01`, `03b` |
| `python restart_gateway.py` | Relance IB Gateway | IBC installé |
| `streamlit run report/Home.py` | Dashboard | au moins une sortie du pipeline |
| `pytest` | Suite de tests | — |
