# Rapport dynamique — pipeline options US

Dashboard Streamlit à deux pages, lu directement depuis les fichiers produits
par le pipeline (`01_build_universe.py` à `08_recuperation_options.py`). Le
rapport ne relance jamais de collecte lui-même : il ne fait que lire `./data/`.

## Rafraîchissement trimestriel (04b, 04c, 07b, run_pipeline_quarterly.py)

Le pipeline de base (04→07) est annuel (un 10-K par an). Ces scripts
permettent un rafraîchissement TRIMESTRIEL de la valorisation elle-même,
point-in-time (chaque donnée datée de son dépôt SEC réel) :

    04b_recuperation_10q.py       -> 10-Q + reconstruction TTM (voir sa
                                      docstring : TTM vs trimestre brut, et
                                      la discrétisation cumul YTD -> trimestre)
    04c_recuperation_8k.py        -> événements matériels (8-K) entre deux
                                      trimestres TTM connus, classifiés par LLM
    07b_validation_qualitative.py -> verdict LLM de cohérence qualitative
                                      (texte du 10-K/10-Q à sa date de dépôt)
                                      vs l'écart de valorisation quantitatif
    run_pipeline_quarterly.py     -> orchestre 04b→04c→05→06→06b→07→07b→08 en
                                      conditions réelles (mode live), ou
                                      reconstitue une valorisation point-in-time
                                      passée sans aucun appel réseau
                                      (--as-of-date, mode replay)

04c et 07b réutilisent `sec_filings_text.py` (recherche/téléchargement de
filings SEC + appel Mistral générique) et nécessitent `MISTRAL_API_KEY` (voir
02_categoriser_secteurs.py) pour produire un verdict -- sans cette variable,
ils journalisent "non_evalue" plutôt que de planter.

05/06b/07 consomment automatiquement le TTM (`FINANCIALS_TTM_FILE`) dès que
04b a tourné une fois, en plus de l'annuel -- sans régression : identique à
avant si 04b n'a jamais tourné.

```bash
python 04b_recuperation_10q.py
python 04c_recuperation_8k.py
python 05_calcul_multiples.py && python 06_calcul_multiples_moyens.py && python 06b_calcul_valorisation_combinee.py
python 07_calcul_dcf.py
python 07b_validation_qualitative.py
# ou, en une commande :
python run_pipeline_quarterly.py --skip-options   # sans 08 (pas besoin d'IB Gateway)
```

Cron (exemple, peu après chaque fenêtre de dépôt 10-Q habituelle) :

```
0 6 5 2,5,8,11 *  cd /chemin/vers/CalculRisque_Mark3 && python3 run_pipeline_quarterly.py --skip-options >> logs/quarterly.log 2>&1
```

Reconstitution point-in-time (backtest manuel, aucun appel réseau) :

```bash
python run_pipeline_quarterly.py --as-of-date 2024-06-30
```

## Backtest (01b, 03b, 09)

Trois scripts complètent le pipeline pour permettre de backtester une
stratégie construite sur l'écart entre cours de bourse et valorisation DCF
(07_calcul_dcf.py), sans biais de survivance et sans look-ahead bias :

    01b_historique_univers_sp500.py   -> univers POINT-IN-TIME (composants
                                          actuels + radiés, avec dates
                                          d'entrée/sortie de l'indice)
    03b_recuperation_cours_quotidiens.py -> cours QUOTIDIENS (IBKR + repli
                                          Stooq gratuit pour les radiés,
                                          qu'IBKR ne résout plus)
    09_backtest.py                    -> moteur de backtest événementiel

Ordre de lancement pour un backtest complet (en plus de 01/02 déjà connus) :

```bash
python 01b_historique_univers_sp500.py
python 03b_recuperation_cours_quotidiens.py --tickers data/universe/sp500_universe_full.csv
python 04_recuperation_10k.py --tickers data/universe/sp500_universe_full.csv
python 07_calcul_dcf.py
python 09_backtest.py --strategy valuation_gap_dcf --start-date 2015-01-01
```

`04_recuperation_10k.py` et `03b` doivent recevoir l'univers COMPLET
(`sp500_universe_full.csv`, sortie de 01b) pour backfiller aussi les
entreprises sorties du S&P 500 -- sinon le backtest retombe sur l'univers
actuel appliqué rétroactivement (biais de survivance, signalé par un warning
au lancement de 09).

Hypothèses du moteur (`backtest/engine.py`), à garder en tête pour
interpréter des résultats :
    - Décision à la clôture du jour J, exécution à l'ouverture de J+1 (aucune
      information n'est utilisée avant sa date réelle de publication : le
      signal DCF utilise `filed_date`, la date de dépôt SEC du 10-K, pas la
      date de clôture d'exercice).
    - Une position n'est JAMAIS fermée simplement parce que son écart de
      valorisation s'est refermé : seuls un stop-loss, un take-profit, ou une
      disparition des données de prix (radiation non couverte par Stooq) la
      clôturent. Elle reste sinon "gelée" à taille inchangée.
    - Un signal DCF vieux de plus de `BACKTEST_SIGNAL_MAX_AGE_DAYS` (config.py,
      400 jours par défaut) n'est plus considéré comme une base valable pour
      une NOUVELLE entrée (mais n'affecte pas une position déjà ouverte).

Résultats sauvegardés intégralement sous `data/backtest/<run_id>/` :
`equity_curve.parquet`, `positions_history.parquet`, `trades.parquet`,
`signals_history.parquet`, `metrics.json`, `run_config.json`.

### Ajouter une nouvelle stratégie

Créer un fichier dans `backtest/strategies/`, y définir une classe héritant
de `Strategy` (`backtest/strategies/base.py`) et décorée par
`@register_strategy("mon_nom")`, puis l'importer dans
`backtest/strategies/__init__.py`. Elle devient disponible via
`python 09_backtest.py --strategy mon_nom` sans toucher au moteur : la
stratégie ne gère que le choix des candidats et leur pondération relative,
l'engine gère uniformément le capital, les plafonds de positions, le
stop-loss/take-profit et les coûts de transaction pour toutes les stratégies.

## Backtest OPTIONS (06b, 10)

Stratégie distincte de `09_backtest.py` (actions) : achète des CALL sur les
entreprises sous-évaluées, des PUT sur les survalorisées, dimensionnés par
le delta pour une exposition $ cible ("hedge par les greeks").

    06b_calcul_valorisation_combinee.py -> valorisation théorique combinée :
                                          multiples sectoriels PAR ANNÉE en
                                          priorité (cross-sectionnel, pas
                                          blendé comme 06), DCF (07) en repli
                                          quand le secteur a trop peu de pairs
    10_backtest_options.py            -> moteur de backtest options

```bash
python 05_calcul_multiples.py
python 07_calcul_dcf.py
python 06b_calcul_valorisation_combinee.py
python 10_backtest_options.py --strategy valuation_gap_options --start-date 2015-01-01
```

Hypothèses du moteur (`backtest/options_engine.py`) :
    - Entrée : cherche un snapshot RÉEL archivé par `08_recuperation_options.py`
      à proximité de la date (fenêtre `OPTIONS_REAL_SNAPSHOT_TOLERANCE_DAYS`,
      14 jours par défaut) ; sinon simule par Black-Scholes (strike ATM,
      échéance ~9 mois, volatilité réalisée glissante en repli). **Lance
      régulièrement `08_recuperation_options.py` sur un compte paper trading
      pour accumuler des snapshots réels au fil du temps** : plus il y en a,
      moins le backtest s'appuie sur du Black-Scholes simulé.
      Depuis l'ajout d'Alpha Vantage (source gratuite, voir la docstring de
      `08_recuperation_options.py` et `ALPHAVANTAGE_API_KEY`), deux leviers
      supplémentaires réduisent cette dépendance : (1) l'IV/greeks de chaque
      snapshot IBKR viennent en priorité d'Alpha Vantage (calculés côté
      Alpha Vantage, pas par notre propre Black-Scholes) ; (2)
      `08_recuperation_options.py --av-backfill-dates 2024-01-15 2024-02-15 ...`
      reconstitue directement un VRAI historique d'options déjà expirées
      (impossible via IBKR seul, qui ne résout plus les contrats expirés),
      sans attendre l'accumulation de runs futurs.
    - Repricing quotidien TOUJOURS par Black-Scholes (aucune source ne fournit
      un flux d'options continu), à strike et échéance fixés à l'entrée. La
      volatilité, elle, dépend de `--vol-mode` : `frozen` (défaut, comportement
      historique) la fige à l'entrée pour toute la vie de la position ;
      `rolling` la fait suivre la volatilité réalisée du jour, remise à
      l'échelle de l'entrée (le rapport implicite/réalisé constaté à l'entrée
      est conservé, donc aucun saut de prime le premier jour). Plus l'échéance
      est longue, plus figer la volatilité est une approximation forte : le
      mode `rolling` est donc l'implicite de
      `valuation_gap_multiples_options` (échéance 2 ans), et reste optionnel
      ailleurs. Le coût en temps est nul (volatilité glissante précalculée en
      une passe vectorisée sur tout le panel de cours).
    - Stop-loss/take-profit sur la PRIME (`OPTIONS_STOP_LOSS_PCT`/
      `OPTIONS_TAKE_PROFIT_PCT`, -50%/+100% par défaut -- des seuils bien
      plus larges que pour les actions, effet de levier oblige), expiration
      (réglée à la valeur intrinsèque), ou disparition des données du
      sous-jacent. Même règle "jamais fermé juste parce que l'écart s'est
      refermé" que la stratégie actions -- position gelée jusqu'à un de ces
      déclencheurs.
    - Une nouvelle stratégie options s'ajoute de la même façon que pour les
      actions : fichier dans `backtest/strategies/`, classe héritant de
      `OptionsStrategy` (`backtest/strategies/options_base.py`), décorée par
      `@register_options_strategy("mon_nom")`.

### Stratégie `valuation_gap_multiples_options` (convergence long terme)

Seconde stratégie options, à côté de `valuation_gap_options` (inchangée).
Elle compare la valorisation théorique issue des **multiples sectoriels
seuls** à la valorisation boursière, et parie sur la convergence de la
seconde vers la première à horizon 2 ans :

```bash
python 10_backtest_options.py --strategy valuation_gap_multiples_options --start-date 2015-01-01
```

| | `valuation_gap_options` | `valuation_gap_multiples_options` |
|---|---|---|
| Signal | multiples, **DCF en repli** | **multiples seuls** (repli DCF écarté) |
| Écart de ±20% rapporté à | cours de bourse | **valeur théorique** |
| Strike | ATM | **à mi-chemin théorique/cours** |
| Échéance | ~9 mois | **2 ans, roulée à 9 mois** |
| Stop-loss / take-profit | −50% / +100% de la **prime** | **−25% / +30% du cours du sous-jacent** |
| Écart refermé | position gelée | **vendue au trimestre suivant** |
| Volatilité de repricing | figée à l'entrée | **suivie au jour le jour** |

Comme la valorisation boursière (nb d'actions × cours) et la valorisation
théorique (nb d'actions × valeur théorique par action) portent sur le même
nombre d'actions, leur rapport se calcule directement par action : la
stratégie lit les colonnes de `06b_calcul_valorisation_combinee.py` sans
reconstruire de capitalisation.

**Pourquoi les stops portent sur le sous-jacent et non sur la prime.** Sur le
cas type de la stratégie (théorique 120, cours 100 → strike 110, 2 ans, vol
30%), le levier effectif est de 3,5x : un stop à −25% de la prime se
déclencherait sur une baisse de seulement −7,6% du titre. Surtout, **à cours
strictement inchangé, la seule érosion de la valeur temps fait perdre 29% à
la prime en 9 mois** — le stop se déclencherait donc tout seul avant que la
convergence visée ait le temps de se produire. Appliqués au cours du
sous-jacent, les seuils décrivent bien le scénario voulu, et sont orientés
dans le sens de la position (pour un PUT, une hausse du titre est la perte).

Le strike à mi-chemin est **volontairement hors de la monnaie**, de la moitié
de l'écart : l'option ne devient gagnante que si le titre parcourt au moins
la moitié du chemin vers sa valeur théorique — une convergence partielle
suffit, un simple bruit de marché non.

**Plafond de pondération** (`OPTIONS_MULTIPLES_WEIGHT_CAP_PCT`, 100% par
défaut). Rapporté à la valeur théorique, l'écart est borné à +100% du côté
sous-évalué (le cours ne peut pas passer sous zéro) mais **non borné du côté
survalorisé** : une valeur théorique proche de zéro produit un écart de
plusieurs milliers de %. Comme le poids est proportionnel à l'écart, une
seule ligne capterait alors l'essentiel du capital — mesuré à **92% du
portefeuille** pour une théorique à 5$ contre un cours à 100$, contre 38%
une fois plafonnée. Le plafond ne s'applique qu'au **dimensionnement** : le
classement des candidats reste fait sur l'écart brut, une survalorisation
extrême restant une forte conviction. `--strategy-param weight_cap_pct=0`
le désactive.

**Réévaluation trimestrielle** : elle est automatique, sans réglage
supplémentaire. Chaque publication (10-Q via `04b_recuperation_10q.py`, 10-K
via `04`) produit un signal daté de sa date de dépôt SEC réelle ; à chacun,
les entreprises passant le seuil sont achetées et celles qui le repassent en
sens inverse sont vendues. Lance `run_pipeline_quarterly.py` pour rafraîchir
ces signaux.

Ces quatre réglages de moteur font partie de la thèse de la stratégie : elle
les déclare (attribut de classe `engine_defaults`) et `10_backtest_options.py`
les applique automatiquement, sauf si l'option correspondante est passée
explicitement en ligne de commande (`--stop-basis`, `--roll-when-days-left`,
`--no-exit-when-signal-lost`, `--target-tenor-days`...).

Côté moteur (`backtest/options_engine.py`), ces comportements sont **optionnels
et désactivés par défaut** : `valuation_gap_options` est strictement inchangée
(vérifié : sorties identiques au bit près sur le banc synthétique, à la seule
résolution du dtype `expiry` près, µs → ns).

Résultats sauvegardés sous `data/backtest_options/<run_id>/` (mêmes fichiers
que le backtest actions).

**Correctif** : `07_calcul_dcf.py::calculer_dcf` calculait
`equity_value = ev - dette_nette + cash`, alors que `net_debt` (produit par
`04_recuperation_10k.py`) est déjà net de cash (`dette brute - cash`) — le
cash était donc compté deux fois (valeur DCF surestimée pour les entreprises
avec beaucoup de trésorerie nette). Corrigé en `equity_value = ev - dette_nette`
(paramètre `cash` supprimé de `calculer_dcf`, devenu inutile) ; c'est la
formule que `06b_calcul_valorisation_combinee.py` utilisait déjà. Si tu as
des runs `07`/`09_backtest.py` antérieurs à ce correctif, relance
`07_calcul_dcf.py` pour régénérer des valeurs DCF exactes avant de
retro-comparer des résultats de backtest.

## Installation

```bash
pip install -r report/requirements.txt
```

## Lancement

**Important : lance la commande depuis la racine du dépôt** (là où se
trouvent `config.py` et `01_build_universe.py`), pas depuis `report/` —
`config.py` résout ses chemins (`./data/...`) relativement au répertoire
d'exécution, exactement comme les scripts `01` à `08`.

```bash
streamlit run report/Home.py
```

## Pages

- **📊 Data** — tableau de couverture par entreprise : années de cours,
  exercices 10-K, contrats d'options collectés, dernière mise à jour de
  chaque source. Filtres par secteur / recherche / couverture complète.
- **📈 Analyse** — pour l'entreprise sélectionnée dans la barre latérale :
  nappe de volatilité implicite (lissée par processus gaussien,
  scikit-learn), structure par terme, skew, greeks par strike, indicateurs
  de liquidité (spread, volume, OI, put/call, détection d'anomalies par
  IsolationForest), et repères de valorisation (multiples sectoriels, DCF, et
  la valorisation combinée de `06b_calcul_valorisation_combinee.py` utilisée
  par la stratégie options). En bas de page : clustering KMeans des multiples
  de valorisation sur
  l'ensemble du portefeuille suivi, avec projection PCA 2D et recoupement
  avec le secteur GICS déclaré.

## Historique des nappes de volatilité

Depuis le correctif apporté à `08_recuperation_options.py`, chaque run
archive en plus un snapshot horodaté dans `data/options/history/`
(jamais écrasé), en parallèle du fichier courant `data/options/option_chains.parquet`
(toujours écrasé par le run suivant). La page Analyse lit l'historique
complet et propose un sélecteur de date dès qu'au moins deux snapshots sont
disponibles pour l'entreprise choisie — donc **relance `04` plusieurs fois
dans le temps** (ex: une fois par semaine) pour voir la nappe évoluer.

Avant le premier run depuis ce correctif, `data/options/history/` est vide :
le rapport retombe alors sur le seul snapshot courant.

`08_recuperation_options.py --av-backfill-dates AAAA-MM-JJ ...` peuple aussi
`data/options/history/` directement avec de vraies dates passées (source
Alpha Vantage, gratuite avec clé), sans attendre l'accumulation de runs
futurs : voir la docstring en tête de `08_recuperation_options.py`.
