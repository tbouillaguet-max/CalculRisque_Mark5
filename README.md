# Rapport dynamique — pipeline options US

Dashboard Streamlit à deux pages, lu directement depuis les fichiers produits
par le pipeline (`01_build_universe.py` à `08_recuperation_options.py`). Le
rapport ne relance jamais de collecte lui-même : il ne fait que lire `./data/`.

## Configuration requise

```bash
export SEC_CONTACT_EMAIL="ton.adresse@exemple.fr"   # obligatoire pour 04, 04b, 04c, 07b
export MISTRAL_API_KEY="ta_cle"                     # optionnel : 02, 04c, 07b
export ALPHAVANTAGE_API_KEY="ta_cle"                # optionnel : 08 --av-backfill-dates
```

`SEC_CONTACT_EMAIL` n'a **pas** de valeur par défaut : la SEC exige un
User-Agent identifiant un contact réel, et un User-Agent générique se fait
bloquer (403/429). Les scripts qui interrogent la SEC échouent au démarrage
avec un message explicite si elle est absente, plutôt que de dégrader
silencieusement.

Sans `MISTRAL_API_KEY`, `04c` et `07b` journalisent leurs lignes en
`non_evalue_pas_de_cle_api` au lieu d'appeler le modèle — les filtres
qualitatifs restent alors sans effet, ce qui est le comportement voulu.

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
    - Le nombre de positions simultanées n'est **pas** plafonné : toutes les
      entreprises retenues par la stratégie sont ouvertes. La concentration
      reste bornée par le seul plafond de pondération
      (`BACKTEST_MAX_WEIGHT_PER_POSITION_PCT`), qui limite la part du
      portefeuille d'UNE ligne sans limiter leur nombre.

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
l'engine gère uniformément le capital, le stop-loss/take-profit et les coûts
de transaction pour toutes les stratégies.

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
    11_optimize_options_stops.py      -> grid-search stop-loss/take-profit sur ce moteur
    12_analyse_put_call.py            -> décomposition CALL/PUT d'un run sauvegardé
    13_diagnostic_friction.py         -> plan 2x2 thèse / friction / churn
    11b_optimize_rebalance_threshold.py -> grid-search sur ε (rebalancement sur dépôt SEC)

```bash
python 05_calcul_multiples.py
python 07_calcul_dcf.py
python 06b_calcul_valorisation_combinee.py
python 10_backtest_options.py --strategy valuation_gap_options --start-date 2015-01-01
```

Hypothèses du moteur (`backtest/options_engine.py`) :
    - Entrée : cherche le DERNIER snapshot RÉEL archivé par
      `08_recuperation_options.py` **au plus tard à la date d'exécution**
      (fenêtre `OPTIONS_REAL_SNAPSHOT_TOLERANCE_DAYS`, 14 jours par défaut) ;
      sinon simule par Black-Scholes (strike ATM,
      échéance 2 ans, volatilité réalisée glissante en repli). **Lance
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
      Du snapshot, le moteur retient le strike, l'échéance et **l'IV** — pas
      la prime : celle-ci a été cotée à un autre spot que celui d'exécution,
      et la reprendre telle quelle faisait apparaître un saut de P&L au
      premier repricing. L'IV est la grandeur transportable d'une date à
      l'autre, le prix ne l'est pas ; la prime d'entrée en est dérivée par
      Black-Scholes au spot du jour.
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
    - **Échéance 2 ans à l'entrée** (`OPTIONS_TARGET_TENOR_DAYS`), avec un
      **point de décision à 9 mois de l'expiration**
      (`OPTIONS_ROLL_WHEN_DAYS_LEFT`) : la position y est réexaminée à l'aune du
      signal courant. Écart toujours au-dessus du seuil d'entrée → le contrat
      est roulé sur une nouvelle échéance pleine, à exposition inchangée ;
      écart repassé sous le seuil (ou retourné de sens) → la position est
      clôturée (`exit_reason` `signal_lost`). On ne porte donc jamais un
      contrat sur sa dernière année de vie, là où la valeur temps s'érode le
      plus vite. `--roll-when-days-left 0` désactive ce réexamen (les positions
      vont alors jusqu'à l'expiration).
    - Stop-loss/take-profit **sur le cours du SOUS-JACENT**
      (`OPTIONS_STOP_BASIS`, `OPTIONS_STOP_LOSS_PCT`/`OPTIONS_TAKE_PROFIT_PCT`,
      −20%/+80% par défaut), orientés dans le sens de la position (pour un PUT,
      une hausse du titre est la perte) ; puis expiration (réglée à la valeur
      intrinsèque) ou disparition des données du sous-jacent. Entre deux
      réexamens de roulement, un écart qui se referme ne ferme pas la position :
      elle reste gelée jusqu'à l'un de ces déclencheurs.
      Adossés à la **prime**, ces seuils seraient atteints par la seule érosion
      de la valeur temps — une ATM à 2 ans perd ~20% de sa prime en 15 mois à
      cours strictement inchangé — et l'effet de levier ferait correspondre
      −20% de prime à une baisse du titre de quelques points seulement.
      `--stop-basis premium` rétablit l'ancienne base.
    - Le nombre de positions simultanées n'est pas plafonné, comme pour le
      moteur actions : toutes les entreprises retenues par la stratégie sont
      ouvertes. Le levier reste borné par `OPTIONS_MAX_DELTA_NOTIONAL_PCT` et
      la concentration par le plafond de pondération.
    - **Rebalancement en DEUX mécanismes disjoints** (voir la section dédiée
      plus bas) : un mécanisme JOURNALIER qui n'ouvre que des positions
      neuves, et un mécanisme SUR DÉPÔT SEC filtré par ε
      (`--rebalance-log-gap-threshold`, `OPTIONS_REBALANCE_LOG_GAP_THRESHOLD`)
      qui ne redimensionne une position déjà détenue que si son écart en log
      a suffisamment bougé depuis le dernier trade réel dessus.
    - Une nouvelle stratégie options s'ajoute de la même façon que pour les
      actions : fichier dans `backtest/strategies/`, classe héritant de
      `OptionsStrategy` (`backtest/strategies/options_base.py`), décorée par
      `@register_options_strategy("mon_nom")`.

### Stratégie `valuation_gap_multiples_options` (convergence long terme)

Seconde stratégie options, à côté de `valuation_gap_options`. Elle compare la
valorisation théorique issue des **multiples sectoriels seuls** à la
valorisation boursière, et parie sur la convergence de la seconde vers la
première à horizon 2 ans. Les deux stratégies partagent désormais l'échéance
2 ans, le roulement à 9 mois et les stops sur le sous-jacent ; ce qui les
sépare tient au signal, au strike et au traitement d'un écart refermé :

```bash
python 10_backtest_options.py --strategy valuation_gap_multiples_options --start-date 2015-01-01
```

| | `valuation_gap_options` | `valuation_gap_multiples_options` |
|---|---|---|
| Signal | multiples, **DCF en repli** | **multiples seuls** (repli DCF écarté) |
| Mesure de l'écart | ±20% rapporté au cours | **100 × ln(théorique/cours)**, symétrique |
| Strike | ATM | **à mi-chemin théorique/cours** |
| Échéance | 2 ans, roulée à 9 mois | 2 ans, roulée à 9 mois |
| Stop-loss | −20% du cours du sous-jacent | **−25%** du cours du sous-jacent |
| Take-profit | +80% du cours du sous-jacent | **80% du chemin vers la théorique** |
| Écart refermé | position gelée jusqu'au **roulement à 9 mois**, où elle est clôturée | **vendue au trimestre suivant** |
| Volatilité de repricing | figée à l'entrée | **suivie au jour le jour** |

Comme la valorisation boursière (nb d'actions × cours) et la valorisation
théorique (nb d'actions × valeur théorique par action) portent sur le même
nombre d'actions, leur rapport se calcule directement par action : la
stratégie lit les colonnes de `06b_calcul_valorisation_combinee.py` sans
reconstruire de capitalisation.

**Pourquoi les stops portent sur le sous-jacent et non sur la prime** (le
raisonnement qui vaut maintenant pour les deux stratégies, cf.
`OPTIONS_STOP_BASIS`). Sur le cas type de cette stratégie (théorique 120,
cours 100 → strike 110, 2 ans, vol 30%), le levier effectif est de 3,5x : un
stop à −25% de la prime se
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

Ces réglages de moteur font partie de la thèse de la stratégie : elle les
déclare (attribut de classe `engine_defaults`) et `10_backtest_options.py` les
applique automatiquement, sauf si l'option correspondante est passée
explicitement en ligne de commande (`--stop-basis`, `--roll-when-days-left`,
`--no-exit-when-signal-lost`, `--target-tenor-days`...).

Côté moteur (`backtest/options_engine.py`), l'échéance 2 ans, le roulement
(`OPTIONS_ROLL_WHEN_DAYS_LEFT`) et les stops sur le sous-jacent
(`OPTIONS_STOP_BASIS`) sont actifs par défaut, donc pour les deux stratégies ;
la vente sur perte de signal et la réévaluation quotidienne restent
**optionnelles et désactivées par défaut**, propres à
`valuation_gap_multiples_options`.

Résultats sauvegardés sous `data/backtest_options/<run_id>/` (mêmes fichiers
que le backtest actions).

### Optimisation stop-loss / take-profit (11_optimize_options_stops.py)

Rejoue le backtest pour **tous les binômes** (stop_loss_pct, take_profit_pct)
d'une grille, sur des données chargées une seule fois, et classe les
résultats par une métrique choisie (Sharpe par défaut) :

```bash
python 11_optimize_options_stops.py
python 11_optimize_options_stops.py --strategy valuation_gap_multiples_options
python 11_optimize_options_stops.py --start-date 2015-01-01 --objective calmar_ratio
python 11_optimize_options_stops.py \
    --stop-loss-grid -10 -15 -20 -25 -30 -35 -40 -50 \
    --take-profit-grid 20 30 40 60 80 100 150 200
python 11_optimize_options_stops.py --workers 4   # parallélise sur des process (fork)
```

Tous les autres réglages moteur (échéance, base des stops, roulement...)
restent ceux résolus par `10_backtest_options.resolve_engine_settings` --
donc ceux imposés par `engine_defaults` de la stratégie choisie, sauf
override explicite. Seuls stop_loss_pct/take_profit_pct varient d'une
combinaison à l'autre.

Sorties :
    - `data/backtest_options/optimize_<stratégie>_<horodatage>.csv` : une
      ligne par binôme, toutes les métriques (Sharpe, Sortino, Calmar, CAGR,
      max drawdown, profit factor, win rate, nombre de trades...).
    - Console : tableau des `--top-n` meilleures combinaisons, et une
      heatmap texte (stop_loss en lignes, take_profit en colonnes) pour
      repérer un plateau ou un optimum en bord de grille.

`--min-trades` (15 par défaut) écarte du CLASSEMENT (pas du CSV) les
combinaisons trop peu tradées pour être statistiquement significatives : un
stop si serré qu'il ne laisse que 3 trades peut afficher un Sharpe flatteur
par pur hasard d'échantillon. Si la meilleure combinaison retenue tombe en
bord de grille, un avertissement invite à élargir `--stop-loss-grid`/
`--take-profit-grid` -- un optimum au bord de la plage testée n'est pas prouvé
être un optimum réel.

**Séparation apprentissage / test** (`--train-fraction`, 0,60 par défaut). Le
classement se fait sur `train_<objectif>`, calculé sur les 60 % initiaux de
l'historique ; `test_<objectif>`, calculé sur le reste, est affiché à côté et
écrit au CSV — et c'est lui qui compte. Un grid-search classé sur l'historique
complet retient, par construction, la combinaison qui colle le mieux à cet
historique-là : avec 64 combinaisons sur une seule période, le meilleur Sharpe
est en grande partie du bruit sélectionné. `--min-trades` et l'avertissement de
bord de grille écartent les artefacts d'échantillon trop petit, pas le
sur-ajustement ; le seul test qui le détecte est de regarder ce que la
combinaison retenue fait sur des données qui n'ont pas servi à la choisir.

Le portefeuille n'est **pas** remis à zéro au changement de période : le run
est unique et on découpe sa courbe de NAV — les positions ouvertes à la fin de
l'apprentissage sont bien celles qu'on porterait en entrant dans le test. Un
avertissement explicite est émis si l'objectif s'effondre hors échantillon.
`--no-walk-forward` rétablit le classement in-sample.

`11b_optimize_rebalance_threshold.py` applique exactement la même séparation.

### Score d'écart symétrique (base `log`)

L'écart de `valuation_gap_multiples_options` se mesure en **points de log** :
`100 × ln(théorique / cours)`, seuil d'entrée à `100 × ln(1,20) ≈ 18,23`.

Les deux conventions en pourcentage conservées (`--strategy-param
gap_basis=theoretical` ou `close`, pour rejouer un run ancien) sont
**asymétriques en miroir l'une de l'autre** — aucune des deux n'est neutre :

| | base `theoretical` | base `close` | base `log` |
|---|---|---|---|
| côté CALL | borné à **+100%** | non borné | non borné |
| côté PUT | non borné | borné à **−100%** | non borné |
| symétrique ? | non | non | **oui** |

L'asymétrie ne déformait pas que le classement, elle déformait la
**sélection** : à seuil 20% en base `theoretical`, un CALL exigeait un cours à
≤80% de la théorique (|ln| = 0,223) là où un PUT se contentait de ≥120%
(|ln| = 0,182). Le PUT était donc structurellement plus facile à qualifier —
et recevait en prime une conviction plus élevée du côté non borné. Le livre
penchait vers le PUT par convention de calcul, pas par signal.

**Ce que le passage au log ne change pas** : la correction d'inflation reste
asymétrique, et c'est voulu. La valeur théorique est nominale, donc la
convergence se fait vers `V × (1+π)^T` — la dérive des prix aide un CALL et
durcit la thèse d'un PUT. En log cette correction devient simplement additive
(`+ T × ln(1+π)`, cf. `base.inflation_adjusted_log_gap`). Le log supprime
l'asymétrie **de convention** ; il laisse intacte celle qui a un contenu
**économique**.

### Refonte du rebalancement : deux mécanismes disjoints

Un seul point d'entrée (`_rebalance`) recalculait auparavant les poids de
TOUTES les positions détenues à CHAQUE occasion (dépôt SEC comme
réévaluation quotidienne), produisant un churn massif sans lien avec de
l'information nouvelle. Le moteur (`backtest/options_engine.py`) sépare
maintenant deux mécanismes à des périmètres volontairement différents :

1. **Mécanisme JOURNALIER** (`_rebalance_daily`, jours SANS nouveau dépôt,
   `daily_rebalance=True`) : uniquement des **ouvertures** de symboles
   nouvellement éligibles, à leur poids ISOLÉ (`conviction_X` / somme des
   convictions de tous les éligibles du jour — ce que la stratégie calcule
   déjà via `base.capped_weights`). **Aucun redimensionnement** sur une
   position déjà détenue : un mouvement de cours pur, sans nouvelle
   publication, ouvre une opportunité neuve mais ne justifie pas de retoucher
   une thèse déjà engagée.

2. **Mécanisme SUR DÉPÔT SEC** (`_rebalance_on_signals`, 10-K/10-Q/8-K) :
   scopé au(x) SEUL(S) symbole(s) dont le dépôt du jour vient de mettre à
   jour `known_signals` — jamais aux autres positions détenues, même si une
   renormalisation globale les aurait affectées. Nouveau candidat : ouvert
   sans condition. Position déjà détenue : redimensionnée seulement si
   l'écart en log a bougé de plus de `rebalance_log_gap_threshold` (ε,
   `OPTIONS_REBALANCE_LOG_GAP_THRESHOLD`, 0.15 par défaut = un rapport
   théorique/cours qui a bougé d'un facteur e^0.15 ≈ 1.16 depuis le dernier
   trade) **depuis le dernier TRADE RÉEL sur cette position**
   (`OptionPosition.last_rebalance_log_gap`), pas depuis le dernier signal
   connu. En dessous du seuil, `known_signals` est mis à jour (les calculs
   futurs utilisent la nouvelle valorisation théorique) mais aucun ordre
   n'est généré et la référence ne bouge pas : le seuil suivant continue de
   se mesurer depuis ce dernier trade, pour qu'une dérive lente qui ne
   franchit jamais ε d'un coup mais s'accumule sur plusieurs dépôts
   successifs finisse par se rattraper.

Dans les deux cas, `exit_when_signal_lost` (sortie sur perte de signal ou
retournement de direction) reste actif à son périmètre habituel : ce n'est
pas un redimensionnement, c'est une décision d'exposition orthogonale à ce
que ces deux mécanismes contraignent. `min_resize_relative_pct` reste un
**second filet** après ε : un changement de conviction qui passe ε peut
encore ne se traduire que par un micro-ajustement en nombre de contrats.

**Ce même filtre borne aussi `_deploy_idle_cash`** (le redéploiement du cash
oisif, config.OPTIONS_MIN_DEPLOYMENT_PCT — **désactivé par défaut depuis
l'audit**, voir plus bas). Cette méthode est appelée CHAQUE
jour de bourse, sans mémoire d'un renfort récent : sans le filtre, une
position à peine sous le plancher de déploiement se fait renforcer d'un ou
deux contrats CHAQUE JOUR, indéfiniment, en payant plein tarif de slippage
et de commission minimum à chaque fois — sur un backtest de plusieurs années,
cette friction pure peut consommer la quasi-totalité du capital initial,
sans qu'aucune thèse n'ait perdu quoi que ce soit. Sur un scénario de test à
2 positions / 120 jours, ce filtre à 15% fait passer les renforcements de
AAA de 18 micro-ajustements distincts à... 1 (l'ouverture initiale). `--min-
resize-relative-pct 0` rétablit l'ancien comportement (tout changement,
même infime, déclenche un ordre) pour les deux mécanismes à la fois.

**Provenance des positions** (`trades.parquet`, colonne `open_reason`) :
`"rebalance"` (dépôt SEC), `"rebalance_daily"` (mécanisme journalier) ou
`"roll"` (réouverture après roulement) — distinct d'`exit_reason`, qui décrit
toujours la SORTIE et reste inchangé par cette refonte.

```bash
python 10_backtest_options.py --strategy valuation_gap_multiples_options --rebalance-log-gap-threshold 0.20
```

`11b_optimize_rebalance_threshold.py` (calqué sur
`11_optimize_options_stops.py`) rejoue le backtest pour chaque valeur de ε
d'une grille et rapporte, par ε : `total_friction_dollar`/
`total_friction_pct_of_initial`, `num_trades` total et par motif,
`num_rebalance_trades` (le churn RÉSIDUEL du mécanisme sur dépôt SEC après
filtrage — le chiffre qui dit si ε a réellement coupé le churn), `cagr_pct`,
`sharpe_ratio`, `max_drawdown_pct` :

```bash
python 11b_optimize_rebalance_threshold.py
python 11b_optimize_rebalance_threshold.py --epsilon-grid 0 0.05 0.10 0.15 0.20 0.30
```

### Take-profit par fraction de convergence

Pour les stratégies qui visent une valeur théorique, le take-profit n'est plus
un seuil fixe mais une **fraction du chemin parcouru** vers cette valeur
(`OPTIONS_TAKE_PROFIT_CONVERGENCE_FRACTION`, 0,80 par défaut).

Un seuil fixe n'est pas atteignable de la même façon des deux côtés, par pure
géométrie. Au seuil d'entrée, la convergence **complète** vaut :

| | position entrée à | convergence complète | take-profit fixe +25% |
|---|---|---|---|
| CALL | cours = 83,3% de V | **+20,0%** de cours | atteignable |
| PUT | cours = 120% de V | **−16,7%** de cours | exige un **dépassement** |

Le take-profit fixe (+30%, ou même +25%) ne se déclenchait donc pratiquement
jamais côté PUT : la jambe ne savait pas prendre ses gains. À 0,80, un PUT
entré à P = 1,20 V sort à P = 1,04 V (−13,3% de cours) et un CALL entré à
P = 0,83 V sort à P = 0,96 V (+20,0%) — atteignable des deux côtés, et
proportionnel à l'écart réellement constaté à l'entrée.

Le **stop-loss**, lui, reste un seuil fixe en % du sous-jacent : il décrit une
perte, pas un degré d'avancement de la thèse. `valuation_gap_options` (ATM,
sans valeur théorique cible) garde `--take-profit-pct` inchangé.

Conséquence pour l'optimisation : sur une stratégie de convergence,
`take_profit_pct` est **inerte**. `11_optimize_options_stops.py` le détecte
(`OptionsStrategy.targets_convergence`) et balaie la fraction de convergence
à la place — `--take-profit-mode` force le choix, et le mode retenu est
reporté dans le CSV.

### Neutralité directionnelle de la file d'exécution

Les achats du jour s'exécutent en **alternance CALL / PUT**, chaque côté trié
par montant décroissant (les ventes passent toujours en premier : leur produit
finance les achats du même jour).

Auparavant l'ordre d'exécution suivait l'ordre d'insertion, c'est-à-dire le
classement par conviction de la stratégie. Or quand le cash s'épuise en cours
de file, `_affordable` tronque **les derniers servis** : un classement qui
place systématiquement une direction en tête la finance intégralement et
laisse l'autre absorber toute la troncature. Le classement par conviction
devenait ainsi un **filtre directionnel**, alors qu'il n'est censé exprimer
qu'une conviction.

L'alternance est **déterministe** et non aléatoire : un mélange par tirage
rendrait un run non reproductible d'une exécution à l'autre, ce qui
interdirait toute comparaison de paramètres. Une direction vide ne réserve
rien — l'autre prend tout le budget.

### Plafond par ordre (`OPTIONS_MAX_TRADE_PCT_OF_NAV`)

Aucun **ordre d'achat** ne décaisse plus de 10 % du NAV (frais inclus).
C'est un plafond **par ordre, pas par position** : une ligne peut le dépasser
en cumulant plusieurs renforcements sur des jours différents — c'est
`BACKTEST_MAX_WEIGHT_PER_POSITION_PCT` qui borne la taille d'une position.
Jamais appliqué aux **ventes** : plafonner une sortie interdirait de liquider
une position devenue grosse, exactement quand il faut pouvoir sortir. Jamais
appliqué non plus au **roulement** : celui-ci ne crée pas une position, il en
renouvelle une, et le plafonner revenait à plafonner une continuation.

Le plafond était auparavant un montant ABSOLU (15 000 $), calibré sur un
capital de 1 000 000 $ sans le dire et ne suivant ni la croissance ni la
baisse du portefeuille. Tant que le plancher de primes reconstruisait les
positions jour après jour, ce sous-dimensionnement se rattrapait tout seul ;
le plancher désactivé, il devenait la contrainte qui mord — l'ouverture visée
par le dimensionnement par delta était ramenée à 1/25e de sa taille.
`OPTIONS_MAX_TRADE_DOLLAR` existe toujours comme plafond absolu additionnel,
désactivé par défaut.

Les ordres ramenés à ce plafond sont comptés séparément des ordres tronqués
faute de cash (`capped_orders_count` vs `truncated_orders_count`) : les deux
causes n'ont rien à voir et les confondre rendrait le diagnostic illisible.
Les ordres **purement abandonnés** (frais excessifs, cash nul, plafond de
levier…) sont eux aussi comptés à part, par motif, dans
`dropped_orders_by_reason` — un ordre abandonné ne laisse aucune trace dans
`trades.parquet` ni dans l'equity_curve, et un run pouvait rester
intégralement en cash sans que rien ne le signale.

### Dimensionnement : une seule base, deux plafonds qui mordent vraiment

Voir `RAPPORT_AUDIT.md` pour les mesures. Trois réglages ont changé de valeur
par défaut, et il faut les connaître pour lire un run.

**Le plancher de primes est désactivé** (`OPTIONS_MIN_DEPLOYMENT_PCT = 0`).
Il exigeait que 25 % du NAV soit investi *en primes* — or une prime baisse
quand la thèse échoue, donc le plancher se trouvait violé précisément quand la
position perdait, et le moteur rachetait. Moins l'option valait cher, plus un
dollar achetait de contrats : le renforcement accélérait à mesure que la thèse
se dégradait, et une position gagnante n'était au contraire jamais renforcée.
Sur un CALL dont le sous-jacent perd 47 %, à signal et chemin de cours
identiques : **NAV finale 976 766 $ sans le plancher, 437 074 $ avec**. Le
mécanisme reste disponible (`--min-deployment-pct 25`) pour rejouer un run
ancien.

**Le plafond de levier borne désormais le portefeuille**, et plus seulement
le redéploiement du cash oisif. Il est vérifié à l'ouverture de chaque ordre
(`_open_or_resize`) *et* réévalué chaque jour : au-delà de
`OPTIONS_MAX_DELTA_NOTIONAL_PCT` majoré de `OPTIONS_DELEVER_TOLERANCE_PCT`,
toutes les positions sont réduites au même prorata. La bande de tolérance
évite de vendre trois contrats à chaque oscillation du marché ; la réduction
au prorata préserve la hiérarchie décidée par la stratégie, le plafond
n'exprimant aucune opinion sur la thèse à abandonner.

**Le roulement conserve l'exposition établie.** `target_dollar` enregistre
désormais l'exposition *réellement prise* — pas celle qui avait été demandée
avant que le cash, le plafond par ordre ou le plafond de levier ne rognent
l'ordre — et suit les renforcements. Sans quoi le roulement rejouait une cible
sans rapport avec la position détenue, dans un sens comme dans l'autre.

### Plancher de delta (`OPTIONS_MIN_DELTA_FOR_SIZING`)

Le moteur convertit une exposition $ visée en contrats par
`nb = target_dollar / (|delta| x spot x multiplicateur)`. Cette expression
**diverge** quand le delta tend vers zéro : à delta 0,01 elle attribue cent
fois plus de contrats qu'à delta 1,0, pour la même exposition notionnelle
affichée. Le seul garde-fou était `abs(delta) < 1e-6`, qui protège d'un
`OverflowError` mais pas de l'absurdité économique.

Sans stop-loss, les positions perdantes survivent et dérivent loin hors de la
monnaie ; leur delta et leur prime tendent vers 0, et chaque renforcement leur
attribue un nombre de contrats colossal. Mesuré sur deux runs identiques à un
paramètre près (`--stop-loss-pct -1000`) : **commissions ×26** (17 910 $ →
470 888 $) pendant que le **slippage baissait**. La commission suit le NOMBRE
de contrats, le slippage leur VALEUR — le volume avait explosé sans que la
valeur engagée bouge.

Ni le plafond par ordre ni le plafond de levier n'y suffisaient : le premier
est en dollars (sur une option à 0,02 $, 10 % d'un NAV de 1 M$ autorise
50 000 contrats), et le second a exactement la même forme que le
dimensionnement, donc il diverge avec lui.

**Le plancher ne s'applique qu'aux ACHATS.** Une position sous le plancher
doit rester vendable — par stop-loss, perte de signal, roulement, expiration
ou réduction. Le test le vérifie explicitement : le plancher est évalué
*après* le calcul du delta de contrats, uniquement dans la branche « achat ».

### Intérêts sur le cash oisif (`OPTIONS_CREDIT_IDLE_CASH`)

Le cash est capitalisé au taux sans risque de l'année, sur les jours
calendaires écoulés (base 365 : un week-end rapporte). Cette stratégie porte
en moyenne **74 % de cash** — le dimensionnement par delta n'engage qu'une
prime, soit une fraction de l'exposition — et le laisser stérile la pénalisait
pour une raison étrangère à la thèse.

C'est aussi ce biais qui rendait `OPTIONS_MIN_DEPLOYMENT_PCT` tentant : le
plancher de primes ne faisait que compenser un manque à gagner artificiel, en
payant frais et slippage pour le faire.

Les intérêts sont publiés à part (`total_cash_interest_dollar`, colonne
`total_cash_interest` de l'`equity_curve`) pour qu'une performance portée par
les taux ne se confonde pas avec une performance portée par la thèse — et
`put_call_analysis` les retire du NAV avant d'attribuer quoi que ce soit à une
jambe, sinon sa réconciliation ne boucle plus.

### Durée de détention minimale (`OPTIONS_MIN_HOLDING_DAYS`)

Les contrats sont achetés à 730 jours d'échéance, mais la durée de détention
médiane d'une sortie `signal_lost` était de **79 jours**, avec un minimum
mesuré à **un jour**. Tous motifs confondus, la moyenne est de 193 jours : la
stratégie consommait 26 % de l'optionalité qu'elle achète et jetait le reste.

Ne s'applique **jamais** aux motifs `stop_loss`, `take_profit`, `roll`,
`expiry` ni `data_gap` : un garde-fou de risque ou une échéance ne se négocie
pas contre un calendrier. À périmètre égal (`--exit-when-signal-lost` des deux
côtés), `--min-holding-days 180` fait passer la médiane `signal_lost` de 87 à
202 jours, et retire 24 % des trades comme de la friction.

### Diagnostics du dimensionnement

`metrics.json` porte désormais `total_contracts_traded`,
`max_contracts_single_order` (+ symbole et date), `min_delta_at_sizing`,
`days_above_delta_cap` / `pct_days_above_delta_cap` /
`median_excess_above_delta_cap_pct`.

Ce sont les chiffres qui auraient rendu les deux défauts ci-dessus visibles
immédiatement : le **volume de contrats** est la seule grandeur qui distingue
« j'engage plus de capital » de « j'achète des milliers d'options mortes »
— la friction totale, elle, ne le dit pas, puisque ses deux composantes
bougent alors en sens inverse.

### Hystérésis entre l'entrée et la sortie (`OPTIONS_EXIT_THRESHOLD_RATIO`)

Entrée et sortie ne partagent plus le même seuil. Une position s'ouvre à
`|écart| >= 18,23` points de log et n'est vendue qu'une fois l'écart repassé
sous `0,70 x 18,23 = 12,76` — un rapport théorique/cours de 1,136 au lieu de
1,20.

Avec la réévaluation quotidienne, un seuil unique faisait qu'un titre
oscillant autour de la barre déclenchait des allers-retours complets, chacun
payant deux fois le slippage, deux commissions minimum, et **abandonnant toute
la valeur temps déjà achetée** sur un contrat à deux ans. Le filtre ε et
`min_resize_relative_pct` protègent tous les deux le *redimensionnement* et lui
seul, jamais la décision d'ouvrir ou de fermer — qui est pourtant la plus
chère. Une convergence de 30 % n'est pas une raison de solder un pari à deux
ans : c'est le début de ce qu'on attendait. `--strategy-param
exit_threshold_ratio=1` rétablit l'ancien comportement.

### Coût de portage et exposition vega dans les métriques

`metrics.json` porte maintenant `total_theta_decay_dollar` /
`total_theta_decay_pct_of_initial` et `avg_vega_notional_pct`, et
l'`equity_curve` les colonnes `theta_per_day` / `vega_notional`.

Le theta est le **seuil que la thèse doit battre avant de gagner quoi que ce
soit** : mesuré sur un sous-jacent quasi plat, acheter des options à deux ans
coûte ~18 % de NAV par deux ans en pure valeur temps, et aucune métrique ne
l'exprimait. Le vega, lui, mesure la part du P&L que ce backtest **ne simule
pas du tout** : la stratégie achetant CALL *et* PUT, elle est longue de vega
des deux côtés, et le repricing ne suit jamais l'implicite.

### Journal des exécutions (`executions.parquet`)

`trades.parquet` n'enregistre **que les ventes**, et présente chacune comme un
aller-retour : `entry_date` y est la date de **première** ouverture de la
position et `entry_price` le prix de revient **moyen** de tous les achats qui
l'ont constituée. Or une position se construit en plusieurs fois
(renforcement au rebalancement, redéploiement du cash oisif) — un trade peut
donc légitimement solder **285 contrats** alors que **94 seulement** ont été
achetés à son `entry_date`. La comptabilité du moteur est juste (aucun contrat
n'est vendu sans avoir été acheté), mais elle était **invérifiable depuis les
sorties** : aucun achat intermédiaire n'apparaissait nulle part.

`executions.parquet` corrige ça : **une ligne par fill, achat comme vente**,
avec `contracts`, `price`, `cash_flow` (signé, frais inclus), `commission`,
`slippage` et `reason`. Trois invariants en découlent, tous testés :

| invariant | vérifie |
|---|---|
| Σ achats − Σ ventes = contrats encore détenus, par symbole | rien n'est vendu sans avoir été acheté |
| capital initial + Σ `cash_flow` = cash final | le journal explique tout le mouvement de cash |
| Σ `commission` / Σ `slippage` = totaux du moteur | aucun fill n'échappe à la friction |

**Bug corrigé au passage** : `_deploy_idle_cash` (le redéploiement du cash
oisif, appelé **chaque jour de bourse**) débitait le cash **sans jamais
comptabiliser sa friction**. `total_friction_dollar` sous-estimait donc le
coût réel sur le chemin de code le plus actif du moteur — mesuré à **44% de
la friction totale** sur un run de test. Les coûts et le journal passent
désormais par un point d'entrée unique (`_record_fill`), ce qui rend cet
oubli structurellement impossible.

### Diagnostic de friction (13_diagnostic_friction.py)

Le moteur accumule désormais la friction payée, décomposée, et la publie jour
par jour dans l'equity_curve (`total_commission`, `total_slippage`) comme en
cumul dans `metrics.json` (`total_friction_dollar`,
`total_friction_pct_of_initial`). Le slippage est compté **des deux côtés** :
la prime est payée majorée à l'achat et encaissée minorée à la vente.

`13_diagnostic_friction.py` rejoue la stratégie sur les 4 combinaisons du plan
`slippage × {réel, 0}` par `rebalancement quotidien × {activé, désactivé}` et
sépare les trois postes :

```bash
python 13_diagnostic_friction.py --strategy valuation_gap_multiples_options --start-date 2015-01-01
```

    thèse    = rendement du run slippage 0 + rebalancement désactivé
    friction = écart imputable au seul slippage, à rebalancement égal
    churn    = écart imputable au seul rebalancement quotidien, à slippage égal

Les deux ne se corrigent pas de la même façon — la friction en tradant moins
gros ou moins souvent, le churn en ne réagissant qu'aux publications — d'où
l'intérêt de ne pas les confondre dans un seul chiffre de perte. Les
commissions et frais tiers restent payés dans **les quatre cases** (les
annuler ne décrirait plus aucun courtier réel) : la colonne « pure thèse » est
un plafond, pas un contrefactuel atteignable.

### Décomposition CALL / PUT (12_analyse_put_call.py)

Le résumé de `10_backtest_options.py` agrège les deux paris en un seul NAV :
il dit combien la stratégie gagne, jamais **laquelle des deux jambes** le
gagne. Or un call parie sur une convergence à la hausse et un put sur la
baisse d'un titre survalorisé : sur un marché haussier de long terme, la
seconde peut saigner en silence pendant que la première la masque. Ce script
rouvre un run déjà sauvegardé et sépare tout ce qui peut l'être :

```bash
python 10_backtest_options.py --strategy valuation_gap_multiples_options --start-date 2015-01-01
python 12_analyse_put_call.py            # dernier run de cette stratégie
python 12_analyse_put_call.py --run-id 20260810_143000 --export
```

Sept tableaux :
    1. **Métriques habituelles en trois colonnes** ALL / CALL / PUT (mêmes
       définitions que `metrics.py`, donc lisibles ligne à ligne).
    2. **Répartition des valorisations** : prime immobilisée par chaque jambe
       (moyenne, pic, part du portefeuille, jours d'exposition).
    3. **Répartition des volumes de trade** : nombre de trades, contrats,
       montants décaissés/encaissés, P&L réalisé, et leurs parts en %.
    4. **Coûts d'exécution par jambe** : commission (frais tiers ORF/CAT/OCC/
       TAF/SEC inclus), slippage, friction totale et friction rapportée au
       volume échangé. Lus dans `executions.parquet`, donc **frais d'achat
       inclus** — invisibles dans `trades.parquet`, qui n'a que les ventes.
    5. **Coûts d'exécution par motif d'ordre** : dit quel mécanisme du moteur
       consomme la friction (`deploy_idle_cash`, `rebalance`,
       `rebalance_daily`, `roll`, `stop_loss`…).
    6. **Réconciliation des quantités** : contrats achetés / vendus / encore
       détenus par jambe, dont l'`ecart` doit valoir zéro.
    7. **P&L par (jambe, motif de sortie)** puis **P&L réalisé par année et
       par jambe** — les tableaux qui localisent réellement la perte : un
       `stop_loss` négatif dit que les seuils coupent au mauvais moment, une
       `expiry` négative que le pari ne se réalise pas dans le temps imparti.

Les tableaux 4 à 6 sont omis (avec un avertissement) pour un run antérieur à
`executions.parquet`.

`--export` écrit en plus ces tableaux en CSV dans le répertoire du run, avec
la contribution cumulée quotidienne de chaque jambe (pour voir *quand* une
jambe décroche, ce qu'aucun agrégat de fin de période ne montre).

**Ce que mesure une colonne de jambe.** La contribution d'une jambe est son
P&L réalisé cumulé + son P&L latent du jour. L'identité
`NAV = capital initial + réalisé + latent` est **exacte** (le moteur
incorpore commission et slippage à `entry_premium`), et le script la vérifie
à chaque exécution plutôt que de la supposer — il affiche l'écart de
réconciliation et avertit s'il dépasse 0,01% du capital. Pour rendre Sharpe
et drawdown calculables par jambe, chaque jambe est rejouée comme un
portefeuille partant du **capital initial complet** et ne recevant que son
propre P&L : ce n'est donc pas « ce qu'aurait donné une stratégie qui
n'achète que des calls » (le dimensionnement aurait été tout autre), mais la
**contribution** de la jambe, dans une unité comparable à l'agrégat. Seule la
ligne `leg_pnl_dollar` s'additionne entre CALL et PUT.

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

## Biais et limites connus

Ces points sont **documentés mais NON corrigés** : soit la donnée nécessaire
n'est pas disponible, soit la correction est un rattrapage long laissé à ta
décision. Ils vont tous dans le même sens — ils rendent les résultats de
backtest **plus optimistes** que la réalité. À garder en tête avant de
conclure quoi que ce soit d'un run.

### Médianes sectorielles calculées sur les survivants (06b)

Les multiples sont calculés par millésime de publication, et — depuis l'audit —
`compute_pit_sector_multiples` restreint en plus la médiane de chaque ligne aux
pairs **déjà déposés à sa propre `filed_date`**. Le regroupement par millésime
supprimait le mélange *entre* millésimes ; il laissait intact le décalage *à
l'intérieur* d'un millésime, où les 10-K s'étalent sur près de trois mois et où
le multiple d'un pair porte son cours à *sa* date de dépôt. La médiane servant
à valoriser un déposant de février intégrait donc les cours de ses pairs
jusqu'en avril. Conséquence assumée du correctif : les premiers déposants d'un
millésime voient moins de pairs, parfois moins que le minimum requis, et
retombent alors sur le repli DCF — c'est la réalité de l'information
disponible à cette date. Le nombre de pairs réellement utilisés est reporté
**par ligne** (`n_peers`, propagé jusqu'au signal).

Reste le biais de composition, lui **non corrigé** : les multiples viennent de
`multiples.parquet`, qui ne contient que l'univers **actuel** : les médianes
sectorielles de 2012 sont établies sur les seules entreprises encore
présentes dans l'indice aujourd'hui.

Les entreprises disparues (faillite, rachat, sortie d'indice) étaient en
moyenne moins bien valorisées que les survivantes : la médiane sectorielle
historique est donc probablement **surestimée**, et avec elle les
valorisations théoriques et les écarts calculés contre elles.

`06b` journalise le nombre de pairs par (secteur, millésime) et, si `01b` a
tourné, avertit des tickers radiés absents de `multiples.parquet`.

**Correction complète** (longue : plusieurs milliers de requêtes SEC et IBKR) :

```bash
python 01b_historique_univers_sp500.py                              # produit UNIVERSE_FULL_FILE
python 03b_recuperation_cours_quotidiens.py --tickers data/universe/sp500_universe_full.csv
python 04_recuperation_10k.py  --tickers data/universe/sp500_universe_full.csv --force-refresh
python 04b_recuperation_10q.py --tickers data/universe/sp500_universe_full.csv --force-refresh
python 05_calcul_multiples.py && python 07_calcul_dcf.py && python 06b_calcul_valorisation_combinee.py
```

### WACC indexé sur la courbe de taux (07)

*(Corrigé — le WACC était auparavant figé de 2010 à 2026.)*

`config.sector_dcf_params(secteur, année)` calcule désormais
`wacc = taux sans risque de l'année + prime de risque du secteur`, la prime
étant celle implicite dans `SECTOR_DCF_PARAMS` (WACC calibré moins
`WACC_CALIBRATION_RISK_FREE_RATE`, 4 %). L'année retenue est celle du **dépôt
SEC** — c'est au moment où l'information devient publique que le marché
actualise.

Un WACC figé était un pari de taux non voulu, et systématiquement à
contretemps : le dépôt connaît pourtant la courbe réelle
(`RISK_FREE_RATE_BY_YEAR`, de 0,05 % à 5,3 %) et s'en sert déjà pour pricer les
options et calculer le Sharpe, mais pas pour actualiser les flux — alors que
le taux est le premier déterminant d'un DCF.

| | WACC figé à 10 % | Effet sur la valeur théorique | Conséquence |
|---|---|---|---|
| 2020-2021 (taux ~0 %) | trop **haut** | sous-estimée | excès de **PUT** |
| 2023-2024 (taux ~5 %) | trop **bas** | surestimée | excès de **CALL** |

Un plancher (`DCF_MIN_WACC_MINUS_TERMINAL_GROWTH`, 3 points) garde le WACC
suffisamment au-dessus de la croissance terminale : en 2011-2015 le
sans-risque tombe à 0,05 %, et la valeur terminale
`FCF x (1+g) / (wacc - g)` cesse d'être une estimation dès que le dénominateur
s'approche de zéro. `DCF_WACC_FOLLOWS_RATE_CURVE = False` rétablit le WACC
figé.

**Régénération nécessaire** : `python 07_calcul_dcf.py` puis
`python 06b_calcul_valorisation_combinee.py`.

### Secteur GICS rétroactif (05, 07)

*(Corrigé pour les médianes sectorielles de 06b — voir la section suivante.
Ce qui suit ne vaut plus que pour 07.)*

La colonne `sector` produite par 02 est la classification GICS
**d'aujourd'hui**. Elle pilote le WACC et les taux de croissance du DCF
(`config.SECTOR_DCF_PARAMS`), les multiples jugés pertinents
(`config.SECTOR_MULTIPLES`), le rendement du dividende du pricing d'options
(`config.SECTOR_DIVIDEND_YIELD`) et l'exclusion du DCF
(`config.SECTORS_SANS_DCF`).

`06b_calcul_valorisation_combinee.py` la ramène désormais au secteur d'époque
avant de composer ses groupes de pairs (`sector_history.sector_asof`), mais
`07_calcul_dcf.py` applique toujours le secteur actuel à tout l'historique :
une entreprise reclassée depuis se voit encore appliquer les mauvaises
hypothèses de DCF sur sa partie ancienne.

### Composition point-in-time du groupe de pairs (06b)

Les médianes sectorielles de `06b` sont calculées sur les pairs tels qu'ils
étaient **à la `filed_date` de chaque ligne**, sur deux dimensions
(`sector_history.py`) :

- **Appartenance à l'indice.** Un pair n'est retenu que s'il était membre du
  S&P 500 à cette date (spans de `01b`). Cela écarte les entreprises entrées
  depuis — une entrée dans l'indice récompense en général un parcours
  boursier, donc les laisser peser sur une médiane de 2012 la pousse vers le
  haut — et, une fois les données backfillées, remet les radiées dans les
  millésimes où elles comptaient.
- **Secteur d'époque.** Les remaniements GICS documentés sont rejoués à
  l'envers : immobilier sorti des financières (2016), création de
  Communication Services (2018 — Alphabet et Meta étaient en technologie
  avant), paiements passés en financières (2023). La colonne `sector` du
  parquet de sortie porte le secteur d'alors, `sector_current` celui
  d'aujourd'hui.

`--no-point-in-time-peers` rétablit l'ancien comportement, pour chiffrer
l'écart entre les deux.

**Ce que le code ne peut pas faire seul.** Restreindre les pairs aux membres
d'alors ne crée pas les lignes manquantes : tant que `03b`/`04`/`04b` n'ont
pas été backfillés sur `sp500_universe_full.csv`, les radiées restent absentes
de `multiples.parquet`. La différence est que le trou est maintenant **mesuré**
— 06b journalise la couverture réelle de l'univers point-in-time par millésime
et avertit en dessous de 95 %.

```bash
python 01b_historique_univers_sp500.py
python 02_categoriser_secteurs.py --universe data/universe/sp500_universe_full.csv
python 03b_recuperation_cours_quotidiens.py --tickers data/universe/sp500_universe_full.csv
python 04_recuperation_10k.py  --tickers data/universe/sp500_universe_full.csv
python 04b_recuperation_10q.py --tickers data/universe/sp500_universe_full.csv
python 05_calcul_multiples.py && python 07_calcul_dcf.py
python 06b_calcul_valorisation_combinee.py
```

L'étape `02 --universe` n'est pas optionnelle : les entreprises radiées
n'ayant pas de secteur GICS (elles ne figurent plus dans la table Wikipédia
des membres actuels), sans elle le backfill coûte des milliers de requêtes SEC
pour des lignes que 06b écarte faute de secteur. `05` le signale.

Limite résiduelle : la table des changements de Wikipédia ne remonte qu'à
~1996-2000, et `sector_history.GICS_RECLASSIFICATIONS` ne couvre que les
remaniements structurels de la nomenclature — pas les reclassements
individuels au fil de l'eau, faute de source historique gratuite.

### Risque de volatilité non modélisé (10, backtest/options_pricing.py)

Le repricing quotidien des positions d'options se fait à volatilité **figée**
(`--vol-mode frozen`) ou suivant la volatilité **réalisée**
(`--vol-mode rolling`), jamais suivant la volatilité **implicite** : le
pipeline ne collecte aucune surface de volatilité historique.

Or une option longue est longue de vega. Un effondrement de l'implicite lui
fait perdre de l'argent même quand le sous-jacent va dans son sens, et ce P&L
n'apparaît nulle part dans les résultats. Les rendements du backtest options
sont donc une **borne optimiste** sur toute période de compression de
volatilité. `10_backtest_options.py` l'annonce au démarrage de chaque run.

Le pricing simulé corrige en revanche deux approximations qui allaient dans
un sens systématique : les dividendes (`config.SECTOR_DIVIDEND_YIELD`, faute
de donnée par titre) et le taux sans risque
(`config.RISK_FREE_RATE_BY_YEAR`).

### Cours IBKR sans dividendes (03, 03b)

`03_recuperation_cours.py` et `03b_recuperation_cours_quotidiens.py`
demandent `whatToShow="TRADES"` à IBKR : des cours de transaction, ajustés
des splits mais **pas des dividendes**. Le P&L du backtest actions ignore
donc les dividendes réinvestis, soit environ **2 %/an** de rendement manquant
sur le S&P 500 — davantage sur les secteurs à haut rendement, précisément
ceux qu'un signal « value » sélectionne.

Ce biais joue **contre** la stratégie (il la sous-estime), à l'inverse des
trois précédents. Il joue en revanche aussi contre l'indice de référence
reconstruit en équipondéré (`build_benchmark_series`), donc l'`alpha_pct`
reste à peu près comparable ; un `SPY` collecté par la même voie porterait la
même sous-estimation.

### Biais de survivance de l'univers (09, 10)

Si `01b_historique_univers_sp500.py` n'a jamais tourné, l'univers **actuel**
du S&P 500 est appliqué à toutes les dates passées. Les deux moteurs le
signalent par un avertissement au démarrage. Lance `01b` pour l'éliminer.

Même avec `01b`, la table des changements de Wikipédia ne remonte qu'à
~1996-2000 : une entreprise sortie de l'indice avant le début de ce suivi
n'apparaît pas.

### Exposition delta : plafonnée en continu, avec un jour de retard (10)

*(Corrigé — cette section décrivait auparavant une limite bien plus large :
le plafond n'était vérifié que dans `_deploy_idle_cash`, et 281 % de delta
notionnel ont été mesurés pour un plafond déclaré à 100 %.)*

`config.OPTIONS_MAX_DELTA_NOTIONAL_PCT` borne désormais le levier **à l'ordre
et à la position** : vérifié dans `_open_or_resize`, puis réévalué chaque jour
de bourse, avec réduction au prorata au-delà de la bande de tolérance
(`OPTIONS_DELEVER_TOLERANCE_PCT`).

Il reste un dépassement résiduel, et il est structurel : comme tout le reste
du moteur, le dé-levier est **décidé à la clôture de J et exécuté à
l'ouverture de J+1**. Entre les deux, le sous-jacent bouge et le NAV avec lui —
or le ratio a le NAV au dénominateur, si bien qu'un portefeuille qui perd voit
son levier monter sans avoir rien acheté. Sur un scénario adverse (sous-jacent
en baisse de 47 %), le maximum observé passe de 281 % à ~112 % pour un plafond
à 100 % et une tolérance de 10 %. La colonne `delta_notional_pct` de
l'`equity_curve` et les champs `avg_delta_notional_pct` /
`max_delta_notional_pct_observed` / `delever_events_count` de `metrics.json`
permettent de le constater run par run.

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
