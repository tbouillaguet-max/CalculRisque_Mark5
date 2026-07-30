# Rapport dynamique — pipeline options US

Dashboard Streamlit à deux pages, lu directement depuis les fichiers produits
par le pipeline (`01_build_universe.py` à `08_recuperation_options.py`). Le
rapport ne relance jamais de collecte lui-même : il ne fait que lire `./data/`.

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
  IsolationForest), et repères de valorisation (multiples sectoriels, DCF).
  En bas de page : clustering KMeans des multiples de valorisation sur
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
