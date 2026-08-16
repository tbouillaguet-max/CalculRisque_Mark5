# ✅ Checklist d'optimisation Kelly

Suivez cette checklist étape par étape pour optimiser la stratégie Kelly complètement.

**Durée estimée:** 4 semaines | **Données:** 2015-2024 | **Validation OOS:** 2024+

---

## 📋 Avant de commencer

- [ ] Lire `KELLY_STRATEGY_README.md` (overview)
- [ ] Lire `docs/optimization_guide_kelly_strategy.md` (détails techniques)
- [ ] Vérifier que tous les tests passent:
  ```bash
  python -m pytest tests/test_expected_value_strategy.py -v
  python -m pytest tests/test_compare_options_strategies.py -v
  ```
  - [ ] 17 tests Kelly passent ✓
  - [ ] 7 tests comparaison passent ✓

---

## 🔵 PHASE 1: BASELINE (Semaine 1, 10-15 min)

### Étape 1.1 — Backtest de référence
```bash
python 10_backtest_options.py \
  --strategy valuation_gap_expected_value_options \
  --start-date 2015-01-01 \
  --end-date 2024-01-01
```

**À vérifier après:** 
- [ ] Script termine sans erreur
- [ ] Créé un dossier en `data/backtest/options/<date>/`
- [ ] Le dossier contient `metrics.json` et `trades.parquet`

**Lire les résultats:**
- [ ] Ouvrir `data/backtest/options/<date>/metrics.json`
- [ ] Noter les valeurs:
  - [ ] `cagr_pct` = ? (rendement annualisé)
  - [ ] `sharpe_ratio` = ? (rendement/risque)
  - [ ] `max_drawdown_pct` = ?
  - [ ] `num_trades` = ? (nombre de positions)
  - [ ] `win_rate_pct` = ? (% rentables)

### Étape 1.2 — Comparaison avec ATM et Multiples
```bash
python compare_options_strategies.py \
  --start-date 2015-01-01 \
  --end-date 2024-01-01
```

**À vérifier après:**
- [ ] Script termine sans erreur
- [ ] Génère 2 CSV: `comparison_*.csv` et `comparison_*_by_exit_reason.csv`
- [ ] Les trois stratégies sont présentes

**Analyser le tableau comparatif:**
```
Ouvrir: data/backtest/comparisons/comparison_*.csv

Colonnes = stratégies:
- valuation_gap_options (ATM)
- valuation_gap_multiples_options (Multiples)
- valuation_gap_expected_value_options (Kelly ← Celle-ci)

Lignes clés à comparer:
```

| Métrique | ATM | Multiples | Kelly | Observations |
|----------|-----|-----------|-------|--------------|
| CAGR % | ? | ? | ? | [ ] Kelly gagne? |
| Sharpe | ? | ? | ? | [ ] Kelly gagne? |
| Max drawdown | ? | ? | ? | [ ] Kelly contrôlé? |
| Nombre de trades | ? | ? | ? | [ ] Volume similaire? |
| Win rate % | ? | ? | ? | [ ] Kelly plus juste? |

**Décomposition P&L par motif (CSV _by_exit_reason):**

- [ ] Ouvrir `comparison_*_by_exit_reason.csv`
- [ ] Filtrer par stratégie = `valuation_gap_expected_value_options`
- [ ] Analyser par motif:
  - [ ] `take_profit`: P&L positif? (thèse réalisée)
  - [ ] `stop_loss`: P&L négatif? (normal)
  - [ ] `expiry`: P&L ? (dérive de temps)

**Points de décision:**
- [ ] Kelly SURPERFORME (CAGR ou Sharpe > autres): → Continuer optimisation
- [ ] Kelly SOUS-PERFORME: → Diagnostiquer (voir section "Problèmes")
- [ ] Performance similaire: → Optimiser quand même (peut gagner finement)

---

## 🟢 PHASE 2: OPTIMISER CONVERGENCE_FRACTION (Semaine 2, 20-45 min)

**Objectif:** Trouver la fraction de convergence optimale (impact maximal sur strike)

### Étape 2.1 — Grid-search complet
```bash
python 11c_optimize_convergence_fraction.py \
  --start-date 2015-01-01 \
  --end-date 2024-01-01 \
  --fraction-grid 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --workers 4
```

**À vérifier pendant:**
- [ ] Voir des logs de progession (chaque fraction)
- [ ] Pas de crash

**À vérifier après:**
- [ ] Script termine
- [ ] Créé `data/backtest_options/optimize_convergence_<stratégie>_<horodatage>.csv`

### Étape 2.2 — Analyser la grille

**Lire la CSV:**
```bash
# Exemple d'affichage rapide
head -20 data/backtest_options/optimize_convergence_*.csv
```

**Tableau à créer:**

| Fraction | CAGR % | Sharpe | Num Trades | Notes |
|----------|--------|--------|------------|-------|
| 0.2 | ? | ? | ? | [ ] |
| 0.3 | ? | ? | ? | [ ] |
| 0.4 | ? | ? | ? | [ ] |
| 0.5 | ? | ? | ? | [ ] (défaut) |
| 0.6 | ? | ? | ? | [ ] |
| 0.7 | ? | ? | ? | [ ] |
| 0.8 | ? | ? | ? | [ ] |
| 0.9 | ? | ? | ? | [ ] |
| 1.0 | ? | ? | ? | [ ] |

**Interprétation:**
- [ ] CAGR doit être unimodal (un pic unique)
- [ ] Pic = fraction optimale
- [ ] Si pic au bord (0.2 ou 1.0): → Élargir grille (voir 2.3)

**Exemples:**
- Pic à 0.5 (défaut): → Bien calibré, passer à 2.3
- Pic à 0.7: → Fraction optimale = 0.7, passer à 2.3
- Pic au bord (0.2): → Grille trop étroite, faire 2.3

### Étape 2.3 — Affiner autour du pic (optionnel)

Si le pic n'est pas au bord de la grille, on peut s'arrêter. Sinon:

```bash
# Exemple: si pic observé à 0.8, affiner autour
python 11c_optimize_convergence_fraction.py \
  --start-date 2015-01-01 \
  --end-date 2024-01-01 \
  --fraction-grid 0.7 0.75 0.8 0.85 0.9 \
  --workers 4
```

- [ ] Créé nouvelle grille fine

### Étape 2.4 — Mettre à jour le défaut

Une fois la fraction optimale identifiée (ex: 0.65):

```bash
# Éditer: backtest/strategies/valuation_gap_expected_value_options.py
# Trouver: self.convergence_fraction = 0.5
# Remplacer par: self.convergence_fraction = 0.65
```

- [ ] Édité `valuation_gap_expected_value_options.py`
- [ ] Changé `convergence_fraction = <optimal_value>`

### Étape 2.5 — Re-run baseline avec nouvelle fraction

```bash
python 10_backtest_options.py \
  --strategy valuation_gap_expected_value_options \
  --start-date 2015-01-01 \
  --end-date 2024-01-01
```

**Comparer avec baseline phase 1:**

| Métrique | Baseline (0.5) | Optimisé | Gain |
|----------|----------------|----------|------|
| CAGR % | ? | ? | +? % |
| Sharpe | ? | ? | +? |

- [ ] CAGR amélioré? 
- [ ] Sharpe stable ou meilleur?

---

## 🟡 PHASE 3: OPTIMISER STOPS (Semaine 3, 30-60 min)

**Objectif:** Trouver la meilleure paire (stop-loss, take-profit)

### Étape 3.1 — Grid-search stop-loss/take-profit

```bash
python 11_optimize_options_stops.py \
  --strategy valuation_gap_expected_value_options \
  --start-date 2015-01-01 \
  --end-date 2024-01-01 \
  --stop-loss-grid -30 -25 -20 -15 \
  --take-profit-grid 0.6 0.8 1.0 1.2
```

⚠️ **Unité du take-profit.** Kelly porte `targets_convergence = True` : la grille attend
des **fractions de convergence** (0.4–1.2), pas des pourcentages. Passer `20 25 30 35`
ici testerait des fractions de 2000 % à 3500 %. Sur cette stratégie, `take_profit_pct`
est inerte — c'est `take_profit_convergence_fraction` qui agit.

**À vérifier pendant:**
- [ ] Voir des logs de progession (chaque paire)
- [ ] Pas de crash

**À vérifier après:**
- [ ] Script termine
- [ ] Créé `data/backtest_options/optimize_<stratégie>_<horodatage>.csv`

### Étape 3.2 — Analyser la heatmap

**Lire la CSV et construire une heatmap:**

```python
import glob
import pandas as pd

stops = pd.read_csv(sorted(glob.glob("data/backtest_options/optimize_*.csv"))[-1])
# take_profit porte la FRACTION de convergence pour Kelly (colonne "take_profit")
print(stops.pivot_table(index="stop_loss_pct", columns="take_profit",
                        values="train_sharpe_ratio"))
```

**Heatmap attendue (exemple, valeurs = Sharpe):**

```
Take-Profit →  0.6   0.8   1.0   1.2      (fractions de convergence)
Stop-Loss ↓
-30            0.82  0.85  0.83  0.81
-25            0.84  0.87  0.86  0.84   ← OPTIMUM (0.87)
-20            0.81  0.84  0.85  0.83
-15            0.79  0.82  0.84  0.82
```

**Interprétation:**
- [ ] Trouver la cellule avec `train_sharpe_ratio` max
- [ ] Optimale = (-25 %, fraction 0.8) dans cet exemple
- [ ] Vérifier que c'est un pic, pas un plateau
- [ ] Regarder `test_sharpe_ratio` sur cette cellule : si l'écart avec le train est fort,
      l'optimum ne survit pas aux données qui ne l'ont pas choisi

### Étape 3.3 — Mettre à jour les engine_defaults

Une fois les stops optimaux identifiés:

```bash
# Éditer: backtest/strategies/valuation_gap_expected_value_options.py
# Ou son engine_defaults dict

# Changer:
stop_loss_pct = -25  (old default -25%)
take_profit_pct = 30  (old default 30%)

# En:
stop_loss_pct = <optimal_stop>
take_profit_pct = <optimal_profit>
```

- [ ] Édité fichier stratégie
- [ ] Changé `stop_loss_pct = <optimal_value>`
- [ ] Changé `take_profit_pct = <optimal_value>`

### Étape 3.4 — Re-run comparaison globale

```bash
python compare_options_strategies.py \
  --start-date 2015-01-01 \
  --end-date 2024-01-01
```

**Comparer Kelly avec les 2 autres stratégies (post-optimisation):**

| Stratégie | CAGR % (Phase 1) | CAGR % (Phase 3) | Changement |
|-----------|-----------------|-----------------|-----------|
| Kelly | ? | ? | +? % |
| ATM | ? | ? | 0% (pas changé) |
| Multiples | ? | ? | 0% (pas changé) |

- [ ] Kelly s'est amélioré?
- [ ] Gap vs ATM/Multiples réduit?

---

## 🟠 PHASE 3bis: OPTIMISER LE SEUIL D'ENTRÉE (Semaine 3, 15-45 min)

**Objectif:** Trouver le seuil de valorisation qui fait entrer les bonnes lignes

### Étape 3bis.1 — Grid-search sur le seuil

```bash
python 11d_optimize_entry_threshold.py \
  --strategy valuation_gap_expected_value_options \
  --start-date 2015-01-01 \
  --end-date 2024-01-01 \
  --gap-grid 10 15 20 25 30 40
```

⚠️ **Unité.** `entry_threshold_pct` est en **points de log × 100**, pas en pourcentage.
`--gap-grid` accepte des % d'écart et fait la conversion ; `--entry-threshold-grid`
attend l'unité interne. Le défaut en production est 18.23 (= écart de 20 %).

**À vérifier après:**
- [ ] Script termine sans erreur
- [ ] Créé `data/backtest_options/optimize_entry_threshold_*.csv`

### Étape 3bis.2 — Analyser la grille

| Seuil (log) | Écart % | Sortie | Trades | Exposition % | CAGR % | train_sharpe | test_sharpe |
|-------------|---------|--------|--------|--------------|--------|--------------|-------------|
| 9.53 | 10 | ? | ? | ? | ? | ? | ? |
| 13.98 | 15 | ? | ? | ? | ? | ? | ? |
| 18.23 | 20 | ? | ? | ? | ? | ? | ? (défaut) |
| 22.31 | 25 | ? | ? | ? | ? | ? | ? |
| 26.24 | 30 | ? | ? | ? | ? | ? | ? |
| 33.65 | 40 | ? | ? | ? | ? | ? | ? |

**Ce qu'il faut lire — et pas seulement le Sharpe:**
- [ ] **`num_trades`** : un seuil haut finit par n'ouvrir que quelques positions. Le script
      écarte du classement celles sous `--min-trades` (15 par défaut), mais les garde au CSV
- [ ] **`avg_exposure_pct`** : un seuil haut peut « améliorer » le Sharpe en n'investissant
      plus. Du cash qui dort n'est pas une performance
- [ ] **`exit_threshold_pct`** : il suit l'entrée (× 0.70). La moitié du changement testé
      est là, pas dans le seuil d'entrée seul
- [ ] **`test_sharpe_ratio`** vs `train_sharpe_ratio` : c'est le second qui dit si l'optimum
      survit à des données qui ne l'ont pas choisi

### Étape 3bis.3 — Mettre à jour le défaut

- [ ] Éditer `config.OPTIONS_MULTIPLES_ENTRY_THRESHOLD_PCT` (en points de log)
- [ ] Re-run phase 1 pour mesurer l'impact total

---

## 🔴 PHASE 4: VALIDATION OUT-OF-SAMPLE (Semaine 4, 10-30 min)

**Objectif:** Tester les paramètres optimisés sur données non-vues (généralisation)

### Étape 4.1 — Backtest OOS (2024+)

```bash
python 10_backtest_options.py \
  --strategy valuation_gap_expected_value_options \
  --start-date 2024-01-01 \
  --end-date 2025-08-16
```

**À vérifier après:**
- [ ] Script termine sans erreur
- [ ] Créé `metrics.json` OOS

### Étape 4.2 — Comparer in-sample vs out-of-sample

**Charger les deux metrics.json:**
- [ ] `metrics_IS.json` = backtest 2015-2024 (optimisation)
- [ ] `metrics_OOS.json` = backtest 2024-2025 (validation)

**Tableau de comparaison:**

| Métrique | In-Sample (2015-2024) | Out-of-Sample (2024-2025) | Écart (%) |
|----------|----------------------|--------------------------|-----------|
| CAGR % | ? | ? | ? |
| Sharpe | ? | ? | ? |
| Max Drawdown | ? | ? | ? |
| Win Rate | ? | ? | ? |

**Interprétation:**
- [ ] Écart CAGR < 10%: → Bonne généralisation ✓
- [ ] Écart CAGR 10-20%: → Acceptable (attention overfitting)
- [ ] Écart CAGR > 20%: → Problème ⚠ (revoir optimisation)

### Étape 4.3 — Décision finale

**Si généralisation bonne (écarts < 10%):**
- [ ] Paramètres sont robustes
- [ ] Prêt pour production
- [ ] Procéder à 4.4

**Si généralisation faible (écarts > 20%):**
- [ ] Revoir l'optimisation (overfitting possible)
- [ ] Essayer validation croisée (walk-forward)
- [ ] ⚠ Ne pas utiliser en production sans révision

### Étape 4.4 — Production ready

```bash
# Vérifier que tous les commits sont poussés
git status
git log --oneline -5

# Tester le pipeline trimestriel avec Kelly
python run_pipeline_quarterly.py \
  --strategy valuation_gap_expected_value_options \
  --as-of-date 2025-08-16
```

- [ ] Tous les commits poussés vers `claude/options-expected-value-kelly-strategy`
- [ ] Pipeline trimestriel s'exécute sans erreur
- [ ] Prêt pour intégration au trading réel

---

## 🚨 Diagnostics — Problèmes courants

### ❌ Problème: Phase 1 — Kelly under-performe

**Symptôme:** CAGR Kelly < ATM ET CAGR Kelly < Multiples

**Causes possibles:**
- [ ] Convergence fraction mal calibrée → trop pessimiste (< 0.5)
- [ ] Strike selectionné trop ITM → stops coupent trop
- [ ] Données manquantes → univers réduit

**Actions:**
1. [ ] Vérifier `num_trades` Kelly vs autres (pas d'issue de données?)
2. [ ] Analyser P&L par motif: où perd-on?
3. [ ] Si stops coupent trop: élargir stops manuellement
4. [ ] Continuer quand même jusqu'à phase 2

### ❌ Problème: Phase 2 — CAGR pic au bord

**Symptôme:** Meilleure fraction = 0.2 (bord bas) ou 1.0 (bord haut)

**Cause:** Grille trop étroite (chercher le vrai optimum dehors)

**Actions:**
1. [ ] Élargir la grille — mais **seulement vers le bas** :
   ```bash
   python 11c_optimize_convergence_fraction.py \
     --fraction-grid 0.05 0.1 0.15 0.2 0.3 0.4 0.5
   ```
   ⚠️ La stratégie **rejette** toute fraction hors de `]0, 1]` (`ValueError` au démarrage).
   Un pic collé à 1.0 ne peut donc pas être « débordé » : il signifie que la thèse
   supporterait une convergence complète, et c'est le résultat lui-même.
2. [ ] Re-analyser (2.2)

### ❌ Problème: Phase 3 — Pas de pic clair (plateau)

**Symptôme:** CAGR similaire pour toutes les paires stop/profit

**Cause:** Stops ne sont pas un levier important (thèse domine)

**Actions:**
1. [ ] Accepter la plateau (stop-loss est moins critique)
2. [ ] Choisir stops conservateurs (-25%, 30%)
3. [ ] Continuer quand même

### ❌ Problème: Phase 4 — Écart IS/OOS > 20%

**Symptôme:** OOS CAGR << IS CAGR

**Cause probable:** Overfitting (optimisation sur bruit de 2015-2024)

**Actions:**
1. [ ] Ne PAS utiliser ces paramètres en production
2. [ ] Revoir l'optimisation:
   - [ ] Utiliser validation croisée (walk-forward)
   - [ ] Élargir les grilles
   - [ ] Augmenter min_trades pour filtrer bruits
3. [ ] Consulter `docs/optimization_guide_kelly_strategy.md` section "Diagnostics"

---

## 📝 Résumé final

Après toutes les phases:

- [ ] **Phase 1 OK:** Baseline établie, Kelly évalué
- [ ] **Phase 2 OK:** `convergence_fraction` optimisée
- [ ] **Phase 3 OK:** `(stop_loss, take_profit)` optimisés
- [ ] **Phase 3bis OK:** `entry_threshold_pct` optimisé
- [ ] **Phase 4 OK:** Généralisation OOS validée

**Fichiers modifiés:**
- [ ] `backtest/strategies/valuation_gap_expected_value_options.py` (convergence_fraction, stops)
- [ ] `config.py` (`OPTIONS_MULTIPLES_ENTRY_THRESHOLD_PCT` si phase 3bis a bougé le seuil)
- [ ] Tous les commits poussés vers branche

**Résultats à sauvegarder** (tous sous `data/backtest_options/`):
- [ ] `optimize_convergence_<stratégie>_<horodatage>.csv`
- [ ] `optimize_<stratégie>_<horodatage>.csv` (stops)
- [ ] `optimize_entry_threshold_<stratégie>_<horodatage>.csv`
- [ ] `data/backtest/comparisons/comparison_*.csv`
- [ ] Screenshots/tableaux comparatifs IS vs OOS

**Prochaines étapes:**
1. [ ] Intégrer Kelly au pipeline trimestriel (`run_pipeline_quarterly.py`)
2. [ ] Monitorer performance réelle vs prédite
3. [ ] Revisiter optimisation annuellement (driften de marché)

---

**Durée totale:** 4 semaines | **Effort:** ~10-15 heures de backtest (surtout automatisé)

**Besoin d'aide?** Consulter:
- `KELLY_STRATEGY_README.md` (vue d'ensemble)
- `docs/optimization_guide_kelly_strategy.md` (technique)
- Tests dans `tests/test_expected_value*.py` (fonctionnement)
