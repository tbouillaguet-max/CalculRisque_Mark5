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
- [ ] Créé `data/backtest/optimization/convergence_*.csv`

### Étape 2.2 — Analyser la grille

**Lire la CSV:**
```bash
# Exemple d'affichage rapide
head -20 data/backtest/optimization/convergence_*.csv
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
  --stop-grid -30 -25 -20 -15 \
  --profit-grid 20 25 30 35
```

**À vérifier pendant:**
- [ ] Voir des logs de progession (chaque paire)
- [ ] Pas de crash

**À vérifier après:**
- [ ] Script termine
- [ ] Créé `data/backtest/optimization/stops_*.csv`

### Étape 3.2 — Analyser la heatmap

**Lire la CSV et construire une heatmap:**

```python
import pandas as pd
stops = pd.read_csv("data/backtest/optimization/stops_*.csv")
# Créer pivot: stop-loss (lignes) x take-profit (colonnes)
# Valeurs = CAGR ou Sharpe
```

**Heatmap attendue (exemple):**

```
Take-Profit →  20    25    30    35
Stop-Loss ↓
-30            8.2%  8.5%  8.3%  8.1%
-25            8.4%  8.7%  8.6%  8.4%   ← OPTIMUM (8.7%)
-20            8.1%  8.4%  8.5%  8.3%
-15            7.9%  8.2%  8.4%  8.2%
```

**Interprétation:**
- [ ] Trouver la cellule avec CAGR max
- [ ] Optimale = (-25%, 25%) dans cet exemple
- [ ] Vérifier que c'est un pic, pas un plateau

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
1. [ ] Élargir grille:
   ```bash
   python 11c_optimize_convergence_fraction.py \
     --fraction-grid 0.1 0.15 0.2 ... 1.0 1.1 1.2 1.3
   ```
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
- [ ] **Phase 4 OK:** Généralisation OOS validée

**Fichiers modifiés:**
- [ ] `backtest/strategies/valuation_gap_expected_value_options.py` (mis à jour)
- [ ] Tous les commits poussés vers branche

**Résultats à sauvegarder:**
- [ ] `data/backtest/optimization/convergence_*.csv`
- [ ] `data/backtest/optimization/stops_*.csv`
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
