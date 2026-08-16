# Stratégie Kelly de valorisation d'options

## Vue générale

La stratégie **ValuationGapExpectedValueOptionsStrategy** (Kelly) choisit le strike qui **maximise la croissance log-optimale (Kelly criterion)** pour chaque position, plutôt que de le poser par convention comme les deux autres stratégies.

- **valuation_gap_options**: Strike ATM (à la monnaie)
- **valuation_gap_multiples_options**: Strike à mi-chemin cours/valeur théorique
- **valuation_gap_expected_value_options**: Strike qui maximise Kelly ← **Cette branche**

## Statut du code

✅ **Complètement implémentée et testée**
- 17 tests passent, 1 omis
- Scripts d'optimisation opérationnels
- Documentation exhaustive

## Démarrage rapide

### 1️⃣ Exécuter un backtest de référence
```bash
python 10_backtest_options.py \
  --strategy valuation_gap_expected_value_options \
  --start-date 2015-01-01 \
  --end-date 2024-01-01
```

### 2️⃣ Comparer les trois stratégies
```bash
python compare_options_strategies.py \
  --start-date 2015-01-01 \
  --end-date 2024-01-01
```

### 3️⃣ Lancer le workflow d'optimisation complet (4 phases)
```bash
python demo_kelly_optimization_workflow.py --phase all
```

## Architecture de la stratégie

### Équation fondamentale
```
mu = convergence_fraction × ln(V / S₀) / T
Strike optimal K* = argmax_K E[log(1 + f × R(K))]
```

Où:
- `convergence_fraction` ∈ (0, 1] contrôle la dérive (0.5 = défaut)
- `mu` = rendement annualisé attendu sous la thèse de valorisation
- `R(K)` = rendement du contrat (payoff/prime - 1)
- `f` = allocation de Kelly (% du capital sur ce contrat)

### Motif de Kelly vs autres critères

Kelly est choisi car:
- **Pas de dégénérescence aux bords**: Contrairement aux ratios (Sharpe, Sortino), Kelly refuse les contrats qui ne paient presque jamais
- **Continuité**: Le strike optimal se déplace continuement avec la conviction, pas par sauts
- **Optimalité en croissance**: Maximise le taux de croissance log-optimal long terme

**Référence:** Mesures empiriques dans `backtest/strategies/valuation_gap_expected_value_options.py` (docstring, section "POURQUOI KELLY, ET PAS UN RATIO GAIN/RISQUE")

## Variables à optimiser

Classées par importance (détails: `docs/optimization_guide_kelly_strategy.md`):

### TIER 1 — CRITIQUE
| Variable | Défaut | Optimisation |
|----------|--------|--------------|
| `convergence_fraction` | 0.5 | `11c_optimize_convergence_fraction.py` |
| `entry_threshold_pct` | 18.23 (= écart 20 %) | `11d_optimize_entry_threshold.py` |

⚠️ `entry_threshold_pct` s'exprime en **points de log × 100**, pas en pourcentage :
`log(1.20) × 100 = 18.23` correspond à un écart de 20 %.

### TIER 2 — ÉLEVÉE
| Variable | Défaut | Optimisation |
|----------|--------|--------------|
| `strike_grid_n_sigma` | 3.0 | Tuning manuel (si optimum au bord) |
| `strike_grid_step_sigma` | 0.25 | Fixe (bien calibré) |
| `quadrature_nodes` | 128 | Fixe (suffisant en précision) |

### TIER 3 — MOYENNE
| Variable | Défaut | Optimisation |
|----------|--------|--------------|
| `weight_cap_pct` | 100% | Tuning par secteur |
| `exit_threshold_ratio` | 0.70 | Se fixe via `11d --exit-threshold-ratio` |
| `rebalance_log_gap_threshold` (ε) | 0.15 | `11b_optimize_rebalance_threshold.py` |
| `daily_rebalance` | True | Choix stratégique (architectural) |

### TIER 4 — FAIBLE (Architecturaux — Ne pas changer)
| Variable | Défaut | Justification |
|----------|--------|---------------|
| `target_tenor_days` | 730 (2 ans) | Thèse convergence multiples |
| `roll_when_days_left` | 270 | Maintient tenor ≈ 2 ans |
| `stop_loss_pct` | -25% | Calibré pour OTM |
| `take_profit_pct` | 30% | % convergence théorique |
| `vol_mode` | "rolling" | Repricing quotidien |

## Scripts d'optimisation

Tous écrivent leur CSV sous `data/backtest_options/`.

### 11c_optimize_convergence_fraction.py
**Optimise:** `convergence_fraction` ∈ `]0, 1]` — valeurs hors bornes **rejetées**
**Temps:** 20-45 min | **Classement:** in-sample (pas de walk-forward)
```bash
python 11c_optimize_convergence_fraction.py \
  --start-date 2015-01-01 \
  --end-date 2024-01-01 \
  --fraction-grid 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0
```
**Sortie:** `optimize_convergence_<stratégie>_<horodatage>.csv`

### 11_optimize_options_stops.py
**Optimise:** `stop_loss_pct` × take-profit (surface 2D)
**Temps:** 30-60 min | **Classement:** walk-forward
```bash
python 11_optimize_options_stops.py \
  --strategy valuation_gap_expected_value_options \
  --start-date 2015-01-01 \
  --end-date 2024-01-01 \
  --stop-loss-grid -30 -25 -20 -15 \
  --take-profit-grid 0.6 0.8 1.0 1.2
```
⚠️ Kelly porte `targets_convergence = True` : le take-profit se balaie en **fractions de
convergence** (défaut `[0.4 … 1.2]`), pas en pourcentages. `take_profit_pct` est inerte
sur cette stratégie.

**Sortie:** `optimize_<stratégie>_<horodatage>.csv`

### 11d_optimize_entry_threshold.py
**Optimise:** `entry_threshold_pct` (seuil d'entrée)
**Temps:** 15-45 min | **Classement:** walk-forward
```bash
python 11d_optimize_entry_threshold.py \
  --strategy valuation_gap_expected_value_options \
  --start-date 2015-01-01 \
  --end-date 2024-01-01 \
  --gap-grid 10 15 20 25 30 40      # en % d'écart, converti en points de log
```
Le seuil de sortie suit l'entrée (`sortie = entrée × exit_threshold_ratio`) : le CSV
porte les deux, plus `ecart_equivalent_pct` pour la conversion d'unité.

**Sortie:** `optimize_entry_threshold_<stratégie>_<horodatage>.csv`

### 11b_optimize_rebalance_threshold.py
**Optimise:** `rebalance_log_gap_threshold` (ε) — churn de rééquilibrage, **pas** le seuil d'entrée
**Temps:** 15-45 min | **Classement:** walk-forward
```bash
python 11b_optimize_rebalance_threshold.py \
  --strategy valuation_gap_expected_value_options \
  --start-date 2015-01-01 \
  --epsilon-grid 0 0.05 0.10 0.15 0.20 0.30
```
**Sortie:** `optimize_rebalance_<stratégie>_<horodatage>.csv`

## Workflow d'optimisation recommandé (4 semaines)

### Semaine 1: Baseline
1. Backtest Kelly sur 2015-2024
2. Comparer Kelly vs ATM vs Multiples
3. Analyser P&L par motif de sortie
```bash
python demo_kelly_optimization_workflow.py --phase 1
```

### Semaine 2: Convergence
1. Grid-search `convergence_fraction`
2. Identifier pic optimal
3. Affiner autour du pic
```bash
python demo_kelly_optimization_workflow.py --phase 2
```

### Semaine 3: Stops
1. Grid-search `(stop_loss, take_profit)`
2. Analyser heatmap
3. Mettre à jour défauts
```bash
python demo_kelly_optimization_workflow.py --phase 3
```

### Semaine 4: Validation
1. Backtest OOS (2024+)
2. Vérifier généralisation
3. Production ready
```bash
python demo_kelly_optimization_workflow.py --phase 4
```

**Voir aussi:** `docs/optimization_guide_kelly_strategy.md` pour détails complets

## Diagnostics et troubleshooting

### ❌ Problème: Trop peu de trades
**Cause:** `entry_threshold_pct` trop haut ou `convergence_fraction` mal calibrée
**Solution:** Baisser le seuil ou affiner la fraction via 11c

### ❌ Problème: Optimum au bord de la grille
**Cause:** Grille trop étroite
**Solution:** Augmenter `strike_grid_n_sigma` (3.0 → 4.0)

### ❌ Problème: Écart in-sample / out-of-sample > 20%
**Cause:** Overfitting
**Solution:** Réviser la grille ou utiliser une validation croisée

## Fichiers clés

```
backtest/strategies/
  └─ valuation_gap_expected_value_options.py    # Stratégie Kelly
  └─ valuation_gap_multiples_options.py          # Référence (multiples)
  └─ valuation_gap_options.py                    # Référence (ATM)

backtest/
  └─ expected_value.py                           # Maths Kelly (pur)
  └─ options_engine.py                           # Moteur d'exécution

tests/
  └─ test_expected_value_strategy.py             # Tests Kelly (17/18 ✓)
  └─ test_expected_value.py                      # Tests maths (isolé)
  └─ test_compare_options_strategies.py          # Tests comparaison

scripts d'optimisation:
  11c_optimize_convergence_fraction.py           # Grid-search fraction de convergence
  11_optimize_options_stops.py                   # Grid-search stops (surface 2D)
  11d_optimize_entry_threshold.py                # Grid-search seuil d'entrée
  11b_optimize_rebalance_threshold.py            # Grid-search ε (churn de rééquilibrage)

scripts de comparaison:
  compare_options_strategies.py                  # Côte à côte 3 stratégies
  10_backtest_options.py                         # Backtest unitaire
  demo_kelly_optimization_workflow.py            # Pipeline 4 phases

documentation:
  docs/optimization_guide_kelly_strategy.md      # Guide complet (CRUCIAL)
  KELLY_STRATEGY_README.md                       # Ce fichier
```

## Tests

Exécuter tous les tests:
```bash
python -m pytest tests/test_expected_value_strategy.py -v
python -m pytest tests/test_expected_value.py -v
python -m pytest tests/test_compare_options_strategies.py -v
```

Résumé:
- ✅ 17 tests passent sur la stratégie Kelly
- ✅ 7 tests passent sur la comparaison
- ⏭️ 1 test omis (strike très ITM, cas limite)

## Impact empirique

Mesures de Kelly vs autres critères (sur données d'optimisation):

### Kelly vs Sharpe
- Kelly refuse les strikes OTM improbables (Sharpe les cherche)
- Kelly produit une allocation **continue** (Sharpe = oscillations binaires)
- Kelly maximise croissance log (Sharpe = rendement/vol)

### Kelly vs Sortino
- Sortino diverge hors-monnaie (prime limite risque baissier)
- Kelly borne par probabilité de perte totale
- Résultat: Kelly reste centré, Sortino → extrême OTM

**Voir:** `backtest/strategies/valuation_gap_expected_value_options.py` (docstring) pour mesures quantitatives

## Intégration à la production

Une fois optimisée:
```bash
# Lancer le pipeline trimestriel avec Kelly
python run_pipeline_quarterly.py \
  --strategy valuation_gap_expected_value_options \
  --as-of-date 2025-08-16
```

## Liens

- **Guide complet:** `docs/optimization_guide_kelly_strategy.md`
- **Workflow guidé:** `python demo_kelly_optimization_workflow.py --phase all`
- **Maths Kelly:** `backtest/expected_value.py` (avec dérivations)
- **Tests:** `tests/test_expected_value*.py`

## Historique de branche

Commits clés:
- `821e83a` — Stratégie « espérance de gain » : strike choisi par Kelly
- `cdee1ec` — Moments tronqués et risque baissier du payoff
- `a10fbea` — Espérance, variance et Sharpe du payoff
- `d99d78a` — Guide complet d'optimisation (TIER 1-4)
- `a1cbe48` — Workflow guidé 4 phases

## Contact

Pour questions ou améliorations, voir la documentation dans `docs/` ou les docstrings des scripts.

---

**Dernière mise à jour:** 2025-08-16
**État:** Production-ready après optimisation in-sample + validation OOS
