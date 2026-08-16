# Guide complet d'optimisation de ValuationGapExpectedValueOptionsStrategy

## Vue d'ensemble

La stratégie « espérance de gain » (Kelly) choisit le strike qui maximise la croissance log-optimale (Kelly criterion) plutôt que de le positionner par convention (ATM pour `valuation_gap_options`, mi-chemin pour `valuation_gap_multiples_options`).

### Équation fondamentale
```
mu = convergence_fraction × ln(V / S₀) / T
Strike optimal = argmax_K E[log(1 + f × R(K))]
R(K) = (payoff(K) / prime(K)) - 1
```

---

## Classification des variables par importance

### TIER 1: CRITIQUE — Impact direct sur la sélection du strike

#### 1. `convergence_fraction` (défaut: 0.5)
- **Rôle:** Contrôle la dérive annualisée μ alimentée au critère de Kelly
- **Impact:** Détermine directement quel strike Kelly sélectionne
- **Plage:** (0, 1] où 1.0 suppose une convergence complète vers la valeur théorique
- **Mesure empirique:** 
  - À σ = 20%, T = 2 ans:
    - μ = 6% (fraction 0.5) → K/S₀ ≈ 0.43
    - μ = 20% (fraction 1.0) → K/S₀ ≈ 0.99
    - μ = 35% (fraction 1.75) → K/S₀ ≈ 1.42

**Optimisation:**
- **Outil:** `11c_optimize_convergence_fraction.py` (Kelly est sa stratégie par défaut)
- **Méthode:** Grid-search (défaut: `[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]`)
- **Approche:** In-sample uniquement — pas de walk-forward dans ce script, contrairement à 11, 11b et 11d
- **Contrainte:** la stratégie **rejette** toute fraction hors de `]0, 1]` (`ValueError`). Une grille contenant 1.1 ou 1.2 échoue avant de tourner.
- **Commandes typiques:**
  ```bash
  # Grid-search complet sur la période 2015-2024
  python 11c_optimize_convergence_fraction.py --start-date 2015-01-01 --end-date 2024-01-01
  
  # Affiner autour d'une valeur optimale connue (ex: 0.5-0.7)
  python 11c_optimize_convergence_fraction.py --fraction-grid 0.5 0.55 0.6 0.65 0.7 \
    --start-date 2015-01-01
  ```

---

#### 2. `entry_threshold_pct` (hérité de MultiplesStrategy, défaut: 18.23)
- **Rôle:** Filtre de valorisation — écart minimal qui fait entrer une ligne dans l'univers des candidates
- **⚠️ Unité:** **points de log × 100**, PAS un pourcentage d'écart. Le défaut `config.OPTIONS_MULTIPLES_ENTRY_THRESHOLD_PCT` vaut `log(1.20) × 100 = 18.23`, soit un écart de **20 %**. Lire « 30 » comme « 30 % » (c'est 35 %) fausse toute la grille.
- **Impact:** Affecte le nombre de candidates, donc la concentration du portefeuille ET l'exposition moyenne (un seuil haut laisse dormir du cash)
- **Couplage:** `exit_threshold_pct = entry_threshold_pct × exit_threshold_ratio` — balayer l'entrée déplace **aussi** la sortie, à hystérésis constante
- **Calibration:** Trop bas = bruit du modèle de multiples ; trop haut = titres en difficulté réelle, où l'écart s'explique autrement qu'par une erreur de marché

**Optimisation:**
- **Outil:** `11d_optimize_entry_threshold.py`
- **Méthode:** Grid-search avec walk-forward (`--train-fraction 0.60` par défaut)
- **Commandes typiques:**
  ```bash
  # Grille par défaut (écarts de 10 % à 50 %)
  python 11d_optimize_entry_threshold.py --start-date 2015-01-01 --end-date 2024-01-01

  # Saisie directe en % d'écart, sans conversion mentale
  python 11d_optimize_entry_threshold.py --gap-grid 10 15 20 25 30 40

  # Ou en points de log, si on préfère l'unité interne
  python 11d_optimize_entry_threshold.py --entry-threshold-grid 9.53 13.98 18.23 22.31
  ```

---

### TIER 2: ÉLEVÉE — Affectent la qualité et la précision de l'optimisation

#### 3. `strike_grid_n_sigma` (défaut: 3.0)
- **Rôle:** Largeur de la grille de recherche de strikes
- **Formule:** [S₀·exp(-3σ√T), S₀·exp(+3σ√T)]
- **Impact:** Détermine si le vrai optimum de Kelly tombe dans la grille
- **Mesure empirique:** ±3σ capture ~99.7% de la distribution lognormale

**Optimisation:**
- **Méthode:** Exogène, basée sur des observations de backtests
- **Cas d'utilisation:**
  - Élargir à ±4σ ou ±5σ si l'optimum observé se colle systématiquement à un bord
  - Laisser à 3.0 en conditions normales (bien calibré depuis les tests)

---

#### 4. `strike_grid_step_sigma` (défaut: 0.25)
- **Rôle:** Granularité de discrétisation de la grille
- **Impact:** Finer = précision meilleure mais plus lent; Coarser = rapide mais risque de rater l'optimum
- **Mesure empirique:** 0.25σ offre un bon équilibre (testé via scans de grille dans les conversations antérieures)

**Optimisation:**
- **Méthode:** Rarement modifié (0.25 est bien calibré)
- **Cas d'utilisation:** Ajustement de performance uniquement si tuning d'exécution demandé

---

#### 5. `quadrature_nodes` (défaut: 128, de OPTIONS_EV_QUADRATURE_NODES)
- **Rôle:** Nombre de nœuds de Gauss-Legendre pour l'évaluation du critère de Kelly
- **Mesure empirique:** 128 nœuds → précision ~1e-13 (32 nœuds suffisent mais 128 est sûr)
- **Impact:** Plus de nœuds = plus de précision, plus lent

**Optimisation:**
- **Méthode:** Fixe, rare ajustement (uniquement sur hardware limité)

---

### TIER 3: MOYENNE — Affectent le cycle de vie des positions et la concentration

#### 6. `weight_cap_pct` (hérité de MultiplesStrategy, défaut: 100%)
- **Rôle:** Plafond de poids individuel d'une position (prévient les allocations outsourcées)
- **Mesure empirique:** Une position seule peut consommer 92% du portefeuille sans cap
- **Impact:** Contrôle la concentration extrême mais rare

**Optimisation:**
- **Méthode:** Tuning par secteur ou global selon cibles de concentration
- **Cas d'utilisation:** Rarement modifié, typiquement défaut de 100% suffit

---

#### 7. `exit_threshold_ratio` (hérité de MultiplesStrategy, défaut: 1.0)
- **Rôle:** Hystérésis — exit_threshold = entry_threshold × ratio
- **Impact:** Prévient l'oscillation autour du seuil d'entrée due au bruit quotidien
- **Formule:** ratio < 1 → sortie plus tôt qu'entrée

**Optimisation:**
- **Méthode:** Rarement modifié (1.0 est standard, peut être affiné via 11b)
- **Approche:** Grid-search exogène si volatilité du signal très élevée

---

#### 8. `daily_rebalance` (engine_default: True)
- **Rôle:** Architecture — Rééquilibre quotidien vs. seulement aux dépôts SEC
- **Impact:** True = capture les opportunités entre les dépôts, False = moins de churn
- **Défaut du référentiel:** True pour la stratégie multiples (Kelly hérite)

**Optimisation:**
- **Méthode:** Choix stratégique (rarement modifié)
- **Considération:** Affecte le profil de friction coûts d'exécution vs. capture

---

### TIER 4: FAIBLE — Choix architecturaux (rarement optimisés)

#### 9. `stop_loss_pct` (engine_default: -25%)
- **Rôle:** Trigger de sortie mesuré sur le sous-jacent, non la prime
- **Calibration:** -25% est plus serré que `valuation_gap_options` (-20%) pour compenser les strikes OTM
- **Mesure empirique:** Optimisé empiriquement pour la thèse Kelly

**Optimisation:**
- **Outil:** `11_optimize_options_stops.py` (grid-search sur paires stop-loss/take-profit)
- **Approche:** Joint optimization avec `take_profit_pct`

---

#### 10. `take_profit_pct` (engine_default: 30%)
- **Rôle:** Trigger de prise de profit
- **Interprétation:** 30% du chemin de convergence vers la valeur théorique (targets_convergence=True)
- **Impact:** Non un montant fixe en dollars, mais une fraction du mouvement

**Optimisation:**
- **Outil:** `11_optimize_options_stops.py`
- **Approche:** Co-optimization avec `stop_loss_pct`

---

#### 11. `roll_when_days_left` (engine_default: 270 jours = 9 mois)
- **Rôle:** Trigger de roulement sur contrats 2 ans
- **Impact:** Conserve la tenor proche de 2 ans sur la durée de vie du trade
- **Calibration:** Fixe par la thèse (rarement changé)

**Optimisation:**
- **Méthode:** Pas d'optimisation (choix architectural)

---

#### 12. `vol_mode` (engine_default: "rolling")
- **Rôle:** Stratégie de repricing volatilité
- **"rolling":** Repricing quotidien sur vol réalisée
- **"frozen":** Utilise la vol à l'entrée
- **Impact:** Critique pour contrats 2 ans où volatilité change

**Optimisation:**
- **Méthode:** Pas d'optimisation (choix architectural fixed par la thèse)

---

#### 13. `target_tenor_days` (engine_default: 730 jours = 2 ans)
- **Rôle:** Horizon d'échéance fixe pour tous les contrats
- **Justification:** Partie intégrante de la thèse convergence multiples
- **Impact:** Architectural — change la thèse si modifié

**Optimisation:**
- **Méthode:** Pas d'optimisation sans réévaluation complète de la stratégie

---

## Chemins d'optimisation organisés

### 1. OPTIMISATION PRIMAIRE — Scripts grid-search explicites

#### Chemin A: `convergence_fraction` (CRITIQUE)
```bash
# Étape 1: Scan complet
python 11c_optimize_convergence_fraction.py \
  --start-date 2015-01-01 \
  --end-date 2024-01-01 \
  --fraction-grid 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

# Étape 2: Affiner autour du pic
python 11c_optimize_convergence_fraction.py \
  --fraction-grid 0.45 0.50 0.55 0.60 0.65 \
  --start-date 2015-01-01 \
  --end-date 2024-01-01

# Lecture: data/backtest_options/optimize_convergence_<stratégie>_<horodatage>.csv
# Colonne "cagr_pct" indique le CAGR pour chaque fraction
```

**Interprétation des résultats:**
- Un pic unique = convergence fraction bien calibrée
- Plateau = insensibilité à la fraction (améliorer le signal d'entrée)
- Pic de bord = grille trop étroite (voir strike_grid_n_sigma)

---

#### Chemin B: `(stop_loss, take_profit)` (HAUTE)
```bash
# Grid-search sur les paires stop-loss/take-profit
python 11_optimize_options_stops.py \
  --strategy valuation_gap_expected_value_options \
  --start-date 2015-01-01 \
  --end-date 2024-01-01 \
  --stop-loss-grid -30 -25 -20 -15 \
  --take-profit-grid 0.6 0.8 1.0 1.2

# Lecture: data/backtest_options/optimize_<stratégie>_<horodatage>.csv
# Heatmap (stop_loss vs take_profit) avec Sharpe ou CAGR
```

**⚠️ Unité du take-profit pour Kelly.** La stratégie porte `targets_convergence = True` :
le mode `auto` bascule donc sur **`--take-profit-mode convergence`**, et la grille attend
des **fractions de convergence** (défaut `[0.4 … 1.2]`), pas des pourcentages. Sur une
stratégie de convergence, `take_profit_pct` est **inerte** — le balayer ne mesurerait rien.

**Considérations:**
- Stops plus serrés (-30%) = moins de perte/trade mais plus de whipsaws
- Fraction de convergence plus haute (1.0+) = viser la valeur théorique entière, donc attendre plus longtemps et subir plus de theta

---

#### Chemin C: `entry_threshold_pct` — seuil d'entrée (HAUTE)
```bash
# Grid-search sur le seuil d'entrée, saisi en % d'écart
python 11d_optimize_entry_threshold.py \
  --strategy valuation_gap_expected_value_options \
  --start-date 2015-01-01 \
  --end-date 2024-01-01 \
  --gap-grid 10 15 20 25 30 40

# Lecture: data/backtest_options/optimize_entry_threshold_<stratégie>_<horodatage>.csv
```

Le tableau porte `ecart_equivalent_pct` et `exit_threshold_pct` à côté du seuil, pour que
la conversion d'unité et le couplage entrée/sortie restent visibles.

---

#### Chemin D: `rebalance_log_gap_threshold` (ε) — churn de rééquilibrage (MOYENNE)
```bash
# Grid-search sur ε — un paramètre du MOTEUR, pas de la stratégie
python 11b_optimize_rebalance_threshold.py \
  --strategy valuation_gap_expected_value_options \
  --start-date 2015-01-01 \
  --epsilon-grid 0 0.05 0.10 0.15 0.20 0.30

# Lecture: data/backtest_options/optimize_rebalance_<stratégie>_<horodatage>.csv
```

**À ne pas confondre avec le chemin C.** ε ne filtre pas les entrées : il contrôle le
redimensionnement d'une position **déjà détenue** sur dépôt SEC (une position n'est
retouchée que si `|log(V/P) − last_rebalance_log_gap|` dépasse ε). C'est un levier de
friction et de churn, pas de sélection. La colonne à lire est `num_rebalance_trades`.

---

### 2. OPTIMISATION SECONDAIRE — Tuning manuel

Ces variables n'ont **pas** de script de grid-search dédié : elles s'ajustent à la main,
en relançant `10_backtest_options.py` et en comparant.

- **`strike_grid_n_sigma`:** Observer l'optimum sur plusieurs backtests; élargir si systématiquement au bord
- **`weight_cap_pct`:** Par secteur, selon concentration maximale tolérable
- **`exit_threshold_ratio`:** Si signal oscillant, tester 0.8–1.2. Se fixe aussi depuis `11d_optimize_entry_threshold.py --exit-threshold-ratio`, où il reste constant sur toute la grille

**Exemple: Tuning de `strike_grid_n_sigma`**
```python
# Dans backtest/strategies/valuation_gap_expected_value_options.py
# Vérifier que le strike optimal Kelly n'est pas au bord de la grille

# Si observé au bord (K/S0 tout au bout du range), augmenter:
strike_grid_n_sigma = 4.0  # au lieu de 3.0

# Re-run backtest et vérifier la stabilité du strike choisi
python 10_backtest_options.py \
  --strategy valuation_gap_expected_value_options \
  --start-date 2015-01-01 \
  --end-date 2024-01-01
```

---

### 3. VARIABLES FIXES — Ne pas optimiser

Les variables suivantes sont des choix architecturaux et font partie de la **thèse** de la stratégie. Les modifier change fondamentalement le comportement:

| Variable | Défaut | Raison |
|----------|--------|--------|
| `target_tenor_days` | 730 (2 ans) | Convergence multiples = 2 ans |
| `roll_when_days_left` | 270 jours | Maintient tenor ≈ 2 ans |
| `vol_mode` | "rolling" | Critique pour 2 ans (vol change) |
| `stop_basis` | "underlying" | Cohérence avec la thèse |
| `quadrature_nodes` | 128 | Suffisant en précision |
| `strike_grid_step_sigma` | 0.25 | Bien calibré |
| `exit_when_signal_lost` | True | Désynchronisation univervers |

---

## Flux d'optimisation recommandé (workflow)

### Phase 1: Établir la baseline (Semaine 1)
1. **Exécuter un backtest de référence** avec tous les défauts:
   ```bash
   python 10_backtest_options.py \
     --strategy valuation_gap_expected_value_options \
     --start-date 2015-01-01 \
     --end-date 2024-01-01 \
     > baseline_run.log 2>&1
   ```
   - Sauvegarder metrics.json (baseline)
   - Noter CAGR, Sharpe, max drawdown, nombre de trades

2. **Comparer avec les deux autres stratégies:**
   ```bash
   python compare_options_strategies.py \
     --start-date 2015-01-01 \
     --end-date 2024-01-01
   ```
   - Analyser la décomposition par motif de sortie
   - Identifier si Kelly performe mieux/pire et **pourquoi**

---

### Phase 2: Optimiser `convergence_fraction` (Semaine 2)
1. **Grid-search complet** (parallélisable):
   ```bash
   python 11c_optimize_convergence_fraction.py \
     --start-date 2015-01-01 \
     --end-date 2024-01-01 \
     --workers 4  # 4 grid-search parallèles
   ```

2. **Analyser les résultats:**
   - Plotter CAGR vs fraction (devrait être unimodal)
   - Sélectionner la fraction au pic
   - Affiner autour du pic (±0.1)

3. **Committter la meilleure fraction:**
   ```bash
   # Mettre à jour le défaut dans valuation_gap_expected_value_options.py
   convergence_fraction=<optimal_value>
   ```

---

### Phase 3: Optimiser stops/profils (Semaine 3)
1. **Grid-search stop-loss/take-profit:**
   ```bash
   python 11_optimize_options_stops.py \
     --strategy valuation_gap_expected_value_options \
     --start-date 2015-01-01 \
     --end-date 2024-01-01
   ```

2. **Analyser la heatmap** résultante
3. **Re-comparer** les trois stratégies avec les nouveaux stops

---

### Phase 4: Validation (Semaine 4)
1. **Out-of-sample validation** (données ultérieures à l'optimisation):
   ```bash
   python 10_backtest_options.py \
     --strategy valuation_gap_expected_value_options \
     --start-date 2024-01-01 \
     --end-date 2025-01-01  # Données non-vues lors de l'optimisation
   ```
   - Vérifier que Sharpe/CAGR reste proche de l'optimisation in-sample

2. **Rouler le pipeline trimestriel** pour conditions réelles:
   ```bash
   python run_pipeline_quarterly.py --as-of-date 2025-08-01
   ```

---

## Diagnostic des problèmes courants

### Problème: Optimum de Kelly au bord de la grille
**Diagnostic:**
```python
# Dans les logs: "Kelly-optimal strike K*: X.XX"
# Si K* ≈ S0 x exp(±3σ√T), c'est au bord
```

**Solution:**
- Augmenter `strike_grid_n_sigma` de 3.0 → 4.0 ou 5.0
- Re-run backtest et vérifier stabilité

---

### Problème: Trop peu de trades ouvertes
**Diagnostic:** `num_trades < 10 sur 10 ans`

**Causes probables:**
- `entry_threshold_pct` trop élevé (seuil d'écart trop strict)
- `convergence_fraction` trop proche de 1.0 (dérive trop optimiste)

**Solution:**
- Baisser `entry_threshold_pct` ou augmenter `convergence_fraction`
- Grid-search via 11b ou 11c

---

### Problème: Strikes choisis par Kelly très hors-la-monnaie
**Diagnostic:** Tous les strikes K/S0 < 0.8 ou K/S0 > 1.3

**Causes:**
- Dérive μ mal calibrée (convergence_fraction mal choisie)
- Volatilité impliquée vs réalisée mal alignées

**Solution:**
- Vérifier la cohérence : vol_implied vs vol_realized
- Affiner convergence_fraction via 11c

---

## Mesures de performance à suivre

### Métriques clés:
1. **CAGR %** — rendement annualisé (critère principal)
2. **Sharpe** — rendement ajusté au risque
3. **Sortino** — rendement/risque baissier
4. **Max drawdown %** — perte maximale à partir d'un pic
5. **Win rate %** — % de trades rentables
6. **Profit factor** — (gains bruts) / (pertes brutes)
7. **Nombre de trades** — activité portefeuille

### Métriques de diagnostic:
- **P&L par motif de sortie** — décompose où vient le gain (take-profit vs expiration vs stop-loss)
- **Average holding days** — durée moyenne des positions
- **Exposition moyenne %** — % du capital investi

---

## Récapitulatif des scripts

| Script | Optimise | Méthode | Temps typique |
|--------|----------|---------|---------------|
| `11c_optimize_convergence_fraction.py` | `convergence_fraction` | Grid-search **in-sample** | 10–30 min |
| `11_optimize_options_stops.py` | `stop_loss_pct` × take-profit | Grid-search + walk-forward | 20–60 min |
| `11d_optimize_entry_threshold.py` | `entry_threshold_pct` | Grid-search + walk-forward | 15–45 min |
| `11b_optimize_rebalance_threshold.py` | `rebalance_log_gap_threshold` (ε) | Grid-search + walk-forward | 15–45 min |
| `compare_options_strategies.py` | Comparaison 3 stratégies | Backtest complet | 30–90 min |
| `10_backtest_options.py` | Single strategy run | Backtest complet | 10–30 min |

Seul 11c classe **in-sample** ; les trois autres réservent une part de l'historique
(`--train-fraction`, 0.60 par défaut) pour juger l'optimum sur des données qui ne l'ont
pas choisi, et affichent `test_sharpe_ratio` à côté de `train_sharpe_ratio`.

---

## Prochaines étapes

1. **Établir baseline:** Exécuter un backtest de référence Kelly sur 2015-2024
2. **Grid-search `convergence_fraction`:** Identifier la fraction optimale
3. **Optimiser stops:** Joint optimization stop-loss/take-profit
4. **Validation OOS:** Tester sur données post-2024
5. **Production:** Intégrer la stratégie au pipeline de trading réel

