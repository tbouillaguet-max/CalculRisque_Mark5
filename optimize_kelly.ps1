# --- Optimisation complète de valuation_gap_expected_value_options ---
$strategy  = "valuation_gap_expected_value_options"
$startDate = "2015-01-01"
$endDate   = "2024-01-01"
$workers   = 4

# 1. convergence_fraction (Kelly est la strategie par defaut de ce script)
python 11c_optimize_convergence_fraction.py `
  --start-date $startDate --end-date $endDate `
  --fraction-grid 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 `
  --workers $workers

# 2. entry_threshold_pct (saisi en % d'ecart, converti en points de log)
python 11d_optimize_entry_threshold.py `
  --strategy $strategy `
  --start-date $startDate --end-date $endDate `
  --gap-grid 10 15 20 25 30 40 `
  --workers $workers

# 3. stop_loss_pct x take-profit (fractions de convergence pour Kelly, pas des %)
python 11_optimize_options_stops.py `
  --strategy $strategy `
  --start-date $startDate --end-date $endDate `
  --stop-loss-grid -30 -25 -20 -15 `
  --take-profit-grid 0.6 0.8 1.0 1.2 `
  --workers $workers

# 4. rebalance_log_gap_threshold (epsilon)
python 11b_optimize_rebalance_threshold.py `
  --strategy $strategy `
  --start-date $startDate --end-date $endDate `
  --epsilon-grid 0 0.05 0.10 0.15 0.20 0.30 `
  --workers $workers

Write-Host "Termine. Resultats sous data/backtest_options/optimize_*.csv"