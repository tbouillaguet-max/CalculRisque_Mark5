"""
Grid-search sur `entry_threshold_pct` (seuil d'entrée) pour une stratégie options.

Rejoue le backtest de 10_backtest_options.py pour CHAQUE valeur du seuil, sur
les MÊMES données chargées UNE SEULE FOIS (comme
11b_optimize_rebalance_threshold.py, dont ce script reprend la structure).
Tous les autres réglages -- moteur ET stratégie -- sont FIXÉS à leurs valeurs
courantes, résolus comme dans 10_backtest_options.py (engine_defaults de la
stratégie sinon config) : le seuil d'entrée est le SEUL paramètre qui varie
d'un run à l'autre.

CE QUE MESURE LE SEUIL. `entry_threshold_pct` est l'écart de valorisation
minimal qu'une ligne doit présenter pour entrer dans l'univers des candidates.
Il s'exprime en POINTS DE LOG x 100, pas en pourcentage d'écart : le défaut
config.OPTIONS_MULTIPLES_ENTRY_THRESHOLD_PCT vaut log(1,20) x 100 = 18,23, ce
qui correspond à un écart de 20 %. La colonne `ecart_equivalent_pct` du tableau
fait la conversion dans les deux sens, parce que lire « 30 » comme « 30 % »
(alors que c'est 35 %) fausserait l'interprétation de toute la grille.

CE QUE LE SEUIL DÉPLACE, ET QU'ON NE VOIT PAS DANS LE SHARPE SEUL :

    - le NOMBRE de candidates, donc le nombre de positions et la
      concentration du portefeuille. Un seuil haut ne rend pas la stratégie
      « plus sélective » gratuitement : il la rend plus concentrée, et le
      Sharpe d'un portefeuille de trois lignes n'est pas comparable à celui
      d'un portefeuille de trente.
    - le SEUIL DE SORTIE, qui n'est pas indépendant : la stratégie multiples
      pose exit_threshold_pct = entry_threshold_pct x exit_threshold_ratio
      (cf. valuation_gap_multiples_options.__init__). Balayer l'entrée
      déplace donc AUSSI la sortie, à hystérésis constante. C'est voulu --
      découpler les deux demanderait une surface 2D, et le ratio est le
      paramètre qui a un sens économique (« de combien le signal doit-il se
      dégrader avant qu'on lâche »), pas la valeur absolue de la sortie. Le
      ratio se fixe avec --exit-threshold-ratio et reste CONSTANT sur toute
      la grille.
    - l'EXPOSITION moyenne : moins de candidates éligibles, c'est du cash qui
      dort. Le tableau affiche avg_exposure_pct à côté du CAGR pour cette
      raison.

POURQUOI CE SEUIL N'A PAS D'OPTIMUM ÉVIDENT. Trop bas, il fait entrer du
bruit : des écarts de valorisation qui ne sont que l'imprécision du modèle de
multiples, et que rien ne fera converger. Trop haut, il ne laisse passer que
les cas extrêmes -- souvent des titres en difficulté réelle, où l'écart
s'explique par autre chose qu'une erreur de marché. Les deux échecs coûtent, et
ils ne se ressemblent pas : le premier se lit dans la friction et le taux de
réussite, le second dans le nombre de trades et le drawdown.

Usage :
    python 11d_optimize_entry_threshold.py
    python 11d_optimize_entry_threshold.py --strategy valuation_gap_multiples_options
    python 11d_optimize_entry_threshold.py --entry-threshold-grid 10 15 20 25 30
    python 11d_optimize_entry_threshold.py --gap-grid 10 15 20 25 30   # en % d'écart
    python 11d_optimize_entry_threshold.py --workers 4                 # process (fork)

Le CSV complet (une ligne par seuil, toutes les métriques) est écrit sous
data/backtest_options/optimize_entry_threshold_<stratégie>_<horodatage>.csv.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import pandas as pd

import config
from backtest import data_loader, metrics as metrics_mod
from backtest.options_engine import OptionsBacktestEngine
from backtest.strategies import OPTIONS_STRATEGY_REGISTRY

logger = logging.getLogger("optimize_entry_threshold")

# 10_backtest_options.py n'est pas un nom de module Python valide (commence
# par un chiffre) : import par importlib, comme le font déjà 11, 11b et 11c.
_cli = importlib.import_module("10_backtest_options")

DEFAULT_STRATEGY = "valuation_gap_expected_value_options"

# Grille en POINTS DE LOG x 100. Le seuil courant y figure toujours (arrondi au
# centième) pour que la ligne de référence soit lisible dans le tableau : une
# grille qui ne contient pas la valeur en production ne dit pas si le
# changement proposé est un gain.
DEFAULT_ENTRY_THRESHOLD_GRID = sorted({
    9.53,    # ecart 10 %
    13.98,   # ecart 15 %
    round(config.OPTIONS_MULTIPLES_ENTRY_THRESHOLD_PCT, 2),  # defaut (ecart 20 %)
    22.31,   # ecart 25 %
    26.24,   # ecart 30 %
    33.65,   # ecart 40 %
    40.55,   # ecart 50 %
})

# Rempli une fois dans le process parent (voir main()), puis rendu disponible
# aux workers par _init_worker : jamais passé en argument de tâche, ce qui
# coûterait une sérialisation d'un DataFrame volumineux par point de grille.
_DATA: dict = {}


def _log_points_to_gap_pct(log_points: float) -> float:
    """18,23 points de log -> 20,0 % d'écart. La grille se raisonne en écart,
    le code de la stratégie travaille en log : la conversion doit apparaître
    dans le tableau, sinon la grille se lit de travers."""
    return (math.exp(log_points / 100.0) - 1.0) * 100.0


def _gap_pct_to_log_points(gap_pct: float) -> float:
    """Réciproque de _log_points_to_gap_pct, pour --gap-grid."""
    return math.log(1.0 + gap_pct / 100.0) * 100.0


def _load_data(benchmark_symbol: str) -> dict:
    logger.info("Chargement des données (une seule fois pour toute la grille)...")
    daily_prices = data_loader.load_daily_prices()
    price_panel = data_loader.build_price_panel(daily_prices)
    valorisation_combinee = data_loader.load_valorisation_combinee_history()
    signal_events = data_loader.build_options_signal_events(valorisation_combinee)
    universe_history = data_loader.load_universe_history()
    fallback_symbols = data_loader.load_current_universe_symbols()
    option_snapshots = data_loader.load_option_snapshots_history()
    material_events = data_loader.load_material_events_8k()
    universe = data_loader.UniverseResolver(universe_history, fallback_symbols)
    benchmark_prices, benchmark_label = data_loader.build_benchmark_series(
        price_panel, universe, symbol=benchmark_symbol,
    )
    return {
        "price_panel": price_panel,
        "signal_events": signal_events,
        "universe_history": universe_history,
        "fallback_symbols": fallback_symbols,
        "option_snapshots": option_snapshots,
        "material_events": material_events,
        "benchmark_prices": benchmark_prices,
        "benchmark_label": benchmark_label,
    }


def _init_worker(benchmark_symbol: str) -> None:
    """Garantit que _DATA est rempli DANS le worker, quelle que soit la façon
    dont le pool a démarré le process.

    Sous fork (défaut Linux), le worker hérite du _DATA déjà rempli par le
    parent : il n'y a rien à charger, et le test ci-dessous court-circuite.
    Sous spawn (défaut Windows, et macOS depuis 3.8), le worker ré-importe le
    module à neuf -- _DATA y vaut {} et CHAQUE tâche échouerait en
    KeyError: 'price_panel'. C'est exactement ce qui se produisait tant que ce
    module supposait le fork."""
    global _DATA
    if _DATA:
        return
    _DATA = _load_data(benchmark_symbol)


def _run_one(
    entry_threshold: float,
    strategy_name: str,
    strategy_params: dict,
    baseline_settings: dict,
    engine_kwargs: dict,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    train_fraction: Optional[float] = None,
) -> dict:
    """Un run complet pour ce seuil d'entrée. Isolé en fonction de module
    (plutôt que méthode/closure) pour rester picklable par
    ProcessPoolExecutor -- les données volumineuses (_DATA) ne sont PAS des
    arguments : elles viennent du fork, voir le commentaire sur _DATA
    plus haut."""
    row = {
        "entry_threshold_pct": entry_threshold,
        "ecart_equivalent_pct": _log_points_to_gap_pct(entry_threshold),
        "error": None,
    }
    try:
        strategy_cls = OPTIONS_STRATEGY_REGISTRY[strategy_name]
        strategy = strategy_cls(**{**strategy_params, "entry_threshold_pct": entry_threshold})
        # Le seuil de sortie est DÉRIVÉ du seuil d'entrée (cf. docstring) :
        # on le journalise pour que le CSV porte les deux, sinon la moitié du
        # changement testé reste invisible.
        row["exit_threshold_pct"] = getattr(strategy, "exit_threshold_pct", None)

        engine = OptionsBacktestEngine(
            price_panel=_DATA["price_panel"],
            signal_events=_DATA["signal_events"],
            universe_history=_DATA["universe_history"],
            fallback_universe_symbols=_DATA["fallback_symbols"],
            option_snapshots=_DATA["option_snapshots"],
            material_events_8k=_DATA["material_events"],
            strategy=strategy,
            stop_loss_pct=baseline_settings["stop_loss_pct"],
            take_profit_pct=baseline_settings["take_profit_pct"],
            target_tenor_days=baseline_settings["target_tenor_days"],
            stop_basis=baseline_settings["stop_basis"],
            exit_when_signal_lost=baseline_settings["exit_when_signal_lost"],
            roll_when_days_left=baseline_settings["roll_when_days_left"],
            daily_rebalance=baseline_settings["daily_rebalance"],
            vol_mode=baseline_settings["vol_mode"],
            min_resize_relative_pct=baseline_settings["min_resize_relative_pct"],
            min_holding_days=baseline_settings["min_holding_days"],
            rebalance_log_gap_threshold=baseline_settings["rebalance_log_gap_threshold"],
            start_date=start_date,
            end_date=end_date,
            **engine_kwargs,
        )
        equity_curve, _positions_history, trades, _signals_history = engine.run()
        run_metrics = metrics_mod.compute_metrics(
            equity_curve, trades,
            risk_free_rate=config.RISK_FREE_RATE,
            benchmark_prices=_DATA["benchmark_prices"],
            extra=engine.execution_diagnostics(),
        )
        for key in (
            "num_trades", "win_rate_pct", "profit_factor", "total_return_pct",
            "cagr_pct", "annualized_volatility_pct", "sharpe_ratio", "sortino_ratio",
            "max_drawdown_pct", "calmar_ratio", "avg_holding_days",
            # L'exposition dit ce que le seuil coûte en capital qui dort : un
            # seuil haut peut « améliorer » le Sharpe en n'investissant plus.
            "avg_exposure_pct",
            "total_commission_dollar", "total_slippage_dollar",
            "total_friction_dollar", "total_friction_pct_of_initial",
            "dropped_orders_count",
            # Compteurs propres à la stratégie « espérance de gain » (absents
            # des deux autres, laissés à None dans ce cas) : ils disent si la
            # contrainte d'espérance positive a mordu en plus du seuil.
            "expected_value_evaluations_count", "dropped_expected_value_negative_count",
        ):
            row[key] = run_metrics.get(key)

        # Séparation apprentissage / test : même raison qu'en 11 et 11b -- un
        # seuil choisi sur l'historique complet est choisi sur les données qui
        # servent ensuite à le juger (cf. metrics.split_period_metrics).
        if train_fraction:
            row.update(metrics_mod.split_period_metrics(
                equity_curve, trades,
                metrics_mod.resolve_split_date(equity_curve, train_fraction),
                risk_free_rate=config.RISK_FREE_RATE,
                benchmark_prices=_DATA["benchmark_prices"],
            ))

        if not trades.empty and "exit_reason" in trades.columns:
            row["exits_by_reason"] = trades["exit_reason"].value_counts().to_dict()
    except Exception as exc:  # noqa: BLE001 -- un seuil qui plante ne doit pas tuer la grille
        logger.exception("Échec pour entry_threshold_pct=%s", entry_threshold)
        row["error"] = str(exc)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, choices=sorted(OPTIONS_STRATEGY_REGISTRY))
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument(
        "--entry-threshold-grid", type=float, nargs="+", default=None,
        help="Seuils à tester, en POINTS DE LOG x 100 (défaut: "
             f"{DEFAULT_ENTRY_THRESHOLD_GRID}, soit des écarts de 10 %% à 50 %%). "
             "Voir --gap-grid pour saisir directement des %% d'écart.",
    )
    parser.add_argument(
        "--gap-grid", type=float, nargs="+", default=None,
        help="Mêmes seuils, saisis en %% D'ÉCART de valorisation (ex: 10 20 30) et "
             "convertis en points de log. Exclusif avec --entry-threshold-grid.",
    )
    parser.add_argument(
        "--exit-threshold-ratio", type=float, default=config.OPTIONS_EXIT_THRESHOLD_RATIO,
        help="Hystérésis, CONSTANTE sur toute la grille (défaut: %(default)s). Le seuil de "
             "sortie vaut entry_threshold_pct x ce ratio : balayer l'entrée déplace donc "
             "aussi la sortie, à hystérésis constante (voir la docstring du script).",
    )
    parser.add_argument(
        "--train-fraction", type=float, default=0.60,
        help="Part initiale de l'historique servant à CHOISIR le seuil ; le reste sert à le "
             "juger (défaut: %(default)s). Le classement se fait sur train_sharpe_ratio et le "
             "CSV porte test_sharpe_ratio à côté -- c'est ce second chiffre qui dit si "
             "l'optimum survit à des données qui ne l'ont pas choisi.",
    )
    parser.add_argument(
        "--no-walk-forward", dest="train_fraction", action="store_const", const=None,
        help="Classe sur l'historique COMPLET (in-sample). Le résultat est alors un optimum "
             "in-sample, à ne pas lire comme une performance attendue.",
    )
    parser.add_argument(
        "--min-trades", type=int, default=15,
        help="Seuils avec moins de trades écartés du CLASSEMENT (pas du CSV, défaut: "
             "%(default)s). Un seuil haut finit toujours par n'ouvrir que deux positions "
             "chanceuses : sans ce plancher, il gagnerait le classement sans rien prouver.",
    )
    parser.add_argument("--entry-threshold-pct", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--benchmark-symbol", default=config.BENCHMARK_SYMBOL)
    parser.add_argument("--initial-capital", type=float, default=config.OPTIONS_INITIAL_CAPITAL)
    parser.add_argument("--commission-per-contract", type=float, default=config.OPTIONS_COMMISSION_PER_CONTRACT)
    parser.add_argument("--slippage-pct-of-premium", type=float, default=config.OPTIONS_SLIPPAGE_PCT_OF_PREMIUM)
    parser.add_argument("--commission-min-per-order", type=float, default=config.OPTIONS_COMMISSION_MIN_PER_ORDER)
    parser.add_argument("--max-fee-pct-of-trade", type=float, default=config.OPTIONS_MAX_FEE_PCT_OF_TRADE)
    parser.add_argument("--fee-bump-max-extra-pct", type=float, default=config.OPTIONS_FEE_BUMP_MAX_EXTRA_PCT)
    parser.add_argument("--min-deployment-pct", type=float, default=config.OPTIONS_MIN_DEPLOYMENT_PCT)
    parser.add_argument("--max-delta-notional-pct", type=float, default=config.OPTIONS_MAX_DELTA_NOTIONAL_PCT)
    parser.add_argument("--max-trade-pct-of-nav", type=float, default=config.OPTIONS_MAX_TRADE_PCT_OF_NAV)
    parser.add_argument("--max-trade-dollar", type=float, default=config.OPTIONS_MAX_TRADE_DOLLAR)
    parser.add_argument(
        "--fractional-contracts", dest="whole_contracts", action="store_false",
        default=config.OPTIONS_WHOLE_CONTRACTS,
    )
    parser.add_argument("--real-snapshot-tolerance-days", type=int, default=config.OPTIONS_REAL_SNAPSHOT_TOLERANCE_DAYS)
    parser.add_argument("--momentum-min-pct", type=float, default=config.BACKTEST_MOMENTUM_MIN_PCT)
    parser.add_argument("--no-momentum-filter", dest="momentum_min_pct", action="store_const", const=None)
    parser.add_argument("--workers", type=int, default=1, help="Process en parallèle (fork). 1 = séquentiel.")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.entry_threshold_grid and args.gap_grid:
        raise SystemExit(
            "--entry-threshold-grid et --gap-grid décrivent la MÊME grille dans deux unités "
            "différentes : n'en donne qu'une."
        )
    if args.gap_grid:
        if any(g <= -100 for g in args.gap_grid):
            raise SystemExit("--gap-grid attend des écarts en %% strictement supérieurs à -100.")
        threshold_grid = sorted(_gap_pct_to_log_points(g) for g in args.gap_grid)
    else:
        threshold_grid = sorted(args.entry_threshold_grid or DEFAULT_ENTRY_THRESHOLD_GRID)

    if any(t < 0 for t in threshold_grid):
        raise SystemExit(
            "Le seuil d'entrée ne peut pas être négatif : il mesure un écart ABSOLU de "
            "valorisation (|log(V/P)|), pas un écart signé."
        )
    if any(t == 0 for t in threshold_grid):
        logger.warning(
            "Un seuil de 0 fait entrer TOUTE ligne valorisée, y compris celles dont l'écart "
            "n'est que le bruit du modèle de multiples. C'est une borne de référence utile, "
            "pas un réglage."
        )

    strategy_cls = OPTIONS_STRATEGY_REGISTRY[args.strategy]

    # Réglages moteur FIXÉS pour toute la grille, résolus UNE FOIS via la même
    # logique que 10_backtest_options.py (engine_defaults de la stratégie
    # sinon fallback config). Le seuil d'entrée n'est PAS un réglage moteur :
    # c'est un paramètre de la stratégie, injecté par _run_one.
    settings_args = argparse.Namespace(
        stop_loss_pct=None, take_profit_pct=None, target_tenor_days=None, stop_basis=None,
        exit_when_signal_lost=None, roll_when_days_left=None, daily_rebalance=None,
        vol_mode=None, min_resize_relative_pct=None,
    )
    baseline_settings, imposed = _cli.resolve_engine_settings(strategy_cls, settings_args)
    if imposed:
        logger.info("Réglages imposés par la stratégie '%s' : %s.", args.strategy, ", ".join(imposed))

    # L'hystérésis reste CONSTANTE sur toute la grille : c'est ce qui rend les
    # points comparables entre eux (cf. docstring).
    strategy_params = {"exit_threshold_ratio": args.exit_threshold_ratio}

    start_date = pd.Timestamp(args.start_date) if args.start_date else None
    end_date = pd.Timestamp(args.end_date) if args.end_date else None

    engine_kwargs = dict(
        initial_capital=args.initial_capital,
        commission_per_contract=args.commission_per_contract,
        slippage_pct_of_premium=args.slippage_pct_of_premium,
        commission_min_per_order=args.commission_min_per_order,
        max_fee_pct_of_trade=args.max_fee_pct_of_trade,
        min_deployment_pct=args.min_deployment_pct,
        max_delta_notional_pct=args.max_delta_notional_pct,
        fee_bump_max_extra_pct=args.fee_bump_max_extra_pct,
        whole_contracts=args.whole_contracts,
        real_snapshot_tolerance_days=args.real_snapshot_tolerance_days,
        momentum_min_pct=args.momentum_min_pct,
        max_trade_dollar=args.max_trade_dollar,
        max_trade_pct_of_nav=args.max_trade_pct_of_nav,
    )

    global _DATA
    _DATA = _load_data(args.benchmark_symbol)

    logger.info(
        "Backtest options '%s' : %d seuils d'entrée %s (soit des écarts de %s)...",
        args.strategy, len(threshold_grid),
        [round(t, 2) for t in threshold_grid],
        ", ".join(f"{_log_points_to_gap_pct(t):.0f}%" for t in threshold_grid),
    )

    rows: list[dict] = []
    if args.workers <= 1:
        for i, seuil in enumerate(threshold_grid, 1):
            logger.info(
                "[%d/%d] entry_threshold_pct=%.2f (écart %.1f%%)",
                i, len(threshold_grid), seuil, _log_points_to_gap_pct(seuil),
            )
            rows.append(_run_one(
                seuil, args.strategy, strategy_params, baseline_settings, engine_kwargs,
                start_date, end_date, args.train_fraction,
            ))
    else:
        # initializer plutôt que fork implicite : sous fork le worker hérite
        # de _DATA et _init_worker ne fait rien, sous spawn il le charge. Sans
        # cela, tout le pool échoue sur les plateformes qui ne forkent pas.
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.benchmark_symbol,),
        ) as pool:
            futures = {
                pool.submit(
                    _run_one, seuil, args.strategy, strategy_params,
                    baseline_settings, engine_kwargs, start_date, end_date,
                    args.train_fraction,
                ): seuil
                for seuil in threshold_grid
            }
            for i, future in enumerate(as_completed(futures), 1):
                seuil = futures[future]
                logger.info("[%d/%d] terminé : entry_threshold_pct=%.2f", i, len(threshold_grid), seuil)
                rows.append(future.result())

    results = pd.DataFrame(rows).sort_values("entry_threshold_pct").reset_index(drop=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_csv or (
        config.DIR_BACKTEST_OPTIONS / f"optimize_entry_threshold_{args.strategy}_{run_id}.csv"
    )
    out_path = str(out_path)
    config.DIR_BACKTEST_OPTIONS.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    logger.info("Résultats complets (%d lignes) sauvegardés dans %s", len(results), out_path)

    n_errors = results["error"].notna().sum()
    if n_errors:
        logger.warning("%d/%d seuils ont échoué (voir la colonne 'error' du CSV).", n_errors, len(results))

    display_cols = [
        "entry_threshold_pct", "ecart_equivalent_pct", "exit_threshold_pct",
        "num_trades", "avg_exposure_pct", "win_rate_pct",
        "total_friction_pct_of_initial",
        "cagr_pct", "sharpe_ratio", "max_drawdown_pct", "calmar_ratio",
        "train_sharpe_ratio", "test_sharpe_ratio",
    ]
    display_cols = [c for c in display_cols if c in results.columns]
    with pd.option_context("display.width", 240, "display.max_columns", 30, "display.float_format", "{:,.3f}".format):
        print(f"\n=== entry_threshold_pct -- stratégie: {args.strategy} ===")
        print(results[display_cols].to_string(index=False, na_rep="-"))

    usable = results[results["error"].isna()].copy()
    if usable.empty:
        logger.error("Aucun seuil exploitable (tous en erreur).")
        sys.exit(1)

    # Le plancher de trades s'applique au CLASSEMENT seulement : les lignes
    # écartées restent dans le CSV, parce que « ce seuil n'ouvre que 3
    # positions » est un résultat, pas une donnée manquante.
    eligible = usable[usable["num_trades"].fillna(0) >= args.min_trades]
    if eligible.empty:
        logger.error(
            "Aucun seuil n'atteint %d trades : la grille entière est trop sélective sur cette "
            "période. Élargis --entry-threshold-grid vers le bas, ou abaisse --min-trades en "
            "sachant que le classement portera alors sur des échantillons minuscules.",
            args.min_trades,
        )
        sys.exit(1)
    n_ecartes = len(usable) - len(eligible)
    if n_ecartes:
        logger.info(
            "%d seuil(s) écarté(s) du classement pour moins de %d trades (conservés dans le CSV).",
            n_ecartes, args.min_trades,
        )

    # Classement sur la période d'APPRENTISSAGE quand elle existe (cf. 11, 11b).
    ranking_key = (
        "train_sharpe_ratio"
        if "train_sharpe_ratio" in eligible.columns and not eligible["train_sharpe_ratio"].isna().all()
        else "sharpe_ratio"
    )
    ranked = eligible.sort_values(ranking_key, ascending=False, na_position="last")
    best = ranked.iloc[0]
    with pd.option_context("display.width", 240, "display.float_format", "{:,.3f}".format):
        print(f"\n=== Top {min(args.top_n, len(ranked))} par {ranking_key} ===")
        print(ranked[display_cols].head(args.top_n).to_string(index=False, na_rep="-"))

    if best["entry_threshold_pct"] in (min(threshold_grid), max(threshold_grid)):
        logger.warning(
            "L'optimum (seuil=%.2f, écart %.1f%%) est en BORD de grille -- élargis "
            "--entry-threshold-grid pour vérifier qu'il ne s'agit pas d'un optimum tronqué "
            "par la plage testée.",
            best["entry_threshold_pct"], best["ecart_equivalent_pct"],
        )
    if "train_sharpe_ratio" in ranked.columns and pd.notna(best.get("test_sharpe_ratio")):
        logger.info(
            "Hors échantillon (coupure au %s) : sharpe %.3f en apprentissage -> %.3f en test.",
            best.get("split_date", "?"), best.get("train_sharpe_ratio") or float("nan"),
            best.get("test_sharpe_ratio") or float("nan"),
        )
    logger.info(
        "Meilleur seuil : %.2f points de log (écart %.1f%%, sortie à %.2f) -- sharpe=%.2f, "
        "%d trades, exposition moyenne %.1f%%, max_drawdown=%.1f%%.",
        best["entry_threshold_pct"], best["ecart_equivalent_pct"],
        best.get("exit_threshold_pct") or float("nan"),
        best.get("sharpe_ratio") or float("nan"), best.get("num_trades") or 0,
        best.get("avg_exposure_pct") or float("nan"),
        best.get("max_drawdown_pct") or float("nan"),
    )


if __name__ == "__main__":
    main()
