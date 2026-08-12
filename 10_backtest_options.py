"""
Backtest d'une stratégie OPTIONS construite sur la valorisation combinée
(multiples sectoriels par année en priorité, DCF en repli -- voir
06b_calcul_valorisation_combinee.py) : achète des CALL sur les entreprises
sous-évaluées, des PUT sur les survalorisées, dimensionnés par le delta pour
une exposition $ cible. Voir backtest/options_engine.py pour les hypothèses
détaillées (repli Black-Scholes, positions gelées, stop-loss/take-profit sur
la prime, expiration).

Distinct de 09_backtest.py (stratégie actions, DCF seul, inchangée).

Prérequis : 01(b), 02, 03b, 04, 05, 07, 06b_calcul_valorisation_combinee.py.
08_recuperation_options.py est optionnel mais recommandé : plus tu accumules
de snapshots réels (data/options/history/), moins le backtest s'appuie sur
du Black-Scholes simulé.

Deux stratégies options sont disponibles (`--list-strategies`) :

    valuation_gap_options            Signal combiné (multiples, DCF en repli),
                                     contrat ATM à ~9 mois, stops sur la PRIME,
                                     positions gelées tant qu'un stop ou
                                     l'expiration ne les ferme pas.
    valuation_gap_multiples_options  Multiples sectoriels SEULS, écart rapporté
                                     à la valeur théorique, strike à mi-chemin
                                     théorique/cours, échéance 2 ans roulée à
                                     9 mois, stops sur le COURS DU SOUS-JACENT,
                                     et vente dès que l'écart repasse sous le
                                     seuil. Voir sa docstring pour le détail.

Chaque stratégie peut imposer les réglages du moteur qui font partie de sa
thèse (attribut de classe engine_defaults) ; une option passée explicitement
en ligne de commande reste prioritaire.

Usage :
    python 10_backtest_options.py
    python 10_backtest_options.py --start-date 2015-01-01
    python 10_backtest_options.py --entry-threshold-pct 25 --stop-loss-pct -40 --take-profit-pct 80
    python 10_backtest_options.py --strategy valuation_gap_multiples_options --start-date 2015-01-01
    python 10_backtest_options.py --list-strategies
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

import pandas as pd

import config
from backtest import data_loader, metrics as metrics_mod
from backtest.options_engine import OptionsBacktestEngine
from backtest.strategies import OPTIONS_STRATEGY_REGISTRY

logger = logging.getLogger("backtest_options.cli")


# Réglages du moteur qu'une stratégie peut vouloir imposer (via son attribut
# de classe engine_defaults) parce qu'ils font partie intégrante de sa thèse --
# une échéance 2 ans avec des stops sur la prime n'aurait par exemple aucun
# sens (voir la docstring de backtest/strategies/valuation_gap_multiples_options.py).
# Priorité : option de ligne de commande explicite > engine_defaults de la
# stratégie > valeur ci-dessous (comportement historique).
ENGINE_SETTING_FALLBACKS = {
    "stop_loss_pct": config.OPTIONS_STOP_LOSS_PCT,
    "take_profit_pct": config.OPTIONS_TAKE_PROFIT_PCT,
    "target_tenor_days": config.OPTIONS_TARGET_TENOR_DAYS,
    "stop_basis": config.OPTIONS_STOP_BASIS,
    "exit_when_signal_lost": False,
    "roll_when_days_left": config.OPTIONS_ROLL_WHEN_DAYS_LEFT,
    "daily_rebalance": False,
    "vol_mode": "frozen",
    "min_resize_relative_pct": config.OPTIONS_MIN_RESIZE_RELATIVE_PCT,
    "rebalance_log_gap_threshold": config.OPTIONS_REBALANCE_LOG_GAP_THRESHOLD,
}


def resolve_engine_settings(strategy_cls, args: argparse.Namespace) -> tuple[dict, list[str]]:
    """Retourne (réglages, liste des réglages imposés par la stratégie) --
    la seconde valeur sert uniquement à les journaliser au lancement, pour
    qu'un réglage non demandé en ligne de commande ne soit pas une surprise."""
    declared = getattr(strategy_cls, "engine_defaults", {}) or {}
    settings, from_strategy = {}, []
    for key, fallback in ENGINE_SETTING_FALLBACKS.items():
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            settings[key] = cli_value
        elif key in declared:
            settings[key] = declared[key]
            from_strategy.append(f"{key}={declared[key]}")
        else:
            settings[key] = fallback
    return settings, from_strategy


def parse_strategy_params(pairs: list[str]) -> dict:
    params = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--strategy-param attend 'clé=valeur', reçu: {pair!r}")
        key, raw_value = pair.split("=", 1)
        try:
            value = float(raw_value)
            if value.is_integer():
                value = int(value)
        except ValueError:
            value = raw_value
        params[key] = value
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="valuation_gap_options")
    parser.add_argument("--list-strategies", action="store_true")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--initial-capital", type=float, default=config.OPTIONS_INITIAL_CAPITAL)
    parser.add_argument("--commission-per-contract", type=float, default=config.OPTIONS_COMMISSION_PER_CONTRACT)
    parser.add_argument("--slippage-pct-of-premium", type=float, default=config.OPTIONS_SLIPPAGE_PCT_OF_PREMIUM)
    parser.add_argument(
        "--commission-min-per-order", type=float, default=config.OPTIONS_COMMISSION_MIN_PER_ORDER,
        help="Minimum facturé PAR ORDRE (IBKR : 1,00$), pas par contrat. 0 pour désactiver.",
    )
    parser.add_argument(
        "--max-fee-pct-of-trade", type=float, default=config.OPTIONS_MAX_FEE_PCT_OF_TRADE,
        help="Abandonne un ordre d'ENTRÉE/renforcement dont les frais dépassent ce %% de sa "
             "valeur (jamais appliqué aux sorties). 0 pour désactiver.",
    )
    parser.add_argument(
        "--fee-bump-max-extra-pct", type=float, default=config.OPTIONS_FEE_BUMP_MAX_EXTRA_PCT,
        help="Tolérance d'exposition supplémentaire pour remonter un ordre à la taille où le "
             "minimum par ordre cesse de mordre. 0 pour désactiver la remontée.",
    )
    parser.add_argument(
        "--min-deployment-pct", type=float, default=config.OPTIONS_MIN_DEPLOYMENT_PCT,
        help="Part minimale du NAV investie en primes (défaut: %(default)s) : en dessous, les "
             "positions ouvertes sont renforcées au prorata. Arbitre cash contre theta contre "
             "levier -- voir le commentaire de config.OPTIONS_MIN_DEPLOYMENT_PCT. 0 pour désactiver.",
    )
    parser.add_argument(
        "--max-delta-notional-pct", type=float, default=config.OPTIONS_MAX_DELTA_NOTIONAL_PCT,
        help="Exposition notionnelle delta-équivalente maximale, en %% du NAV (défaut: "
             "%(default)s). Borne le levier : le redéploiement du cash s'arrête dès ce plafond "
             "atteint, même si --min-deployment-pct n'est pas satisfait. 0 pour désactiver.",
    )
    parser.add_argument(
        "--delever-tolerance-pct", type=float, default=config.OPTIONS_DELEVER_TOLERANCE_PCT,
        help="Marge au-delà du plafond de levier avant de réduire les positions au prorata "
             "(défaut: %(default)s). Le plafond est désormais appliqué à l'ORDRE et à la "
             "POSITION : sans cette bande, le gamma ferait vendre quelques contrats presque "
             "chaque jour. 0 dé-lève au moindre dépassement.",
    )
    parser.add_argument(
        "--max-trade-pct-of-nav", type=float, default=config.OPTIONS_MAX_TRADE_PCT_OF_NAV,
        help="Montant maximal décaissé par ORDRE D'ACHAT, en %% du NAV (défaut: %(default)s). Remplace le plafond en dollars absolus, qui ne suivait pas la taille du portefeuille. 0 pour désactiver.",
    )
    parser.add_argument(
        "--max-trade-dollar", type=float, default=config.OPTIONS_MAX_TRADE_DOLLAR,
        help="Montant maximal décaissé par ORDRE D'ACHAT, frais inclus (défaut: %(default)s). "
             "Plafond par ordre et non par position : une ligne peut le dépasser en cumulant "
             "plusieurs renforcements. Jamais appliqué aux ventes. 0 pour désactiver.",
    )
    parser.add_argument(
        "--take-profit-convergence-fraction", type=float,
        default=config.OPTIONS_TAKE_PROFIT_CONVERGENCE_FRACTION,
        help="Fraction du chemin vers la valeur théorique déclenchant la prise de gain, pour les "
             "stratégies qui visent une convergence (défaut: %(default)s). Sans effet sur une "
             "stratégie ATM (valuation_gap_options), qui garde --take-profit-pct. 0 pour "
             "revenir au seuil fixe partout.",
    )
    parser.add_argument(
        "--fractional-contracts", dest="whole_contracts", action="store_false",
        default=config.OPTIONS_WHOLE_CONTRACTS,
        help="Autorise des contrats fractionnaires (irréaliste : les options se négocient "
             "par contrats entiers). À n'utiliser que pour comparer avec l'ancien comportement.",
    )
    # default=None sur les réglages qu'une stratégie peut imposer : distingue
    # "non demandé" (la stratégie décide) de "demandé explicitement" (priorité
    # à la ligne de commande). Voir resolve_engine_settings.
    parser.add_argument("--stop-loss-pct", type=float, default=None, help="Négatif, ex: -50. Base fixée par --stop-basis.")
    parser.add_argument("--take-profit-pct", type=float, default=None)
    parser.add_argument("--entry-threshold-pct", type=float, default=None)
    parser.add_argument("--target-tenor-days", type=int, default=None)
    parser.add_argument(
        "--stop-basis", choices=("premium", "underlying"), default=None,
        help="Sur quoi mesurer stop-loss/take-profit : la prime de l'option, ou le cours du sous-jacent.",
    )
    parser.add_argument(
        "--exit-when-signal-lost", dest="exit_when_signal_lost", action="store_true", default=None,
        help="Vend une position dont l'écart est repassé sous le seuil, au lieu de la geler.",
    )
    parser.add_argument("--no-exit-when-signal-lost", dest="exit_when_signal_lost", action="store_false")
    parser.add_argument(
        "--roll-when-days-left", type=int, default=None,
        help="Réexamine la position à N jours de l'expiration : roulée sur une échéance pleine "
             "si elle passe encore les filtres de la stratégie, clôturée sinon (0 = jamais).",
    )
    parser.add_argument(
        "--daily-rebalance", dest="daily_rebalance", action="store_true", default=None,
        help="Réévalue l'éligibilité tous les jours de bourse (cours du jour), pas seulement "
             "aux dates de dépôt SEC.",
    )
    parser.add_argument("--no-daily-rebalance", dest="daily_rebalance", action="store_false")
    parser.add_argument(
        "--vol-mode", choices=("frozen", "rolling"), default=None,
        help="Volatilité de repricing : figée à l'entrée, ou suivie au jour le jour "
             "(volatilité réalisée courante remise à l'échelle de l'entrée).",
    )
    parser.add_argument("--real-snapshot-tolerance-days", type=int, default=config.OPTIONS_REAL_SNAPSHOT_TOLERANCE_DAYS)
    parser.add_argument(
        "--min-resize-relative-pct", type=float, default=None,
        help="Une position déjà ouverte n'est resize/renforcée que si le changement dépasse ce %% "
             "de ses contrats actuels (défaut de la stratégie, sinon config.OPTIONS_MIN_RESIZE_"
             "RELATIVE_PCT = 15). S'applique aussi bien au redimensionnement sur dépôt SEC qu'au "
             "redéploiement quotidien du cash oisif (_deploy_idle_cash) -- ce dernier tourne CHAQUE "
             "jour de bourse sans lui, ce qui peut consommer tout le capital en friction pure sur "
             "un backtest de plusieurs années. Ex: --min-resize-relative-pct 5 (plus réactif) ou "
             "25 (moins de friction). 0 désactive le filtre (tout changement déclenche un ordre).",
    )
    parser.add_argument(
        "--rebalance-log-gap-threshold", type=float, default=None,
        help="ε (points de log(V/P), défaut config.OPTIONS_REBALANCE_LOG_GAP_THRESHOLD = 0.15) : "
             "sur le mécanisme de rebalancement SUR DÉPÔT SEC, une position déjà détenue n'est "
             "redimensionnée que si |log(V/P) - last_rebalance_log_gap| dépasse ce seuil depuis "
             "le dernier trade réel sur cette position. Un nouveau candidat s'ouvre toujours sans "
             "ce test. 0 désactive le filtre (redimensionne dès que le signal a un effet).",
    )
    parser.add_argument(
        "--momentum-min-pct", type=float, default=config.BACKTEST_MOMENTUM_MIN_PCT,
        help="Momentum 12-1 minimal (en %%), ORIENTÉ dans le sens de la position (un titre en "
             "forte hausse est écarté côté put). Ex: -10. --no-momentum-filter pour désactiver.",
    )
    parser.add_argument(
        "--no-momentum-filter", dest="momentum_min_pct", action="store_const", const=None,
        help="Désactive le filtre momentum.",
    )
    parser.add_argument("--strategy-param", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--benchmark-symbol", default=config.BENCHMARK_SYMBOL,
        help="Indice de référence (défaut: %(default)s). Doit être présent dans les cours "
             "quotidiens ; sinon un indice équipondéré de l'univers point-in-time est reconstruit.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.list_strategies:
        for name in sorted(OPTIONS_STRATEGY_REGISTRY):
            print(name)
        return

    if args.strategy not in OPTIONS_STRATEGY_REGISTRY:
        logger.error("Stratégie options inconnue: %s. Disponibles: %s", args.strategy, sorted(OPTIONS_STRATEGY_REGISTRY))
        sys.exit(1)

    logger.warning(
        "RISQUE DE VOLATILITÉ NON MODÉLISÉ. Le repricing quotidien des positions se fait à "
        "volatilité figée (--vol-mode frozen) ou suivant la volatilité RÉALISÉE "
        "(--vol-mode rolling), jamais suivant la volatilité IMPLICITE : le pipeline ne "
        "collecte aucune surface de volatilité historique. Or une option longue est longue "
        "de vega -- un effondrement de l'implicite lui fait perdre de l'argent même quand le "
        "sous-jacent va dans son sens, et ce P&L n'apparaît nulle part dans les résultats "
        "ci-dessous. Les rendements de ce backtest sont donc à lire comme une borne "
        "OPTIMISTE sur toute période de compression de volatilité."
    )

    logger.info("Chargement des données...")
    daily_prices = data_loader.load_daily_prices()
    price_panel = data_loader.build_price_panel(daily_prices)
    valorisation_combinee = data_loader.load_valorisation_combinee_history()
    signal_events = data_loader.build_options_signal_events(valorisation_combinee)
    universe_history = data_loader.load_universe_history()
    fallback_symbols = data_loader.load_current_universe_symbols()
    option_snapshots = data_loader.load_option_snapshots_history()
    material_events = data_loader.load_material_events_8k()

    strategy_cls = OPTIONS_STRATEGY_REGISTRY[args.strategy]
    engine_settings, imposed = resolve_engine_settings(strategy_cls, args)
    if imposed:
        logger.info("Réglages imposés par la stratégie '%s' : %s.", args.strategy, ", ".join(imposed))

    # Seuls les paramètres réellement demandés sont transmis : sinon chaque
    # stratégie recevrait les valeurs par défaut d'une autre, au lieu des
    # siennes (les deux stratégies options n'ont ni le même seuil ni la même
    # base de calcul de l'écart).
    strategy_params = {}
    if args.entry_threshold_pct is not None:
        strategy_params["entry_threshold_pct"] = args.entry_threshold_pct
    strategy_params.update(parse_strategy_params(args.strategy_param))
    strategy = strategy_cls(**strategy_params)

    start_date = pd.Timestamp(args.start_date) if args.start_date else None
    end_date = pd.Timestamp(args.end_date) if args.end_date else None

    engine = OptionsBacktestEngine(
        price_panel=price_panel,
        signal_events=signal_events,
        universe_history=universe_history,
        fallback_universe_symbols=fallback_symbols,
        option_snapshots=option_snapshots,
        strategy=strategy,
        initial_capital=args.initial_capital,
        commission_per_contract=args.commission_per_contract,
        slippage_pct_of_premium=args.slippage_pct_of_premium,
        commission_min_per_order=args.commission_min_per_order,
        max_fee_pct_of_trade=args.max_fee_pct_of_trade,
        min_deployment_pct=args.min_deployment_pct,
        max_delta_notional_pct=args.max_delta_notional_pct,
        delever_tolerance_pct=args.delever_tolerance_pct,
        max_trade_dollar=args.max_trade_dollar,
        max_trade_pct_of_nav=args.max_trade_pct_of_nav,
        take_profit_convergence_fraction=args.take_profit_convergence_fraction,
        fee_bump_max_extra_pct=args.fee_bump_max_extra_pct,
        whole_contracts=args.whole_contracts,
        stop_loss_pct=engine_settings["stop_loss_pct"],
        take_profit_pct=engine_settings["take_profit_pct"],
        target_tenor_days=engine_settings["target_tenor_days"],
        real_snapshot_tolerance_days=args.real_snapshot_tolerance_days,
        momentum_min_pct=args.momentum_min_pct,
        min_resize_relative_pct=engine_settings["min_resize_relative_pct"],
        rebalance_log_gap_threshold=engine_settings["rebalance_log_gap_threshold"],
        material_events_8k=material_events,
        stop_basis=engine_settings["stop_basis"],
        exit_when_signal_lost=engine_settings["exit_when_signal_lost"],
        roll_when_days_left=engine_settings["roll_when_days_left"],
        daily_rebalance=engine_settings["daily_rebalance"],
        vol_mode=engine_settings["vol_mode"],
        start_date=start_date,
        end_date=end_date,
    )

    logger.info(
        "Backtest options '%s' du %s au %s (%d jours de bourse)...",
        args.strategy, engine.calendar[0].date(), engine.calendar[-1].date(), len(engine.calendar),
    )
    equity_curve, positions_history, trades, signals_history = engine.run()
    benchmark_prices, benchmark_label = data_loader.build_benchmark_series(
        price_panel, engine.universe, symbol=args.benchmark_symbol,
    )
    run_metrics = metrics_mod.compute_metrics(
        equity_curve, trades,
        risk_free_rate=config.RISK_FREE_RATE,
        benchmark_prices=benchmark_prices,
        extra=engine.execution_diagnostics(),
    )
    run_metrics["benchmark_label"] = benchmark_label

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = config.DIR_BACKTEST_OPTIONS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_curve.to_parquet(out_dir / "equity_curve.parquet", index=False, engine="pyarrow")
    positions_history.to_parquet(out_dir / "positions_history.parquet", index=False, engine="pyarrow")
    trades.to_parquet(out_dir / "trades.parquet", index=False, engine="pyarrow")
    signals_history.to_parquet(out_dir / "signals_history.parquet", index=False, engine="pyarrow")
    # Journal des exécutions : une ligne par fill, ACHAT COMME VENTE (cf.
    # options_engine._record_fill). trades.parquet, lui, n'enregistre que les
    # ventes et les présente comme des allers-retours dont l'entry_date est la
    # PREMIÈRE ouverture de la position -- une position construite en
    # plusieurs achats y solde donc plus de contrats qu'il n'en a été acheté à
    # cette date. Ce journal est ce qui rend la conservation des quantités et
    # la friction auditables ligne à ligne.
    pd.DataFrame(engine.executions).to_parquet(
        out_dir / "executions.parquet", index=False, engine="pyarrow",
    )
    (out_dir / "metrics.json").write_text(json.dumps(run_metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps({
        "strategy": args.strategy, "strategy_params": strategy_params,
        "initial_capital": args.initial_capital, "commission_per_contract": args.commission_per_contract,
        "slippage_pct_of_premium": args.slippage_pct_of_premium,
        "min_deployment_pct": args.min_deployment_pct,
        "max_delta_notional_pct": args.max_delta_notional_pct,
        "delever_tolerance_pct": args.delever_tolerance_pct,
        "max_trade_dollar": args.max_trade_dollar,
        "max_trade_pct_of_nav": args.max_trade_pct_of_nav,
        "take_profit_convergence_fraction": args.take_profit_convergence_fraction,
        **engine_settings,
        "real_snapshot_tolerance_days": args.real_snapshot_tolerance_days,
        "start_date": str(engine.calendar[0].date()), "end_date": str(engine.calendar[-1].date()),
        "has_pit_universe": universe_history is not None, "n_real_option_snapshots": len(option_snapshots),
        "benchmark_symbol": args.benchmark_symbol, "benchmark_label": benchmark_label,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Résultats sauvegardés dans %s", out_dir)
    logger.info("--- Résumé ---")
    for key in [
        "total_return_pct", "cagr_pct", "annualized_volatility_pct", "sharpe_ratio", "sortino_ratio",
        "max_drawdown_pct", "calmar_ratio", "num_trades", "win_rate_pct", "profit_factor", "avg_exposure_pct",
        "truncated_orders_count", "truncated_orders_pct", "capped_orders_count",
        "dropped_orders_count", "dropped_orders_by_reason", "avg_cash_pct",
        "avg_delta_notional_pct", "max_delta_notional_pct_observed", "delever_events_count",
        "total_commission_dollar", "total_slippage_dollar",
        "total_friction_dollar", "total_friction_pct_of_initial",
        # Coût de portage (theta) et exposition vega : le seuil que la thèse
        # doit battre, et la part du P&L que ce backtest ne simule pas.
        "total_theta_decay_dollar", "total_theta_decay_pct_of_initial", "avg_vega_notional_pct",
    ]:
        if key in run_metrics:
            logger.info("%s: %s", key, run_metrics[key])
    logger.info("%s", metrics_mod.format_benchmark_summary(run_metrics, benchmark_label))
    if not trades.empty and "exit_reason" in trades.columns:
        logger.info("Répartition des sorties: %s", trades["exit_reason"].value_counts().to_dict())
    if not trades.empty and "open_reason" in trades.columns:
        # Origine des positions sorties : distingue les ouvertures issues du
        # mécanisme journalier (mouvement de cours pur, sans nouveau dépôt)
        # de celles issues d'un dépôt SEC -- cf. la docstring de
        # backtest/options_engine.py.
        logger.info("Positions sorties, par origine d'ouverture: %s", trades["open_reason"].value_counts().to_dict())
    if not positions_history.empty and "source" in positions_history.columns:
        logger.info("Positions par source: %s", positions_history.drop_duplicates("symbol")["source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
