"""
Grid-search MULTI-PARAMÈTRES sur une stratégie ACTIONS (09_backtest.py).

Rejoue le backtest pour chaque combinaison de la grille, sur les MÊMES données
chargées UNE SEULE FOIS -- même structure que les quatre optimiseurs options
(11, 11b, 11c, 11d), dont ce script reprend le pool de process, le partage de
`_DATA` et la séparation apprentissage/test.

CE QUI LE DISTINGUE DES QUATRE AUTRES : il balaie un PRODUIT CARTÉSIEN, pas un
axe. Les optimiseurs options font varier un paramètre à la fois, ce qui suffit
quand les réglages sont séparables. Ici ils ne le sont pas : le stop-loss et la
prise de gain forment un couple (resserrer l'un sans l'autre change la
distribution des sorties, pas seulement sa moyenne), et le seuil d'entrée
déplace le nombre de lignes, donc l'effet du plafond de concentration. Une
descente axe par axe trouverait un optimum de coordonnée, pas un optimum.

LES CINQ AXES, ET CE QU'ILS DÉPLACENT

    stop_loss_pct / take_profit_pct -- les deux seules sorties du moteur
        actions (une position ne se ferme JAMAIS sur simple refermeture de
        l'écart, cf. engine.py). Mesuré sur la configuration d'origine
        (-15%/+30%) : 916 stop-loss à -15,5% de moyenne pour -3,5 M$, contre
        977 prises de gain à +25,0%. Un stop serré sur une stratégie *value*
        vend précisément les titres devenus PLUS décotés -- c'est-à-dire ceux
        dont la thèse s'est renforcée. C'est l'axe où il y a le plus à gagner,
        et le premier à balayer largement.

    entry_threshold_pct -- ne se lit PAS pareil d'une stratégie à l'autre :
        écart au cours pour valuation_gap_dcf (20% par défaut), écart à la
        médiane du SECTEUR pour valuation_gap_sector_neutral (10%). La grille
        par défaut s'adapte donc à la stratégie choisie.

    momentum_min_pct -- filtre "value trap" sur les NOUVELLES entrées
        uniquement. None le désactive.

    rebalance_band_pct -- zone de non-négociation, en points de NAV (cf.
        engine._drift_is_material). Sans elle le portefeuille est repesé 9
        séances sur 10, pour 722% de rotation annualisée.

    max_weight_pct -- plafond de concentration par ligne. Arbitrage de
        diversification pur : moins de plafond, plus de conviction et plus de
        variance.

POURQUOI LE CLASSEMENT SE FAIT SUR LA FENÊTRE D'APPRENTISSAGE SEULE. Une
grille de plusieurs centaines de points classée sur l'historique complet
retient la combinaison qui colle le mieux à CE chemin, et son Sharpe est celui
d'un maximum sur autant de tirages -- un maximum qui n'est pas nul même quand
la vraie performance l'est (cf. backtest/metrics.py, « Sharpe déflaté »). Le
classement porte donc sur `train_sharpe_ratio`, et `test_sharpe_ratio` est
affiché à côté sans jamais entrer dans la sélection : c'est le seul chiffre
produit par des données qui n'ont pas choisi la combinaison.

LA CONTRAINTE DE RENDEMENT, ET POURQUOI ELLE N'EST PAS OPTIONNELLE. Maximiser
un ratio autorise à l'améliorer en désinvestissant : moins de volatilité au
dénominateur, et tant pis pour le numérateur. `--min-cagr-vs-benchmark` écarte
du CLASSEMENT (pas du CSV) toute combinaison dont le CAGR d'apprentissage
tombe sous celui de l'indice de référence sur la MÊME fenêtre. Le portefeuille
doit rester une alternative à l'indice, pas un livret à faible variance.

Usage :
    python 16_optimize_strategie_actions.py
    python 16_optimize_strategie_actions.py --strategy valuation_gap_sector_neutral
    python 16_optimize_strategie_actions.py --stop-loss-grid -15 -25 -40 --take-profit-grid 30 80
    python 16_optimize_strategie_actions.py --split-date 2022-01-01 --workers 4
    python 16_optimize_strategie_actions.py --quick        # grille réduite, pour vérifier le montage

Le CSV complet (une ligne par combinaison, toutes les métriques) est écrit sous
data/backtest/optimize_actions_<stratégie>_<horodatage>.csv.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

import config
from backtest import data_loader, metrics as metrics_mod
from backtest.engine import BacktestEngine
from backtest.strategies import STRATEGY_REGISTRY

logger = logging.getLogger("optimize_actions")

DEFAULT_STRATEGY = "valuation_gap_dcf"

# Sentinelles de désactivation, plutôt que None : le moteur compare
# `move_pct <= stop_loss_pct` et `move_pct >= take_profit_pct`, donc un seuil
# hors d'atteinte suffit à neutraliser la sortie sans rendre le moteur
# conditionnel. -100% est le plancher d'une action (le cours ne passe pas sous
# zéro), +10000% un plafond qu'aucune ligne de l'historique n'approche.
STOP_LOSS_OFF = -100.0
TAKE_PROFIT_OFF = 10_000.0

# Les grilles par défaut CONTIENNENT TOUJOURS la valeur en production : une
# grille qui ne porte pas la configuration actuelle ne dit pas si le changement
# proposé est un gain, seulement lequel de ses concurrents gagne.
DEFAULT_STOP_LOSS_GRID = [-15.0, -25.0, -40.0, STOP_LOSS_OFF]
DEFAULT_TAKE_PROFIT_GRID = [30.0, 60.0, 100.0, TAKE_PROFIT_OFF]
DEFAULT_MOMENTUM_GRID = [None, -10.0, -25.0]
DEFAULT_REBALANCE_BAND_GRID = [0.0, 5.0, 15.0]
DEFAULT_MAX_WEIGHT_GRID = [10.0, 20.0]

# Le seuil d'entrée ne se lit pas pareil d'une stratégie à l'autre (écart au
# cours vs écart à la médiane sectorielle) : sa grille dépend donc de la
# stratégie, contrairement aux quatre autres axes qui sont des réglages de
# MOTEUR et se lisent identiquement partout.
DEFAULT_ENTRY_GRIDS = {
    "valuation_gap_dcf": [15.0, 20.0, 30.0],
    "valuation_gap_sector_neutral": [5.0, 10.0, 20.0],
}

# Rempli une fois par process. Sur Linux, les workers du pool en héritent par
# copy-on-write (fork) sans repasser par le chargement disque. Sur Windows (et
# sur macOS depuis Python 3.8), ProcessPoolExecutor n'utilise PAS fork : chaque
# worker est un interpréteur NEUF qui réimporte ce module de zéro, et repart
# donc avec _DATA = {} -- voir _pool_initializer, qui répare cet écart en
# repeuplant _DATA UNE FOIS par worker via initializer/initargs (cf.
# 11_optimize_options_stops.py, qui a exactement le même besoin).
_DATA: dict = {}


def _pool_initializer(data: dict) -> None:
    """Exécuté une fois par worker à la création du pool. Sur fork, c'est un
    no-op (_DATA déjà hérité) ; sur spawn, c'est le SEUL moyen par lequel un
    worker reçoit jamais ces données -- sans lui, _run_one échoue avec
    `KeyError: 'price_panel'` sur 100% des points de la grille."""
    global _DATA
    _DATA = data


def _load_data(benchmark_symbol: str) -> dict:
    logger.info("Chargement des données (une seule fois pour toute la grille)...")
    daily_prices = data_loader.load_daily_prices()
    price_panel = data_loader.build_price_panel(daily_prices)
    dcf_history = data_loader.load_dcf_history()
    signal_events = data_loader.build_signal_events(dcf_history)
    universe_history = data_loader.load_universe_history()
    fallback_symbols = data_loader.load_current_universe_symbols()
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
        "material_events": material_events,
        "benchmark_prices": benchmark_prices,
        "benchmark_label": benchmark_label,
    }


def _benchmark_cagr_pct(prices: Optional[pd.Series], start: pd.Timestamp, end: pd.Timestamp) -> Optional[float]:
    """CAGR de l'indice de référence sur EXACTEMENT la fenêtre demandée.

    Le plancher de rendement (--min-cagr-vs-benchmark) doit se comparer à
    l'indice sur la MÊME période que la stratégie : le SPY ne fait pas le même
    CAGR sur 2015-2021 (marché porteur) que sur 2022-2026. Un plancher en dur
    comparerait la fenêtre d'apprentissage à la performance d'une autre."""
    if prices is None or prices.empty:
        return None
    window = prices.loc[(prices.index >= start) & (prices.index <= end)].dropna()
    if len(window) < 2 or window.iloc[0] <= 0:
        return None
    years = (window.index[-1] - window.index[0]).days / 365.25
    if years <= 0:
        return None
    return ((window.iloc[-1] / window.iloc[0]) ** (1 / years) - 1) * 100


def _run_one(
    combo: dict,
    strategy_name: str,
    fixed_strategy_params: dict,
    engine_kwargs: dict,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    split_date: Optional[pd.Timestamp],
    n_trials: int = 1,
    garder_courbe: bool = False,
) -> dict:
    """Un run complet pour cette combinaison. Fonction de MODULE (et non
    méthode ni closure) pour rester picklable par ProcessPoolExecutor -- les
    données volumineuses (_DATA) ne sont PAS des arguments : elles viennent du
    fork ou de l'initializer, voir le commentaire sur _DATA plus haut.

    `n_trials` : taille de la grille balayée, transmise à compute_metrics pour
    que chaque ligne porte son Sharpe DÉFLATÉ. Le Sharpe brut du meilleur point
    d'une grille est celui d'un MAXIMUM sur n_trials tirages, et le maximum de
    n tirages n'est pas nul même quand la vraie performance l'est."""
    row = dict(combo)
    row["error"] = None
    try:
        strategy_cls = STRATEGY_REGISTRY[strategy_name]
        strategy = strategy_cls(**{
            **fixed_strategy_params,
            "entry_threshold_pct": combo["entry_threshold_pct"],
            "max_weight_pct": combo["max_weight_pct"],
        })

        engine = BacktestEngine(
            price_panel=_DATA["price_panel"],
            signal_events=_DATA["signal_events"],
            universe_history=_DATA["universe_history"],
            fallback_universe_symbols=_DATA["fallback_symbols"],
            material_events_8k=_DATA["material_events"],
            strategy=strategy,
            stop_loss_pct=combo["stop_loss_pct"],
            take_profit_pct=combo["take_profit_pct"],
            momentum_min_pct=combo["momentum_min_pct"],
            rebalance_band_pct=combo["rebalance_band_pct"],
            start_date=start_date,
            end_date=end_date,
            **engine_kwargs,
        )
        equity_curve, _positions, trades, _signals = engine.run()
        run_metrics = metrics_mod.compute_metrics(
            equity_curve, trades,
            risk_free_rate=config.RISK_FREE_RATE,
            benchmark_prices=_DATA["benchmark_prices"],
            extra=engine.execution_diagnostics(),
            n_trials=n_trials,
        )

        for key in (
            "cagr_pct", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "annualized_volatility_pct", "max_drawdown_pct", "total_return_pct",
            "annualized_turnover_pct", "num_trades",
            "num_positions_closed", "win_rate_positions_pct", "profit_factor_positions",
            "alpha_pct", "beta", "information_ratio",
            "deflated_sharpe_ratio", "sharpe_noise_floor",
            # Un stop très large ou un seuil d'entrée très haut peut
            # « améliorer » un ratio en n'investissant plus : ces deux
            # colonnes sont là pour que ça se voie dans le tableau.
            "avg_exposure_pct", "unfilled_dollar_pct",
            "rebalance_skipped_days_pct",
        ):
            row[key] = run_metrics.get(key)

        if split_date is not None:
            row.update(metrics_mod.split_period_metrics(
                equity_curve, trades, split_date,
                risk_free_rate=config.RISK_FREE_RATE,
                benchmark_prices=_DATA["benchmark_prices"],
            ))
            # CAGR de l'indice sur CHAQUE fenêtre : c'est la référence du
            # plancher de rendement, et elle diffère d'une fenêtre à l'autre.
            dates = pd.DatetimeIndex(equity_curve["date"])
            row["train_benchmark_cagr_pct"] = _benchmark_cagr_pct(
                _DATA["benchmark_prices"], dates[0], split_date)
            row["test_benchmark_cagr_pct"] = _benchmark_cagr_pct(
                _DATA["benchmark_prices"], split_date, dates[-1])

        if not trades.empty and "exit_reason" in trades.columns:
            counts = trades["exit_reason"].value_counts().to_dict()
            for reason in ("stop_loss", "take_profit", "rebalance", "data_gap"):
                row[f"exits_{reason}"] = int(counts.get(reason, 0))

        if garder_courbe:
            # La courbe de NAV permet de RE-DÉCOUPER après coup, en autant de
            # fenêtres qu'on veut, sans relancer un seul backtest : c'est ce
            # qui rend le walk-forward abordable (une passe de grille au lieu
            # d'une par fenêtre). 576 courbes de ~3000 points tiennent dans
            # une quinzaine de mégaoctets.
            row["_nav"] = equity_curve["nav"].to_numpy(dtype=float)
            row["_dates"] = pd.DatetimeIndex(equity_curve["date"])
    except Exception as exc:  # noqa: BLE001 -- une combinaison qui plante ne doit pas tuer la grille
        logger.exception("Échec pour %s", combo)
        row["error"] = str(exc)
    return row


def _build_grid(args, strategy_name: str) -> list[dict]:
    entry_grid = args.entry_threshold_grid or DEFAULT_ENTRY_GRIDS.get(
        strategy_name, DEFAULT_ENTRY_GRIDS[DEFAULT_STRATEGY])
    momentum_grid = (
        [None if m <= -1000 else m for m in args.momentum_grid]
        if args.momentum_grid else DEFAULT_MOMENTUM_GRID
    )
    axes = [
        args.stop_loss_grid or DEFAULT_STOP_LOSS_GRID,
        args.take_profit_grid or DEFAULT_TAKE_PROFIT_GRID,
        entry_grid,
        momentum_grid,
        args.rebalance_band_grid or DEFAULT_REBALANCE_BAND_GRID,
        args.max_weight_grid or DEFAULT_MAX_WEIGHT_GRID,
    ]
    if args.quick:
        # Montage vérifiable en une minute : on ne garde que les bornes de
        # chaque axe. Sert à valider la plomberie, jamais à conclure.
        axes = [axe[:: max(len(axe) - 1, 1)] for axe in axes]
    noms = ("stop_loss_pct", "take_profit_pct", "entry_threshold_pct",
            "momentum_min_pct", "rebalance_band_pct", "max_weight_pct")
    return [dict(zip(noms, valeurs)) for valeurs in itertools.product(*axes)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, choices=sorted(STRATEGY_REGISTRY))
    parser.add_argument("--start-date", type=str, default="2015-01-01")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument(
        "--split-date", type=str, default="2022-01-01",
        help="Date qui sépare la fenêtre d'APPRENTISSAGE (classement) de la fenêtre de TEST "
             "(jugement). Défaut: %(default)s. Voir --no-split pour classer in-sample.",
    )
    parser.add_argument(
        "--no-split", dest="split_date", action="store_const", const=None,
        help="Classe sur l'historique COMPLET (in-sample). Le résultat est alors un optimum "
             "in-sample, à ne pas lire comme une performance attendue.",
    )
    parser.add_argument(
        "--min-cagr-vs-benchmark", type=float, default=0.0,
        help="Écart de CAGR minimal vs l'indice, en points, sur la fenêtre d'APPRENTISSAGE, "
             "pour qu'une combinaison entre au classement (défaut: %(default)s, soit "
             "« au moins l'indice »). Maximiser un ratio autorise sinon à l'améliorer en "
             "désinvestissant. Passe une valeur très négative pour lever la contrainte.",
    )
    parser.add_argument(
        "--min-positions", type=int, default=50,
        help="Combinaisons ayant fermé moins de thèses que ça, écartées du CLASSEMENT (pas du "
             "CSV, défaut: %(default)s). Un réglage très sélectif finit toujours par n'ouvrir "
             "que quelques positions chanceuses : sans ce plancher, il gagnerait sans rien prouver.",
    )
    parser.add_argument(
        "--plateau-tolerance", type=float, default=0.05,
        help="Écart de Sharpe d'apprentissage en deçà duquel deux combinaisons sont tenues pour "
             "indiscernables (défaut: %(default)s). Le meilleur point est alors celui qui NÉGOCIE "
             "LE MOINS parmi elles, pas celui dont l'estimation est la plus haute -- l'erreur-type "
             "d'un Sharpe sur sept ans dépasse 0,45. 0 rétablit l'argmax strict.",
    )
    parser.add_argument(
        "--n-trials-prior", type=int, default=0,
        help="Nombre de combinaisons DÉJÀ essayées avant cette grille, dans les études "
             "précédentes. Ajouté à la taille de la grille pour le Sharpe déflaté. Le "
             "surapprentissage se compte sur le PROGRAMME DE RECHERCHE entier, pas sur la "
             "dernière grille : repartir de zéro à chaque lancement revient à effacer le "
             "compteur juste avant de le lire.",
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Évalue en plus la performance hors échantillon CONCATÉNÉE sur fenêtres "
             "glissantes : chaque fenêtre choisit sa combinaison sur son seul passé. Une "
             "coupure unique ne laisse que 4,7 ans hors échantillon, trop peu pour trancher.",
    )
    parser.add_argument("--wf-annees-apprentissage", type=float, default=5.0)
    parser.add_argument("--wf-annees-test", type=float, default=1.0)
    parser.add_argument("--stop-loss-grid", type=float, nargs="+", default=None)
    parser.add_argument("--take-profit-grid", type=float, nargs="+", default=None)
    parser.add_argument("--entry-threshold-grid", type=float, nargs="+", default=None)
    parser.add_argument(
        "--momentum-grid", type=float, nargs="+", default=None,
        help="Seuils de momentum. Une valeur <= -1000 vaut « filtre désactivé ».",
    )
    parser.add_argument("--rebalance-band-grid", type=float, nargs="+", default=None)
    parser.add_argument("--max-weight-grid", type=float, nargs="+", default=None)
    parser.add_argument("--commission-bps", type=float, default=config.BACKTEST_COMMISSION_BPS)
    parser.add_argument("--slippage-bps", type=float, default=config.BACKTEST_SLIPPAGE_BPS)
    parser.add_argument("--initial-capital", type=float, default=config.BACKTEST_INITIAL_CAPITAL)
    parser.add_argument("--benchmark-symbol", default=config.BENCHMARK_SYMBOL)
    parser.add_argument("--workers", type=int, default=1, help="Process en parallèle (fork).")
    parser.add_argument("--quick", action="store_true", help="Grille réduite aux bornes, pour vérifier le montage.")
    parser.add_argument("--output", default=None, help="Chemin du CSV (défaut: horodaté sous data/backtest/).")
    parser.add_argument(
        "--report-only", default=None, metavar="CSV",
        help="Relit un CSV déjà produit et réaffiche le classement, sans relancer un seul "
             "backtest. Sert à changer de garde-fou (--min-positions, --min-cagr-vs-benchmark) "
             "ou de critère sans repayer l'heure de calcul de la grille.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.report_only:
        results = pd.read_csv(args.report_only)
        # Le CSV rend les None de momentum_min_pct sous forme de cases vides :
        # relus par pandas, ce sont des NaN, et « filtre désactivé » doit
        # rester lisible comme tel dans le tableau.
        _report(results, args, pd.Timestamp(args.split_date) if args.split_date else None)
        return
    # Le moteur avertit à chaque run sur la couverture de l'univers et le
    # sous-investissement : pertinent pour UN backtest, illisible répété des
    # centaines de fois. Les deux chiffres restent dans le CSV.
    logging.getLogger("backtest.engine").setLevel(logging.ERROR)

    grid = _build_grid(args, args.strategy)
    logger.info("Grille : %d combinaisons, stratégie '%s'.", len(grid), args.strategy)

    global _DATA
    _DATA = _load_data(args.benchmark_symbol)

    start_date = pd.Timestamp(args.start_date) if args.start_date else None
    end_date = pd.Timestamp(args.end_date) if args.end_date else None
    split_date = pd.Timestamp(args.split_date) if args.split_date else None

    engine_kwargs = {
        "initial_capital": args.initial_capital,
        "cost_bps": args.commission_bps + args.slippage_bps,
    }
    fixed_strategy_params: dict = {}

    # Le surapprentissage se compte sur le PROGRAMME entier (cf.
    # --n-trials-prior) : la grille du jour n'en est qu'une tranche.
    n_trials = len(grid) + max(args.n_trials_prior, 0)
    if args.n_trials_prior:
        logger.info(
            "Sharpe deflate calcule sur %d essais : %d dans cette grille, %d anterieurs.",
            n_trials, len(grid), args.n_trials_prior)

    rows: list[dict] = []
    if args.workers > 1:
        # initializer/initargs : cf. le commentaire sur _pool_initializer.
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_pool_initializer, initargs=(_DATA,),
        ) as pool:
            futures = {
                pool.submit(
                    _run_one, combo, args.strategy, fixed_strategy_params, engine_kwargs,
                    start_date, end_date, split_date, n_trials, args.walk_forward,
                ): combo
                for combo in grid
            }
            for i, future in enumerate(as_completed(futures), 1):
                rows.append(future.result())
                if i % 10 == 0 or i == len(grid):
                    logger.info("  %d/%d combinaisons", i, len(grid))
    else:
        for i, combo in enumerate(grid, 1):
            rows.append(_run_one(
                combo, args.strategy, fixed_strategy_params, engine_kwargs,
                start_date, end_date, split_date, n_trials, args.walk_forward,
            ))
            if i % 10 == 0 or i == len(grid):
                logger.info("  %d/%d combinaisons", i, len(grid))

    # Les courbes de NAV servent au walk-forward et n'ont rien à faire dans le
    # CSV (une colonne de 3000 nombres par ligne le rendrait illisible et
    # énorme) : elles restent en mémoire, sous des clés préfixées d'un
    # underscore, et sont retirées avant écriture.
    resultat_wf = walk_forward(rows, args, args.plateau_tolerance) if args.walk_forward else None

    results = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    output = args.output or (
        config.DIR_BACKTEST / f"optimize_actions_{args.strategy}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    config.DIR_BACKTEST.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    logger.info("Grille complète écrite dans %s", output)

    _report(results, args, split_date)

    if resultat_wf:
        logger.info(
            "--- Walk-forward : %d fenêtres, %d séances hors échantillon ---\n"
            "Sharpe HORS ÉCHANTILLON concaténé : %.3f\n"
            "Chaque fenêtre a choisi sa combinaison sur son seul passé ; aucune portion de "
            "cette courbe n'a servi à choisir le réglage qui la produit.",
            resultat_wf["n_fenetres"], resultat_wf["n_jours"],
            resultat_wf["sharpe_hors_echantillon"],
        )
        logger.info(
            "Combinaison retenue par fenêtre :\n%s",
            pd.DataFrame(resultat_wf["choix"]).to_string(index=False),
        )


def fenetres_walk_forward(
    debut: pd.Timestamp, fin: pd.Timestamp,
    annees_apprentissage: float = 5.0, annees_test: float = 1.0,
) -> list[tuple]:
    """Fenêtres glissantes (début_apprentissage, coupure, fin_test).

    POURQUOI, À CÔTÉ DE LA COUPURE UNIQUE. Une seule coupure 2015/2022 ne
    laisse que 4,7 ans hors échantillon, sur lesquels l'erreur-type d'un Sharpe
    dépasse 0,5 : c'est trop court pour trancher, et c'est ce qui rendait
    l'écart mesuré (+0,08) non significatif à lui seul. Le walk-forward
    réutilise CHAQUE année comme fenêtre de test à son tour, après un
    apprentissage qui ne voit que son passé. La performance hors échantillon
    est alors la concaténation de toutes ces années -- toute la période moins
    le premier apprentissage, au lieu d'un tiers.

    Ce n'est pas une astuce pour obtenir plus : c'est la même exigence
    appliquée plus souvent. Un réglage qui ne tient que sur une fenêtre
    particulière apparaît ici pour ce qu'il est."""
    fenetres = []
    coupure = debut + pd.DateOffset(years=int(annees_apprentissage))
    pas = pd.DateOffset(years=int(annees_test)) if annees_test >= 1 else pd.DateOffset(months=int(annees_test * 12))
    while coupure < fin:
        fin_test = min(coupure + pas, fin)
        if (fin_test - coupure).days < 60:
            break
        fenetres.append((debut, coupure, fin_test))
        coupure = fin_test
    return fenetres


def _erreur_type_sharpe(sharpe: float, annees: float) -> float:
    """Erreur-type d'un Sharpe annualisé estimé sur `annees` années
    (Lo, 2002) : `sqrt((1 + SR²/2) / n)`, n en années.

    Elle répond à la seule question qui permette de lire un classement de
    grille : l'écart entre le premier et le dixième est-il une différence, ou
    du bruit d'estimation ? Sur 7 ans et un Sharpe de 1, elle vaut 0,46 --
    c'est-à-dire que presque toute la grille est indiscernable de son
    maximum."""
    if annees <= 0:
        return float("nan")
    return ((1 + 0.5 * sharpe * sharpe) / annees) ** 0.5


def _choisir_sur_le_plateau(
    ranked: pd.DataFrame, rank_key: str, tolerance: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Meilleur point = le moins coûteux en ROTATION parmi ceux que la fenêtre
    d'apprentissage ne sait pas départager.

    POURQUOI PAS L'ARGMAX. Le Sharpe d'une combinaison est une ESTIMATION, et
    son erreur-type sur sept ans dépasse 0,45 (cf. _erreur_type_sharpe).
    Retenir le maximum d'une grille de plusieurs centaines de points revient
    alors à retenir le tirage le plus chanceux d'un ensemble statistiquement
    homogène : le premier du classement n'est pas meilleur que le dixième, il a
    juste mieux collé à ce chemin-là.

    LE DÉPARTAGE NE REGARDE PAS LA FENÊTRE DE TEST -- ce serait la consommer, et
    elle ne vaut que tant qu'elle n'a rien choisi. Il porte sur la ROTATION, qui
    n'est pas une mesure de performance mais d'EXPOSITION À UNE HYPOTHÈSE : tout
    le backtest suppose 10 bps par aller simple. À 720% de rotation annuelle,
    se tromper de 20 bps coûte 1,4 point de CAGR par an ; à 360%, 0,7. À
    performance d'apprentissage indiscernable, la combinaison qui négocie moins
    est celle dont le résultat dépend le moins d'un chiffre qu'on a supposé.

    `tolerance` à 0 rétablit l'argmax strict."""
    if tolerance <= 0 or ranked.empty:
        return ranked.iloc[0], ranked.head(1)
    plafond = float(ranked.iloc[0][rank_key])
    plateau = ranked[ranked[rank_key] >= plafond - tolerance]
    if "annualized_turnover_pct" not in plateau.columns or plateau["annualized_turnover_pct"].isna().all():
        return plateau.iloc[0], plateau
    return plateau.sort_values("annualized_turnover_pct").iloc[0], plateau


def _lisible(cle: str, valeur) -> str:
    """Les trois façons dont un axe peut valoir « désactivé » ne se lisent pas
    d'elles-mêmes : un momentum à NaN (le filtre est absent, pas indéfini), un
    stop à -100% et une prise de gain à +10000% sont des sentinelles, pas des
    seuils qu'on pourrait atteindre. Les afficher bruts laisserait croire à un
    réglage extrême là où il n'y a tout simplement plus de règle."""
    if cle == "momentum_min_pct" and (valeur is None or pd.isna(valeur)):
        return "désactivé (aucun filtre momentum)"
    if cle == "stop_loss_pct" and valeur is not None and valeur <= STOP_LOSS_OFF:
        return f"{valeur} -> désactivé (hors d'atteinte)"
    if cle == "take_profit_pct" and valeur is not None and valeur >= TAKE_PROFIT_OFF:
        return f"{valeur} -> désactivé (hors d'atteinte)"
    if cle == "rebalance_band_pct" and valeur == 0:
        return "0 (aucune zone de non-négociation)"
    return str(valeur)


AXES = ("stop_loss_pct", "take_profit_pct", "entry_threshold_pct",
        "momentum_min_pct", "rebalance_band_pct", "max_weight_pct")


def _lire_le_plateau(
    ranked: pd.DataFrame, plateau: pd.DataFrame, rank_key: str, tolerance: float,
) -> None:
    """Ce que la grille établit vraiment, par opposition à ce qu'elle classe.

    Un axe sur lequel TOUTES les combinaisons du plateau s'accordent est un
    résultat : la fenêtre d'apprentissage ne sait pas départager le reste, mais
    elle exclut l'autre valeur de cet axe-là. Un axe où le plateau reste partagé
    ne conclut rien, et le dire évite de présenter comme un réglage optimisé ce
    qui n'est qu'une valeur tirée au sort parmi des équivalentes."""
    if tolerance <= 0 or plateau.empty:
        return
    meilleur = float(ranked.iloc[0][rank_key])
    annees = 7.0
    logger.info(
        "--- Ce que la grille établit ---\n"
        "Erreur-type d'un Sharpe estimé sur ~%.0f ans : %.2f. Sur %d combinaisons, %d sont à "
        "moins d'une erreur-type du maximum : le classement ne les départage donc PAS.\n"
        "Plateau retenu (à %.2f du maximum) : %d combinaisons.",
        annees, _erreur_type_sharpe(meilleur, annees), len(ranked),
        int((ranked[rank_key] >= meilleur - _erreur_type_sharpe(meilleur, annees)).sum()),
        tolerance, len(plateau),
    )
    for axe in AXES:
        if axe not in plateau.columns:
            continue
        # Les NaN de momentum_min_pct sont une VALEUR (« filtre désactivé »),
        # pas une absence : dropna=False, sans quoi un plateau unanimement sans
        # filtre passerait pour un axe vide.
        valeurs = plateau[axe].value_counts(dropna=False)
        if len(valeurs) == 1:
            logger.info("  %-20s UNANIME : %s", axe, _lisible(axe, plateau[axe].iloc[0]))
        else:
            retenues = ", ".join(_lisible(axe, v) for v in sorted(valeurs.index, key=lambda x: (pd.isna(x), x)))
            logger.info("  %-20s indécis : %s", axe, retenues)


def _sharpe_fenetre(nav: "np.ndarray", dates: pd.DatetimeIndex, debut, fin) -> float:
    """Sharpe annualisé sur ]debut, fin], calculé depuis la courbe de NAV."""
    masque = (dates > pd.Timestamp(debut)) & (dates <= pd.Timestamp(fin))
    valeurs = nav[masque]
    if len(valeurs) < 20:
        return float("nan")
    rendements = np.diff(valeurs) / valeurs[:-1]
    ecart_type = rendements.std()
    if not ecart_type > 0:
        return float("nan")
    return float(rendements.mean() / ecart_type * math.sqrt(metrics_mod.TRADING_DAYS_PER_YEAR))


def walk_forward(rows: list[dict], args, tolerance: float) -> Optional[dict]:
    """Performance hors échantillon CONCATÉNÉE sur des fenêtres glissantes.

    Pour chaque fenêtre, la combinaison est choisie sur le SEUL apprentissage
    de cette fenêtre-là (même règle de plateau que le reste du script), puis
    ses rendements de la fenêtre de test sont conservés. Mis bout à bout, ces
    segments forment une courbe qu'aucune de ses portions n'a servi à choisir.

    Le départage sur le plateau reste la ROTATION GLOBALE de la combinaison,
    faute de rotation par fenêtre dans le tableau. C'est une approximation
    assumée : la rotation est une propriété assez stable d'un réglage, et elle
    ne sert qu'à départager des combinaisons déjà jugées indiscernables."""
    utilisables = [r for r in rows if r.get("error") is None and r.get("_nav") is not None]
    if not utilisables:
        return None

    dates = utilisables[0]["_dates"]
    fenetres = fenetres_walk_forward(
        dates[0], dates[-1], args.wf_annees_apprentissage, args.wf_annees_test)
    if not fenetres:
        return None

    segments, choix = [], []
    for debut, coupure, fin_test in fenetres:
        classees = []
        for r in utilisables:
            sharpe_train = _sharpe_fenetre(r["_nav"], r["_dates"], debut, coupure)
            if not math.isnan(sharpe_train):
                classees.append((sharpe_train, r))
        if not classees:
            continue
        classees.sort(key=lambda t: -t[0])
        plafond = classees[0][0]
        plateau = [r for s, r in classees if s >= plafond - tolerance]
        retenue = min(
            plateau,
            key=lambda r: r.get("annualized_turnover_pct") or float("inf"),
        )

        masque = (retenue["_dates"] > coupure) & (retenue["_dates"] <= fin_test)
        valeurs = retenue["_nav"][masque]
        if len(valeurs) >= 2:
            segments.append(np.diff(valeurs) / valeurs[:-1])
        choix.append({
            "coupure": str(pd.Timestamp(coupure).date()),
            "fin_test": str(pd.Timestamp(fin_test).date()),
            **{axe: retenue.get(axe) for axe in AXES},
        })

    if not segments:
        return None
    rendements = np.concatenate(segments)
    ecart_type = rendements.std()
    return {
        "sharpe_hors_echantillon": (
            float(rendements.mean() / ecart_type * math.sqrt(metrics_mod.TRADING_DAYS_PER_YEAR))
            if ecart_type > 0 else float("nan")
        ),
        "n_fenetres": len(choix),
        "n_jours": int(len(rendements)),
        "choix": choix,
    }


def _reference_combo(strategy_name: str) -> dict:
    """La configuration EN PRODUCTION, celle que la grille doit contenir pour
    que le classement dise si le changement proposé est un gain -- et non
    seulement lequel de ses concurrents gagne."""
    entry = (
        config.BACKTEST_SECTOR_NEUTRAL_ENTRY_THRESHOLD_PCT
        if strategy_name == "valuation_gap_sector_neutral"
        else config.BACKTEST_ENTRY_THRESHOLD_PCT
    )
    return {
        "stop_loss_pct": config.BACKTEST_STOP_LOSS_PCT,
        "take_profit_pct": config.BACKTEST_TAKE_PROFIT_PCT,
        "entry_threshold_pct": entry,
        "momentum_min_pct": config.BACKTEST_STOCKS_MOMENTUM_MIN_PCT,
        "rebalance_band_pct": config.BACKTEST_REBALANCE_BAND_PCT,
        "max_weight_pct": config.BACKTEST_MAX_WEIGHT_PER_POSITION_PCT,
    }


def _compare_a_la_reference(
    ok: pd.DataFrame, best: pd.Series, strategy_name: str,
    rank_key: str, split_date: Optional[pd.Timestamp],
) -> None:
    """Meilleur point CONTRE configuration actuelle, sur les mêmes fenêtres.

    Sans cette comparaison, un classement dit seulement quelle combinaison est
    en tête ; il ne dit pas si elle vaut mieux que ce qui tourne déjà. Et c'est
    la comparaison des fenêtres de TEST qui tranche : celle d'apprentissage a
    choisi le gagnant, elle ne peut pas l'arbitrer."""
    reference = _reference_combo(strategy_name)
    masque = pd.Series(True, index=ok.index)
    for cle, valeur in reference.items():
        if cle not in ok.columns:
            return
        colonne = ok[cle]
        # momentum_min_pct = None se relit en NaN depuis le CSV : comparer par
        # égalité renverrait False partout et la ligne de référence resterait
        # introuvable, sans que rien ne le signale.
        masque &= colonne.isna() if valeur is None else (colonne - valeur).abs() < 1e-9
    lignes = ok[masque]
    if lignes.empty:
        logger.warning(
            "La configuration actuelle (%s) n'est pas dans la grille : le classement ne dit "
            "donc pas si le meilleur point est un GAIN, seulement qu'il est en tête.", reference,
        )
        return

    ref = lignes.iloc[0]
    colonnes = [rank_key, "test_sharpe_ratio", "train_cagr_pct", "test_cagr_pct"] if split_date is not None \
        else ["sharpe_ratio", "cagr_pct"]
    colonnes += ["annualized_turnover_pct", "max_drawdown_pct", "num_positions_closed"]
    colonnes = [c for c in colonnes if c in ok.columns]

    tableau = pd.DataFrame(
        [ref[colonnes], best[colonnes]], index=["configuration actuelle", "meilleur point"],
    )
    logger.info("--- Meilleur point contre configuration actuelle ---\n%s", tableau.to_string())
    if split_date is not None and pd.notna(ref.get("test_sharpe_ratio")) and pd.notna(best.get("test_sharpe_ratio")):
        ecart = best["test_sharpe_ratio"] - ref["test_sharpe_ratio"]
        logger.info(
            "Écart de Sharpe HORS ÉCHANTILLON : %+.3f. C'est le seul chiffre de cette "
            "comparaison que la sélection n'a pas pu fabriquer.", ecart,
        )


def _report(results: pd.DataFrame, args, split_date: Optional[pd.Timestamp]) -> None:
    """Classement, garde-fous, et lecture du meilleur point."""
    ok = results[results["error"].isna()]
    if ok.empty:
        logger.error("Toutes les combinaisons ont échoué -- voir le CSV.")
        sys.exit(1)

    rank_key = "train_sharpe_ratio" if split_date is not None and "train_sharpe_ratio" in ok.columns else "sharpe_ratio"
    cagr_key = "train_cagr_pct" if rank_key.startswith("train_") else "cagr_pct"
    bench_key = "train_benchmark_cagr_pct" if rank_key.startswith("train_") else None

    eligible = ok[ok[rank_key].notna()]
    ecarte_positions = len(eligible)
    if "num_positions_closed" in eligible.columns:
        eligible = eligible[eligible["num_positions_closed"].fillna(0) >= args.min_positions]
    ecarte_positions -= len(eligible)

    ecarte_cagr = len(eligible)
    if bench_key and bench_key in eligible.columns:
        plancher = eligible[bench_key] + args.min_cagr_vs_benchmark
        eligible = eligible[eligible[cagr_key] >= plancher]
    ecarte_cagr -= len(eligible)

    if eligible.empty:
        logger.error(
            "Aucune combinaison ne passe les garde-fous (%d écartées faute de thèses, "
            "%d faute de rendement). Le CSV porte la grille entière.",
            ecarte_positions, ecarte_cagr,
        )
        sys.exit(1)

    ranked = eligible.sort_values(rank_key, ascending=False)
    best, plateau = _choisir_sur_le_plateau(ranked, rank_key, args.plateau_tolerance)

    logger.info(
        "%d combinaisons retenues sur %d (%d écartées faute de thèses, %d faute de rendement).",
        len(eligible), len(ok), ecarte_positions, ecarte_cagr,
    )
    colonnes = [
        "stop_loss_pct", "take_profit_pct", "entry_threshold_pct", "momentum_min_pct",
        "rebalance_band_pct", "max_weight_pct", rank_key,
    ]
    if split_date is not None:
        colonnes += ["test_sharpe_ratio", "train_cagr_pct", "test_cagr_pct"]
    else:
        colonnes += ["cagr_pct"]
    colonnes += ["annualized_turnover_pct", "max_drawdown_pct", "avg_exposure_pct"]
    colonnes = [c for c in colonnes if c in ranked.columns]
    logger.info("--- Dix meilleures combinaisons ---\n%s",
                ranked[colonnes].head(10).to_string(index=False))

    _lire_le_plateau(ranked, plateau, rank_key, args.plateau_tolerance)

    _compare_a_la_reference(ok, best, args.strategy, rank_key, split_date)

    logger.info("--- Meilleur point ---")
    for key in ("stop_loss_pct", "take_profit_pct", "entry_threshold_pct", "momentum_min_pct",
                "rebalance_band_pct", "max_weight_pct"):
        logger.info("  %s = %s", key, _lisible(key, best.get(key)))
    if split_date is not None and pd.notna(best.get("test_sharpe_ratio")):
        logger.info(
            "Sharpe apprentissage %.3f -> test %.3f (coupure %s). C'est le SECOND chiffre qui "
            "dit si l'optimum survit à des données qui ne l'ont pas choisi.",
            best.get(rank_key), best.get("test_sharpe_ratio"), split_date.date(),
        )
    logger.info(
        "Sharpe plein échantillon %.3f, plancher de bruit %.3f pour %d essais, Sharpe déflaté %.3f. "
        "Un Sharpe sous son plancher de bruit n'est pas distinguable de la sélection elle-même.",
        best.get("sharpe_ratio") or float("nan"),
        best.get("sharpe_noise_floor") or float("nan"),
        len(results),
        best.get("deflated_sharpe_ratio") or float("nan"),
    )


if __name__ == "__main__":
    main()
