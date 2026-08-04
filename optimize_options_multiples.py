"""
Recherche aléatoire des hyperparamètres de la stratégie options
'valuation_gap_multiples_options' (backtest/strategies/valuation_gap_multiples_options.py),
sur le moteur backtest/options_engine.py.

Portée VOLONTAIREMENT restreinte à 3 paramètres (SEARCH_SPACE ci-dessous) :
entry_threshold_pct, stop_loss_pct, take_profit_pct -- pour en isoler
l'effet sans le bruit des 14 autres réglages possibles. Tous les autres
(FIXED_PARAMS) sont FIGÉS à la valeur actuelle de config.py, à UNE exception
près : min_deployment_pct est figé à 0 (désactivé), pas à sa valeur de
config.py (90). Il dominait le classement précédent (corrélation -0,88 au
Calmar sur un sweep à 17 paramètres) au point de masquer l'effet de tout le
reste -- le laisser à 90 aurait rendu ce sweep ciblé inexploitable, chaque
essai restant dominé par ce seul réglage plutôt que par les 3 qu'on regarde
ici. Reviens dessus séparément une fois entry/stop/take-profit calibrés.

Les paramètres qui façonnent le SIGNAL en amont (MIN_PEERS_PER_SECTOR_YEAR,
MULTIPLE_PLAUSIBLE_RANGE, SECTOR_MULTIPLES dans
06b_calcul_valorisation_combinee.py) restent hors périmètre : les faire
varier exigerait de régénérer valorisation_combinee_historique.parquet à
chaque essai (minutes, pas secondes).

Méthode : ÉCHANTILLONNAGE ALÉATOIRE, pas un quadrillage complet -- même à 3
paramètres, un tirage aléatoire couvre l'espace au moins aussi bien qu'un
quadrillage régulier pour un budget d'essais donné, et le code reste
identique si tu élargis SEARCH_SPACE plus tard. Un tirage aléatoire couvre
l'espace de recherche de façon représentative avec un nombre de runs
raisonnable (--trials, 300 par défaut).

Classement : RATIO DE CALMAR (CAGR / |drawdown max|) par défaut -- mesure
directe du compromis rendement/risque demandé. --objectif permet de changer
pour sharpe_ratio, total_return_pct, etc. Les runs à moins de
--min-trades (15 par défaut) sont écartés du classement : trop peu de trades
pour que la mesure soit autre chose que du bruit.

ATTENTION -- ceci est de la RECHERCHE D'HYPERPARAMÈTRES SUR DONNÉES
HISTORIQUES : le risque de surapprentissage (overfitting) au passé est réel,
d'autant plus que l'espace exploré est grand. Les meilleurs résultats du
classement ne sont pas une prédiction de performance future -- vérifie que
la configuration retenue reste plausible économiquement (cf. les bornes ci-
dessous, choisies pour rester dans un voisinage défendable des valeurs
actuelles), et regarde comment le score se dégrade autour de l'optimum
(--sensibilite) plutôt que le seul meilleur point : un optimum isolé au
milieu d'un plateau médiocre est un signe de surapprentissage, un optimum
au centre d'une zone stable est plus digne de confiance.

Usage :
    python optimize_options_multiples.py
    python optimize_options_multiples.py --trials 1000 --seed 42
    python optimize_options_multiples.py --objectif sharpe_ratio --min-trades 25
    python optimize_options_multiples.py --start-date 2015-01-01 --workers 4
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from backtest import data_loader, metrics as metrics_mod
from backtest.options_engine import OptionsBacktestEngine
from backtest.strategies.valuation_gap_multiples_options import ValuationGapMultiplesOptionsStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("optimize_options_multiples")


# ----------------------------------------------------------------------------
# Espace de recherche -- SEULS ces paramètres varient d'un essai à l'autre.
# ----------------------------------------------------------------------------
# (nom du paramètre, borne basse, borne haute, "int"/"float"/liste de choix).
# Bornes choisies comme un voisinage DÉFENDABLE de la valeur actuelle (cf.
# config.py) -- pas un intervalle arbitrairement large, qui favoriserait le
# surapprentissage en explorant des réglages économiquement incohérents
# (un stop-loss à -90% ou un take-profit à 2% n'ont pas de sens réel).
SEARCH_SPACE: dict[str, tuple] = {
    "entry_threshold_pct":  (10.0, 50.0, "float"),
    "stop_loss_pct":        (-45.0, -10.0, "float"),   # négatif : mouvement du sous-jacent
    "take_profit_pct":      (15.0, 60.0, "float"),
}

# ----------------------------------------------------------------------------
# Paramètres FIGÉS -- valeur actuelle de config.py pour tous, à UNE exception
# près : min_deployment_pct est figé à 0 (désactivé), pas à sa valeur de
# config.py (90). Sur le sweep précédent à 17 paramètres, il dominait le
# classement (corrélation -0,88 au Calmar) au point de masquer l'effet de
# tout le reste -- le laisser actif ici rendrait ce sweep ciblé inexploitable.
# Reviens dessus séparément une fois entry/stop/take-profit calibrés.
# ----------------------------------------------------------------------------
FIXED_PARAMS: dict = {
    "weight_cap_pct": config.OPTIONS_MULTIPLES_WEIGHT_CAP_PCT,
    "gap_basis": config.OPTIONS_MULTIPLES_GAP_BASIS,
    "max_positions": config.OPTIONS_MAX_POSITIONS,
    "min_positions": config.BACKTEST_MIN_POSITIONS,
    "tenor_days": config.OPTIONS_MULTIPLES_TENOR_DAYS,
    "roll_when_days_left": config.OPTIONS_MULTIPLES_ROLL_WHEN_DAYS_LEFT,
    "min_deployment_pct": 0.0,  # cf. note ci-dessus -- PAS config.OPTIONS_MIN_DEPLOYMENT_PCT
    "min_resize_relative_pct": config.OPTIONS_MIN_RESIZE_RELATIVE_PCT,
    "momentum_min_pct": config.BACKTEST_MOMENTUM_MIN_PCT,
    "slippage_pct_of_premium": config.OPTIONS_SLIPPAGE_PCT_OF_PREMIUM,
    "fee_bump_max_extra_pct": config.OPTIONS_FEE_BUMP_MAX_EXTRA_PCT,
    "max_fee_pct_of_trade": config.OPTIONS_MAX_FEE_PCT_OF_TRADE,
    "realized_vol_lookback_days": config.OPTIONS_REALIZED_VOL_LOOKBACK_DAYS,
    "inflation_adjust": config.INFLATION_ADJUST_GAP,
}

# Colonnes de metrics.compute_metrics affichées dans le classement.
REPORT_COLUMNS = [
    "calmar_ratio", "sharpe_ratio", "total_return_pct", "cagr_pct",
    "max_drawdown_pct", "num_trades", "win_rate_pct", "profit_factor",
]


def sample_params(rng: np.random.Generator) -> dict:
    """Un tirage sur SEARCH_SPACE, complété par FIXED_PARAMS, respectant les
    contraintes de cohérence entre paramètres -- un tirage indépendant par
    paramètre en violerait certaines si SEARCH_SPACE est élargi plus tard
    (ex: roulement à 400 jours de l'échéance sur un contrat de 365 jours
    n'a pas de sens). Avec seulement 3 paramètres sans contrainte entre eux,
    la boucle de rejet réussit systématiquement au premier tirage ici -- elle
    reste générique pour un SEARCH_SPACE élargi sans rien changer au reste
    du script."""
    for _ in range(200):  # rejet-échantillonnage : quelques tentatives suffisent
        sampled = {}
        for name, (lo, hi, kind) in SEARCH_SPACE.items():
            if kind == "choice":
                sampled[name] = lo[rng.integers(0, len(lo))]
            elif kind == "int":
                sampled[name] = int(rng.integers(lo, hi + 1))
            else:
                sampled[name] = float(rng.uniform(lo, hi))
        params = {**FIXED_PARAMS, **sampled}

        if params["min_positions"] > params["max_positions"]:
            continue
        # Le roulement doit laisser au moins 90 jours de vie au nouveau
        # contrat, sinon la position roule en boucle sans jamais capter
        # l'échéance visée.
        if params["tenor_days"] - params["roll_when_days_left"] < 90:
            continue
        return params
    raise RuntimeError("Impossible de tirer une combinaison valide après 200 tentatives : resserre SEARCH_SPACE.")


@dataclass
class LoadedData:
    price_panel: object
    signal_events: pd.DataFrame
    universe_history: Optional[pd.DataFrame]
    fallback_symbols: set
    option_snapshots: pd.DataFrame
    material_events: Optional[pd.DataFrame]


def load_data() -> LoadedData:
    """Chargé UNE SEULE FOIS pour tous les essais -- c'est le coût fixe du
    sweep, pas quelque chose à refaire à chaque tirage."""
    logger.info("Chargement des données (une seule fois pour tous les essais)...")
    daily_prices = data_loader.load_daily_prices()
    price_panel = data_loader.build_price_panel(daily_prices)
    valorisation_combinee = data_loader.load_valorisation_combinee_history()
    signal_events = data_loader.build_options_signal_events(valorisation_combinee)
    return LoadedData(
        price_panel=price_panel,
        signal_events=signal_events,
        universe_history=data_loader.load_universe_history(),
        fallback_symbols=data_loader.load_current_universe_symbols(),
        option_snapshots=data_loader.load_option_snapshots_history(),
        material_events=data_loader.load_material_events_8k(),
    )


def run_trial(
    params: dict, data: LoadedData, initial_capital: float,
    start_date: Optional[pd.Timestamp], end_date: Optional[pd.Timestamp],
) -> Optional[dict]:
    """Un essai : construit la stratégie et le moteur avec `params`, lance le
    backtest, retourne {params..., metrics...}. None si l'essai n'a produit
    aucun résultat exploitable (ex: aucun jour de bourse dans la plage)."""
    # inflation_adjust est un réglage GLOBAL (config.INFLATION_ADJUST_GAP),
    # lu par la stratégie à chaque appel -- pas un paramètre de constructeur.
    # Cf. backtest/strategies/base.inflation_adjusted_gap.
    previous_inflation_setting = config.INFLATION_ADJUST_GAP
    config.INFLATION_ADJUST_GAP = params["inflation_adjust"]
    try:
        strategy = ValuationGapMultiplesOptionsStrategy(
            entry_threshold_pct=params["entry_threshold_pct"],
            max_positions=params["max_positions"],
            min_positions=params["min_positions"],
            tenor_days=params["tenor_days"],
            gap_basis=params["gap_basis"],
            weight_cap_pct=params["weight_cap_pct"],
        )
        engine = OptionsBacktestEngine(
            price_panel=data.price_panel,
            signal_events=data.signal_events,
            universe_history=data.universe_history,
            fallback_universe_symbols=data.fallback_symbols,
            option_snapshots=data.option_snapshots,
            strategy=strategy,
            initial_capital=initial_capital,
            commission_per_contract=config.OPTIONS_COMMISSION_PER_CONTRACT,
            slippage_pct_of_premium=params["slippage_pct_of_premium"],
            commission_min_per_order=config.OPTIONS_COMMISSION_MIN_PER_ORDER,
            max_fee_pct_of_trade=params["max_fee_pct_of_trade"],
            min_deployment_pct=params["min_deployment_pct"],
            fee_bump_max_extra_pct=params["fee_bump_max_extra_pct"],
            stop_loss_pct=params["stop_loss_pct"],
            take_profit_pct=params["take_profit_pct"],
            max_positions=params["max_positions"],
            target_tenor_days=params["tenor_days"],
            realized_vol_lookback_days=params["realized_vol_lookback_days"],
            momentum_min_pct=params["momentum_min_pct"],
            min_resize_relative_pct=params["min_resize_relative_pct"],
            material_events_8k=data.material_events,
            stop_basis="underlying",
            exit_when_signal_lost=True,
            roll_when_days_left=params["roll_when_days_left"],
            vol_mode="rolling",
            start_date=start_date,
            end_date=end_date,
        )
        equity_curve, _, trades, _ = engine.run()
    except (ValueError, ZeroDivisionError) as exc:
        logger.debug("Essai écarté (%s) : %s", params, exc)
        return None
    finally:
        config.INFLATION_ADJUST_GAP = previous_inflation_setting

    if equity_curve.empty:
        return None
    run_metrics = metrics_mod.compute_metrics(equity_curve, trades, risk_free_rate=config.RISK_FREE_RATE)
    if not run_metrics:
        return None
    return {**params, **{k: run_metrics.get(k) for k in REPORT_COLUMNS}}


# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=300, help="Nombre de combinaisons tirées.")
    parser.add_argument("--seed", type=int, default=None, help="Graine aléatoire, pour reproduire un run.")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--initial-capital", type=float, default=config.OPTIONS_INITIAL_CAPITAL)
    parser.add_argument(
        "--objectif", default="calmar_ratio", choices=REPORT_COLUMNS,
        help="Métrique de classement (défaut : calmar_ratio = rendement/risque).",
    )
    parser.add_argument(
        "--min-trades", type=int, default=15,
        help="Écarte du classement les essais avec moins de trades (mesure trop bruitée). "
             "0 pour ne rien écarter.",
    )
    parser.add_argument("--top", type=int, default=20, help="Nombre de lignes affichées en tête de classement.")
    parser.add_argument(
        "--output", default=None,
        help="Fichier CSV de sortie (défaut : data/backtest_options/optimize_multiples_<horodatage>.csv).",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=25,
        help="Sauvegarde intermédiaire tous les N essais (pour ne rien perdre sur un run interrompu).",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    start_date = pd.Timestamp(args.start_date) if args.start_date else None
    end_date = pd.Timestamp(args.end_date) if args.end_date else None

    output_path = Path(args.output) if args.output else (
        config.DIR_BACKTEST_OPTIONS / f"optimize_multiples_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_data()

    logger.info("Lancement de %d essais (objectif de classement : %s)...", args.trials, args.objectif)
    rows: list[dict] = []
    failures = 0
    started = time.monotonic()
    for i in range(1, args.trials + 1):
        params = sample_params(rng)
        result = run_trial(params, data, args.initial_capital, start_date, end_date)
        if result is None:
            failures += 1
            continue
        rows.append(result)

        if i % max(args.checkpoint_every, 1) == 0 or i == args.trials:
            pd.DataFrame(rows).to_csv(output_path, index=False)
            elapsed = time.monotonic() - started
            logger.info(
                "%d/%d essais (%d écartés, %.1fs écoulées, ~%.2fs/essai) -- checkpoint : %s",
                i, args.trials, failures, elapsed, elapsed / i, output_path,
            )

    if not rows:
        logger.error("Aucun essai exploitable -- vérifie les données d'entrée (08/06b/09).")
        return

    df = pd.DataFrame(rows)
    ranked = df[df["num_trades"].fillna(0) >= args.min_trades].sort_values(args.objectif, ascending=False)
    if ranked.empty:
        logger.warning(
            "Aucun essai n'atteint --min-trades=%d : classement sur l'ensemble des essais à la place.",
            args.min_trades,
        )
        ranked = df.sort_values(args.objectif, ascending=False)

    df.to_csv(output_path, index=False)
    logger.info("Résultats complets (%d essais) sauvegardés : %s", len(df), output_path)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"\n=== Top {min(args.top, len(ranked))} par {args.objectif} ===")
    cols = list(SEARCH_SPACE.keys()) + REPORT_COLUMNS
    print(ranked[cols].head(args.top).to_string(index=False))

    print(f"\n=== Référence : réglages actuels de config.py pour les 3 paramètres balayés "
          f"(le reste = FIXED_PARAMS, dont min_deployment_pct désactivé) ===")
    # Construit à partir de FIXED_PARAMS (même source que les essais, aucune
    # duplication) : seuls les 3 paramètres balayés prennent leur valeur
    # actuelle de config.py plutôt que la valeur figée.
    baseline = {
        **FIXED_PARAMS,
        "entry_threshold_pct": config.OPTIONS_MULTIPLES_ENTRY_THRESHOLD_PCT,
        "stop_loss_pct": config.OPTIONS_MULTIPLES_STOP_LOSS_PCT,
        "take_profit_pct": config.OPTIONS_MULTIPLES_TAKE_PROFIT_PCT,
    }
    baseline_result = run_trial(baseline, data, args.initial_capital, start_date, end_date)
    if baseline_result:
        print(pd.DataFrame([baseline_result])[cols].to_string(index=False))
    else:
        print("(réglages actuels : aucun résultat exploitable sur cette période)")

    # Sensibilité : corrélation de chaque paramètre numérique à l'objectif,
    # sur l'ensemble des essais -- indique lesquels PÈSENT vraiment, à
    # regarder AVANT de figer le meilleur essai isolé (cf. avertissement en
    # tête de fichier sur le surapprentissage).
    print(f"\n=== Sensibilité : corrélation de chaque paramètre à {args.objectif} (tous essais) ===")
    numeric_params = [p for p, (_, _, k) in SEARCH_SPACE.items() if k in ("int", "float")]
    corr = df[numeric_params + [args.objectif]].corr(numeric_only=True)[args.objectif].drop(args.objectif)
    print(corr.sort_values(key=abs, ascending=False).to_string())


if __name__ == "__main__":
    main()
