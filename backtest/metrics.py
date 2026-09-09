"""Métriques de performance et de risque calculées à partir de la sortie de
BacktestEngine.run() (equity_curve, trades). Objectif de 09_backtest.py :
répondre à "cette stratégie est-elle bénéficiaire ?", pas seulement via le
rendement total mais avec de quoi juger la robustesse (drawdown, Sharpe,
Sortino, VaR/CVaR, profit factor...).

"Bénéficiaire" ne suffit pas : un CAGR de 4% sur 2015-2026 est une
SOUS-PERFORMANCE massive, et aucune des métriques ci-dessus ne le dit. D'où
`benchmark_prices` (voir compute_metrics) : tant qu'une stratégie n'est pas
comparée à ce qu'aurait rapporté le fait de ne rien décider, ses chiffres ne
se lisent pas."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

import config

TRADING_DAYS_PER_YEAR = 252

_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    """Même implémentation que backtest/options_pricing.py (math.erfc plutôt
    que scipy.stats.norm) : ce module n'a pas d'autre besoin de scipy, et une
    dépendance entière pour deux appels scalaires ne se justifie pas."""
    return 0.5 * math.erfc(-x * _INV_SQRT_2)


def _norm_ppf(p: float) -> float:
    """Quantile de la loi normale centrée réduite, par l'approximation
    rationnelle d'Acklam (erreur relative < 1,15e-9 sur tout le support) --
    largement au-delà de ce qu'exige un seuil de bruit dont les entrées, elles,
    sont connues à deux chiffres près.

    Le raffiner par une itération de Newton n'aurait aucun sens ici : `p` vient
    d'un NOMBRE D'ESSAIS approximatif, pas d'une mesure."""
    if not 0.0 < p < 1.0:
        return math.copysign(math.inf, p - 0.5)

    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_bas, p_haut = 0.02425, 1 - 0.02425

    if p < p_bas:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > p_haut:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def _max_drawdown_duration_days(equity_curve: pd.DataFrame) -> int:
    running_max = equity_curve["nav"].cummax()
    in_drawdown = equity_curve["nav"] < running_max
    if not in_drawdown.any():
        return 0
    # Longueur (en jours calendaires) de la plus longue séquence continue en drawdown.
    group_id = (~in_drawdown).cumsum()
    durations = equity_curve.loc[in_drawdown].groupby(group_id[in_drawdown])["date"].agg(lambda d: (d.max() - d.min()).days + 1)
    return int(durations.max()) if not durations.empty else 0


def _risk_free_daily(dates: pd.Series, fallback_annual_rate: float) -> pd.Series:
    """Taux sans risque quotidien, pris sur la courbe annuelle
    config.RISK_FREE_RATE_BY_YEAR quand elle existe.

    Repli sur `fallback_annual_rate` (constante, comportement d'avant) pour
    les années absentes de la table, ou si config ne porte pas la courbe --
    on ne veut pas qu'une métrique dépende de la présence d'un réglage
    optionnel."""
    by_year = getattr(config, "RISK_FREE_RATE_BY_YEAR", None)
    if not by_year:
        return pd.Series(fallback_annual_rate / TRADING_DAYS_PER_YEAR, index=dates.index)
    annual = pd.DatetimeIndex(dates).year.map(lambda y: by_year.get(int(y), fallback_annual_rate))
    return pd.Series(np.asarray(annual, dtype=float) / TRADING_DAYS_PER_YEAR, index=dates.index)


# ----------------------------------------------------------------------------
# Sharpe DÉFLATÉ : ce que le Sharpe vaut une fois retiré ce que le hasard donne
# ----------------------------------------------------------------------------
#
# LE PROBLÈME. Un grid-search classé sur un unique chemin historique retient, par
# construction, la combinaison qui colle le mieux à CE chemin-là. Le Sharpe qu'il
# affiche n'est donc pas le Sharpe d'une stratégie, c'est le MAXIMUM de N tirages
# -- et le maximum de N tirages n'est pas nul même quand la vraie performance
# l'est. Il croît en racine(2·ln N).
#
# Ordre de grandeur, avec l'écart-type d'un Sharpe annualisé estimé sur T années
# (~1/racine(T)) : sur 10 ans d'historique, 8 essais donnent déjà 0,46 de Sharpe
# gratuit, 64 essais 0,75, une centaine 0,80. Le seul optimiseur des stops de ce
# dépôt en balaie 64 (8 stops x 8 take-profits), et il n'est pas seul.
#
# LA CORRECTION. Bailey & Lopez de Prado (Journal of Portfolio Management, 2014)
# proposent de rendre non plus un ratio mais une PROBABILITÉ : celle que le vrai
# Sharpe dépasse ce seuil de bruit, en tenant compte de l'asymétrie et des queues
# de la distribution des rendements -- deux corrections qui comptent ici, un
# portefeuille d'options longues n'ayant rien de gaussien.
#
#     DSR = Phi[ (SR - SR0)·racine(N-1) / racine(1 - g3·SR + (g4-1)/4·SR²) ]
#
# avec N le nombre d'observations, g3 l'asymétrie, g4 le kurtosis (3 = normal),
# et SR, SR0 exprimés PAR OBSERVATION (jamais annualisés : la formule mélange
# sinon deux échelles). Lecture : DSR = 0,95 veut dire « 95% de chances que la
# vraie performance dépasse ce que N essais auraient produit par hasard ».
#
# Harvey, Liu & Zhu (Review of Financial Studies, 2016) arrivent au même endroit
# par la voie des tests multiples : la barre pour un nouveau facteur n'est pas un
# t de 2,0 mais d'environ 3,0.

_EULER_MASCHERONI = 0.5772156649015329


def expected_maximum_sharpe(n_trials: int, sharpe_std: float) -> float:
    """SR0 : Sharpe MAXIMUM attendu sur `n_trials` essais indépendants quand la
    vraie performance est nulle. Exprimé dans l'unité de `sharpe_std`.

    Approximation de l'espérance du maximum de n_trials gaussiennes :

        SR0 = sharpe_std · [ (1-gamma)·Z⁻¹(1 - 1/N) + gamma·Z⁻¹(1 - 1/(N·e)) ]

    `sharpe_std` est la dispersion des Sharpe ENTRE ESSAIS. Quand on dispose des
    Sharpe réellement obtenus par le grid-search, c'est leur écart-type qu'il
    faut passer ; sinon, l'écart-type d'échantillonnage d'un Sharpe sous
    l'hypothèse nulle (~1/racine(N observations)) en est l'approximation usuelle.

    0 pour un seul essai : sans sélection, il n'y a pas de biais de sélection à
    retirer, et le Sharpe déflaté se réduit alors au Sharpe probabiliste testé
    contre zéro."""
    if n_trials <= 1 or sharpe_std <= 0 or not math.isfinite(sharpe_std):
        return 0.0
    n = float(n_trials)
    quantile = (
        (1 - _EULER_MASCHERONI) * _norm_ppf(1 - 1 / n)
        + _EULER_MASCHERONI * _norm_ppf(1 - 1 / (n * math.e))
    )
    return sharpe_std * quantile


def probabilistic_sharpe_ratio(
    returns: pd.Series, benchmark_sharpe: float = 0.0,
) -> Optional[float]:
    """P(vrai Sharpe > `benchmark_sharpe`), corrigée de l'asymétrie et des
    queues des rendements.

    `returns` et `benchmark_sharpe` sont PAR OBSERVATION (rendements quotidiens
    -> Sharpe quotidien). Annualiser l'un sans l'autre est l'erreur classique et
    donne des probabilités absurdes."""
    n = len(returns)
    if n < 4:
        return None
    ecart_type = float(returns.std())
    if not ecart_type or not math.isfinite(ecart_type) or ecart_type <= 0:
        return None

    sharpe = float(returns.mean()) / ecart_type
    asymetrie = float(returns.skew())
    kurtosis = float(returns.kurt()) + 3.0        # pandas rend l'EXCÈS de kurtosis
    if not all(math.isfinite(x) for x in (sharpe, asymetrie, kurtosis)):
        return None

    # Variance de l'estimateur du Sharpe (Mertens) : gaussienne -> 1, puis
    # corrigée par l'asymétrie et l'épaisseur des queues.
    variance = 1.0 - asymetrie * sharpe + (kurtosis - 1.0) / 4.0 * sharpe * sharpe
    if variance <= 0:
        return None
    return _norm_cdf((sharpe - benchmark_sharpe) * math.sqrt(n - 1) / math.sqrt(variance))


def deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int, sharpe_std: Optional[float] = None,
) -> Optional[float]:
    """DSR : probabilité que le vrai Sharpe dépasse ce que `n_trials` essais
    auraient produit par pur hasard (cf. le pavé ci-dessus).

    `sharpe_std` : dispersion des Sharpe entre essais, PAR OBSERVATION. À
    défaut, l'écart-type d'échantillonnage sous l'hypothèse nulle,
    1/racine(N-1) -- approximation prudente et standard quand les Sharpe des
    autres essais n'ont pas été conservés."""
    n = len(returns)
    if n < 4:
        return None
    if sharpe_std is None:
        sharpe_std = 1.0 / math.sqrt(n - 1)
    return probabilistic_sharpe_ratio(
        returns, benchmark_sharpe=expected_maximum_sharpe(n_trials, sharpe_std))


def _position_level_metrics(trades: pd.DataFrame) -> dict:
    """Mêmes trades, regroupés par THÈSE (une entrée, ses renforts, ses
    allègements, sa sortie) au lieu d'une ligne par exécution.

    POURQUOI. `trades.parquet` logue une ligne par VENTE, y compris les ventes
    partielles de rebalancement. `num_trades`, `win_rate_pct` et
    `profit_factor` comptent donc des exécutions, pas des paris -- et le biais
    n'est pas neutre : une position qui monte est rognée à chaque
    rebalancement, ce qui inscrit une longue série de petits gains, puis sort
    d'un coup au stop, ce qui n'en inscrit qu'UNE perte. Mesuré sur une thèse
    unique perdante de -226 k$, la version par exécution affichait 32 trades et
    91% de réussite.

    La clé de regroupement est (symbol, entry_date) : engine.Position ne
    remet jamais entry_date à jour lors d'un renfort, et une position
    entièrement soldée est supprimée du portefeuille -- une réouverture
    ultérieure du même symbole porte donc une entry_date différente et compte
    bien comme une thèse distincte."""
    if trades is None or trades.empty or "entry_date" not in trades.columns:
        return {}
    par_these = trades.groupby(["symbol", "entry_date"], sort=False).agg(
        pnl=("pnl", "sum"), holding_days=("holding_days", "max"),
    )
    gains = par_these.loc[par_these["pnl"] > 0, "pnl"].sum()
    pertes = -par_these.loc[par_these["pnl"] < 0, "pnl"].sum()
    return {
        "num_positions_closed": int(len(par_these)),
        "win_rate_positions_pct": float((par_these["pnl"] > 0).mean() * 100),
        "profit_factor_positions": float(gains / pertes) if pertes > 0 else None,
        "avg_holding_days_positions": float(par_these["holding_days"].mean()),
    }


def _benchmark_metrics(equity_curve: pd.DataFrame, benchmark_prices: pd.Series, n_years: float) -> dict:
    """Comparaison à un indice de référence, sur EXACTEMENT le même
    calendrier que la courbe de NAV : la série de l'indice est réalignée sur
    les dates de l'equity_curve (report de la dernière clôture connue), sinon
    un jour férié d'un côté seulement décalerait toute la série de rendements
    et le beta comme le tracking error deviendraient du bruit.

    Retourne {} si l'indice ne couvre pas réellement la période -- mieux vaut
    pas de benchmark qu'un benchmark faux. Le décompte porte sur les
    observations RÉELLES tombant dans la fenêtre, pas sur la série remplie :
    une série qui s'arrête au premier jour serait reportée en ligne plate sur
    tout le run et afficherait un rendement de 0%, ce qui se lirait comme une
    surperformance imaginaire au lieu d'une absence de donnée."""
    dates = pd.DatetimeIndex(equity_curve["date"])
    observed = pd.to_numeric(benchmark_prices, errors="coerce").dropna()
    in_window = observed[(observed.index >= dates.min()) & (observed.index <= dates.max())]
    if len(in_window) < 2:
        return {}

    aligned = pd.to_numeric(
        observed.reindex(observed.index.union(dates)).ffill().reindex(dates), errors="coerce",
    )
    if aligned.notna().sum() < 2:
        return {}

    bench_start, bench_end = aligned.dropna().iloc[0], aligned.dropna().iloc[-1]
    if not bench_start or bench_start <= 0:
        return {}

    bench_total_return_pct = (bench_end / bench_start - 1) * 100
    bench_cagr_pct = (
        ((bench_end / bench_start) ** (1 / n_years) - 1) * 100
        if n_years and n_years > 0 else np.nan
    )

    # Rendements quotidiens appariés position par position : les deux séries
    # sont déjà sur le même calendrier (réalignement ci-dessus), on ne garde
    # que les jours où les DEUX sont définis (premier jour, trous de
    # couverture de l'indice).
    paired = pd.DataFrame({
        "strategy": equity_curve["nav"].pct_change(fill_method=None).to_numpy(),
        "benchmark": aligned.pct_change(fill_method=None).to_numpy(),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    metrics = {
        "benchmark_total_return_pct": float(bench_total_return_pct),
        "benchmark_cagr_pct": float(bench_cagr_pct) if pd.notna(bench_cagr_pct) else None,
    }

    strategy_cagr_pct = (
        ((equity_curve["nav"].iloc[-1] / equity_curve["nav"].iloc[0]) ** (1 / n_years) - 1) * 100
        if n_years and n_years > 0 else np.nan
    )
    # alpha = excès de CAGR, pas l'alpha de Jensen : c'est la question posée
    # ("combien ai-je gagné de plus qu'en achetant l'indice ?"), et elle se lit
    # sans avoir à supposer un modèle de marché.
    metrics["alpha_pct"] = (
        float(strategy_cagr_pct - bench_cagr_pct)
        if pd.notna(strategy_cagr_pct) and pd.notna(bench_cagr_pct) else None
    )

    if len(paired) > 1:
        bench_var = paired["benchmark"].var()
        beta = paired["strategy"].cov(paired["benchmark"]) / bench_var if bench_var > 0 else np.nan
        active = paired["strategy"] - paired["benchmark"]
        tracking_error = active.std()
        metrics.update({
            "beta": float(beta) if pd.notna(beta) else None,
            "tracking_error_pct": float(tracking_error * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)
            if pd.notna(tracking_error) else None,
            "information_ratio": float(active.mean() / tracking_error * np.sqrt(TRADING_DAYS_PER_YEAR))
            if pd.notna(tracking_error) and tracking_error > 0 else None,
        })
    return metrics


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.0,
    benchmark_prices: Optional[pd.Series] = None,
    extra: Optional[dict] = None,
    n_trials: int = 1,
) -> dict:
    """benchmark_prices : clôtures quotidiennes de l'indice de référence
    (Series indexée par date). Absent -> les champs benchmark_*, alpha_pct,
    beta, information_ratio et tracking_error_pct ne sont simplement pas
    produits.

    `n_trials` : nombre de configurations essayées avant de retenir celle-ci.
    1 pour un run isolé ; la taille de la grille pour un run issu d'un
    grid-search. Il pilote `deflated_sharpe_ratio` et `sharpe_noise_floor`, les
    deux seuls chiffres qui disent si le Sharpe affiché dépasse ce que la
    sélection aurait produit toute seule (cf. le pavé plus haut). Le laisser à 1
    quand plusieurs dizaines de combinaisons ont été balayées ne rend pas le
    résultat plus prudent : il le rend simplement muet sur la question.

    `extra` : métriques calculées par le moteur lui-même (ordres tronqués,
    exposition delta...), fusionnées telles quelles dans le dictionnaire de
    sortie -- elles n'ont pas leur place ici, mais elles ont leur place dans
    metrics.json."""
    if equity_curve.empty:
        return {}
    equity_curve = equity_curve.sort_values("date").reset_index(drop=True)
    daily_returns = equity_curve["nav"].pct_change(fill_method=None).dropna()

    nav_start, nav_end = equity_curve["nav"].iloc[0], equity_curve["nav"].iloc[-1]
    n_days = (equity_curve["date"].iloc[-1] - equity_curve["date"].iloc[0]).days
    n_years = n_days / 365.25 if n_days > 0 else np.nan

    total_return_pct = (nav_end / nav_start - 1) * 100
    cagr_pct = ((nav_end / nav_start) ** (1 / n_years) - 1) * 100 if n_years and n_years > 0 else np.nan

    ann_vol_pct = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100 if len(daily_returns) > 1 else np.nan
    # Taux sans risque JOUR PAR JOUR quand la courbe annuelle est disponible :
    # appliquer 4% à 2012 (taux réel ~0,09%) fabriquait une prime de risque
    # négative sur une année pourtant positive, et le Sharpe s'en trouvait
    # écrasé sur toute la première moitié de l'historique.
    rf_daily = _risk_free_daily(equity_curve["date"].iloc[1:], risk_free_rate)
    rf_daily.index = daily_returns.index
    excess = daily_returns - rf_daily
    # Sharpe = moyenne des EXCÈS / écart-type des EXCÈS. L'ancienne version
    # divisait par l'écart-type des rendements BRUTS : le taux sans risque
    # étant une constante, les deux écarts-types coïncident tant que le taux
    # est constant, mais plus dès que RISK_FREE_RATE_BY_YEAR le fait varier
    # (cf. C7). Autant écrire la définition.
    excess_std = excess.std()
    sharpe = (excess.mean() / excess_std) * np.sqrt(TRADING_DAYS_PER_YEAR) if excess_std > 0 else np.nan

    # DÉVIATION À LA BAISSE au sens standard : racine de la moyenne des carrés
    # des excès NÉGATIFS, mesurés par rapport à la cible (0 sur des excès), et
    # moyennés sur TOUTES les observations.
    #
    # L'ancienne version prenait `daily_returns[daily_returns < 0].std()`, qui
    # se trompe deux fois : elle mélange un numérateur en excès et un
    # dénominateur en rendements bruts, et surtout `.std()` mesure la
    # dispersion autour de la MOYENNE DES NÉGATIFS, pas autour de zéro. Cette
    # moyenne étant négative, les écarts sont systématiquement rétrécis et le
    # Sortino systématiquement flatté -- d'un facteur 1,19 sur une
    # distribution normale, davantage sur une distribution asymétrique, ce qui
    # est précisément le cas d'un portefeuille d'options longues.
    downside_deviation = float(np.sqrt(np.square(np.minimum(excess, 0.0)).mean()))
    sortino = (
        (excess.mean() / downside_deviation) * np.sqrt(TRADING_DAYS_PER_YEAR)
        if len(excess) > 1 and downside_deviation > 0 else np.nan
    )

    running_max = equity_curve["nav"].cummax()
    drawdown = equity_curve["nav"] / running_max - 1
    max_drawdown_pct = drawdown.min() * 100
    max_drawdown_duration_days = _max_drawdown_duration_days(equity_curve)
    # `max_drawdown_pct not in (0, np.nan)` ne fonctionnait que par accident :
    # `in` teste d'abord l'IDENTITÉ, or np.nan est un singleton -- le test
    # "n'est pas NaN" passait donc pour le np.nan du module numpy et pour lui
    # seul, pas pour un NaN calculé. Test explicite.
    calmar = (
        cagr_pct / abs(max_drawdown_pct)
        if pd.notna(max_drawdown_pct) and max_drawdown_pct != 0 and pd.notna(cagr_pct)
        else np.nan
    )

    var_95_pct = daily_returns.quantile(0.05) * 100 if len(daily_returns) > 1 else np.nan
    cvar_95_pct = daily_returns[daily_returns <= daily_returns.quantile(0.05)].mean() * 100 if len(daily_returns) > 1 else np.nan

    metrics = {
        "start_date": str(equity_curve["date"].iloc[0].date()),
        "end_date": str(equity_curve["date"].iloc[-1].date()),
        "initial_nav": float(nav_start),
        "final_nav": float(nav_end),
        "total_return_pct": float(total_return_pct),
        "cagr_pct": float(cagr_pct) if pd.notna(cagr_pct) else None,
        "annualized_volatility_pct": float(ann_vol_pct) if pd.notna(ann_vol_pct) else None,
        "sharpe_ratio": float(sharpe) if pd.notna(sharpe) else None,
        # Nombre d'essais dont ce run est le survivant, et les deux chiffres
        # qui en découlent. `sharpe_noise_floor` est ANNUALISÉ pour se lire
        # directement à côté de `sharpe_ratio` ; le calcul, lui, reste par
        # observation (cf. expected_maximum_sharpe).
        "n_trials": int(n_trials),
        "sharpe_noise_floor": float(
            expected_maximum_sharpe(n_trials, 1.0 / math.sqrt(len(daily_returns) - 1))
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        ) if len(daily_returns) > 1 else None,
        "deflated_sharpe_ratio": deflated_sharpe_ratio(excess, n_trials),
        "probabilistic_sharpe_ratio": probabilistic_sharpe_ratio(excess),
        "sortino_ratio": float(sortino) if pd.notna(sortino) else None,
        "max_drawdown_pct": float(max_drawdown_pct) if pd.notna(max_drawdown_pct) else None,
        "max_drawdown_duration_days": max_drawdown_duration_days,
        "calmar_ratio": float(calmar) if pd.notna(calmar) else None,
        "var_95_daily_pct": float(var_95_pct) if pd.notna(var_95_pct) else None,
        "cvar_95_daily_pct": float(cvar_95_pct) if pd.notna(cvar_95_pct) else None,
        "skewness_daily_returns": float(daily_returns.skew()) if len(daily_returns) > 2 else None,
        "kurtosis_daily_returns": float(daily_returns.kurt()) if len(daily_returns) > 3 else None,
        "best_day_pct": float(daily_returns.max() * 100) if len(daily_returns) > 0 else None,
        "worst_day_pct": float(daily_returns.min() * 100) if len(daily_returns) > 0 else None,
        "avg_num_positions": float(equity_curve["num_positions"].mean()),
        "max_num_positions": int(equity_curve["num_positions"].max()),
        "avg_exposure_pct": float((equity_curve["invested_value"] / equity_curve["nav"]).mean() * 100),
    }

    if trades is not None and not trades.empty:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]
        gross_profit = wins["pnl"].sum()
        gross_loss = -losses["pnl"].sum()
        turnover_dollar = (trades["shares"] * trades["exit_price"]).sum() + (trades["shares"] * trades["entry_price"]).sum()

        metrics.update({
            "num_trades": int(len(trades)),
            "win_rate_pct": float(len(wins) / len(trades) * 100),
            "avg_win_pct": float(wins["return_pct"].mean()) if not wins.empty else None,
            "avg_loss_pct": float(losses["return_pct"].mean()) if not losses.empty else None,
            "avg_trade_return_pct": float(trades["return_pct"].mean()),
            "best_trade_pct": float(trades["return_pct"].max()),
            "worst_trade_pct": float(trades["return_pct"].min()),
            "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else None,
            "avg_holding_days": float(trades["holding_days"].mean()),
            "exits_by_reason": trades["exit_reason"].value_counts().to_dict(),
            "annualized_turnover_pct": float(turnover_dollar / equity_curve["nav"].mean() / n_years * 100) if n_years and n_years > 0 else None,
        })
        # num_trades/win_rate_pct/profit_factor comptent des EXÉCUTIONS ; les
        # champs *_positions comptent des THÈSES. Voir _position_level_metrics
        # : ce sont ces derniers qu'il faut lire pour juger la stratégie.
        metrics.update(_position_level_metrics(trades))
    else:
        metrics.update({"num_trades": 0})

    if benchmark_prices is not None and not benchmark_prices.empty:
        metrics.update(_benchmark_metrics(equity_curve, benchmark_prices, n_years))

    if extra:
        metrics.update(extra)

    return metrics


def resolve_split_date(equity_curve: pd.DataFrame, train_fraction: float) -> Optional[pd.Timestamp]:
    """Date qui coupe la courbe de NAV en `train_fraction` d'apprentissage et
    le reste en test. None si la courbe est trop courte pour que les deux
    fenêtres aient un sens."""
    if equity_curve.empty or not 0 < train_fraction < 1:
        return None
    dates = pd.DatetimeIndex(equity_curve["date"]).sort_values()
    if len(dates) < 20:
        return None
    return dates[int(len(dates) * train_fraction)]


def split_period_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    split_date: pd.Timestamp,
    risk_free_rate: float = 0.0,
    benchmark_prices: Optional[pd.Series] = None,
    keys: tuple = ("cagr_pct", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
                   "max_drawdown_pct", "total_return_pct", "num_trades", "profit_factor"),
) -> dict:
    """Mêmes métriques, calculées SÉPARÉMENT avant et après `split_date`, et
    préfixées `train_` / `test_`.

    À QUOI ÇA SERT. Un grid-search classé sur l'historique complet retient,
    par construction, la combinaison qui colle le mieux à cet historique-là.
    Avec plusieurs dizaines de combinaisons sur une seule période, le meilleur
    Sharpe est en grande partie du bruit sélectionné : les garde-fous déjà en
    place (nombre minimal de trades, avertissement en bord de grille) écartent
    les artefacts d'échantillon trop petit, mais pas le sur-ajustement. Le
    seul test qui le détecte est de regarder ce que la combinaison retenue
    fait sur des données qui n'ont pas servi à la choisir.

    LE PORTEFEUILLE N'EST PAS REMIS À ZÉRO au changement de période : le run
    est unique et on découpe sa courbe de NAV. C'est volontaire et plus
    réaliste qu'un second run indépendant -- les positions ouvertes à la fin
    de la période d'apprentissage sont bien celles qu'on porterait en entrant
    dans la période de test."""
    out: dict = {}
    if equity_curve.empty or split_date is None:
        return out

    dates = pd.DatetimeIndex(equity_curve["date"])
    windows = {
        "train": equity_curve[dates <= split_date],
        "test": equity_curve[dates > split_date],
    }
    for label, window in windows.items():
        if len(window) < 2:
            continue
        if trades is not None and not trades.empty and "exit_date" in trades.columns:
            exits = pd.DatetimeIndex(trades["exit_date"])
            window_trades = trades[exits <= split_date] if label == "train" else trades[exits > split_date]
        else:
            window_trades = trades
        window_metrics = compute_metrics(
            window, window_trades,
            risk_free_rate=risk_free_rate, benchmark_prices=benchmark_prices,
        )
        for key in keys:
            out[f"{label}_{key}"] = window_metrics.get(key)
    out["split_date"] = str(pd.Timestamp(split_date).date())
    return out


def format_benchmark_summary(metrics: dict, benchmark_label: str) -> str:
    """Ligne de résumé de fin de run. Sortie explicite quand l'indice n'a pas
    pu être construit : l'absence de comparaison doit se voir, pas se
    deviner à l'absence de la ligne."""
    if "benchmark_cagr_pct" not in metrics:
        return (
            "Pas de comparaison à un indice de référence sur ce run "
            "(ni SPY ni univers point-in-time exploitable dans les cours quotidiens)."
        )

    def pct(key: str) -> str:
        value = metrics.get(key)
        return "n/a" if value is None else f"{value:+.2f}%"

    def ratio(key: str) -> str:
        value = metrics.get(key)
        return "n/a" if value is None else f"{value:.2f}"

    return (
        f"Comparaison à {benchmark_label} : CAGR stratégie {pct('cagr_pct')} vs "
        f"{pct('benchmark_cagr_pct')} -> alpha {pct('alpha_pct')} "
        f"(beta {ratio('beta')}, tracking error {pct('tracking_error_pct')}, "
        f"information ratio {ratio('information_ratio')})."
    )
