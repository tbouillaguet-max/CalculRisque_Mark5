"""Relit les sorties d'un run de 09_backtest.py et répond aux questions que
metrics.json ne pose pas : ces chiffres sont-ils crédibles, et de quoi
sont-ils faits ?

Aucun recalcul de backtest : tout est reconstruit depuis equity_curve.parquet,
trades.parquet et signals_history.parquet, donc l'audit d'un run existant
coûte quelques secondes.

Cinq questions, dans l'ordre où elles peuvent invalider un résultat :

    1. COUVERTURE DES SIGNAUX. Un univers point-in-time (01b) interdit
       d'acheter une entreprise avant son entrée dans l'indice, mais il ne
       CRÉE pas les fondamentaux des entreprises radiées. 03b se rabat sur
       l'univers complet quand il existe, alors que 04/04b s'en tiennent par
       défaut à l'univers ACTUEL : les cours des radiées nourrissent alors
       l'indice de référence équipondéré, mais leurs signaux n'existent pas et
       la stratégie ne choisit QUE parmi des survivantes. C'est le biais qui
       fabrique le plus d'alpha imaginaire, et c'est celui qu'on mesure ici en
       premier.

    2. MÉTRIQUES PAR THÈSE. trades.parquet logue une ligne par VENTE, ventes
       partielles de rebalancement comprises. num_trades, win_rate_pct et
       profit_factor comptent donc des exécutions, pas des paris -- et pas de
       façon neutre : une ligne qui monte est allégée à chaque rebalancement
       (autant de petits gains inscrits) puis sort d'un coup au stop (UNE
       perte).

    3. STABILITÉ DE L'ALPHA. Un CAGR unique sur 17 ans ne dit pas s'il a été
       gagné régulièrement ou sur une seule fenêtre. L'alpha est donc
       redécoupé par année civile et par sous-période glissante.

    4. SENSIBILITÉ À LA DATE DE DÉPART. Un backtest long-only qui commence
       juste après un krach part avec un avantage qui n'a rien à voir avec la
       stratégie.

    5. SENSIBILITÉ AUX COÛTS. Le run suppose commission + slippage =
       10 bps par aller simple. Cette section chiffre ce que devient le
       résultat si l'exécution réelle coûte davantage.

Usage :
    python 14_audit_backtest.py                        # dernier run
    python 14_audit_backtest.py --run-id 20260816_000429
    python 14_audit_backtest.py --run-dir data/backtest/20260816_000429
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from backtest import data_loader, metrics as metrics_mod

logger = logging.getLogger("backtest.audit")

TRADING_DAYS_PER_YEAR = metrics_mod.TRADING_DAYS_PER_YEAR


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
def resolve_run_dir(run_id: Optional[str], run_dir: Optional[Path]) -> Path:
    if run_dir is not None:
        return run_dir
    if run_id is not None:
        return config.DIR_BACKTEST / run_id
    runs = sorted(p for p in config.DIR_BACKTEST.glob("*") if (p / "equity_curve.parquet").exists())
    if not runs:
        raise FileNotFoundError(
            f"Aucun run exploitable dans {config.DIR_BACKTEST}. Lance d'abord 09_backtest.py."
        )
    return runs[-1]


def load_run(run_dir: Path) -> dict:
    def parquet(name: str) -> pd.DataFrame:
        path = run_dir / f"{name}.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def json_file(name: str) -> dict:
        path = run_dir / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    return {
        "dir": run_dir,
        "equity_curve": parquet("equity_curve"),
        "trades": parquet("trades"),
        "signals_history": parquet("signals_history"),
        "metrics": json_file("metrics"),
        # Les coûts appliqués au run sont dans run_config.json, pas dans
        # metrics.json : sans lui, la section 5 comparerait à un coût supposé.
        "run_config": json_file("run_config"),
    }


# --------------------------------------------------------------------------- #
# 1. Couverture des signaux : le biais de survivance résiduel
# --------------------------------------------------------------------------- #
def audit_signal_coverage(signals_history: pd.DataFrame, equity_curve: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Part de l'indice RÉEL de chaque année que la stratégie pouvait acheter.

    La ventilation membres actuels / anciens membres est le test décisif : une
    couverture proche de 100% sur les premiers et proche de 0% sur les seconds
    signifie que la stratégie n'a jamais pu détenir une entreprise appelée à
    sortir de l'indice, alors que l'indice de référence, lui, les porte
    toutes. L'alpha se lit alors contre un repère que la stratégie n'avait pas
    le droit de perdre."""
    history = data_loader.load_universe_history()
    if history is None:
        logger.warning(
            "Pas de %s : couverture non mesurable. Lance 01b_historique_univers_sp500.py. "
            "Sans lui, l'univers ACTUEL a de toute façon été appliqué à toutes les dates "
            "passées, ce qui est le biais de survivance dans sa forme la plus large.",
            config.UNIVERSE_HISTORY_FILE,
        )
        return None
    if signals_history.empty:
        logger.warning("signals_history.parquet vide ou absent : couverture non mesurable.")
        return None

    history = history.copy()
    history["symbol"] = history["ric"].map(config.to_ib_symbol)
    # "Encore membre aujourd'hui" = au moins un span sans date de sortie.
    toujours_membre = set(history.loc[history["end_date"].isna(), "symbol"].dropna())

    publies = signals_history.copy()
    publies["date"] = pd.to_datetime(publies["date"])
    premier_signal = publies.groupby("symbol")["date"].min()

    dates = pd.DatetimeIndex(equity_curve["date"])
    rows = []
    for annee, fin in pd.Series(dates, index=dates.year).groupby(level=0).last().items():
        actifs = history["start_date"].isna() | (history["start_date"] <= fin)
        sortis = history["end_date"].isna() | (history["end_date"] > fin)
        membres = set(history.loc[actifs & sortis, "symbol"].dropna())
        if not membres:
            continue
        avec_signal = {s for s in membres if s in premier_signal.index and premier_signal[s] <= fin}

        courants = membres & toujours_membre
        anciens = membres - toujours_membre
        rows.append({
            "annee": int(annee),
            "membres": len(membres),
            "avec_signal": len(avec_signal),
            "couverture": len(avec_signal) / len(membres),
            "couverture_membres_actuels": len(avec_signal & courants) / len(courants) if courants else np.nan,
            "couverture_anciens_membres": len(avec_signal & anciens) / len(anciens) if anciens else np.nan,
        })
    return pd.DataFrame(rows) if rows else None


# --------------------------------------------------------------------------- #
# 3-4. Stabilité de l'alpha, sensibilité à la date de départ
# --------------------------------------------------------------------------- #
def _aligned_benchmark(equity_curve: pd.DataFrame) -> Optional[pd.Series]:
    """Indice de référence reconstruit exactement comme 09_backtest.py.

    Nécessite les cours quotidiens (03b) et, pour l'univers point-in-time,
    01b -- sinon la comparaison n'est simplement pas produite."""
    try:
        panel = data_loader.build_price_panel(data_loader.load_daily_prices())
    except FileNotFoundError:
        logger.warning("Cours quotidiens introuvables : sections alpha non produites.")
        return None
    universe = data_loader.UniverseResolver(
        data_loader.load_universe_history(), data_loader.load_current_universe_symbols(),
    )
    serie, _ = data_loader.build_benchmark_series(panel, universe)
    if serie is None:
        return None
    dates = pd.DatetimeIndex(equity_curve["date"])
    return serie.reindex(serie.index.union(dates)).ffill().reindex(dates)


def _cagr(serie: pd.Series) -> float:
    valides = serie.dropna()
    if len(valides) < 2 or valides.iloc[0] <= 0:
        return np.nan
    annees = (valides.index[-1] - valides.index[0]).days / 365.25
    if annees <= 0:
        return np.nan
    return ((valides.iloc[-1] / valides.iloc[0]) ** (1 / annees) - 1) * 100


def audit_par_annee(equity_curve: pd.DataFrame, benchmark: Optional[pd.Series]) -> pd.DataFrame:
    nav = pd.Series(equity_curve["nav"].to_numpy(), index=pd.DatetimeIndex(equity_curve["date"]))
    rows = []
    for annee, groupe in nav.groupby(nav.index.year):
        strategie = (groupe.iloc[-1] / groupe.iloc[0] - 1) * 100
        row = {"annee": int(annee), "strategie_pct": strategie}
        if benchmark is not None:
            fenetre = benchmark[benchmark.index.year == annee].dropna()
            if len(fenetre) > 1 and fenetre.iloc[0] > 0:
                row["indice_pct"] = (fenetre.iloc[-1] / fenetre.iloc[0] - 1) * 100
                row["ecart_pct"] = strategie - row["indice_pct"]
        rows.append(row)
    return pd.DataFrame(rows)


def audit_sous_periodes(
    equity_curve: pd.DataFrame, benchmark: Optional[pd.Series], fenetre_annees: int = 5,
) -> pd.DataFrame:
    """Alpha sur des fenêtres glissantes : un alpha gagné une fois puis jamais
    revu ne se voit pas dans un CAGR unique."""
    nav = pd.Series(equity_curve["nav"].to_numpy(), index=pd.DatetimeIndex(equity_curve["date"]))
    rows = []
    debut, fin = nav.index[0], nav.index[-1]
    curseur = debut
    while curseur + pd.DateOffset(years=fenetre_annees) <= fin:
        borne = curseur + pd.DateOffset(years=fenetre_annees)
        fenetre = nav[(nav.index >= curseur) & (nav.index <= borne)]
        row = {
            "debut": curseur.date(), "fin": borne.date(),
            "strategie_cagr_pct": _cagr(fenetre),
        }
        if benchmark is not None:
            ref = benchmark[(benchmark.index >= curseur) & (benchmark.index <= borne)]
            row["indice_cagr_pct"] = _cagr(ref)
            row["alpha_pct"] = row["strategie_cagr_pct"] - row["indice_cagr_pct"]
        rows.append(row)
        curseur += pd.DateOffset(years=1)
    return pd.DataFrame(rows)


def audit_date_de_depart(equity_curve: pd.DataFrame, benchmark: Optional[pd.Series]) -> pd.DataFrame:
    """CAGR obtenu si le run avait commencé 1, 2, 3... ans plus tard.

    Un long-only qui démarre juste après un krach hérite d'un rebond qui n'est
    pas de son fait. Si le CAGR s'effondre en décalant le départ d'un an, ce
    n'est pas la stratégie qu'on mesure, c'est le point d'entrée."""
    nav = pd.Series(equity_curve["nav"].to_numpy(), index=pd.DatetimeIndex(equity_curve["date"]))
    rows = []
    for decalage in (0, 1, 2, 3, 5):
        depart = nav.index[0] + pd.DateOffset(years=decalage)
        if depart >= nav.index[-1]:
            break
        fenetre = nav[nav.index >= depart]
        row = {
            "depart": depart.date(), "annees": round((nav.index[-1] - depart).days / 365.25, 1),
            "strategie_cagr_pct": _cagr(fenetre),
        }
        if benchmark is not None:
            ref = benchmark[benchmark.index >= depart]
            row["indice_cagr_pct"] = _cagr(ref)
            row["alpha_pct"] = row["strategie_cagr_pct"] - row["indice_cagr_pct"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Sensibilité aux coûts d'exécution
# --------------------------------------------------------------------------- #
def audit_couts(run: dict) -> Optional[pd.DataFrame]:
    """Ce que devient le CAGR si l'exécution réelle coûte plus cher que les
    10 bps par aller simple supposés par le run.

    Approximation ASSUMÉE, et conservatrice dans sa forme : le surcoût est
    appliqué au volume échangé tel qu'il a eu lieu, sans que le portefeuille
    réagisse (moins de trades, positions plus grosses). Un vrai chiffre
    demande de relancer 09 avec --slippage-bps ; l'ordre de grandeur suffit
    à savoir si la stratégie survit ou non à une exécution réaliste."""
    trades, equity_curve = run["trades"], run["equity_curve"]
    if trades.empty or equity_curve.empty or "shares" not in trades.columns:
        return None

    volume = float((trades["shares"] * trades["exit_price"]).sum()
                   + (trades["shares"] * trades["entry_price"]).sum())
    nav_moyen = float(equity_curve["nav"].mean())
    nav_final = float(equity_curve["nav"].iloc[-1])
    annees = (equity_curve["date"].iloc[-1] - equity_curve["date"].iloc[0]).days / 365.25
    cagr_reference = run["metrics"].get("cagr_pct")
    run_config = run.get("run_config") or {}
    applique_bps = (
        float(run_config.get("commission_bps", config.BACKTEST_COMMISSION_BPS))
        + float(run_config.get("slippage_bps", config.BACKTEST_SLIPPAGE_BPS))
    )

    rows = []
    for total_bps in (applique_bps, 20.0, 30.0, 50.0):
        surcout = volume * (total_bps - applique_bps) / 10_000
        nav_ajuste = nav_final - surcout
        rows.append({
            "cout_aller_simple_bps": total_bps,
            "surcout_total_dollar": surcout,
            "cagr_approx_pct": (
                ((nav_ajuste / float(equity_curve["nav"].iloc[0])) ** (1 / annees) - 1) * 100
                if nav_ajuste > 0 and annees > 0 else np.nan
            ),
        })
    out = pd.DataFrame(rows)
    out.attrs["volume"] = volume
    out.attrs["rotation_annuelle_pct"] = volume / nav_moyen / annees * 100 if annees > 0 else np.nan
    out.attrs["cagr_reference"] = cagr_reference
    return out


# --------------------------------------------------------------------------- #
def _pct(value) -> str:
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:+.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default=None, help="Sous-dossier de data/backtest (défaut: le plus récent).")
    parser.add_argument("--run-dir", type=Path, default=None, help="Chemin complet, prioritaire sur --run-id.")
    parser.add_argument("--fenetre-annees", type=int, default=5, help="Largeur des sous-périodes glissantes (défaut: %(default)s).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pd.set_option("display.width", 140)

    try:
        run_dir = resolve_run_dir(args.run_id, args.run_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    if not (run_dir / "equity_curve.parquet").exists():
        logger.error("%s ne contient pas de equity_curve.parquet.", run_dir)
        sys.exit(1)

    run = load_run(run_dir)
    equity_curve, trades = run["equity_curve"], run["trades"]
    print(f"\n=== Audit de {run_dir} ===")
    print(f"Période : {equity_curve['date'].iloc[0].date()} -> {equity_curve['date'].iloc[-1].date()}"
          f"  |  CAGR affiché : {_pct(run['metrics'].get('cagr_pct'))}"
          f"  |  alpha affiché : {_pct(run['metrics'].get('alpha_pct'))}")

    # 1 ------------------------------------------------------------------ #
    print("\n--- 1. Couverture de l'univers point-in-time par les signaux ---")
    couverture = audit_signal_coverage(run["signals_history"], equity_curve)
    if couverture is None:
        print("non mesurable (voir avertissements ci-dessus).")
    else:
        affichage = couverture.copy()
        for col in ("couverture", "couverture_membres_actuels", "couverture_anciens_membres"):
            affichage[col] = (affichage[col] * 100).round(1)
        print(affichage.to_string(index=False))
        actuels = couverture["couverture_membres_actuels"].mean()
        anciens = couverture["couverture_anciens_membres"].mean()
        print(f"\nMoyenne : {couverture['couverture'].mean() * 100:.1f}% de l'indice couvert "
              f"({actuels * 100:.1f}% des membres actuels, {anciens * 100:.1f}% des anciens).")
        if pd.notna(anciens) and pd.notna(actuels) and anciens < actuels / 2:
            print(
                "  /!\\ BIAIS DE SURVIVANCE. La stratégie n'a quasiment jamais pu détenir une\n"
                "      entreprise appelée à sortir de l'indice, alors que l'indice de référence\n"
                "      équipondéré les porte toutes : l'alpha affiché est surestimé, et un signal\n"
                "      'value' est précisément celui que ce biais flatte le plus.\n"
                f"      Correction : 04_recuperation_10k.py --tickers {config.UNIVERSE_FULL_FILE}\n"
                "      (idem 04b), puis 07_calcul_dcf.py, puis relancer 09."
            )
        elif pd.isna(anciens):
            print(
                "  /!\\ TEST DÉCISIF NON DÉCLENCHÉ. 'couverture_anciens_membres' est NaN sur\n"
                "      TOUTES les années : sp500_universe_history.parquet ne contient aucune\n"
                "      entreprise radiée qui recouvre la période auditée -- le biais de\n"
                "      survivance n'a donc pas pu être mesuré, ce qui n'est PAS la même chose\n"
                "      que 'pas de biais'. Vérifie que 01b_historique_univers_sp500.py a bien\n"
                "      trouvé des radiations (regarde son log : 'X tickers ... radiés') avant\n"
                "      de faire confiance à l'alpha affiché."
            )

    # 2 ------------------------------------------------------------------ #
    print("\n--- 2. Trades affichés (exécutions) vs thèses réelles ---")
    par_these = metrics_mod._position_level_metrics(trades)
    if not par_these:
        print("trades.parquet vide ou sans colonne entry_date.")
    else:
        print(f"num_trades affiché      : {run['metrics'].get('num_trades')} exécutions "
              f"(ventes partielles de rebalancement comprises)")
        print(f"thèses réellement closes : {par_these['num_positions_closed']}")
        print(f"win_rate affiché         : {_pct(run['metrics'].get('win_rate_pct'))} "
              f"-> par thèse : {_pct(par_these['win_rate_positions_pct'])}")
        print(f"profit_factor affiché    : {run['metrics'].get('profit_factor')} "
              f"-> par thèse : {par_these['profit_factor_positions']}")
        if "exit_reason" in trades.columns:
            print(f"motifs de sortie         : {trades['exit_reason'].value_counts().to_dict()}")
            if int((trades['exit_reason'] == 'data_gap').sum()) == 0:
                print("  note : aucune sortie 'data_gap' -- aucune position n'a jamais cessé d'être\n"
                      "         cotée, ce qui confirme en général le biais mesuré en section 1.")

    benchmark = _aligned_benchmark(equity_curve)

    # 3 ------------------------------------------------------------------ #
    print("\n--- 3. Année par année ---")
    print(audit_par_annee(equity_curve, benchmark).round(2).to_string(index=False))

    print(f"\n--- 3b. Sous-périodes glissantes de {args.fenetre_annees} ans ---")
    sous_periodes = audit_sous_periodes(equity_curve, benchmark, args.fenetre_annees)
    if sous_periodes.empty:
        print("historique trop court.")
    else:
        print(sous_periodes.round(2).to_string(index=False))
        if "alpha_pct" in sous_periodes.columns:
            negatifs = int((sous_periodes["alpha_pct"] < 0).sum())
            print(f"\nAlpha négatif sur {negatifs}/{len(sous_periodes)} fenêtres "
                  f"(médiane {_pct(sous_periodes['alpha_pct'].median())}).")

    # 4 ------------------------------------------------------------------ #
    print("\n--- 4. Sensibilité à la date de départ ---")
    print(audit_date_de_depart(equity_curve, benchmark).round(2).to_string(index=False))

    # 5 ------------------------------------------------------------------ #
    print("\n--- 5. Sensibilité aux coûts d'exécution ---")
    couts = audit_couts(run)
    if couts is None:
        print("non calculable (trades.parquet vide).")
    else:
        print(f"volume échangé : {couts.attrs['volume']:,.0f} $  |  "
              f"rotation annuelle : {couts.attrs['rotation_annuelle_pct']:.0f}% du NAV moyen")
        print(couts.round(2).to_string(index=False))
        print("(approximation : le surcoût est appliqué au volume constaté, le portefeuille ne\n"
              " réagit pas. Pour un chiffre exact, relancer 09_backtest.py --slippage-bps N.)")

    print()


if __name__ == "__main__":
    main()
