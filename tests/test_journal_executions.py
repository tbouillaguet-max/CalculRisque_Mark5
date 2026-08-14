"""Journal des exécutions : conservation des quantités, bouclage du cash, et
friction comptée sur TOUS les chemins d'achat.

Motivé par un rapport d'incohérence ("je vends des quantités que je n'ai pas
achetées") qui a mis au jour deux problèmes distincts :

    1. trades.parquet n'enregistre que les VENTES et les présente comme des
       allers-retours dont `entry_date` est la PREMIÈRE ouverture de la
       position -- une position construite en plusieurs achats y solde donc
       plus de contrats qu'il n'en a été acheté à cette date. La comptabilité
       était juste, mais INVÉRIFIABLE depuis les sorties.
    2. _deploy_idle_cash débitait le cash sans comptabiliser sa friction :
       total_friction_dollar sous-estimait donc le coût réel sur le chemin de
       code le plus actif du moteur.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest import put_call_analysis as pca
from backtest.options_engine import OptionsBacktestEngine
from backtest.strategies.valuation_gap_multiples_options import ValuationGapMultiplesOptionsStrategy
from tests.options_harness import cours, moteur, serie_bruitee, signaux

CAPITAL = 2_000_000.0


@pytest.fixture
def run_avec_renforcements():
    """Un run où les positions sont RENFORCÉES après leur ouverture (plusieurs
    achats par position) : c'est la seule configuration où le problème se
    manifeste -- sans renforcement, chaque vente correspond trivialement à son
    unique achat."""
    panel = cours({f"S{i}": serie_bruitee(100.0, 400, graine=i) for i in range(6)})
    dates = panel.close.index
    evenements = pd.concat(
        [
            signaux([f"S{i}"], d.strftime("%Y-%m-%d"),
                    theoretical=(170.0 if i < 3 else 60.0) * (1 + 0.04 * k))
            for i in range(6) for k, d in enumerate(dates[2::70])
        ],
        ignore_index=True,
    )
    engine = OptionsBacktestEngine(
        price_panel=panel, signal_events=evenements, universe_history=None,
        fallback_universe_symbols=set(panel.close.columns), option_snapshots=pd.DataFrame(),
        strategy=ValuationGapMultiplesOptionsStrategy(), initial_capital=CAPITAL,
        commission_per_contract=0.65, slippage_pct_of_premium=2.5,
        stop_loss_pct=-25.0, take_profit_pct=30.0, target_tenor_days=180,
        roll_when_days_left=60, stop_basis="underlying", exit_when_signal_lost=True,
        daily_rebalance=True, vol_mode="rolling", momentum_min_pct=None,
        max_trade_dollar=15_000.0, take_profit_convergence_fraction=0.80,
        # Le plancher de primes est DÉSACTIVÉ par défaut depuis l'audit
        # (config.OPTIONS_MIN_DEPLOYMENT_PCT = 0 : c'était une martingale sur
        # les positions perdantes). Il est réactivé EXPLICITEMENT ici parce
        # que ce fichier teste le JOURNAL de _deploy_idle_cash, pas la
        # pertinence du mécanisme : sans lui, ce chemin de code ne s'exécute
        # simplement jamais et les tests ne vérifient plus rien.
        min_deployment_pct=25.0,
    )
    equity_curve, positions_history, trades, _ = engine.run()
    return engine, equity_curve, positions_history, trades, pd.DataFrame(engine.executions)


# --------------------------------------------------------------------------- #
# Le journal existe et couvre les DEUX sens
# --------------------------------------------------------------------------- #

def test_le_journal_enregistre_les_achats_ET_les_ventes(run_avec_renforcements):
    _, _, _, trades, executions = run_avec_renforcements
    assert not executions.empty
    assert set(executions["side"]) == {"buy", "sell"}
    # Le point de départ du problème : il y a bien PLUS d'achats que de ventes,
    # et trades.parquet n'en voyait aucun.
    assert (executions["side"] == "buy").sum() > (executions["side"] == "sell").sum()
    assert (executions["side"] == "sell").sum() == len(trades)


def test_le_renforcement_par_le_cash_oisif_apparait_au_journal(run_avec_renforcements):
    """C'est l'achat le plus fréquent du moteur (appelé chaque jour de bourse)
    et celui qui n'était tracé nulle part."""
    _, _, _, _, executions = run_avec_renforcements
    assert (executions["reason"] == "deploy_idle_cash").any()


# --------------------------------------------------------------------------- #
# Conservation : rien n'est vendu qui n'ait été acheté
# --------------------------------------------------------------------------- #

def test_aucun_contrat_n_est_vendu_sans_avoir_ete_achete(run_avec_renforcements):
    """L'invariant que le rapport d'incohérence mettait en cause. Vérifié
    SYMBOLE PAR SYMBOLE depuis le seul journal : achats - ventes = ce qui
    reste ouvert à la fin."""
    engine, _, _, _, executions = run_avec_renforcements
    signe = executions["contracts"] * executions["side"].map({"buy": 1, "sell": -1})
    net = executions.assign(net=signe).groupby("symbol")["net"].sum()

    restant = pd.Series(
        {s: p.contracts for s, p in engine.positions.items()}, dtype=float,
    ).reindex(net.index).fillna(0.0)

    assert (net - restant).abs().max() < 1e-9, (net - restant)


def test_un_trade_peut_solder_plus_de_contrats_que_son_entry_date_n_en_a_achete(run_avec_renforcements):
    """Le symptôme rapporté, reproduit et EXPLIQUÉ : c'est le comportement
    attendu de trades.parquet, pas une erreur -- entry_date est la première
    ouverture, la position ayant été renforcée ensuite. Si ce test cessait de
    trouver un tel cas, la documentation de trades.parquet deviendrait
    inutilement alarmiste."""
    _, _, _, trades, executions = run_avec_renforcements
    achats = executions[executions["side"] == "buy"]
    achete_ce_jour = achats.groupby(["symbol", "date"])["contracts"].sum()

    ecarts = [
        t["contracts"] - achete_ce_jour.get((t["symbol"], t["entry_date"]), 0.0)
        for _, t in trades.iterrows()
    ]
    assert max(ecarts) > 0, (
        "aucun trade ne solde plus que son entry_date n'a acheté : le banc ne "
        "reproduit pas le cas signalé"
    )


def test_la_reconciliation_du_rapport_donne_un_ecart_nul(run_avec_renforcements):
    _, _, positions_history, _, executions = run_avec_renforcements
    table = pca.quantity_reconciliation(executions, positions_history)
    assert not table.empty
    assert table.loc["ecart"].abs().max() < 1e-6


# --------------------------------------------------------------------------- #
# Bouclage du cash
# --------------------------------------------------------------------------- #

def test_les_cash_flow_du_journal_reconstituent_le_cash_final(run_avec_renforcements):
    """cash_flow est signé et frais compris : sa somme doit expliquer tout
    l'écart entre capital initial et cash final, aux INTÉRÊTS près.

    Les intérêts du cash oisif (cf. options_engine._accrue_cash_interest) sont
    le seul mouvement de trésorerie qui ne passe pas par un fill : le journal
    des exécutions décrit ce qui a été NÉGOCIÉ, pas tout ce qui a bougé sur le
    compte. Les compter à part est volontaire -- une performance portée par
    les taux ne doit pas se confondre avec une performance portée par la
    thèse."""
    engine, _, _, _, executions = run_avec_renforcements
    attendu = CAPITAL + executions["cash_flow"].sum() + engine.total_cash_interest
    assert attendu == pytest.approx(engine.cash, abs=1e-6)


def test_les_achats_sortent_du_cash_et_les_ventes_y_rentrent(run_avec_renforcements):
    _, _, _, _, executions = run_avec_renforcements
    assert (executions[executions["side"] == "buy"]["cash_flow"] < 0).all()
    assert (executions[executions["side"] == "sell"]["cash_flow"] >= 0).all()


# --------------------------------------------------------------------------- #
# Friction : comptée sur TOUS les chemins d'achat
# --------------------------------------------------------------------------- #

def test_la_friction_du_journal_egale_les_totaux_du_moteur(run_avec_renforcements):
    engine, _, _, _, executions = run_avec_renforcements
    diagnostics = engine.execution_diagnostics()
    assert executions["commission"].sum() == pytest.approx(diagnostics["total_commission_dollar"])
    assert executions["slippage"].sum() == pytest.approx(diagnostics["total_slippage_dollar"])


def test_le_redeploiement_du_cash_oisif_paie_sa_friction():
    """Le bug corrigé : _deploy_idle_cash débitait le cash sans jamais
    incrémenter total_commission/total_slippage. Isolé ici sur un run où
    c'est le SEUL acheteur après l'ouverture initiale."""
    panel = cours({"AAA": serie_bruitee(100.0, 120, graine=1)})
    engine = moteur(
        panel, signaux(["AAA"], "2020-01-06"), {"AAA": "CALL"},
        min_deployment_pct=60.0, slippage_pct_of_premium=2.5, min_resize_relative_pct=None,
    )
    engine.run()

    executions = pd.DataFrame(engine.executions)
    renforts = executions[executions["reason"] == "deploy_idle_cash"]
    assert not renforts.empty, "le banc ne déclenche aucun redéploiement"
    assert renforts["commission"].sum() > 0
    assert renforts["slippage"].sum() > 0
    # Et ces montants sont bien inclus dans les totaux publiés.
    diagnostics = engine.execution_diagnostics()
    assert diagnostics["total_commission_dollar"] >= renforts["commission"].sum() - 1e-9
    assert diagnostics["total_slippage_dollar"] >= renforts["slippage"].sum() - 1e-9


def test_sans_slippage_le_journal_n_en_enregistre_aucun():
    panel = cours({"AAA": serie_bruitee(100.0, 120, graine=1)})
    engine = moteur(
        panel, signaux(["AAA"], "2020-01-06"), {"AAA": "CALL"},
        min_deployment_pct=60.0, slippage_pct_of_premium=0.0,
    )
    engine.run()
    executions = pd.DataFrame(engine.executions)
    assert not executions.empty
    assert executions["slippage"].abs().max() == pytest.approx(0.0)
    assert executions["commission"].sum() > 0    # les frais, eux, restent dus


# --------------------------------------------------------------------------- #
# Tableaux de coûts du rapport
# --------------------------------------------------------------------------- #

def test_les_couts_par_jambe_s_additionnent_au_total(run_avec_renforcements):
    _, _, _, _, executions = run_avec_renforcements
    table = pca.cost_breakdown(executions)
    for ligne in ("commission_dollar", "slippage_dollar", "friction_dollar", "num_fills"):
        assert table.loc[ligne, "CALL"] + table.loc[ligne, "PUT"] == pytest.approx(
            table.loc[ligne, "ALL"], rel=1e-9,
        )


def test_les_couts_par_motif_couvrent_toute_la_friction(run_avec_renforcements):
    _, _, _, _, executions = run_avec_renforcements
    par_motif = pca.cost_by_reason(executions)
    total = pca.cost_breakdown(executions).loc["friction_dollar", "ALL"]
    assert par_motif["friction_dollar"].sum() == pytest.approx(total, rel=1e-9)
    assert par_motif["pct_of_friction"].sum() == pytest.approx(100.0, rel=1e-9)


def test_les_tableaux_de_couts_tolerent_un_run_sans_journal():
    """Les runs antérieurs au journal n'ont pas executions.parquet : le
    rapport doit les omettre, pas planter."""
    vide = pd.DataFrame()
    assert pca.cost_breakdown(vide).empty
    assert pca.cost_by_reason(vide).empty
    assert pca.quantity_reconciliation(vide, pd.DataFrame()).empty
