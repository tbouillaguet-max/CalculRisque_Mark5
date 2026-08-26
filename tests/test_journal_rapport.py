"""Journal achats/ventes du rapport (page Stratégies, onglet Options) :
conservation des quantités telle qu'elle est AFFICHÉE.

Motivé par un rapport d'incohérence : "en triant par date croissante et en
sommant les volumes achetés et vendus, j'arrive à des soldes négatifs -- plus
de ventes à un instant que je n'ai de contrats". Le moteur, lui, est juste
(tests/test_journal_executions.py vérifie déjà qu'aucun contrat n'est vendu
sans avoir été acheté) : c'est report/utils.build_trade_log qui fabriquait le
solde négatif, pour deux raisons indépendantes.

    1. UNITÉS MÉLANGÉES. trades.parquet porte `contracts` ET `shares`
       (contracts x 100) côté options ; le journal lisait `shares` en priorité
       pour les ventes et `contracts` pour les achats. Chaque vente pesait donc
       100 fois son poids réel.

    2. ACHATS MANQUANTS. Les achats étaient reconstruits en comparant la
       quantité détenue d'une ligne de positions_history à la ligne précédente
       du même symbole. Or positions_history n'enregistre que les positions
       ENCORE OUVERTES en fin de journée : la comparaison enjambe les périodes
       où la position n'existait pas, et rate tout achat qui ne fait pas monter
       le solde de fin de journée (roulement d'échéance, ré-entrée après une
       sortie complète, vente partielle suivie d'un renfort le même jour). Ces
       achats-là étaient pourtant bien vendus plus tard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "report") not in sys.path:
    sys.path.insert(0, str(ROOT / "report"))

from backtest.options_engine import OptionsBacktestEngine  # noqa: E402
from backtest.strategies.valuation_gap_multiples_options import (  # noqa: E402
    ValuationGapMultiplesOptionsStrategy,
)
from tests.options_harness import cours, moteur, serie_bruitee, signaux  # noqa: E402
from utils import build_trade_log  # noqa: E402

CAPITAL = 2_000_000.0


def _chronologique(log: pd.DataFrame) -> pd.DataFrame:
    """Le journal est rendu du plus RÉCENT au plus ancien (ordre d'affichage) :
    les contrôles de solde le remettent à l'endroit, comme le fait
    l'utilisateur en triant par date croissante."""
    return log.iloc[::-1].reset_index(drop=True)


def _soldes(log: pd.DataFrame) -> pd.DataFrame:
    """Solde recalculé À LA MAIN depuis les seules colonnes affichées -- c'est
    le geste du rapport d'incohérence, pas la colonne `solde` du journal."""
    chrono = _chronologique(log)
    signe = chrono["quantite"] * chrono["action"].map({"Achat": 1.0, "Vente": -1.0})
    chrono = chrono.assign(cumul=signe.groupby(chrono["symbol"]).cumsum())
    return chrono.groupby("symbol")["cumul"].agg(["min", "last"])


# --------------------------------------------------------------------------- #
# Un run complet : roulements, sorties totales, ré-entrées, renforcements
# --------------------------------------------------------------------------- #

def _sorties(engine) -> dict:
    _, positions_history, trades, signals_history = engine.run()
    return {
        "engine": engine,
        "positions_history": positions_history,
        "trades": trades,
        "signals_history": signals_history,
        "executions": pd.DataFrame(engine.executions),
        "detenu_a_la_fin": pd.Series(
            {s: p.contracts for s, p in engine.positions.items()}, dtype=float,
        ),
    }


@pytest.fixture(scope="module")
def run_rebalancements():
    """Sorties complètes suivies de ré-entrées, allègements de dé-levier et
    renforcements quotidiens : la moitié des chemins que la reconstruction
    ratait."""
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
    return _sorties(OptionsBacktestEngine(
        price_panel=panel, signal_events=evenements, universe_history=None,
        fallback_universe_symbols=set(panel.close.columns), option_snapshots=pd.DataFrame(),
        strategy=ValuationGapMultiplesOptionsStrategy(), initial_capital=CAPITAL,
        commission_per_contract=0.65, slippage_pct_of_premium=2.5,
        stop_loss_pct=-25.0, take_profit_pct=30.0, target_tenor_days=180,
        roll_when_days_left=60, stop_basis="underlying", exit_when_signal_lost=True,
        daily_rebalance=True, vol_mode="rolling", momentum_min_pct=None,
        max_trade_dollar=15_000.0, take_profit_convergence_fraction=0.80,
        min_deployment_pct=25.0,
    ))


@pytest.fixture(scope="module")
def run_roulements():
    """L'autre moitié : des positions portées assez longtemps pour atteindre
    leur point de décision de roulement (clôture de N contrats et réouverture
    de M le même jour). C'est le chemin où un M <= N faisait purement
    disparaître l'achat du contrat renouvelé."""
    panel = cours({
        "AAA": serie_bruitee(100.0, 400, graine=7),
        "BBB": serie_bruitee(100.0, 400, graine=11),
    })
    evenements = signaux(["AAA", "BBB"], "2020-01-02", theoretical=180.0)
    return _sorties(moteur(
        panel, evenements, {"AAA": "CALL", "BBB": "CALL"},
        vol_mode="rolling", min_deployment_pct=40.0,
    ))


@pytest.fixture(params=["rebalancements", "roulements"])
def run_options(request, run_rebalancements, run_roulements):
    return {"rebalancements": run_rebalancements, "roulements": run_roulements}[request.param]


@pytest.fixture(params=["avec_journal", "sans_journal"])
def journal(request, run_options):
    """Les deux sources d'achats doivent tenir le même invariant : le journal
    des exécutions du moteur options (exact, fill par fill) et la
    reconstruction depuis positions_history, seule disponible pour la stratégie
    actions et pour les runs options antérieurs au journal."""
    executions = run_options["executions"] if request.param == "avec_journal" else None
    return build_trade_log(
        run_options["positions_history"], run_options["trades"],
        run_options["signals_history"], executions,
    )


def test_le_solde_ne_devient_jamais_negatif(journal):
    """L'incohérence signalée. Le backtest est non margé : il n'y a aucune
    vente à découvert, donc aucun solde cumulé ne peut passer sous zéro."""
    negatifs = _soldes(journal)[lambda d: d["min"] < -1e-6]
    assert negatifs.empty, f"soldes négatifs :\n{negatifs}"


def test_le_solde_final_est_la_position_reellement_detenue(journal, run_options):
    """Plus fort que la seule positivité : achats - ventes doit tomber
    EXACTEMENT sur ce que le moteur détient encore à la fin du run."""
    fin = _soldes(journal)["last"]
    attendu = run_options["detenu_a_la_fin"].reindex(fin.index).fillna(0.0)
    assert (fin - attendu).abs().max() < 1e-6, (fin - attendu)


def test_la_colonne_solde_egale_le_cumul_recalcule(journal):
    """La colonne `solde` affichée est bien le cumul que l'utilisateur
    referait à la main -- sinon elle donnerait une fausse assurance."""
    chrono = _chronologique(journal)
    signe = chrono["quantite"] * chrono["action"].map({"Achat": 1.0, "Vente": -1.0})
    attendu = signe.groupby(chrono["symbol"]).cumsum()
    assert (chrono["solde"] - attendu).abs().max() < 1e-6


def test_les_ventes_sont_comptees_en_contrats_et_non_en_actions(journal, run_options):
    """La cause n°1 : trades.parquet porte `shares` = contracts x 100 côté
    options, et le journal lisait `shares`. Un facteur 100 sur les seules
    ventes suffisait à rendre tous les soldes négatifs."""
    ventes = journal[journal["action"] == "Vente"]["quantite"].sum()
    assert ventes == pytest.approx(run_options["trades"]["contracts"].sum())
    # Et le piège est bien présent dans les données : `shares` vaut 100 fois
    # plus. Si ce n'était plus le cas, ce test ne protégerait plus rien.
    assert run_options["trades"]["shares"].sum() == pytest.approx(ventes * 100)


def test_les_achats_totalisent_ce_que_le_moteur_a_reellement_achete(journal, run_options):
    """La cause n°2 : les achats ratés par la reconstruction. Référence = le
    journal des exécutions du moteur, qui les enregistre tous."""
    executions = run_options["executions"]
    attendu = executions.loc[executions["side"] == "buy", "contracts"].sum()
    achats = journal[journal["action"] == "Achat"]["quantite"].sum()
    assert achats == pytest.approx(attendu)


def test_le_banc_des_roulements_en_declenche_vraiment(run_roulements):
    """Sans roulement, ce banc ne prouverait rien : c'est le chemin qui faisait
    disparaître l'achat du contrat renouvelé."""
    assert (run_roulements["executions"]["reason"] == "roll").any()


def test_le_banc_des_rebalancements_solde_vraiment_des_positions(run_rebalancements):
    """Idem pour l'autre chemin : une position soldée en entier ne laisse plus
    aucune ligne dans positions_history, et c'est ce trou que l'ancienne
    reconstruction enjambait."""
    executions = run_rebalancements["executions"]
    signe = executions["contracts"] * executions["side"].map({"buy": 1.0, "sell": -1.0})
    cumul = signe.groupby(executions["symbol"]).cumsum()
    assert (cumul.abs() < 1e-9).any(), "aucune sortie complète dans le run"


def test_l_ancienne_reconstruction_produisait_bien_des_soldes_negatifs(run_options):
    """Reproduit le bug corrigé, pour que la régression soit détectable : la
    comparaison à la ligne PRÉCÉDENTE du même symbole (sans réintégrer les
    ventes du jour, et en enjambant les périodes sans position) rate assez
    d'achats pour faire plonger le cumul sous zéro."""
    ph = run_options["positions_history"].sort_values(["symbol", "date"]).copy()
    ph["prev"] = ph.groupby("symbol")["contracts"].shift(1).fillna(0.0)
    achats = (ph["contracts"] - ph["prev"]).clip(lower=0.0).groupby(ph["symbol"]).sum()
    ventes = run_options["trades"].groupby("symbol")["contracts"].sum()
    manquants = (ventes - achats.reindex(ventes.index).fillna(0.0)).max()
    assert manquants > 0, (
        "l'ancienne reconstruction ne rate plus aucun achat sur ce banc : "
        "il ne reproduit plus le cas signalé"
    )


# --------------------------------------------------------------------------- #
# Cas isolés, sur des tables minimales : chaque chemin que la reconstruction
# ratait, sans avoir à faire tourner un moteur complet.
# --------------------------------------------------------------------------- #

def _ph(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp(d), "symbol": s, "option_type": "CALL", "strike": k,
             "contracts": float(n), "premium": 5.0}
            for d, s, k, n in rows
        ]
    )


def _trades(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": s, "entry_date": pd.Timestamp("2020-01-01"), "exit_date": pd.Timestamp(d),
             "shares": float(n) * 100, "contracts": float(n), "exit_price": 4.0,
             "exit_reason": r, "option_type": "CALL", "strike": 100.0,
             "pnl": 0.0, "return_pct": 0.0}
            for d, s, n, r in rows
        ]
    )


def test_un_roulement_ne_fait_pas_disparaitre_l_achat_du_nouveau_contrat():
    """Le moteur solde 10 contrats et en rouvre 6 le MÊME JOUR (échéance
    pleine, exposition $ inchangée mais prime plus chère). La quantité de fin
    de journée BAISSE : l'ancienne reconstruction n'y voyait aucun achat, et
    les 6 contrats rouverts étaient vendus plus tard sans avoir jamais été
    achetés."""
    positions = _ph([
        ("2020-01-02", "AAA", 100.0, 10),
        ("2020-01-03", "AAA", 120.0, 6),     # jour du roulement : nouveau contrat
        ("2020-01-06", "AAA", 120.0, 6),
    ])
    trades = _trades([("2020-01-03", "AAA", 10, "roll"), ("2020-01-07", "AAA", 6, "expiry")])

    log = build_trade_log(positions, trades, pd.DataFrame())
    achats = log[log["action"] == "Achat"]
    assert achats["quantite"].sum() == pytest.approx(16.0)   # 10 à l'ouverture + 6 au roulement
    assert _soldes(log).loc["AAA", "min"] >= -1e-9
    assert _soldes(log).loc["AAA", "last"] == pytest.approx(0.0)


def test_une_re_entree_apres_sortie_complete_est_bien_comptee():
    """Position soldée en entier, puis rouverte des semaines plus tard.
    positions_history ne garde aucune ligne entre les deux : l'ancienne
    reconstruction comparait la ré-entrée à la quantité d'AVANT la sortie et
    la prenait pour un allègement."""
    positions = _ph([
        ("2020-01-02", "AAA", 100.0, 20),
        ("2020-01-03", "AAA", 100.0, 20),
        ("2020-02-03", "AAA", 100.0, 8),     # ré-entrée, plus petite
        ("2020-02-04", "AAA", 100.0, 8),
    ])
    trades = _trades([("2020-01-06", "AAA", 20, "stop_loss"), ("2020-02-05", "AAA", 8, "take_profit")])

    log = build_trade_log(positions, trades, pd.DataFrame())
    assert log[log["action"] == "Achat"]["quantite"].sum() == pytest.approx(28.0)
    assert _soldes(log).loc["AAA", "min"] >= -1e-9
    assert _soldes(log).loc["AAA", "last"] == pytest.approx(0.0)


def test_une_vente_partielle_suivie_d_un_renfort_le_meme_jour_est_comptee():
    """Solde de fin de journée inchangé (12 vendus, 12 rachetés) : la variation
    quotidienne est nulle, l'achat n'existait donc pas pour l'ancienne
    reconstruction."""
    positions = _ph([
        ("2020-01-02", "AAA", 100.0, 30),
        ("2020-01-03", "AAA", 100.0, 30),
        ("2020-01-06", "AAA", 100.0, 30),
    ])
    trades = _trades([("2020-01-03", "AAA", 12, "delever"), ("2020-01-07", "AAA", 30, "expiry")])

    log = build_trade_log(positions, trades, pd.DataFrame())
    assert log[log["action"] == "Achat"]["quantite"].sum() == pytest.approx(42.0)
    assert _soldes(log).loc["AAA", "min"] >= -1e-9
    assert _soldes(log).loc["AAA", "last"] == pytest.approx(0.0)


def test_le_journal_des_executions_prime_sur_la_reconstruction():
    """Quand le run a executions.parquet, les achats en sont lus tels quels --
    y compris ceux qu'aucune variation de positions_history ne trahit."""
    positions = _ph([("2020-01-02", "AAA", 100.0, 10), ("2020-01-03", "AAA", 120.0, 6)])
    trades = _trades([("2020-01-03", "AAA", 10, "roll")])
    executions = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-02"), "symbol": "AAA", "side": "buy", "option_type": "CALL",
         "contracts": 10.0, "price": 5.0, "reason": "rebalance"},
        {"date": pd.Timestamp("2020-01-03"), "symbol": "AAA", "side": "sell", "option_type": "CALL",
         "contracts": 10.0, "price": 4.0, "reason": "roll"},
        {"date": pd.Timestamp("2020-01-03"), "symbol": "AAA", "side": "buy", "option_type": "CALL",
         "contracts": 6.0, "price": 7.0, "reason": "roll"},
    ])

    log = build_trade_log(positions, trades, pd.DataFrame(), executions)
    achats = log[log["action"] == "Achat"]
    assert list(achats["quantite"]) == [6.0, 10.0]           # rendu du plus récent au plus ancien
    assert _soldes(log).loc["AAA", "last"] == pytest.approx(6.0)
    # Le motif du moteur est repris tel quel, au lieu du générique
    # "Renforcement (rebalancement)" que produisait la reconstruction.
    assert "Roulement d'échéance" in achats.iloc[0]["raison"]


def test_une_ouverture_est_rattachee_au_dernier_signal_connu():
    positions = _ph([("2020-01-06", "AAA", 100.0, 10)])
    signals = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-02"), "symbol": "AAA", "gap_pct": 42.0},
        {"date": pd.Timestamp("2020-06-01"), "symbol": "AAA", "gap_pct": -80.0},   # postérieur
    ])
    log = build_trade_log(positions, pd.DataFrame(), signals)
    assert "sous-évaluation" in log.iloc[0]["raison"]
    assert "42.0" in log.iloc[0]["raison"]


# --------------------------------------------------------------------------- #
# Schéma actions et cas dégénérés
# --------------------------------------------------------------------------- #

def test_le_schema_actions_reste_lu_en_actions():
    """La stratégie actions n'a ni `contracts` ni journal des exécutions : ses
    quantités sont des ACTIONS des deux côtés, et doivent le rester."""
    positions = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-02"), "symbol": "AAA", "shares": 150.0, "price": 20.0},
        {"date": pd.Timestamp("2020-01-03"), "symbol": "AAA", "shares": 150.0, "price": 21.0},
    ])
    trades = pd.DataFrame([{
        "symbol": "AAA", "entry_date": pd.Timestamp("2020-01-02"),
        "exit_date": pd.Timestamp("2020-01-06"), "shares": 150.0, "exit_price": 22.0,
        "exit_reason": "take_profit", "pnl": 300.0, "return_pct": 10.0,
    }])

    log = build_trade_log(positions, trades, pd.DataFrame())
    assert log[log["action"] == "Achat"]["quantite"].sum() == pytest.approx(150.0)
    assert log[log["action"] == "Vente"]["quantite"].sum() == pytest.approx(150.0)
    assert _soldes(log).loc["AAA", "last"] == pytest.approx(0.0)


def test_un_run_sans_aucune_execution_donne_un_journal_vide():
    log = build_trade_log(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert log.empty
    assert "solde" in log.columns
