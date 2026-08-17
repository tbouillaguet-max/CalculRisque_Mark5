"""Journal achats/ventes DU RAPPORT (report/utils.py::build_trade_log).

Motivé par un rapport d'incohérence sur la page Stratégies : sur un même
contrat, le journal affichait 24 + 9 contrats achetés puis 3 300 « vendus ».
Deux défauts distincts, tous deux dans la MISE EN FORME et non dans le moteur
(dont la comptabilité est vérifiée par tests/test_journal_executions.py) :

    1. UNITÉS MÉLANGÉES. trades.parquet porte à la fois `contracts` et
       `shares` = contracts x 100 (le multiplicateur du contrat). Le journal
       lisait `shares` côté vente et des contrats côté achat : facteur 100
       d'écart entre les deux moitiés du même tableau.

    2. ROULEMENTS INVISIBLES. Les achats étaient reconstruits depuis les
       variations de quantité détenue, regroupées PAR SYMBOLE. Or un
       roulement (options_engine._roll_position) clôture un contrat et en
       rouvre un autre le même jour : la quantité nette bouge à peine, et
       l'achat du contrat de remplacement n'apparaissait donc pas -- alors
       que sa vente, elle, apparaissait des mois plus tard.

Le correctif est de lire executions.parquet, le journal de fills du moteur,
plutôt que de déduire les achats d'un état de portefeuille.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.options_harness import cours, moteur, serie_bruitee, signaux

utils = pytest.importorskip("report.utils", reason="le rapport Streamlit n'est pas installé")

MULTIPLIER = 100.0


@pytest.fixture(scope="module")
def run_avec_roulements():
    """Un run dont TOUTES les sorties sauf la première sont des roulements :
    c'est la configuration qui met en défaut la reconstruction par symbole.

    Cours peu volatil et stops très larges pour qu'aucun stop-loss ni
    take-profit ne vienne clôturer avant l'échéance, plafond de levier levé
    pour qu'aucun dé-levier ne réduise la ligne : il ne reste que le cycle
    « échéance proche -> on roule »."""
    panel = cours({"AAA": serie_bruitee(100.0, 500, sigma=0.008, graine=3)})
    engine = moteur(
        panel, signaux(["AAA"], "2020-01-06"), {"AAA": "CALL"},
        target_tenor_days=180, roll_when_days_left=45,
        stop_loss_pct=-95.0, take_profit_pct=500.0, exit_when_signal_lost=False,
        max_delta_notional_pct=None,
    )
    _, positions_history, trades, signals_history = engine.run()
    executions = pd.DataFrame(engine.executions)
    return engine, positions_history, trades, signals_history, executions


@pytest.fixture(scope="module")
def journal(run_avec_roulements):
    _, positions_history, trades, signals_history, executions = run_avec_roulements
    return utils.build_trade_log(positions_history, trades, signals_history, executions)


def test_le_banc_produit_bien_des_roulements(run_avec_roulements):
    """Sans roulement, ce fichier ne teste rien : le cas signalé n'apparaît
    QUE là où un contrat en remplace un autre le même jour."""
    _, _, _, _, executions = run_avec_roulements
    roulements = executions[executions["reason"] == "roll"]
    assert set(roulements["side"]) == {"buy", "sell"}
    assert len(roulements) >= 4


# --------------------------------------------------------------------------- #
# L'invariant réclamé : on ne vend pas ce qu'on n'a pas
# --------------------------------------------------------------------------- #

def test_le_solde_de_contrats_n_est_jamais_negatif(journal):
    assert (journal["solde"] >= -1e-9).all(), journal[journal["solde"] < -1e-9]


def test_chaque_contrat_est_achete_avant_d_etre_vendu(journal):
    """Vérifié CONTRAT PAR CONTRAT (sens + strike + échéance), et non symbole
    par symbole : c'est précisément la maille que la reconstruction perdait."""
    signe = journal["quantite"] * journal["action"].map({"Achat": 1.0, "Vente": -1.0})
    net = journal.assign(net=signe).groupby(["symbol", "contrat"])["net"].sum()
    assert (net >= -1e-9).all(), net[net < -1e-9]

    achats = journal[journal["action"] == "Achat"]["quantite"].sum()
    ventes = journal[journal["action"] == "Vente"]["quantite"].sum()
    assert achats >= ventes - 1e-9, (
        f"{ventes} contrats vendus pour {achats} achetés : le journal décrit "
        "des ventes à découvert que le moteur n'a pas passées"
    )


def test_le_solde_final_egale_la_position_encore_ouverte(journal, run_avec_roulements):
    engine, _, _, _, _ = run_avec_roulements
    signe = journal["quantite"] * journal["action"].map({"Achat": 1.0, "Vente": -1.0})
    net = journal.assign(net=signe).groupby("symbol")["net"].sum()
    restant = pd.Series({s: p.contracts for s, p in engine.positions.items()}, dtype=float)
    restant = restant.reindex(net.index).fillna(0.0)
    assert (net - restant).abs().max() < 1e-9


# --------------------------------------------------------------------------- #
# Les deux défauts, nommément
# --------------------------------------------------------------------------- #

def test_les_quantites_sont_en_contrats_des_deux_cotes(journal, run_avec_roulements):
    """Le facteur 100 : une vente doit être affichée en CONTRATS, alors que
    trades.parquet la porte aussi en `shares` = contrats x multiplicateur."""
    _, _, trades, _, _ = run_avec_roulements
    ventes = journal[journal["action"] == "Vente"]
    assert sorted(ventes["quantite"]) == sorted(trades["contracts"])
    assert ventes["quantite"].sum() * MULTIPLIER == pytest.approx(trades["shares"].sum())


def test_l_achat_du_contrat_de_remplacement_apparait(journal, run_avec_roulements):
    """Un roulement doit se lire comme une vente ET un achat à la même date,
    pas comme une vente esseulée."""
    _, _, _, _, executions = run_avec_roulements
    dates_de_roulement = sorted(set(executions[executions["reason"] == "roll"]["date"]))
    for date in dates_de_roulement:
        ce_jour = journal[journal["date"] == date]
        assert set(ce_jour["action"]) == {"Achat", "Vente"}, (
            f"{date:%Y-%m-%d} : roulement sans achat du contrat de remplacement"
        )
        achete = ce_jour[ce_jour["action"] == "Achat"]["quantite"].sum()
        attendu = executions[
            (executions["date"] == date) & (executions["side"] == "buy")
        ]["contracts"].sum()
        assert achete == pytest.approx(attendu)


def test_le_contrat_est_identifie_par_son_strike_et_son_echeance(journal, run_avec_roulements):
    """Sans l'échéance, deux contrats successifs d'un même roulement peuvent
    porter des strikes proches et se confondre à l'oeil."""
    _, _, _, _, executions = run_avec_roulements
    assert journal["contrat"].str.match(r"^(CALL|PUT) \d+\.\d{2} éch\. \d{4}-\d{2}-\d{2}$").all()
    # Autant de contrats distincts au journal que le moteur en a négociés.
    attendu = executions.groupby(["option_type", "strike", "expiry"]).ngroups
    assert journal["contrat"].nunique() == attendu


# --------------------------------------------------------------------------- #
# Ce que la reconstruction de repli, elle, ne peut pas faire
# --------------------------------------------------------------------------- #

def test_la_reconstruction_sans_journal_perd_les_roulements(run_avec_roulements):
    """Documente POURQUOI le journal de fills est nécessaire, et non un
    confort : sans lui, l'incohérence signalée est inévitable. Si ce test
    cessait d'échouer à équilibrer, la bannière d'avertissement de la page
    Stratégies n'aurait plus lieu d'être."""
    _, positions_history, trades, signals_history, _ = run_avec_roulements
    replis = utils.build_trade_log(positions_history, trades, signals_history)

    achats = replis[replis["action"] == "Achat"]["quantite"].sum()
    ventes = replis[replis["action"] == "Vente"]["quantite"].sum()
    assert ventes > achats, (
        "le banc ne reproduit plus le déséquilibre de la reconstruction : "
        "ce test ne protège plus rien"
    )
    # ... mais les unités, elles, sont bien corrigées sur ce chemin aussi.
    assert sorted(replis[replis["action"] == "Vente"]["quantite"]) == sorted(trades["contracts"])


def test_le_repli_actions_reste_intact():
    """Un run actions n'a pas de journal de fills (backtest/engine.py n'en
    tient pas) : la reconstruction reste sa seule source, et elle est juste --
    un symbole y désigne un seul instrument, sans roulement possible."""
    positions_history = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-06"), "symbol": "AAA", "shares": 100.0, "price": 10.0},
        {"date": pd.Timestamp("2020-01-07"), "symbol": "AAA", "shares": 150.0, "price": 11.0},
    ])
    trades = pd.DataFrame([{
        "symbol": "AAA", "entry_date": pd.Timestamp("2020-01-06"),
        "exit_date": pd.Timestamp("2020-01-08"), "shares": 150.0,
        "entry_price": 10.3, "exit_price": 12.0, "pnl": 255.0, "return_pct": 16.5,
        "holding_days": 2, "exit_reason": "stop_loss",
    }])
    log = utils.build_trade_log(positions_history, trades, pd.DataFrame())

    assert list(log["action"]) == ["Vente", "Achat", "Achat"]
    assert sorted(log["quantite"]) == [50.0, 100.0, 150.0]
    assert "solde" not in log.columns    # non calculable sans journal de fills
    assert "contrat" not in log.columns  # une action n'est pas un contrat


# --------------------------------------------------------------------------- #
# P&L des ventes
# --------------------------------------------------------------------------- #

def test_le_pnl_des_ventes_vient_de_trades_parquet(journal, run_avec_roulements):
    _, _, trades, _, _ = run_avec_roulements
    ventes = journal[journal["action"] == "Vente"]
    assert ventes["pnl"].notna().all()
    assert ventes["pnl"].sum() == pytest.approx(trades["pnl"].sum())


def test_les_achats_n_ont_ni_pnl_ni_rendement(journal):
    """Un achat n'a pas de résultat : il ouvre une position, il n'en solde
    aucune."""
    achats = journal[journal["action"] == "Achat"]
    assert achats["pnl"].isna().all()
    assert achats["return_pct"].isna().all()


def test_un_journal_desynchronise_laisse_le_pnl_vide_plutot_que_faux():
    """L'appariement vente <-> trade est positionnel (cf. _attach_sell_pnl) :
    quand l'hypothèse ne tient pas, la colonne doit rester vide."""
    executions = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-06"), "symbol": "AAA", "side": "buy",
         "option_type": "CALL", "strike": 100.0, "expiry": pd.Timestamp("2020-07-06"),
         "contracts": 10.0, "price": 5.0, "reason": "rebalance"},
        {"date": pd.Timestamp("2020-02-06"), "symbol": "AAA", "side": "sell",
         "option_type": "CALL", "strike": 100.0, "expiry": pd.Timestamp("2020-07-06"),
         "contracts": 10.0, "price": 6.0, "reason": "stop_loss"},
    ])
    trades_incoherents = pd.DataFrame([{
        "symbol": "BBB", "exit_date": pd.Timestamp("2020-02-06"),
        "contracts": 10.0, "pnl": 999.0, "return_pct": 20.0,
    }])
    log = utils.build_trade_log(pd.DataFrame(), trades_incoherents, pd.DataFrame(), executions)
    assert log["pnl"].isna().all()
    # Le reste du journal, lui, est bien rendu.
    assert list(log["solde"]) == [0.0, 10.0]
