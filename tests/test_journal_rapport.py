"""Journal achats/ventes du RAPPORT (report.utils.build_trade_log).

Suite du rapport d'incohérence « je vends des contrats que je n'ai pas ».
Le moteur, lui, avait déjà été mis hors de cause et instrumenté
(tests/test_journal_executions.py) : sa comptabilité est juste et son journal
des exécutions le prouve ligne à ligne. Restait la PAGE, qui n'a jamais lu ce
journal et rebâtissait les achats à partir de positions_history, avec deux
défauts qui produisaient exactement le symptôme à l'écran :

    1. les ventes s'affichaient en ACTIONS SOUS-JACENTES (trades["shares"] =
       contrats x 100) en regard d'achats en CONTRATS -- 221 contrats vendus
       s'affichaient "22100" sous un achat de "230" ;
    2. positions_history n'a de ligne que les jours de DÉTENTION : après une
       sortie totale, la ligne suivante du symbole est une position neuve, mais
       le shift(1) y voyait la quantité de l'ancienne. Une ré-entrée plus
       petite ne produisait donc AUCUN achat, et une plus grande n'en montrait
       que l'écart.
"""

from __future__ import annotations

import pandas as pd
import pytest

from report.utils import build_trade_log

MULT = 100.0


def _executions(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp(d), "symbol": s, "side": side, "option_type": ot,
             "contracts": c, "price": p, "cash_flow": (-1 if side == "buy" else 1) * c * MULT * p,
             "multiplier": MULT, "reason": r, "commission": 0.0, "slippage": 0.0}
            for d, s, side, ot, c, p, r in rows
        ]
    )


def _trades(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": s, "entry_date": pd.Timestamp(ed), "exit_date": pd.Timestamp(xd),
             # Le piège : le schéma options écrit les DEUX colonnes.
             "shares": c * MULT, "contracts": c,
             "entry_price": ep, "exit_price": xp,
             "pnl": (xp - ep) * c * MULT, "return_pct": (xp - ep) / ep * 100,
             "holding_days": 30, "exit_reason": reason,
             "option_type": ot, "strike": k, "source": "real", "open_reason": "rebalance"}
            for s, ed, xd, c, ep, xp, reason, ot, k in rows
        ]
    )


def _positions(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp(d), "symbol": s, "option_type": ot, "strike": k,
             "expiry": pd.Timestamp("2031-01-17"), "contracts": c, "entry_premium": p,
             "entry_date": pd.Timestamp(ed), "premium": p, "market_value": c * MULT * p,
             "unrealized_pnl": 0.0, "source": "real"}
            for d, s, ot, k, c, p, ed in rows
        ]
    )


# --------------------------------------------------------------------------- #
# Le chemin exact : on lit le journal des exécutions
# --------------------------------------------------------------------------- #

@pytest.fixture
def run_options():
    """Le cas signalé, réduit à l'os : une position ouverte puis RENFORCÉE,
    soldée en deux fois, puis une SECONDE position sur le même symbole, plus
    petite que ce que la première détenait."""
    executions = _executions([
        ("2010-03-02", "PWR", "buy", "PUT", 230.0, 4.35, "rebalance"),
        ("2010-06-01", "PWR", "buy", "PUT", 100.0, 3.10, "deploy_idle_cash"),
        ("2011-03-02", "PWR", "sell", "PUT", 221.0, 1.31, "rebalance"),
        ("2011-05-05", "PWR", "sell", "PUT", 109.0, 2.76, "signal_lost"),
        # Position NEUVE, plus petite que ce que détenait la précédente.
        ("2012-03-01", "PWR", "buy", "PUT", 2.0, 32.82, "rebalance"),
        ("2012-12-13", "PWR", "sell", "PUT", 2.0, 25.81, "stop_loss"),
    ])
    trades = _trades([
        ("PWR", "2010-03-02", "2011-03-02", 221.0, 4.0, 1.31, "rebalance", "PUT", 19.24),
        ("PWR", "2010-03-02", "2011-05-05", 109.0, 4.0, 2.76, "signal_lost", "PUT", 19.24),
        ("PWR", "2012-03-01", "2012-12-13", 2.0, 32.82, 25.81, "stop_loss", "PUT", 53.35708896003361),
    ])
    positions = _positions([
        ("2010-03-02", "PWR", "PUT", 19.24, 230.0, 4.35, "2010-03-02"),
        ("2010-06-01", "PWR", "PUT", 19.24, 330.0, 3.10, "2010-03-02"),
        ("2011-03-02", "PWR", "PUT", 19.24, 109.0, 1.31, "2010-03-02"),
        ("2012-03-01", "PWR", "PUT", 53.35708896003361, 2.0, 32.82, "2012-03-01"),
    ])
    return positions, trades, executions


def test_les_quantites_se_conservent(run_options):
    """L'invariant que le tableau violait : sur un symbole, ce qui est vendu a
    été acheté. Vérifié SUR CE QUI EST AFFICHÉ, pas sur les tables sources."""
    positions, trades, executions = run_options
    log = build_trade_log(positions, trades, pd.DataFrame(), executions)

    signe = log["quantite"] * log["action"].map({"Achat": 1, "Vente": -1})
    net = log.assign(net=signe).groupby("symbol")["net"].sum()
    assert (net >= -1e-9).all(), f"plus vendu qu'acheté : {net.to_dict()}"
    assert net["PWR"] == pytest.approx(0.0)


def test_les_ventes_sont_en_contrats_et_non_en_actions(run_options):
    """Le facteur 100. trades porte `shares` (= contrats x 100) ET `contracts` :
    lire `shares` affichait 22100 pour 221 contrats vendus."""
    positions, trades, executions = run_options
    log = build_trade_log(positions, trades, pd.DataFrame(), executions)

    ventes = log[log["action"] == "Vente"].set_index("date")["quantite"]
    assert ventes[pd.Timestamp("2011-03-02")] == pytest.approx(221.0)
    assert ventes[pd.Timestamp("2012-12-13")] == pytest.approx(2.0)


def test_tous_les_achats_apparaissent(run_options):
    """Y compris le renfort par le cash oisif et la ré-entrée plus petite que
    la position précédente -- les deux que la reconstruction perdait."""
    positions, trades, executions = run_options
    log = build_trade_log(positions, trades, pd.DataFrame(), executions)

    achats = log[log["action"] == "Achat"]
    assert len(achats) == 3
    assert achats.set_index("date")["quantite"][pd.Timestamp("2012-03-01")] == pytest.approx(2.0)
    assert achats["raison"].str.contains("Redéploiement du cash oisif").any()


def test_le_pnl_et_le_strike_des_ventes_viennent_de_trades(run_options):
    positions, trades, executions = run_options
    log = build_trade_log(positions, trades, pd.DataFrame(), executions)

    vente = log[(log["action"] == "Vente") & (log["date"] == pd.Timestamp("2012-12-13"))].iloc[0]
    assert vente["pnl"] == pytest.approx((25.81 - 32.82) * 2 * MULT)
    assert vente["return_pct"] is not None
    # Strike arrondi : 53.35708896003361 n'est pas une précision de contrat.
    assert "PUT 53.36" in vente["raison"]
    assert "53.357088" not in vente["raison"]


def test_les_achats_n_ont_ni_pnl_ni_rendement(run_options):
    positions, trades, executions = run_options
    log = build_trade_log(positions, trades, pd.DataFrame(), executions)
    achats = log[log["action"] == "Achat"]
    assert achats["pnl"].isna().all()
    assert achats["return_pct"].isna().all()


def test_les_motifs_sont_traduits(run_options):
    positions, trades, executions = run_options
    log = build_trade_log(positions, trades, pd.DataFrame(), executions)
    assert log["raison"].str.startswith("Stop-loss").any()
    assert log["raison"].str.startswith("Signal disparu").any()
    assert not log["raison"].str.contains("stop_loss").any()


def test_la_source_exacte_est_annoncee(run_options):
    """La page distingue les deux provenances dans sa légende : cet attribut
    est le seul moyen qu'elle a de savoir laquelle a servi."""
    positions, trades, executions = run_options
    assert build_trade_log(positions, trades, pd.DataFrame(), executions).attrs["source_exacte"] is True
    assert build_trade_log(positions, trades, pd.DataFrame(), None).attrs["source_exacte"] is False


def test_plusieurs_ventes_du_meme_symbole_le_meme_jour_gardent_leur_pnl():
    """L'appariement vente <-> trade est POSITIONNEL : deux sorties du même
    symbole le même jour (stop puis rebalancement) ne doivent pas échanger
    leur P&L ni leur strike."""
    executions = _executions([
        ("2020-01-06", "AAA", "buy", "CALL", 10.0, 5.0, "rebalance"),
        ("2020-06-01", "AAA", "sell", "CALL", 4.0, 6.0, "take_profit"),
        ("2020-06-01", "AAA", "sell", "CALL", 6.0, 3.0, "stop_loss"),
    ])
    trades = _trades([
        ("AAA", "2020-01-06", "2020-06-01", 4.0, 5.0, 6.0, "take_profit", "CALL", 90.0),
        ("AAA", "2020-01-06", "2020-06-01", 6.0, 5.0, 3.0, "stop_loss", "CALL", 90.0),
    ])
    log = build_trade_log(pd.DataFrame(), trades, pd.DataFrame(), executions)

    ventes = log[log["action"] == "Vente"]
    tp = ventes[ventes["raison"].str.startswith("Take-profit")].iloc[0]
    sl = ventes[ventes["raison"].str.startswith("Stop-loss")].iloc[0]
    assert tp["quantite"] == pytest.approx(4.0)
    assert tp["pnl"] == pytest.approx((6.0 - 5.0) * 4 * MULT)
    assert sl["quantite"] == pytest.approx(6.0)
    assert sl["pnl"] == pytest.approx((3.0 - 5.0) * 6 * MULT)


# --------------------------------------------------------------------------- #
# Le repli : runs sans journal d'exécutions (actions, runs options anciens)
# --------------------------------------------------------------------------- #

def test_le_repli_affiche_les_ventes_en_contrats(run_options):
    """Même sans executions.parquet, plus de facteur 100 : `contracts` prime
    sur `shares` quand les deux sont là."""
    positions, trades, _ = run_options
    log = build_trade_log(positions, trades, pd.DataFrame(), None)

    ventes = log[log["action"] == "Vente"].set_index("date")["quantite"]
    assert ventes[pd.Timestamp("2011-03-02")] == pytest.approx(221.0)


def test_le_repli_voit_la_reentree_comme_une_position_neuve(run_options):
    """La ré-entrée du 2012-03-01 (2 contrats) suit une position qui en
    détenait 109 : sans le découpage par entry_date, la baisse de quantité
    n'émettait aucun achat -- la vente du 2012-12-13 restait donc sans achat
    en regard."""
    positions, trades, _ = run_options
    log = build_trade_log(positions, trades, pd.DataFrame(), None)

    achats = log[log["action"] == "Achat"].set_index("date")
    assert pd.Timestamp("2012-03-01") in achats.index
    assert achats.loc[pd.Timestamp("2012-03-01"), "quantite"] == pytest.approx(2.0)
    assert "Nouvelle position" in achats.loc[pd.Timestamp("2012-03-01"), "raison"]


def test_le_repli_sur_un_run_actions_reste_en_actions():
    """La stratégie actions n'a ni contracts ni option_type : son repli doit
    continuer de fonctionner sur shares/price, sans détail de contrat."""
    positions = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-06"), "symbol": "AAA", "shares": 100.0,
         "entry_price": 10.0, "entry_date": pd.Timestamp("2020-01-06"),
         "price": 10.0, "market_value": 1000.0, "unrealized_pnl": 0.0,
         "unrealized_return_pct": 0.0},
        {"date": pd.Timestamp("2020-02-03"), "symbol": "AAA", "shares": 150.0,
         "entry_price": 10.5, "entry_date": pd.Timestamp("2020-01-06"),
         "price": 11.0, "market_value": 1650.0, "unrealized_pnl": 75.0,
         "unrealized_return_pct": 4.8},
    ])
    trades = pd.DataFrame([
        {"symbol": "AAA", "entry_date": pd.Timestamp("2020-01-06"),
         "exit_date": pd.Timestamp("2020-03-02"), "shares": 150.0, "entry_price": 10.5,
         "exit_price": 12.0, "pnl": 225.0, "return_pct": 14.3, "holding_days": 56,
         "exit_reason": "take_profit"},
    ])
    log = build_trade_log(positions, trades, pd.DataFrame(), None)

    signe = log["quantite"] * log["action"].map({"Achat": 1, "Vente": -1})
    assert signe.sum() == pytest.approx(0.0)
    assert log["raison"].str.startswith("Take-profit").any()
    # Aucun détail de contrat : une action n'a ni jambe ni strike.
    assert not log["raison"].str.contains("PUT|CALL", regex=True).any()


def test_un_run_vide_ne_casse_pas():
    log = build_trade_log(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert log.empty
    assert list(log.columns) == ["date", "symbol", "action", "quantite", "prix", "raison", "pnl", "return_pct"]
