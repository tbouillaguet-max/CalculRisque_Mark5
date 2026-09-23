"""Commission minimum, plancher relatif au NAV, et seuil de viabilité.

CE QUE CES RÉGLAGES CORRIGENT. Le coût du moteur était purement proportionnel :
un ordre de 7 $ y payait 0,7 centime, ce qu'aucun courtier ne facture. C'est ce
qui faisait passer un portefeuille de 1 000 $ pour viable -- mesuré, il l'est
jusqu'à 1,8 centime de frais fixe par ordre, et pas au-delà.

LE GARDE-FOU QUI COMPTE LE PLUS est le dernier bloc : un plancher de taille ne
doit JAMAIS s'appliquer à une liquidation. Stop-loss, take-profit, stop
suiveur, perte de signal et symbole périmé visent une cible de zéro ; leur
opposer un plancher emprisonnerait dans le portefeuille toute ligne devenue
plus petite que lui -- c'est-à-dire que le stop-loss cesserait de fonctionner
sur exactement les positions qui en ont le plus besoin, celles qui se sont
effondrées.
"""

from __future__ import annotations

import pandas as pd
import pytest

import config
from backtest.data_loader import build_price_panel
from backtest.engine import MIN_TRADE_DOLLAR, BacktestEngine, Position
from backtest.strategies.base import Strategy


def _cours(n: int = 120, prix: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame([
        {"symbol": s, "date": d, "close": prix, "open": prix, "volume": 1e6}
        for s in ("AAA", "BBB") for d in dates
    ])


class _Poids(Strategy):
    def __init__(self, poids: dict):
        super().__init__()
        self._poids = poids

    def generate_target_weights(self, signals, current_positions):
        return dict(self._poids)


def _moteur(strategie=None, capital: float = 1_000_000.0, **kwargs) -> BacktestEngine:
    panel = build_price_panel(_cours())
    evenements = pd.DataFrame([{
        "symbol": "AAA", "published_date": panel.close.index[10], "fiscal_year": 2019,
        "sector": "Technologie", "close_at_filing": 100.0,
        "valuation_dcf_per_share": 200.0, "gap_pct": 100.0,
    }])
    base = dict(
        price_panel=panel, signal_events=evenements, universe_history=None,
        fallback_universe_symbols={"AAA", "BBB"},
        strategy=strategie or _Poids({"AAA": 1.0}), initial_capital=capital,
        cost_bps=10.0, stop_loss_pct=-99.0, take_profit_pct=1e9, rebalance_band_pct=0.0,
    )
    base.update(kwargs)
    return BacktestEngine(**base)


# --------------------------------------------------------------------------- #
# Les défauts ne changent rien
# --------------------------------------------------------------------------- #
def test_les_trois_reglages_sont_neutres_par_defaut():
    """Ils touchent le chemin d'exécution de TOUS les runs : à 0, le moteur
    doit se comporter exactement comme avant, sinon les chiffres de référence
    du README deviennent faux en silence."""
    assert config.BACKTEST_MIN_COMMISSION_DOLLAR == 0.0
    assert config.BACKTEST_MIN_TRADE_PCT_OF_NAV == 0.0
    assert config.BACKTEST_MAX_FEE_PCT_OF_TRADE == 0.0

    m = _moteur()
    assert m._montant_minimal(1_000_000.0) == MIN_TRADE_DOLLAR
    assert m._taux_de_cout("AAA", 10_000.0, m.calendar[20]) == pytest.approx(10.0 / 10_000)


# --------------------------------------------------------------------------- #
# Commission minimum
# --------------------------------------------------------------------------- #
def test_la_commission_minimum_domine_les_petits_ordres():
    m = _moteur(min_commission_dollar=1.0)
    d = m.calendar[20]
    # 100 $ d'ordre : 10 bps feraient 1 centime, le minimum impose 1 $ -> 100 %.
    assert m._taux_de_cout("AAA", 100.0, d) == pytest.approx(0.01)
    # 10 000 $ : 10 bps font 10 $, bien au-dessus du minimum -> inchangé.
    assert m._taux_de_cout("AAA", 10_000.0, d) == pytest.approx(0.001)
    # Le point de bascule est exactement là où les deux s'égalisent.
    assert m._taux_de_cout("AAA", 1_000.0, d) == pytest.approx(0.001)


def test_la_commission_minimum_est_payee_a_l_aller_ET_au_retour():
    """« 1 $ aller et 1 $ retour » : le taux est calculé par EXÉCUTION, donc
    un aller-retour en paie deux."""
    m = _moteur(min_commission_dollar=1.0)
    d = m.calendar[20]
    cout_un_sens = 500.0 * m._taux_de_cout("AAA", 500.0, d)
    assert cout_un_sens == pytest.approx(1.0)
    assert 2 * cout_un_sens == pytest.approx(2.0)


def test_sans_commission_minimum_le_cout_reste_proportionnel():
    m = _moteur(min_commission_dollar=0.0)
    d = m.calendar[20]
    for notionnel in (7.0, 100.0, 1e6):
        assert m._taux_de_cout("AAA", notionnel, d) == pytest.approx(0.001)


# --------------------------------------------------------------------------- #
# Les trois planchers
# --------------------------------------------------------------------------- #
def test_le_plancher_relatif_suit_le_NAV():
    """Le plancher absolu de 1 $ ne coupe rien à l'échelle : c'est 0,000036 %
    d'un NAV de 2,8 M$. Un plancher relatif, lui, tient."""
    m = _moteur(min_trade_pct_of_nav=0.05)
    assert m._montant_minimal(10_000.0) == pytest.approx(5.0)
    assert m._montant_minimal(1_000_000.0) == pytest.approx(500.0)
    assert m._montant_minimal(8_000_000.0) == pytest.approx(4_000.0)


def test_le_seuil_de_viabilite_se_deduit_de_la_commission():
    """1 $ de commission minimum et 1 % de frais maximum tolérés : un ordre
    sous 100 $ n'a pas de raison d'exister."""
    m = _moteur(min_commission_dollar=1.0, max_fee_pct_of_trade=1.0)
    assert m._montant_minimal(1_000_000.0) == pytest.approx(100.0)

    strict = _moteur(min_commission_dollar=1.0, max_fee_pct_of_trade=0.1)
    assert strict._montant_minimal(1_000_000.0) == pytest.approx(1_000.0)


def test_le_plancher_retenu_est_le_plus_contraignant_des_trois():
    m = _moteur(min_commission_dollar=1.0, max_fee_pct_of_trade=1.0, min_trade_pct_of_nav=0.05)
    # NAV modeste : la viabilité (100 $) l'emporte sur le relatif (5 $).
    assert m._montant_minimal(10_000.0) == pytest.approx(100.0)
    # NAV élevé : le relatif (4 000 $) l'emporte sur la viabilité.
    assert m._montant_minimal(8_000_000.0) == pytest.approx(4_000.0)


def test_un_seuil_de_viabilite_sans_commission_minimum_ne_fait_rien():
    """Le seuil se DÉDUIT de la commission : sans commission fixe, il n'a rien
    à borner et ne doit pas inventer un plancher."""
    m = _moteur(max_fee_pct_of_trade=1.0)
    assert m._montant_minimal(1_000_000.0) == MIN_TRADE_DOLLAR


# --------------------------------------------------------------------------- #
# LE garde-fou : une liquidation passe toujours
# --------------------------------------------------------------------------- #
def _moteur_avec_position(valeur: float, **kwargs) -> BacktestEngine:
    m = _moteur(**kwargs)
    m.positions["BBB"] = Position(
        symbol="BBB", shares=valeur / 100.0, entry_price=100.0, entry_date=m.calendar[0],
    )
    m.cash = 1_000_000.0 - valeur
    return m


def test_une_liquidation_passe_meme_sous_le_plancher():
    """LE garde-fou. Une ligne tombée à 50 $ doit pouvoir être soldée alors que
    le plancher de viabilité est à 100 $ -- sinon un stop-loss ne peut plus
    fermer une position effondrée, et elle reste au portefeuille pour toujours."""
    m = _moteur_avec_position(50.0, min_commission_dollar=1.0, max_fee_pct_of_trade=1.0)
    assert m._montant_minimal(1_000_000.0) == pytest.approx(100.0)

    m._queue_order("BBB", 0.0, "stop_loss", m.calendar[20])
    m._execute_pending_orders(m.calendar[21])

    assert "BBB" not in m.positions, "la position n'a pas pu être soldée"
    assert len(m.trades) == 1
    assert m.trades[0]["exit_reason"] == "stop_loss"


def test_un_ajustement_sous_le_plancher_est_ignore():
    """La contrepartie : un allègement de confort, lui, est bien filtré."""
    m = _moteur_avec_position(10_000.0, min_commission_dollar=1.0, max_fee_pct_of_trade=1.0)
    # Cible à 9 950 $ : un allègement de 50 $, sous le plancher de 100 $.
    m._queue_order("BBB", 9_950.0, "rebalance", m.calendar[20])
    m._execute_pending_orders(m.calendar[21])

    assert "BBB" in m.positions
    assert len(m.trades) == 0, "un allègement sous le plancher a été exécuté"


def test_un_ajustement_au_dessus_du_plancher_passe():
    m = _moteur_avec_position(10_000.0, min_commission_dollar=1.0, max_fee_pct_of_trade=1.0)
    m._queue_order("BBB", 9_000.0, "rebalance", m.calendar[20])  # allègement de 1 000 $
    m._execute_pending_orders(m.calendar[21])

    assert len(m.trades) == 1
    assert m.trades[0]["exit_reason"] == "rebalance"


# --------------------------------------------------------------------------- #
# Comptabilité de la friction
# --------------------------------------------------------------------------- #
def test_la_friction_payee_est_totalisee():
    """Le moteur facturait sa friction sans jamais la totaliser : impossible de
    répondre à « moins de transactions, est-ce moins de frais ? ». La réponse
    n'est pas évidente -- la friction suit les DOLLARS négociés, pas le nombre
    d'ordres -- d'où la nécessité de la mesurer."""
    m = _moteur_avec_position(10_000.0)
    m._queue_order("BBB", 0.0, "stop_loss", m.calendar[20])
    m._execute_pending_orders(m.calendar[21])

    assert m.executions_count == 1
    # 10 000 $ vendus à 10 bps.
    assert m.total_friction_dollar == pytest.approx(10.0)


def test_la_commission_minimum_apparait_dans_la_friction_totale():
    """Un petit ordre paie le minimum, et le total doit le refléter -- sinon le
    compteur mesurerait une friction théorique et non celle qui a été payée."""
    m = _moteur_avec_position(200.0, min_commission_dollar=1.0)
    m._queue_order("BBB", 0.0, "stop_loss", m.calendar[20])
    m._execute_pending_orders(m.calendar[21])

    assert m.executions_count == 1
    # 200 $ à 10 bps feraient 0,20 $ : le minimum de 1 $ s'y substitue.
    assert m.total_friction_dollar == pytest.approx(1.0)


def test_le_compteur_part_de_zero_et_ne_decroit_jamais():
    m = _moteur_avec_position(10_000.0)
    assert m.total_friction_dollar == 0.0 and m.executions_count == 0
    precedent = 0.0
    for i, cible in enumerate((9_000.0, 8_000.0, 7_000.0)):
        m._queue_order("BBB", cible, "rebalance", m.calendar[20 + 2 * i])
        m._execute_pending_orders(m.calendar[21 + 2 * i])
        assert m.total_friction_dollar > precedent
        assert m.executions_count == i + 1
        precedent = m.total_friction_dollar


def test_une_candidate_inachetable_ne_declenche_pas_de_repesage():
    """Si la commission minimum rend une candidate inachetable, elle n'a aucune
    raison de forcer le repesage de tout le portefeuille pour être achetée --
    elle ne le serait pas."""
    m = _moteur_avec_position(
        500_000.0, min_commission_dollar=1.0, max_fee_pct_of_trade=1.0,
        rebalance_band_pct=15.0,
    )
    # Cible de 50 $ : sous le seuil de viabilité de 100 $.
    assert m._drift_is_material({"AAA": 50.0}, m.calendar[20], 1_000_000.0) is False
    # Cible de 200 $ : au-dessus, le coupe-circuit joue comme avant.
    assert m._drift_is_material({"AAA": 200.0}, m.calendar[20], 1_000_000.0) is True
