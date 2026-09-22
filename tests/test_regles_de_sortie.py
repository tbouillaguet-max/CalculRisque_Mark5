"""Les trois sorties facultatives du moteur ACTIONS.

LE CONTEXTE QUI LES REND DÉLICATES. Le moteur ne ferme une position que sur
stop-loss ou prise de gain : un écart qui se referme ne vend pas, la ligne
devient GELÉE. C'est un choix EXPLICITE de l'utilisateur, documenté dans le
README et dans la docstring du module. Les réglages testés ici le rendent
mesurable sans le renverser -- deux d'entre eux restent désactivés par défaut,
et le test `test_les_defauts_ne_renversent_pas_la_regle_des_positions_gelees`
existe pour que cela ne change pas par inadvertance.

Ce qui a été retenu, et pourquoi : seul le stop SUIVEUR est activé (-20).
L'effet est une dose-réponse lisse -- nul à -35 où le stop ne se déclenche
jamais, croissant jusqu'à -15 -- ce qu'un pic de bruit ne produit pas, et il
est significatif sur la fenêtre de test (+0,120, p = 0,008). La sortie sur
perte de signal fait mieux encore (+0,162, p = 0,001) mais touche directement
la règle des positions gelées : elle attend une décision, pas un défaut.
"""

from __future__ import annotations

import pandas as pd
import pytest

import config
from backtest.data_loader import PricePanel
from backtest.engine import BacktestEngine
from backtest.strategies.base import Strategy


def _panel(prices: dict[str, list[float]], dates) -> PricePanel:
    close = pd.DataFrame(prices, index=dates)
    return PricePanel(close, close.copy(), close.apply(lambda c: c.last_valid_index()))


def _evt(symbol: str, published: pd.Timestamp, gap_pct: float = 100.0) -> dict:
    return {
        "symbol": symbol, "published_date": published, "fiscal_year": 2019,
        "sector": "Technologie", "close_at_filing": 100.0,
        "valuation_dcf_per_share": 200.0, "gap_pct": gap_pct, "period_type": None,
    }


class _ToutAcheter(Strategy):
    def generate_target_weights(self, signals, current_positions):
        symbols = list(signals["symbol"])
        return {s: 1.0 / len(symbols) for s in symbols} if symbols else {}


def _moteur(panel, events, **kwargs) -> BacktestEngine:
    defaults = dict(
        universe_history=None, initial_capital=1_000_000.0, cost_bps=0.0,
        stop_loss_pct=-99.0, take_profit_pct=1e6, momentum_min_pct=None,
        rebalance_band_pct=0.0, trailing_stop_pct=None,
    )
    defaults.update(kwargs)
    return BacktestEngine(
        price_panel=panel, signal_events=events, strategy=_ToutAcheter(),
        fallback_universe_symbols=set(panel.close.columns), **defaults,
    )


def _raisons(moteur) -> set[str]:
    return {t["exit_reason"] for t in moteur.trades}


# --------------------------------------------------------------------------- #
# Stop suiveur
# --------------------------------------------------------------------------- #
def test_le_stop_suiveur_ferme_sur_un_recul_depuis_le_PLUS_HAUT():
    """LE cas que le stop fixe ne voit pas : le titre monte de 60% puis rend
    presque tout. Depuis le prix d'entrée il est encore gagnant, donc le stop
    fixe ne se déclenche jamais."""
    dates = pd.bdate_range("2020-01-01", periods=30)
    cours = [100.0] * 5 + [160.0] * 5 + [110.0] * 20   # +60% puis -31% du sommet
    panel = _panel({"AAA": cours, "BBB": [100.0] * 30}, dates)
    events = pd.DataFrame([_evt("AAA", dates[1]), _evt("BBB", dates[1])])

    moteur = _moteur(panel, events, trailing_stop_pct=-25.0, stop_loss_pct=-30.0)
    moteur.run()

    assert "trailing_stop" in _raisons(moteur)
    assert "AAA" not in moteur.positions


def test_le_stop_fixe_seul_ne_verrait_rien_sur_ce_scenario():
    """Contrôle négatif du précédent : sans stop suiveur, la même trajectoire
    ne ferme rien. C'est ce qui rend le réglage utile plutôt que redondant."""
    dates = pd.bdate_range("2020-01-01", periods=30)
    panel = _panel({"AAA": [100.0] * 5 + [160.0] * 5 + [110.0] * 20, "BBB": [100.0] * 30}, dates)
    events = pd.DataFrame([_evt("AAA", dates[1]), _evt("BBB", dates[1])])

    moteur = _moteur(panel, events, trailing_stop_pct=None, stop_loss_pct=-30.0)
    moteur.run()
    assert "trailing_stop" not in _raisons(moteur)
    assert "AAA" in moteur.positions


def test_un_stop_suiveur_trop_large_ne_se_declenche_jamais():
    dates = pd.bdate_range("2020-01-01", periods=30)
    panel = _panel({"AAA": [100.0] * 5 + [160.0] * 5 + [110.0] * 20, "BBB": [100.0] * 30}, dates)
    events = pd.DataFrame([_evt("AAA", dates[1]), _evt("BBB", dates[1])])

    moteur = _moteur(panel, events, trailing_stop_pct=-50.0, stop_loss_pct=-99.0)
    moteur.run()
    assert "trailing_stop" not in _raisons(moteur)


def test_le_sommet_suit_le_cours_a_la_hausse():
    dates = pd.bdate_range("2020-01-01", periods=20)
    panel = _panel({"AAA": [100.0 + 5 * i for i in range(20)], "BBB": [100.0] * 20}, dates)
    events = pd.DataFrame([_evt("AAA", dates[1]), _evt("BBB", dates[1])])

    moteur = _moteur(panel, events, trailing_stop_pct=-10.0)
    moteur.run()
    # Aucune baisse : rien ne doit se déclencher, et le sommet doit avoir suivi.
    assert "trailing_stop" not in _raisons(moteur)
    assert moteur.positions["AAA"].peak_price >= 190.0


# --------------------------------------------------------------------------- #
# Durée de détention maximale
# --------------------------------------------------------------------------- #
def test_la_duree_maximale_ferme_une_position_immobile():
    """Une thèse qui ne s'est pas réalisée reste sinon éternellement en
    portefeuille : seuls les stops la ferment, et un cours plat n'en touche
    aucun."""
    dates = pd.bdate_range("2020-01-01", periods=60)
    panel = _panel({"AAA": [100.0] * 60, "BBB": [100.0] * 60}, dates)
    events = pd.DataFrame([_evt("AAA", dates[1]), _evt("BBB", dates[1])])

    moteur = _moteur(panel, events, max_holding_days=30)
    moteur.run()
    assert "max_holding" in _raisons(moteur)


def test_sans_duree_maximale_la_position_immobile_reste():
    dates = pd.bdate_range("2020-01-01", periods=60)
    panel = _panel({"AAA": [100.0] * 60, "BBB": [100.0] * 60}, dates)
    events = pd.DataFrame([_evt("AAA", dates[1]), _evt("BBB", dates[1])])

    moteur = _moteur(panel, events, max_holding_days=None)
    moteur.run()
    assert "max_holding" not in _raisons(moteur)
    assert set(moteur.positions) == {"AAA", "BBB"}


# --------------------------------------------------------------------------- #
# Sortie sur perte de signal : ce qui touche à la règle des positions gelées
# --------------------------------------------------------------------------- #
def test_la_sortie_sur_perte_de_signal_vend_quand_l_ecart_se_referme():
    dates = pd.bdate_range("2020-01-01", periods=40)
    panel = _panel({"AAA": [100.0] * 40, "BBB": [100.0] * 40}, dates)
    events = pd.DataFrame([
        _evt("AAA", dates[1], gap_pct=100.0), _evt("BBB", dates[1], gap_pct=100.0),
        _evt("AAA", dates[20], gap_pct=-5.0),   # la thèse s'est refermée
    ])

    moteur = _moteur(panel, events, exit_gap_threshold_pct=0.0)
    moteur.run()
    assert "signal_lost" in _raisons(moteur)
    assert "AAA" not in moteur.positions
    assert "BBB" in moteur.positions, "seule la ligne dont l'écart s'est refermé doit sortir"


def test_un_signal_PERIME_ne_declenche_pas_de_vente():
    """Distinction essentielle : la péremption GÈLE une ligne, elle ne la vend
    pas (cf. _signal_is_actionable). Vendre sur la dernière valeur connue d'un
    signal trop vieux reviendrait à agir sur une information qu'on vient de
    déclarer inutilisable."""
    dates = pd.bdate_range("2020-01-01", periods=250)
    panel = _panel({"AAA": [100.0] * 250, "BBB": [100.0] * 250}, dates)
    events = pd.DataFrame([
        _evt("AAA", dates[1], gap_pct=100.0), _evt("BBB", dates[1], gap_pct=100.0),
        _evt("AAA", dates[5], gap_pct=-5.0),
    ])

    moteur = _moteur(panel, events, exit_gap_threshold_pct=0.0, signal_max_age_days=30)
    moteur.run()
    ventes = [t for t in moteur.trades if t["exit_reason"] == "signal_lost"]
    # La vente a lieu tant que le signal est frais, pas après sa péremption.
    assert all((t["exit_date"] - dates[5]).days <= 30 for t in ventes)


def test_les_defauts_ne_renversent_pas_la_regle_des_positions_gelees():
    """LE garde-fou du module. La règle des positions gelées est un choix
    explicite de l'utilisateur : aucun de ces réglages ne doit l'annuler sans
    décision. Si ce test échoue, c'est que le défaut a changé -- pas que le
    test est faux."""
    assert config.BACKTEST_EXIT_GAP_THRESHOLD_PCT is None
    assert config.BACKTEST_MAX_HOLDING_DAYS is None


def test_le_stop_suiveur_est_actif_par_defaut():
    """Lui, en revanche, EST retenu : il ne touche pas à la règle des positions
    gelées (c'est un stop, pas une sortie sur signal) et son effet est
    significatif hors échantillon."""
    assert config.BACKTEST_TRAILING_STOP_PCT == -20.0
    dates = pd.bdate_range("2020-01-01", periods=10)
    panel = _panel({"AAA": [100.0] * 10}, dates)
    moteur = BacktestEngine(
        price_panel=panel, signal_events=pd.DataFrame([_evt("AAA", dates[1])]),
        universe_history=None, fallback_universe_symbols={"AAA"},
        strategy=_ToutAcheter(), initial_capital=1e6, cost_bps=0.0,
        stop_loss_pct=-99.0, take_profit_pct=1e6,
    )
    assert moteur.trailing_stop_pct == config.BACKTEST_TRAILING_STOP_PCT
