"""Levée du coupe-circuit « une candidate neuve repèse tout ».

CE QUE CES TESTS PROTÈGENT. `engine._drift_is_material` renvoyait `True` dès
qu'une candidate absente du portefeuille dépassait le trade minimum : la zone
de non-négociation était donc court-circuitée presque tous les jours, puisque
des dépôts SEC amènent des candidates neuves 2624 séances sur 2936. Mesuré,
élargir la bande de 15 à l'infini ne changeait que 22 ventes sur 39044.

La levée est réservée à `valuation_gap_combined_ancre`. Les deux garde-fous
qui comptent sont donc :
  1. les trois stratégies actions existantes gardent le comportement EXACT
     d'avant -- c'est un changement de moteur, il pouvait tout déplacer ;
  2. la levée ne réintroduit pas le défaut latent qu'elle supprime (une
     candidate seule qui n'est jamais achetée).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from backtest.data_loader import PricePanel, build_price_panel
from backtest.engine import MIN_TRADE_DOLLAR, BacktestEngine
from backtest.strategies import STRATEGY_REGISTRY
from backtest.strategies.base import Strategy
from backtest.strategies.valuation_gap_combined import ValuationGapCombinedStrategy
from backtest.strategies.valuation_gap_combined_ancre import ValuationGapCombinedAncreeStrategy


# --------------------------------------------------------------------------- #
# Ce que déclarent les stratégies
# --------------------------------------------------------------------------- #
def test_la_strategie_ancree_est_enregistree_et_leve_le_coupe_circuit():
    assert STRATEGY_REGISTRY["valuation_gap_combined_ancre"] is ValuationGapCombinedAncreeStrategy
    assert ValuationGapCombinedAncreeStrategy.entree_neuve_force_repesage is False


def test_les_trois_strategies_existantes_gardent_le_coupe_circuit():
    """LE test de non-régression au niveau de la déclaration. Si l'une d'elles
    basculait à False par héritage mal placé, son backtest changerait sans
    qu'aucun autre test n'échoue -- les chiffres de référence du README
    deviendraient faux en silence."""
    for nom in ("valuation_gap_dcf", "valuation_gap_sector_neutral", "valuation_gap_combined"):
        assert STRATEGY_REGISTRY[nom].entree_neuve_force_repesage is True, nom


def test_le_defaut_de_la_classe_de_base_est_le_comportement_historique():
    assert Strategy.entree_neuve_force_repesage is True


def test_la_strategie_ancree_n_herite_que_de_cet_attribut():
    """Elle doit rester identique à la combinée sur tout le reste : même
    signal, même seuil, mêmes poids. Sinon la comparaison entre les deux ne
    mesure plus la seule levée du coupe-circuit."""
    ancree, combinee = ValuationGapCombinedAncreeStrategy(), ValuationGapCombinedStrategy()
    assert ancree.signal_source == combinee.signal_source == "combinee"
    assert ancree.entry_threshold_pct == combinee.entry_threshold_pct

    signaux = pd.DataFrame([
        {"symbol": s, "gap_pct": g, "sector": "Technologie",
         "published_date": pd.Timestamp("2020-06-01"), "fiscal_year": 2019,
         "close_at_filing": 100.0, "valuation_dcf_per_share": 100.0 * (1 + g / 100)}
        for s, g in {"AAA": 60.0, "BBB": 40.0, "CCC": 25.0}.items()
    ])
    assert ancree.generate_target_weights(signaux, set()) == combinee.generate_target_weights(signaux, set())


# --------------------------------------------------------------------------- #
# Le comportement du moteur
# --------------------------------------------------------------------------- #
def _cours(n: int = 120) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    lignes = []
    for symbole in ("AAA", "BBB", "CCC"):
        for d in dates:
            lignes.append({"date": d, "symbol": symbole, "open": 100.0, "close": 100.0})
    return pd.DataFrame(lignes)


class _Cibles(Strategy):
    """Stratégie de test : rend des poids fixés d'avance."""

    def __init__(self, poids: dict, force: bool = True):
        super().__init__()
        self._poids = poids
        self.entree_neuve_force_repesage = force

    def generate_target_weights(self, signals, current_positions):
        return dict(self._poids)


def _moteur(strategie, band: float, positions: dict | None = None) -> BacktestEngine:
    panel = build_price_panel(_cours())
    evenements = pd.DataFrame([{
        "symbol": "AAA", "published_date": panel.close.index[1], "fiscal_year": 2019,
        "sector": "Technologie", "close_at_filing": 100.0,
        "valuation_dcf_per_share": 150.0, "gap_pct": 50.0,
    }])
    moteur = BacktestEngine(
        price_panel=panel, signal_events=evenements, universe_history=None,
        fallback_universe_symbols={"AAA", "BBB", "CCC"},
        strategy=strategie, initial_capital=1_000_000.0, cost_bps=0.0,
        stop_loss_pct=-99.0, take_profit_pct=1e9, rebalance_band_pct=band,
    )
    if positions:
        from backtest.engine import Position
        for symbole, valeur in positions.items():
            moteur.positions[symbole] = Position(
                symbol=symbole, shares=valeur / 100.0, entry_price=100.0,
                entry_date=panel.close.index[0],
            )
        moteur.cash = 1_000_000.0 - sum(positions.values())
    return moteur


def test_avec_le_coupe_circuit_une_petite_candidate_neuve_declenche_tout():
    """Comportement HISTORIQUE, celui des trois stratégies en place : une
    candidate à 1 point de NAV franchit la zone de 15 points."""
    moteur = _moteur(_Cibles({"AAA": 0.01}, force=True), band=15.0,
                     positions={"BBB": 500_000.0})
    assert moteur._drift_is_material({"AAA": 10_000.0}, moteur.calendar[5], 1_000_000.0) is True


def test_sans_le_coupe_circuit_une_petite_candidate_neuve_ne_declenche_rien():
    """LE changement. La même candidate à 1 point de NAV reste sous la zone :
    on ne repèse pas 82 lignes pour elle."""
    moteur = _moteur(_Cibles({"AAA": 0.01}, force=False), band=15.0,
                     positions={"BBB": 500_000.0})
    # BBB vaut 500 000 et n'a pas de cible : il ne compte pas dans la dérive.
    assert moteur._drift_is_material({"AAA": 10_000.0}, moteur.calendar[5], 1_000_000.0) is False


def test_sans_le_coupe_circuit_une_grosse_candidate_neuve_declenche_encore():
    """La levée n'est pas un blocage : une candidate qui pèse plus que la zone
    la franchit toute seule, comme n'importe quel écart."""
    moteur = _moteur(_Cibles({"AAA": 0.20}, force=False), band=15.0,
                     positions={"BBB": 500_000.0})
    assert moteur._drift_is_material({"AAA": 200_000.0}, moteur.calendar[5], 1_000_000.0) is True


def test_sans_le_coupe_circuit_un_portefeuille_vide_declenche_toujours():
    """LE défaut latent que le coupe-circuit corrigeait, et qu'il faut corriger
    autrement : sans position, la dérive ne s'accumule pas -- elle vaut la
    cible de la candidate, indéfiniment. Une candidate sous le seuil ne serait
    donc JAMAIS achetée, et la zone deviendrait un filtre de SIGNAL."""
    moteur = _moteur(_Cibles({"AAA": 0.01}, force=False), band=15.0)
    assert not moteur.positions
    assert moteur._drift_is_material({"AAA": 10_000.0}, moteur.calendar[5], 1_000_000.0) is True


def test_l_amorcage_ne_s_applique_pas_au_mode_historique():
    """Garde-fou de séparation : le mode historique n'a pas besoin de
    l'amorçage (son coupe-circuit fait déjà le travail) et ne doit pas voir son
    décompte de journées sautées changer."""
    moteur = _moteur(_Cibles({}, force=True), band=15.0)
    assert moteur._drift_is_material({}, moteur.calendar[5], 1_000_000.0) is False


def test_une_cible_sous_le_trade_minimum_ne_compte_pas_dans_la_derive():
    """Poussière : une candidate à moins d'un dollar ne doit ni déclencher, ni
    gonfler la dérive -- dans les deux modes."""
    for force in (True, False):
        moteur = _moteur(_Cibles({"AAA": 1e-9}, force=force), band=15.0,
                         positions={"BBB": 500_000.0})
        assert moteur._drift_is_material(
            {"AAA": MIN_TRADE_DOLLAR / 2}, moteur.calendar[5], 1_000_000.0) is False, force


def test_une_bande_desactivee_reste_sans_effet_dans_les_deux_modes():
    for force in (True, False):
        moteur = _moteur(_Cibles({"AAA": 0.01}, force=force), band=0.0,
                         positions={"BBB": 500_000.0})
        assert moteur._drift_is_material({"AAA": 1.0}, moteur.calendar[5], 1_000_000.0) is True, force


# --------------------------------------------------------------------------- #
# Non-régression de bout en bout
# --------------------------------------------------------------------------- #
def test_un_run_complet_est_inchange_pour_une_strategie_qui_garde_le_coupe_circuit():
    """La preuve qui compte : le changement touche `_drift_is_material`, donc
    le chemin de TOUS les runs. Deux moteurs identiques -- l'un dont la
    stratégie déclare explicitement True, l'autre une classe qui n'a pas du
    tout l'attribut -- doivent rendre la même courbe, au centime près."""
    declare = _moteur(_Cibles({"AAA": 0.5, "BBB": 0.5}, force=True), band=15.0)

    class SansAttribut(Strategy):
        def generate_target_weights(self, signals, current_positions):
            return {"AAA": 0.5, "BBB": 0.5}

    # SansAttribut ne déclare rien : elle hérite du défaut de Strategy, ce qui
    # est exactement le cas à protéger -- une stratégie écrite avant ce
    # changement, et qui ne connaît pas l'attribut.
    assert "entree_neuve_force_repesage" not in SansAttribut.__dict__
    implicite = _moteur(SansAttribut(), band=15.0)

    courbe_a = declare.run()[0]
    courbe_b = implicite.run()[0]
    pd.testing.assert_frame_equal(courbe_a, courbe_b)
