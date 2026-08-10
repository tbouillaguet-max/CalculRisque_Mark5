"""Instrumentation de la friction, plafond par ordre, take-profit par
convergence, score symétrique et neutralité directionnelle de la file."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import config
from backtest.options_engine import OptionPosition
from backtest.strategies.base import inflation_adjusted_log_gap
from backtest.strategies.valuation_gap_multiples_options import ValuationGapMultiplesOptionsStrategy
from tests.options_harness import cours, moteur, serie_bruitee, signaux

N_JOURS = 150


@pytest.fixture
def panel():
    return cours({
        "AAA": serie_bruitee(100.0, N_JOURS, graine=1),
        "BBB": serie_bruitee(80.0, N_JOURS, graine=2),
    })


@pytest.fixture
def evenements():
    return signaux(["AAA", "BBB"], "2020-01-06")


# --------------------------------------------------------------------------- #
# Friction : total_commission / total_slippage
# --------------------------------------------------------------------------- #

def test_la_friction_est_publiee_dans_l_equity_curve(panel, evenements):
    engine = moteur(
        panel, evenements, {"AAA": "CALL", "BBB": "PUT"},
        slippage_pct_of_premium=5.0, min_deployment_pct=60.0,
    )
    equity_curve, positions_history, _, _ = engine.run()

    assert {"total_commission", "total_slippage"} <= set(equity_curve.columns)
    # Cumuls : monotones (la friction ne se rembourse pas) et non nuls dès la
    # première entrée -- inutile d'attendre une sortie, l'achat coûte déjà.
    assert not positions_history.empty
    assert (equity_curve["total_commission"].diff().dropna() >= -1e-9).all()
    assert (equity_curve["total_slippage"].diff().dropna() >= -1e-9).all()
    assert equity_curve["total_commission"].iloc[-1] > 0
    assert equity_curve["total_slippage"].iloc[-1] > 0


def test_sans_slippage_le_compteur_de_slippage_reste_nul(panel, evenements):
    """C'est ce run qui sert de référence "pure thèse" : si le compteur ne
    tombait pas à zéro, la comparaison avec le run frotté ne mesurerait rien."""
    engine = moteur(
        panel, evenements, {"AAA": "CALL", "BBB": "PUT"},
        slippage_pct_of_premium=0.0, min_deployment_pct=60.0,
    )
    equity_curve, _, _, _ = engine.run()
    assert equity_curve["total_slippage"].iloc[-1] == pytest.approx(0.0)
    assert equity_curve["total_commission"].iloc[-1] > 0    # les frais, eux, restent


def test_le_slippage_est_compte_a_l_achat_ET_a_la_vente(panel, evenements):
    """Le moteur paie la prime majorée à l'achat et l'encaisse minorée à la
    vente : ne compter qu'un côté sous-estimerait la friction de moitié."""
    engine = moteur(panel, evenements, {"AAA": "CALL"}, slippage_pct_of_premium=10.0)
    engine.positions["AAA"] = OptionPosition(
        symbol="AAA", option_type="CALL", strike=50.0, expiry=panel.close.index[-1],
        contracts=10.0, entry_premium=5.0, entry_date=panel.close.index[0],
        vol=0.25, multiplier=100.0, source="simulated", entry_spot=100.0,
        stop_reference_premium=5.0, stop_reference_spot=100.0,
    )
    engine.total_slippage = 0.0
    engine._close_position("AAA", panel.close.index[10], "take_profit")
    assert engine.total_slippage > 0, "une VENTE seule n'a produit aucun slippage"


def test_la_friction_apparait_dans_les_diagnostics(panel, evenements):
    engine = moteur(panel, evenements, {"AAA": "CALL"}, slippage_pct_of_premium=5.0)
    engine.run()
    diagnostics = engine.execution_diagnostics()
    assert diagnostics["total_friction_dollar"] == pytest.approx(
        diagnostics["total_commission_dollar"] + diagnostics["total_slippage_dollar"]
    )
    assert diagnostics["total_friction_pct_of_initial"] > 0


# --------------------------------------------------------------------------- #
# Plafond par ordre
# --------------------------------------------------------------------------- #

def test_aucun_ordre_d_achat_ne_depasse_le_plafond(panel, evenements):
    plafond = 15_000.0
    engine = moteur(
        panel, evenements, {"AAA": "CALL", "BBB": "PUT"},
        max_trade_dollar=plafond, min_deployment_pct=90.0,
    )
    cash_avant = [engine.cash]
    achats = []
    vrai_open = type(engine)._open_or_resize

    def espion(self, *args, **kwargs):
        avant = self.cash
        result = vrai_open(self, *args, **kwargs)
        if self.cash < avant:
            achats.append(avant - self.cash)
        return result

    type(engine)._open_or_resize = espion
    try:
        engine.run()
    finally:
        type(engine)._open_or_resize = vrai_open

    assert achats, "aucun achat exécuté : le test ne prouve rien"
    assert max(achats) <= plafond + 1e-6, f"ordre à {max(achats):.2f}$ au-dessus du plafond"
    assert cash_avant  # garde-fou de lisibilité


def test_le_plafond_desactive_laisse_passer_de_plus_gros_ordres(panel, evenements):
    """Sans ce contrôle, le test ci-dessus passerait même si aucun ordre
    n'avait jamais approché le plafond."""
    def plus_gros_achat(max_trade_dollar):
        engine = moteur(
            panel, evenements, {"AAA": "CALL", "BBB": "PUT"},
            max_trade_dollar=max_trade_dollar, min_deployment_pct=90.0,
        )
        achats = []
        vrai_open = type(engine)._open_or_resize

        def espion(self, *args, **kwargs):
            avant = self.cash
            result = vrai_open(self, *args, **kwargs)
            if self.cash < avant:
                achats.append(avant - self.cash)
            return result

        type(engine)._open_or_resize = espion
        try:
            engine.run()
        finally:
            type(engine)._open_or_resize = vrai_open
        return max(achats) if achats else 0.0

    assert plus_gros_achat(None) > plus_gros_achat(15_000.0)


def test_le_plafond_ne_bride_jamais_une_vente(panel, evenements):
    """Plafonner une sortie interdirait de liquider une position devenue
    grosse -- exactement quand il faut pouvoir sortir."""
    engine = moteur(panel, evenements, {"AAA": "CALL"}, max_trade_dollar=100.0)
    engine.positions["AAA"] = OptionPosition(
        symbol="AAA", option_type="CALL", strike=10.0, expiry=panel.close.index[-1],
        contracts=500.0, entry_premium=5.0, entry_date=panel.close.index[0],
        vol=0.25, multiplier=100.0, source="simulated", entry_spot=100.0,
        stop_reference_premium=5.0, stop_reference_spot=100.0,
    )
    cash_avant = engine.cash
    engine._close_position("AAA", panel.close.index[10], "stop_loss")
    assert "AAA" not in engine.positions
    assert engine.cash - cash_avant > 100.0, "la vente a été bridée par le plafond d'achat"


# --------------------------------------------------------------------------- #
# Take-profit par fraction de convergence
# --------------------------------------------------------------------------- #

def _position_convergence(engine, panel, option_type, theorique, spot_entree=100.0):
    pos = OptionPosition(
        symbol="AAA", option_type=option_type, strike=(theorique + spot_entree) / 2,
        expiry=panel.close.index[-1], contracts=1.0, entry_premium=5.0,
        entry_date=panel.close.index[0], vol=0.25, multiplier=100.0, source="simulated",
        entry_spot=spot_entree, stop_reference_premium=5.0, stop_reference_spot=spot_entree,
        strike_reference_price=theorique,
    )
    engine.positions["AAA"] = pos
    return pos


@pytest.mark.parametrize(
    "option_type, theorique, cours_du_jour, fraction_attendue",
    [
        # CALL entré à 100 avec une théorique à 125 : la convergence complète
        # vaut +25% de cours, 80% du chemin vaut 120.
        ("CALL", 125.0, 100.0, 0.0),
        ("CALL", 125.0, 120.0, 0.8),
        ("CALL", 125.0, 125.0, 1.0),
        # PUT entré à 100 avec une théorique à 83,33 (cours à 120% de V) : la
        # convergence complète vaut -16,7%, 80% du chemin vaut 86,67.
        ("PUT", 100 / 1.2, 100.0, 0.0),
        ("PUT", 100 / 1.2, 100 - 0.8 * (100 - 100 / 1.2), 0.8),
        ("PUT", 100 / 1.2, 100 / 1.2, 1.0),
    ],
)
def test_la_fraction_de_convergence_est_orientee_dans_les_deux_sens(
    panel, evenements, option_type, theorique, cours_du_jour, fraction_attendue,
):
    """Le point de la manoeuvre : la même fraction décrit le même degré
    d'avancement de la thèse pour un call comme pour un put, alors qu'un
    seuil fixe en % du sous-jacent, lui, n'est pas atteignable pareil des
    deux côtés."""
    dates = panel.close.index
    close = pd.DataFrame({"AAA": [cours_du_jour] * len(dates)}, index=dates)
    from backtest.data_loader import PricePanel

    plat = PricePanel(close, close.copy(), close.apply(lambda c: c.last_valid_index()))
    engine = moteur(plat, signaux(["AAA"], "2020-01-06"), {"AAA": option_type})
    pos = _position_convergence(engine, plat, option_type, theorique)
    assert engine._convergence_fraction(pos, dates[10]) == pytest.approx(fraction_attendue, abs=1e-6)


def test_le_take_profit_de_convergence_se_declenche_pour_un_put(panel, evenements):
    """Le cas qui motive le changement : au seuil d'entrée, la convergence
    complète d'un put ne vaut que -16,7% de cours -- un take-profit fixe à
    +25% ou +30% exigeait un dépassement de la cible et ne partait jamais."""
    from backtest.data_loader import PricePanel

    dates = panel.close.index
    theorique = 100 / 1.2
    cours_a_80pct = 100 - 0.8 * (100 - theorique)
    close = pd.DataFrame({"AAA": [cours_a_80pct] * len(dates)}, index=dates)
    plat = PricePanel(close, close.copy(), close.apply(lambda c: c.last_valid_index()))

    engine = moteur(
        plat, signaux(["AAA"], "2020-01-06"), {"AAA": "PUT"},
        take_profit_pct=30.0, take_profit_convergence_fraction=0.80,
    )
    _position_convergence(engine, plat, "PUT", theorique)
    declenches = engine._check_stop_loss_take_profit(dates[10])

    assert "AAA" in declenches
    assert engine.pending_orders["AAA"].reason == "take_profit"
    # Le seuil fixe, lui, n'aurait rien fait : -13,3% de cours pour un put,
    # c'est +13,3% dans le sens de la position, loin des +30% demandés.
    assert engine._position_move_pct(engine.positions["AAA"], dates[10]) < 30.0


def test_une_position_atm_garde_le_take_profit_en_pourcentage(panel, evenements):
    """valuation_gap_options ne passe pas de strike_reference_price : son
    take-profit ne doit pas changer de nature au passage."""
    engine = moteur(panel, evenements, {"AAA": "CALL"}, take_profit_convergence_fraction=0.80)
    pos = OptionPosition(
        symbol="AAA", option_type="CALL", strike=100.0, expiry=panel.close.index[-1],
        contracts=1.0, entry_premium=5.0, entry_date=panel.close.index[0],
        vol=0.25, multiplier=100.0, source="simulated", entry_spot=100.0,
        stop_reference_premium=5.0, stop_reference_spot=100.0,
    )
    assert engine._convergence_fraction(pos, panel.close.index[10]) is None


def test_un_ecart_d_entree_degenere_ne_declenche_pas_de_prise_de_gain(panel, evenements):
    """Théorique confondue avec le cours d'entrée : la fraction serait une
    division par zéro. Peut arriver après un roulement, qui repose la
    référence sur une théorique rafraîchie."""
    engine = moteur(panel, evenements, {"AAA": "CALL"}, take_profit_convergence_fraction=0.80)
    pos = _position_convergence(engine, panel, "CALL", theorique=100.0, spot_entree=100.0)
    assert engine._convergence_fraction(pos, panel.close.index[10]) is None


# --------------------------------------------------------------------------- #
# Score symétrique
# --------------------------------------------------------------------------- #

def _signal(theorique: float, close: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "symbol": "AAA", "published_date": pd.Timestamp("2020-01-06"), "fiscal_year": 2019,
        "sector": "Technologie", "close": close, "valuation_theoretical_per_share": theorique,
        "source": "multiples", "gap_pct": 0.0, "period_type": "FY",
    }])


@pytest.fixture
def sans_inflation(monkeypatch):
    """DEUX asymétries se superposent dans le score, et il ne faut pas les
    confondre :

    1. Celle de la CONVENTION de calcul (bases "theoretical"/"close"), qui
       n'a aucun contenu économique -- c'est elle que le passage en log
       supprime, et c'est elle que testent les tests qui prennent ce banc.
    2. Celle de l'INFLATION, qui en a un : la valeur théorique étant
       nominale, la dérive des prix aide un call et pénalise un put. Elle
       DOIT subsister, et le test ci-dessous la fixe explicitement.

    Neutraliser (2) est donc le seul moyen d'observer (1) isolément."""
    monkeypatch.setattr(config, "INFLATION_ADJUST_GAP", False)


def test_moitie_prix_et_double_prix_donnent_la_meme_conviction(sans_inflation):
    """Le coeur du point 3 : deux erreurs de valorisation d'ampleur égale au
    signe près doivent produire un score égal au signe près."""
    strategie = ValuationGapMultiplesOptionsStrategy()
    sous_evaluee = strategie._candidates(_signal(theorique=200.0, close=100.0))
    survalorisee = strategie._candidates(_signal(theorique=100.0, close=200.0))

    assert sous_evaluee["_gap"].iloc[0] == pytest.approx(-survalorisee["_gap"].iloc[0], rel=1e-9)
    assert sous_evaluee["_abs_gap"].iloc[0] == pytest.approx(survalorisee["_abs_gap"].iloc[0], rel=1e-9)


def test_le_seuil_filtre_les_deux_sens_a_identite(sans_inflation):
    """Le biais de SÉLECTION, distinct du biais de classement : à 20% en base
    théorique, un put qualifiait à P/V >= 1,20 là où un call exigeait
    P/V <= 0,80 -- le put était structurellement plus facile à retenir."""
    strategie = ValuationGapMultiplesOptionsStrategy()
    # Juste au-dessus du seuil des deux côtés (ratio 1,21 et son inverse).
    assert not strategie._candidates(_signal(theorique=121.0, close=100.0)).empty
    assert not strategie._candidates(_signal(theorique=100.0, close=121.0)).empty
    # Juste en dessous des deux côtés (ratio 1,19 et son inverse).
    assert strategie._candidates(_signal(theorique=119.0, close=100.0)).empty
    assert strategie._candidates(_signal(theorique=100.0, close=119.0)).empty


def test_l_inflation_reste_une_asymetrie_voulue_par_dessus_le_score_symetrique():
    """Le score log est symétrique, l'ajustement d'inflation ne l'est pas --
    et c'est le comportement recherché : sur 2 ans de dérive nominale, un put
    doit voir sa thèse durcie et un call assouplie. Ce test empêche de
    "corriger" cette asymétrie-là en croyant achever le passage au log."""
    strategie = ValuationGapMultiplesOptionsStrategy()
    call = strategie._candidates(_signal(theorique=200.0, close=100.0))["_gap"].iloc[0]
    put = strategie._candidates(_signal(theorique=100.0, close=200.0))["_gap"].iloc[0]

    assert call > abs(put), "l'inflation doit favoriser le call"
    # Décalage uniforme : les deux scores sont déplacés du même montant, ce qui
    # préserve les ÉCARTS entre candidats du même côté.
    assert call - 100 * math.log(2) == pytest.approx(put + 100 * math.log(2), rel=1e-9)


def test_l_ancienne_base_theorique_reste_rejouable(sans_inflation):
    """Conservée pour pouvoir comparer un run à l'ancien comportement -- et
    ce test fixe l'asymétrie qu'elle porte, pour qu'on ne la reprenne pas
    par mégarde comme défaut."""
    ancienne = ValuationGapMultiplesOptionsStrategy(gap_basis="theoretical", entry_threshold_pct=20.0)
    # P/V = 1,20 : retenu en base théorique (écart -20,0%)...
    assert not ancienne._candidates(_signal(theorique=100.0, close=120.0)).empty
    # ...alors que son symétrique exact (V/P = 1,20) ne l'est pas (+16,7%).
    assert ancienne._candidates(_signal(theorique=120.0, close=100.0)).empty


def test_gap_basis_inconnu_est_refuse():
    with pytest.raises(ValueError, match="gap_basis"):
        ValuationGapMultiplesOptionsStrategy(gap_basis="racine_carree")


def test_le_seuil_par_defaut_vaut_bien_vingt_pourcent_de_rapport():
    assert config.OPTIONS_MULTIPLES_ENTRY_THRESHOLD_PCT == pytest.approx(math.log(1.20) * 100)
    assert config.OPTIONS_MULTIPLES_GAP_BASIS == "log"


def test_la_correction_d_inflation_est_additive_en_log():
    """En log, la convergence vers V x (1+pi)^T se traduit par un simple
    ajout de T x ln(1+pi) -- appliquer la formule en pourcentage traiterait
    18,23 points de log comme un écart de 18,23%."""
    gap = pd.Series([0.0, 50.0])
    dates = pd.Series([pd.Timestamp("2020-01-06")] * 2)
    corrige = inflation_adjusted_log_gap(gap, dates, horizon_years=2.0, enabled=True)

    inflation = config.inflation_known_at(pd.Timestamp("2020-01-06")) / 100
    attendu = np.log1p(inflation) * 2.0 * 100
    assert (corrige - gap).round(9).nunique() == 1, "la correction doit être un décalage uniforme"
    assert (corrige - gap).iloc[0] == pytest.approx(attendu)


def test_l_inflation_desactivee_laisse_le_score_intact():
    gap = pd.Series([10.0, -10.0])
    dates = pd.Series([pd.Timestamp("2020-01-06")] * 2)
    assert list(inflation_adjusted_log_gap(gap, dates, 2.0, enabled=False)) == [10.0, -10.0]


# --------------------------------------------------------------------------- #
# Neutralité directionnelle de la file d'exécution
# --------------------------------------------------------------------------- #

def _file(engine, ordres: dict[str, tuple[float, str]]):
    """ordres : {symbole: (target_dollar, sens)}."""
    from backtest.options_engine import _PendingOrder

    for symbol, (montant, sens) in ordres.items():
        engine.pending_orders[symbol] = _PendingOrder(montant, "rebalance", pd.Timestamp("2020-01-06"))
        engine._pending_spec[symbol] = {"option_type": sens}


def test_les_achats_alternent_call_et_put(panel, evenements):
    """Sans alternance, l'ordre d'exécution suivait le classement par
    conviction : quand le cash s'épuise, _affordable tronque les derniers
    servis, donc une direction systématiquement en tête finance tout et
    l'autre absorbe toute la troncature."""
    engine = moteur(panel, evenements, {"AAA": "CALL"})
    _file(engine, {
        "C1": (900.0, "CALL"), "C2": (800.0, "CALL"), "C3": (700.0, "CALL"),
        "P1": (500.0, "PUT"), "P2": (400.0, "PUT"),
    })
    assert [s for s, _ in engine._execution_order()] == ["C1", "P1", "C2", "P2", "C3"]


def test_les_ventes_passent_avant_tous_les_achats(panel, evenements):
    """Leur produit finance les achats du même jour : l'alternance ne doit
    pas défaire cette priorité."""
    engine = moteur(panel, evenements, {"AAA": "CALL"})
    _file(engine, {"C1": (900.0, "CALL"), "P1": (500.0, "PUT")})
    _file(engine, {"V1": (0.0, "CALL")})
    ordre = [s for s, _ in engine._execution_order()]
    assert ordre[0] == "V1"


def test_une_direction_vide_ne_reserve_rien(panel, evenements):
    engine = moteur(panel, evenements, {"AAA": "CALL"})
    _file(engine, {"C1": (900.0, "CALL"), "C2": (800.0, "CALL")})
    assert [s for s, _ in engine._execution_order()] == ["C1", "C2"]


def test_l_ordre_est_deterministe(panel, evenements):
    """Un mélange par tirage rendrait un run non reproductible d'une
    exécution à l'autre, ce qui interdirait toute comparaison de paramètres."""
    def ordre():
        engine = moteur(panel, evenements, {"AAA": "CALL"})
        _file(engine, {
            "C1": (900.0, "CALL"), "P1": (500.0, "PUT"),
            "C2": (800.0, "CALL"), "P2": (400.0, "PUT"),
        })
        return [s for s, _ in engine._execution_order()]

    assert ordre() == ordre() == ordre()


def test_un_roulement_sans_spec_garde_le_sens_de_sa_position(panel, evenements):
    """_check_rolls met en file sans _pending_spec : sans repli sur la
    position détenue, un roulement serait classé "direction inconnue" et
    sortirait de l'alternance."""
    from backtest.options_engine import _PendingOrder

    engine = moteur(panel, evenements, {"AAA": "CALL"})
    engine.positions["AAA"] = OptionPosition(
        symbol="AAA", option_type="PUT", strike=100.0, expiry=panel.close.index[-1],
        contracts=1.0, entry_premium=5.0, entry_date=panel.close.index[0],
        vol=0.25, multiplier=100.0, source="simulated", entry_spot=100.0,
        stop_reference_premium=5.0, stop_reference_spot=100.0,
    )
    engine.pending_orders["AAA"] = _PendingOrder(500.0, "roll", pd.Timestamp("2020-01-06"), roll=True)
    assert engine._order_direction("AAA") == "PUT"
