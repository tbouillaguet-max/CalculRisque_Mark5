"""Impact de marché (capacité) et ciblage de volatilité.

CES DEUX RÉGLAGES NE SONT PAS LÀ POUR AMÉLIORER UN SHARPE, et c'est ce qui les
distingue du reste de l'étude.

L'IMPACT DE MARCHÉ répond à une question que le coût forfaitaire de 10 bps ne
peut pas poser, puisqu'il ne dépend pas de la taille de l'ordre : jusqu'à quel
ENCOURS la stratégie tient-elle ? Mesuré avec un impact de 100 bps pour un
ordre égal au volume quotidien : CAGR 17,7% à un million de dollars, 16,9% à
cent millions, 14,6% à un milliard, 11,2% à cinq milliards -- la stratégie
cesse de battre l'indice (11,99%) vers deux à trois milliards.

LE CIBLAGE DE VOLATILITÉ échange du Sharpe contre du drawdown : à une cible de
12%, le drawdown maximal passe de -34,4% à -25,8% mais le Sharpe descend de
0,918 à 0,868. Il n'est donc pas retenu par défaut -- l'objectif de l'étude
était le Sharpe -- mais il est disponible pour qui préfère l'autre côté de cet
arbitrage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from backtest.data_loader import PricePanel, build_price_panel
from backtest.engine import BacktestEngine
from backtest.strategies.base import Strategy


def _cours(n: int = 60, volume: float = 1e6) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    lignes = []
    for symbol in ("AAA", "BBB"):
        for d in dates:
            lignes.append({"symbol": symbol, "date": d, "close": 100.0, "open": 100.0, "volume": volume})
    return pd.DataFrame(lignes)


def _evt(symbol: str, published: pd.Timestamp) -> dict:
    return {
        "symbol": symbol, "published_date": published, "fiscal_year": 2019,
        "sector": "Technologie", "close_at_filing": 100.0,
        "valuation_dcf_per_share": 200.0, "gap_pct": 100.0, "period_type": None,
    }


class _ToutAcheter(Strategy):
    def generate_target_weights(self, signals, current_positions):
        symbols = list(signals["symbol"])
        return {s: 1.0 / len(symbols) for s in symbols} if symbols else {}


def _moteur(panel, events, **kwargs) -> BacktestEngine:
    defaults = dict(
        universe_history=None, initial_capital=1_000_000.0, cost_bps=10.0,
        stop_loss_pct=-99.0, take_profit_pct=1e6, momentum_min_pct=None,
        rebalance_band_pct=0.0, trailing_stop_pct=None,
    )
    defaults.update(kwargs)
    return BacktestEngine(
        price_panel=panel, signal_events=events, strategy=_ToutAcheter(),
        fallback_universe_symbols=set(panel.close.columns), **defaults,
    )


# --------------------------------------------------------------------------- #
# Impact de marché
# --------------------------------------------------------------------------- #
def test_le_panel_expose_le_volume_en_dollars():
    """En dollars et non en titres : un volume de 10 millions d'actions ne dit
    rien tant qu'on ignore si l'action vaut 3 $ ou 300 $, et c'est bien un
    MONTANT qu'on cherche à exécuter."""
    panel = build_price_panel(_cours(volume=2e6))
    date = panel.close.index[40]
    assert panel.dollar_volume_at("AAA", date) == pytest.approx(2e6 * 100.0)


def test_un_panel_sans_volume_ne_fait_pas_echouer_le_modele():
    """Les caches de cours antérieurs à la collecte du volume existent : le
    modèle d'impact doit se désactiver de lui-même, pas planter."""
    cours = _cours().drop(columns=["volume"])
    panel = build_price_panel(cours)
    assert panel.dollar_volume_at("AAA", panel.close.index[40]) is None

    events = pd.DataFrame([_evt("AAA", panel.close.index[1])])
    moteur = _moteur(panel, events, impact_coefficient_bps=100.0)
    moteur.run()  # ne doit pas lever
    assert moteur.positions


def test_l_impact_suit_la_racine_de_la_part_de_volume():
    """Forme empirique standard : à 1% du volume quotidien on paie le dixième
    de l'impact d'un ordre égal au volume entier, pas le centième."""
    panel = build_price_panel(_cours())
    events = pd.DataFrame([_evt("AAA", panel.close.index[1])])
    moteur = _moteur(panel, events, impact_coefficient_bps=100.0)
    date = panel.close.index[40]
    volume = panel.dollar_volume_at("AAA", date)

    assert moteur._impact_bps("AAA", volume, date) == pytest.approx(100.0)
    assert moteur._impact_bps("AAA", volume * 0.01, date) == pytest.approx(10.0)
    assert moteur._impact_bps("AAA", volume * 0.25, date) == pytest.approx(50.0)


def test_un_coefficient_nul_laisse_le_cout_forfaitaire():
    panel = build_price_panel(_cours())
    events = pd.DataFrame([_evt("AAA", panel.close.index[1])])
    moteur = _moteur(panel, events, impact_coefficient_bps=0.0)
    assert moteur._impact_bps("AAA", 1e9, panel.close.index[40]) == 0.0


def test_un_encours_plus_gros_coute_plus_cher():
    """LE test de capacité, en miniature : à volume de marché identique, un
    portefeuille plus gros doit finir avec une performance plus faible. C'est
    exactement ce que le coût forfaitaire ne sait pas exprimer."""
    panel = build_price_panel(_cours(volume=1e4))  # marché volontairement étroit
    events = pd.DataFrame([_evt("AAA", panel.close.index[1]), _evt("BBB", panel.close.index[1])])

    petit = _moteur(panel, events, initial_capital=1e6, impact_coefficient_bps=100.0)
    gros = _moteur(panel, events, initial_capital=1e9, impact_coefficient_bps=100.0)
    petit.run()
    gros.run()

    rendement = lambda m, c: m.equity_curve_rows[-1]["nav"] / c - 1  # noqa: E731
    assert rendement(gros, 1e9) < rendement(petit, 1e6)


# --------------------------------------------------------------------------- #
# Ciblage de volatilité
# --------------------------------------------------------------------------- #
def _facteur_apres(moteur, chocs) -> float:
    """Facteur de ciblage après avoir alimenté le moteur d'une courbe de NAV
    donnée. La méthode ne prend pas de date : elle lit les dernières séances
    ENREGISTRÉES, ce qui rend un look-ahead impossible par construction."""
    nav = 1_000_000.0
    moteur.equity_curve_rows = []
    for choc in chocs:
        nav *= (1 + choc)
        moteur.equity_curve_rows.append({"nav": nav, "cash": 0.0, "invested_value": nav, "num_positions": 1})
    return moteur._echelle_ciblage_volatilite()


def test_le_ciblage_reduit_l_exposition_quand_ca_secoue():
    panel = build_price_panel(_cours())
    events = pd.DataFrame([_evt("AAA", panel.close.index[1])])
    moteur = _moteur(panel, events, vol_target_pct=10.0)

    rng = np.random.default_rng(0)
    calme = _facteur_apres(moteur, rng.normal(0, 0.002, 120))
    agite = _facteur_apres(moteur, rng.normal(0, 0.05, 120))

    assert agite < calme, "l'exposition n'a pas été réduite en régime agité"
    assert agite < 0.5, "un régime cinq fois plus agité que la cible doit fortement désinvestir"


def test_le_ciblage_ne_leve_jamais_de_levier():
    """Borné à 1 : le portefeuille peut se désinvestir quand ça secoue, jamais
    s'endetter quand c'est calme. Le moteur n'est pas margé, et un ciblage qui
    lèverait du levier changerait la nature du produit."""
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame({"AAA": [100.0] * n, "BBB": [100.0] * n}, index=dates)
    panel = PricePanel(close, close.copy(), close.apply(lambda c: c.last_valid_index()))
    events = pd.DataFrame([_evt("AAA", dates[1])])

    moteur = _moteur(panel, events, vol_target_pct=30.0)  # cible très au-dessus du réalisé
    moteur.run()
    assert moteur._echelle_ciblage_volatilite() == 1.0


def test_sans_cible_le_facteur_vaut_un():
    panel = build_price_panel(_cours())
    events = pd.DataFrame([_evt("AAA", panel.close.index[1])])
    moteur = _moteur(panel, events, vol_target_pct=None)
    assert moteur._echelle_ciblage_volatilite() == 1.0


def test_l_impact_de_marche_reste_desactive_par_defaut():
    """L'impact décrit un coût réel que le backtest à un million de dollars ne
    voit pas : une ligne y pèse quelques dizaines de milliers contre un volume
    quotidien médian de 113 millions. L'activer par défaut ne changerait rien
    aux chiffres tout en rendant chaque run dépendant du panel de volumes. Son
    emploi est l'étude de capacité, à la demande."""
    assert config.BACKTEST_IMPACT_COEFFICIENT_BPS == 0.0


def test_le_ciblage_de_volatilite_est_actif_a_douze():
    """ACTIVÉ sur décision de l'utilisateur, et le test le fige pour qu'un
    retour à `None` par inadvertance ne passe pas inaperçu -- il déplacerait
    tous les chiffres de référence du README.

    Ce n'est pas un réglage qui améliore le Sharpe, et le test ne doit pas
    laisser croire le contraire : mesuré, le ciblage le dégrade légèrement
    (−0,014 à −0,056 selon la stratégie, jamais significatif) et réduit
    nettement le drawdown maximal (−36,1 % → −26,5 % sur la combinée). C'est
    l'arbitrage inverse de celui que la seule optimisation du Sharpe aurait
    retenu, et il est assumé comme tel."""
    assert config.BACKTEST_VOL_TARGET_PCT == 12.0
    assert config.BACKTEST_VOL_TARGET_LOOKBACK_DAYS == 60
