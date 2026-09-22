"""Les deux réglages ajoutés au moteur ACTIONS pour l'optimisation du Sharpe :
la zone de non-négociation du rebalancement, et le plafond de concentration
exposé en paramètre de stratégie.

CE QUE CES TESTS PROTÈGENT EN PRIORITÉ : l'absence de régression. Les deux
réglages ont un défaut qui reproduit EXACTEMENT le comportement d'avant leur
ajout (zone à 0, plafond lu dans config). Un run d'archive doit rester
reproductible à l'identique, sinon le réglage n'est pas un réglage mais un
changement de moteur déguisé.

LE PIÈGE QUE LE SECOND GROUPE DE TESTS DOCUMENTE. Une première version filtrait
LIGNE À LIGNE : ne toucher une position que si SA cible s'écartait de plus de
x% de SA valeur. Elle divisait bien les exécutions par 15 -- et elle filtrait
du même coup les ALLÈGEMENTS, qui sont précisément ce qui finance les achats du
même jour (cf. engine._execute_pending_orders, où les ventes passent avant les
achats pour cette raison). Mesuré sur 2015-2026 : 52% du montant d'achat
demandé devenait infinançable, contre 5% sans filtre. D'où la forme retenue --
un seuil sur la DÉRIVE TOTALE, qui laisse le choix entre repeser tout le
portefeuille ou n'y pas toucher, jamais entre les deux.
"""

from __future__ import annotations

import pandas as pd

import config
from backtest.data_loader import PricePanel
from backtest.engine import BacktestEngine
from backtest.strategies import STRATEGY_REGISTRY
from backtest.strategies.base import Strategy


def _panel(prices: dict[str, list[float]], dates) -> PricePanel:
    close = pd.DataFrame(prices, index=dates)
    return PricePanel(close, close.copy(), close.apply(lambda c: c.last_valid_index()))


def _evt(symbol: str, published: pd.Timestamp, gap_pct: float = 100.0) -> dict:
    """period_type=None : sans quoi data_loader.signal_max_age_for préfère
    config.BACKTEST_SIGNAL_MAX_AGE_DAYS_BY_PERIOD au paramètre du moteur."""
    return {
        "symbol": symbol, "published_date": published, "fiscal_year": 2019,
        "sector": "Technologie", "close_at_filing": 100.0,
        "valuation_dcf_per_share": 200.0, "gap_pct": gap_pct, "period_type": None,
    }


def _moteur(panel: PricePanel, events: pd.DataFrame, strategy: Strategy, **kwargs) -> BacktestEngine:
    defaults = dict(
        universe_history=None, initial_capital=1_000_000.0, cost_bps=0.0,
        stop_loss_pct=-99.0, take_profit_pct=1e6, momentum_min_pct=None,
    )
    defaults.update(kwargs)
    return BacktestEngine(
        price_panel=panel, signal_events=events, strategy=strategy,
        fallback_universe_symbols=set(panel.close.columns), **defaults,
    )


class _ToutesAPoidsEgaux(Strategy):
    """Poids égaux sur toutes les candidates connues. La cible d'une ligne
    bouge donc dès qu'une AUTRE entreprise publie -- c'est exactement le
    mécanisme qui produit la rotation qu'on cherche à filtrer."""

    def generate_target_weights(self, signals, current_positions):
        symbols = list(signals["symbol"])
        return {s: 1.0 / len(symbols) for s in symbols} if symbols else {}


# --------------------------------------------------------------------------- #
# Non-régression : le défaut ne change rien
# --------------------------------------------------------------------------- #
def _scenario(n: int = 60):
    """Trois titres dont les cours DÉRIVENT doucement dans des sens opposés,
    et qui republient régulièrement.

    La dérive est indispensable au scénario : à prix strictement constants, les
    lignes restent exactement à leur cible et le moteur ne passe déjà plus
    d'ordre (l'écart tombe sous MIN_TRADE_DOLLAR), zone ou pas -- le test ne
    distinguerait alors rien. Ici chaque republication trouve le portefeuille
    à quelques points de sa cible : assez pour qu'un repesage soit émis sans
    zone, trop peu pour qu'il en vaille la peine."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    panel = _panel({
        "AAA": [100.0 * 1.002 ** t for t in range(n)],
        "BBB": [100.0 * 0.998 ** t for t in range(n)],
        "CCC": [100.0] * n,
    }, dates)
    events = pd.DataFrame(
        [_evt("AAA", dates[1]), _evt("BBB", dates[1]), _evt("CCC", dates[1])]
        # Republications : sans zone de non-négociation, chacune remet les
        # trois lignes en mouvement pour rattraper quelques points de dérive.
        + [_evt(sym, dates[j]) for j in range(5, n, 5) for sym in ("AAA", "BBB", "CCC")]
    )
    return panel, events


def test_zone_a_zero_reproduit_exactement_le_comportement_dorigine():
    """LE test de non-régression : un moteur construit sans le paramètre et un
    moteur construit avec la valeur 0 doivent produire la MÊME courbe de NAV,
    les mêmes trades et le même cash."""
    panel, events = _scenario()
    sans = _moteur(panel, events, _ToutesAPoidsEgaux())
    avec_zero = _moteur(panel, events, _ToutesAPoidsEgaux(), rebalance_band_pct=0.0)
    sans.run()
    avec_zero.run()

    assert [r["nav"] for r in sans.equity_curve_rows] == [r["nav"] for r in avec_zero.equity_curve_rows]
    assert len(sans.trades) == len(avec_zero.trades)
    assert sans.cash == avec_zero.cash
    assert avec_zero.rebalance_skipped_days == 0


def test_le_defaut_du_moteur_est_celui_de_config():
    panel, events = _scenario()
    moteur = _moteur(panel, events, _ToutesAPoidsEgaux())
    assert moteur.rebalance_band_pct == config.BACKTEST_REBALANCE_BAND_PCT


# --------------------------------------------------------------------------- #
# La zone de non-négociation fait ce qu'elle annonce
# --------------------------------------------------------------------------- #
def test_une_zone_large_supprime_les_repesages_sans_objet():
    """Une zone très large ne doit laisser passer que les repesages qui font
    réellement bouger le portefeuille -- typiquement l'arrivée d'une ligne
    neuve, dont la cible entière compte dans la dérive."""
    panel, events = _scenario()
    serre = _moteur(panel, events, _ToutesAPoidsEgaux(), rebalance_band_pct=0.0)
    large = _moteur(panel, events, _ToutesAPoidsEgaux(), rebalance_band_pct=25.0)
    serre.run()
    large.run()

    assert large.rebalance_skipped_days > 0, "la zone n'a rien filtré : le scénario ne reproduit rien"
    assert len(large.trades) < len(serre.trades)


def test_la_zone_n_empeche_jamais_une_entree_neuve():
    """Une entrée neuve porte sa cible ENTIÈRE dans la dérive : un signal
    vraiment nouveau déclenche donc lui-même le repesage, au lieu d'être
    retardé jusqu'à ce qu'un autre événement l'autorise. Sans cette propriété,
    la zone deviendrait un filtre de signal, ce qu'elle n'est pas."""
    panel, events = _scenario()
    large = _moteur(panel, events, _ToutesAPoidsEgaux(), rebalance_band_pct=25.0)
    large.run()
    assert set(large.positions) == {"AAA", "BBB", "CCC"}


def test_la_zone_ne_fait_pas_mourir_le_portefeuille_de_faim():
    """Le défaut de la version LIGNE À LIGNE, en contrôle négatif : filtrer les
    allègements affamait les achats (52% du montant demandé infinançable).
    Avec un seuil sur la dérive totale, le sous-investissement ne doit pas
    empirer par rapport à l'absence de zone."""
    panel, events = _scenario()
    serre = _moteur(panel, events, _ToutesAPoidsEgaux(), rebalance_band_pct=0.0)
    large = _moteur(panel, events, _ToutesAPoidsEgaux(), rebalance_band_pct=25.0)
    serre.run()
    large.run()

    assert large.execution_diagnostics()["unfilled_dollar_pct"] <= (
        serre.execution_diagnostics()["unfilled_dollar_pct"] + 1e-9
    )


def test_la_zone_ne_touche_pas_aux_stops():
    """Les sorties par stop-loss/take-profit ne passent pas par _rebalance :
    une zone large ne doit pas retarder d'un jour la fermeture d'une ligne qui
    a franchi son seuil."""
    dates = pd.bdate_range("2020-01-01", periods=20)
    # AAA s'effondre à la clôture de J10 : le stop doit partir, zone ou pas.
    panel = _panel({"AAA": [100.0] * 10 + [50.0] * 10, "BBB": [100.0] * 20}, dates)
    events = pd.DataFrame([_evt("AAA", dates[1]), _evt("BBB", dates[1])])

    large = _moteur(
        panel, events, _ToutesAPoidsEgaux(),
        rebalance_band_pct=90.0, stop_loss_pct=-20.0,
    )
    large.run()
    stops = [t for t in large.trades if t["exit_reason"] == "stop_loss"]
    assert stops, "le stop-loss n'a pas été exécuté malgré une chute de 50%"
    assert "AAA" not in large.positions


def test_le_diagnostic_rapporte_ce_que_la_zone_a_filtre():
    panel, events = _scenario()
    moteur = _moteur(panel, events, _ToutesAPoidsEgaux(), rebalance_band_pct=25.0)
    moteur.run()
    diag = moteur.execution_diagnostics()

    assert diag["rebalance_band_pct"] == 25.0
    # Des JOURS de repesage, pas des événements : les trois titres publient le
    # même jour, et le portefeuille n'est repesé qu'une fois pour les trois.
    assert diag["rebalance_days_count"] == events["published_date"].nunique()
    assert 0 < diag["rebalance_skipped_days_pct"] <= 100


# --------------------------------------------------------------------------- #
# Plafond de concentration exposé en paramètre de stratégie
# --------------------------------------------------------------------------- #
def _signaux_deux_lignes() -> pd.DataFrame:
    """Deux candidates d'écarts très inégaux : sans plafond, la première
    capterait 90% du capital."""
    return pd.DataFrame([
        {**_evt("AAA", pd.Timestamp("2020-01-02"), gap_pct=900.0)},
        {**_evt("BBB", pd.Timestamp("2020-01-02"), gap_pct=100.0)},
    ])


def test_le_plafond_par_defaut_reste_celui_de_config():
    for nom in ("valuation_gap_dcf", "valuation_gap_sector_neutral"):
        strategie = STRATEGY_REGISTRY[nom]()
        assert strategie.max_weight_pct == config.BACKTEST_MAX_WEIGHT_PER_POSITION_PCT
        # Le paramètre doit aussi apparaître dans params : c'est lui qui est
        # consigné dans run_config.json, donc ce qui rend un run reproductible.
        assert strategie.params["max_weight_pct"] == config.BACKTEST_MAX_WEIGHT_PER_POSITION_PCT


def test_un_plafond_plus_bas_borne_vraiment_les_poids():
    signaux = _signaux_deux_lignes()
    strategie = STRATEGY_REGISTRY["valuation_gap_dcf"](entry_threshold_pct=0.0, max_weight_pct=5.0)
    poids = strategie.generate_target_weights(signaux, set())

    assert poids, "aucune candidate retenue : le scénario ne teste rien"
    assert max(poids.values()) <= 0.05 + 1e-9


def test_le_plafond_borne_aussi_la_strategie_neutre_au_secteur():
    """Même garde-fou côté sector_neutral, où le plafond par position se
    combine à un plafond par SECTEUR -- deux limites distinctes qu'il ne faut
    pas confondre."""
    signaux = _signaux_deux_lignes()
    strategie = STRATEGY_REGISTRY["valuation_gap_sector_neutral"](
        entry_threshold_pct=-1e9, min_absolute_gap_pct=0.0,
        max_weight_per_sector_pct=0.0, max_weight_pct=5.0,
    )
    poids = strategie.generate_target_weights(signaux, set())

    assert poids, "aucune candidate retenue : le scénario ne teste rien"
    assert max(poids.values()) <= 0.05 + 1e-9
