"""Défauts du moteur ACTIONS relevés à l'audit du run 20260816_000429, et
leurs correctifs.

Ce run affichait 19,00% de CAGR, +5,49% d'alpha... et 17,9% d'ordres d'achat
tronqués. L'audit a montré que la troncature n'était pas seulement la
conséquence assumée de la règle des positions gelées, mais tenait pour partie
à trois défauts distincts :

    A1  la file d'exécution classait les ordres sur le SIGNE DE LA CIBLE et
        non sur leur sens réel, si bien que le résultat du run dépendait de
        l'ordre d'itération d'un dictionnaire ;
    A2  une position détenue court-circuitait TOUS les filtres de péremption
        du signal, et était donc renforcée sur une thèse déclarée invalide ;
    A3  num_trades / win_rate_pct / profit_factor comptaient des exécutions et
        non des thèses, ce qui gonflait mécaniquement le taux de réussite ;
    A4  l'indice de référence équipondéré déversait une baisse étalée sur
        plusieurs semaines en une seule séance.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest import data_loader, metrics as metrics_mod
from backtest.data_loader import PricePanel, UniverseResolver
from backtest.engine import BacktestEngine
from backtest.strategies.base import Strategy


def _panel(prices: dict[str, list[float]], dates) -> PricePanel:
    close = pd.DataFrame(prices, index=dates)
    return PricePanel(close, close.copy(), close.apply(lambda c: c.last_valid_index()))


def _evt(symbol: str, published: pd.Timestamp, gap_pct: float = 100.0, period_type=None) -> dict:
    """period_type=None par défaut : c'est le seul moyen que le paramètre
    signal_max_age_days du moteur s'applique vraiment. Avec "FY" ou "TTM",
    data_loader.signal_max_age_for lui préfère
    config.BACKTEST_SIGNAL_MAX_AGE_DAYS_BY_PERIOD (270 / 120 jours), et un
    test qui croirait régler l'âge maximal à 30 jours testerait 270."""
    return {
        "symbol": symbol, "published_date": published, "fiscal_year": 2019,
        "sector": "Technologie", "close_at_filing": 100.0,
        "valuation_dcf_per_share": 200.0, "gap_pct": gap_pct, "period_type": period_type,
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


# --------------------------------------------------------------------------- #
# A1 : le sens d'un ordre n'est pas le signe de sa cible
# --------------------------------------------------------------------------- #
class _PoidsFixes(Strategy):
    """Poids identiques, énumérés dans un ORDRE imposé -- c'est la seule
    variable de l'expérience."""

    def __init__(self, ordre: tuple[str, ...]):
        super().__init__(ordre=ordre)
        self.ordre = ordre

    def generate_target_weights(self, signals, current_positions):
        presents = set(signals["symbol"])
        return {s: 0.5 for s in self.ordre if s in presents}


def _run_ordre(ordre: tuple[str, ...]):
    dates = pd.bdate_range("2020-01-01", periods=10)
    # AAA quadruple avant le rebalancement de J5 : sa ligne pèse alors bien
    # plus que sa cible de 50%, l'ordre émis est donc une VENTE partielle --
    # dont la cible reste POSITIVE. BBB entre au même rebalancement (achat).
    panel = _panel({"AAA": [100.0] * 4 + [400.0] * 6, "BBB": [100.0] * 10}, dates)
    events = pd.DataFrame([_evt("AAA", dates[1]), _evt("AAA", dates[5]), _evt("BBB", dates[5])])
    engine = _moteur(panel, events, _PoidsFixes(ordre))
    engine.run()
    return engine


def test_A1_le_resultat_ne_depend_pas_de_l_ordre_d_enumeration():
    """Mêmes prix, mêmes poids cibles : seul l'ordre d'énumération change.

    Avant correctif, l'ordre ("BBB", "AAA") laissait 750 000 $ en cash et BBB
    à 500 000 $ au lieu de 1 250 000 $ -- l'achat de BBB était tronqué faute
    de cash alors que la vente partielle d'AAA, exécutée juste après, le
    couvrait largement."""
    aaa_dabord = _run_ordre(("AAA", "BBB"))
    bbb_dabord = _run_ordre(("BBB", "AAA"))

    for engine in (aaa_dabord, bbb_dabord):
        assert engine.cash == 0.0, "du cash est resté oisif alors qu'une vente le couvrait"

    assert aaa_dabord.equity_curve_rows[-1]["nav"] == bbb_dabord.equity_curve_rows[-1]["nav"]
    for symbol in ("AAA", "BBB"):
        assert aaa_dabord.positions[symbol].shares == bbb_dabord.positions[symbol].shares


def test_A1_une_vente_partielle_est_executee_avant_les_achats():
    engine = _run_ordre(("BBB", "AAA"))
    ventes = [t for t in engine.trades if t["exit_reason"] == "rebalance"]
    assert ventes, "aucune vente de rebalancement : le scénario ne reproduit rien"
    assert engine.truncated_orders_count == 0


def test_A1_un_manque_reel_de_cash_reduit_TOUS_les_achats_du_meme_facteur():
    """Un manque de cash RÉEL, et parfaitement légitime : _rebalance
    dimensionne sur le NAV de la clôture de J, l'exécution a lieu à
    l'ouverture de J+1, et entre les deux la vente qui devait financer les
    achats se fait à un cours bien plus bas.

    Il n'y a alors pas assez de cash quoi qu'on fasse -- et c'est exactement
    la situation où l'ancienne file servait le premier ordre du dictionnaire
    en entier et abandonnait le second."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    panel = _panel({
        # AAA décroche de 20% à la clôture de J4 (stop-loss), puis ouvre 50%
        # plus bas encore le lendemain : la vente rapporte la moitié de ce que
        # le budget de rebalancement avait supposé.
        "AAA": [100.0] * 4 + [80.0, 40.0, 40.0, 40.0, 40.0, 40.0],
        "BBB": [100.0] * 10,
        "CCC": [100.0] * 10,
    }, dates)
    open_ = panel.close.copy()
    open_.loc[dates[5], "AAA"] = 40.0
    panel = PricePanel(panel.close, open_, panel.last_valid_date)

    events = pd.DataFrame([
        _evt("AAA", dates[1]),
        _evt("BBB", dates[4]), _evt("CCC", dates[4]),
    ])

    class Cibles(Strategy):
        def generate_target_weights(self, signals, current_positions):
            presents = set(signals["symbol"])
            if presents == {"AAA"}:
                return {"AAA": 1.0}
            return {s: 0.5 for s in ("BBB", "CCC") if s in presents}

    engine = _moteur(panel, events, Cibles(), stop_loss_pct=-15.0)
    engine.run()

    valeurs = {s: p.shares * panel.close_at(s, dates[-1]) for s, p in engine.positions.items()}
    assert set(valeurs) == {"BBB", "CCC"}, "un des deux achats a été sacrifié"
    assert valeurs["BBB"] == pytest.approx(valeurs["CCC"]), (
        "les poids demandés n'ont pas été respectés : un ordre servi en entier, l'autre rogné"
    )
    assert engine.truncated_orders_count == 2, "la réduction au prorata doit rester signalée"


# --------------------------------------------------------------------------- #
# A2 : une position détenue n'est pas renforcée sur un signal périmé
# --------------------------------------------------------------------------- #
class _SeuilGap(Strategy):
    def generate_target_weights(self, signals, current_positions):
        candidates = signals[signals["gap_pct"] >= 50.0]
        return {s: 0.5 for s in candidates["symbol"]}


def _run_peremption(**kwargs):
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    # BBB monte de 1%/jour : le NAV monte avec lui, donc la cible en dollars
    # de AAA (50% du budget) monte aussi -- c'est ce qui déclenchait les
    # renforcements de AAA sur son unique et vieux signal.
    panel = _panel(
        {"AAA": [100.0] * n, "BBB": [100.0 * (1.01 ** i) for i in range(n)]}, dates,
    )
    events = pd.DataFrame(
        [_evt("AAA", dates[1])] + [_evt("BBB", dates[i]) for i in range(1, n - 1, 20)]
    )
    engine = _moteur(panel, events, _SeuilGap(), **kwargs)
    engine.run()
    detenu = pd.DataFrame(engine.positions_history_rows)
    return engine, detenu[detenu["symbol"] == "AAA"].reset_index(drop=True)


def _apres_peremption(detenu: pd.DataFrame, jours: int = 60) -> pd.Series:
    """Quantités détenues bien après l'expiration du signal -- les
    renforcements passés PENDANT la fenêtre de validité, eux, sont légitimes."""
    return detenu[detenu["date"] >= detenu["date"].iloc[0] + pd.Timedelta(days=jours)]["shares"]


def test_A2_un_signal_perime_ne_renforce_plus_une_position_detenue():
    """AAA n'a qu'un signal, du 2e jour, avec un âge maximal de 30 jours.

    Avant correctif la ligne était renforcée x3,88 sur toute la période : le
    commentaire du code affirmait qu'une position détenue était "de toute
    façon gelée, pas rebalancée sur la base de ce vieux signal", ce que le
    code ne faisait pas."""
    _, aaa = _run_peremption(signal_max_age_days=30)
    assert not aaa.empty, "AAA n'a jamais été acheté : le scénario ne teste rien"
    apres = _apres_peremption(aaa)
    assert len(apres) > 100, "fenêtre d'observation trop courte"
    assert apres.nunique() == 1, (
        "la position a été renforcée alors que son signal était périmé"
    )


def test_A2_un_8k_materiel_gele_aussi_la_position():
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    huit_k = pd.DataFrame([{"symbol": "AAA", "filed_date": dates[5]}])
    _, aaa = _run_peremption(signal_max_age_days=100_000, material_events_8k=huit_k)
    assert not aaa.empty
    assert aaa["shares"].nunique() == 1, (
        "la position a été renforcée malgré un 8-K matériel postérieur au signal"
    )


def test_A2_une_position_gelee_n_est_pas_vendue_pour_autant():
    """Le gel ne doit pas devenir une sortie déguisée : la règle utilisateur
    est que seuls stop-loss/take-profit ferment une position."""
    engine, aaa = _run_peremption(signal_max_age_days=30)
    assert "AAA" in engine.positions
    assert not [t for t in engine.trades if t["symbol"] == "AAA"]


def test_A2_un_signal_frais_renforce_toujours():
    """Contrôle négatif : sans péremption, le renforcement doit rester actif
    -- sinon le correctif aurait simplement éteint le rebalancement."""
    _, aaa = _run_peremption(signal_max_age_days=100_000)
    assert _apres_peremption(aaa).nunique() > 1


def test_A2_le_momentum_ne_filtre_que_les_nouvelles_entrees():
    """Le filtre momentum est documenté comme un garde-fou d'ENTRÉE. Appliqué
    à une position détenue, il deviendrait un signal de sortie déguisé."""
    n = 400
    dates = pd.bdate_range("2020-01-01", periods=n)
    # AAA s'effondre après son entrée : son momentum 12-1 devient très négatif.
    panel = _panel(
        {"AAA": [100.0] * 30 + [100.0 * (0.99 ** i) for i in range(n - 30)],
         "BBB": [100.0] * n},
        dates,
    )
    events = pd.DataFrame([_evt("AAA", dates[i]) for i in range(1, n - 1, 20)]
                          + [_evt("BBB", dates[i]) for i in range(1, n - 1, 20)])
    engine = _moteur(panel, events, _SeuilGap(), momentum_min_pct=-10.0)
    engine.run()
    assert "AAA" in engine.positions, "le filtre momentum a fermé une position détenue"


# --------------------------------------------------------------------------- #
# A3 : un aller-retour partiel n'est pas un trade
# --------------------------------------------------------------------------- #
def test_A3_les_metriques_par_these_ne_comptent_pas_les_allegements():
    """Une thèse perdante, allégée 29 fois pendant sa hausse puis soldée au
    stop, s'affichait comme 32 trades à 91% de réussite."""
    trades = pd.DataFrame(
        [{"symbol": "AAA", "entry_date": pd.Timestamp("2020-01-02"), "pnl": 1_000.0,
          "holding_days": 10 * i, "exit_reason": "rebalance"} for i in range(1, 30)]
        + [{"symbol": "AAA", "entry_date": pd.Timestamp("2020-01-02"), "pnl": -255_000.0,
            "holding_days": 300, "exit_reason": "stop_loss"}]
    )
    par_these = metrics_mod._position_level_metrics(trades)

    assert par_these["num_positions_closed"] == 1
    assert par_these["win_rate_positions_pct"] == 0.0
    assert par_these["profit_factor_positions"] == 0.0  # aucun gain à opposer
    assert par_these["avg_holding_days_positions"] == 300


def test_A3_une_reouverture_compte_comme_une_these_distincte():
    trades = pd.DataFrame([
        {"symbol": "AAA", "entry_date": pd.Timestamp("2020-01-02"), "pnl": 500.0,
         "holding_days": 30, "exit_reason": "take_profit"},
        {"symbol": "AAA", "entry_date": pd.Timestamp("2020-06-01"), "pnl": -200.0,
         "holding_days": 20, "exit_reason": "stop_loss"},
    ])
    par_these = metrics_mod._position_level_metrics(trades)
    assert par_these["num_positions_closed"] == 2
    assert par_these["win_rate_positions_pct"] == 50.0
    assert par_these["profit_factor_positions"] == 2.5


def test_A3_les_metriques_par_these_sont_dans_compute_metrics():
    equity = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=30),
        "nav": [1_000_000.0 * (1.001 ** i) for i in range(30)],
        "cash": [0.0] * 30, "invested_value": [1_000_000.0] * 30,
        "num_positions": [1] * 30,
    })
    trades = pd.DataFrame([{
        "symbol": "AAA", "entry_date": pd.Timestamp("2020-01-02"),
        "exit_date": pd.Timestamp("2020-02-01"), "shares": 10.0, "entry_price": 100.0,
        "exit_price": 110.0, "pnl": 100.0, "return_pct": 10.0, "holding_days": 30,
        "exit_reason": "take_profit",
    }])
    computed = metrics_mod.compute_metrics(equity, trades)
    assert computed["num_positions_closed"] == 1
    assert computed["num_trades"] == 1


# --------------------------------------------------------------------------- #
# A4 : l'indice de référence ne concentre pas une baisse étalée sur une séance
# --------------------------------------------------------------------------- #
def test_A4_un_trou_de_cotation_ne_fabrique_pas_un_krach_d_un_jour():
    """AAA disparaît des données 15 jours (au-delà de FORWARD_FILL_MAX_DAYS)
    puis reprend 40% plus bas.

    Avant correctif, pct_change reportait la dernière valeur connue SANS
    limite -- annulant le forward-fill borné du panel -- et la totalité de la
    baisse tombait sur la séance de reprise : -20% en un jour sur un indice à
    deux composantes."""
    dates = pd.bdate_range("2020-01-01", periods=40)
    close_raw = pd.DataFrame(
        {"AAA": [100.0] * 10 + [float("nan")] * 15 + [60.0] * 15, "BBB": [100.0] * 40},
        index=dates,
    )
    close = close_raw.ffill(limit=data_loader.FORWARD_FILL_MAX_DAYS)
    panel = PricePanel(close, close_raw.copy(), close_raw.apply(lambda c: c.last_valid_index()))

    serie, _ = data_loader.build_benchmark_series(
        panel, UniverseResolver(None, {"AAA", "BBB"}), symbol="SPY",
    )
    pires = serie.pct_change(fill_method=None).min()
    assert pires > -0.05, f"baisse concentrée sur une séance ({pires:.1%})"


def test_A4_les_rendements_reels_restent_comptes():
    """Contrôle négatif : sans trou, la baisse doit bien apparaître."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    close = pd.DataFrame(
        {"AAA": [100.0] * 5 + [60.0] * 5, "BBB": [100.0] * 10}, index=dates,
    )
    panel = PricePanel(close, close.copy(), close.apply(lambda c: c.last_valid_index()))
    serie, _ = data_loader.build_benchmark_series(
        panel, UniverseResolver(None, {"AAA", "BBB"}), symbol="SPY",
    )
    assert serie.pct_change(fill_method=None).min() == pytest.approx(-0.20)


# --------------------------------------------------------------------------- #
# A5 : la couverture de l'univers par les signaux est mesurée, pas supposée
# --------------------------------------------------------------------------- #
def test_A5_la_couverture_des_signaux_est_reportee():
    """Le biais de survivance RÉSIDUEL : un univers point-in-time empêche
    d'acheter trop tôt, mais ne crée pas les signaux des entreprises radiées.
    Ici, seul AAA a des signaux alors que l'indice compte deux membres."""
    dates = pd.bdate_range("2020-01-01", periods=300)
    panel = _panel({"AAA": [100.0] * 300, "BBB": [100.0] * 300}, dates)
    events = pd.DataFrame([_evt("AAA", dates[i]) for i in range(1, 299, 20)])
    engine = _moteur(panel, events, _SeuilGap())
    engine.run()

    diagnostics = engine.execution_diagnostics()
    assert diagnostics["signal_coverage_avg_ratio"] == 0.5
    assert diagnostics["signal_coverage_min_ratio"] == 0.5
    assert not engine.signal_coverage().empty


def test_A5_l_avertissement_nomme_l_annee_la_moins_couverte(caplog):
    dates = pd.bdate_range("2020-01-01", periods=300)
    panel = _panel({"AAA": [100.0] * 300, "BBB": [100.0] * 300}, dates)
    events = pd.DataFrame([_evt("AAA", dates[i]) for i in range(1, 299, 20)])
    engine = _moteur(panel, events, _SeuilGap())
    engine.run()
    with caplog.at_level("WARNING"):
        engine.execution_diagnostics()
    assert any("Couverture moyenne de l'univers" in m for m in caplog.messages)


# --------------------------------------------------------------------------- #
# A6 : un dépôt un jour non coté ne disparaît pas
# --------------------------------------------------------------------------- #
def test_A6_un_signal_publie_un_jour_ferme_est_connu_a_la_seance_suivante():
    """La SEC accepte les dépôts des jours où le NYSE est fermé (le Vendredi
    saint, tous les ans). Le calendrier du moteur vient des COURS (03b), pas
    d'EDGAR : un dict indexé sur la date exacte ne retrouvait jamais ces
    signaux, qui étaient perdus en silence."""
    dates = pd.bdate_range("2020-01-06", periods=20)  # que des jours ouvrés
    panel = _panel({"AAA": [100.0] * 20}, dates)
    dimanche = pd.Timestamp("2020-01-12")
    assert dimanche not in set(dates), "le scénario suppose une date hors calendrier"

    engine = _moteur(panel, pd.DataFrame([_evt("AAA", dimanche)]), _SeuilGap())
    engine.run()
    assert "AAA" in engine.known_signals, "signal publié hors séance jamais vu"
    assert "AAA" in engine.positions, "signal vu mais jamais joué"


def test_A6_chaque_signal_n_est_consomme_qu_une_fois():
    dates = pd.bdate_range("2020-01-06", periods=20)
    panel = _panel({"AAA": [100.0] * 20}, dates)
    engine = _moteur(panel, pd.DataFrame([_evt("AAA", dates[2])]), _SeuilGap())
    engine.run()
    lignes = pd.DataFrame(engine.signals_history_rows)
    assert len(lignes) == 1, "le même dépôt a été rejoué plusieurs jours de suite"


def test_A6_les_signaux_anterieurs_au_debut_du_run_sont_connus_sans_etre_joues():
    """--start-date ne doit pas rendre invisible ce qui a été publié avant :
    le signal est connu (il l'était vraiment), mais son âge l'écarte de toute
    nouvelle entrée."""
    dates = pd.bdate_range("2020-01-06", periods=200)
    panel = _panel({"AAA": [100.0] * 200}, dates)
    engine = _moteur(
        panel, pd.DataFrame([_evt("AAA", dates[0]), _evt("BBB2", dates[100])]),
        _SeuilGap(), start_date=dates[150], signal_max_age_days=30,
    )
    engine.run()
    assert "AAA" in engine.known_signals
    assert "AAA" not in engine.positions, "un signal vieux de 150 jours a déclenché un achat"


# --------------------------------------------------------------------------- #
# A7 : les frais ne sont pas une pénurie de cash
# --------------------------------------------------------------------------- #
def test_A7_le_manque_a_hauteur_des_frais_n_est_pas_compte_comme_troncature():
    """_rebalance alloue une VALEUR DE POSITION égale au NAV ; l'acquérir
    consomme en plus la commission. Un portefeuille pleinement investi est
    donc court d'exactement cost_bps à chaque rebalancement, et ne peut pas ne
    pas l'être -- ce n'est pas un défaut d'exécution.

    Avant correctif, ce manque de 0,1% faisait afficher "100% des ordres
    tronqués" sur un moteur qui fonctionnait, et noyait les vraies pénuries."""
    dates = pd.bdate_range("2020-01-01", periods=12)
    panel = _panel({"AAA": [100.0] * 12, "BBB": [100.0] * 12}, dates)
    events = pd.DataFrame([_evt("AAA", dates[1]), _evt("BBB", dates[1])])
    engine = _moteur(panel, events, _SeuilGap(), cost_bps=10.0)
    engine.run()

    assert engine.buy_orders_count > 0
    assert engine.truncated_orders_count == 0, (
        "le coût de transaction a été compté comme une pénurie de cash"
    )
    diagnostics = engine.execution_diagnostics()
    assert diagnostics["unfilled_dollar_pct"] < 0.5


def test_A7_une_vraie_penurie_reste_signalee():
    """Contrôle négatif : le seuil ne doit pas absorber un manque réel."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    panel = _panel({
        "AAA": [100.0] * 4 + [80.0, 40.0, 40.0, 40.0, 40.0, 40.0],
        "BBB": [100.0] * 10, "CCC": [100.0] * 10,
    }, dates)
    open_ = panel.close.copy()
    open_.loc[dates[5], "AAA"] = 40.0
    panel = PricePanel(panel.close, open_, panel.last_valid_date)

    class Cibles(Strategy):
        def generate_target_weights(self, signals, current_positions):
            presents = set(signals["symbol"])
            if presents == {"AAA"}:
                return {"AAA": 1.0}
            return {s: 0.5 for s in ("BBB", "CCC") if s in presents}

    events = pd.DataFrame([_evt("AAA", dates[1]), _evt("BBB", dates[4]), _evt("CCC", dates[4])])
    engine = _moteur(panel, events, Cibles(), stop_loss_pct=-15.0, cost_bps=10.0)
    engine.run()

    assert engine.truncated_orders_count == 2
    assert engine.execution_diagnostics()["unfilled_dollar_pct"] > 5.0
