"""Réglages de construction de portefeuille des stratégies ACTIONS, et le
filtre qui les a tous rendus lisibles.

LE FILTRE DE PLAUSIBILITÉ EST LE RÉSULTAT PRINCIPAL DE CE BLOC, et il est né
d'un faux positif. En plafonnant le nombre de candidates retenues par
rebalancement (`max_positions`), le Sharpe montait de façon MONOTONE sur les
deux fenêtres -- 0,908 en illimité, 1,055 à quinze lignes, 1,192 à cinq --
avec un test apparié significatif. Tout indiquait un vrai effet.

La volatilité annualisée, elle, ne bougeait pas : 18,7% que le portefeuille
tienne 115 lignes ou 12. Impossible pour une vraie concentration. En regardant
ce que `max_positions` sélectionnait -- les plus fortes convictions, donc les
plus grands écarts -- l'explication est apparue : la distribution des écarts de
la valorisation combinée monte jusqu'à +1 817 436 625%, et plus on restreignait
aux « meilleures » convictions, plus le portefeuille était piloté par des
valorisations cassées (90e centile des lignes détenues : 2 331% en illimité,
102 654% à cinq lignes).

Une fois les écarts aberrants écartés, l'effet n'est plus significatif
(p = 0,42 à trente lignes). Le réglage reste disponible, mais il n'est pas
devenu un défaut -- et c'est tout l'objet de ces tests : empêcher que le filtre
disparaisse et rende ce faux positif à nouveau invisible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from backtest.data_loader import PricePanel
from backtest.engine import BacktestEngine
from backtest.strategies import STRATEGY_REGISTRY
from backtest.strategies.base import (
    Strategy, cap_per_sector, construire_poids, risk_adjusted_conviction, top_n_candidates,
)


def _candidates(gaps: dict[str, float], secteurs: dict | None = None, vols: dict | None = None) -> pd.DataFrame:
    lignes = []
    for symbol, gap in gaps.items():
        lignes.append({
            "symbol": symbol, "gap_pct": gap,
            "sector": (secteurs or {}).get(symbol, "Technologie"),
            "realized_vol": (vols or {}).get(symbol),
        })
    return pd.DataFrame(lignes)


# --------------------------------------------------------------------------- #
# Filtre de plausibilité : le garde-fou qui a démasqué le faux positif
# --------------------------------------------------------------------------- #
def _panel(prices: dict[str, list[float]], dates) -> PricePanel:
    close = pd.DataFrame(prices, index=dates)
    return PricePanel(close, close.copy(), close.apply(lambda c: c.last_valid_index()))


def _evt(symbol: str, published: pd.Timestamp, gap_pct: float) -> dict:
    return {
        "symbol": symbol, "published_date": published, "fiscal_year": 2019,
        "sector": "Technologie", "close_at_filing": 100.0,
        "valuation_dcf_per_share": 100.0 * (1 + gap_pct / 100),
        "gap_pct": gap_pct, "period_type": None,
    }


class _ToutCeQuiPasse(Strategy):
    def generate_target_weights(self, signals, current_positions):
        symbols = list(signals["symbol"])
        return {s: 1.0 / len(symbols) for s in symbols} if symbols else {}


def _moteur(events, **kwargs) -> BacktestEngine:
    dates = pd.bdate_range("2020-01-01", periods=30)
    panel = _panel({"SAIN": [100.0] * 30, "CASSE": [100.0] * 30}, dates)
    defaults = dict(
        universe_history=None, initial_capital=1_000_000.0, cost_bps=0.0,
        stop_loss_pct=-99.0, take_profit_pct=1e6, momentum_min_pct=None,
        rebalance_band_pct=0.0,
    )
    defaults.update(kwargs)
    return BacktestEngine(
        price_panel=panel, signal_events=events, strategy=_ToutCeQuiPasse(),
        fallback_universe_symbols={"SAIN", "CASSE"}, **defaults,
    )


def test_un_ecart_aberrant_n_entre_jamais_en_portefeuille():
    """+1 817 436 625% figure RÉELLEMENT dans l'archive du dépôt : une valeur
    théorique par action de plusieurs millions de dollars, produite par un
    multiple appliqué à un dénominateur proche de zéro. Ce n'est pas une
    sous-évaluation, et le classement des candidates se fait sur cette
    grandeur."""
    dates = pd.bdate_range("2020-01-01", periods=30)
    events = pd.DataFrame([
        _evt("SAIN", dates[1], 120.0),
        _evt("CASSE", dates[1], 1_817_436_625.0),
    ])
    moteur = _moteur(events)
    moteur.run()

    assert "SAIN" in moteur.positions
    assert "CASSE" not in moteur.positions, "un écart absurde a été traité comme une conviction"


def test_le_filtre_est_symetrique():
    """Une valeur théorique absurdement BASSE est tout aussi cassée qu'une
    absurdement haute. Le filtre porte sur la valeur absolue : le moteur
    actions est long-only, mais l'écart sert aussi au classement, et une
    stratégie directionnelle ajoutée plus tard hériterait du même garde-fou."""
    dates = pd.bdate_range("2020-01-01", periods=30)
    events = pd.DataFrame([_evt("SAIN", dates[1], 120.0), _evt("CASSE", dates[1], -99_000.0)])
    moteur = _moteur(events)
    moteur.run()
    assert "CASSE" not in moteur.positions


def test_le_seuil_est_desactivable_et_laisse_alors_tout_passer():
    dates = pd.bdate_range("2020-01-01", periods=30)
    events = pd.DataFrame([_evt("SAIN", dates[1], 120.0), _evt("CASSE", dates[1], 5_000.0)])
    moteur = _moteur(events, max_plausible_gap_pct=0)
    moteur.run()
    assert "CASSE" in moteur.positions


def test_le_defaut_du_moteur_vient_de_config():
    dates = pd.bdate_range("2020-01-01", periods=30)
    events = pd.DataFrame([_evt("SAIN", dates[1], 120.0)])
    assert _moteur(events).max_plausible_gap_pct == config.BACKTEST_MAX_PLAUSIBLE_GAP_PCT
    assert config.BACKTEST_MAX_PLAUSIBLE_GAP_PCT > 0, (
        "le filtre est désactivé par défaut : le faux positif de max_positions redevient invisible"
    )


def test_un_ecart_large_mais_plausible_passe():
    """Le seuil est volontairement LARGE : une décote réelle de 300% existe sur
    une société en difficulté temporaire, et le filtre ne doit écarter que ce
    qu'aucune thèse n'explique."""
    dates = pd.bdate_range("2020-01-01", periods=30)
    events = pd.DataFrame([_evt("SAIN", dates[1], 300.0)])
    moteur = _moteur(events)
    moteur.run()
    assert "SAIN" in moteur.positions


# --------------------------------------------------------------------------- #
# Pondération par le risque
# --------------------------------------------------------------------------- #
def test_la_ponderation_par_le_risque_penalise_le_plus_volatil():
    conviction = pd.Series([100.0, 100.0], index=["CALME", "AGITE"])
    vol = pd.Series([0.15, 0.60], index=["CALME", "AGITE"])
    ajustee = risk_adjusted_conviction(conviction, vol, exponent=1.0)
    assert ajustee["CALME"] == pytest.approx(4 * ajustee["AGITE"])


def test_un_exposant_nul_ne_change_rien():
    conviction = pd.Series([100.0, 50.0], index=["A", "B"])
    vol = pd.Series([0.15, 0.60], index=["A", "B"])
    assert risk_adjusted_conviction(conviction, vol, 0.0).equals(conviction)


def test_une_volatilite_manquante_ne_fait_pas_sortir_la_ligne():
    """Un titre récemment coté n'a pas d'historique suffisant. L'écarter ferait
    du filtre de risque un filtre de SIGNAL -- la même erreur que la zone de
    non-négociation avait failli commettre sur les entrées neuves."""
    conviction = pd.Series([100.0, 100.0], index=["CONNU", "NOUVEAU"])
    vol = pd.Series([0.20, np.nan], index=["CONNU", "NOUVEAU"])
    ajustee = risk_adjusted_conviction(conviction, vol, 1.0)
    assert ajustee["NOUVEAU"] == 100.0, "la ligne sans volatilité doit garder sa conviction brute"
    assert np.isfinite(ajustee).all()


def test_une_volatilite_nulle_est_traitee_comme_manquante():
    """Sinon la division ferait exploser la conviction à l'infini, et cette
    ligne capterait tout le capital."""
    ajustee = risk_adjusted_conviction(
        pd.Series([100.0], index=["A"]), pd.Series([0.0], index=["A"]), 1.0)
    assert np.isfinite(ajustee).all()


# --------------------------------------------------------------------------- #
# Plafond de nombre, plafond sectoriel, pondération par rang
# --------------------------------------------------------------------------- #
def test_le_plafond_de_nombre_garde_les_plus_fortes_convictions():
    candidates = _candidates({"A": 10.0, "B": 90.0, "C": 50.0, "D": 20.0})
    retenues = top_n_candidates(candidates, candidates["gap_pct"], 2)
    assert set(candidates.loc[retenues, "symbol"]) == {"B", "C"}


def test_un_plafond_inatteignable_ne_retire_personne():
    candidates = _candidates({"A": 10.0, "B": 90.0})
    assert len(top_n_candidates(candidates, candidates["gap_pct"], 10)) == 2
    assert len(top_n_candidates(candidates, candidates["gap_pct"], 0)) == 2


def test_le_plafond_sectoriel_borne_le_cumul_sans_redistribuer():
    """L'excédent n'est PAS redistribué : la somme descend et le reste va en
    cash. Redistribuer concentrerait davantage sur les secteurs restants --
    l'inverse du but."""
    poids = pd.Series([0.3, 0.3, 0.2], index=["T1", "T2", "S1"])
    secteurs = pd.Series(["Tech", "Tech", "Santé"], index=["T1", "T2", "S1"])
    borne = cap_per_sector(poids, secteurs, 30.0)

    assert borne[["T1", "T2"]].sum() == pytest.approx(0.30)
    assert borne["S1"] == pytest.approx(0.20), "un secteur sous son plafond ne doit pas bouger"
    assert borne.sum() < poids.sum()


def test_la_ponderation_par_rang_ecrase_les_ecarts_d_ampleur():
    candidates = _candidates({"A": 10.0, "B": 20.0, "C": 1000.0})
    par_ampleur = construire_poids(candidates, candidates["gap_pct"], max_weight_pct=0)
    par_rang = construire_poids(candidates, candidates["gap_pct"], max_weight_pct=0, rank_weighting=True)

    assert par_ampleur["C"] > 0.9, "sans rang, l'écart extrême capte presque tout"
    assert par_rang["C"] == pytest.approx(3 / 6), "par rang, il ne pèse que son rang"


def test_les_defauts_de_construction_ne_changent_rien():
    """Les quatre réglages de ce bloc existent pour être BALAYÉS, pas pour
    modifier le comportement : aucun n'a été retenu comme défaut, et les trois
    mesurés se sont révélés neutres ou négatifs."""
    for nom in ("valuation_gap_dcf", "valuation_gap_combined", "valuation_gap_sector_neutral"):
        strategie = STRATEGY_REGISTRY[nom]()
        assert strategie.vol_weight_exponent == 0.0
        assert strategie.max_positions == 0
        assert strategie.rank_weighting is False
    # Le plafond sectoriel reste propre à la stratégie neutre au secteur.
    assert STRATEGY_REGISTRY["valuation_gap_dcf"]().max_weight_per_sector_pct == 0.0
    assert STRATEGY_REGISTRY["valuation_gap_sector_neutral"]().max_weight_per_sector_pct > 0
