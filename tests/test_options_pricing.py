"""C7 : dividendes dans Black-Scholes, courbe de taux annuelle."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import config
from backtest import options_pricing as bs
from tests.options_harness import cours, moteur, serie_bruitee, signaux


# --------------------------------------------------------------------------- #
# C7.1 : dividendes
# --------------------------------------------------------------------------- #

def test_defaut_sans_dividende_reproduit_l_ancien_comportement():
    """q=0 doit rendre exactement l'ancienne formule : le correctif ne change
    rien tant que l'appelant ne fournit pas de rendement."""
    args = (100.0, 100.0, 0.75, 0.25, "CALL")
    assert bs.bs_price(*args, q=0.0) == pytest.approx(bs.bs_price(*args))
    assert bs.bs_greeks(*args, q=0.0)["delta"] == pytest.approx(bs.bs_greeks(*args)["delta"])


def test_le_dividende_abaisse_le_call_et_releve_le_put():
    """Le biais corrigé : sans dividende, les calls sont surévalués et les
    puts sous-évalués, systématiquement."""
    args = (100.0, 100.0, 0.75, 0.25)
    call_sans = bs.bs_price(*args, "CALL", q=0.0)
    call_avec = bs.bs_price(*args, "CALL", q=0.04)
    put_sans = bs.bs_price(*args, "PUT", q=0.0)
    put_avec = bs.bs_price(*args, "PUT", q=0.04)

    assert call_avec < call_sans
    assert put_avec > put_sans


def test_l_ecart_croit_avec_le_rendement():
    """Donc l'erreur est la plus grande sur les secteurs à haut rendement --
    précisément ceux qu'un signal value sélectionne."""
    base = bs.bs_price(100.0, 100.0, 0.75, 0.25, "CALL", q=0.0)
    ecarts = [base - bs.bs_price(100.0, 100.0, 0.75, 0.25, "CALL", q=q) for q in (0.01, 0.02, 0.045)]
    assert ecarts == sorted(ecarts)
    assert all(e > 0 for e in ecarts)


def test_parite_call_put_avec_dividende():
    """Contrôle d'intégrité de la formule : C - P = S e^-qT - K e^-rT."""
    spot, strike, t, vol, r, q = 105.0, 100.0, 0.6, 0.3, 0.03, 0.025
    call = bs.bs_price(spot, strike, t, vol, "CALL", r=r, q=q)
    put = bs.bs_price(spot, strike, t, vol, "PUT", r=r, q=q)
    assert call - put == pytest.approx(spot * math.exp(-q * t) - strike * math.exp(-r * t))


def test_delta_borne_par_le_facteur_dividende():
    """Le delta d'un call vaut e^-qT N(d1) : il ne peut plus atteindre 1."""
    greeks = bs.bs_greeks(200.0, 100.0, 1.0, 0.2, "CALL", q=0.05)
    assert 0 < greeks["delta"] <= math.exp(-0.05 * 1.0)


def test_greeks_restent_de_signe_correct():
    call = bs.bs_greeks(100.0, 100.0, 0.75, 0.25, "CALL", q=0.03)
    put = bs.bs_greeks(100.0, 100.0, 0.75, 0.25, "PUT", q=0.03)
    assert call["delta"] > 0 and put["delta"] < 0
    assert call["gamma"] > 0 and put["gamma"] > 0
    assert call["vega"] > 0 and put["vega"] > 0


def test_rendement_sectoriel_configure():
    assert config.dividend_yield_for("Télécommunications") > config.dividend_yield_for("Technologie")
    assert config.dividend_yield_for("secteur inexistant") == config.SECTOR_DIVIDEND_YIELD["_default"]
    assert config.dividend_yield_for(None) == config.SECTOR_DIVIDEND_YIELD["_default"]


def test_le_moteur_applique_le_rendement_du_secteur():
    """Deux entreprises identiques dans deux secteurs de rendements très
    différents ne doivent pas donner la même prime."""
    panel = cours({"AAA": serie_bruitee(100.0, 60, graine=4)})
    jour = panel.close.index[30]

    primes = {}
    for secteur in ("Technologie", "Télécommunications"):
        ev = signaux(["AAA"], "2020-01-06")
        ev["sector"] = secteur
        engine = moteur(panel, ev, {"AAA": "CALL"})
        primes[secteur] = engine._select_contract("AAA", "CALL", 100.0, jour)["premium"]

    assert primes["Télécommunications"] < primes["Technologie"]


# --------------------------------------------------------------------------- #
# C7.2 : courbe de taux annuelle
# --------------------------------------------------------------------------- #

def test_courbe_de_taux_par_annee():
    assert config.risk_free_rate_for(2012) < 0.005      # ère des taux zéro
    assert config.risk_free_rate_for(2023) > 0.04       # resserrement
    assert config.risk_free_rate_for(1990) == config.RISK_FREE_RATE   # hors table -> repli
    assert config.risk_free_rate_for(None) == config.RISK_FREE_RATE


def test_le_taux_influence_la_prime():
    bas = bs.bs_price(100.0, 100.0, 0.75, 0.25, "CALL", r=config.risk_free_rate_for(2012))
    haut = bs.bs_price(100.0, 100.0, 0.75, 0.25, "CALL", r=config.risk_free_rate_for(2023))
    assert haut > bas


def test_le_moteur_utilise_le_taux_de_l_annee_simulee():
    panel_2012 = cours({"AAA": serie_bruitee(100.0, 60, graine=4)}, start="2012-01-02")
    panel_2023 = cours({"AAA": serie_bruitee(100.0, 60, graine=4)}, start="2023-01-02")

    primes = []
    for panel, date_signal in ((panel_2012, "2012-01-06"), (panel_2023, "2023-01-06")):
        engine = moteur(panel, signaux(["AAA"], date_signal), {"AAA": "CALL"})
        primes.append(engine._select_contract("AAA", "CALL", 100.0, panel.close.index[30])["premium"])

    assert primes[1] > primes[0], "le taux de l'année simulée n'est pas utilisé"


def test_metrics_utilise_la_courbe_annuelle():
    """Un taux à 4% appliqué à 2012 (taux réel ~0,09%) écrasait le Sharpe de
    toute la première moitié de l'historique."""
    from backtest import metrics as metrics_mod

    import numpy as np
    n = 500
    navs = list(100 * (1.0004) ** np.arange(n))

    def sharpe(start):
        curve = pd.DataFrame({
            "date": pd.bdate_range(start=start, periods=n), "nav": navs,
            "cash": 0.0, "invested_value": navs, "num_positions": 1,
        })
        return metrics_mod.compute_metrics(curve, pd.DataFrame(), risk_free_rate=0.04)["sharpe_ratio"]

    # Même courbe de NAV, deux époques : le Sharpe de 2012 doit être MEILLEUR
    # (le taux sans risque à battre y était quasi nul).
    assert sharpe("2012-01-02") > sharpe("2023-01-02")


# --------------------------------------------------------------------------- #
# Volatilité COTÉE : ce qu'on paie, et non ce que le titre réalise
# --------------------------------------------------------------------------- #

def test_a_la_monnaie_on_paie_la_realisee_plus_l_ecart():
    """Le niveau : une option ne s'achète pas à la volatilité réalisée mais à
    l'implicite cotée, plus haute (prime de risque de variance)."""
    cotee = bs.quoted_implied_vol(0.30, 100.0, 100.0, 2.0)
    assert cotee == pytest.approx(0.30 + config.OPTIONS_IMPLIED_VOL_SPREAD)


def test_les_strikes_bas_coutent_plus_cher():
    """Le skew. Un put hors de la monnaie vise un strike SOUS le cours : c'est
    là que la volatilité cotée monte, et c'est exactement la jambe que le
    backtest sous-facturait."""
    atm = bs.quoted_implied_vol(0.30, 100.0, 100.0, 2.0)
    bas = bs.quoted_implied_vol(0.30, 100.0, 70.0, 2.0)
    plus_bas = bs.quoted_implied_vol(0.30, 100.0, 50.0, 2.0)
    assert plus_bas > bas > atm


def test_les_strikes_hauts_restent_au_niveau_de_la_monnaie():
    """Bridage volontaire (cf. la docstring de quoted_implied_vol) : incliner
    aussi du côté haut, face à une loi de S_T lognormale à volatilité unique,
    fabriquerait un edge de bord de grille sans rapport avec la thèse."""
    atm = bs.quoted_implied_vol(0.30, 100.0, 100.0, 2.0)
    for strike in (120.0, 200.0, 1000.0):
        assert bs.quoted_implied_vol(0.30, 100.0, strike, 2.0) == pytest.approx(atm)


def test_la_correction_ne_rend_jamais_une_option_moins_chere():
    """Propriété qui donne son sens au correctif : il retire de l'optimisme, il
    ne peut pas en ajouter. Vrai à tout strike et à toute maturité."""
    for t_years in (0.25, 1.0, 2.0, 5.0):
        for strike in (40.0, 70.0, 100.0, 150.0, 400.0):
            assert bs.quoted_implied_vol(0.30, 100.0, strike, t_years) >= 0.30


def test_la_pente_est_normalisee_en_ecarts_types():
    """Normaliser par sigma·racine(T) plutôt que par un pourcentage du spot
    rend la pente cohérente d'une maturité à l'autre : un même déplacement en
    ÉCARTS-TYPES doit coûter le même supplément de volatilité, que l'échéance
    soit courte ou longue."""
    def a_z_moins_un(sigma: float, t_years: float) -> float:
        atm = sigma + config.OPTIONS_IMPLIED_VOL_SPREAD
        strike = 100.0 * math.exp(-atm * math.sqrt(t_years))   # z = -1
        return bs.quoted_implied_vol(sigma, 100.0, strike, t_years) - atm

    court = a_z_moins_un(0.30, 0.25)
    long = a_z_moins_un(0.30, 3.0)
    nerveux = a_z_moins_un(0.70, 1.0)
    assert court == pytest.approx(config.OPTIONS_VOL_SKEW_SLOPE)
    assert long == pytest.approx(config.OPTIONS_VOL_SKEW_SLOPE)
    assert nerveux == pytest.approx(config.OPTIONS_VOL_SKEW_SLOPE)


def test_desactivable_par_configuration():
    """Écart et pente à zéro reproduisent exactement le comportement d'avant --
    ce sont des hypothèses, elles doivent pouvoir être retirées."""
    assert bs.quoted_implied_vol(
        0.30, 100.0, 70.0, 2.0, spread=0.0, skew_slope=0.0) == pytest.approx(0.30)


def test_cas_degeneres_rendent_la_realisee():
    """Maturité nulle, prix ou volatilité non positifs : la formule est
    indéfinie, et l'interception du pricing dégénéré appartient à bs_price."""
    assert bs.quoted_implied_vol(0.30, 100.0, 100.0, 0.0) == 0.30
    assert bs.quoted_implied_vol(0.0, 100.0, 100.0, 2.0) == 0.0
    assert bs.quoted_implied_vol(0.30, 0.0, 100.0, 2.0) == 0.30
    assert bs.quoted_implied_vol(0.30, 100.0, 0.0, 2.0) == 0.30


def test_le_moteur_ouvre_a_la_volatilite_cotee():
    """Bout en bout : la position ouverte porte la volatilité COTÉE, pas la
    réalisée -- c'est elle qui a fixé la prime payée, et c'est elle que
    `vol_ratio` fera vivre ensuite sans saut de P&L."""
    panel = cours({"AAA": serie_bruitee(100.0, 200, graine=11)})
    evenements = signaux(["AAA"], "2020-01-02", theoretical=200.0)
    engine = moteur(panel, evenements, {"AAA": "CALL"}, vol_mode="rolling")
    engine.run()

    position = engine.positions.get("AAA")
    assert position is not None
    assert position.source == "simulated"

    # La volatilité d'entrée est la COTÉE, dérivée de la réalisée du jour.
    realisee = engine.prices.realized_vol_at(
        "AAA", position.entry_date, engine.realized_vol_lookback_days)
    attendue = bs.quoted_implied_vol(
        realisee if realisee else config.OPTIONS_FALLBACK_VOL,
        position.entry_spot, position.strike,
        (position.expiry - position.entry_date).days / 365.0,
    )
    assert position.vol == pytest.approx(attendue)
    assert position.vol > (realisee or config.OPTIONS_FALLBACK_VOL)

    # Et `vol_ratio` la fait vivre sans saut : au jour de l'entrée, le
    # repricing doit redonner exactement la volatilité d'entrée.
    assert engine._repricing_vol(position, position.entry_date) == pytest.approx(position.vol)
