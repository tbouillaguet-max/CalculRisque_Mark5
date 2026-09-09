"""Une prime Black-Scholes sous-coulée (subnormale, non nulle) ne doit jamais
faire déborder un dimensionnement vers l'infini.

Reproduit un OverflowError réel : `python 13_diagnostic_friction.py` sur
données de production a planté dans `_deploy_idle_cash` ("cannot convert
float infinity to integer"). Cause : `math.erfc` (options_pricing._norm_cdf)
traverse une bande de flottants subnormaux (~1e-310 à 5e-324) avant de tomber
exactement à 0.0 -- une prime de cette bande passe le test `> 0` mais fait
déborder `share / (multiplier x premium)` avant même d'atteindre
`_round_contracts`."""

from __future__ import annotations

import math

import pytest

from backtest.options_engine import MIN_PREMIUM_FOR_SIZING, OptionPosition
from tests.options_harness import cours, moteur, serie_bruitee, signaux

N_JOURS = 60


@pytest.fixture
def panel():
    return cours({"AAA": serie_bruitee(100.0, N_JOURS, graine=1)})


@pytest.fixture
def evenements():
    return signaux(["AAA"], "2020-01-06")


def _position_prime_subnormale(engine, panel) -> OptionPosition:
    """Un CALL dont le strike tombe pile dans la bande subnormale de
    Black-Scholes (cf. docstring du module) -- trouvé par bissection sur le
    VRAI pricer plutôt qu'inventé, pour que le test reproduise le mécanisme
    réel et reste robuste à un changement de dates ou de formule.

    En production ce strike vient de gap_basis="log" (non borné :
    valuation_theoretical_per_share peut être aberrante) ; ici on le cherche
    directement plutôt que de fabriquer un signal aberrant."""
    today = panel.close.index[10]
    # Le spot d'OUVERTURE réel de ce jour -- pas une valeur ronde -- car
    # _deploy_idle_cash repricera avec exactement ce nombre : la bissection
    # doit chercher la frontière sur le même spot pour rester valide.
    spot = engine.prices.open_at("AAA", today)
    pos = OptionPosition(
        symbol="AAA", option_type="CALL", strike=spot, expiry=panel.close.index[-1],
        contracts=10.0, entry_premium=0.01, entry_date=panel.close.index[0],
        vol=0.05, multiplier=100.0, source="simulated", entry_spot=spot,
        stop_reference_premium=0.01, stop_reference_spot=spot,
    )
    lo, hi = spot, spot * 1e10
    for _ in range(200):
        mid = (lo + hi) / 2
        pos.strike = mid
        premium = engine._current_premium(pos, today, spot=spot)
        if premium and premium > 0:
            lo = mid
        else:
            hi = mid
    pos.strike = lo
    engine.positions["AAA"] = pos
    return pos


def test_le_pricer_produit_bien_une_prime_subnormale_sur_ce_cas(panel, evenements):
    """Garde-fou du test lui-même : si Black-Scholes cessait de sous-couler
    sur ce strike (changement de formule, de vol...), les tests suivants ne
    prouveraient plus rien -- ce test le dit explicitement plutôt qu'un faux
    négatif silencieux."""
    engine = moteur(panel, evenements, {"AAA": "CALL"})
    pos = _position_prime_subnormale(engine, panel)
    today = panel.close.index[10]
    premium = engine._current_premium(pos, today, spot=engine.prices.open_at("AAA", today))
    assert premium is not None
    assert 0 < premium < MIN_PREMIUM_FOR_SIZING, (
        f"prime={premium!r} : ajuster le strike du fixture pour retomber dans la bande subnormale"
    )


def test_le_redeploiement_ne_plante_pas_sur_une_prime_subnormale(panel, evenements):
    """Le crash exact rapporté : OverflowError dans _deploy_idle_cash."""
    engine = moteur(panel, evenements, {"AAA": "CALL"}, min_deployment_pct=90.0)
    _position_prime_subnormale(engine, panel)
    cash_avant = engine.cash

    engine._deploy_idle_cash(panel.close.index[10])  # ne doit pas lever OverflowError

    # Position invendable : aucun renforcement n'a dû avoir lieu dessus.
    assert engine.cash == pytest.approx(cash_avant)
    assert engine.positions["AAA"].contracts == pytest.approx(10.0)


def test_round_contracts_neutralise_l_infini_plutot_que_de_planter(panel, evenements):
    """Filet de sécurité direct sur le point de passage unique avant int()."""
    engine = moteur(panel, evenements, {"AAA": "CALL"})
    assert engine._round_contracts(float("inf")) == 0.0
    assert engine._round_contracts(float("-inf")) == 0.0
    assert engine._round_contracts(float("nan")) == 0.0
    assert engine._round_contracts(3.6) == 4.0  # comportement normal inchangé


def test_une_prime_ordinaire_continue_de_se_redeployer(panel, evenements):
    """Le garde-fou ne doit pas bloquer un renforcement légitime : seule la
    bande subnormale est écartée, pas toute prime faible."""
    engine = moteur(panel, evenements, {"AAA": "CALL"}, min_deployment_pct=90.0)
    engine.positions["AAA"] = OptionPosition(
        symbol="AAA", option_type="CALL", strike=100.0, expiry=panel.close.index[-1],
        contracts=1.0, entry_premium=5.0, entry_date=panel.close.index[0],
        vol=0.25, multiplier=100.0, source="simulated", entry_spot=100.0,
        stop_reference_premium=5.0, stop_reference_spot=100.0,
    )
    cash_avant = engine.cash
    engine._deploy_idle_cash(panel.close.index[10])
    assert engine.cash < cash_avant, "le redéploiement normal doit toujours avoir lieu"
