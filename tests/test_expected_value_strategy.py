"""Stratégie « espérance de gain » branchée sur le moteur.

Le module de maths est testé isolément (tests/test_expected_value.py). Ce
fichier-ci vérifie l'INTÉGRATION, et en particulier l'astuce du strike : le
moteur ne sait pas recevoir un strike, seulement un prix de référence dont il
fait la moyenne avec le spot. La stratégie inverse cette formule, et rien
d'autre ne garantit que l'inversion tienne.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

import config
from backtest import expected_value, options_pricing
from backtest.strategies.options_base import MarketContext
from backtest.strategies.valuation_gap_expected_value_options import (
    ValuationGapExpectedValueOptionsStrategy,
)
from tests.options_harness import cours, moteur, serie_bruitee, signaux

SIGMA = 0.35
SPOT = 100.0


def _contexte(sigma=SIGMA, implied=None, quoted=(), today="2020-06-01") -> MarketContext:
    return MarketContext(
        today=pd.Timestamp(today),
        realized_vol=lambda symbol: sigma,
        pricing_context=lambda symbol: (0.04, 0.0),
        implied_vol=lambda symbol, option_type, tenor: implied,
        quoted_strikes=lambda symbol, option_type, tenor: list(quoted),
    )


def _signaux(theoretical=150.0, spot=SPOT, symbols=("AAA",)) -> pd.DataFrame:
    df = signaux(symbols, "2020-05-01", theoretical=theoretical)
    return df.assign(close=spot)


def _strategie(**kwargs) -> ValuationGapExpectedValueOptionsStrategy:
    strategie = ValuationGapExpectedValueOptionsStrategy(**kwargs)
    strategie.bind_market_context(_contexte(**kwargs.pop("_contexte", {})))
    return strategie


# --------------------------------------------------------------------------- #
# Transmission du strike (en MONEYNESS)
# --------------------------------------------------------------------------- #

def _strike_transmis(cible: dict, spot: float = SPOT) -> float:
    """Strike que le moteur retiendra : moneyness x spot d'exécution."""
    return cible["strike_moneyness"] * spot


def _strike_attendu(
    strategie, spot: float = SPOT, theoretical: float = 150.0,
    sigma_p: float = SIGMA, iv_marche: float | None = None,
    option_type: str = "CALL", r: float = 0.04, q: float = 0.0,
    strikes=None,
):
    """Contrôle indépendant du strike que la stratégie doit retenir, refait à
    la main avec les DEUX mesures (cf. _physical_volatility) :

        sigma_P  -> loi de S_T : espérance, variance, taux de croissance
        sigma_Q  -> prime payée : implicite cotée si le snapshot en donne une,
                    sinon modélisée par options_pricing.quoted_implied_vol

    Recopier une seule volatilité des deux côtés, comme le faisaient ces tests,
    revient à supposer que l'on paie la volatilité que le titre réalise -- soit
    exactement l'hypothèse que la séparation des mesures a supprimée."""
    t_years = strategie.tenor_days / 365.0
    mu = expected_value.convergence_drift(
        spot, theoretical, t_years, strategie.convergence_fraction)

    def prime(strike: float) -> float:
        sigma_q = (
            iv_marche if iv_marche is not None
            else options_pricing.quoted_implied_vol(sigma_p, spot, strike, t_years)
        )
        return options_pricing.bs_price(spot, strike, t_years, sigma_q, option_type, r=r, q=q)

    resultat = expected_value.optimal_strike(
        spot, mu, sigma_p, t_years, option_type,
        expected_value.strike_grid(spot, sigma_p, t_years) if strikes is None else strikes,
        r=r, q=q, criterion="kelly", premium_fn=prime,
    )
    return None if resultat is None else resultat["strike"]


def test_le_strike_est_transmis_en_moneyness():
    """La moneyness est relative au spot, donc valide à toute date -- y compris
    au roulement, des mois plus tard."""
    strategie = _strategie()
    cibles = strategie.generate_option_targets(_signaux(), {})

    assert "AAA" in cibles
    attendu = strategie._optimal_strike("AAA", "CALL", SPOT, 150.0)
    assert _strike_transmis(cibles["AAA"]) == pytest.approx(attendu)
    assert cibles["AAA"]["strike_moneyness"] > 0


def test_le_prix_de_reference_reste_la_valeur_theorique():
    """Il pilote la prise de gain par convergence (_convergence_fraction) et le
    rafraîchissement au roulement : lui donner autre chose qu'une valeur
    théorique casse les deux, silencieusement."""
    strategie = _strategie()
    cibles = strategie.generate_option_targets(_signaux(theoretical=150.0), {})
    assert cibles["AAA"]["strike_reference_price"] == pytest.approx(150.0)


def test_le_strike_retenu_est_bien_celui_qui_maximise_la_croissance():
    """Contrôle indépendant : on refait la sélection à la main sur la même
    grille et on doit retomber sur le même contrat."""
    strategie = _strategie()
    t_years = strategie.tenor_days / 365.0
    mu = expected_value.convergence_drift(SPOT, 150.0, t_years, strategie.convergence_fraction)

    cibles = strategie.generate_option_targets(_signaux(), {})
    strike_retenu = _strike_transmis(cibles["AAA"])

    meilleur, meilleur_score = None, -math.inf
    for strike in expected_value.strike_grid(SPOT, SIGMA, t_years):
        # sigma_Q pour la prime (modélisée, aucune implicite cotée dans ce
        # contexte), sigma_P pour la loi et pour Kelly. Les confondre ferait
        # payer la volatilité que le titre réalise, pas celle qu'on cote.
        sigma_q = options_pricing.quoted_implied_vol(SIGMA, SPOT, strike, t_years)
        premium = options_pricing.bs_price(SPOT, strike, t_years, sigma_q, "CALL", r=0.04, q=0.0)
        optimum = expected_value.kelly_optimum(
            SPOT, strike, mu, SIGMA, t_years, "CALL", premium, r=0.04)
        if optimum is not None and optimum[1] > meilleur_score:
            meilleur, meilleur_score = strike, optimum[1]

    assert strike_retenu == pytest.approx(meilleur)


def test_un_strike_profondement_dans_la_monnaie_reste_positif():
    """Le strike transmis est toujours strictement positif, quel que soit le
    spot auquel on l'applique -- c'est précisément ce que l'ancienne
    transmission par prix de référence inversé ne garantissait pas."""
    strategie = _strategie()
    cibles = strategie.generate_option_targets(_signaux(theoretical=104.0), {})
    if not cibles:
        pytest.skip("aucune ligne retenue à cette conviction")
    for spot_futur in (SPOT, SPOT / 2, SPOT / 10, SPOT * 3):
        assert _strike_transmis(cibles["AAA"], spot_futur) > 0


# --------------------------------------------------------------------------- #
# Volatilité de sélection
# --------------------------------------------------------------------------- #

def test_l_implicite_reelle_fixe_le_prix_la_realisee_fixe_la_loi():
    """LES DEUX MESURES NE S'EXCLUENT PAS -- elles répondent à deux questions
    différentes, et c'est le cœur de la sélection.

    L'implicite cotée dit ce que le contrat COÛTE ; la réalisée dit comment le
    titre BOUGE. La version précédente n'en gardait qu'une, en préférant
    l'implicite : elle supposait donc que le titre bougerait d'autant que le
    marché le facture, ce qui rend l'achat de volatilité gratuit par
    construction et efface la prime de risque de variance du calcul."""
    strategie = ValuationGapExpectedValueOptionsStrategy()
    strategie.bind_market_context(_contexte(sigma=0.20, implied=0.60))
    strategie.generate_option_targets(_signaux(), {})

    diagnostics = strategie.diagnostics()
    assert diagnostics["expected_value_implied_vol_count"] == 1   # prix
    assert diagnostics["expected_value_realized_vol_count"] == 1  # loi
    assert diagnostics["expected_value_market_priced_pct"] == 100.0
    # Résolue UNE fois par évaluation, pas une fois par strike candidat.
    assert diagnostics["expected_value_evaluations_count"] == 1


def test_une_implicite_trop_riche_fait_disparaitre_l_edge():
    """La conséquence économique de la séparation, et la raison d'être du
    changement : à thèse identique, un contrat que le marché price bien
    au-dessus de ce que le titre réalise cesse d'être achetable.

    C'est le prédicteur de rendement d'options de Goyal & Saretto (JFE, 2009),
    obtenu sans filtre ajouté -- il tombe de la condition d'existence de Kelly,
    E[R] > 0.

    GRILLE RESTREINTE À LA MONNAIE, et c'est indispensable : sur une grille
    complète, une implicite délirante ne ferme pas la ligne, elle la POUSSE
    très dans la monnaie (mesuré : moneyness 0,23), là où le contrat n'a
    presque plus de vega et où le prix ne dépend donc plus guère de la
    volatilité. Le comportement est correct -- « si la volatilité est hors de
    prix, achète celle qui n'en contient pas » -- mais il masque l'effet qu'on
    veut mesurer ici, qui ne concerne que les contrats réellement porteurs de
    volatilité. Voir le test suivant pour cette échappatoire, testée pour
    elle-même."""
    signaux = _signaux(theoretical=130.0)

    bon_marche = ValuationGapExpectedValueOptionsStrategy()
    bon_marche.bind_market_context(_contexte(sigma=0.35, implied=0.30, quoted=(100.0,)))
    assert bon_marche.generate_option_targets(signaux, {}) != {}

    hors_de_prix = ValuationGapExpectedValueOptionsStrategy()
    hors_de_prix.bind_market_context(_contexte(sigma=0.35, implied=1.50, quoted=(100.0,)))
    assert hors_de_prix.generate_option_targets(signaux, {}) == {}
    assert hors_de_prix.diagnostics()["dropped_expected_value_negative_count"] == 1


def test_une_implicite_trop_riche_pousse_vers_le_moins_de_vega():
    """L'échappatoire du test précédent, vérifiée pour elle-même : laissée
    libre de sa grille, la stratégie répond à une volatilité hors de prix en
    choisissant un strike bien plus dans la monnaie -- donc bien moins sensible
    à la volatilité -- plutôt qu'en renonçant.

    C'est économiquement juste, et c'est aussi ce qui explique qu'un filtre de
    richesse de volatilité ne suffirait pas à lui seul : il faut que le critère
    voie le PRIX, ce qu'il fait ici."""
    signaux = _signaux(theoretical=130.0)

    bon_marche = ValuationGapExpectedValueOptionsStrategy()
    bon_marche.bind_market_context(_contexte(sigma=0.35, implied=0.30))
    cible_bon_marche = bon_marche.generate_option_targets(signaux, {})["AAA"]

    hors_de_prix = ValuationGapExpectedValueOptionsStrategy()
    hors_de_prix.bind_market_context(_contexte(sigma=0.35, implied=1.50))
    cible_hors_de_prix = hors_de_prix.generate_option_targets(signaux, {})["AAA"]

    assert cible_hors_de_prix["strike_moneyness"] < cible_bon_marche["strike_moneyness"]


def test_sans_volatilite_on_se_replie_sur_celle_du_moteur():
    """Écarter la ligne serait plus prudent en apparence, mais biaiserait la
    sélection vers les seuls titres à long historique -- alors que le moteur
    sait parfaitement ouvrir la position, au repli de config.OPTIONS_FALLBACK_VOL."""
    strategie = ValuationGapExpectedValueOptionsStrategy()
    strategie.bind_market_context(MarketContext(
        today=pd.Timestamp("2020-06-01"),
        realized_vol=lambda symbol: None,
        pricing_context=lambda symbol: (0.04, 0.0),
        implied_vol=lambda symbol, option_type, tenor: None,
        quoted_strikes=lambda symbol, option_type, tenor: [],
    ))
    cibles = strategie.generate_option_targets(_signaux(), {})

    assert "AAA" in cibles
    assert strategie.diagnostics()["expected_value_fallback_vol_count"] == 1
    # Le repli sert de sigma_P ; la prime reste modélisée par-dessus (aucune
    # implicite cotée ici), donc plus chère que le repli lui-même.
    assert _strike_transmis(cibles["AAA"]) == pytest.approx(
        _strike_attendu(strategie, sigma_p=config.OPTIONS_FALLBACK_VOL))


# --------------------------------------------------------------------------- #
# Grille contrainte par les strikes cotés
# --------------------------------------------------------------------------- #

def test_la_grille_se_restreint_aux_strikes_cotes():
    """Sans cette contrainte, l'optimisation choisirait un strike que personne
    ne cotait, et le moteur se rabattrait silencieusement sur un autre."""
    cotes = [90.0, 110.0, 130.0]
    strategie = ValuationGapExpectedValueOptionsStrategy()
    strategie.bind_market_context(_contexte(quoted=cotes))

    cibles = strategie.generate_option_targets(_signaux(), {})
    strike = _strike_transmis(cibles["AAA"])

    assert strike in cotes
    assert strategie.diagnostics()["expected_value_quoted_grid_count"] == 1


def test_des_strikes_cotes_hors_fenetre_font_retomber_sur_la_grille_theorique():
    """Mieux vaut un contrat simulé dans la fenêtre plausible qu'un contrat
    réel manifestement hors sujet."""
    strategie = ValuationGapExpectedValueOptionsStrategy()
    strategie.bind_market_context(_contexte(quoted=[5000.0, 9000.0]))

    cibles = strategie.generate_option_targets(_signaux(), {})
    strike = _strike_transmis(cibles["AAA"])

    assert strike < 1000.0
    assert strategie.diagnostics()["expected_value_theoretical_grid_count"] == 1


# --------------------------------------------------------------------------- #
# Espérance positive obligatoire
# --------------------------------------------------------------------------- #

def test_une_these_trop_faible_n_ouvre_rien_et_est_comptee():
    """Une valeur théorique à peine au-dessus du cours ne bat pas la prime :
    aucune mise log-optimale n'existe, la ligne est écartée et comptée."""
    strategie = ValuationGapExpectedValueOptionsStrategy(entry_threshold_pct=0.5)
    strategie.bind_market_context(_contexte(sigma=0.80))

    cibles = strategie.generate_option_targets(_signaux(theoretical=100.5), {})
    assert cibles == {}
    assert strategie.diagnostics()["dropped_expected_value_negative_count"] == 1


def test_la_fraction_de_convergence_est_validee():
    for invalide in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            ValuationGapExpectedValueOptionsStrategy(convergence_fraction=invalide)


def test_une_fraction_plus_forte_deplace_le_strike_vers_le_dehors():
    """Plus la convergence supposée est grande, plus la dérive est forte, donc
    plus Kelly accepte de convexité."""
    strikes = []
    for fraction in (0.2, 0.5, 1.0):
        strategie = ValuationGapExpectedValueOptionsStrategy(convergence_fraction=fraction)
        strategie.bind_market_context(_contexte())
        cibles = strategie.generate_option_targets(_signaux(theoretical=200.0), {})
        strikes.append(_strike_transmis(cibles["AAA"]))
    assert all(a <= b for a, b in zip(strikes, strikes[1:]))
    assert strikes[-1] > strikes[0]


# --------------------------------------------------------------------------- #
# Héritage : le signal doit rester rigoureusement celui de la stratégie multiples
# --------------------------------------------------------------------------- #

def test_les_candidates_et_les_poids_sont_ceux_de_la_strategie_multiples():
    """La comparaison entre les deux stratégies ne doit mesurer QUE l'effet de
    la sélection de strike."""
    from backtest.strategies.valuation_gap_multiples_options import (
        ValuationGapMultiplesOptionsStrategy,
    )

    donnees = _signaux(symbols=("AAA", "BBB", "CCC"))
    reference = ValuationGapMultiplesOptionsStrategy()
    strategie = _strategie()

    attendues = reference.generate_option_targets(donnees, {})
    obtenues = strategie.generate_option_targets(donnees, {})

    assert set(obtenues) == set(attendues)
    for symbol in attendues:
        assert obtenues[symbol]["weight"] == pytest.approx(attendues[symbol]["weight"])
        assert obtenues[symbol]["option_type"] == attendues[symbol]["option_type"]


# --------------------------------------------------------------------------- #
# Bout en bout à travers le moteur
# --------------------------------------------------------------------------- #

def test_un_run_complet_ouvre_des_positions_et_expose_ses_compteurs():
    n = 400
    panel = cours({"AAA": serie_bruitee(100.0, n, graine=1),
                   "BBB": serie_bruitee(100.0, n, graine=2)})
    evenements = signaux(["AAA", "BBB"], "2020-01-02", theoretical=180.0)

    strategie = ValuationGapExpectedValueOptionsStrategy()
    engine = moteur(panel, evenements, {}, strategy=strategie, vol_mode="rolling")
    engine.run()

    metriques = engine.diagnostics_summary() if hasattr(engine, "diagnostics_summary") else None
    assert engine.executions, "aucune exécution : le moteur n'a rien ouvert"

    # Les compteurs de la stratégie doivent avoir remonté jusqu'aux métriques.
    resume = engine._strategy_diagnostics()
    assert resume["expected_value_evaluations_count"] > 0
    assert "dropped_expected_value_negative_count" in resume
    assert metriques is None or "dropped_expected_value_negative_count" in metriques


def test_le_strike_ouvert_par_le_moteur_est_proche_du_strike_optimise():
    """Le spot de décision (clôture J) et le spot d'exécution (ouverture J+1)
    diffèrent : l'astuce 2K* - spot laisse donc dériver le strike de la MOITIÉ
    du mouvement de nuit. Ce test chiffre cette dérive et vérifie qu'elle reste
    très inférieure au pas de grille -- sans quoi l'optimisation choisirait un
    contrat et le moteur en ouvrirait un autre."""
    # Assez court pour qu'aucun roulement ne vienne remplacer le contrat.
    panel = cours({"AAA": serie_bruitee(100.0, 200, graine=7)})
    evenements = signaux(["AAA"], "2020-01-02", theoretical=180.0)

    strategie = ValuationGapExpectedValueOptionsStrategy()
    engine = moteur(panel, evenements, {}, strategy=strategie, vol_mode="rolling")
    engine.run()

    position = engine.positions.get("AAA")
    assert position is not None, "aucune position ouverte"

    # Volatilité effectivement retenue : au tout début du panel, l'historique
    # est trop court, donc le repli partagé avec le moteur.
    assert strategie.diagnostics()["expected_value_fallback_vol_count"] == 1
    sigma = config.OPTIONS_FALLBACK_VOL
    t_years = strategie.tenor_days / 365.0

    dates = panel.close.index
    spot_decision = float(panel.close.loc[dates[1], "AAA"])   # clôture du jour du signal
    spot_execution = float(panel.open.loc[dates[2], "AAA"])   # ouverture du lendemain

    # Fraction de convergence LUE SUR LA STRATÉGIE, pas recopiée : elle est
    # réglable (config.OPTIONS_EV_CONVERGENCE_FRACTION_DEFAULT, déjà passée de
    # 0,5 à 0,8) et la figer ici faisait comparer le strike réellement ouvert à
    # l'optimum d'une AUTRE thèse que celle jouée par le moteur.
    r, q = engine._pricing_context("AAA", dates[1])
    optimal = _strike_attendu(
        strategie, spot=spot_decision, theoretical=180.0, sigma_p=sigma, r=r, q=q)

    derive = abs(position.strike - optimal)
    pas_de_grille = optimal * (math.exp(0.25 * sigma * math.sqrt(t_years)) - 1.0)

    # La dérive est de l'ordre de la moitié d'un mouvement journalier...
    assert derive < abs(spot_execution - spot_decision)
    # ... donc très en dessous du pas de grille : le contrat que le moteur
    # ouvre est bien celui que l'optimisation a choisi. Mesuré sur ce jeu :
    # dérive 0,22 $ pour un pas de grille de 5,4 $, soit 4 %.
    assert derive < 0.1 * pas_de_grille


# --------------------------------------------------------------------------- #
# Régression : le roulement d'un sous-jacent effondré
# --------------------------------------------------------------------------- #

def _serie_effondree(depart: float, n: int, fin: float, graine: int = 11) -> list:
    """Cours qui décroît fortement, avec du bruit pour que la volatilité
    réalisée reste définie."""
    import numpy as np

    rng = np.random.default_rng(graine)
    tendance = np.linspace(0.0, math.log(fin / depart), n)
    bruit = np.cumsum(rng.normal(0.0, 0.012, n))
    return list(depart * np.exp(tendance + bruit - bruit[0]))


def test_un_ordre_execute_bien_plus_bas_ouvre_toujours_un_strike_valide():
    """Régression du run réel (ValueError: math domain error).

    Un ordre reste en attente tant que l'ouverture du sous-jacent est
    indisponible -- titre suspendu, par exemple -- et peut donc s'exécuter des
    jours plus tard, à un cours très inférieur à celui de la décision. Avec la
    transmission d'origine (prix de référence inversé 2K* - spot), la moyenne
    (référence + spot) / 2 passait alors sous zéro et math.log(spot / strike)
    levait ValueError en plein backtest.

    La chute nécessaire dépend de la volatilité, parce qu'elle fixe la borne
    basse de la grille : -44% à sigma = 30%, mais seulement -70% à 45% et -84%
    à 60%, la borne valant S0 x exp(-3 sigma racine(T)). Sur seize ans et une
    centaine de positions, c'est un scénario ordinaire.

    En moneyness, le strike suit le spot d'exécution par construction : il
    reste strictement positif quel que soit l'effondrement."""
    panel = cours({"AAA": serie_bruitee(100.0, 200, graine=3)})
    evenements = signaux(["AAA"], "2020-01-02", theoretical=180.0)
    strategie = ValuationGapExpectedValueOptionsStrategy()
    engine = moteur(panel, evenements, {}, strategy=strategie, vol_mode="rolling")

    today = panel.close.index[60]
    cibles = strategie.generate_option_targets(_signaux(theoretical=180.0), {})
    moneyness = cibles["AAA"]["strike_moneyness"]

    for spot_effondre in (100.0, 40.0, 10.0, 1.0):
        contrat = engine._select_contract(
            "AAA", "CALL", spot=spot_effondre, today=today,
            strike_reference_price=cibles["AAA"]["strike_reference_price"],
            strike_moneyness=moneyness,
        )
        assert contrat["strike"] == pytest.approx(moneyness * spot_effondre)
        assert contrat["strike"] > 0
        assert contrat["premium"] >= 0


def test_l_ancienne_transmission_produisait_bien_un_strike_negatif():
    """Témoin du bug corrigé : sur la même position, le prix de référence
    inversé devient négatif dès que le spot d'exécution s'écarte assez."""
    strategie = _strategie()
    cibles = strategie.generate_option_targets(_signaux(theoretical=150.0), {})
    strike = cibles["AAA"]["strike_moneyness"] * SPOT

    reference_ancienne = 2.0 * strike - SPOT
    seuil = SPOT - 2.0 * strike  # spot d'exécution sous lequel la moyenne passe à zéro

    assert (reference_ancienne + SPOT) / 2.0 == pytest.approx(strike)   # au spot de décision
    if seuil > 0:
        assert (reference_ancienne + (seuil - 1.0)) / 2.0 < 0           # plus bas : négatif
    # La transmission actuelle, elle, reste positive au même spot.
    assert cibles["AAA"]["strike_moneyness"] * max(seuil - 1.0, 0.01) > 0


def test_le_roulement_conserve_la_moneyness_optimisee():
    """Second bug du même mécanisme, silencieux celui-là : quand un signal
    frais existait, _roll_position ÉCRASAIT la référence par la valeur
    théorique, et le contrat renouvelé repartait sur le strike à mi-chemin de
    la stratégie multiples -- l'optimisation était perdue à chaque roulement."""
    panel = cours({"AAA": serie_bruitee(100.0, 400, graine=7)})
    evenements = signaux(["AAA"], "2020-01-02", theoretical=180.0)

    strategie = ValuationGapExpectedValueOptionsStrategy()
    engine = moteur(panel, evenements, {}, strategy=strategie, vol_mode="rolling")
    engine.run()

    roulements = [e for e in engine.executions if e.get("reason") == "roll"]
    assert roulements, "le scénario n'a pas déclenché de roulement"

    position = engine.positions.get("AAA")
    assert position is not None
    assert position.strike_moneyness is not None

    date_roll = max(e["date"] for e in roulements)
    spot_roll = engine.prices.open_at("AAA", date_roll)
    # Le strike du contrat renouvelé est recentré sur le cours du jour, à la
    # moneyness choisie par Kelly à l'entrée -- et non à (théorique + spot) / 2.
    assert position.strike == pytest.approx(position.strike_moneyness * spot_roll)
    mi_chemin = (180.0 + spot_roll) / 2.0
    assert abs(position.strike - mi_chemin) > 1e-6


def test_un_strike_vise_non_positif_retombe_sur_l_atm():
    """Garde-fou du moteur : une référence qui produirait un strike nul ou
    négatif ne doit jamais atteindre le logarithme de Black-Scholes."""
    panel = cours({"AAA": serie_bruitee(100.0, 120, graine=3)})
    evenements = signaux(["AAA"], "2020-01-02", theoretical=180.0)
    engine = moteur(panel, evenements, {"AAA": "CALL"})

    today = panel.close.index[60]
    contrat = engine._select_contract(
        "AAA", "CALL", spot=10.0, today=today, strike_reference_price=-500.0,
    )
    assert contrat["strike"] == pytest.approx(10.0)  # repli ATM
