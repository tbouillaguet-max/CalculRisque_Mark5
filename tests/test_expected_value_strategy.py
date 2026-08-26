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
        premium = options_pricing.bs_price(SPOT, strike, t_years, SIGMA, "CALL", r=0.04, q=0.0)
        optimum = expected_value.kelly_optimum(SPOT, strike, mu, SIGMA, t_years, "CALL", premium)
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

def test_l_implicite_reelle_prime_sur_la_realisee():
    """Sélectionner sur la réalisée un contrat que le marché price à une tout
    autre implicite reviendrait à trouver des bonnes affaires partout où le
    marché anticipe simplement plus de mouvement."""
    strategie = ValuationGapExpectedValueOptionsStrategy()
    strategie.bind_market_context(_contexte(sigma=0.20, implied=0.60))
    strategie.generate_option_targets(_signaux(), {})

    assert strategie.diagnostics()["expected_value_implied_vol_count"] == 1
    assert strategie.diagnostics()["expected_value_realized_vol_count"] == 0
    assert strategie.diagnostics()["expected_value_implied_vol_pct"] == 100.0


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
    # Le strike doit être celui qu'on obtiendrait à la volatilité de repli.
    attendu = expected_value.optimal_strike(
        SPOT,
        expected_value.convergence_drift(
            SPOT, 150.0, strategie.tenor_days / 365.0, strategie.convergence_fraction,
        ),
        config.OPTIONS_FALLBACK_VOL, strategie.tenor_days / 365.0, "CALL",
        expected_value.strike_grid(SPOT, config.OPTIONS_FALLBACK_VOL, strategie.tenor_days / 365.0),
        r=0.04, q=0.0,
    )["strike"]
    assert _strike_transmis(cibles["AAA"]) == pytest.approx(attendu)


# --------------------------------------------------------------------------- #
# Grille contrainte par les strikes cotés
# --------------------------------------------------------------------------- #

def test_la_grille_se_restreint_aux_strikes_cotes():
    """Sans cette contrainte, l'optimisation choisirait un strike que personne
    ne cotait, et le moteur se rabattrait silencieusement sur un autre.

    Fraction POSÉE ICI, alors que le sujet du test n'est pas la convergence :
    ce test a seulement besoin qu'une thèse viable existe, et au défaut le plus
    bas admissible la thèse de `_signaux()` n'a plus d'espérance nette positive
    (dérive 2 %/an contre un taux sans risque à 4 %). Le laisser suivre
    config.OPTIONS_EV_CONVERGENCE_FRACTION_DEFAULT le ferait tomber sur un
    `KeyError` en disant « la grille cotée est cassée », alors que la ligne
    aurait simplement été écartée pour une raison sans rapport."""
    cotes = [90.0, 110.0, 130.0]
    strategie = ValuationGapExpectedValueOptionsStrategy(convergence_fraction=0.5)
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
    aucune mise log-optimale n'existe, la ligne est écartée et comptée.

    `convergence_fraction=1.0` est posée EXPLICITEMENT, et c'est le point du
    test : c'est l'hypothèse la plus généreuse que la stratégie accepte (le
    cours atteint exactement sa valeur théorique à l'échéance). Écarter la
    ligne sous cette hypothèse-là l'écarte sous toutes les autres, puisque
    l'espérance nette croît avec la fraction (cf. le test de monotonie
    ci-dessous). Sans cette fixation, le test suivait
    config.OPTIONS_EV_CONVERGENCE_FRACTION_DEFAULT et rejetait ce contrat avec
    une marge de 7 MILLIÈMES de dollar sur une prime de 97 $ : le porter de 0,5
    à 0,8 a suffi à faire changer le signe de l'espérance nette et à faire
    tomber le test, sans qu'aucun code de production ne soit en cause."""
    strategie = ValuationGapExpectedValueOptionsStrategy(
        entry_threshold_pct=0.5, convergence_fraction=1.0,
    )
    strategie.bind_market_context(_contexte(sigma=SIGMA))

    cibles = strategie.generate_option_targets(_signaux(theoretical=100.5), {})
    assert cibles == {}
    assert strategie.diagnostics()["dropped_expected_value_negative_count"] == 1


def test_la_garde_d_esperance_est_monotone_en_fraction_de_convergence():
    """L'invariant qui rend le test précédent concluant : l'espérance nette
    d'un contrat CROÎT avec la fraction de convergence (elle ne dépend de la
    fraction que par la dérive `mu`, et E[payoff] est croissante en `mu`).

    Une thèse écartée à fraction 1,0 est donc écartée à toute fraction plus
    faible, et une thèse retenue à une fraction donnée l'est à toute fraction
    plus forte. C'est ce qui permet de tester la garde une seule fois, au cas
    le plus favorable, plutôt que de la re-calibrer à chaque changement du
    défaut de config."""
    retenues = []
    for fraction in (0.1, 0.25, 0.5, 0.8, 1.0):
        strategie = ValuationGapExpectedValueOptionsStrategy(
            # Plancher de matérialité DÉSACTIVÉ : ce test porte sur la garde
            # d'espérance et sur elle seule. Les deux garde-fous écartent la
            # même thèse marginale, et les laisser agir ensemble ferait mesurer
            # leur composition au lieu de la monotonie recherchée.
            entry_threshold_pct=0.5, convergence_fraction=fraction, min_kelly_fraction=0.0,
        )
        strategie.bind_market_context(_contexte(sigma=SIGMA))
        retenues.append(bool(strategie.generate_option_targets(_signaux(theoretical=103.0), {})))

    # Aucune fraction plus forte ne peut faire écarter ce qu'une plus faible
    # retenait : la suite des booléens est croissante.
    assert retenues == sorted(retenues), retenues
    # Et la thèse est choisie pour que la bascule tombe DANS la plage balayée :
    # une suite constante satisferait la croissance sans rien démontrer.
    assert len(set(retenues)) == 2, (
        f"aucune bascule sur cette thèse ({retenues}) : le banc ne teste plus "
        "la monotonie, seulement une constante"
    )


# --------------------------------------------------------------------------- #
# Plancher de matérialité sur la mise log-optimale
# --------------------------------------------------------------------------- #

def _contrat_retenu(sigma, theoretical, **kwargs):
    """(cibles, f* du contrat que l'optimisation retiendrait sans plancher)."""
    strategie = ValuationGapExpectedValueOptionsStrategy(entry_threshold_pct=0.5, **kwargs)
    strategie.bind_market_context(_contexte(sigma=sigma))
    cibles = strategie.generate_option_targets(_signaux(theoretical=theoretical), {})
    return strategie, cibles


def _f_star_sans_plancher(sigma, theoretical, fraction):
    t_years = ValuationGapExpectedValueOptionsStrategy().tenor_days / 365.0
    mu = expected_value.convergence_drift(SPOT, theoretical, t_years, fraction)
    resultat = expected_value.optimal_strike(
        SPOT, mu, sigma, t_years, "CALL",
        expected_value.strike_grid(SPOT, sigma, t_years), r=0.04, q=0.0,
    )
    return None if resultat is None else resultat["kelly_fraction"]


def test_une_mise_log_optimale_sous_le_plancher_n_ouvre_rien():
    """Le défaut corrigé : la contrainte d'espérance positive est un test de
    SIGNE. Un contrat d'espérance nette +0,15 % de la prime la passait, alors
    que Kelly y dimensionne 0,05 % du capital -- et la stratégie l'ouvrait au
    poids de portefeuille ordinaire, 400 fois la taille recommandée."""
    sigma, theoretical, fraction = 0.80, 100.5, 0.8

    # Sans plancher, cette ligne est bien ouverte, et sur une mise dérisoire :
    # c'est ce qui rend le test concluant plutôt que tautologique.
    f_star = _f_star_sans_plancher(sigma, theoretical, fraction)
    assert f_star is not None and f_star < 0.01, f_star
    _, sans = _contrat_retenu(sigma, theoretical, convergence_fraction=fraction, min_kelly_fraction=0.0)
    assert "AAA" in sans

    strategie, avec = _contrat_retenu(
        sigma, theoretical, convergence_fraction=fraction, min_kelly_fraction=0.05,
    )
    assert avec == {}
    assert strategie.diagnostics()["dropped_kelly_below_floor_count"] == 1
    # Compté À PART du rejet pour espérance négative : les deux diagnostics ne
    # disent pas la même chose sur le signal.
    assert strategie.diagnostics()["dropped_expected_value_negative_count"] == 0


def test_une_these_franche_n_est_pas_touchee_par_le_plancher():
    """Le plancher ne doit pas se transformer en filtre général : sur une thèse
    franche, la mise log-optimale est largement au-dessus et la ligne passe."""
    strategie, cibles = _contrat_retenu(SIGMA, 150.0, min_kelly_fraction=0.05)
    assert "AAA" in cibles
    diag = strategie.diagnostics()
    assert diag["dropped_kelly_below_floor_count"] == 0
    # Et la marge est reportée, pour qu'un plancher trop bas se voie.
    assert diag["min_kelly_fraction_retained"] > 0.05
    assert diag["min_kelly_fraction_floor"] == 0.05


def test_le_plancher_desactive_retablit_le_comportement_anterieur():
    strategie, cibles = _contrat_retenu(0.80, 100.5, convergence_fraction=0.8, min_kelly_fraction=0.0)
    assert "AAA" in cibles
    assert strategie.diagnostics()["dropped_kelly_below_floor_count"] == 0


def test_le_plancher_est_monotone():
    """Monter le plancher ne peut qu'écarter davantage : la propriété qui rend
    le paramètre optimisable sans surprise."""
    ouvertes = []
    for plancher in (0.0, 0.01, 0.05, 0.20, 0.60, 0.95):
        _, cibles = _contrat_retenu(
            0.50, 130.0, convergence_fraction=0.8, min_kelly_fraction=plancher,
        )
        ouvertes.append(bool(cibles))
    assert ouvertes == sorted(ouvertes, reverse=True), ouvertes
    assert len(set(ouvertes)) == 2, (
        f"aucune bascule ({ouvertes}) : le banc ne teste plus la monotonie"
    )


def test_le_plancher_est_valide():
    for invalide in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            ValuationGapExpectedValueOptionsStrategy(min_kelly_fraction=invalide)
    # 0 (désactivé) et une valeur strictement sous 1 restent admises.
    ValuationGapExpectedValueOptionsStrategy(min_kelly_fraction=0.0)
    ValuationGapExpectedValueOptionsStrategy(min_kelly_fraction=0.999)


def test_le_plancher_par_defaut_ne_mord_pas_au_seuil_d_entree_usuel():
    """Garde-fou de CALIBRAGE : le défaut de config est censé ne retirer que la
    queue dégénérée. Si un futur réglage le faisait mordre sur une thèse qui
    passe tout juste le seuil d'entrée de production, à une volatilité
    ordinaire, ce serait un changement de politique -- pas un garde-fou."""
    t_years = ValuationGapExpectedValueOptionsStrategy().tenor_days / 365.0
    seuil = math.exp(config.OPTIONS_MULTIPLES_ENTRY_THRESHOLD_PCT / 100)
    for sigma in (0.15, 0.25, 0.35, 0.50):
        f_star = _f_star_sans_plancher(
            sigma, SPOT * seuil, config.OPTIONS_EV_CONVERGENCE_FRACTION_DEFAULT,
        )
        assert f_star is not None, sigma
        assert f_star >= config.OPTIONS_EV_MIN_KELLY_FRACTION, (
            f"à sigma={sigma}, f*={f_star:.4f} tombe sous le plancher par défaut "
            f"{config.OPTIONS_EV_MIN_KELLY_FRACTION} pour une thèse au seuil d'entrée"
        )
    assert t_years > 0


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

    # Fraction LUE SUR LA STRATÉGIE, jamais recopiée : ce test refait à la main
    # le calcul que la stratégie vient de faire, donc il doit partir des mêmes
    # paramètres qu'elle. Un littéral (0.5, la valeur de
    # config.OPTIONS_EV_CONVERGENCE_FRACTION_DEFAULT à l'époque) faisait
    # comparer le strike du moteur à l'optimum d'une AUTRE dérive dès que ce
    # défaut bougeait -- porté à 0.8, l'écart mesuré passait de 0,22 $ à 53 $
    # et le test tombait, sans qu'aucun code de production ne soit en cause.
    mu = expected_value.convergence_drift(
        spot_decision, 180.0, t_years, strategie.convergence_fraction,
    )
    r, q = engine._pricing_context("AAA", dates[1])
    optimal = expected_value.optimal_strike(
        spot_decision, mu, sigma, t_years, "CALL",
        expected_value.strike_grid(spot_decision, sigma, t_years), r=r, q=q,
    )["strike"]

    derive = abs(position.strike - optimal)
    # Idem pour le pas de grille : config.OPTIONS_EV_STRIKE_GRID_STEP_SIGMA, pas
    # une copie.
    pas_de_grille = optimal * (
        math.exp(config.OPTIONS_EV_STRIKE_GRID_STEP_SIGMA * sigma * math.sqrt(t_years)) - 1.0
    )

    # La dérive a une FORME FERMÉE, et la vérifier vaut mieux que la borner à
    # vue : le moteur applique la moneyness au spot d'EXÉCUTION
    # (options_engine._select_contract, `target_strike = strike_moneyness * spot`),
    # donc le strike ouvert vaut K* x S_exécution / S_décision, et l'écart est
    # exactement proportionnel au mouvement de nuit.
    #
    # L'ancienne borne, `derive < |S_exécution - S_décision|`, revenait par ce
    # calcul à exiger K* < S_décision, c'est-à-dire un optimum DANS la monnaie
    # -- une propriété de la thèse testée, pas de la transmission du strike.
    # Elle tombait dès que l'optimum remontait à la monnaie (mesuré : égalité
    # stricte à fraction de convergence 1,0, où K* est exactement le point ATM
    # de la grille).
    assert derive == pytest.approx(optimal * abs(spot_execution - spot_decision) / spot_decision)
    # C'est ici la vraie conclusion : cette dérive reste très en dessous du pas
    # de grille, donc le contrat que le moteur ouvre est bien celui que
    # l'optimisation a choisi. Mesuré sur ce jeu, à fraction 0,8 : dérive
    # 0,33 $ pour un pas de grille de 9,1 $, soit 3,7 %.
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
