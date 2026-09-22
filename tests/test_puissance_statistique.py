"""Outils qui disent si un écart de Sharpe VEUT DIRE quelque chose.

CE QUI A MOTIVÉ CE MODULE. L'étude des réglages actions a produit un gain de
+0,08 de Sharpe. Lu à l'aune de l'erreur-type d'un Sharpe sur sept ans (0,46),
ce gain est indiscernable de zéro -- et cette lecture est fausse : elle vaut
pour deux stratégies INDÉPENDANTES, alors que deux variantes d'un même
backtest ont des courbes corrélées à 0,97. L'écart qui les sépare est APPARIÉ,
et sa dispersion est bien plus faible que celle de chacun de ses termes.
Sans test apparié, tout est « non significatif » et plus aucune décision n'est
possible.

Trois choses sont vérifiées ici : que le test apparié détecte un vrai écart,
qu'il n'en invente pas là où il n'y en a pas, et que la version vectorisée de
`capped_weights` donne EXACTEMENT les mêmes poids que celle qu'elle remplace.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

import config
from backtest import metrics as metrics_mod
from backtest.strategies.base import capped_weights

_opt = importlib.import_module("16_optimize_strategie_actions")


def _serie(valeurs, debut="2015-01-01") -> pd.Series:
    dates = pd.bdate_range(debut, periods=len(valeurs))
    return pd.Series(valeurs, index=dates)


# --------------------------------------------------------------------------- #
# Bootstrap apparié
# --------------------------------------------------------------------------- #
def test_le_test_apparie_detecte_un_ecart_reel():
    """B surpasse A d'une dérive constante, sur des chocs IDENTIQUES : c'est le
    cas le plus favorable à un test apparié, et il doit conclure."""
    rng = np.random.default_rng(1)
    chocs = rng.normal(0, 0.01, 2000)
    a = _serie(chocs)
    b = _serie(chocs + 0.0004)  # même bruit, dérive en plus

    resultat = metrics_mod.paired_sharpe_difference(a, b, n_bootstrap=800)
    assert resultat["sharpe_difference"] > 0
    assert resultat["p_value"] < 0.05
    assert resultat["difference_ci_low"] > 0
    assert resultat["correlation"] > 0.99


def test_le_test_apparie_n_invente_pas_d_ecart():
    """Contrôle négatif, sous forme de CALIBRATION et non d'un cas unique.

    Une paire unique de séries indépendantes ne prouve rien : un intervalle à
    95% exclut zéro une fois sur vingt PAR CONSTRUCTION, et c'est exactement ce
    qu'on observe pour certaines graines (à la graine 2, deux tirages de la
    même loi donnent des Sharpe de -0,42 et +0,94 -- un écart de 1,36 dû au
    seul hasard, que le test signale à juste titre comme réel POUR CET
    ÉCHANTILLON). Ce qu'il faut vérifier, c'est le TAUX de ces rejets.

    Le seuil est volontairement lâche. Mesuré sur 60 paires : environ 10% de
    rejets contre 5% nominaux -- un bootstrap par blocs est connu pour être
    légèrement libéral, et 40 paires ne permettent pas de trancher plus fin.
    À retenir en lisant une p-value marginale : elle est un peu optimiste."""
    rejets = 0
    essais = 40
    for graine in range(essais):
        rng = np.random.default_rng(5000 + graine)
        a = _serie(rng.normal(0.0003, 0.01, 1500))
        b = _serie(rng.normal(0.0003, 0.01, 1500))
        resultat = metrics_mod.paired_sharpe_difference(a, b, n_bootstrap=300)
        if resultat["difference_ci_low"] > 0 or resultat["difference_ci_high"] < 0:
            rejets += 1

    assert rejets / essais <= 0.25, (
        f"{rejets}/{essais} rejets sur des séries sans écart : le test sur-rejette"
    )


def test_l_appariement_resserre_vraiment_l_intervalle():
    """LE point du module : à écart identique, deux séries corrélées doivent
    donner un intervalle plus étroit que deux séries indépendantes. C'est
    exactement l'information que l'erreur-type marginale jette."""
    rng = np.random.default_rng(3)
    chocs = rng.normal(0, 0.01, 2000)
    a = _serie(chocs)
    couplee = _serie(chocs + 0.0004)
    independante = _serie(rng.normal(0.0004, 0.01, 2000))

    large = metrics_mod.paired_sharpe_difference(a, independante, n_bootstrap=800)
    etroit = metrics_mod.paired_sharpe_difference(a, couplee, n_bootstrap=800)

    largeur = lambda r: r["difference_ci_high"] - r["difference_ci_low"]  # noqa: E731
    assert largeur(etroit) < largeur(large)


def test_une_serie_trop_courte_ne_rend_rien_plutot_qu_un_chiffre_faux():
    court = _serie(np.zeros(10))
    assert metrics_mod.paired_sharpe_difference(court, court) == {}


def test_daily_returns_of_decoupe_la_fenetre():
    courbe = pd.DataFrame({
        "date": pd.bdate_range("2015-01-01", periods=400),
        "nav": np.linspace(1e6, 2e6, 400),
    })
    complet = metrics_mod.daily_returns_of(courbe)
    fenetre = metrics_mod.daily_returns_of(courbe, start="2016-01-01")

    assert len(complet) == 399
    assert len(fenetre) < len(complet)
    assert fenetre.index.min() > pd.Timestamp("2016-01-01")


# --------------------------------------------------------------------------- #
# capped_weights vectorisée : mêmes poids, au bit près
# --------------------------------------------------------------------------- #
def _capped_weights_reference(conviction, cap_pct=None, max_iter=20):
    """Implémentation d'ORIGINE, en pandas. Conservée ici et nulle part
    ailleurs : elle est l'étalon de la version vectorisée, qui n'a le droit
    d'être plus rapide qu'à condition d'être identique."""
    cap = config.BACKTEST_MAX_WEIGHT_PER_POSITION_PCT if cap_pct is None else cap_pct
    total = conviction.sum()
    if total <= 0:
        return conviction
    weights = conviction / total
    if not cap or cap <= 0:
        return weights
    cap = cap / 100
    if cap * len(weights) <= 1:
        return pd.Series(cap, index=weights.index)
    for _ in range(max_iter):
        over = weights > cap
        if not over.any():
            break
        excess = float((weights[over] - cap).sum())
        weights = weights.where(~over, cap)
        under = ~over
        room = float(weights[under].sum())
        if room <= 0:
            break
        weights = weights.where(over, weights + excess * weights / room)
    return weights


@pytest.mark.parametrize("graine", [0, 1, 2, 3, 4])
def test_la_version_vectorisee_est_identique_a_l_originale(graine):
    """Sur des tirages aléatoires, écarts ABERRANTS compris -- l'historique en
    contient à plusieurs milliers de pour cent, et c'est précisément là qu'une
    réécriture d'un point fixe peut diverger."""
    rng = np.random.default_rng(graine)
    for _ in range(400):
        n = int(rng.integers(1, 150))
        loi = rng.integers(0, 3)
        if loi == 0:
            valeurs = rng.uniform(0.1, 100, n)
        elif loi == 1:
            valeurs = rng.lognormal(0, 3, n)
        else:
            valeurs = rng.uniform(1, 50, n)
            valeurs[0] = rng.uniform(1e3, 1e6)
        serie = pd.Series(valeurs, index=[f"S{i}" for i in range(n)], name="w")
        cap = float(rng.choice([0, 5, 10, 20, 33.3, 100]))

        attendu = _capped_weights_reference(serie.copy(), cap)
        obtenu = capped_weights(serie.copy(), cap)
        assert (attendu - obtenu).abs().max() == 0.0, f"divergence pour n={n}, cap={cap}"


def test_le_plafond_est_respecte_et_la_somme_ne_depasse_jamais_un():
    serie = pd.Series(np.arange(1, 21, dtype=float), index=[f"S{i}" for i in range(20)])
    poids = capped_weights(serie, 10.0)
    assert poids.max() <= 0.10 + 1e-12
    assert poids.sum() <= 1.0 + 1e-12


# --------------------------------------------------------------------------- #
# Fenêtres glissantes
# --------------------------------------------------------------------------- #
def test_les_fenetres_glissantes_ne_se_recouvrent_pas_et_couvrent_la_fin():
    debut, fin = pd.Timestamp("2015-01-01"), pd.Timestamp("2026-09-04")
    fenetres = _opt.fenetres_walk_forward(debut, fin, annees_apprentissage=5, annees_test=1)

    assert len(fenetres) >= 5
    for (_, coupure, fin_test) in fenetres:
        assert coupure < fin_test <= fin
    # Les fenêtres de test s'enchaînent sans trou ni recouvrement : chaque
    # séance hors échantillon est comptée une fois et une seule.
    for precedente, suivante in zip(fenetres, fenetres[1:]):
        assert precedente[2] == suivante[1]


def test_chaque_apprentissage_precede_son_test():
    """La propriété qui fait tout : aucune fenêtre ne peut choisir son réglage
    sur des données postérieures à celles qui la jugent."""
    fenetres = _opt.fenetres_walk_forward(
        pd.Timestamp("2015-01-01"), pd.Timestamp("2026-01-01"), 5, 1)
    for debut, coupure, fin_test in fenetres:
        assert debut < coupure < fin_test


def test_une_periode_trop_courte_ne_produit_aucune_fenetre():
    assert _opt.fenetres_walk_forward(
        pd.Timestamp("2015-01-01"), pd.Timestamp("2017-01-01"), 5, 1) == []
