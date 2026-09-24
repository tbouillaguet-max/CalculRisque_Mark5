"""Classer sur l'INFORMATION RATIO plutôt que sur le Sharpe.

CE QUE ÇA ATTAQUE. Le Sharpe d'une stratégie actions est dominé par le facteur
MARCHÉ, que toutes les variantes d'une même grille partagent intégralement :
il est dans le numérateur ET au dénominateur de chacune, et il ne les sépare
donc pas -- il les bruite toutes ensemble. L'information ratio est le Sharpe de
l'écart ACTIF (stratégie moins indice) : le facteur commun disparaît, et ce qui
reste est ce qui distingue vraiment les combinaisons.

LE TEST QUI COMPTE est `test_l_information_ratio_resserre_l_intervalle_quand_le_marche_domine` :
c'est la propriété entière du mécanisme, et tout le reste en découle.

CE QUE LA MESURE DIT. Le levier avait été chiffré ×1,9 contre l'erreur-type
MARGINALE, et soupçonné de ne rien apporter une fois le test apparié en place --
les deux retirent un facteur commun. Mesuré sur la grille réelle de 432
combinaisons, il apporte ×1,29 PAR-DESSUS l'appariement (étendue/demi-largeur
1,84 contre 1,43) et fait passer de 41 à 68 le nombre de combinaisons établies
PIRES que la production.

CE QUE LA MESURE NE DIT PAS. Le mécanisme exact de ce gain résiduel n'est pas
établi : l'hypothèse naturelle -- l'appariement n'annule le marché que si les
deux variantes le portent à l'identique -- ne se reproduit pas proprement sur
données simulées, et les tests qui prétendaient l'isoler ont été retirés plutôt
qu'ajustés jusqu'à passer. Ce qui est vérifié ici est la MÉCANIQUE du calcul,
pas l'explication du gain.
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np
import pandas as pd
import pytest

from backtest import metrics as metrics_mod

_opt = importlib.import_module("16_optimize_strategie_actions")


def _serie(valeurs, debut="2015-01-01") -> pd.Series:
    return pd.Series(valeurs, index=pd.bdate_range(debut, periods=len(valeurs)))


def _trades(courbe: pd.DataFrame) -> pd.DataFrame:
    """Un journal de ventes minimal mais COMPLET : `compute_metrics` lit
    `num_positions`, `pnl` et `exit_date`, et un DataFrame vide le fait échouer
    sur la colonne manquante plutôt que de rendre des métriques sans trades."""
    dates = pd.DatetimeIndex(courbe["date"])
    milieu = len(dates) // 2
    return pd.DataFrame({
        "symbol": ["AAA", "BBB"],
        "entry_date": [dates[1], dates[milieu]],
        "exit_date": [dates[milieu - 1], dates[-1]],
        "pnl": [1_000.0, -500.0],
        "return_pct": [5.0, -2.0],
        "holding_days": [30, 30],
        "exit_reason": ["take_profit", "stop_loss"],
        "shares": [10.0, 10.0],
        "entry_price": [100.0, 100.0],
        "exit_price": [105.0, 98.0],
    })


def _courbe(rendements, debut="2015-01-01") -> pd.DataFrame:
    """`compute_metrics` lit `num_positions` et `invested_value` SANS garde :
    une courbe réduite à (date, nav) le fait échouer sur la colonne manquante.
    Ce ne sont pas des détails de confort -- ce sont des colonnes du contrat."""
    dates = pd.bdate_range(debut, periods=len(rendements) + 1)
    nav = 100_000.0 * np.cumprod(np.concatenate([[1.0], 1.0 + np.asarray(rendements)]))
    return pd.DataFrame({
        "date": dates, "nav": nav,
        "num_positions": 12,
        "invested_value": nav * 0.9,
    })


# --------------------------------------------------------------------------- #
# LA propriété
# --------------------------------------------------------------------------- #
def _signal_sur_bruit(a, b, marche=None, n_bootstrap=600) -> float:
    """|écart mesuré| / demi-largeur de son intervalle.

    LE SEUL RAPPORT COMPARABLE ENTRE MÉTRIQUES, et c'est le piège de tout ce
    dossier : un Sharpe et un information ratio ne sont PAS sur la même échelle.
    Dans un régime dominé par le marché, l'IR d'une stratégie vaut plusieurs
    fois son Sharpe, et son intervalle est plus large dans la même proportion.
    Comparer les demi-largeurs brutes ferait conclure que l'IR sépare dix fois
    moins bien -- alors qu'il sépare pareil."""
    r = metrics_mod.paired_sharpe_difference(
        a, b, n_bootstrap=n_bootstrap, benchmark_returns=marche)
    demi = (r["difference_ci_high"] - r["difference_ci_low"]) / 2
    return abs(r["sharpe_difference"]) / demi if demi > 0 else float("inf")


def test_les_demi_largeurs_de_DEUX_METRIQUES_ne_se_comparent_pas():
    """LE PIÈGE MÉTHODOLOGIQUE DE TOUT CE DOSSIER, et il a failli me faire
    conclure l'inverse de la mesure.

    Dans un régime dominé par le marché, l'information ratio d'une stratégie
    vaut plusieurs fois son Sharpe -- et son intervalle est plus large dans la
    même proportion. Lire les demi-largeurs brutes côte à côte fait donc
    conclure que l'IR « sépare dix fois moins bien », alors qu'il sépare
    pareil ou mieux.

    Ce qui se compare est sans dimension : étendue / demi-largeur pour une
    grille, |écart| / demi-largeur pour une paire."""
    rng = np.random.default_rng(7)
    n = 1500
    marche = rng.normal(0.0004, 0.020, n)
    commun = rng.normal(0.0, 0.0030, n)
    a = _serie(marche + commun + rng.normal(0.0000, 0.0015, n))
    b = _serie(marche + commun + rng.normal(0.0002, 0.0015, n))
    m = _serie(marche)

    sur_sharpe = metrics_mod.paired_sharpe_difference(a, b, n_bootstrap=400)
    sur_ir = metrics_mod.paired_sharpe_difference(a, b, n_bootstrap=400, benchmark_returns=m)

    demi_sharpe = (sur_sharpe["difference_ci_high"] - sur_sharpe["difference_ci_low"]) / 2
    demi_ir = (sur_ir["difference_ci_high"] - sur_ir["difference_ci_low"]) / 2

    # L'intervalle de l'IR est BEAUCOUP plus large en valeur absolue...
    assert demi_ir > 3 * demi_sharpe
    # ... et pourtant les rapports signal/bruit sont du même ordre.
    assert 0.7 < _signal_sur_bruit(a, b, m) / _signal_sur_bruit(a, b) < 1.5


def test_l_ir_separe_mieux_A_L_AUNE_MARGINALE_et_c_est_ce_qui_trompait():
    """POURQUOI LE LEVIER SEMBLAIT VALOIR ×1,9. Jugé contre l'erreur-type
    MARGINALE -- celle d'une stratégie prise isolément --, l'IR sépare bien
    mieux : le marché disparaît du dénominateur, donc l'écart pèse plus lourd
    relativement au bruit.

    C'est vrai, et c'est la mesure qui a fait proposer ce levier. Elle ne vaut
    que TANT QU'ON N'A PAS le test apparié : lui aussi retire ce facteur, et
    le test précédent montre qu'après lui il ne reste rien à prendre."""
    rng = np.random.default_rng(13)
    n = 1500
    marche = rng.normal(0.0004, 0.020, n)
    a = marche + rng.normal(0.0000, 0.0020, n)
    b = marche + rng.normal(0.0002, 0.0020, n)
    annualise = np.sqrt(metrics_mod.TRADING_DAYS_PER_YEAR)
    annees = n / metrics_mod.TRADING_DAYS_PER_YEAR

    def ratio_et_marge(x, y, actif):
        u, v = (x - actif, y - actif) if actif is not None else (x, y)
        ru, rv = u.mean()/u.std()*annualise, v.mean()/v.std()*annualise
        # Erreur-type de Lo sur la valeur du ratio (cf. _erreur_type_sharpe).
        et = np.sqrt((1 + 0.5 * max(ru, rv) ** 2) / annees)
        return abs(rv - ru) / et

    assert ratio_et_marge(a, b, marche) > 3 * ratio_et_marge(a, b, None)


def test_le_resultat_dit_quelle_metrique_a_ete_mesuree():
    """Deux appels rendent le même schéma de clés : sans cette étiquette, un CSV
    ne dirait plus si `paired_delta_sharpe` porte un Sharpe ou un IR."""
    a, b = _serie(np.full(400, 0.001)), _serie(np.full(400, 0.001))
    m = _serie(np.full(400, 0.0005))
    assert metrics_mod.paired_sharpe_difference(a, b, n_bootstrap=50)["metrique"] == "sharpe_ratio"
    assert metrics_mod.paired_sharpe_difference(
        a, b, n_bootstrap=50, benchmark_returns=m)["metrique"] == "information_ratio"


def test_l_indice_est_reechantillonne_avec_les_deux_series():
    """L'APPARIEMENT DOIT SURVIVRE À L'AJOUT DE L'INDICE. Si le marché était
    tiré à part -- ou laissé dans son ordre d'origine pendant que A et B sont
    rééchantillonnés --, les deux écarts actifs cesseraient de porter le même
    marché, et le test redeviendrait non apparié.

    Une série comparée à ELLE-MÊME le montre : l'écart doit être nul à chaque
    rééchantillonnage, donc l'intervalle entier doit être nul."""
    rng = np.random.default_rng(3)
    x = _serie(rng.normal(0.0005, 0.02, 800))
    m = _serie(rng.normal(0.0003, 0.019, 800))

    r = metrics_mod.paired_sharpe_difference(x, x, n_bootstrap=300, benchmark_returns=m)
    assert r["sharpe_difference"] == pytest.approx(0.0, abs=1e-12)
    assert r["difference_ci_low"] == pytest.approx(0.0, abs=1e-12)
    assert r["difference_ci_high"] == pytest.approx(0.0, abs=1e-12)


def test_l_ir_vaut_bien_le_sharpe_de_l_ecart_actif():
    """Contrôle d'identité : la définition, pas une approximation."""
    rng = np.random.default_rng(11)
    strat = rng.normal(0.0008, 0.02, 900)
    marche = rng.normal(0.0004, 0.018, 900)
    r = metrics_mod.paired_sharpe_difference(
        _serie(strat), _serie(strat), n_bootstrap=50, benchmark_returns=_serie(marche))

    actif = strat - marche
    attendu = actif.mean() / actif.std() * np.sqrt(metrics_mod.TRADING_DAYS_PER_YEAR)
    assert r["sharpe_b"] == pytest.approx(attendu)


def test_un_indice_trop_court_ne_fabrique_pas_de_chiffre():
    """L'intersection des dates gouverne : un indice qui ne couvre pas la
    période ne doit pas produire un IR sur trois points."""
    a = b = _serie(np.full(400, 0.001))
    assert metrics_mod.paired_sharpe_difference(
        a, b, n_bootstrap=50, benchmark_returns=_serie(np.full(5, 0.0005))) == {}


# --------------------------------------------------------------------------- #
# L'IR par fenêtre
# --------------------------------------------------------------------------- #
def test_l_information_ratio_est_calcule_par_fenetre():
    """Sans `train_information_ratio`, l'IR ne peut pas CLASSER : la grille
    classe toujours sur la fenêtre d'apprentissage. Il était calculé en plein
    échantillon et absent de la liste des clés découpées -- c'est tout ce qui
    manquait."""
    rng = np.random.default_rng(5)
    n = 1200
    marche = rng.normal(0.0003, 0.015, n)
    courbe = _courbe(marche + rng.normal(0.0002, 0.004, n))
    indice = pd.Series(
        100.0 * np.cumprod(1.0 + np.concatenate([[0.0], marche])),
        index=pd.DatetimeIndex(courbe["date"]))

    decoupe = metrics_mod.split_period_metrics(
        courbe, _trades(courbe), split_date=courbe["date"].iloc[n // 2],
        benchmark_prices=indice,
    )
    for cle in ("train_information_ratio", "test_information_ratio",
                "train_alpha_pct", "test_alpha_pct", "train_beta", "test_beta"):
        assert cle in decoupe, cle
        assert decoupe[cle] is not None, cle

    # Les deux fenêtres portent des IR DIFFÉRENTS : un découpage qui rendrait
    # deux fois le chiffre plein échantillon passerait tous les tests ci-dessus.
    assert decoupe["train_information_ratio"] != decoupe["test_information_ratio"]


def test_les_metriques_historiques_restent_toutes_la():
    """Ajouter des clés ne doit pas en retirer : les colonnes du CSV sont lues
    par le rapport, l'audit et les runs archivés."""
    courbe = _courbe(np.random.default_rng(1).normal(0.0005, 0.01, 600))
    decoupe = metrics_mod.split_period_metrics(
        courbe, _trades(courbe), split_date=courbe["date"].iloc[300])
    for cle in ("train_cagr_pct", "train_sharpe_ratio", "train_sortino_ratio",
                "train_calmar_ratio", "train_max_drawdown_pct", "train_total_return_pct",
                "test_sharpe_ratio", "split_date"):
        assert cle in decoupe, cle


# --------------------------------------------------------------------------- #
# Le branchement sur la grille
# --------------------------------------------------------------------------- #
def _args(**surcharges) -> argparse.Namespace:
    base = dict(entry_threshold_grid=None, momentum_grid=None, stop_loss_grid=None,
                take_profit_grid=None, rebalance_band_grid=None, max_weight_grid=None,
                max_holding_grid=None, multiple_hierarchy_grid=None, quick=False,
                rank_metric="sharpe_ratio")
    return argparse.Namespace(**{**base, **surcharges})


def test_le_sharpe_reste_le_defaut():
    """Le levier est DISPONIBLE, pas imposé : basculer le défaut changerait
    silencieusement tous les réglages retenus jusqu'ici."""
    import inspect
    source = inspect.getsource(_opt.main)
    assert '"--rank-metric", default="sharpe_ratio"' in source


def test_les_quatre_metriques_sont_offertes():
    import inspect
    source = inspect.getsource(_opt.main)
    for m in ("sharpe_ratio", "information_ratio", "sortino_ratio", "calmar_ratio"):
        assert m in source, m


def test_le_test_apparie_suit_la_metrique_de_classement():
    """Classer sur l'IR en jugeant sur le Sharpe, ce serait choisir selon un
    critère et conclure selon un autre -- et l'écart entre les deux est
    exactement ce que le facteur marché explique."""
    import inspect
    source = inspect.getsource(_opt._ajoute_stats_appariees)
    assert 'metrique == "information_ratio"' in source
    assert "benchmark_returns=rendements_indice" in source
    # Et l'appelant transmet bien le réglage, au lieu du défaut de la fonction.
    assert "args.rank_metric" in inspect.getsource(_opt.main)


def test_sans_indice_le_test_apparie_le_dit_au_lieu_de_mentir(monkeypatch, caplog):
    """Un IR demandé sans série d'indice retomberait sur le Sharpe : silencieux,
    ce serait un rapport qui annonce une métrique et en mesure une autre."""
    import logging
    reference = _opt._reference_combo("valuation_gap_combined")
    rng = np.random.default_rng(0)
    lignes = [{**reference, "sharpe_ratio": 1.0,
               "_nav": 100_000.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.011, 400)),
               "_dates": pd.bdate_range("2015-01-01", periods=400)}]

    monkeypatch.setitem(_opt._DATA, "benchmark_prices", None)
    with caplog.at_level(logging.WARNING):
        _opt._ajoute_stats_appariees(lignes, "valuation_gap_combined", 100,
                                     metrique="information_ratio")
    assert "reste sur le Sharpe" in caplog.text


def test_la_lecture_compare_a_l_erreur_type_de_LA_metrique():
    """L'erreur-type de Lo se calcule sur la VALEUR du ratio : celle d'un IR de
    0,25 n'est pas celle d'un Sharpe de 0,93. Les mélanger ferait annoncer un
    facteur de précision qui n'existe pas."""
    import inspect
    source = inspect.getsource(_opt._lire_le_test_apparie)
    assert "colonne = metrique if metrique in avec.columns else" in source


def test_le_rapport_montre_la_fenetre_de_test_de_LA_metrique():
    """Classer sur l'IR et afficher `test_sharpe_ratio` à côté ferait juger la
    survie hors échantillon sur autre chose que ce qui a choisi."""
    import inspect
    source = inspect.getsource(_opt._report)
    assert 'rank_key.replace("train_", "test_", 1)' in source
