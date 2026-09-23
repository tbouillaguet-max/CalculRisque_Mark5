"""La grille classe-t-elle CONTRE ce qui tourne, et sait-elle le dire ?

CE QUE CES TESTS DÉFENDENT. La grille produisait un classement et rien de
plus : 864 combinaisons dont l'étendue entière (0,133 de Sharpe) tenait dans
le tiers d'une erreur-type marginale (0,35). Autrement dit, elle désignait un
gagnant sans jamais pouvoir dire s'il valait mieux que la configuration en
production. Le test apparié répond à cette question-là -- les courbes sont
corrélées à 0,990, leur écart se mesure quatre fois plus finement (0,079 contre
0,353), et 26 combinaisons sur 108 s'en trouvent établies différentes de ce qui
tourne. Mais il ne sert à rien si le câblage se trompe de ligne de référence,
et rien ne le signalerait : on lirait des intervalles parfaitement calculés
autour de la mauvaise combinaison.

LE TEST QUI COMPTE est `test_la_ligne_de_reference_se_retrouve_malgre_le_none`.
`momentum_min_pct = None` est la valeur EN PRODUCTION, et c'est exactement la
valeur qu'une comparaison par égalité ne retrouve pas (None passé par un
DataFrame revient en NaN, et NaN != NaN). Sans ce traitement explicite, la
référence serait introuvable sur la configuration réelle, et seulement sur
elle.
"""

from __future__ import annotations

import importlib
import logging

import numpy as np
import pandas as pd
import pytest

_opt = importlib.import_module("16_optimize_strategie_actions")


def _reference() -> dict:
    return _opt._reference_combo("valuation_gap_combined")


def _ligne(reference: dict, n: int = 400, derive: float = 0.0, graine: int = 0, **surcharges) -> dict:
    """Une ligne de grille telle que `_run_one` la rend : les axes, plus la
    courbe de NAV sous les clés préfixées d'un underscore."""
    rng = np.random.default_rng(graine)
    dates = pd.bdate_range("2015-01-01", periods=n)
    rendements = rng.normal(0.0004 + derive, 0.011, n)
    return {
        **reference, **surcharges,
        "sharpe_ratio": 1.0,
        "_nav": 100_000.0 * np.cumprod(1.0 + rendements),
        "_dates": pd.DatetimeIndex(dates),
    }


# --------------------------------------------------------------------------- #
# Retrouver la configuration en production dans la grille
# --------------------------------------------------------------------------- #
def test_la_ligne_de_reference_se_retrouve_malgre_le_none():
    """LE point : `momentum_min_pct = None` est la valeur en production, et
    c'est la seule que l'égalité ne retrouve pas."""
    reference = _reference()
    assert reference["momentum_min_pct"] is None, (
        "Ce test perd son objet si la production fixe un momentum : le remettre "
        "à None ici masquerait le vrai réglage.")

    lignes = [
        _ligne(reference, momentum_min_pct=0.0),
        _ligne(reference, momentum_min_pct=-10.0),
        _ligne(reference),
    ]
    assert _opt._index_de_reference(lignes, reference) == 2


def test_un_nan_vaut_le_none_attendu():
    """La même ligne relue depuis un CSV porte NaN, pas None. Les deux doivent
    désigner la même configuration."""
    reference = _reference()
    ligne = _ligne(reference)
    ligne["momentum_min_pct"] = float("nan")
    assert _opt._index_de_reference([ligne], reference) == 0


def test_une_grille_sans_la_configuration_actuelle_rend_none():
    reference = _reference()
    lignes = [_ligne(reference, entry_threshold_pct=reference["entry_threshold_pct"] + 5.0)]
    assert _opt._index_de_reference(lignes, reference) is None


def test_une_ligne_en_erreur_n_est_jamais_prise_pour_la_reference():
    """Une combinaison qui a planté porte les bons axes mais aucune courbe :
    la retenir donnerait un test apparié contre rien."""
    reference = _reference()
    plantee = {**reference, "error": "boom"}
    assert _opt._index_de_reference([plantee], reference) is None
    assert _opt._index_de_reference([plantee, _ligne(reference)], reference) == 1


def test_un_ecart_infime_sur_un_axe_reste_la_meme_configuration():
    """Les seuils transitent par un CSV : 15.0 peut revenir 14.999999999. La
    comparaison est une tolérance, pas une égalité."""
    reference = _reference()
    ligne = _ligne(reference)
    ligne["entry_threshold_pct"] = float(reference["entry_threshold_pct"]) + 1e-12
    assert _opt._index_de_reference([ligne], reference) == 0


def test_un_vrai_ecart_sur_un_axe_n_est_pas_la_reference():
    """Contrôle de la tolérance : sans lui, le test précédent passerait aussi
    avec une comparaison qui accepte tout."""
    reference = _reference()
    ligne = _ligne(reference)
    ligne["entry_threshold_pct"] = float(reference["entry_threshold_pct"]) + 0.01
    assert _opt._index_de_reference([ligne], reference) is None


# --------------------------------------------------------------------------- #
# Le test apparié, branché sur la grille
# --------------------------------------------------------------------------- #
def test_chaque_combinaison_recoit_son_ecart_et_son_intervalle():
    reference = _reference()
    lignes = [
        _ligne(reference),
        _ligne(reference, derive=0.0006, graine=1, take_profit_pct=80.0),
    ]
    assert _opt._ajoute_stats_appariees(lignes, "valuation_gap_combined", 200) == 0

    for ligne in lignes:
        for cle in ("paired_delta_sharpe", "paired_ci_low", "paired_ci_high",
                    "paired_p_value", "paired_correlation", "paired_n_observations"):
            assert cle in ligne, cle
        assert ligne["paired_ci_low"] <= ligne["paired_delta_sharpe"] <= ligne["paired_ci_high"]


def test_la_reference_est_comparee_a_elle_meme_et_donne_zero():
    """Garde-fou d'appariement : si les deux séries n'étaient pas rééchantillonnées
    sur les MÊMES dates, la ligne de référence s'écarterait d'elle-même."""
    reference = _reference()
    lignes = [_ligne(reference)]
    _opt._ajoute_stats_appariees(lignes, "valuation_gap_combined", 200)

    assert lignes[0]["paired_delta_sharpe"] == pytest.approx(0.0, abs=1e-12)
    assert lignes[0]["paired_ci_low"] == pytest.approx(0.0, abs=1e-12)
    assert lignes[0]["paired_ci_high"] == pytest.approx(0.0, abs=1e-12)
    assert lignes[0]["paired_correlation"] == pytest.approx(1.0)


def test_une_combinaison_franchement_meilleure_a_un_intervalle_a_droite_de_zero():
    """Sans ça, le test apparié serait un calcul qui ne conclut jamais."""
    reference = _reference()
    lignes = [_ligne(reference), _ligne(reference, derive=0.002, take_profit_pct=80.0)]
    _opt._ajoute_stats_appariees(lignes, "valuation_gap_combined", 400)

    assert lignes[1]["paired_delta_sharpe"] > 0
    assert lignes[1]["paired_ci_low"] > 0
    assert lignes[1]["paired_p_value"] < 0.05


def test_sans_la_configuration_actuelle_rien_n_est_calcule_mais_c_est_dit(caplog):
    """Le silence serait le pire des cas : une grille classée comme avant,
    sans qu'on sache que la comparaison qui décide n'a pas eu lieu."""
    reference = _reference()
    lignes = [_ligne(reference, entry_threshold_pct=reference["entry_threshold_pct"] + 5.0)]

    with caplog.at_level(logging.WARNING):
        assert _opt._ajoute_stats_appariees(lignes, "valuation_gap_combined", 200) is None

    assert "absente de la grille" in caplog.text
    assert "paired_delta_sharpe" not in lignes[0]


def test_zero_reechantillonnage_desactive_le_test_sans_rien_casser():
    reference = _reference()
    lignes = [_ligne(reference)]
    assert _opt._ajoute_stats_appariees(lignes, "valuation_gap_combined", 0) is None
    assert "paired_delta_sharpe" not in lignes[0]


def test_une_ligne_en_erreur_est_sautee_sans_faire_planter_la_grille():
    reference = _reference()
    lignes = [_ligne(reference), {**reference, "error": "boom", "take_profit_pct": 80.0}]
    _opt._ajoute_stats_appariees(lignes, "valuation_gap_combined", 200)

    assert "paired_delta_sharpe" in lignes[0]
    assert "paired_delta_sharpe" not in lignes[1]


def test_les_courbes_de_nav_ne_partent_pas_dans_le_csv():
    """Une colonne de 3000 nombres par ligne rendrait le CSV illisible : les
    clés soulignées restent en mémoire. C'est le filtre qu'applique `main`."""
    reference = _reference()
    lignes = [_ligne(reference)]
    _opt._ajoute_stats_appariees(lignes, "valuation_gap_combined", 200)

    colonnes = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in lignes]).columns
    assert "_nav" not in colonnes and "_dates" not in colonnes
    assert "paired_ci_low" in colonnes


# --------------------------------------------------------------------------- #
# La lecture
# --------------------------------------------------------------------------- #
def _tableau(lignes: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(lignes)


def test_la_lecture_compte_les_meilleures_les_pires_et_les_indistinctes(caplog):
    ok = _tableau([
        {"sharpe_ratio": 1.0, "paired_delta_sharpe": 0.20, "paired_ci_low": 0.05,
         "paired_ci_high": 0.35, "paired_p_value": 0.01, "paired_correlation": 0.99},
        {"sharpe_ratio": 1.0, "paired_delta_sharpe": -0.20, "paired_ci_low": -0.35,
         "paired_ci_high": -0.05, "paired_p_value": 0.99, "paired_correlation": 0.99},
        {"sharpe_ratio": 1.0, "paired_delta_sharpe": 0.01, "paired_ci_low": -0.10,
         "paired_ci_high": 0.12, "paired_p_value": 0.45, "paired_correlation": 0.99},
    ])
    with caplog.at_level(logging.INFO):
        _opt._lire_le_test_apparie(ok, tolerance=0.02)

    assert "1 MEILLEURES" in caplog.text
    assert "1 PIRES" in caplog.text
    assert "1 indistinguables" in caplog.text


def test_la_lecture_dit_quand_aucune_combinaison_ne_bat_la_production(caplog):
    """« Aucune » est un résultat, et il doit s'afficher comme tel -- pas comme
    un tableau vide que personne ne remarque."""
    ok = _tableau([
        {"sharpe_ratio": 1.0, "paired_delta_sharpe": 0.01, "paired_ci_low": -0.10,
         "paired_ci_high": 0.12, "paired_p_value": 0.45, "paired_correlation": 0.99},
    ])
    with caplog.at_level(logging.INFO):
        _opt._lire_le_test_apparie(ok, tolerance=0.02)

    assert "AUCUNE combinaison" in caplog.text


def test_la_lecture_met_les_deux_precisions_cote_a_cote(caplog):
    """Tout l'intérêt du test apparié tient dans ce rapprochement : c'est ce qui
    fait passer la grille de « rien n'est distinguable » à une décision."""
    ok = _tableau([
        {"sharpe_ratio": 1.0, "paired_delta_sharpe": 0.0, "paired_ci_low": -0.04,
         "paired_ci_high": 0.04, "paired_p_value": 0.5, "paired_correlation": 0.997,
         "paired_n_observations": 7 * 252},
    ])
    with caplog.at_level(logging.INFO):
        _opt._lire_le_test_apparie(ok, tolerance=0.02)

    # Demi-largeur 0,040 contre une erreur-type marginale de 0,463 sur sept ans
    # à Sharpe 1 : le rapport annoncé doit être celui-là, et pas un autre.
    assert "0.040" in caplog.text
    assert "0.463" in caplog.text
    assert "12x plus précis" in caplog.text


def test_les_deux_precisions_portent_sur_la_MEME_fenetre(caplog):
    """LE piège de ce rapprochement. L'intervalle apparié est mesuré sur la
    courbe ENTIÈRE (onze ans) ; le comparer à l'erreur-type de la seule fenêtre
    d'apprentissage (sept ans) gonflerait le rapport de sqrt(11/7), soit 25 %
    de précision annoncée qui n'existe pas. Rien ne le signalerait : les deux
    nombres sont justes, c'est leur mise côte à côte qui serait fausse."""
    ligne = {"sharpe_ratio": 1.0, "paired_delta_sharpe": 0.0, "paired_ci_low": -0.04,
             "paired_ci_high": 0.04, "paired_p_value": 0.5, "paired_correlation": 0.997}

    with caplog.at_level(logging.INFO):
        _opt._lire_le_test_apparie(_tableau([{**ligne, "paired_n_observations": 11 * 252}]),
                                   tolerance=0.02)
    assert "sur les mêmes 11.0 ans" in caplog.text
    assert "0.369" in caplog.text          # et non 0,463, celle de sept ans

    caplog.clear()
    with caplog.at_level(logging.INFO):
        _opt._lire_le_test_apparie(_tableau([{**ligne, "paired_n_observations": 7 * 252}]),
                                   tolerance=0.02)
    assert "sur les mêmes 7.0 ans" in caplog.text
    assert "0.463" in caplog.text


def test_la_precision_marginale_suit_le_sharpe_plein_echantillon(caplog):
    """`train_sharpe_ratio` porte sur la fenêtre d'apprentissage, donc pas sur
    celle du test apparié : c'est le Sharpe plein échantillon qui va avec."""
    ok = _tableau([
        {"sharpe_ratio": 1.0, "train_sharpe_ratio": 3.0, "paired_delta_sharpe": 0.0,
         "paired_ci_low": -0.04, "paired_ci_high": 0.04, "paired_p_value": 0.5,
         "paired_correlation": 0.997, "paired_n_observations": 7 * 252},
    ])
    with caplog.at_level(logging.INFO):
        _opt._lire_le_test_apparie(ok, tolerance=0.02)

    assert "0.463" in caplog.text          # Sharpe 1 ; à Sharpe 3 ce serait 0,886


def test_la_lecture_ne_dit_rien_quand_le_test_n_a_pas_tourne(caplog):
    with caplog.at_level(logging.INFO):
        _opt._lire_le_test_apparie(_tableau([{"sharpe_ratio": 1.0}]), tolerance=0.02)
        _opt._lire_le_test_apparie(
            _tableau([{"sharpe_ratio": 1.0, "paired_ci_low": float("nan")}]), tolerance=0.02)

    assert "test apparié départage" not in caplog.text


# --------------------------------------------------------------------------- #
# Les deux axes retirés de la grille
# --------------------------------------------------------------------------- #
def test_la_grille_par_defaut_garde_la_configuration_en_production():
    """Retirer un axe ne doit jamais retirer le point de comparaison : une
    grille qui ne contient plus la production ne peut plus rien départager."""
    reference = _reference()
    assert _opt.DEFAULT_STOP_LOSS_GRID == [reference["stop_loss_pct"]]
    assert _opt.DEFAULT_MAX_WEIGHT_GRID == [reference["max_weight_pct"]]


def test_les_deux_axes_retires_sont_bien_reduits_a_un_point():
    """Mesurés sans effet (η² de 0,6 % et 0,1 %), ils multipliaient le calcul
    par huit. Le réglage reste paramétrable en ligne de commande."""
    assert len(_opt.DEFAULT_STOP_LOSS_GRID) == 1
    assert len(_opt.DEFAULT_MAX_WEIGHT_GRID) == 1


def test_un_axe_reduit_a_un_point_n_est_pas_annonce_UNANIME(caplog):
    """LE piège de la réduction d'axes. Un axe à une seule valeur est unanime
    sur le plateau par construction, et l'afficher comme tel présenterait
    l'ABSENCE DE BALAYAGE comme un résultat -- le genre de non-résultat qu'on
    relit six mois plus tard comme une conclusion."""
    grille = pd.DataFrame([
        {"train_sharpe_ratio": s, "stop_loss_pct": -15.0, "take_profit_pct": tp,
         "annualized_turnover_pct": 700.0}
        for s, tp in ((1.00, 30.0), (0.99, 30.0), (0.98, 30.0), (0.97, 60.0))
    ])
    with caplog.at_level(logging.INFO):
        _opt._lire_le_plateau(grille, grille.head(3), "train_sharpe_ratio", tolerance=0.05)

    assert "stop_loss_pct        non balayé" in caplog.text
    # take_profit_pct, lui, est balayé ET unanime sur le plateau : c'en est un.
    assert "take_profit_pct      UNANIME" in caplog.text


def test_un_axe_reellement_unanime_reste_annonce_comme_tel(caplog):
    """Contrôle du garde-fou précédent : sans lui, on pourrait le satisfaire en
    ne disant plus jamais « unanime », et perdre le seul vrai résultat que le
    plateau produise."""
    grille = pd.DataFrame([
        {"train_sharpe_ratio": s, "take_profit_pct": tp, "annualized_turnover_pct": 700.0}
        for s, tp in ((1.00, 30.0), (0.99, 30.0), (0.80, 60.0))
    ])
    with caplog.at_level(logging.INFO):
        _opt._lire_le_plateau(grille, grille.head(2), "train_sharpe_ratio", tolerance=0.05)

    assert "take_profit_pct      UNANIME" in caplog.text
    assert "non balayé" not in caplog.text
