"""Agrégation des multiples sectoriels, et combinaison des valeurs implicites.

Deux choix qui étaient implicites et le sont devenus explicites :

    B3a  COMMENT résumer les multiples d'un groupe de pairs. La médiane est un
         choix robuste raisonnable ; la moyenne HARMONIQUE est l'estimateur de
         variance minimale d'un ratio sous erreur multiplicative (Baker &
         Ruback, 1999), parce qu'elle revient à moyenner des rendements.
    B3b  COMMENT combiner les valeurs implicites des trois multiples. La
         médiane à trois voix les traite comme également informatifs ; ils ne
         le sont pas (Liu, Nissim & Thomas, 2002 : les multiples de chiffre
         d'affaires sont les moins précis, de loin).

Les deux restent commutables par configuration, pour rejouer un run ancien et
pour rendre l'A/B possible sans toucher au code.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

import config

m06b = importlib.import_module("06b_calcul_valorisation_combinee")


# --------------------------------------------------------------------------- #
# B3a -- moyenne harmonique
# --------------------------------------------------------------------------- #

def test_la_moyenne_harmonique_est_l_inverse_du_rendement_moyen():
    """C'est sa définition économique, et toute sa justification : moyenner des
    P/E n'a pas de sens, moyenner des rendements bénéficiaires en a un."""
    multiples = pd.Series([10.0, 20.0, 40.0])
    rendement_moyen = (1 / multiples).mean()

    assert m06b.aggregate_multiple(multiples, "harmonic") == pytest.approx(1 / rendement_moyen)
    # 3 / (1/10 + 1/20 + 1/40) = 3 / 0.175 = 17.14...
    assert m06b.aggregate_multiple(multiples, "harmonic") == pytest.approx(17.142857, rel=1e-5)


def test_la_moyenne_harmonique_reste_sous_la_moyenne_arithmetique():
    """Propriété structurelle -- harmonique <= géométrique <= arithmétique --
    et c'est exactement le biais à la hausse que Baker & Ruback reprochent à la
    moyenne arithmétique : sur une distribution étalée à droite, comme l'est
    toujours celle des multiples, l'écart n'a rien d'anecdotique.

    AUCUN ORDRE GÉNÉRAL FACE À LA MÉDIANE, en revanche : il dépend de la forme
    de la distribution, et l'affirmer serait faux. Sur l'échantillon ci-dessous,
    la moyenne harmonique passe AU-DESSUS de la médiane, que le paquet de
    valeurs basses tire vers le bas."""
    multiples = pd.Series([8.0, 10.0, 12.0, 15.0, 45.0])
    harmonique = m06b.aggregate_multiple(multiples, "harmonic")

    assert harmonique < multiples.mean()                    # toujours vrai
    assert harmonique == pytest.approx(12.587412, rel=1e-5)
    assert harmonique > multiples.median()                  # propre à cet échantillon

    # L'ordre s'inverse dès que ce sont les multiples BAS qui sont isolés : la
    # moyenne harmonique leur donne un poids que la médiane leur refuse.
    # Vérifié dans les deux sens, précisément parce qu'aucun des deux n'est la
    # règle.
    autre_forme = pd.Series([6.0, 19.0, 21.0, 23.0, 26.0])
    assert m06b.aggregate_multiple(autre_forme, "harmonic") < autre_forme.median()
    assert m06b.aggregate_multiple(autre_forme, "harmonic") < autre_forme.mean()


def test_le_mode_median_reproduit_le_comportement_historique():
    multiples = pd.Series([8.0, 10.0, 12.0, 15.0, 45.0])
    assert m06b.aggregate_multiple(multiples, "median") == pytest.approx(12.0)


def test_un_agregateur_inconnu_est_refuse():
    with pytest.raises(ValueError):
        m06b.aggregate_multiple(pd.Series([10.0]), "moyenne")


def test_un_groupe_vide_ne_rend_rien():
    assert m06b.aggregate_multiple(pd.Series([], dtype=float)) is None


def test_le_plancher_de_plausibilite_protege_la_moyenne_harmonique():
    """LA contrepartie du passage à l'harmonique. Un multiple minuscule devient
    un rendement gigantesque et tire toute l'agrégation -- là où la médiane
    l'ignorait. C'est ce que les planchers de MULTIPLE_PLAUSIBLE_RANGE
    écartent, et la raison pour laquelle ils ne sont plus à zéro."""
    sains = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
    pollues = pd.Series([0.01, 10.0, 12.0, 14.0, 16.0, 18.0])

    # Sans filtrage, une seule valeur aberrante écrase l'agrégation...
    assert m06b.aggregate_multiple(pollues, "harmonic") < 0.07
    # ...alors qu'elle laisse la médiane presque intacte : c'est bien la
    # moyenne harmonique qui a besoin du plancher, pas la médiane.
    assert m06b.aggregate_multiple(pollues, "median") == pytest.approx(13.0)

    # _clean_multiple l'écarte, et l'agrégation retrouve sa valeur.
    nettoyes = m06b._clean_multiple(pollues, "EV/EBITDA")
    assert m06b.aggregate_multiple(nettoyes, "harmonic") == pytest.approx(
        m06b.aggregate_multiple(sains, "harmonic"))


def test_les_planchers_ecartent_les_erreurs_pas_les_valorisations_reelles():
    """Les bornes basses ne doivent retirer que ce qui est presque certainement
    une erreur d'extraction pour une société de l'indice."""
    for col, plausible, aberrant in (
        ("EV/EBITDA", 4.0, 0.2),
        ("P/E", 6.0, 0.4),
        # EV/Sales a une distribution réellement plus étalée : un distributeur
        # à 0,15 x le chiffre d'affaires est une valorisation légitime.
        ("EV/Sales", 0.15, 0.01),
    ):
        garde = m06b._clean_multiple(pd.Series([plausible, aberrant]), col)
        assert list(garde) == [plausible], col


# --------------------------------------------------------------------------- #
# B3b -- hiérarchie de fiabilité
# --------------------------------------------------------------------------- #

def _implicites(ev_ebitda=np.nan, ev_sales=np.nan, pe=np.nan) -> pd.DataFrame:
    return pd.DataFrame(
        {"EV/EBITDA": [ev_ebitda], "EV/Sales": [ev_sales], "P/E": [pe]})


def test_ev_sales_ne_tranche_plus_entre_les_deux_multiples_de_resultats():
    """LE défaut corrigé. Avec une médiane à trois voix, quand EV/EBITDA et P/E
    divergent, c'est EV/Sales -- le moins précis des trois -- qui décide."""
    implicites = _implicites(ev_ebitda=100.0, pe=140.0, ev_sales=101.0)

    # Ancien comportement : la médiane des trois, donc EV/Sales.
    assert m06b.combine_implied_prices(implicites, "flat")[0] == pytest.approx(101.0)
    # Nouveau : moyenne des deux multiples de résultats, EV/Sales ignoré.
    assert m06b.combine_implied_prices(implicites, "tiers")[0] == pytest.approx(120.0)


def test_un_seul_multiple_de_resultats_suffit_a_ecarter_ev_sales():
    implicites = _implicites(ev_ebitda=100.0, ev_sales=250.0)
    assert m06b.combine_implied_prices(implicites, "tiers")[0] == pytest.approx(100.0)


def test_ev_sales_sert_quand_rien_de_mieux_n_est_disponible():
    """Le repli n'est pas un cas résiduel : une entreprise en perte n'a pas de
    P/E exploitable et souvent pas d'EBITDA positif non plus -- c'est
    exactement là qu'un multiple de chiffre d'affaires est la seule
    valorisation possible."""
    implicites = _implicites(ev_sales=80.0)
    assert m06b.combine_implied_prices(implicites, "tiers")[0] == pytest.approx(80.0)


def test_aucun_multiple_exploitable_ne_rend_rien():
    """La ligne doit rester NaN pour que le repli DCF de 06b prenne le relais."""
    assert pd.isna(m06b.combine_implied_prices(_implicites(), "tiers")[0])


def test_la_combinaison_est_vectorisee_ligne_a_ligne():
    """Chaque ligne choisit son propre rang : une entreprise valorisable par
    ses résultats et une autre qui ne l'est pas coexistent dans le même appel."""
    implicites = pd.DataFrame({
        "EV/EBITDA": [100.0, np.nan, np.nan],
        "EV/Sales":  [250.0, 80.0, np.nan],
        "P/E":       [140.0, np.nan, np.nan],
    })
    resultat = m06b.combine_implied_prices(implicites, "tiers")
    assert resultat[0] == pytest.approx(120.0)   # rang 1, EV/Sales ignoré
    assert resultat[1] == pytest.approx(80.0)    # rang 2, seul disponible
    assert pd.isna(resultat[2])


def test_une_combinaison_inconnue_est_refusee():
    with pytest.raises(ValueError):
        m06b.combine_implied_prices(_implicites(pe=100.0), "ponderee")


def test_les_deux_reglages_sont_lus_dans_config_par_defaut(monkeypatch):
    """Un run se rejoue à l'identique en basculant la configuration, sans
    toucher au code -- même mécanique que OPTIONS_MULTIPLES_GAP_BASIS."""
    implicites = _implicites(ev_ebitda=100.0, pe=140.0, ev_sales=101.0)
    multiples = pd.Series([8.0, 10.0, 12.0, 15.0, 45.0])

    monkeypatch.setattr(config, "MULTIPLE_COMBINATION", "flat")
    monkeypatch.setattr(config, "SECTOR_MULTIPLE_AGGREGATOR", "median")
    assert m06b.combine_implied_prices(implicites)[0] == pytest.approx(101.0)
    assert m06b.aggregate_multiple(multiples) == pytest.approx(12.0)

    monkeypatch.setattr(config, "MULTIPLE_COMBINATION", "tiers")
    monkeypatch.setattr(config, "SECTOR_MULTIPLE_AGGREGATOR", "harmonic")
    assert m06b.combine_implied_prices(implicites)[0] == pytest.approx(120.0)
    assert m06b.aggregate_multiple(multiples) == pytest.approx(12.587412, rel=1e-5)
