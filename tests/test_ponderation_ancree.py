"""Pondération ANCRÉE : des cibles qui ne dépendent pas des autres candidates.

CE QU'ELLE ATTAQUE. `base.capped_weights` calcule
`poids = conviction / SOMME(convictions)`. Un seul dépôt SEC change donc le
dénominateur, et avec lui la cible de TOUTES les lignes -- c'est la cause
première des 93 % de ventes qui ne sont que du repesage. Ni la zone de
non-négociation ni la levée du coupe-circuit ne peuvent rien contre ça :
elles suppriment des ordres, elles ne réduisent pas l'AMPLITUDE de ce que
chaque dépôt déplace.

LE TEST QUI COMPTE est `test_une_candidate_neuve_ne_deplace_pas_les_autres` :
c'est la propriété entière de ce mécanisme, et tout le reste en découle.

MESURÉ, ELLE NE PAIE PAS -- voir le README. Le mécanisme reste disponible et
testé, désactivé par défaut, comme les autres réglages que ce dépôt a mesurés
puis écartés.
"""

from __future__ import annotations

import pandas as pd
import pytest

import config
from backtest.strategies import STRATEGY_REGISTRY
from backtest.strategies.base import capped_weights, construire_poids, poids_ancres
from backtest.strategies.valuation_gap_combined_ancre import ValuationGapCombinedAncreeStrategy


def _conviction(valeurs: dict) -> pd.Series:
    return pd.Series(valeurs, dtype=float)


# --------------------------------------------------------------------------- #
# LA propriété
# --------------------------------------------------------------------------- #
def test_une_candidate_neuve_ne_deplace_pas_les_autres():
    """LE point du mécanisme, et la seule chose qu'il promet."""
    avant = poids_ancres(_conviction({"AAA": 60.0, "BBB": 40.0}), ancre=1_000.0)
    apres = poids_ancres(_conviction({"AAA": 60.0, "BBB": 40.0, "CCC": 90.0}), ancre=1_000.0)

    assert apres["AAA"] == avant["AAA"]
    assert apres["BBB"] == avant["BBB"]
    assert apres["CCC"] == pytest.approx(0.09)


def test_la_ponderation_historique_deplace_bien_tout_le_monde():
    """Contrôle : c'est le comportement dont on cherche à sortir. Sans lui, le
    test précédent ne prouverait pas grand-chose.

    ASSEZ DE CANDIDATES POUR QUE LE PLAFOND SOIT ATTEIGNABLE. Avec deux lignes
    et un plafond de 20 %, `capped_weights` emprunte sa branche documentée --
    plafond hors d'atteinte, donc chaque ligne prend le plafond et le reste va
    en cash -- et les poids ne bougent plus quand une candidate arrive. On
    mesurerait le plafond, pas la renormalisation."""
    base = {f"S{i}": 10.0 * (i + 1) for i in range(10)}
    avant = capped_weights(_conviction(base), cap_pct=20.0)
    apres = capped_weights(_conviction({**base, "NEUVE": 90.0}), cap_pct=20.0)

    assert apres["S0"] != avant["S0"]
    assert apres["S9"] != avant["S9"]


# --------------------------------------------------------------------------- #
# Mécanique
# --------------------------------------------------------------------------- #
def test_le_poids_est_la_conviction_divisee_par_l_ancre():
    poids = poids_ancres(_conviction({"AAA": 50.0, "BBB": 200.0}), ancre=1_000.0, cap_pct=0)
    assert poids["AAA"] == pytest.approx(0.05)
    assert poids["BBB"] == pytest.approx(0.20)


def test_le_plafond_par_ligne_s_applique():
    poids = poids_ancres(_conviction({"AAA": 5_000.0}), ancre=1_000.0, cap_pct=20.0)
    assert poids["AAA"] == pytest.approx(0.20)


def test_la_somme_peut_rester_sous_un_et_le_solde_va_en_cash():
    """Pas de renormalisation : forcer la somme à 1 rétablirait exactement le
    couplage qu'on vient de supprimer."""
    poids = poids_ancres(_conviction({"AAA": 10.0, "BBB": 20.0}), ancre=1_000.0)
    assert poids.sum() == pytest.approx(0.03)


def test_une_conviction_negative_ne_donne_pas_un_poids_negatif():
    """Long-only : une conviction négative ne peut pas devenir une vente à
    découvert par accident arithmétique."""
    poids = poids_ancres(_conviction({"AAA": -50.0, "BBB": 50.0}), ancre=1_000.0)
    assert poids["AAA"] == 0.0
    assert poids["BBB"] > 0


@pytest.mark.parametrize("ancre", [0, -1.0, None])
def test_une_ancre_invalide_echoue_bruyamment(ancre):
    """Une ancre nulle produirait des poids infinis. Mieux vaut planter que
    renvoyer un portefeuille absurde."""
    with pytest.raises(ValueError, match="Ancre de conviction invalide"):
        poids_ancres(_conviction({"AAA": 50.0}), ancre=ancre)


# --------------------------------------------------------------------------- #
# Branchement
# --------------------------------------------------------------------------- #
def _candidates(n: int = 10) -> pd.DataFrame:
    """Assez de lignes pour que le plafond par position soit atteignable : en
    deçà, `capped_weights` renvoie le plafond pour chacune et la somme ne vaut
    pas 1 (cf. sa docstring)."""
    return pd.DataFrame([{"symbol": f"S{i}", "sector": "Technologie"} for i in range(n)])


def _conv(n: int = 10) -> pd.Series:
    return pd.Series({i: 10.0 * (i + 1) for i in range(n)}, dtype=float)


def test_construire_poids_bascule_sur_l_ancre_quand_elle_est_donnee():
    historique = construire_poids(_candidates(), _conv(), max_weight_per_sector_pct=0.0)
    ancree = construire_poids(_candidates(), _conv(), max_weight_per_sector_pct=0.0,
                              conviction_anchor=1_000.0)

    assert sum(historique.values()) == pytest.approx(1.0)
    # Somme des convictions = 550, divisée par l'ancre de 1 000.
    assert sum(ancree.values()) == pytest.approx(0.55)


def test_sans_ancre_la_ponderation_reste_celle_d_avant():
    assert (construire_poids(_candidates(), _conv(), max_weight_per_sector_pct=0.0)
            == construire_poids(_candidates(), _conv(), max_weight_per_sector_pct=0.0,
                                conviction_anchor=None))


def test_les_trois_strategies_existantes_gardent_la_ponderation_historique():
    """Garde-fou de séparation : l'ancre n'est demandée par aucune d'elles, et
    le défaut de config reste None -- leurs chiffres ne bougent pas."""
    assert config.BACKTEST_CONVICTION_ANCHOR is None
    for nom in ("valuation_gap_dcf", "valuation_gap_sector_neutral", "valuation_gap_combined"):
        assert STRATEGY_REGISTRY[nom]().conviction_anchor is None, nom


def test_la_strategie_ancree_lit_le_defaut_de_config(monkeypatch):
    """L'ancre se CALIBRE : elle doit pouvoir bouger sans toucher au code."""
    assert ValuationGapCombinedAncreeStrategy().conviction_anchor is None
    monkeypatch.setattr(config, "BACKTEST_CONVICTION_ANCHOR", 8_000.0)
    assert ValuationGapCombinedAncreeStrategy().conviction_anchor == 8_000.0
    # Un argument explicite l'emporte sur le défaut.
    assert ValuationGapCombinedAncreeStrategy(conviction_anchor=4_000.0).conviction_anchor == 4_000.0
