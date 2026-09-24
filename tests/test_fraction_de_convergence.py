"""Le README annonce la fraction de convergence que la stratégie utilise vraiment.

CE QUI S'EST PASSÉ. `config.OPTIONS_EV_CONVERGENCE_FRACTION_DEFAULT` valait
0.8 depuis le premier commit, pendant que le commentaire juste au-dessus et le
README justifiaient 0,5 (« la même hypothèse implicite que le strike à
mi-chemin »). Personne ne lisait l'un contre l'autre. La valeur est restée,
mesure à l'appui (voir le commentaire de config) ; c'est le texte qui a été
corrigé -- et ce test l'empêche de redevenir faux sans bruit.
"""

from __future__ import annotations

import pathlib
import re

import config
from backtest.strategies.valuation_gap_expected_value_options import (
    ValuationGapExpectedValueOptionsStrategy,
)

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


def test_le_readme_annonce_la_valeur_de_config():
    texte = README.read_text(encoding="utf-8")
    annonces = re.findall(
        r"\*\*Fraction de convergence\*\* \(`OPTIONS_EV_CONVERGENCE_FRACTION_DEFAULT`, ([0-9]+,[0-9]+)\)",
        texte)
    assert annonces, "le README n'annonce plus la fraction de convergence par défaut"
    assert {float(a.replace(",", ".")) for a in annonces} == {
        config.OPTIONS_EV_CONVERGENCE_FRACTION_DEFAULT}


def test_la_strategie_utilise_la_valeur_de_config():
    """Le défaut documenté est bien celui que le backtest joue quand on ne
    passe pas --strategy-param convergence_fraction."""
    assert ValuationGapExpectedValueOptionsStrategy().convergence_fraction == \
        config.OPTIONS_EV_CONVERGENCE_FRACTION_DEFAULT
