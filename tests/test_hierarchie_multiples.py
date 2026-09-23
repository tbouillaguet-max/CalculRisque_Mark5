"""HIÉRARCHIE DES MULTIPLES : lequel tranche quand les trois divergent ?

CE QUI EST EN JEU. Trois multiples donnent trois valeurs théoriques pour la même
action, et elles divergent -- sur les 13 240 lignes où P/E et EV/EBITDA
coexistent, l'écart médian entre les deux vaut 19,5 points de cours. Le choix de
celui qui tranche est un réglage de SIGNAL, et jusqu'ici il n'avait jamais été
mesuré : `MULTIPLE_COMBINATION` a été fixé sur un argument de littérature
(Liu, Nissim & Thomas 2002), pas sur ce dépôt.

LES DEUX TESTS QUI COMPTENT :

`test_la_recombinaison_reproduit_exactement_le_parquet` -- toute la mesure
repose sur le fait que rejouer la combinaison depuis `price_from_*` donne
EXACTEMENT ce que 06b a écrit. Si ce n'était pas le cas, l'axe comparerait des
réimplémentations et non des hiérarchies, et rien ne le signalerait.

`test_le_parquet_en_production_porte_flat_pas_tiers` -- ce que la recombinaison
a révélé sans qu'on le cherche : le fichier de signal et la configuration se
contredisent. Ce test échouera le jour où 06b sera rejoué, et c'est son objet :
il oblige à venir relire ce que la contradiction impliquait plutôt qu'à la
découvrir une seconde fois.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
import hierarchie_multiples as hm

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _implied(**colonnes) -> pd.DataFrame:
    return pd.DataFrame(colonnes, dtype=float)


# --------------------------------------------------------------------------- #
# La mécanique des rangs
# --------------------------------------------------------------------------- #
def test_un_rang_ne_sert_qu_aux_lignes_qu_aucun_meilleur_n_a_servies():
    """LE point de la hiérarchie : EV/Sales ne doit jamais départager quand un
    multiple de résultats existe."""
    implied = _implied(**{"P/E": [100.0, np.nan], "EV/EBITDA": [50.0, np.nan],
                          "EV/Sales": [10.0, 10.0]})
    valeurs = hm.combiner(implied, "tiers")
    assert valeurs.iloc[0] == pytest.approx(75.0)      # médiane de P/E et EV/EBITDA
    assert valeurs.iloc[1] == pytest.approx(10.0)      # repli sur EV/Sales


def test_la_mediane_a_plat_laisse_ev_sales_departager():
    """Contrôle : c'est le comportement dont la hiérarchie cherche à sortir, et
    sans lui le test précédent ne prouverait rien."""
    implied = _implied(**{"P/E": [100.0], "EV/EBITDA": [50.0], "EV/Sales": [10.0]})
    assert hm.combiner(implied, "flat").iloc[0] == pytest.approx(50.0)
    assert hm.combiner(implied, "tiers").iloc[0] == pytest.approx(75.0)


def test_pe_first_et_ebitda_first_choisissent_vraiment():
    implied = _implied(**{"P/E": [100.0], "EV/EBITDA": [50.0], "EV/Sales": [10.0]})
    assert hm.combiner(implied, "pe_first").iloc[0] == pytest.approx(100.0)
    assert hm.combiner(implied, "ebitda_first").iloc[0] == pytest.approx(50.0)


def test_le_repli_descend_de_rang_en_rang():
    """`pe_first` a trois rangs : sans P/E on prend EV/EBITDA, sans lui EV/Sales."""
    implied = _implied(**{"P/E": [np.nan, np.nan], "EV/EBITDA": [50.0, np.nan],
                          "EV/Sales": [10.0, 10.0]})
    valeurs = hm.combiner(implied, "pe_first")
    assert valeurs.iloc[0] == pytest.approx(50.0)
    assert valeurs.iloc[1] == pytest.approx(10.0)


def test_une_ligne_sans_aucun_multiple_ne_recoit_rien():
    implied = _implied(**{"P/E": [np.nan], "EV/EBITDA": [np.nan], "EV/Sales": [np.nan]})
    for nom in hm.HIERARCHIES:
        assert pd.isna(hm.combiner(implied, nom).iloc[0]), nom


def test_une_hierarchie_inconnue_echoue_bruyamment():
    """Une faute de frappe qui retomberait en silence sur le défaut ferait
    croire à une mesure qui n'a pas eu lieu."""
    with pytest.raises(ValueError, match="Hiérarchie de multiples inconnue"):
        hm.combiner(_implied(**{"P/E": [1.0]}), "pe_frist")


def test_une_table_de_rangs_explicite_est_acceptee():
    """Pour essayer une hiérarchie hors catalogue sans modifier le module."""
    implied = _implied(**{"P/E": [100.0], "EV/EBITDA": [50.0], "EV/Sales": [10.0]})
    assert hm.combiner(implied, {"EV/Sales": 1, "P/E": 2, "EV/EBITDA": 3}).iloc[0] == 10.0


# --------------------------------------------------------------------------- #
# LA propriété qui rend l'axe propre
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nom", sorted(hm.HIERARCHIES))
def test_la_couverture_ne_depend_pas_de_la_hierarchie(nom):
    """Sans cette invariance, l'axe mesurerait un effet de COUVERTURE déguisé en
    effet de hiérarchie : une hiérarchie qui valorise moins de lignes change le
    nombre de candidates, donc tout le portefeuille, et on attribuerait à la
    qualité du multiple ce qui vient du nombre de titres."""
    implied = _implied(**{
        "P/E": [100.0, np.nan, np.nan, np.nan],
        "EV/EBITDA": [50.0, 50.0, np.nan, np.nan],
        "EV/Sales": [10.0, 10.0, 10.0, np.nan],
    })
    assert list(hm.combiner(implied, nom).notna()) == [True, True, True, False]


# --------------------------------------------------------------------------- #
# La recombinaison d'un parquet déjà écrit
# --------------------------------------------------------------------------- #
def _parquet() -> pd.DataFrame:
    if not config.VALORISATION_COMBINEE_FILE.exists():
        pytest.skip("valorisation_combinee_historique.parquet absent")
    try:
        return pd.read_parquet(config.VALORISATION_COMBINEE_FILE)
    except (OSError, ValueError) as exc:          # pointeur LFS non matérialisé
        pytest.skip(f"parquet illisible : {exc}")


def test_la_recombinaison_reproduit_exactement_le_parquet():
    """LE test dont dépend toute la mesure. Rejouer la combinaison depuis les
    prix implicites stockés doit rendre EXACTEMENT le gap_pct que 06b a écrit --
    sinon l'axe compare des réimplémentations, pas des hiérarchies.

    La hiérarchie testée est celle que le fichier porte RÉELLEMENT (`flat`, cf.
    le test suivant), pas celle que la config annonce."""
    d = _parquet()
    refait = hm.recombiner(d, "flat")
    assert (refait["gap_pct"] - d["gap_pct"]).abs().max() == 0.0
    assert (refait["valuation_multiples_per_share"]
            - d["valuation_multiples_per_share"]).abs().max() == 0.0
    assert (refait["valuation_theoretical_per_share"]
            - d["valuation_theoretical_per_share"]).abs().max() == 0.0


def test_le_parquet_en_production_porte_flat_pas_tiers():
    """CE QUE LA RECOMBINAISON A RÉVÉLÉ. Le fichier de signal et la
    configuration se contredisent : tous les backtests `combinee` de ce dépôt
    ont tourné sur la médiane à trois voix, pas sur la hiérarchie que
    `MULTIPLE_COMBINATION` documente.

    Ce test tombera le jour où 06b sera rejoué -- c'est voulu. Il force à venir
    relire ce que la contradiction impliquait (les chiffres publiés du README
    portent sur `flat`) au lieu de la redécouvrir par accident."""
    d = _parquet()
    assert config.MULTIPLE_COMBINATION == "tiers"
    ecart_flat = (hm.recombiner(d, "flat")["gap_pct"] - d["gap_pct"]).abs().max()
    ecart_tiers = (hm.recombiner(d, "tiers")["gap_pct"] - d["gap_pct"]).abs().max()
    assert ecart_flat == 0.0, "le parquet ne porte plus `flat`"
    assert ecart_tiers > 0.0, (
        "le parquet porte maintenant `tiers` : 06b a été rejoué. Relis la section "
        "« hiérarchie des multiples » du README -- les chiffres publiés portaient sur "
        "`flat`, et la référence du test apparié doit suivre (_reference_combo)."
    )


@pytest.mark.parametrize("nom", sorted(hm.HIERARCHIES))
def test_la_recombinaison_preserve_le_jeu_de_lignes_et_la_source(nom):
    """`source` distingue une valorisation par multiples d'un repli DCF, et la
    stratégie options filtre dessus. La couverture étant invariante, il ne doit
    jamais bouger -- le recalculer plutôt que le supposer est ce qui le vérifie."""
    d = _parquet()
    refait = hm.recombiner(d, nom)
    assert len(refait) == len(d)
    assert (refait["source"] == d["source"]).all()
    assert (refait["valuation_multiples_per_share"].notna()
            == d["valuation_multiples_per_share"].notna()).all()


def test_les_hierarchies_changent_reellement_le_signal():
    """Un axe qui ne déplace rien se lirait comme un axe mesuré et sans effet.
    Ici il déplace des milliers de lignes -- la mesure porte sur du réel."""
    d = _parquet()
    reference = hm.recombiner(d, "flat")["gap_pct"]
    for nom in ("tiers", "pe_first", "ebitda_first"):
        change = (hm.recombiner(d, nom)["gap_pct"] - reference).abs().gt(1e-9).sum()
        assert change > 5_000, f"{nom} ne change que {change} lignes"


def test_sans_hierarchie_le_parquet_n_est_pas_touche():
    d = _parquet()
    assert hm.recombiner(d, None) is d


def test_un_parquet_sans_les_prix_implicites_echoue_avec_le_remede():
    """Un 06b antérieur à l'écriture de ces colonnes : le message doit dire quoi
    relancer, pas seulement qu'une colonne manque."""
    d = _parquet().drop(columns=["price_from_pe"])
    with pytest.raises(ValueError, match="relance 06b"):
        hm.recombiner(d, "tiers")


# --------------------------------------------------------------------------- #
# 06b et le module partagé ne peuvent pas diverger
# --------------------------------------------------------------------------- #
def test_06b_delegue_au_module_partage():
    """Deux implémentations de la même combinaison finiraient par diverger, et
    l'écart ne se verrait que dans les chiffres."""
    import importlib
    mod = importlib.import_module("06b_calcul_valorisation_combinee")
    implied = _implied(**{"P/E": [100.0], "EV/EBITDA": [50.0], "EV/Sales": [10.0]})
    assert mod.combine_implied_prices(implied, "flat").iloc[0] == pytest.approx(50.0)
    assert mod.combine_implied_prices(implied, "tiers").iloc[0] == pytest.approx(75.0)
    # Et le défaut suit toujours config.MULTIPLE_COMBINATION.
    attendu = 75.0 if config.MULTIPLE_COMBINATION == "tiers" else 50.0
    assert mod.combine_implied_prices(implied).iloc[0] == pytest.approx(attendu)


def test_06b_refuse_toujours_un_mode_inconnu():
    import importlib
    mod = importlib.import_module("06b_calcul_valorisation_combinee")
    with pytest.raises(ValueError, match="attend 'tiers' ou 'flat'"):
        mod.combine_implied_prices(_implied(**{"P/E": [1.0]}), "mediane")


def test_06b_lit_bien_la_table_de_rangs_de_config(monkeypatch):
    """MULTIPLE_RELIABILITY_TIERS doit rester modifiable sans toucher au
    catalogue de hiérarchies."""
    import importlib
    mod = importlib.import_module("06b_calcul_valorisation_combinee")
    implied = _implied(**{"P/E": [100.0], "EV/EBITDA": [50.0], "EV/Sales": [10.0]})
    monkeypatch.setattr(config, "MULTIPLE_RELIABILITY_TIERS",
                        {"EV/Sales": 1, "P/E": 2, "EV/EBITDA": 2})
    assert mod.combine_implied_prices(implied, "tiers").iloc[0] == pytest.approx(10.0)
