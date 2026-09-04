"""Multiple mérité : la régression, et le banc d'essai qui décide de la brancher.

Le banc d'essai (15_test_multiple_merite.py) doit trancher DANS LES DEUX SENS,
et c'est ce que ces tests vérifient sur des données dont on connaît la vérité :

    - quand les fondamentaux expliquent réellement le multiple, le mérité doit
      gagner ;
    - quand ils n'expliquent RIEN, il doit perdre (une régression sur du bruit
      ajoute de la variance sans ajouter d'information).

Un banc d'essai qui ne sait dire que « oui » ne décide de rien.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

import warranted_multiple

m15 = importlib.import_module("15_test_multiple_merite")


# --------------------------------------------------------------------------- #
# Construction des régresseurs
# --------------------------------------------------------------------------- #

def test_les_regresseurs_sont_construits_depuis_les_colonnes_du_pipeline():
    df = pd.DataFrame({
        "symbol": ["AAA", "AAA"],
        "revenue": [1000.0, 1200.0],
        "ebitda": [200.0, 250.0],
        "ebit": [150.0, 190.0],
        "total_assets": [900.0, 1000.0],
        "total_liabilities": [400.0, 420.0],
        "net_debt": [100.0, 120.0],
        "tax_rate": [0.21, 0.21],
    })
    features = warranted_multiple.build_features(df)

    assert list(features.columns) == list(warranted_multiple.FEATURES)
    assert features["marge_ebitda"].iloc[0] == pytest.approx(0.20)
    assert features["levier"].iloc[0] == pytest.approx(0.5)
    # Croissance en log entre les deux périodes de la même entreprise.
    assert features["croissance_revenue"].iloc[1] == pytest.approx(np.log(1.2))
    # La première période n'a pas de précédente : croissance inconnue, pas 0.
    assert pd.isna(features["croissance_revenue"].iloc[0])
    # ROIC = EBIT(1-t) / (capitaux propres + dette nette) = 150x0,79 / 600
    assert features["roic"].iloc[0] == pytest.approx(150 * 0.79 / 600)


def test_un_denominateur_negatif_ne_produit_pas_un_ratio_inverse():
    """La garde porte sur le DÉNOMINATEUR, et seulement sur lui.

    Un numérateur négatif est une information légitime : une marge d'EBITDA
    de -5% décrit une entreprise en perte, et la régression doit la voir. Un
    dénominateur négatif, lui, INVERSE le signe du ratio sans que la
    performance change -- un levier « négatif » sur EBITDA négatif se lirait
    comme un bilan sain."""
    df = pd.DataFrame({
        "symbol": ["AAA"], "revenue": [1000.0], "ebitda": [-50.0], "ebit": [100.0],
        "total_assets": [500.0], "total_liabilities": [900.0], "net_debt": [200.0],
        "tax_rate": [0.21],
    })
    features = warranted_multiple.build_features(df)

    # Numérateur négatif, dénominateur sain : valeur conservée, et négative.
    assert features["marge_ebitda"].iloc[0] == pytest.approx(-0.05)
    # Dénominateurs négatifs : neutralisés.
    assert pd.isna(features["levier"].iloc[0])   # EBITDA négatif
    assert pd.isna(features["roic"].iloc[0])     # capital employé négatif


def test_les_colonnes_absentes_donnent_un_regresseur_vide_pas_une_exception():
    """Un cache régénéré par une version antérieure du pipeline n'a pas
    forcément tous les postes."""
    features = warranted_multiple.build_features(pd.DataFrame({"symbol": ["AAA"]}))
    assert list(features.columns) == list(warranted_multiple.FEATURES)
    assert features.isna().all().all()


# --------------------------------------------------------------------------- #
# Régression
# --------------------------------------------------------------------------- #

def _panel(n: int, effet_marge: float, graine: int = 0, bruit: float = 0.05) -> pd.DataFrame:
    """Panel dont le multiple est expliqué par la marge à hauteur de
    `effet_marge`, plus un bruit. effet_marge=0 -> les fondamentaux
    n'expliquent rien."""
    rng = np.random.default_rng(graine)
    marge = rng.uniform(0.05, 0.40, n)
    log_multiple = 2.3 + effet_marge * marge + rng.normal(0, bruit, n)
    return pd.DataFrame({
        "symbol": [f"S{i:03d}" for i in range(n)],
        "sector": ["Technologie"] * n,
        "marge_ebitda": marge,
        "croissance_revenue": rng.normal(0.05, 0.02, n),
        "levier": rng.uniform(0.5, 3.0, n),
        "roic": rng.uniform(0.05, 0.25, n),
        "taille": rng.uniform(6.0, 12.0, n),
        "EV/EBITDA": np.exp(log_multiple),
    })


def test_la_regression_retrouve_une_relation_qui_existe():
    """Contrôle de base : sur des données où le multiple dépend fortement de la
    marge, le multiple mérité doit suivre la marge."""
    panel = _panel(300, effet_marge=4.0, graine=1)
    pairs, cibles = panel.iloc[:280], panel.iloc[280:]

    predit = warranted_multiple.fit_predict(pairs, cibles, "EV/EBITDA", sector_column=None)
    assert predit is not None

    observe = cibles["EV/EBITDA"]
    erreur = np.abs(np.log(observe / predit))
    erreur_agregat = np.abs(np.log(observe / pairs["EV/EBITDA"].median()))
    assert erreur.median() < erreur_agregat.median()


def test_un_echantillon_trop_court_ne_rend_rien():
    """En dessous de MIN_OBS_PER_FEATURE par régresseur, les coefficients sont
    du bruit et le « multiple mérité » un surajustement déguisé. Mieux vaut
    None -- l'appelant se rabat sur le multiple sectoriel."""
    panel = _panel(20, effet_marge=4.0, graine=2)
    assert warranted_multiple.fit_predict(
        panel.iloc[:15], panel.iloc[15:], "EV/EBITDA", sector_column=None) is None


def test_les_indicatrices_sectorielles_permettent_d_ajuster_sur_toute_la_coupe():
    """C'est ce qui rend le mode --grouping millesime viable : la pente des
    fondamentaux obtient des centaines d'observations, les indicatrices
    récupèrent le niveau sectoriel."""
    a = _panel(150, effet_marge=3.0, graine=3)
    b = _panel(150, effet_marge=3.0, graine=4).assign(sector="Santé")
    b["EV/EBITDA"] *= 2.0                       # niveau sectoriel franchement différent
    panel = pd.concat([a, b], ignore_index=True)

    cibles = panel.groupby("sector", group_keys=False).tail(5)
    pairs = panel.drop(index=cibles.index)
    predit = warranted_multiple.fit_predict(pairs, cibles, "EV/EBITDA", sector_column="sector")
    assert predit is not None

    # Les prédictions doivent séparer les deux secteurs, pas les moyenner.
    par_secteur = predit.groupby(cibles["sector"]).median()
    assert par_secteur["Santé"] > 1.5 * par_secteur["Technologie"]


def test_une_cible_extreme_est_ramenee_dans_le_domaine_des_pairs():
    """Extrapoler une régression loin hors du domaine d'estimation produit des
    multiples absurdes. La cible est écrêtée aux bornes DES PAIRS -- jamais aux
    siennes, sinon elle redéfinirait le domaine."""
    panel = _panel(300, effet_marge=4.0, graine=5)
    pairs = panel.iloc[:280]
    extreme = panel.iloc[280:281].copy()
    extreme["marge_ebitda"] = 50.0              # hors de tout domaine plausible

    predit = warranted_multiple.fit_predict(pairs, extreme, "EV/EBITDA", sector_column=None)
    assert predit is not None
    assert predit.iloc[0] < pairs["EV/EBITDA"].max() * 2


# --------------------------------------------------------------------------- #
# Le banc d'essai doit savoir dire NON
# --------------------------------------------------------------------------- #

def _millesime(panel: pd.DataFrame) -> pd.DataFrame:
    """Complète un panel des colonnes qu'attend le banc d'essai."""
    return panel.assign(
        period_type="FY", fiscal_year=2020, fiscal_quarter=None,
        filed_date=pd.date_range("2020-02-01", periods=len(panel), freq="D"),
        revenue=np.exp(panel["taille"]),
        ebitda=np.exp(panel["taille"]) * panel["marge_ebitda"],
        ebit=np.exp(panel["taille"]) * panel["marge_ebitda"] * 0.75,
        total_assets=np.exp(panel["taille"]) * 1.2,
        total_liabilities=np.exp(panel["taille"]) * 0.5,
        net_debt=np.exp(panel["taille"]) * panel["marge_ebitda"] * panel["levier"],
        tax_rate=0.21,
    )


def test_le_banc_d_essai_conclut_en_faveur_du_merite_quand_il_est_meilleur():
    """Les fondamentaux expliquent réellement le multiple : le mérité doit
    gagner, et le banc d'essai doit le voir."""
    panel = _millesime(_panel(400, effet_marge=6.0, graine=6, bruit=0.03))
    resultats = m15.comparer(panel, "EV/EBITDA", "millesime", min_peers=60)

    assert not resultats.empty
    assert resultats["err_merite"].median() < resultats["err_sectoriel"].median()
    assert (resultats["err_merite"] < resultats["err_sectoriel"]).mean() > 0.6


def test_le_banc_d_essai_conclut_contre_le_merite_quand_il_n_apporte_rien():
    """LE test qui compte. Les fondamentaux n'expliquent RIEN du multiple : une
    régression sur du bruit ajoute de la variance sans information, et le banc
    d'essai doit le dire plutôt que de récompenser la complexité.

    Sans ce test, rien ne garantirait que le verdict ne soit pas acquis
    d'avance -- une régression ajuste toujours mieux DANS son échantillon, et
    c'est précisément pourquoi la ligne évaluée en est exclue."""
    panel = _millesime(_panel(400, effet_marge=0.0, graine=7, bruit=0.30))
    resultats = m15.comparer(panel, "EV/EBITDA", "millesime", min_peers=60)

    assert not resultats.empty
    assert resultats["err_merite"].median() >= resultats["err_sectoriel"].median()


def test_la_ligne_evaluee_est_exclue_de_ses_propres_pairs():
    """Sans cette exclusion, la régression gagnerait par construction : elle
    ajusterait sur une population contenant la réponse."""
    panel = _millesime(_panel(200, effet_marge=4.0, graine=8))
    resultats = m15.comparer(panel, "EV/EBITDA", "millesime", min_peers=60)

    assert not resultats.empty
    # Une erreur strictement nulle signalerait que la ligne s'est vue elle-même.
    assert (resultats["err_merite"] > 1e-9).all()


def test_seules_les_lignes_comparables_entrent_dans_le_verdict():
    """Comparer une méthode sur les cas faciles à l'autre sur tous les cas ne
    mesurerait rien : chaque ligne retenue porte les DEUX prédicteurs."""
    panel = _millesime(_panel(300, effet_marge=4.0, graine=9))
    resultats = m15.comparer(panel, "EV/EBITDA", "millesime", min_peers=60)

    assert resultats[["sectoriel", "merite", "observe"]].notna().all().all()
    assert (resultats[["sectoriel", "merite", "observe"]] > 0).all().all()


def test_les_premieres_lignes_du_millesime_sont_ignorees_faute_de_pairs():
    """Point-in-time : une ligne déposée tôt n'a pas assez de pairs déjà
    publiés. Elle doit être écartée, pas valorisée sur des pairs futurs."""
    panel = _millesime(_panel(200, effet_marge=4.0, graine=10))
    resultats = m15.comparer(panel, "EV/EBITDA", "millesime", min_peers=60)

    assert len(resultats) < len(panel)
    assert resultats["n_peers"].min() >= 60
