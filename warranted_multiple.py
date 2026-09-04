"""
Multiple MÉRITÉ : quel multiple une entreprise devrait avoir compte tenu de ses
propres fondamentaux, estimé par régression en coupe sur ses pairs.

LE PROBLÈME QU'IL ADRESSE
-------------------------
06b_calcul_valorisation_combinee.py valorise aujourd'hui par le multiple
sectoriel appliqué aux fondamentaux de l'entreprise. C'est un repère
INCONDITIONNEL : il ne sait pas distinguer une entreprise mal valorisée d'une
entreprise justement valorisée pour ce qu'elle est. Or une décote sur la
médiane sectorielle est le plus souvent MÉRITÉE -- moins de croissance, marges
plus faibles, plus de levier, rentabilité du capital inférieure.

L'enjeu n'est pas théorique : Fama & French (JFE, 2015) montrent que HML
devient redondant une fois la rentabilité (RMW) et l'investissement (CMA)
inclus. Autrement dit, la part de la « cheapness » corrélée à une faible
rentabilité ne paie pas -- et c'est précisément celle qu'un écart brut à la
médiane sectorielle charge le plus.

Bhojraj & Lee (Journal of Accounting Research, 2002, « Who Is My Peer? ») posent
le remède : estimer par régression en coupe le multiple que les fondamentaux
JUSTIFIENT, et ne retenir comme candidat à la mispricing que le RÉSIDU.

    ln(M_i) = a + b·x_i + e_i        M_mérité_i = exp(a + b·x_i)

CE MODULE NE DÉCIDE RIEN
-------------------------
Il calcule le multiple mérité, il ne le branche nulle part. C'est délibéré :
15_test_multiple_merite.py compare d'abord sa capacité de prédiction à celle du
multiple sectoriel, HORS ÉCHANTILLON. Si le multiple mérité ne prédit pas mieux,
il n'y a aucune raison d'espérer qu'il fasse mieux en stratégie, et le brancher
dans 06b serait une complication gratuite.

POURQUOI EN LOG
---------------
Un multiple est un ratio : borné par zéro, fortement asymétrique à droite.
Régresser le niveau laisserait un P/E de sortie de perte dominer l'ajustement,
et autoriserait des multiples mérités négatifs. En log, les erreurs deviennent
multiplicatives -- la structure d'erreur qui convient à un ratio, et celle-là
même qui fonde la moyenne harmonique (cf. config.SECTOR_MULTIPLE_AGGREGATOR).

LE PIÈGE DE CIRCULARITÉ
------------------------
Expliquer EV/EBITDA par la marge d'EBITDA met partiellement la même grandeur
des deux côtés. Les régresseurs sont donc peu nombreux, économiquement motivés,
et aucun n'est une transformation directe du multiple expliqué. Le jeu reste
volontairement court : chaque régresseur supplémentaire améliore l'ajustement
dans l'échantillon et absorbe un peu plus de ce qu'on cherche, à savoir le
résidu.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Régresseurs, tous construits depuis des colonnes déjà produites par
# 04_recuperation_10k.py / 05_calcul_multiples.py -- aucune collecte nouvelle.
FEATURES: tuple = (
    "marge_ebitda",       # rentabilité opérationnelle
    "croissance_revenue", # croissance
    "levier",             # risque financier
    "roic",               # rentabilité du capital employé
    "taille",             # effet taille
)

# Observations minimales pour ajuster une régression. Une règle usuelle est
# d'exiger au moins ~10 observations par régresseur ; en dessous, les
# coefficients sont du bruit et le « multiple mérité » est un surajustement
# déguisé. C'est LA contrainte qui décide du regroupement viable : un
# secteur x millésime dépasse rarement 30 pairs, une coupe complète de
# l'indice en compte plusieurs centaines.
MIN_OBS_PER_FEATURE = 10


def _safe_ratio(numerateur: pd.Series, denominateur: pd.Series) -> pd.Series:
    """Ratio avec dénominateur non strictement positif neutralisé en NaN.

    Pas de `replace({0: nan})` : un dénominateur NÉGATIF (capitaux propres
    négatifs, EBITDA négatif) produit un ratio de signe inversé qui n'a aucun
    sens économique et que la régression interpréterait comme une observation
    légitime."""
    denominateur = denominateur.where(denominateur > 0)
    return numerateur / denominateur


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Régresseurs, dans l'ordre de FEATURES.

    `df` doit porter les colonnes de 05_calcul_multiples.py (financials +
    cours) et être triée par (symbol, filed_date) pour que la croissance ait un
    sens -- ce que fait build_input_table.

    Les colonnes absentes donnent un régresseur entièrement NaN plutôt qu'une
    exception : un cache régénéré par une version antérieure du pipeline n'a
    pas forcément tous les postes, et c'est à l'appelant de décider s'il
    renonce (cf. fit_predict, qui écarte les lignes incomplètes)."""
    def colonne(nom: str) -> pd.Series:
        if nom not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return pd.to_numeric(df[nom], errors="coerce")

    revenue = colonne("revenue")
    ebitda, ebit = colonne("ebitda"), colonne("ebit")
    total_assets, total_liabilities = colonne("total_assets"), colonne("total_liabilities")
    net_debt = colonne("net_debt")
    tax_rate = colonne("tax_rate").clip(lower=0.0, upper=0.6).fillna(0.21)

    # Capital employé = capitaux propres + dette nette. Le passer par
    # _safe_ratio écarte les capitaux propres négatifs, où un ROIC n'a pas de
    # sens (le dénominateur change de signe, pas la performance).
    capitaux_propres = total_assets - total_liabilities
    capital_employe = capitaux_propres + net_debt.fillna(0)

    # Croissance du chiffre d'affaires : variation d'une période à la
    # PRÉCÉDENTE CHRONOLOGIQUEMENT, par entreprise. En log, pour rester
    # symétrique et borné -- une croissance de +100% et une chute de -50%
    # comptent pour le même mouvement au signe près.
    if "symbol" in df.columns:
        precedent = revenue.groupby(df["symbol"]).shift(1)
    else:
        precedent = pd.Series(np.nan, index=df.index, dtype=float)
    croissance = np.log(_safe_ratio(revenue, precedent))

    features = pd.DataFrame({
        "marge_ebitda": _safe_ratio(ebitda, revenue),
        "croissance_revenue": croissance,
        "levier": _safe_ratio(net_debt, ebitda),
        "roic": _safe_ratio(ebit * (1 - tax_rate), capital_employe),
        # Taille en log : la relation multiple/taille est multiplicative, et le
        # chiffre d'affaires brut s'étale sur quatre ordres de grandeur.
        "taille": np.log(revenue.where(revenue > 0)),
    })
    return features.replace([np.inf, -np.inf], np.nan)[list(FEATURES)]


def _winsorize(features: pd.DataFrame, quantile: float = 0.02) -> pd.DataFrame:
    """Écrête chaque régresseur à ses quantiles extrêmes.

    Une régression par moindres carrés est dominée par ses points extrêmes : un
    levier de 40x (entreprise proche du défaut) ou une croissance de +900%
    (acquisition transformante) déplacerait tous les coefficients pour toute la
    coupe. On écrête plutôt que d'exclure -- exclure retirerait de l'échantillon
    des entreprises qu'il faut bien valoriser."""
    if features.empty:
        return features
    bas = features.quantile(quantile)
    haut = features.quantile(1 - quantile)
    return features.clip(lower=bas, upper=haut, axis=1)


def fit_predict(
    peers: pd.DataFrame,
    targets: pd.DataFrame,
    multiple_col: str,
    features: Sequence[str] = FEATURES,
    sector_column: Optional[str] = "sector",
    min_obs_per_feature: int = MIN_OBS_PER_FEATURE,
) -> Optional[pd.Series]:
    """Multiple mérité de chaque ligne de `targets`, estimé sur `peers`.

    `peers` et `targets` portent le multiple observé (`multiple_col`) et les
    colonnes de régresseurs produites par build_features. `targets` n'a PAS
    besoin d'un multiple observé -- c'est la valeur qu'on cherche à expliquer,
    pas une entrée du calcul.

    HORS ÉCHANTILLON PAR CONSTRUCTION : la ligne valorisée n'appartient pas à
    `peers`. C'est l'appelant qui garantit cette séparation, exactement comme
    06b exclut déjà une ligne de sa propre médiane sectorielle -- se comparer à
    soi tire mécaniquement le résidu vers zéro.

    `sector_column` ajoute des indicatrices sectorielles, ce qui permet
    d'ajuster sur une coupe complète de l'indice plutôt que secteur par
    secteur : les indicatrices récupèrent le niveau sectoriel que la médiane
    capturait, tout en donnant à la pente des fondamentaux les centaines
    d'observations qu'elle exige. None pour ajuster sans effet sectoriel (le
    cas d'un ajustement déjà restreint à un seul secteur).

    None quand l'échantillon est trop court pour que les coefficients veuillent
    dire quelque chose -- l'appelant se rabat alors sur le multiple sectoriel.
    """
    features = list(features)
    if multiple_col not in peers.columns:
        return None
    features = [f for f in features if f in peers.columns and f in targets.columns]
    if not features:
        return None

    # RÉGRESSEURS INEXPLOITABLES ÉCARTÉS PLUTÔT QUE SUBIS. `dropna` sur
    # l'ensemble des colonnes éliminerait toute ligne à laquelle il manque UN
    # régresseur -- et un régresseur peut manquer massivement pour une raison
    # structurelle : la croissance du chiffre d'affaires est inconnue à la
    # PREMIÈRE période de chaque entreprise, et l'est partout si le cache ne
    # porte qu'une période. Exiger les cinq ferait alors perdre non pas
    # quelques lignes mais la régression entière, silencieusement.
    #
    # On retire donc d'abord les régresseurs trop peu renseignés, puis on
    # n'exige que les survivants. Mieux vaut une régression à trois variables
    # sur tout l'échantillon qu'aucune régression du tout.
    utilisables = [
        f for f in features
        if peers[f].notna().mean() >= 0.5 and targets[f].notna().any()
    ]
    if not utilisables:
        return None
    if len(utilisables) < len(features):
        logger.debug(
            "Régresseurs écartés faute d'être renseignés : %s",
            sorted(set(features) - set(utilisables)))
    features = utilisables

    ajustables = peers.dropna(subset=[*features, multiple_col])
    ajustables = ajustables[ajustables[multiple_col] > 0]
    if ajustables.empty or targets.empty:
        return None

    X_brut = _winsorize(ajustables[features])
    y = np.log(ajustables[multiple_col].to_numpy(dtype=float))

    # Indicatrices sectorielles, sans la modalité de référence (colinéarité
    # avec la constante). Un secteur représenté par une seule ligne
    # n'apporterait qu'un degré de liberté consommé pour rien.
    secteurs: list = []
    if sector_column and sector_column in ajustables.columns:
        comptes = ajustables[sector_column].value_counts()
        secteurs = sorted(comptes[comptes >= 2].index.dropna().tolist())[1:]

    n_parametres = 1 + len(features) + len(secteurs)
    if len(ajustables) < max(n_parametres + 1, min_obs_per_feature * len(features)):
        return None

    def matrice(source: pd.DataFrame, valeurs: pd.DataFrame) -> np.ndarray:
        blocs = [np.ones((len(valeurs), 1)), valeurs[features].to_numpy(dtype=float)]
        for secteur in secteurs:
            blocs.append((source[sector_column] == secteur).to_numpy(dtype=float).reshape(-1, 1))
        return np.hstack(blocs)

    X = matrice(ajustables, X_brut)
    coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
    if not np.all(np.isfinite(coefficients)):
        return None

    # Les cibles sont écrêtées aux bornes DES PAIRS, jamais aux leurs : une
    # cible extrême doit être ramenée dans le domaine où les coefficients ont
    # été estimés, pas redéfinir ce domaine.
    cibles = targets.dropna(subset=features)
    if cibles.empty:
        return None
    X_cible = cibles[features].clip(
        lower=X_brut.min(), upper=X_brut.max(), axis=1)
    predites = matrice(cibles, X_cible) @ coefficients

    return pd.Series(np.exp(predites), index=cibles.index, name=f"{multiple_col}_merite")
