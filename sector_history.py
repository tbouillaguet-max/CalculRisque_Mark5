"""
Composition POINT-IN-TIME des groupes de pairs sectoriels, partagé par
06b_calcul_valorisation_combinee.py (médianes sectorielles) et les scripts qui
veulent raisonner "à la date" plutôt qu'"aujourd'hui".

Deux erreurs distinctes sont corrigées ici, et elles n'ont rien à voir l'une
avec l'autre malgré leur symptôme commun (un groupe de pairs qui n'est pas
celui qu'un investisseur aurait observé à la date valorisée).


1. BIAIS DE SURVIVANCE -- QUI était dans l'indice
-------------------------------------------------
multiples.parquet (05) ne contient que l'univers ACTUEL. Une médiane
sectorielle de 2012 y est donc calculée sur les seules entreprises encore
membres aujourd'hui : les faillites, rachats et sorties d'indice ont disparu
de l'échantillon. Comme les disparues étaient en moyenne moins bien
valorisées, la médiane historique est SURESTIMÉE, et avec elle tous les écarts
de valorisation mesurés contre elle.

La symétrie est rarement relevée mais tout aussi fausse : une entreprise
ENTRÉE dans l'indice en 2020 pesait dans la médiane de 2012 alors qu'elle n'en
faisait pas partie -- une entrée dans le S&P 500 récompense en général un
parcours boursier, donc ce biais-là pousse aussi la médiane vers le haut.

`members_asof` répond à la seule question qui vaille : qui était membre de
l'indice à cette date ? Elle s'appuie sur les spans d'appartenance produits
par 01b_historique_univers_sp500.py (config.UNIVERSE_HISTORY_FILE).

Attention : restreindre les pairs aux membres d'alors ne CRÉE pas les données
manquantes. Tant que 03b/04/04b n'ont pas été backfillés sur
config.UNIVERSE_FULL_FILE, les radiées restent absentes de multiples.parquet
et le biais persiste -- il est simplement devenu MESURABLE (voir
`coverage_report`), au lieu d'être invisible.


2. RECLASSIFICATIONS GICS -- DANS QUEL SECTEUR elle était
----------------------------------------------------------
Le secteur vient de la photo GICS ACTUELLE (01 -> 02) et était appliqué
rétroactivement à tout l'historique. Or la nomenclature GICS a été remaniée
plusieurs fois, et un remaniement déplace des entreprises entre secteurs :

    - 2016-09-16 : l'immobilier sort de "Financials" et devient un secteur à
      part entière ("Real Estate").
    - 2018-09-28 : "Telecommunication Services" devient "Communication
      Services" et absorbe les médias/divertissement (venus de "Consumer
      Discretionary") et l'internet grand public (venu d'"Information
      Technology"). C'est le remaniement le plus lourd : Alphabet et Meta
      étaient classées en technologie AVANT cette date.
    - 2023-03-17 : les processeurs de paiement passent d'"Information
      Technology" à "Financials", et quelques distributeurs de
      "Consumer Discretionary" à "Consumer Staples".

Conséquence sur les médianes : le P/E de la technologie avant 2018 est
aujourd'hui recalculé SANS Alphabet ni Meta (parties dans "Medias"), alors
qu'elles en faisaient partie ; et la médiane des médias d'avant 2018 est
calculée AVEC elles, alors qu'elles n'y étaient pas. Les deux séries sont
faussées de part et d'autre de la charnière, et l'entreprise valorisée est
comparée à un groupe de pairs qui n'a jamais existé à cette date.

`sector_asof` remonte le temps à partir du secteur actuel en rejouant à
l'envers les mouvements documentés ci-dessous.

LIMITE ASSUMÉE : cette table couvre les mouvements STRUCTURELS documentés
publiquement par S&P/MSCI, entreprise par entreprise pour les cas où le
mouvement n'est pas déductible du secteur seul. Elle ne couvre pas les
reclassements individuels au fil de l'eau (une entreprise qui change d'activité
et donc de sous-secteur, sans remaniement de la nomenclature) : ceux-là restent
invisibles, faute de source historique gratuite. Elle est délibérément
EXTENSIBLE -- ajouter un événement ou un ticker ne demande que d'éditer
GICS_RECLASSIFICATIONS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Set

import pandas as pd

import config

logger = logging.getLogger("sector_history")


# ----------------------------------------------------------------------------
# Reclassifications GICS
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Reclassification:
    """Un remaniement de la nomenclature GICS.

    `effective_date` : date à partir de laquelle le nouveau classement
    s'applique. AVANT cette date, les entreprises listées étaient dans leur
    secteur d'origine.

    `moves` : {symbole IBKR -> secteur AVANT le remaniement}, dans les
    libellés français de 02_categoriser_secteurs.py (ceux qui figurent
    réellement dans la colonne "sector" du pipeline).

    `sector_moves` : {secteur actuel -> secteur avant}, pour les mouvements
    qui concernent un secteur ENTIER et ne demandent donc pas de liste de
    tickers. Appliqué à toute entreprise du secteur cible qui n'est pas déjà
    nommée dans `moves`.
    """

    label: str
    effective_date: date
    description: str
    moves: Dict[str, str] = field(default_factory=dict)
    sector_moves: Dict[str, str] = field(default_factory=dict)


# Libellés utilisés ci-dessous (sortie de 02_categoriser_secteurs.py, via
# GICS_TO_SECTEUR) : "Technologie" = Information Technology, "Medias" =
# Communication Services, "Distribution" = Consumer Discretionary,
# "Services financiers" = Financials, "Immobilier" = Real Estate,
# "Agro-alimentaire et boissons" = Consumer Staples.
GICS_RECLASSIFICATIONS: List[Reclassification] = [
    Reclassification(
        label="real_estate_2016",
        effective_date=date(2016, 9, 16),
        description=(
            "L'immobilier (hors sociétés de crédit hypothécaire) sort de Financials "
            "et devient le 11e secteur GICS."
        ),
        # Mouvement de SECTEUR ENTIER : toute entreprise aujourd'hui classée
        # en immobilier était auparavant dans les financières. Pas besoin
        # d'énumérer les foncières une par une.
        sector_moves={"Immobilier": "Services financiers"},
    ),
    Reclassification(
        label="communication_services_2018",
        effective_date=date(2018, 9, 28),
        description=(
            "Telecommunication Services devient Communication Services et absorbe "
            "les médias/divertissement (Consumer Discretionary) et l'internet grand "
            "public (Information Technology)."
        ),
        # Ici le secteur d'ORIGINE diffère d'une entreprise à l'autre : les
        # opérateurs télécoms n'ont pas bougé, les médias venaient de la
        # consommation discrétionnaire, l'internet de la technologie. Une
        # règle par secteur ne suffit donc pas.
        moves={
            # Internet grand public : Information Technology -> Communication Services
            "GOOGL": "Technologie", "GOOG": "Technologie", "META": "Technologie",
            "FB": "Technologie", "TWTR": "Technologie",
            # Médias et divertissement : Consumer Discretionary -> Communication Services
            "DIS": "Distribution", "CMCSA": "Distribution", "NFLX": "Distribution",
            "CHTR": "Distribution", "FOXA": "Distribution", "FOX": "Distribution",
            "NWSA": "Distribution", "NWS": "Distribution", "OMC": "Distribution",
            "IPG": "Distribution", "DISCA": "Distribution", "DISCK": "Distribution",
            "VIAB": "Distribution", "CBS": "Distribution", "DISH": "Distribution",
            "TRIP": "Distribution", "LYV": "Distribution", "MTCH": "Distribution",
            # Jeu vidéo : Information Technology -> Communication Services
            "EA": "Technologie", "ATVI": "Technologie", "TTWO": "Technologie",
        },
    ),
    Reclassification(
        label="payments_and_staples_2023",
        effective_date=date(2023, 3, 17),
        description=(
            "Les services de paiement passent d'Information Technology à Financials ; "
            "quelques distributeurs passent de Consumer Discretionary à Consumer Staples."
        ),
        moves={
            # Paiements : Information Technology -> Financials
            "V": "Technologie", "MA": "Technologie", "PYPL": "Technologie",
            "FIS": "Technologie", "FISV": "Technologie", "FI": "Technologie",
            "GPN": "Technologie", "JKHY": "Technologie",
            # Distribution -> biens de consommation de base
            "TGT": "Distribution", "DG": "Distribution", "DLTR": "Distribution",
        },
    ),
]


def _as_date(value) -> Optional[date]:
    """Normalise en `datetime.date`. None si la valeur n'est pas datable :
    l'appelant traite alors la ligne comme non datée (aucune restriction
    point-in-time possible), jamais comme datée d'aujourd'hui."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    stamp = pd.to_datetime(value, errors="coerce")
    if stamp is pd.NaT or pd.isna(stamp):
        return None
    return stamp.date()


def sector_asof(symbol: str, current_sector: Optional[str], as_of) -> Optional[str]:
    """Secteur de `symbol` tel qu'il était classé à la date `as_of`, déduit du
    secteur ACTUEL en rejouant à l'envers les remaniements GICS postérieurs.

    Les événements sont rejoués du plus RÉCENT au plus ancien : une entreprise
    peut avoir été déplacée deux fois (rare mais possible), et remonter le
    temps demande de défaire les mouvements dans l'ordre inverse de leur
    application.

    `as_of` non datable, ou secteur actuel inconnu : le secteur actuel est
    rendu tel quel (aucune correction possible, et inventer serait pire)."""
    jour = _as_date(as_of)
    if jour is None or not isinstance(current_sector, str) or not current_sector:
        return current_sector

    secteur = current_sector
    for event in sorted(GICS_RECLASSIFICATIONS, key=lambda e: e.effective_date, reverse=True):
        if jour >= event.effective_date:
            continue  # remaniement déjà en vigueur à cette date : rien à défaire
        avant = event.moves.get(symbol)
        if avant is None:
            avant = event.sector_moves.get(secteur)
        if avant is not None:
            secteur = avant
    return secteur


def apply_sector_asof(
    df: pd.DataFrame, symbol_col: str = "symbol", sector_col: str = "sector",
    date_col: str = "filed_date", target_col: str = "sector_asof",
) -> pd.DataFrame:
    """Ajoute `target_col` : le secteur point-in-time de chaque ligne.

    Le résultat est mis en cache par (symbole, secteur, année) plutôt que
    calculé ligne à ligne : les remaniements GICS se comptent sur les doigts
    d'une main et aucun ne tombe deux fois dans la même année, donc l'année
    suffit à identifier le régime -- sauf pour l'année de l'événement, traitée
    exactement. Sans ce cache, sector_asof est rappelée des centaines de
    milliers de fois pour quelques centaines de réponses distinctes."""
    df = df.copy()
    if symbol_col not in df.columns or sector_col not in df.columns:
        df[target_col] = df.get(sector_col)
        return df

    dates = pd.to_datetime(df.get(date_col), errors="coerce")
    cache: Dict[tuple, Optional[str]] = {}
    resultats = []
    for symbol, secteur, when in zip(df[symbol_col], df[sector_col], dates):
        if pd.isna(when):
            resultats.append(secteur)
            continue
        cle = (symbol, secteur, when.year, when.month)
        if cle not in cache:
            cache[cle] = sector_asof(symbol, secteur, when)
        resultats.append(cache[cle])
    df[target_col] = resultats
    return df


def reclassified_symbols() -> Set[str]:
    """Symboles concernés par au moins un remaniement documenté. Sert aux
    diagnostics : ce sont les seuls dont le secteur historique diffère du
    secteur actuel."""
    touches: Set[str] = set()
    for event in GICS_RECLASSIFICATIONS:
        touches.update(event.moves)
    return touches


# ----------------------------------------------------------------------------
# Appartenance à l'indice (biais de survivance)
# ----------------------------------------------------------------------------

def load_universe_history(path=None) -> Optional[pd.DataFrame]:
    """Spans d'appartenance produits par 01b (ric, start_date, end_date,
    is_current_member), avec la colonne `symbol` (symbole IBKR canonique)
    ajoutée pour se joindre au reste du pipeline.

    None si 01b n'a jamais tourné : l'appelant doit alors DÉGRADER
    explicitement (et le dire), pas faire comme si l'univers actuel valait
    univers historique."""
    path = path or getattr(config, "UNIVERSE_HISTORY_FILE", None)
    if path is None or not path.exists():
        return None
    history = pd.read_parquet(path)
    if "ric" not in history.columns:
        logger.warning("%s n'a pas de colonne 'ric' : historique d'univers ignoré.", path)
        return None
    history = history.copy()
    history["symbol"] = history["ric"].map(config.to_ib_symbol)
    history["start_date"] = pd.to_datetime(history["start_date"], errors="coerce")
    history["end_date"] = pd.to_datetime(history["end_date"], errors="coerce")
    return history


def members_asof(history: pd.DataFrame, as_of) -> Set[str]:
    """Symboles membres de l'indice à `as_of`.

    Un span est actif si start_date <= as_of < end_date, avec les deux bornes
    manquantes traitées comme ouvertes (start absent = membre depuis avant le
    début du suivi Wikipedia ; end absent = toujours membre). Même convention
    que 01b::universe_asof, dont c'est la transposition sur les symboles
    IBKR."""
    if history is None or history.empty:
        return set()
    jour = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(jour):
        return set()
    debut_ok = history["start_date"].isna() | (history["start_date"] <= jour)
    fin_ok = history["end_date"].isna() | (history["end_date"] > jour)
    return set(history.loc[debut_ok & fin_ok, "symbol"].dropna())


class MembershipIndex:
    """Appartenance à l'indice interrogeable par (symbole, date), sans
    recalculer un masque sur tout l'historique à chaque appel.

    06b interroge l'appartenance une fois par PAIRE (ligne, pair candidat) :
    des dizaines de millions d'interrogations sur un univers complet. La
    version naïve (un masque pandas par appel) y passait l'essentiel du run,
    alors que les spans d'un symbole tiennent dans une petite liste de
    couples."""

    def __init__(self, history: Optional[pd.DataFrame]):
        self._spans: Dict[str, List[tuple]] = {}
        if history is None or history.empty:
            return
        for symbol, group in history.groupby("symbol"):
            self._spans[symbol] = [
                (start, end) for start, end in zip(group["start_date"], group["end_date"])
            ]

    def __bool__(self) -> bool:
        return bool(self._spans)

    @property
    def symbols(self) -> Set[str]:
        return set(self._spans)

    def is_member(self, symbol: str, as_of) -> bool:
        """Vrai si `symbol` était membre à `as_of`.

        Symbole ABSENT de l'historique : True. C'est délibéré -- 01b ne
        remonte qu'à ~1996-2000, et une entreprise présente dans les données
        mais inconnue de l'historique des changements ne doit pas être exclue
        des médianes sur la foi d'une source incomplète. Écarter par défaut
        reviendrait à remplacer un biais mesuré par une amputation
        silencieuse."""
        spans = self._spans.get(symbol)
        if spans is None:
            return True
        jour = pd.to_datetime(as_of, errors="coerce")
        if pd.isna(jour):
            return True  # ligne non datée : aucune restriction possible
        for start, end in spans:
            if (pd.isna(start) or start <= jour) and (pd.isna(end) or end > jour):
                return True
        return False


# ----------------------------------------------------------------------------
# Mesure du biais résiduel
# ----------------------------------------------------------------------------

def coverage_report(
    history: Optional[pd.DataFrame], present_symbols: Set[str], as_of,
) -> Optional[dict]:
    """Part de l'univers RÉEL de la date couverte par les données présentes.

    C'est la mesure directe du biais de survivance résiduel : `members` est
    l'univers qu'il aurait fallu, `present` ce qu'on a. Un taux de 100 %
    signifie que la médiane de cette date porte sur l'indice complet ; 62 %
    dit exactement de combien elle est amputée, et par qui.

    None si l'historique d'univers est indisponible -- on ne peut alors rien
    affirmer, et le silence vaut mieux qu'un chiffre inventé."""
    if history is None or history.empty:
        return None
    membres = members_asof(history, as_of)
    if not membres:
        return None
    presents = membres & set(present_symbols)
    return {
        "as_of": pd.to_datetime(as_of, errors="coerce"),
        "members": len(membres),
        "present": len(presents),
        "missing": len(membres) - len(presents),
        "coverage_ratio": len(presents) / len(membres),
        "missing_symbols": sorted(membres - presents),
    }
