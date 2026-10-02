"""
Recalcul de la valorisation (DCF, multiples, valeur combinée) à partir des
faits chiffrés qu'un 8-K apporte -- la partie CALCUL du traitement des 8-K.

LE PARTAGE DES RÔLES
--------------------
04d_extraction_8k.py demande à Gemini d'EXTRAIRE des faits (un chiffre
d'affaires trimestriel, un nombre d'actions émises, un prix d'acquisition...),
chacun accompagné de la citation qui le justifie, vérifiée mot pour mot dans le
texte. Gemini ne calcule rien. C'est ce module qui calcule, avec les formules du
pipeline (07_calcul_dcf.py pour le DCF, 06b_calcul_valorisation_combinee.py et
hierarchie_multiples.py pour les multiples) : un calcul fait en Python est
reproductible, testable, et ne se trompe pas d'un ordre de grandeur.

CE QUI EST RECALCULÉ, ET COMMENT
--------------------------------
Un signal repose sur une période publiée (10-K ou 10-Q) : ses fondamentaux
(multiples.parquet, sortie de 05) et sa valorisation (06b ou 07). Pour chaque
8-K CHIFFRABLE déposé après elle, les fondamentaux sont modifiés par les faits
du 8-K (appliquer_faits), puis :

  - les MULTIPLES : le multiple appliqué par 06b est retrouvé en inversant le
    prix implicite qu'il a stocké, puis réappliqué aux fondamentaux modifiés.
    Le multiple de référence lui-même ne bouge pas, et c'est exact : la médiane
    sectorielle EXCLUT l'entreprise valorisée (06b), son propre 8-K ne la
    déplace donc pas. (Le multiple MÉRITÉ, lui, dépend des fondamentaux de
    l'entreprise ; il n'est pas re-régressé ici -- voir README.)
  - le DCF : refait avec les fonctions de 07 sur le FCF modifié, et ajouté à la
    valeur stockée sous forme d'ÉCART (nouvelle valeur - valeur recalculée sur
    les fondamentaux d'origine) : sans fait, la valeur reste exactement celle
    du pipeline, au centime près.
  - la valeur COMBINÉE : même hiérarchie de multiples que 06b, repli DCF.

Un risque révélé mais non chiffrable (départ soudain du CFO, désaccord avec
l'auditeur...) n'a pas de flux à modifier : il ajoute une prime au WACC
(config.PRIME_RISQUE_8K_BPS), et la même prime réduit les multiples dans le
rapport où elle réduit le DCF.

LE SIGNAL AJUSTÉ
----------------
Chaque 8-K qui change quelque chose produit une NOUVELLE ligne de signal,
datée du dépôt du 8-K et valorisée au cours de ce jour-là. Elle remplace le
signal précédent dans le moteur (le dernier signal connu d'un symbole fait foi)
et garde la date de la période d'origine dans `age_reference_date` : elle se
périme au même rythme que le 10-K/10-Q qui la fonde, et cède la place au
prochain dépôt périodique, qui reprend les vrais chiffres. Les 8-K suivants
d'une même période se cumulent.

Le sens qu'on autorise est un réglage (config.AJUSTEMENT_8K_MODE) : en
"garder", une bonne nouvelle ne peut que maintenir le signal ; en "renforcer",
elle peut l'accroître. "veto" n'appelle pas ce module.
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import hierarchie_multiples

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Les faits qu'un 8-K peut apporter, et leur unité
# ----------------------------------------------------------------------------
# Table unique, partagée avec 04d : elle donne la liste autorisée du schéma
# Gemini ET l'unité qui sert à vérifier le nombre dans sa citation.
MUSD = "millions_usd"
MACTIONS = "millions_actions"
USD_PAR_ACTION = "usd_par_action"
ANNEES = "annees"

TYPES_FAITS: Dict[str, str] = {
    # Résultats publiés (Item 2.02) : le trimestre ET le même trimestre un an
    # plus tôt -- leur différence fait glisser le TTM d'un trimestre.
    "ca_trimestre": MUSD, "ca_trimestre_n1": MUSD,
    "ebit_trimestre": MUSD, "ebit_trimestre_n1": MUSD,
    "resultat_net_trimestre": MUSD, "resultat_net_trimestre_n1": MUSD,
    # Prévisions de la direction pour l'exercice en cours (prospectif).
    "guidance_ca_bas": MUSD, "guidance_ca_haut": MUSD,
    "guidance_bpa_bas": USD_PAR_ACTION, "guidance_bpa_haut": USD_PAR_ACTION,
    # Capital : émission réalisée, rachat RÉALISÉ (pas une autorisation).
    "actions_emises": MACTIONS, "produit_emission": MUSD,
    "actions_rachetees": MACTIONS, "montant_rachat": MUSD,
    # Opérations FINALISÉES (Item 2.01).
    "acquisition_prix_cash": MUSD, "acquisition_actions_emises": MACTIONS,
    "acquisition_ca_cible": MUSD, "acquisition_ebitda_cible": MUSD,
    "cession_prix_recu": MUSD, "cession_ca_cede": MUSD, "cession_ebitda_cede": MUSD,
    # Charges : un décaissement ponctuel (restructuration, règlement payé).
    # Une dépréciation, charge non décaissée, n'en est pas une.
    "charge_cash_ponctuelle": MUSD,
    # Prospectif, comme la guidance.
    "economies_annuelles": MUSD,
    "contrat_valeur_totale": MUSD, "contrat_duree_annees": ANNEES,
}

# Faits PROSPECTIFS : des chiffres de la direction, pas des faits réalisés.
# Ignorés en mode "sectoriel", pondérés par la prudence en mode "guidance".
FAITS_PROSPECTIFS = frozenset({
    "guidance_ca_bas", "guidance_ca_haut", "guidance_bpa_bas", "guidance_bpa_haut",
    "economies_annuelles", "contrat_valeur_totale", "contrat_duree_annees",
})

PRIMES_RISQUE = tuple(config.PRIME_RISQUE_8K_BPS)

# Période d'origine sans date de clôture (lignes FY : 04 ne la renseigne pas) :
# estimée à la date de dépôt moins ce délai. Un 10-K est déposé 30 à 90 jours
# après la clôture, donc tout trimestre clos après cette date est postérieur à
# l'exercice, et le Q4 (= la clôture) est antérieur.
DELAI_CLOTURE_DEPOT_JOURS = 30

# Au-delà de ce délai sans cotation, le cours d'un jour n'est plus connu.
MAX_JOURS_SANS_COURS = 10


@dataclass(frozen=True)
class Reglages:
    """Les trois réglages du recalcul, comparés par le backtest."""
    mode: str = "veto"
    projections: str = "sectoriel"
    prudence: float = 0.5

    def __post_init__(self):
        if self.mode not in config.AJUSTEMENT_8K_MODES:
            raise ValueError(f"Mode d'ajustement 8-K inconnu : {self.mode!r}. "
                             f"Attendu l'un de {config.AJUSTEMENT_8K_MODES}.")
        if self.projections not in config.PROJECTIONS_8K_MODES:
            raise ValueError(f"Mode de projections 8-K inconnu : {self.projections!r}. "
                             f"Attendu l'un de {config.PROJECTIONS_8K_MODES}.")
        if not 0.0 <= self.prudence <= 1.0:
            raise ValueError(f"La prudence doit être entre 0 et 1, reçu {self.prudence}.")

    @classmethod
    def depuis_config(cls) -> "Reglages":
        return cls(config.AJUSTEMENT_8K_MODE, config.PROJECTIONS_8K_MODE,
                   config.PROJECTIONS_8K_PRUDENCE)

    @property
    def actif(self) -> bool:
        return self.mode != "veto"

    @property
    def croit_la_direction(self) -> bool:
        return self.projections == "guidance" and self.prudence > 0


# ----------------------------------------------------------------------------
# Fondamentaux et application des faits
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Fondamentaux:
    """Les postes d'une période que les faits d'un 8-K peuvent modifier, en
    dollars et en actions (pas en millions)."""
    revenue: float
    ebitda: float
    ebit: float
    net_income: float
    net_debt: float
    shares: float
    da: float = 0.0
    capex: float = 0.0
    wc_change: float = 0.0
    tax_rate: float = 0.21
    period_end: Optional[pd.Timestamp] = None

    @classmethod
    def depuis_ligne(cls, ligne) -> "Fondamentaux":
        def nombre(nom, defaut=np.nan):
            valeur = ligne.get(nom) if hasattr(ligne, "get") else None
            try:
                valeur = float(valeur)
            except (TypeError, ValueError):
                return defaut
            return valeur if np.isfinite(valeur) else defaut

        taux = nombre("tax_rate", config_taux_defaut())
        if not 0 <= taux < 1:
            taux = config_taux_defaut()
        fin = pd.to_datetime(ligne.get("period_end"), errors="coerce")
        if pd.isna(fin):
            depot = pd.to_datetime(ligne.get("filed_date"), errors="coerce")
            fin = depot - pd.Timedelta(days=DELAI_CLOTURE_DEPOT_JOURS) if pd.notna(depot) else None
        return cls(
            revenue=nombre("revenue"), ebitda=nombre("ebitda"), ebit=nombre("ebit"),
            net_income=nombre("net_income"), net_debt=nombre("net_debt", 0.0),
            shares=nombre("shares_outstanding"), da=nombre("da", 0.0),
            capex=nombre("capex", 0.0), wc_change=nombre("working_capital_change", 0.0),
            tax_rate=taux, period_end=fin,
        )

    @property
    def marge_ebit(self) -> float:
        return self.ebit / self.revenue if self.revenue and self.revenue > 0 else np.nan

    @property
    def marge_ebitda(self) -> float:
        return self.ebitda / self.revenue if self.revenue and self.revenue > 0 else np.nan


def config_taux_defaut() -> float:
    return float(_module_07().HYPOTHESES_DEFAUT["taux_imposition_defaut"])


_MODULE_07 = None


def _module_07():
    """07_calcul_dcf, importé à la demande : son nom commence par un chiffre,
    et l'importer charge 05 -- inutile tant qu'aucun 8-K n'est recalculé."""
    global _MODULE_07
    if _MODULE_07 is None:
        _MODULE_07 = importlib.import_module("07_calcul_dcf")
    return _MODULE_07


def charger_faits(valeur) -> List[dict]:
    """La colonne `faits` d'extractions_8k.parquet : du JSON, une liste de
    {type, valeur, citation}. Une valeur illisible ne donne aucun fait."""
    if isinstance(valeur, list):
        faits = valeur
    elif isinstance(valeur, str) and valeur.strip():
        try:
            faits = json.loads(valeur)
        except ValueError:
            return []
    else:
        return []
    return [f for f in faits if isinstance(f, dict) and f.get("type") in TYPES_FAITS]


def faits_par_type(faits: List[dict]) -> Dict[str, float]:
    """Une valeur par type de fait -- la première citée, quand le texte en
    donne plusieurs (le tableau répète souvent le chiffre du titre)."""
    valeurs: Dict[str, float] = {}
    for fait in faits:
        try:
            valeur = float(fait.get("valeur"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(valeur):
            valeurs.setdefault(fait["type"], valeur)
    return valeurs


def _ajouter_resultat_operationnel(f: Fondamentaux, ebitda_ajoute: float) -> Fondamentaux:
    """Ajoute un EBITDA (cible acquise, activité cédée, contrat, économies) :
    l'EBIT bouge dans le rapport EBIT/EBITDA de l'entreprise (la D&A de la
    cible n'est pas publiée), le résultat net de l'EBIT après impôt."""
    rapport = f.ebit / f.ebitda if f.ebitda and f.ebitda > 0 and np.isfinite(f.ebit) else 1.0
    ebit_ajoute = ebitda_ajoute * rapport
    return replace(
        f, ebitda=f.ebitda + ebitda_ajoute, ebit=f.ebit + ebit_ajoute,
        net_income=f.net_income + ebit_ajoute * (1 - f.tax_rate),
    )


def appliquer_faits(
    f: Fondamentaux, valeurs: Dict[str, float], fin_periode_resultats=None,
    reglages: Reglages = Reglages(),
) -> Tuple[Fondamentaux, List[str]]:
    """Fondamentaux modifiés par les faits d'UN 8-K, et la liste des postes
    réellement appliqués (vide : rien de chiffrable).

    Les montants arrivent en millions (unité de l'extraction) et sont
    convertis ici. La guidance n'est PAS appliquée ici : elle ne change pas
    un niveau mais une croissance (voir croissance_guidance)."""
    M = 1e6
    postes: List[str] = []

    # RÉSULTATS PUBLIÉS : TTM nouveau = TTM d'origine + trimestre publié -
    # même trimestre un an plus tôt. Seulement si ce trimestre est postérieur
    # à la période d'origine : sinon il y est déjà, et l'ajouter le compterait
    # deux fois.
    fin = pd.to_datetime(fin_periode_resultats, errors="coerce")
    trimestre_nouveau = (
        pd.notna(fin) and f.period_end is not None and pd.notna(f.period_end) and fin > f.period_end
    )
    if trimestre_nouveau:
        for poste, attribut in (("ca", "revenue"), ("ebit", "ebit"), ("resultat_net", "net_income")):
            courant, avant = valeurs.get(f"{poste}_trimestre"), valeurs.get(f"{poste}_trimestre_n1")
            if courant is None or avant is None:
                continue
            ecart = (courant - avant) * M
            f = replace(f, **{attribut: getattr(f, attribut) + ecart})
            if attribut == "ebit":
                # D&A supposée stable d'un an sur l'autre : l'EBITDA suit l'EBIT.
                f = replace(f, ebitda=f.ebitda + ecart)
            postes.append(f"resultats_{poste}")

    # CAPITAL. Une émission apporte de la trésorerie (la dette nette baisse) ;
    # un rachat réalisé en consomme.
    emises, rachetees = valeurs.get("actions_emises"), valeurs.get("actions_rachetees")
    if emises:
        f = replace(f, shares=f.shares + emises * M,
                    net_debt=f.net_debt - (valeurs.get("produit_emission") or 0.0) * M)
        postes.append("emission")
    if rachetees:
        f = replace(f, shares=max(f.shares - rachetees * M, 0.0),
                    net_debt=f.net_debt + (valeurs.get("montant_rachat") or 0.0) * M)
        postes.append("rachat")

    # ACQUISITION FINALISÉE : la trésorerie (ou la dette) payée s'ajoute à la
    # dette nette, les actions remises au nombre d'actions, la cible au
    # périmètre. Sans CA ni EBITDA de la cible, seul le bilan bouge : c'est
    # pessimiste, et c'est voulu -- on ne devine pas ce qu'on a acheté.
    prix = valeurs.get("acquisition_prix_cash")
    actions_remises = valeurs.get("acquisition_actions_emises")
    if prix or actions_remises:
        f = replace(f, net_debt=f.net_debt + (prix or 0.0) * M,
                    shares=f.shares + (actions_remises or 0.0) * M)
        if valeurs.get("acquisition_ca_cible"):
            f = replace(f, revenue=f.revenue + valeurs["acquisition_ca_cible"] * M)
        if valeurs.get("acquisition_ebitda_cible"):
            f = _ajouter_resultat_operationnel(f, valeurs["acquisition_ebitda_cible"] * M)
        postes.append("acquisition")

    # CESSION FINALISÉE : l'inverse.
    if valeurs.get("cession_prix_recu"):
        f = replace(f, net_debt=f.net_debt - valeurs["cession_prix_recu"] * M)
        if valeurs.get("cession_ca_cede"):
            f = replace(f, revenue=f.revenue - valeurs["cession_ca_cede"] * M)
        if valeurs.get("cession_ebitda_cede"):
            f = _ajouter_resultat_operationnel(f, -valeurs["cession_ebitda_cede"] * M)
        postes.append("cession")

    # DÉCAISSEMENT PONCTUEL : il sort de la trésorerie, une fois. Il ne touche
    # pas le résultat récurrent -- un multiple ou un DCF appliqué à un
    # résultat amputé d'une charge ponctuelle sous-évaluerait l'entreprise.
    if valeurs.get("charge_cash_ponctuelle"):
        f = replace(f, net_debt=f.net_debt + abs(valeurs["charge_cash_ponctuelle"]) * M)
        postes.append("charge_ponctuelle")

    # PROSPECTIF : seulement si l'on croit la direction, et pondéré.
    if reglages.croit_la_direction:
        if valeurs.get("economies_annuelles"):
            f = _ajouter_resultat_operationnel(
                f, reglages.prudence * valeurs["economies_annuelles"] * M)
            postes.append("economies")
        valeur_contrat, duree = valeurs.get("contrat_valeur_totale"), valeurs.get("contrat_duree_annees")
        if valeur_contrat and duree and duree > 0 and np.isfinite(f.marge_ebitda):
            ca_annuel = reglages.prudence * valeur_contrat / duree * M
            f = _ajouter_resultat_operationnel(replace(f, revenue=f.revenue + ca_annuel),
                                               ca_annuel * f.marge_ebitda)
            postes.append("contrat")

    return f, postes


def croissance_guidance(valeurs: Dict[str, float], f: Fondamentaux) -> Optional[float]:
    """Croissance que la guidance implique par rapport aux fondamentaux TTM :
    milieu de fourchette du CA prévu, à défaut du BPA, rapporté au réalisé.
    None sans guidance exploitable, ou hors de CROISSANCE_GUIDANCE_BORNES
    (plus probablement une erreur d'unité ou de période qu'une prévision)."""
    def milieu(bas, haut):
        valeurs_presentes = [v for v in (valeurs.get(bas), valeurs.get(haut)) if v is not None]
        return float(np.mean(valeurs_presentes)) if valeurs_presentes else None

    croissance = None
    ca_prevu = milieu("guidance_ca_bas", "guidance_ca_haut")
    if ca_prevu is not None and f.revenue and f.revenue > 0:
        croissance = ca_prevu * 1e6 / f.revenue - 1
    else:
        bpa_prevu = milieu("guidance_bpa_bas", "guidance_bpa_haut")
        if bpa_prevu is not None and f.shares and f.shares > 0 and f.net_income and f.net_income > 0:
            croissance = bpa_prevu / (f.net_income / f.shares) - 1
    if croissance is None or not np.isfinite(croissance):
        return None
    bas, haut = config.CROISSANCE_GUIDANCE_BORNES
    return croissance if bas <= croissance <= haut else None


# ----------------------------------------------------------------------------
# DCF et multiples
# ----------------------------------------------------------------------------

def trajectoire_croissance(g_secteur: float, g_annee1: Optional[float], periode: int) -> np.ndarray:
    """Croissance de chaque année de prévision : g_annee1 en année 1, puis
    retour linéaire vers le taux sectoriel en dernière année. Sans guidance,
    le taux sectoriel partout -- exactement le DCF de 07."""
    if g_annee1 is None or periode < 2:
        return np.full(periode, g_secteur)
    poids = np.arange(periode) / (periode - 1)
    return g_annee1 + (g_secteur - g_annee1) * poids


def valeur_entreprise_dcf(fcf: float, croissances: np.ndarray, g_terminal: float, wacc: float) -> Optional[float]:
    """Valeur d'entreprise d'un FCF sur une trajectoire de croissance, mêmes
    formules que 07 (calculer_dcf), qu'elle reproduit à l'identique quand la
    croissance est constante."""
    if wacc <= g_terminal:
        return None
    fcf_futurs = fcf * np.cumprod(1 + croissances)
    actualisation = (1 + wacc) ** np.arange(1, len(croissances) + 1)
    terminale = _module_07().calculer_terminal_value(fcf_futurs[-1], g_terminal, wacc)
    return float(np.sum(fcf_futurs / actualisation) + terminale / actualisation[-1])


def hypotheses_dcf(secteur, annee) -> dict:
    m07 = _module_07()
    annee = int(annee) if annee is not None and pd.notna(annee) else None
    return m07.hypotheses_pour_secteur(secteur, m07.HYPOTHESES_DEFAUT, year=annee)


def valeur_dcf_par_action(
    f: Fondamentaux, hyp: dict, g_annee1: Optional[float] = None, prime_bps: float = 0.0,
) -> Optional[float]:
    """DCF par action de 07 sur ces fondamentaux. None dans les cas où 07
    n'en calcule pas (EBIT ou FCF négatif, actions inconnues)."""
    if not (np.isfinite(f.ebit) and f.ebit > 0 and np.isfinite(f.shares) and f.shares > 0):
        return None
    fcf = _module_07().calculer_fcf(f.ebit, f.tax_rate, f.capex, f.wc_change, f.da)
    if not np.isfinite(fcf) or fcf <= 0:
        return None
    croissances = trajectoire_croissance(hyp["taux_croissance_fcf"], g_annee1, hyp["periode_prevision"])
    ev = valeur_entreprise_dcf(fcf, croissances, hyp["taux_croissance_terminal"],
                               hyp["taux_actualisation"] + prime_bps / 1e4)
    if ev is None:
        return None
    return (ev - f.net_debt) / f.shares


def facteur_risque(hyp: dict, prime_bps: float) -> float:
    """Rapport des valeurs d'entreprise DCF avec et sans la prime de risque,
    pour un même flux : la décote que la prime impose, appliquée aux
    multiples pour qu'un même risque pèse pareil sur les deux méthodes."""
    if not prime_bps:
        return 1.0
    croissances = trajectoire_croissance(hyp["taux_croissance_fcf"], None, hyp["periode_prevision"])
    sans = valeur_entreprise_dcf(1.0, croissances, hyp["taux_croissance_terminal"], hyp["taux_actualisation"])
    avec = valeur_entreprise_dcf(1.0, croissances, hyp["taux_croissance_terminal"],
                                 hyp["taux_actualisation"] + prime_bps / 1e4)
    if not sans or avec is None:
        return 1.0
    return avec / sans


# Prix implicite stocké par 06b -> grandeur à laquelle le multiple s'applique.
MULTIPLES_EV = {"EV/EBITDA": ("price_from_ev_ebitda", "ebitda"),
                "EV/Sales": ("price_from_ev_sales", "revenue")}
MULTIPLE_PE = ("P/E", "price_from_pe")


def prix_implicites(
    ligne, base: Fondamentaux, nouveau: Fondamentaux, facteur: float = 1.0, croissance_relative: float = 0.0,
) -> Dict[str, float]:
    """Prix implicite de chaque multiple sur les fondamentaux modifiés.

    Le multiple appliqué par 06b n'est pas stocké, mais il se retrouve
    EXACTEMENT en inversant le prix implicite qui l'est :
        prix = (M x grandeur - dette nette) / actions
        =>  M = (prix x actions + dette nette) / grandeur
    -- quelle que soit la méthode qui l'a produit (médiane, moyenne
    harmonique, multiple mérité).

    `facteur` : décote de risque (facteur_risque). `croissance_relative` :
    surcroît de croissance sur un an prévu par la direction par rapport au
    secteur, appliqué aux grandeurs (mode "guidance") -- les pairs sont
    valorisés sur leur passé, leur croissance attendue est dans le multiple ;
    seul l'écart à cette attente est propre à l'entreprise."""
    ajustement = 1.0 + croissance_relative
    resultat: Dict[str, float] = {}
    for multiple, (colonne, grandeur) in MULTIPLES_EV.items():
        prix_base = _nombre(ligne.get(colonne))
        avant, apres = getattr(base, grandeur), getattr(nouveau, grandeur)
        if not (np.isfinite(prix_base) and avant > 0 and base.shares > 0):
            resultat[multiple] = np.nan
            continue
        m = (prix_base * base.shares + base.net_debt) / avant
        if not (apres > 0 and nouveau.shares > 0):
            resultat[multiple] = np.nan
            continue
        resultat[multiple] = (m * apres * ajustement * facteur - nouveau.net_debt) / nouveau.shares

    multiple, colonne = MULTIPLE_PE
    prix_base = _nombre(ligne.get(colonne))
    if np.isfinite(prix_base) and base.net_income > 0 and base.shares > 0 and \
            nouveau.net_income > 0 and nouveau.shares > 0:
        pe = prix_base / (base.net_income / base.shares)
        resultat[multiple] = pe * facteur * ajustement * nouveau.net_income / nouveau.shares
    else:
        resultat[multiple] = np.nan
    return resultat


def _nombre(valeur) -> float:
    try:
        valeur = float(valeur)
    except (TypeError, ValueError):
        return np.nan
    return valeur if np.isfinite(valeur) else np.nan


def _table_hierarchie(hierarchie):
    if hierarchie is not None:
        return hierarchie
    return None if config.MULTIPLE_COMBINATION == "flat" else config.MULTIPLE_RELIABILITY_TIERS


# ----------------------------------------------------------------------------
# Lignes de signal ajustées
# ----------------------------------------------------------------------------

def _cle_periode(df: pd.DataFrame) -> pd.Series:
    """symbol|period_type|année|trimestre -- l'année est `year` (= fiscal_year
    dans 05, 06b et 07), le trimestre vide pour une ligne annuelle."""
    def norm(col):
        if col not in df.columns:
            return pd.Series([""] * len(df), index=df.index)
        serie = df[col].astype("object")
        if col == "year":
            serie = pd.to_numeric(serie, errors="coerce").astype("Int64").astype("object")
        return serie.where(pd.notna(serie), "").astype(str)
    return norm("symbol") + "|" + norm("period_type") + "|" + norm("year") + "|" + norm("fiscal_quarter")


def preparer_fondamentaux(multiples: pd.DataFrame) -> Dict[str, Fondamentaux]:
    """Fondamentaux par clé de période, à partir de multiples.parquet (05).
    La variation de BFR est calculée comme dans 07 (build_input_table) :
    différence avec la période précédente CHRONOLOGIQUEMENT, par entreprise."""
    df = multiples.copy()
    df["filed_date"] = pd.to_datetime(df.get("filed_date"), errors="coerce")
    df = df.sort_values(["symbol", "filed_date"])
    if "working_capital" in df.columns:
        df["working_capital_change"] = df.groupby("symbol")["working_capital"].diff()
    cles = _cle_periode(df)
    return {cle: Fondamentaux.depuis_ligne(ligne)
            for cle, ligne in zip(cles, df.to_dict("records"))}


class CoursQuotidiens:
    """Cours de clôture connu à une date (le dernier au plus tard ce jour-là)."""

    def __init__(self, cours: Optional[pd.DataFrame]):
        self._par_symbole: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if cours is None or cours.empty:
            return
        cours = cours.dropna(subset=["close"])
        for symbole, groupe in cours.groupby("symbol"):
            groupe = groupe.sort_values("date")
            self._par_symbole[symbole] = (
                pd.to_datetime(groupe["date"]).to_numpy(dtype="datetime64[ns]"),
                groupe["close"].to_numpy(dtype=float),
            )

    def a(self, symbole: str, date) -> Optional[float]:
        donnees = self._par_symbole.get(symbole)
        if donnees is None:
            return None
        dates, closes = donnees
        jour = np.datetime64(pd.Timestamp(date).normalize().to_datetime64())
        position = int(np.searchsorted(dates, jour, side="right")) - 1
        if position < 0 or (jour - dates[position]) > np.timedelta64(MAX_JOURS_SANS_COURS, "D"):
            return None
        return float(closes[position])


@dataclass
class _Etat:
    """Ce qu'une période d'origine a accumulé de 8-K en 8-K."""
    cle_base: str
    fondamentaux: Fondamentaux
    valeur_precedente: float
    prime_bps: float = 0.0
    g_guidance: Optional[float] = None


def ajuster_historique(
    historique: pd.DataFrame,
    fondamentaux: Dict[str, Fondamentaux],
    extractions: pd.DataFrame,
    cours: CoursQuotidiens,
    reglages: Reglages,
    source: str = "combinee",
    hierarchie=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lignes de signal ajustées par les 8-K, au schéma de `historique`, et
    les 8-K de veto (symbol, filed_date).

    `historique` : valorisation_combinee_historique (source "combinee") ou
    dcf_historique (source "dcf"). Rend deux tables vides si le mode est
    "veto" ou s'il n'y a rien à ajuster."""
    colonnes_veto = ["symbol", "filed_date", "accession_number"]
    vide = historique.iloc[0:0].copy()
    if not reglages.actif or historique.empty or extractions is None or extractions.empty:
        return vide, pd.DataFrame(columns=colonnes_veto)

    colonne_valeur = "valuation_theoretical_per_share" if source == "combinee" else "valuation_dcf_per_share"
    hist = historique.copy()
    hist["filed_date"] = pd.to_datetime(hist["filed_date"], errors="coerce")
    hist["_cle"] = _cle_periode(hist)
    hist = hist.dropna(subset=["filed_date"]).sort_values(["symbol", "filed_date"])
    bases_par_symbole = {s: g for s, g in hist.groupby("symbol")}

    ext = extractions.copy()
    ext["filed_date"] = pd.to_datetime(ext["filed_date"], errors="coerce")
    ext = ext.dropna(subset=["filed_date"]).sort_values(["symbol", "filed_date", "accession_number"])
    if "veto" in ext.columns:
        vetos = ext[ext["veto"].fillna(False).astype(bool)][colonnes_veto]
        ext = ext[~ext["veto"].fillna(False).astype(bool)]
    else:
        vetos = pd.DataFrame(columns=colonnes_veto)

    table = _table_hierarchie(hierarchie)
    lignes: List[dict] = []
    ignores = {"sans_signal_avant": 0, "sans_fondamentaux": 0, "sans_cours": 0, "sans_effet": 0}

    for symbole, groupe in ext.groupby("symbol"):
        bases = bases_par_symbole.get(symbole)
        if bases is None:
            ignores["sans_signal_avant"] += len(groupe)
            continue
        dates_bases = bases["filed_date"].to_numpy(dtype="datetime64[ns]")
        etat: Optional[_Etat] = None
        for e in groupe.to_dict("records"):
            # La période d'origine : le dernier dépôt périodique STRICTEMENT
            # antérieur au 8-K. Un 10-Q déposé le même jour que son communiqué
            # de résultats contient déjà ces chiffres.
            position = int(np.searchsorted(dates_bases, np.datetime64(e["filed_date"].to_datetime64()),
                                           side="left")) - 1
            if position < 0:
                ignores["sans_signal_avant"] += 1
                continue
            if position + 1 < len(dates_bases) and \
                    dates_bases[position + 1] == np.datetime64(e["filed_date"].to_datetime64()):
                # Un 10-Q/10-K déposé le même jour porte des chiffres plus
                # complets : c'est lui le signal du jour, pas un ajustement de
                # la période précédente.
                ignores["depot_periodique_le_meme_jour"] = ignores.get("depot_periodique_le_meme_jour", 0) + 1
                continue
            base = bases.iloc[position]
            f_base = fondamentaux.get(base["_cle"])
            if f_base is None:
                ignores["sans_fondamentaux"] += 1
                continue
            if etat is None or etat.cle_base != base["_cle"]:
                etat = _Etat(base["_cle"], f_base, _nombre(base.get(colonne_valeur)))

            valeurs = faits_par_type(charger_faits(e.get("faits")))
            f_nouveau, postes = appliquer_faits(etat.fondamentaux, valeurs,
                                                e.get("fin_periode_resultats"), reglages)
            prime = float(config.PRIME_RISQUE_8K_BPS.get(e.get("prime_risque") or "aucune", 0))
            g = croissance_guidance(valeurs, f_nouveau) if reglages.croit_la_direction else None
            if not postes and prime <= etat.prime_bps and g is None:
                ignores["sans_effet"] += 1
                continue
            etat.fondamentaux = f_nouveau
            etat.prime_bps = max(etat.prime_bps, prime)
            if g is not None:
                etat.g_guidance = g

            close = cours.a(symbole, e["filed_date"])
            if close is None or close <= 0:
                ignores["sans_cours"] += 1
                continue

            ligne = _ligne_ajustee(base, f_base, etat, e, close, reglages, source, colonne_valeur, table)
            if ligne is None:
                ignores["sans_effet"] += 1
                continue
            ligne["postes_8k"] = ",".join(postes + (["prime_risque"] if prime else [])
                                          + (["guidance"] if g is not None else []))
            etat.valeur_precedente = ligne[colonne_valeur]
            lignes.append(ligne)

    if any(ignores.values()):
        logger.info("Recalcul 8-K : %d ligne(s) de signal ajustée(s) ; 8-K sans effet : %s.",
                    len(lignes), ", ".join(f"{k} {v}" for k, v in ignores.items() if v))
    if not lignes:
        return vide, vetos.reset_index(drop=True)
    ajustees = pd.DataFrame(lignes)
    colonnes = [c for c in historique.columns] + [
        c for c in ("age_reference_date", "ajustement_8k", "postes_8k", "valeur_8k_brute")
        if c not in historique.columns]
    return ajustees.reindex(columns=colonnes), vetos.reset_index(drop=True)


def _ligne_ajustee(base, f_base: Fondamentaux, etat: _Etat, e: dict, close: float,
                   reglages: Reglages, source: str, colonne_valeur: str, table) -> Optional[dict]:
    hyp = hypotheses_dcf(base.get("sector"), base["filed_date"].year)
    g_secteur = hyp["taux_croissance_fcf"]
    g_annee1 = None
    if etat.g_guidance is not None:
        g_annee1 = g_secteur + reglages.prudence * (etat.g_guidance - g_secteur)

    # DCF : écart entre le DCF modifié et le DCF d'origine, ajouté à la valeur
    # stockée -- sans fait, la valeur du pipeline reste intacte.
    dcf_stocke = _nombre(base.get("valuation_dcf_per_share"))
    dcf = np.nan
    if np.isfinite(dcf_stocke):
        avant = valeur_dcf_par_action(f_base, hyp)
        apres = valeur_dcf_par_action(etat.fondamentaux, hyp, g_annee1, etat.prime_bps)
        if avant is not None and apres is not None:
            dcf = dcf_stocke + (apres - avant)

    ligne = dict(base)
    ligne.pop("_cle", None)
    ligne["valuation_dcf_per_share"] = dcf

    if source == "combinee":
        facteur = facteur_risque(hyp, etat.prime_bps)
        implicites = prix_implicites(base, f_base, etat.fondamentaux, facteur,
                                     (g_annee1 - g_secteur) if g_annee1 is not None else 0.0)
        implied = pd.DataFrame([implicites])
        multiples = float(hierarchie_multiples.combiner(implied, table).iloc[0])
        ligne["price_from_ev_ebitda"] = implicites["EV/EBITDA"]
        ligne["price_from_ev_sales"] = implicites["EV/Sales"]
        ligne["price_from_pe"] = implicites["P/E"]
        ligne["n_multiples_used"] = int(implied.notna().sum(axis=1).iloc[0])
        ligne["valuation_multiples_per_share"] = multiples
        if np.isfinite(multiples):
            valeur, ligne["source"] = multiples, "multiples"
        elif np.isfinite(dcf):
            valeur, ligne["source"] = dcf, "dcf_fallback"
        else:
            return None
    else:
        if not np.isfinite(dcf):
            return None
        valeur = dcf

    ligne["valeur_8k_brute"] = valeur
    # "garder" : une bonne nouvelle ne fait pas croître la conviction. Le sens
    # de la thèse se lit au cours du jour du 8-K, sur la valeur d'avant.
    precedente = etat.valeur_precedente
    if reglages.mode == "garder" and np.isfinite(precedente):
        valeur = min(valeur, precedente) if precedente >= close else max(valeur, precedente)

    ligne[colonne_valeur] = valeur
    if source == "combinee":
        ligne["valuation_theoretical_per_share"] = valeur
    ligne["close"] = close
    ligne["gap_pct"] = (valeur - close) / close * 100
    ligne["age_reference_date"] = base["filed_date"]
    ligne["filed_date"] = e["filed_date"]
    ligne["ajustement_8k"] = e.get("accession_number")
    return ligne


# ----------------------------------------------------------------------------
# Chargement (pour backtest.data_loader)
# ----------------------------------------------------------------------------

def charger_extractions(path=None) -> Optional[pd.DataFrame]:
    path = path or config.EXTRACTIONS_8K_FILE
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df if not df.empty else None


def lignes_ajustees(historique: pd.DataFrame, reglages: Reglages, source: str,
                    hierarchie=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Point d'entrée du data_loader : charge extractions, fondamentaux (05)
    et cours quotidiens (03b), puis ajuster_historique. Deux tables vides --
    et le comportement historique -- si l'un d'eux manque, en le disant."""
    vides = historique.iloc[0:0].copy(), pd.DataFrame(columns=["symbol", "filed_date", "accession_number"])
    if not reglages.actif:
        return vides
    extractions = charger_extractions()
    if extractions is None:
        logger.warning(
            "Ajustement 8-K '%s' demandé, mais %s est absent ou vide : lance 04d_extraction_8k.py. "
            "Repli sur le veto des 8-K matériels.", reglages.mode, config.EXTRACTIONS_8K_FILE)
        return vides
    for chemin, script in ((config.MULTIPLES_FILE, "05_calcul_multiples.py"),
                           (config.DAILY_PRICES_FILE, "03b_recuperation_cours_quotidiens.py")):
        if not chemin.exists():
            logger.warning("Ajustement 8-K : %s introuvable (lance %s). Repli sur le veto.", chemin, script)
            return vides
    fondamentaux = preparer_fondamentaux(pd.read_parquet(config.MULTIPLES_FILE))
    cours_df = pd.read_parquet(config.DAILY_PRICES_FILE, columns=["symbol", "date", "close"])
    cours_df = cours_df[cours_df["symbol"].isin(set(extractions["symbol"]))]
    lignes, vetos = ajuster_historique(historique, fondamentaux, extractions, CoursQuotidiens(cours_df),
                                       reglages, source=source, hierarchie=hierarchie)
    logger.info("Ajustement 8-K '%s' (projections '%s', prudence %.2f) : %d signal(aux) recalculé(s), "
                "%d 8-K de veto.", reglages.mode, reglages.projections, reglages.prudence,
                len(lignes), len(vetos))
    return lignes, vetos
