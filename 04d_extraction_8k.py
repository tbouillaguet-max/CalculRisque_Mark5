"""
Extrait par Gemini les FAITS CHIFFRÉS des 8-K déposés sur les signaux actifs
et les positions ouvertes, pour que recalcul_8k.py recalcule DCF et multiples
avec ces nouvelles données (au lieu de seulement périmer le signal).

OÙ IL SE PLACE
--------------
04c_recuperation_8k.py liste et classe TOUS les 8-K (matériel ou non, sept
catégories). Ce script en reprend une petite partie -- ceux qui touchent un
signal vivant -- et y lit des CHIFFRES : résultats du trimestre, guidance,
actions émises, prix d'une acquisition finalisée, charges décaissées... Il
écrit extractions_8k.parquet ; backtest/data_loader.py le lit à travers
recalcul_8k.py quand le recalcul est activé (config.AJUSTEMENT_8K_MODE ou
--ajustement-8k sur 09, 10 et 17). Il doit tourner APRÈS 07 et 06b (il lit
leurs signaux pour savoir lesquels sont actifs).

GEMINI EXTRAIT, PYTHON CALCULE
------------------------------
La consigne (CONSIGNE_TEMPLATE) interdit au modèle tout calcul : il recopie des
nombres écrits dans le texte, dans une unité imposée, avec la CITATION exacte
où chacun figure. Chaque fait est ensuite vérifié ici (verifier_faits) :
  - la citation doit se retrouver mot pour mot dans le texte envoyé ;
  - le nombre doit figurer dans la citation, à une échelle près (unité,
    millier, million, milliard -- les tableaux de résultats annoncent leur
    unité en tête, loin de la cellule).
Un fait qui échoue est REJETÉ (gardé à part, pour audit), jamais corrigé. Un
chiffre inventé ne peut donc pas entrer dans une valorisation.

Le modèle juge aussi, sans chiffre : le sens probable de l'événement, une
catégorie de risque non chiffrable (le montant de la prime de risque est
décidé dans config.PRIME_RISQUE_8K_BPS, pas par le modèle), et un veto quand
la survie de l'entreprise ou la fiabilité de ses comptes est en cause.

PÉRIMÈTRE ET COÛT
-----------------
Une extraction coûte bien plus qu'un classement de 04c : le communiqué de
résultats entier, et une réponse détaillée. Elle est donc réservée aux 8-K :
  - déposés depuis --depuis (défaut : config.LLM_8K_FENETRE_JOURS jours, la
    plus longue vie d'un signal) ;
  - dont un Item peut porter des chiffres (config.ITEMS_EXTRACTION_8K) ;
  - d'un symbole qui, à la date du 8-K, avait un signal ACTIF -- non périmé,
    écart au moins égal à config.EXTRACTION_8K_GAP_MIN_PCT -- ou une position
    ouverte dans le compte paper (dernier_run.json de 17_paper_trading.py).
Les 8-K partent par lots de --par-requete déposés le même jour (aucun 8-K plus
récent n'éclaire un plus ancien). Chaque extraction est mémorisée
(cache_extractions_8k.jsonl) : un 8-K est figé, il n'est jamais relu, sauf
changement de VERSION_EXTRACTION ou --refaire.

Contrainte anti-anticipation : comme 04c, chaque 8-K est jugé sur son seul
texte, la consigne interdisant d'utiliser ce que le modèle saurait d'autre.

Usage :
    python 04d_extraction_8k.py --dry-run          # liste les 8-K concernés, n'appelle rien
    python 04d_extraction_8k.py
    python 04d_extraction_8k.py --ticker AAPL
    python 04d_extraction_8k.py --depuis 2024-01-01   # plus d'historique, pour le backtest (coûteux)
    python 04d_extraction_8k.py --refaire          # relit des 8-K déjà extraits
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

import config
import ecriture_atomique
import recalcul_8k
import reprise_jsonl
import sec_filings_text as sft
from backtest import data_loader

logger = logging.getLogger("extraction_8k")

_m04c = importlib.import_module("04c_recuperation_8k")

MEMOIRE_FILENAME = "cache_extractions_8k.jsonl"

# Version de la consigne et des règles de vérification. Une extraction faite
# sous une autre version n'est plus servie par la mémoire : le 8-K est relu.
VERSION_EXTRACTION = 1

# Lot de 8-K par requête : deux, pas dix comme 04c -- chaque document porte un
# communiqué entier, et la réponse détaille chaque fait.
DOCUMENTS_PAR_REQUETE = 2
# Texte du 8-K lui-même (du premier Item aux signatures) et du communiqué
# joint. Celui-ci porte les tableaux de résultats, qui viennent après les
# faits marquants : 20 000 caractères couvrent le compte de résultat d'un
# communiqué ordinaire.
MAX_CARACTERES_8K = 8_000
MAX_CARACTERES_COMMUNIQUE = 20_000
# Budget de RÉPONSE par 8-K : une quinzaine de faits et leurs citations.
JETONS_PAR_DOCUMENT = 1_500
# Longueur maximale d'une citation demandée au modèle ; la vérification
# refuse aussi une citation trop courte pour identifier un passage.
LONGUEUR_CITATION_MAX = 300
LONGUEUR_CITATION_MIN = 8
# Écart relatif toléré entre la valeur extraite et le nombre cité (arrondis
# d'unité : « $1.2 billion » pour 1 213 M$ ne passe pas, et ne doit pas).
TOLERANCE_RELATIVE = 0.005

DIRECTIONS = ("positive", "negative", "neutre")

_DEFINITIONS_FAITS = """\
- ca_trimestre, ca_trimestre_n1 : chiffre d'affaires (revenue, net sales) du trimestre publié, et du même trimestre de l'année précédente.
- ebit_trimestre, ebit_trimestre_n1 : résultat opérationnel GAAP (operating income) du trimestre publié, et du même trimestre de l'année précédente.
- resultat_net_trimestre, resultat_net_trimestre_n1 : résultat net GAAP attribuable aux actionnaires du trimestre publié, et du même trimestre de l'année précédente.
- guidance_ca_bas, guidance_ca_haut : prévision de chiffre d'affaires de la direction pour l'exercice en cours (une prévision unique : la même valeur dans les deux).
- guidance_bpa_bas, guidance_bpa_haut : prévision de BPA dilué GAAP pour l'exercice en cours. Ignore les BPA ajustés ou non-GAAP.
- actions_emises, produit_emission : actions nouvelles émises lors d'une émission réalisée (offre au public, placement privé, programme ATM exécuté), et son produit brut. Pas les plans de rémunération.
- actions_rachetees, montant_rachat : actions effectivement rachetées et montant payé (rachat accéléré livré, par exemple). Une simple autorisation de rachat n'en est pas un.
- acquisition_prix_cash, acquisition_actions_emises, acquisition_ca_cible, acquisition_ebitda_cible : pour une acquisition FINALISÉE seulement, la part payée en numéraire ou en dette, les actions remises aux vendeurs, le chiffre d'affaires annuel et l'EBITDA annuel de la cible. Une acquisition seulement signée ne s'extrait pas.
- cession_prix_recu, cession_ca_cede, cession_ebitda_cede : pour une cession FINALISÉE seulement, le prix reçu, le chiffre d'affaires et l'EBITDA annuels de l'activité cédée.
- charge_cash_ponctuelle : coût décaissé une seule fois (restructuration, coûts de transaction, amende ou règlement payé). Jamais une dépréciation ni une charge non décaissée.
- economies_annuelles : économies de coûts annuelles récurrentes visées par un plan annoncé.
- contrat_valeur_totale, contrat_duree_annees : valeur totale et durée d'un NOUVEAU contrat (pas d'un renouvellement)."""

CONSIGNE_TEMPLATE = """Tu es un analyste financier spécialiste des 8-K. Tu reçois des 8-K déposés le {filed_date}, chacun entre des balises <document id="...">, précédé de l'entreprise et des Items déclarés, et suivi, quand il en a un, du communiqué de presse joint (Exhibit 99). Juge chaque document séparément, uniquement à partir de son propre texte : ignore les autres documents du lot et tout ce que tu pourrais savoir par ailleurs, en particulier après cette date.

1. faits : extrais les faits chiffrés de la liste ci-dessous que le texte énonce explicitement, et seulement ceux-là. Pour chacun, donne son type, sa valeur et une citation : le passage du texte où figure le nombre, recopié mot pour mot, au plus {longueur} caractères. Ne calcule rien, n'estime rien, ne déduis rien : si un nombre n'est pas écrit dans le texte, n'extrais pas le fait. Seule l'unité se convertit. Une citation qui ne figure pas mot pour mot dans le texte fait rejeter le fait.
Unités : montants en millions de dollars US ; nombres d'actions en millions ; BPA en dollars par action ; durée en années. Une perte est une valeur négative.
{definitions}

2. fin_periode_resultats : date de clôture du trimestre dont le document publie les résultats, au format AAAA-MM-JJ ; chaîne vide s'il n'en publie pas.

3. direction : effet probable de l'événement sur la valeur intrinsèque de l'action -- positive, negative ou neutre (routine, ou effet indéterminé).

4. prime_risque : risque supplémentaire révélé par le document et qu'aucun chiffre ne mesure.
- aucune : le cas général, y compris un résultat décevant mais chiffré.
- faible : litige ou enquête mineurs ; départ annoncé d'un dirigeant avec successeur nommé.
- moyenne : départ soudain du CEO ou du CFO sans successeur ; avenant ou renonciation à un covenant bancaire ; enquête réglementaire significative ; changement d'auditeur sans désaccord.
- forte : désaccord avec l'auditeur ; enquête pour fraude ; défaut de paiement évité de justesse.

5. veto : vrai seulement si le document met en cause la survie de l'entreprise ou la fiabilité de ses comptes -- doute sur la continuité d'exploitation, faillite, états financiers publiés qui ne sont plus fiables, exigibilité anticipée d'une dette, radiation de la cote.

6. resume : une phrase courte, en français."""

_SCHEMA_FAIT = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": list(recalcul_8k.TYPES_FAITS)},
        "valeur": {"type": "number"},
        "citation": {"type": "string"},
    },
    "required": ["type", "valeur", "citation"],
    "propertyOrdering": ["type", "valeur", "citation"],
}

# Le FORMAT de la réponse pour UN 8-K. Les faits d'abord : le modèle juge
# sens, risque et veto APRÈS avoir relevé les chiffres, pas avant.
SCHEMA_EXTRACTION = {
    "type": "object",
    "properties": {
        "faits": {"type": "array", "items": _SCHEMA_FAIT},
        "fin_periode_resultats": {"type": "string"},
        "direction": {"type": "string", "enum": list(DIRECTIONS)},
        "prime_risque": {"type": "string", "enum": list(recalcul_8k.PRIMES_RISQUE)},
        "veto": {"type": "boolean"},
        "resume": {"type": "string"},
    },
    "required": ["faits", "fin_periode_resultats", "direction", "prime_risque", "veto", "resume"],
    "propertyOrdering": ["faits", "fin_periode_resultats", "direction", "prime_risque", "veto", "resume"],
}


def build_consigne(filed_date: str) -> str:
    return CONSIGNE_TEMPLATE.format(filed_date=filed_date, longueur=LONGUEUR_CITATION_MAX,
                                    definitions=_DEFINITIONS_FAITS)


# ----------------------------------------------------------------------------
# Vérification des faits
# ----------------------------------------------------------------------------

_EQUIVALENTS = str.maketrans({
    "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-",
    "−": "-", " ": " ", " ": " ", " ": " ",
})


def normaliser(texte: str) -> str:
    """Forme de comparaison d'une citation : minuscules, guillemets et tirets
    unifiés, espaces resserrés -- les différences qu'un recopiage introduit
    sans changer le texte."""
    return " ".join((texte or "").translate(_EQUIVALENTS).lower().split())


_NOMBRE = re.compile(r"\d[\d,]*(?:\.\d+)?|\.\d+")
_NOMBRES_EN_LETTRES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20,
}
# Échelles possibles entre le nombre écrit et la valeur extraite : le nombre
# est en dollars, en milliers, en millions ou en milliards.
_ECHELLES = {
    recalcul_8k.MUSD: (1e-6, 1e-3, 1.0, 1e3),
    recalcul_8k.MACTIONS: (1e-6, 1e-3, 1.0, 1e3),
    recalcul_8k.USD_PAR_ACTION: (1.0,),
    recalcul_8k.ANNEES: (1.0,),
}


def nombres_cites(citation: str, unite: str) -> List[float]:
    nombres = []
    for brut in _NOMBRE.findall(citation):
        try:
            nombres.append(float(brut.replace(",", "")))
        except ValueError:
            continue
    if unite == recalcul_8k.ANNEES:
        mots = re.findall(r"[a-z]+", citation.lower())
        nombres.extend(_NOMBRES_EN_LETTRES[m] for m in mots if m in _NOMBRES_EN_LETTRES)
    return nombres


def nombre_dans_citation(valeur: float, unite: str, citation: str) -> bool:
    """La valeur extraite figure-t-elle dans la citation, à une échelle près ?
    Le signe n'est pas comparé : une perte s'écrit souvent entre parenthèses."""
    cible = abs(valeur)
    for nombre in nombres_cites(citation, unite):
        for echelle in _ECHELLES.get(unite, (1.0,)):
            candidat = nombre * echelle
            if abs(candidat - cible) <= max(TOLERANCE_RELATIVE * cible, 1e-9):
                return True
    return False


def verifier_faits(faits: Iterable[dict], texte_normalise: str) -> Tuple[List[dict], List[dict]]:
    """(faits vérifiés, faits rejetés avec leur raison) -- voir l'en-tête."""
    verifies, rejetes = [], []
    for fait in faits:
        type_fait = fait.get("type")
        citation = str(fait.get("citation") or "")
        unite = recalcul_8k.TYPES_FAITS.get(type_fait)
        try:
            valeur = float(fait.get("valeur"))
        except (TypeError, ValueError):
            valeur = None
        if unite is None:
            raison = "type inconnu"
        elif valeur is None or valeur != valeur:
            raison = "valeur non numérique"
        elif len(normaliser(citation)) < LONGUEUR_CITATION_MIN:
            raison = "citation trop courte"
        elif normaliser(citation) not in texte_normalise:
            raison = "citation introuvable dans le texte"
        elif not nombre_dans_citation(valeur, unite, citation):
            raison = "nombre absent de la citation"
        else:
            verifies.append({"type": type_fait, "valeur": valeur, "citation": citation})
            continue
        rejetes.append({**fait, "raison": raison})
    return verifies, rejetes


# ----------------------------------------------------------------------------
# Périmètre : les 8-K des signaux actifs et des positions ouvertes
# ----------------------------------------------------------------------------

def numeros_items(item_codes) -> Set[str]:
    """Numéros d'Items ("2.02") d'une colonne item_codes ("Item 2.02", liste
    ou tableau numpy selon le chemin de lecture)."""
    if item_codes is None:
        return set()
    if isinstance(item_codes, str):
        item_codes = [item_codes]
    return {_m04c._numero_item(str(code)) for code in item_codes} - {""}


def charger_signaux() -> pd.DataFrame:
    """Les signaux BRUTS des deux sources (06b et 07), sans ajustement 8-K :
    symbol, filed_date, gap_pct, period_type."""
    morceaux = []
    for chemin in (config.VALORISATION_COMBINEE_FILE, config.DCF_HISTORY_FILE):
        if chemin.exists():
            df = pd.read_parquet(chemin, columns=None)
            colonnes = [c for c in ("symbol", "filed_date", "gap_pct", "period_type") if c in df.columns]
            morceaux.append(df[colonnes])
    if not morceaux:
        return pd.DataFrame(columns=["symbol", "filed_date", "gap_pct", "period_type"])
    signaux = pd.concat(morceaux, ignore_index=True)
    signaux["filed_date"] = pd.to_datetime(signaux["filed_date"], errors="coerce")
    return signaux.dropna(subset=["symbol", "filed_date", "gap_pct"])


def positions_ouvertes(chemin: Optional[Path] = None) -> Set[str]:
    """Symboles détenus d'après le dernier run du compte paper (cibles de
    poids non nul). Ensemble vide sans journal : le paper trading est
    facultatif."""
    chemin = chemin or (config.DIR_PAPER_TRADING / "dernier_run.json")
    if not chemin.exists():
        return set()
    try:
        cibles = json.loads(chemin.read_text(encoding="utf-8")).get("cibles") or {}
    except (OSError, ValueError, AttributeError):
        logger.warning("%s illisible : positions ouvertes ignorées.", chemin)
        return set()
    return {s for s, ligne in cibles.items() if isinstance(ligne, dict) and (ligne.get("poids") or 0) > 0}


def huit_k_eligibles(
    evenements: pd.DataFrame, signaux: pd.DataFrame, positions: Set[str],
    depuis: Optional[pd.Timestamp], seuil_gap_pct: float = config.EXTRACTION_8K_GAP_MIN_PCT,
    items: Iterable[str] = config.ITEMS_EXTRACTION_8K,
) -> pd.DataFrame:
    """Les 8-K à extraire (voir l'en-tête). `evenements` : sortie de 04c.

    Le signal regardé est le DERNIER publié strictement avant le 8-K, toutes
    sources confondues : actif s'il n'est pas périmé à la date du 8-K et que
    son écart atteint le seuil."""
    if evenements is None or evenements.empty:
        return pd.DataFrame(columns=list(evenements.columns) if evenements is not None else [])
    ev = evenements.copy()
    ev["filed_date"] = pd.to_datetime(ev["filed_date"], errors="coerce")
    ev = ev.dropna(subset=["symbol", "filed_date", "accession_number"])
    if depuis is not None:
        ev = ev[ev["filed_date"] >= depuis]
    voulus = set(items)
    ev = ev[ev["item_codes"].map(lambda codes: bool(numeros_items(codes) & voulus))]
    if ev.empty:
        return ev

    actif = pd.Series(False, index=ev.index)
    if not signaux.empty:
        sig = signaux.sort_values("filed_date").rename(columns={"filed_date": "date_signal"})
        gauche = ev[["symbol", "filed_date"]].reset_index().sort_values("filed_date")
        rapproche = pd.merge_asof(
            gauche, sig, left_on="filed_date", right_on="date_signal", by="symbol",
            direction="backward", allow_exact_matches=False,
        ).set_index("index")
        age = (rapproche["filed_date"] - rapproche["date_signal"]).dt.days
        age_max = rapproche.apply(
            lambda r: data_loader.signal_max_age_for(r.to_dict(), config.BACKTEST_SIGNAL_MAX_AGE_DAYS), axis=1)
        # Un écart absurde n'est pas un signal mais une erreur de valorisation :
        # le moteur l'écarte (config.BACKTEST_MAX_PLAUSIBLE_GAP_PCT), ses 8-K
        # ne méritent pas d'extraction.
        plafond = getattr(config, "BACKTEST_MAX_PLAUSIBLE_GAP_PCT", None) or float("inf")
        ecart = rapproche["gap_pct"].abs()
        vivant = rapproche["date_signal"].notna() & (age <= age_max) & \
            (ecart >= seuil_gap_pct) & (ecart <= plafond)
        actif = vivant.reindex(ev.index, fill_value=False)
    actif |= ev["symbol"].isin(positions)
    return ev[actif].sort_values(["filed_date", "symbol"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Texte envoyé au modèle
# ----------------------------------------------------------------------------

def texte_pour_extraction(texte_8k: str, communique: Optional[str]) -> str:
    """Le 8-K du premier Item aux signatures, puis le communiqué joint --
    avec des bornes plus larges que 04c : les chiffres sont dans les
    tableaux, en fin de communiqué."""
    debut = _m04c.ITEM_CODE_PATTERN.search(texte_8k)
    corps = texte_8k[debut.start():] if debut else texte_8k
    fin = _m04c._SIGNATURES.search(corps)
    if fin and fin.start() > 0:
        corps = corps[:fin.start()]
    texte = " ".join(corps.split())[:MAX_CARACTERES_8K]
    joint = " ".join((communique or "").split())[:MAX_CARACTERES_COMMUNIQUE]
    return f"{texte}\n[Communiqué joint (Exhibit 99)]\n{joint}" if joint else texte


@dataclass
class AExtraire:
    evenement: dict
    texte: str   # texte_pour_extraction, aussi la référence des citations


class Telechargeur:
    """Texte d'un 8-K à partir de son numéro d'accession -- une seule requête
    submissions par entreprise et par run."""

    def __init__(self):
        self._depots: Dict[str, Dict[str, dict]] = {}

    def texte(self, cik: str, accession: str) -> Optional[str]:
        if cik not in self._depots:
            self._depots[cik] = {d["accession_number"]: d for d in sft.fetch_submissions_strict(cik)}
        depot = self._depots[cik].get(accession)
        if not depot or not depot.get("primary_document"):
            return None
        url = sft.filing_document_url(cik, accession, depot["primary_document"])
        extrait = sft.fetch_filing_text(url, form="8-K")
        if extrait is None:
            return None
        communique = sft.texte_piece_jointe(cik, accession, MAX_CARACTERES_COMMUNIQUE)
        return texte_pour_extraction(extrait[0], communique)


# ----------------------------------------------------------------------------
# Mémoire
# ----------------------------------------------------------------------------

def memoire_path(output_dir: Path) -> Path:
    return output_dir / MEMOIRE_FILENAME


def cle(symbol: str, accession: str) -> str:
    return f"{symbol}:{accession}"


def charger_memoire(output_dir: Path) -> Dict[str, dict]:
    """Extractions déjà faites, de la version courante, par symbole:accession
    (la dernière écrite l'emporte). Une ligne illisible est ignorée."""
    lignes, illisibles = reprise_jsonl.lire_lignes(memoire_path(output_dir))
    if illisibles:
        logger.warning("%s : %d ligne(s) illisible(s) ignorée(s).", memoire_path(output_dir), illisibles)
    return {cle(e["symbol"], e["accession_number"]): e for e in lignes
            if e.get("symbol") and e.get("accession_number")
            and e.get("version_extraction") == VERSION_EXTRACTION}


def memoriser(output_dir: Path, entree: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with memoire_path(output_dir).open("a", encoding="utf-8") as fichier:
        fichier.write(json.dumps(entree, default=str, ensure_ascii=False) + "\n")


def entree_depuis_reponse(evenement: dict, reponse: dict, texte: str) -> dict:
    """Une ligne d'extractions_8k.parquet, faits vérifiés contre `texte`."""
    verifies, rejetes = verifier_faits(reponse.get("faits") or [], normaliser(texte))
    fin = str(reponse.get("fin_periode_resultats") or "").strip()
    fin = fin if pd.notna(pd.to_datetime(fin, errors="coerce", format="%Y-%m-%d")) else ""
    items = evenement.get("item_codes")
    return {
        "symbol": evenement["symbol"], "cik": evenement.get("cik"),
        "accession_number": evenement["accession_number"],
        "filed_date": str(pd.Timestamp(evenement["filed_date"]).date()),
        "item_codes": list(items) if items is not None and not isinstance(items, str) else items,
        "direction": reponse.get("direction"), "veto": bool(reponse.get("veto")),
        "prime_risque": reponse.get("prime_risque") or "aucune",
        "fin_periode_resultats": fin,
        "faits": json.dumps(verifies, ensure_ascii=False),
        "faits_rejetes": json.dumps(rejetes, ensure_ascii=False),
        "chiffrable": bool(verifies), "resume": reponse.get("resume"),
        "modele": sft.dernier_modele_utilise(), "version_extraction": VERSION_EXTRACTION,
        "extraction_timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def extraire(a_extraire: List[AExtraire], output_dir: Path,
             par_requete: int = DOCUMENTS_PAR_REQUETE) -> List[dict]:
    """Soumet les 8-K par lots de même date de dépôt, les plus récents
    d'abord (si le quota s'épuise, ce sont les plus anciens qui attendent).
    Chaque extraction est mémorisée aussitôt : un Ctrl-C ne perd pas une
    requête payée."""
    par_date: Dict[str, List[AExtraire]] = {}
    for element in a_extraire:
        par_date.setdefault(str(pd.Timestamp(element.evenement["filed_date"]).date()), []).append(element)
    entrees = []
    for date in sorted(par_date, reverse=True):
        groupe = par_date[date]
        for debut in range(0, len(groupe), par_requete):
            lot = groupe[debut:debut + par_requete]
            documents = {f"d{i + 1}": _document(element) for i, element in enumerate(lot)}
            reponses = sft.analyser_documents(documents, build_consigne(date), SCHEMA_EXTRACTION,
                                              JETONS_PAR_DOCUMENT)
            if not reponses:
                if sft.llm_coupe_pour_ce_run():
                    logger.warning("Gemini coupé pour ce run : les 8-K restants attendent le prochain.")
                    return entrees
                continue
            for i, element in enumerate(lot):
                reponse = reponses.get(f"d{i + 1}")
                if reponse is None:
                    continue
                entree = entree_depuis_reponse(element.evenement, reponse, element.texte)
                memoriser(output_dir, entree)
                entrees.append(entree)
    return entrees


def _document(element: AExtraire) -> str:
    items = ", ".join(sorted(numeros_items(element.evenement.get("item_codes"))))
    return f"Entreprise : {element.evenement['symbol']}. Items déclarés : {items or 'aucun'}.\n{element.texte}"


def ecrire_sortie(memoire: Dict[str, dict], chemin: Path) -> pd.DataFrame:
    """extractions_8k.parquet : toutes les extractions de la version courante."""
    df = pd.DataFrame(list(memoire.values()))
    if df.empty:
        df = pd.DataFrame(columns=["symbol", "cik", "accession_number", "filed_date", "item_codes",
                                   "direction", "veto", "prime_risque", "fin_periode_resultats", "faits",
                                   "faits_rejetes", "chiffrable", "resume", "modele",
                                   "version_extraction", "extraction_timestamp"])
    df = df.sort_values(["filed_date", "symbol"]).reset_index(drop=True) if len(df) else df
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tmp = chemin.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, engine="pyarrow")
    ecriture_atomique.remplacer(tmp, chemin)
    return df


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", default=None, help="Un seul symbole.")
    parser.add_argument("--depuis", default=None,
                        help="Date de dépôt minimale (AAAA-MM-JJ). Défaut : il y a "
                             f"{config.LLM_8K_FENETRE_JOURS} jours (config.LLM_8K_FENETRE_JOURS).")
    parser.add_argument("--limit", type=int, default=None, help="Au plus N 8-K soumis au modèle.")
    parser.add_argument("--par-requete", type=int, default=DOCUMENTS_PAR_REQUETE,
                        help="8-K du même jour par requête (défaut: %(default)s).")
    parser.add_argument("--seuil-gap-pct", type=float, default=config.EXTRACTION_8K_GAP_MIN_PCT,
                        help="Écart minimal d'un signal pour qu'il soit actif (défaut: %(default)s).")
    parser.add_argument("--refaire", action="store_true",
                        help="Relit aussi les 8-K déjà extraits (chaque appel est payant).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Liste les 8-K concernés sans rien télécharger ni appeler.")
    parser.add_argument("--output-dir", type=Path, default=config.DIR_FINANCIALS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.par_requete < 1:
        parser.error("--par-requete doit valoir au moins 1.")

    if not config.MATERIAL_EVENTS_8K_FILE.exists():
        logger.error("%s introuvable : lance d'abord 04c_recuperation_8k.py.", config.MATERIAL_EVENTS_8K_FILE)
        sys.exit(1)
    evenements = pd.read_parquet(config.MATERIAL_EVENTS_8K_FILE)
    if args.ticker:
        evenements = evenements[evenements["symbol"] == args.ticker]
    depuis = pd.Timestamp(args.depuis) if args.depuis else \
        pd.Timestamp(datetime.now().date() - timedelta(days=config.LLM_8K_FENETRE_JOURS))
    positions = positions_ouvertes()
    eligibles = huit_k_eligibles(evenements, charger_signaux(), positions, depuis, args.seuil_gap_pct)

    memoire = {} if args.refaire else charger_memoire(args.output_dir)
    a_lire = eligibles[[cle(s, a) not in memoire
                        for s, a in zip(eligibles["symbol"], eligibles["accession_number"])]] \
        if len(eligibles) else eligibles
    if args.limit is not None:
        a_lire = a_lire.tail(args.limit)   # les plus récents
    logger.info("8-K des signaux actifs et positions ouvertes (%d positions) depuis le %s : %d, "
                "dont %d déjà extraits, %d à lire.", len(positions), depuis.date(), len(eligibles),
                len(eligibles) - len(a_lire), len(a_lire))

    if args.dry_run:
        for e in a_lire.to_dict("records"):
            logger.info("  %s  %-6s  %s  %s", pd.Timestamp(e["filed_date"]).date(), e["symbol"],
                        e["accession_number"], ", ".join(sorted(numeros_items(e.get("item_codes")))))
        return

    if len(a_lire) and not sft.llm_disponible():
        logger.warning("Aucune clé Gemini : rien n'est extrait. %s", sft.aide_cle_absente())
        a_lire = a_lire.iloc[0:0]
    if len(a_lire):
        logger.info("Modèle : %s", sft.description_llm())

    telechargeur = Telechargeur()
    a_extraire: List[AExtraire] = []
    for e in a_lire.to_dict("records"):
        try:
            texte = telechargeur.texte(str(e["cik"]).zfill(10), e["accession_number"])
        except Exception as exc:  # noqa: BLE001 -- un 8-K illisible n'arrête pas le run
            logger.warning("%s %s : téléchargement impossible (%s).", e["symbol"], e["accession_number"], exc)
            continue
        if texte:
            a_extraire.append(AExtraire(e, texte))
    nouvelles = extraire(a_extraire, args.output_dir, args.par_requete) if a_extraire else []

    memoire = charger_memoire(args.output_dir)
    sortie = ecrire_sortie(memoire, config.EXTRACTIONS_8K_FILE)
    chiffrables = int(sortie["chiffrable"].fillna(False).astype(bool).sum()) if len(sortie) else 0
    logger.info("%d 8-K extrait(s) ce run ; %s : %d extraction(s), dont %d chiffrable(s).",
                len(nouvelles), config.EXTRACTIONS_8K_FILE, len(sortie), chiffrables)
    if a_extraire:
        logger.info("%s", sft.bilan_llm())


if __name__ == "__main__":
    main()
