"""
Détecte les événements MATÉRIELS annoncés par 8-K entre deux trimestres TTM
déjà connus (04b_recuperation_10q.py) : rachats d'actions, changement de
guidance, départ de dirigeant, procédure judiciaire, M&A -- des événements
qui pourraient invalider une thèse de valorisation AVANT le prochain
recalcul trimestriel (07_calcul_dcf.py).

Pour chaque entreprise, la fenêtre de recherche est calculée à partir des
filed_date déjà connues dans FINANCIALS_TTM_FILE (04b) : entre chaque paire
de trimestres consécutifs, plus une fenêtre "ouverte" du dernier trimestre
connu jusqu'à aujourd'hui (pour détecter les 8-K récents pas encore suivis
d'un nouveau TTM). Les 8-K eux-mêmes sont listés et téléchargés via
sec_filings_text.py (réutilisé, pas dupliqué -- même module que
07b_validation_qualitative.py).

Contrainte anti-anticipation
------------------------------
Chaque 8-K est un document déjà intrinsèquement point-in-time (il ne peut
par construction parler que d'événements connus à SA date de dépôt) : la
classification par Gemini ne porte QUE sur le texte de CE 8-K, jamais sur un
résumé agrégé ou une connaissance d'événements postérieurs -- même garantie
structurelle que 07b_validation_qualitative.py (sec_filings_text.py::
fetch_filing_text ne télécharge qu'UN document à la fois).

Classification par Gemini, au moindre coût en requêtes et en jetons
------------------------------------------------------------------
Au palier gratuit de Gemini, c'est le nombre de REQUÊTES par jour qui borne
un run. Chaque 8-K est donc d'abord classé par règles (classify_8k_par_regles),
et seuls vont ensuite à Gemini :
  - les 8-K RÉCENTS (config.LLM_8K_FENETRE_JOURS) : un 8-K plus ancien ne
    touche plus aucun signal actif ;
  - dont les Items ne décident pas seuls (ITEMS_POUR_LE_MODELE : 1.01, 1.02,
    2.02, 5.02, 7.01, 8.01) -- 6 352 des 6 968 8-K récents classés le
    2026-09-27. Chacun part avec le début de son communiqué joint (Exhibit 99,
    sec_filings_text.texte_piece_jointe) : c'est lui qui porte les résultats,
    les prévisions, l'opération annoncée.
Ils partent PAR LOTS (classer_en_attente) : jusqu'à DOCUMENTS_PAR_REQUETE 8-K
par requête, tous déposés le même jour -- aucun 8-K plus récent n'éclaire le
classement d'un plus ancien. Chaque lot réunit le texte des 8-K, allégé de la
page de garde et des signatures (texte_pour_le_modele), une consigne
(CONSIGNE_TEMPLATE) et le format de la réponse (SCHEMA_VERDICT, un schéma
JSON, un verdict par 8-K) : Gemini ne peut répondre, pour chaque 8-K, qu'une
des sept catégories, un booléen pour la matérialité et une phrase de résumé,
et la réponse est revérifiée avant d'être gardée. Sans réponse pour un lot,
ses 8-K gardent leur verdict par règles, que le modèle reprendra au run
suivant.

Pré-classification par regex (codes "Item X.XX", boilerplate standardisé du
formulaire 8-K -- ex: Item 5.02 = départ/nomination de dirigeant, Item 8.01 =
autres événements, Item 1.01 = accord matériel) : gratuite et fiable, gardée
en plus du verdict LLM (pas à sa place) pour un recoupement rapide côté
rapport, sans dépendre uniquement de la classification sémantique du modèle.

Mémoire des classifications (cache_8k.jsonl)
--------------------------------------------
Un 8-K est un document FIGÉ : son texte ne changera plus, donc sa
classification non plus. Chaque 8-K classé -- par Gemini, ou par règles à
défaut -- est mémorisé (clé : symbole + numéro d'accession) dans un cache
JSONL persistant, et n'est jamais ré-analysé -- ni son texte re-téléchargé
auprès de la SEC. Un verdict par règles d'un 8-K récent que Gemini lirait
repart au modèle dès qu'une clé est disponible (voir load_llm_cache).
--vider-memoire efface cette mémoire pour tout reprendre à zéro.

Ce cache est indépendant de --resume : --resume reprend un run interrompu (au
grain du ticker), le cache survit à TOUS les runs (au grain du document). Un
run complet relancé après une interruption ne repaie donc pas les milliers
d'appels déjà effectués. --no-llm-cache force la ré-analyse.

Prérequis :
    pip install requests beautifulsoup4
    GEMINI_API_KEY=ta_cle dans .env (modèle : .env.example). Sans clé, chaque
    8-K est téléchargé et classé PAR RÈGLES à partir de son texte (cf.
    classify_8k_par_regles) ; le modèle reprend les récents quand une clé
    arrive. Facultatifs, dans .env aussi : GEMINI_MODEL (modèle),
    GEMINI_THINKING_LEVEL (réflexion) et GEMINI_REQUESTS_PER_SECOND (débit),
    voir sec_filings_text.

Usage :
    python 04c_recuperation_8k.py
    python 04c_recuperation_8k.py --limit 10
    python 04c_recuperation_8k.py --resume
    python 04c_recuperation_8k.py --ticker AAPL
    python 04c_recuperation_8k.py --no-llm-cache
    python 04c_recuperation_8k.py --vider-memoire     # tout reclasser, 8-K retéléchargés
    python 04c_recuperation_8k.py --par-requete 5
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import config
import ecriture_atomique
import reprise_jsonl
import sec_filings_text as sft

logger = logging.getLogger("recuperation_8k")

CHECKPOINT_EVERY = 10
ITEM_CODE_PATTERN = re.compile(r"Item\s+\d+\.\d+", re.IGNORECASE)

# Cache persistant des 8-K DÉJÀ classifiés (voir le docstring). JSONL
# append-only : écrit ligne par ligne au fil du run, donc utilisable même
# après un Ctrl-C ou une coupure -- un JSON réécrit en bloc en fin de run
# perdrait tout le travail d'un run interrompu, exactement le cas qu'il s'agit
# d'éviter.
LLM_CACHE_FILENAME = "cache_8k.jsonl"
# Son nom du temps de Mistral, l'ancien fournisseur : renommé au premier
# chargement (voir migrer_ancien_cache).
ANCIEN_LLM_CACHE_FILENAME = "cache_8k_mistral.jsonl"
# Classifications qui ne valent PAS mémorisation : ce sont des non-réponses
# (quota épuisé, clé absente, format illisible), pas des verdicts.
NON_CACHEABLE_CATEGORIES = frozenset({None, "", "non_evalue"})

# Qui a rendu un verdict (colonne `classification_source`).
SOURCE_GEMINI = "gemini"
SOURCE_REGLES = "regles_document"
# Seules ces sources sont servies par la mémoire. Une entrée sans source, ou
# d'une autre source, date d'avant Gemini -- de Mistral : elle est écartée
# (effacée de la mémoire au chargement) et le 8-K reclassé comme un neuf, par
# Gemini s'il est récent, par règles sinon. Constaté dans la mémoire du dépôt :
# 1 647 verdicts de Mistral, 77 % jugés matériels, dont une catégorie inventée
# (« aut_materiel ») ; les garder aurait laissé dix entreprises jugées
# autrement que le reste de l'univers.
SOURCES_RECONNUES = frozenset({SOURCE_GEMINI, SOURCE_REGLES})

# Au-delà de cette part d'entreprises en échec RÉSEAU, le run échoue
# bruyamment sans rien écrire. Un material_events_8k.parquet incomplet est
# pire qu'absent : load_material_events_8k rend None sur un fichier vide (le
# filtre reste alors sans effet, ce qui se voit), mais un fichier À MOITIÉ
# rempli passe pour complet et laisse entrer des signaux que des 8-K matériels
# auraient dû périmer.
DEFAULT_MAX_FAILURE_RATIO = 0.10

CATEGORIES = (
    "rachat_actions", "changement_guidance", "depart_dirigeant",
    "procedure_judiciaire", "fusion_acquisition", "autre_materiel", "non_materiel",
)

# Items que Gemini lit : ceux dont SEUL le texte dit la portée -- un accord
# (1.01, 1.02), des résultats (2.02), un départ ou une nomination (5.02), une
# communication (7.01), un « autre événement » (8.01). Tout autre 8-K est
# classé par règles, sans requête : les Items administratifs (9.01, 5.07...) ne
# sont jamais matériels, ceux de config.MATERIAL_8K_ITEM_CATEGORIES le sont par
# définition. Les résultats et la Regulation FD renvoient au communiqué joint
# (Exhibit 99) : 04c le télécharge pour chaque 8-K que Gemini lit
# (sec_filings_text.texte_piece_jointe) -- avant, Gemini n'aurait vu que la page
# de couverture, et ces 8-K lui étaient épargnés. Mesuré sur les 8-K classés le
# 2026-09-27, 400 derniers jours : 6 352 8-K lus sur 6 968, contre 3 697 sans
# 2.02 ni 7.01.
ITEMS_POUR_LE_MODELE = frozenset({"1.01", "1.02", "2.02", "5.02", "7.01", "8.01"})

# 8-K par requête, tous déposés le MÊME jour (voir classer_en_attente). Sur les
# mêmes 8-K, les 6 352 à lire tombent sur 271 jours de dépôt : 757 requêtes, au
# lieu de 6 352.
DOCUMENTS_PAR_REQUETE = 10

# Texte transmis à Gemini par 8-K, page de garde et signatures retirées (voir
# texte_pour_le_modele) : l'objet d'un 8-K tient en quelques paragraphes.
MAX_CARACTERES_MODELE = 6_000
# Et de son communiqué joint : le titre, les faits marquants et, le plus
# souvent, les prévisions viennent en tête ; les tableaux financiers suivent.
MAX_CARACTERES_PIECE_JOINTE = 6_000

# Budget de RÉPONSE par 8-K : une catégorie, un booléen, une phrase.
JETONS_PAR_VERDICT = 120

# La CONSIGNE donnée à Gemini (instruction système), une fois par lot. Les
# 8-K partent à côté, et le format de la réponse dans SCHEMA_VERDICT : la
# consigne ne répète ni les champs ni un exemple de JSON -- Google le
# déconseille, la qualité baisse.
CONSIGNE_TEMPLATE = """Tu es un analyste financier. Tu reçois des 8-K déposés le {filed_date}, chacun entre des balises <document id="...">, précédé de l'entreprise et des Items déclarés, et suivi, quand il en a un, du communiqué de presse joint (Exhibit 99) : c'est souvent lui qui porte l'information -- résultats, prévisions, opération annoncée. Juge chaque document séparément, uniquement à partir de son propre texte : ignore les autres documents du lot et tout ce que tu pourrais savoir par ailleurs, en particulier après cette date.

Pour chacun, dis si l'événement annoncé est matériel pour une thèse de valorisation, c'est-à-dire susceptible de changer significativement la valeur intrinsèque ou le risque perçu de l'entreprise ; classe-le dans la catégorie qui le décrit le mieux (non_materiel pour un dépôt de routine) ; résume-le en une phrase courte, en français."""

# Le FORMAT de la réponse pour UN 8-K ; sec_filings_text.analyser_documents en
# exige un par document du lot. Gemini génère sous sa contrainte : une
# catégorie de la liste, un vrai booléen, une phrase -- et rien d'autre. Sans
# description ni ordre imposé : répétés pour chaque 8-K du lot, ils coûteraient
# des jetons à chaque requête, et la consigne dit déjà ce que chaque champ
# attend.
SCHEMA_VERDICT = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "materiality": {"type": "boolean"},
        "summary": {"type": "string"},
    },
    "required": ["category", "materiality", "summary"],
}


def build_consigne(filed_date: str) -> str:
    return CONSIGNE_TEMPLATE.format(filed_date=filed_date)


def extract_item_codes(text: str) -> List[str]:
    """Codes "Item X.XX" détectés (recherche insensible à la casse, mais
    normalisés en "Item X.XX" avant dédoublonnage -- sinon "Item 5.02" et
    "item 5.02" compteraient comme deux codes distincts)."""
    matches = ITEM_CODE_PATTERN.findall(text)
    normalized = {re.sub(r"^item", "Item", m, flags=re.IGNORECASE) for m in matches}
    return sorted(normalized)


def compute_search_windows(ttm: pd.DataFrame, symbol: str, today: datetime) -> List[tuple]:
    """Fenêtres (start_date, end_date) entre trimestres TTM consécutifs
    déjà connus pour ce symbole, plus une fenêtre ouverte du dernier
    trimestre connu jusqu'à aujourd'hui. Vide si aucun trimestre TTM connu
    pour ce symbole (04b jamais lancé pour lui)."""
    dates = sorted(pd.to_datetime(ttm.loc[ttm["symbol"] == symbol, "filed_date"]).dt.strftime("%Y-%m-%d").unique())
    if not dates:
        return []
    windows = [(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]
    windows.append((dates[-1], today.strftime("%Y-%m-%d")))
    return windows


# ----------------------------------------------------------------------------
# Classification SANS modèle, à partir du texte du document
# ----------------------------------------------------------------------------
# POURQUOI. Sans clé d'API, `classify_8k` renvoyait `non_evalue` et jetait le
# texte qu'il venait de télécharger. Mesuré sur l'archive du dépôt : 99 147
# dépôts, `category` à `non_evalue` sur 100% des lignes, `materiality` et
# `summary` vides partout. Le filtre d'événements matériels -- l'une des deux
# protections anti-value-trap du moteur -- ne s'appliquait donc à RIEN, en
# silence à un avertissement près. Le coûteux (télécharger le document) était
# déjà payé ; seul le jugement manquait.
#
# CE QUE LA RÈGLE LIT, ET POURQUOI C'EST LE DOCUMENT. Deux niveaux, tous deux
# extraits du texte :
#   1. les CODES D'ITEM que le déposant déclare lui-même en tête du 8-K
#      (`extract_item_codes` les parse depuis le document). La SEC les
#      normalise : la matérialité d'un « Item 4.02 » -- non-fiabilité d'états
#      financiers déjà publiés -- tient à la définition du code, pas à une
#      lecture ;
#   2. des FORMULATIONS caractéristiques, cherchées dans le corps du document,
#      qui tranchent là où le code seul est ambigu. C'est le cas décisif de
#      l'Item 5.02, qui couvre aussi bien le départ d'un directeur général que
#      l'élection routinière d'un administrateur.
#
# Déterministe et auditable, donc reproductible d'un run à l'autre -- ce que la
# classification par modèle n'était pas. Elle reste prioritaire quand une clé
# est disponible : la règle est un repli, pas un remplacement.

# Tables lues depuis config : la relecture d'archive (backtest.data_loader) se
# sert des mêmes, et deux copies auraient fini par diverger en silence.
# `_ITEMS_MATERIELS` : matériels par définition SEC, le texte n'ajoute rien.
# `_ITEMS_AMBIGUS` : le texte décide -- un Item 1.01 peut annoncer une fusion
# comme un contrat de fourniture, un Item 5.02 un départ de dirigeant comme une
# élection d'administrateur.
_ITEMS_MATERIELS = dict(config.MATERIAL_8K_ITEM_CATEGORIES)
_ITEMS_AMBIGUS = frozenset(config.AMBIGUOUS_8K_ITEM_CODES)
# Catégories qui ne donnent pas l'alerte (un rachat d'actions est une bonne
# nouvelle) : notées pour mémoire, jamais matérielles. Voir config.
_SANS_ALERTE = frozenset(config.CATEGORIES_8K_SANS_ALERTE)

# Version des règles, écrite dans chaque verdict qu'elles rendent. Un verdict
# d'une autre version est écarté de la mémoire au chargement (load_llm_cache) :
# son 8-K est retéléchargé et reclassé au run suivant.
#   1 (verdicts sans ce champ) : mots-clés cherchés dans tout le document.
#   2 : phrases de formulaire ignorées, départs limités au directeur général et
#       au directeur financier, rachats d'actions sans alerte, Item 2.03
#       retiré des matériels d'office, résumé = la phrase qui a décidé.
VERSION_REGLES = 2

# POURQUOI LA VERSION 2. La version 1 cherchait ses mots-clés n'importe où dans
# le document, et un seul suffisait. Confrontée à la réaction du cours autour
# de chaque dépôt (99 787 8-K classés le 2026-09-27, rendement anormal de la
# veille au lendemain, SPY retiré), 9 863 de ses 22 153 verdicts « matériels »
# venaient de six pièges, aussi inertes qu'un 8-K de routine :
#   - « securities litigation » pris dans la mention légale « Private
#     Securities Litigation Reform Act of 1995 » (1 238) ;
#   - « bankruptcy » pris dans les clauses de défaut d'un contrat de dette,
#     « events of bankruptcy or insolvency » (1 011) ;
#   - l'Item 2.03, une émission d'obligations, matériel d'office (3 380) ;
#   - « tender offer » pris dans un rachat d'OBLIGATIONS, ou dans l'étiquette
#     XBRL « PreCommencement Tender Offer false » (471) ;
#   - « départ de dirigeant » sur des nominations, des rémunérations, un
#     vice-président (2 249) ;
#   - un rachat d'actions -- une bonne nouvelle -- compté comme alerte (1 514).
# Réaction > 5 % : 6 à 12 % de ces dépôts, contre 5,4 % pour la routine ; dérive
# à 60 séances positive. Le filtre bloquait donc des entreprises pour rien.
# Les motifs qui portaient un vrai signal sont gardés : faillite (« chapter
# 11 » : 26,5 % de réactions > 5 %, dérive médiane -4,1 %), guidance abaissée
# (35,8 %, -3,7 %), dépréciation (25,5 %), accord de fusion (13 %).

# Phrases de FORMULAIRE, présentes dans des milliers de 8-K sans rien annoncer :
# avertissement sur les déclarations prospectives, clauses de défaut d'un
# contrat de dette, étiquettes XBRL et cases à cocher de la page de garde. Une
# phrase qui en contient une est ignorée en entier -- c'est dans ces phrases
# que se trouvaient les mots-clés des faux positifs, pas dans l'annonce.
_PHRASE_DE_FORMULAIRE = re.compile("|".join((
    r"forward[- ]looking", r"litigation\s+reform\s+act", r"safe\s+harbor",
    r"actual\s+results\s+(?:to|could|may|might|will)\s+differ",
    r"events?\s+of\s+default", r"\bdefaults?\b",
    r"co-registrant", r"pre-?commencement", r"rule\s+14d-2", r"rule\s+13e-4",
    r"rule\s+425\b", r"rule\s+14a-12",
)), re.IGNORECASE)

# Fin de phrase : un point, un point d'exclamation ou d'interrogation suivi
# d'un blanc. Les abréviations (« Inc. ») coupent des phrases en deux, ce qui
# ne gêne pas : une moitié de phrase de formulaire en porte encore le marqueur.
_FIN_DE_PHRASE = re.compile(r"(?<=[.!?])\s+")

# Le titre de l'Item 5.02, « Departure of Directors or Certain Officers »,
# annonce la RUBRIQUE, pas un départ : sans ce retrait, une simple nomination
# de directeur financier se lisait comme un départ.
_TITRE_ITEM_502 = re.compile(r"departure\s+of\s+directors\s+or\s+(?:certain|principal)\s+officers", re.IGNORECASE)

# Départ de dirigeant : directeur général ou directeur financier SEULEMENT (un
# vice-président, un administrateur ne changent pas une thèse de valorisation),
# et un vrai départ -- pas un mot de contrat de travail.
_DIRIGEANT = (r"(?:chief\s+executive\s+officer|chief\s+financial\s+officer|\bCEO\b|\bCFO\b"
              r"|principal\s+(?:executive|financial)\s+officer)")
_DEPART = (r"(?:\bresign\w*|\bstep(?:s|ped|ping)?\s+down|\bretir(?:e|es|ed|ing)\b"
           r"|\bretirement\s+(?:as|from)\b|\bwill\s+leave\b|\bdepart(?:s|ed|ing|ure)?\b"
           r"|\bterminated\s+(?:the\s+)?(?:his\s+|her\s+)?employment)")

# Le texte d'une phrase qui annonce un départ ou une opération, là où la même
# formulation apparaît aussi dans un autre contexte :
#   - rémunération : « if the Chief Executive Officer retires, his options
#     vest », « upon his resignation for good reason » -- une clause, pas un
#     départ ;
#   - dette : « cash tender offers for its senior notes » -- un rachat
#     d'obligations, pas une offre sur les actions.
_CLAUSE_DE_REMUNERATION = re.compile(
    r"\bin\s+the\s+event\b|\bupon\s+(?:his|her|a|the|such)\s+\w*\s*(?:resignation|retirement|termination|departure)"
    r"|\bif\s+(?:he|she|the\s+executive|mr\.|ms\.)|\beligib|\bvest|\bseverance\b|\bgood\s+reason\b"
    r"|\bwithout\s+cause\b|\bchange\s+(?:in|of)\s+control\b",
    re.IGNORECASE)
_OPERATION_SUR_LA_DETTE = re.compile(
    r"\bnotes?\b|debt\s+securities|debentures?|\bbonds?\b|consent\s+solicitation|principal\s+amount",
    re.IGNORECASE)

# Formulations cherchées dans le CORPS du document, phrase par phrase, avec la
# phrase à écarter le cas échéant. L'ordre compte : la première catégorie dont
# un motif est trouvé l'emporte, du plus spécifique au plus général ; les
# catégories sans alerte ne passent qu'après les codes d'item (voir
# classify_8k_par_regles).
_MOTIFS_PAR_CATEGORIE = (
    ("fusion_acquisition", (
        r"merger agreement", r"agreement and plan of merger", r"business combination",
        r"definitive agreement to (?:acquire|purchase)", r"tender offer",
        r"agreed to (?:acquire|be acquired)", r"asset purchase agreement",
    ), _OPERATION_SUR_LA_DETTE),
    ("procedure_judiciaire", (
        # La faillite de l'entreprise elle-même, pas le mot « bankruptcy » :
        # celui-ci peuplait surtout les clauses de défaut des contrats de dette.
        r"chapter 11", r"chapter 7\b", r"voluntary petitions?", r"petitions? for (?:relief|reorganization)",
        r"(?:file[sd]?|filing) for (?:bankruptcy|chapter)", r"receivership",
        r"class action", r"securities litigation(?!\s+reform)", r"sec investigation",
        r"department of justice", r"subpoena", r"consent decree",
        r"settlement agreement", r"civil penalty",
    ), None),
    ("changement_guidance", (
        # Inchangés : même une guidance « mise à jour » (53 8-K, sens inconnu)
        # était suivie de réactions fortes -- 30 % au-delà de 5 % -- et d'une
        # dérive moyenne de -3,2 % à 60 séances.
        r"(?:revis|updat|lower|rais|reduc|increas)\w*\s+(?:its\s+)?(?:full[- ]year\s+)?(?:financial\s+)?(?:guidance|outlook)",
        r"withdraw\w*\s+(?:its\s+)?(?:guidance|outlook)",
        r"no longer expects", r"now expects", r"suspend\w*\s+(?:its\s+)?guidance",
    ), None),
    ("depart_dirigeant", (
        rf"{_DIRIGEANT}[^.]{{0,150}}?{_DEPART}",
        rf"{_DEPART}[^.]{{0,150}}?{_DIRIGEANT}",
    ), _CLAUSE_DE_REMUNERATION),
    ("autre_materiel", (
        r"impairment charge", r"goodwill impairment", r"restructuring (?:plan|charge|program)",
        r"non[- ]reliance", r"should no longer be relied upon", r"material weakness",
        r"restate\w*\s+(?:its\s+)?(?:financial statements|prior)",
        r"delisting", r"notice of noncompliance", r"going concern",
        r"dismissed\s+\w+\s+as (?:its\s+)?independent registered public accounting firm",
    ), None),
    ("rachat_actions", (
        r"share repurchase (?:program|authorization)", r"stock repurchase (?:program|authorization)",
        r"repurchase up to", r"authorized the repurchase", r"buyback program",
    ), _OPERATION_SUR_LA_DETTE),
)

# Une expression par catégorie (l'alternative de ses motifs) : une recherche
# par phrase et par catégorie, au lieu d'une par motif.
_MOTIFS_COMPILES = tuple(
    (categorie, re.compile("|".join(f"(?:{motif})" for motif in motifs), re.IGNORECASE), exclusion)
    for categorie, motifs, exclusion in _MOTIFS_PAR_CATEGORIE
)


def _numero_item(code: str) -> str:
    """"Item 5.02" -> "5.02". L'archive contient "Item 9.01", "Item  9.01" et
    "Item\\n9.01" comme trois valeurs distinctes : comparer les chaînes
    entières n'en verrait qu'une."""
    trouve = re.search(r"(\d+\.\d+)", str(code))
    return trouve.group(1) if trouve else ""


def phrases_a_lire(texte: str) -> List[str]:
    """Le document en phrases, sans les phrases de formulaire
    (_PHRASE_DE_FORMULAIRE) ni le titre de l'Item 5.02."""
    phrases = _FIN_DE_PHRASE.split(" ".join((texte or "").split()))
    return [_TITRE_ITEM_502.sub(" ", p) for p in phrases if p and not _PHRASE_DE_FORMULAIRE.search(p)]


def _preuve(phrases: List[str], motif: "re.Pattern", exclusion) -> Optional[str]:
    """La première phrase du document qui porte un motif de la catégorie sans
    tomber sous son exclusion."""
    for phrase in phrases:
        if motif.search(phrase) and not (exclusion is not None and exclusion.search(phrase)):
            return phrase
    return None


def classify_8k_par_regles(item_codes: List[str], text: str) -> dict:
    """Catégorie, matérialité et résumé déduits du DOCUMENT, sans modèle.

    Voir les pavés ci-dessus pour le raisonnement. Rend les mêmes clés que la
    voie modèle, plus `classification_source` -- sans quoi on ne saurait plus,
    en relisant le parquet, lequel des deux chemins a produit une ligne -- et
    `version_regles`. Le résumé est la phrase même qui a décidé : vérifiable
    contre la source."""
    numeros = {_numero_item(c) for c in item_codes}
    phrases = phrases_a_lire(text)

    def chercher(sans_alerte: bool):
        for candidate, motif, exclusion in _MOTIFS_COMPILES:
            if (candidate in _SANS_ALERTE) != sans_alerte:
                continue
            preuve = _preuve(phrases, motif, exclusion)
            if preuve:
                return candidate, preuve
        return None, None

    # Une catégorie qui donne l'alerte d'abord, puis les codes matériels par
    # définition, et seulement ensuite une catégorie sans alerte : un rachat
    # d'actions annoncé avec une dépréciation ne doit pas la masquer.
    categorie, preuve = chercher(sans_alerte=False)
    if categorie is None:
        for numero in sorted(numeros):
            if numero in _ITEMS_MATERIELS:
                categorie = _ITEMS_MATERIELS[numero]
                break
    if categorie is None:
        categorie, preuve = chercher(sans_alerte=True)

    # Un motif trouvé dans un document qui ne déclare QUE des codes
    # administratifs (9.01 pièces jointes, 5.07 vote en assemblée, 2.03 dette)
    # est très probablement une mention de passage, pas l'objet du dépôt.
    if categorie and not (numeros & (set(_ITEMS_MATERIELS) | _ITEMS_AMBIGUS)):
        categorie = None

    base = {"item_codes": item_codes, "classification_source": SOURCE_REGLES,
            "version_regles": VERSION_REGLES}
    if categorie is None:
        return {**base, "category": "non_materiel", "materiality": False, "summary": None}
    return {
        **base, "category": categorie, "materiality": categorie not in _SANS_ALERTE,
        "summary": " ".join(preuve.split())[:300] if preuve else None,
    }


def date_limite_llm(aujourd_hui: datetime, jours: Optional[int]) -> Optional[str]:
    """Date (AAAA-MM-JJ) à partir de laquelle un 8-K passe par le modèle ;
    None -- 0 jour ou moins -- pour tout l'historique (cf.
    config.LLM_8K_FENETRE_JOURS)."""
    if not jours or jours <= 0:
        return None
    return (aujourd_hui - timedelta(days=jours)).date().isoformat()


def llm_pour(filed_date, limite: Optional[str]) -> bool:
    """Ce 8-K est-il assez récent pour le modèle ? Une date absente ou
    illisible ne le prive pas du modèle : le doute profite au meilleur verdict."""
    if limite is None or not filed_date:
        return True
    return str(filed_date)[:10] >= limite


def passe_au_modele(item_codes: List[str]) -> bool:
    """Ce 8-K va-t-il à Gemini ? Seulement s'il déclare un Item que seul le
    texte peut trancher (ITEMS_POUR_LE_MODELE) et aucun Item matériel par
    définition -- celui-là décide seul, sans requête."""
    numeros = {_numero_item(c) for c in item_codes}
    return bool(numeros & ITEMS_POUR_LE_MODELE) and not numeros & set(_ITEMS_MATERIELS)


# Le bloc de signatures, qui ferme un 8-K : « SIGNATURE » ou « SIGNATURES »,
# en capitales -- en minuscules, le mot court dans le texte courant.
_SIGNATURES = re.compile(r"\bSIGNATURES?\b")


def texte_pour_le_modele(text: str, piece_jointe: Optional[str] = None) -> str:
    """Ce que Gemini lit d'un 8-K : du premier Item aux signatures, espaces
    resserrés, au plus MAX_CARACTERES_MODELE caractères. La page de garde
    (cases à cocher, adresse, titres cotés) et les signatures sont du
    formulaire, identique d'un dépôt à l'autre : des jetons payés pour rien.
    Suivi, s'il y en a un, du début du communiqué joint (Exhibit 99), au plus
    MAX_CARACTERES_PIECE_JOINTE caractères."""
    debut = ITEM_CODE_PATTERN.search(text)
    corps = text[debut.start():] if debut else text
    fin = _SIGNATURES.search(corps)
    if fin and fin.start() > 0:
        corps = corps[:fin.start()]
    texte = " ".join(corps.split())[:MAX_CARACTERES_MODELE]
    communique = " ".join((piece_jointe or "").split())[:MAX_CARACTERES_PIECE_JOINTE]
    return f"{texte}\n[Communiqué joint (Exhibit 99)]\n{communique}" if communique else texte


@dataclass
class EnAttente:
    """Un 8-K déjà classé par règles, qui attend son lot pour Gemini."""
    ligne: dict      # la ligne par règles, déjà écrite (checkpoint, mémoire)
    texte: str       # ce que Gemini en lira (texte_pour_le_modele)


def _document_pour_le_modele(attente: EnAttente) -> str:
    items = ", ".join(sorted({_numero_item(c) for c in attente.ligne.get("item_codes") or []} - {""}))
    return f"Entreprise : {attente.ligne['symbol']}. Items déclarés : {items or 'aucun'}.\n{attente.texte}"


def classer_en_attente(
    en_attente: List[EnAttente], llm_cache: Optional[Dict[str, dict]] = None,
    output_dir: Optional[Path] = None, par_requete: int = DOCUMENTS_PAR_REQUETE,
) -> List[dict]:
    """Soumet à Gemini les 8-K en attente, par lots d'au plus `par_requete`
    8-K déposés le MÊME JOUR -- jamais un 8-K plus récent à côté d'un plus
    ancien, qu'il pourrait éclairer (anticipation). Les dates les plus
    récentes d'abord : si le quota du jour s'épuise en route, ce sont les
    8-K les plus anciens qui attendent le prochain run.

    Chaque verdict est écrit aussitôt (checkpoint, mémoire) : un Ctrl-C ne
    perd pas une requête payée. Un 8-K sans réponse -- son lot entier, ou
    lui seul quand il fait échouer le lot (voir
    sec_filings_text.analyser_documents) -- garde son verdict par règles, déjà
    écrit, que le modèle reprendra au run suivant. Rend les lignes classées
    par Gemini."""
    par_date: Dict[str, List[EnAttente]] = {}
    for attente in en_attente:
        par_date.setdefault(str(attente.ligne["filed_date"])[:10], []).append(attente)
    lots = [
        (date, par_date[date][debut:debut + par_requete])
        for date in sorted(par_date, reverse=True)
        for debut in range(0, len(par_date[date]), par_requete)
    ]
    if not lots:
        return []
    logger.info(
        "Gemini : %d 8-K à lire, en %d requête(s) -- au plus %d 8-K déposés le même jour par "
        "requête, les plus récents d'abord.", len(en_attente), len(lots), par_requete)

    classees: List[dict] = []
    for numero, (date, lot) in enumerate(lots, start=1):
        if sft.llm_coupe_pour_ce_run():
            logger.warning(
                "Gemini coupé pour ce run : les %d 8-K restants gardent leur verdict par règles, "
                "que le modèle reprendra au prochain run.", sum(len(l) for _, l in lots[numero - 1:]))
            break
        documents = {f"d{rang}": _document_pour_le_modele(a) for rang, a in enumerate(lot, start=1)}
        reponses = sft.analyser_documents(documents, build_consigne(date), SCHEMA_VERDICT,
                                          max_tokens_par_document=JETONS_PAR_VERDICT)
        if reponses is None:
            continue
        lignes = []
        for rang, attente in enumerate(lot, start=1):
            verdict = reponses.get(f"d{rang}")
            if verdict is None:
                continue
            ligne = {
                **attente.ligne,
                "category": verdict["category"], "materiality": verdict["materiality"],
                "summary": verdict["summary"], "classification_source": SOURCE_GEMINI,
                "version_regles": None,
                "modele": sft.dernier_modele_utilise(),
                "fetch_timestamp": datetime.now().isoformat(timespec="seconds"), "from_cache": False,
            }
            lignes.append(ligne)
            if llm_cache is not None:
                llm_cache[cache_key(ligne["symbol"], ligne["accession_number"])] = ligne
                if output_dir is not None:
                    append_llm_cache(output_dir, ligne)
        if output_dir is not None:
            append_checkpoint(output_dir, lignes)
        classees.extend(lignes)
        if numero % 25 == 0:
            logger.info("Gemini : %d/%d requêtes, %d 8-K classés.", numero, len(lots), len(classees))
    return classees


# ----------------------------------------------------------------------------
# Mémoire des 8-K déjà classifiés (cache_8k.jsonl)
# ----------------------------------------------------------------------------

def llm_cache_path(output_dir: Path) -> Path:
    return output_dir / LLM_CACHE_FILENAME


def migrer_ancien_cache(output_dir: Path) -> None:
    """Renomme la mémoire de son ancien nom (ANCIEN_LLM_CACHE_FILENAME) vers le
    nouveau. Si les deux existent, l'ancienne passe DEVANT la nouvelle dans un
    seul fichier : la « dernière écriture gagnante » de load_llm_cache donne
    alors raison aux verdicts les plus récents. Rien n'est perdu ; les verdicts
    de Mistral qu'elle contient sont écartés ensuite, par load_llm_cache."""
    ancien = output_dir / ANCIEN_LLM_CACHE_FILENAME
    if not ancien.exists():
        return
    nouveau = llm_cache_path(output_dir)
    if nouveau.exists():
        contenu = ancien.read_bytes()
        if contenu and not contenu.endswith(b"\n"):
            contenu += b"\n"
        tmp = nouveau.with_suffix(".jsonl.tmp")
        tmp.write_bytes(contenu + nouveau.read_bytes())
        ecriture_atomique.remplacer(tmp, nouveau)
        try:
            ancien.unlink()
        except OSError as exc:
            # Verrou passager (antivirus, synchronisation) : l'ancien fichier
            # sera refusionné au prochain lancement, sans effet sur ce que
            # rend la mémoire -- la dernière écriture gagne toujours.
            logger.warning("%s non supprimé (%s) : il sera refusionné au prochain lancement.",
                           ancien, exc)
            return
    else:
        ecriture_atomique.remplacer(ancien, nouveau)
    logger.info("Mémoire des classifications renommée : %s -> %s.", ancien.name, nouveau.name)


def cache_key(symbol: str, accession_number: str) -> str:
    return f"{symbol}:{accession_number}"


def is_cacheable(classification: dict) -> bool:
    """Vrai seulement si un verdict exploitable a été rendu. Mémoriser un
    "non_evalue" reviendrait à graver dans le marbre l'échec du jour (quota
    atteint) : le 8-K ne serait plus jamais reproposé à l'analyse.

    Un verdict PAR RÈGLES est mémorisé comme les autres -- il est déterministe,
    et c'est le téléchargement du document qu'on évite de repayer, pas le
    calcul. Il reste distingué par `classification_source`, ce dont
    `load_llm_cache` se sert pour le remettre en jeu le jour où une clé d'API
    devient disponible (cf. sa docstring)."""
    return classification.get("category") not in NON_CACHEABLE_CATEGORIES


def regles_perimees(entree: dict) -> bool:
    """Verdict par règles rendu par une autre version qu'aujourd'hui (voir
    VERSION_REGLES) : la mémoire ne doit plus le servir. Un 8-K est figé, mais
    la règle qui l'a lu ne l'est pas."""
    return (entree.get("classification_source") == SOURCE_REGLES
            and entree.get("version_regles") != VERSION_REGLES)


def entrees_a_conserver(entrees: List[dict]) -> List[dict]:
    """Ce que la mémoire doit garder de chaque 8-K : son dernier verdict du
    MODÈLE, plus son dernier verdict PAR RÈGLES s'il est plus récent.

    C'est exactement ce que load_llm_cache peut retenir : avec une clé d'API,
    les verdicts par règles sont ignorés et le dernier verdict du modèle sert ;
    sans clé, le plus récent des deux. Tout le reste est un doublon -- une
    ré-analyse (--no-llm-cache), ou un 8-K repris par le modèle le jour où une
    clé arrive, ajoute une ligne sans effacer l'ancienne."""
    dernier_modele: Dict[str, int] = {}
    dernier_regles: Dict[str, int] = {}
    for i, entree in enumerate(entrees):
        cle = cache_key(entree["symbol"], entree["accession_number"])
        if entree.get("classification_source") == SOURCE_REGLES:
            dernier_regles[cle] = i
        else:
            dernier_modele[cle] = i
    garder = set(dernier_modele.values()) | {
        i for cle, i in dernier_regles.items() if i > dernier_modele.get(cle, -1)}
    return [entree for i, entree in enumerate(entrees) if i in garder]


def load_llm_cache(output_dir: Path, limite_llm: Optional[str] = None) -> Dict[str, dict]:
    """Cache des classifications déjà obtenues, indexé par symbole:accession.

    Tolérant aux lignes corrompues (un run tué en plein write laisse une ligne
    tronquée) : on ignore la ligne fautive plutôt que de perdre tout le cache
    -- une entrée manquante coûte un appel au modèle, un cache illisible en
    coûte des milliers.

    SANS DOUBLONS : à chaque chargement, les lignes illisibles, les verdicts
    remplacés (cf. entrees_a_conserver) et ceux d'avant Gemini (cf.
    SOURCES_RECONNUES) sont retirés du fichier, réécrit d'un bloc. Le fichier
    ne fait qu'ajouter des lignes en cours de run -- c'est ce qui protège un
    appel payé d'un Ctrl-C --, il est donc compacté ici, au seul moment où
    rien d'autre n'y écrit.

    Les verdicts rendus PAR RÈGLES (`classification_source == "regles_document"`)
    sont ignorés dès qu'une clé d'API est disponible : ils ont été produits
    faute de mieux, et les garder empêcherait le modèle de reprendre la main
    le jour où la clé arrive -- un repli qui se transformerait en plafond."""
    migrer_ancien_cache(output_dir)
    path = llm_cache_path(output_dir)
    if not path.exists():
        return {}
    lignes, illisibles = reprise_jsonl.lire_lignes(path)
    entrees = [e for e in lignes if e.get("symbol") and e.get("accession_number")]
    ignorees = illisibles + len(lignes) - len(entrees)
    reconnues = [e for e in entrees if e.get("classification_source") in SOURCES_RECONNUES]
    ecartees = len(entrees) - len(reconnues)
    a_jour = [e for e in reconnues if not regles_perimees(e)]
    perimees = len(reconnues) - len(a_jour)
    conservees = entrees_a_conserver(a_jour)
    doublons = len(a_jour) - len(conservees)
    if ignorees:
        logger.warning("%d ligne(s) illisible(s) ignorée(s) dans %s.", ignorees, path)
    if ecartees:
        logger.info(
            "%d verdict(s) d'avant Gemini (Mistral) écarté(s) de %s : ces 8-K sont reclassés comme "
            "des neufs -- par Gemini les récents quand une clé est disponible, par règles sinon.",
            ecartees, path)
    if perimees:
        logger.info(
            "%d verdict(s) rendu(s) par une version antérieure des règles écarté(s) de %s : ces 8-K "
            "sont retéléchargés et reclassés par les règles v%d (voir VERSION_REGLES). Les verdicts "
            "de Gemini sont gardés.", perimees, path, VERSION_REGLES)
    if doublons or ignorees or ecartees or perimees:
        reprise_jsonl.reecrire(path, conservees)
        logger.info(
            "Mémoire des classifications nettoyée : %d doublon(s), %d ligne(s) illisible(s), "
            "%d verdict(s) d'avant Gemini et %d verdict(s) par règles périmé(s) retirés de %s.",
            doublons, ignorees, ecartees, perimees, path)

    llm_disponible = sft.llm_disponible()
    cache: Dict[str, dict] = {}
    par_regles = set()
    for entree in conservees:
        cle = cache_key(entree["symbol"], entree["accession_number"])
        if (llm_disponible and entree.get("classification_source") == SOURCE_REGLES
                and llm_pour(entree.get("filed_date"), limite_llm)
                and passe_au_modele(entree.get("item_codes") or [])):
            # Seul un 8-K RÉCENT, et que Gemini lirait (voir passe_au_modele),
            # repart au modèle : un ancien, ou un 8-K dont les Items décident
            # seuls, classé par règles, reste servi par le cache (cf.
            # config.LLM_8K_FENETRE_JOURS).
            par_regles.add(cle)
            continue
        # Dernière écriture gagnante : une ré-analyse (--no-llm-cache)
        # remplace l'ancien verdict.
        cache[cle] = entree
    # Remis en jeu : les 8-K qui n'ont QU'UN verdict par règles. Ceux qui ont
    # aussi un verdict du modèle le gardent, et ne repartent pas au modèle.
    remis_en_jeu = len(par_regles - set(cache))
    if remis_en_jeu:
        logger.info(
            "%d 8-K classés par règles remis en jeu : %s est disponible, "
            "le modèle reprend la main dessus.", remis_en_jeu, sft.description_llm(),
        )
    logger.info("Mémoire des classifications : %d 8-K déjà analysés dans %s.", len(cache), path)
    return cache


def append_llm_cache(output_dir: Path, entry: dict) -> None:
    """Écriture IMMÉDIATE, une ligne par 8-K classifié. L'appel à Gemini vient
    d'être payé : il ne doit pas être reperdu par un Ctrl-C dix secondes plus
    tard."""
    path = llm_cache_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
        f.flush()


def row_from_cache(entry: dict, symbol: str, cik: str, filing: dict) -> dict:
    """Ligne de sortie reconstruite depuis le cache. Les champs d'identité
    viennent du filing courant (source de vérité), le verdict du cache."""
    return {
        "symbol": symbol, "cik": cik, "filed_date": filing["filing_date"],
        "accession_number": filing["accession_number"],
        "item_codes": entry.get("item_codes") or [],
        "category": entry.get("category"),
        "materiality": entry.get("materiality"),
        "summary": entry.get("summary"),
        # Reporté depuis le cache : sans lui, une ligne relue ne dirait plus
        # lequel des deux chemins l'a produite, et le parquet deviendrait
        # ininterprétable dès qu'un run mélange les deux.
        "classification_source": entry.get("classification_source"),
        "version_regles": entry.get("version_regles"),
        "modele": entry.get("modele"),
        "fetch_timestamp": entry.get("fetch_timestamp"),
        "from_cache": True,
    }


def process_ticker_8k(
    symbol: str, cik: str, windows: List[tuple],
    llm_cache: Optional[Dict[str, dict]] = None, output_dir: Optional[Path] = None,
    limite_llm: Optional[str] = None, en_attente: Optional[List[EnAttente]] = None,
) -> tuple:
    """Lignes 8-K du ticker, plus le nombre de classifications servies par le
    cache. `llm_cache` à None désactive complètement la mémoire (--no-llm-cache).

    Chaque 8-K est classé ici PAR RÈGLES. Ceux que Gemini doit lire -- déposés
    à partir de `limite_llm`, et dont les Items ne décident pas seuls (voir
    passe_au_modele) -- sont en plus ajoutés à `en_attente`, que
    classer_en_attente soumet ensuite par lots de même date. `en_attente` à
    None : aucun 8-K n'est mis en attente (pas de clé Gemini)."""
    rows = []
    cache_hits = 0
    seen_accessions = set()
    # UNE seule requête submissions par entreprise, puis filtrage en mémoire.
    # Auparavant, list_company_filings était appelée une fois PAR FENÊTRE :
    # une entreprise avec 40 trimestres TTM connus téléchargeait 40 fois le
    # même JSON (~20 000 requêtes SEC pour ~500 nécessaires à l'échelle du
    # S&P 500).
    submissions = sft.fetch_submissions_strict(cik)
    for start_date, end_date in windows:
        filings = sft.filter_filings(submissions, forms=("8-K",), start_date=start_date, end_date=end_date)
        for filing in filings:
            if filing["accession_number"] in seen_accessions:
                continue  # deux fenêtres adjacentes peuvent se recouvrir sur leur borne commune
            seen_accessions.add(filing["accession_number"])

            # Déjà classifié lors d'un run précédent : ni téléchargement SEC,
            # ni appel à Gemini. Le test vient AVANT fetch_filing_text, sinon
            # l'économie se limiterait au LLM.
            if llm_cache is not None:
                connu = llm_cache.get(cache_key(symbol, filing["accession_number"]))
                if connu is not None:
                    rows.append(row_from_cache(connu, symbol, cik, filing))
                    cache_hits += 1
                    continue

            if not filing.get("primary_document"):
                logger.warning(
                    "%s : filing %s du %s sans primary_document, ignoré (filing ancien ?).",
                    symbol, filing["accession_number"], filing["filing_date"],
                )
                continue

            url = sft.filing_document_url(cik, filing["accession_number"], filing["primary_document"])
            # form="8-K" : pas de tentative d'extraction par section (document
            # court, dont l'objet est annoncé dès les premières lignes).
            extracted = sft.fetch_filing_text(url, form=filing["form"])
            if extracted is None:
                continue
            text, _extraction_mode = extracted

            item_codes = extract_item_codes(text)
            classification = classify_8k_par_regles(item_codes, text)
            row = {
                "symbol": symbol, "cik": cik, "filed_date": filing["filing_date"],
                "accession_number": filing["accession_number"],
                **classification,
                "modele": None,
                "fetch_timestamp": datetime.now().isoformat(timespec="seconds"),
                "from_cache": False,
            }
            rows.append(row)

            if llm_cache is not None and is_cacheable(classification):
                llm_cache[cache_key(symbol, filing["accession_number"])] = row
                if output_dir is not None:
                    append_llm_cache(output_dir, row)

            # Le verdict par règles reste écrit : si Gemini ne répond pas pour
            # son lot, c'est lui qui sert, et le modèle le reprendra au run
            # suivant (voir load_llm_cache).
            if (en_attente is not None and llm_pour(filing["filing_date"], limite_llm)
                    and passe_au_modele(item_codes)):
                # Le communiqué joint, pour Gemini seulement : les règles
                # restent sur le 8-K, où leurs motifs ont été mesurés -- un
                # communiqué de résultats cite dépréciations et restructurations
                # à chaque trimestre, dans ses tableaux.
                communique = sft.texte_piece_jointe(cik, filing["accession_number"], MAX_CARACTERES_PIECE_JOINTE)
                en_attente.append(EnAttente(ligne=row, texte=texte_pour_le_modele(text, communique)))
    return rows, cache_hits


# ----------------------------------------------------------------------------
# Checkpoint/reprise façon 08_recuperation_options.py
# ----------------------------------------------------------------------------

def _progress_path(output_dir: Path) -> Path:
    return output_dir / "progress_8k.json"


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint_8k.jsonl"


def load_progress(output_dir: Path) -> set:
    path = _progress_path(output_dir)
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")).get("processed", []))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Fichier de progression illisible (%s), on repart de zéro.", e)
        return set()


def save_progress(output_dir: Path, processed: set) -> None:
    path = _progress_path(output_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"processed": sorted(processed), "updated_at": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=2), encoding="utf-8")
    ecriture_atomique.remplacer(tmp, path)


def append_checkpoint(output_dir: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with _checkpoint_path(output_dir).open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")


def load_checkpoint_rows(output_dir: Path) -> List[dict]:
    """Un 8-K par ligne : un ticker refait après une reprise (--resume) écrit
    ses 8-K une seconde fois (cf. reprise_jsonl)."""
    return reprise_jsonl.lire_sans_doublons(
        _checkpoint_path(output_dir),
        cle=lambda row: (row.get("symbol"), row.get("accession_number")), journal=logger)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tickers", type=Path, default=None,
        help="CSV d'univers (défaut : univers point-in-time de 01b s'il existe, "
             "sinon univers actuel de 01 -- voir config.default_universe_file).",
    )
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--output-dir", default=config.DIR_FINANCIALS, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-llm-cache", action="store_true",
        help="Ignore " + LLM_CACHE_FILENAME + " et re-soumet à Gemini des 8-K déjà classifiés "
             "(à réserver à un changement de consigne ou de modèle : chaque appel est payant).",
    )
    parser.add_argument(
        "--llm-depuis-jours", type=int, default=config.LLM_8K_FENETRE_JOURS,
        help="Seuls les 8-K déposés depuis ce nombre de jours passent par le modèle ; les plus "
             "anciens sont classés par règles, sans appel (défaut: %(default)s, la plus longue "
             "durée de vie d'un signal -- un 8-K plus ancien ne touche plus aucun signal actif). "
             "0 : tout l'historique.",
    )
    parser.add_argument(
        "--par-requete", type=int, default=DOCUMENTS_PAR_REQUETE,
        help="8-K déposés le même jour envoyés à Gemini dans une seule requête (défaut: "
             "%(default)s). Plus haut : moins de requêtes, des requêtes plus longues.",
    )
    parser.add_argument(
        "--vider-memoire", action="store_true",
        help="Efface " + LLM_CACHE_FILENAME + " avant le run : chaque 8-K est retéléchargé à la "
             "SEC et reclassé (règles, puis Gemini pour les récents à lire).",
    )
    parser.add_argument(
        "--max-failure-ratio", type=float, default=DEFAULT_MAX_FAILURE_RATIO,
        help="Part maximale d'entreprises en échec RÉSEAU tolérée avant d'abandonner le run "
             "sans rien écrire (défaut: %(default)s). Un material_events_8k.parquet incomplet "
             "désactive silencieusement le filtre d'événements matériels du backtest.",
    )
    args = parser.parse_args()
    if args.vider_memoire and args.resume:
        parser.error("--vider-memoire repart de zéro : incompatible avec --resume.")
    if args.par_requete < 1:
        parser.error("--par-requete doit valoir au moins 1.")
    limite_llm = date_limite_llm(datetime.now(), args.llm_depuis_jours)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if sft.sec_http.require_contact_email(logger) is None:
        sys.exit(1)

    if not sft.llm_disponible():
        # Ce message annonçait « category='non_evalue' », le comportement
        # d'AVANT la classification par règles : il faisait croire à un run
        # inutile alors que chaque 8-K est bel et bien lu et classé.
        logger.info(
            "Aucune clé Gemini (%s) : chaque 8-K est téléchargé et classé PAR RÈGLES à partir "
            "de son texte. Pour que Gemini classe les 8-K récents : %s",
            sft.GEMINI_API_KEY_ENV, sft.aide_cle_absente(),
        )
    else:
        logger.info(
            "Classification par %s %s déclarant un Item %s, jusqu'à %d par requête ; tous les "
            "autres sont classés par règles. Débit : un appel toutes les %.2fs (%s pour l'ajuster "
            "au quota de ton offre). Le débit se resserre automatiquement en cas de 429.",
            sft.description_llm(),
            "de tous les 8-K" if limite_llm is None else
            f"des 8-K déposés depuis le {limite_llm} ({args.llm_depuis_jours} jours)",
            "/".join(sorted(ITEMS_POUR_LE_MODELE)), args.par_requete,
            sft.GEMINI_RATE_LIMITER.interval, sft.GEMINI_REQUESTS_PER_SECOND_ENV,
        )

    if not config.FINANCIALS_TTM_FILE.exists():
        logger.error(
            "%s introuvable. Lance d'abord 04b_recuperation_10q.py : ce script a besoin des "
            "trimestres TTM déjà connus pour délimiter les fenêtres de recherche des 8-K.",
            config.FINANCIALS_TTM_FILE,
        )
        sys.exit(1)
    ttm = pd.read_parquet(config.FINANCIALS_TTM_FILE)

    if args.ticker:
        symbols_ric = [args.ticker.upper()]
    else:
        tickers_file = args.tickers or config.default_universe_file()
        if args.tickers is None:
            config.journaliser_univers_retenu(logger, tickers_file)
        universe = pd.read_csv(tickers_file, encoding="utf-8-sig")
        symbols_ric = universe["RIC"].dropna().unique().tolist()
        if args.limit:
            symbols_ric = symbols_ric[: args.limit]

    # cik par symbole : porté par FINANCIALS_TTM_FILE (04b), pas besoin de
    # re-résoudre via company_tickers.json comme 04/04b.
    cik_by_symbol = ttm.drop_duplicates(subset=["symbol"]).set_index("symbol")["cik"].astype(str).to_dict()

    processed_keys: set = set()
    if args.resume:
        processed_keys = load_progress(args.output_dir)
        logger.info("Reprise : %d tickers déjà traités.", len(processed_keys))
    else:
        # Progression et checkpoint sont propres à UN run et repartent de zéro.
        # Le cache des classifications, lui, est délibérément conservé : c'est
        # une mémoire de documents figés, pas l'état d'avancement d'un run.
        _progress_path(args.output_dir).unlink(missing_ok=True)
        _checkpoint_path(args.output_dir).unlink(missing_ok=True)

    if args.vider_memoire:
        for chemin in (llm_cache_path(args.output_dir), args.output_dir / ANCIEN_LLM_CACHE_FILENAME):
            if chemin.exists():
                chemin.unlink()
                logger.info("--vider-memoire : %s effacé, chaque 8-K sera retéléchargé et reclassé.", chemin)

    llm_cache: Optional[Dict[str, dict]] = None
    if args.no_llm_cache:
        logger.warning(
            "--no-llm-cache : les 8-K déjà classifiés seront re-téléchargés et re-soumis à Gemini."
        )
    else:
        llm_cache = load_llm_cache(args.output_dir, limite_llm)

    today = datetime.now()
    to_process = []
    for ric in symbols_ric:
        symbol = config.to_ib_symbol(ric)
        if symbol in processed_keys:
            continue
        if symbol not in cik_by_symbol:
            continue  # pas de trimestre TTM connu pour ce symbole -- rien à borner, ignoré silencieusement
        to_process.append(symbol)

    logger.info("%d/%d tickers avec un historique TTM à interroger.", len(to_process), len(symbols_ric))

    # Trois compteurs distincts, et non un seul "échec" : une entreprise sans
    # 8-K et une entreprise dont la SEC n'a pas répondu ne veulent pas dire la
    # même chose. Seuls les échecs RÉSEAU remettent en cause la complétude du
    # fichier de sortie (cf. le seuil en fin de run).
    ok_count, not_found_count, network_fail_count, event_count = 0, 0, 0, 0
    cache_hit_count = 0
    since_checkpoint = 0
    interrompu = False
    # Les 8-K que Gemini lira, soumis par lots de même date une fois tous les
    # tickers parcourus (classer_en_attente). None sans clé : rien à attendre.
    en_attente: Optional[List[EnAttente]] = [] if sft.llm_disponible() else None
    classes_par_gemini = 0
    try:
        for i, symbol in enumerate(to_process, start=1):
            cik = cik_by_symbol[symbol]
            windows = compute_search_windows(ttm, symbol, today)
            logger.info("[%d/%d] %s (CIK %s, %d fenêtre(s))...", i, len(to_process), symbol, cik, len(windows))
            hits = 0
            try:
                rows, hits = process_ticker_8k(symbol, cik, windows, llm_cache, args.output_dir, limite_llm,
                                               en_attente)
            except KeyboardInterrupt:
                # Ctrl-C pendant une attente de quota : sortie propre (le
                # cache et le checkpoint sont déjà sur disque), pas une trace
                # de pile de vingt lignes.
                logger.warning("Interruption demandée : arrêt après %d/%d tickers.", i - 1, len(to_process))
                interrompu = True
                break
            except sft.sec_http.SecNotFound:
                # CIK inconnu de la SEC : réponse définitive, il n'y a rien à
                # récupérer et rien à réessayer. Ce n'est pas un incident.
                logger.info("  -> CIK %s inconnu de la SEC pour %s, ignoré.", cik, symbol)
                rows = []
                not_found_count += 1
            except sft.sec_http.SecUnavailable as exc:
                # La SEC n'a pas répondu : les 8-K de cette entreprise existent
                # peut-être. C'est CE cas qui creusait des trous silencieux.
                logger.warning("  -> SEC indisponible pour %s : %s (à relancer)", symbol, exc)
                rows = []
                network_fail_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("  -> ECHEC pour %s : %s (ticker ignoré, on continue)", symbol, exc)
                rows = []
                network_fail_count += 1
            else:
                ok_count += 1
                event_count += len(rows)
                cache_hit_count += hits
                if rows:
                    append_checkpoint(args.output_dir, rows)

            processed_keys.add(symbol)
            since_checkpoint += 1
            if since_checkpoint >= CHECKPOINT_EVERY:
                save_progress(args.output_dir, processed_keys)
                since_checkpoint = 0
    finally:
        save_progress(args.output_dir, processed_keys)

    if en_attente and not interrompu:
        try:
            classes_par_gemini = len(classer_en_attente(en_attente, llm_cache, args.output_dir, args.par_requete))
        except KeyboardInterrupt:
            # Les lots déjà classés sont écrits ; les autres gardent leur
            # verdict par règles, que le modèle reprendra au prochain run.
            logger.warning("Interruption demandée pendant la classification par Gemini.")
            interrompu = True

    logger.info(
        "Terminé. OK: %d | CIK introuvables: %d | Échecs réseau: %d | 8-K détectés: %d "
        "(dont %d servis par la mémoire, %d nouvellement analysés, dont %d lus par Gemini)",
        ok_count, not_found_count, network_fail_count, event_count,
        cache_hit_count, event_count - cache_hit_count, classes_par_gemini,
    )
    if sft.llm_disponible():
        # Le modèle a-t-il vraiment travaillé ? Une ligne, plutôt que de le
        # déduire de centaines de lignes de réessais.
        logger.info("Modèle : %s", sft.bilan_llm())
        if sft.documents_sans_modele():
            logger.info("Les 8-K récents classés par règles faute de réponse du modèle lui seront "
                        "rendus au prochain run.")
    if llm_cache is not None:
        logger.info("Mémoire des classifications : %d 8-K au total dans %s.",
                    len(llm_cache), llm_cache_path(args.output_dir))

    if interrompu:
        # Le fichier de sortie serait tronqué à l'endroit exact du Ctrl-C, et
        # un material_events_8k.parquet partiel désactive silencieusement le
        # filtre d'événements matériels du backtest. Rien n'est écrit.
        logger.warning(
            "Run interrompu : aucun fichier de sortie généré. Le travail déjà payé est "
            "conservé (%s) ; relance avec --resume pour reprendre là où tu en étais.",
            llm_cache_path(args.output_dir),
        )
        sys.exit(130)

    failure_ratio = network_fail_count / len(to_process) if to_process else 0.0
    if failure_ratio > args.max_failure_ratio:
        logger.error(
            "%.0f%% des entreprises (%d/%d) ont échoué pour cause d'indisponibilité SEC, "
            "au-delà du seuil de %.0f%%. Le fichier de sortie serait INCOMPLET, et un "
            "material_events_8k.parquet incomplet désactive silencieusement le filtre "
            "d'événements matériels du backtest (load_material_events_8k rend None et le "
            "run continue). Rien n'est écrit : relance 04c_recuperation_8k.py --resume "
            "quand la SEC répond de nouveau.",
            failure_ratio * 100, network_fail_count, len(to_process), args.max_failure_ratio * 100,
        )
        sys.exit(1)

    rows = load_checkpoint_rows(args.output_dir)
    if not rows:
        logger.warning("Aucun 8-K collecté, pas de fichier de sortie généré.")
        return

    df = pd.DataFrame(rows)
    if args.ticker or args.limit or args.tickers:
        # Run PARTIEL : il ne remplace que ses propres 8-K dans le fichier
        # complet, que le backtest et le paper trading lisent (cf. reprise_jsonl).
        df, conserves = reprise_jsonl.fusionner_run_partiel(
            df, config.MATERIAL_EVENTS_8K_FILE, ["symbol", "accession_number"])
        logger.info(
            "Run partiel (--ticker, --limit ou --tickers) : %d 8-K de ce run fusionnés dans %s, "
            "%d autres conservés tels quels.", len(rows), config.MATERIAL_EVENTS_8K_FILE, conserves)
    if "from_cache" in df.columns:
        # Un checkpoint écrit par une version antérieure n'a pas la colonne :
        # sans normalisation, le mélange bool/NaN part en colonne "object" et
        # pyarrow refuse d'inférer un type.
        df["from_cache"] = df["from_cache"].fillna(False).astype(bool)
    config.MATERIAL_EVENTS_8K_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.MATERIAL_EVENTS_8K_FILE, index=False, engine="pyarrow")
    logger.info("8-K sauvegardés : %s (%d lignes, %d entreprises).", config.MATERIAL_EVENTS_8K_FILE, len(df), df["symbol"].nunique())
    if "materiality" in df.columns:
        logger.info("Répartition matérialité : %s", df["materiality"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
