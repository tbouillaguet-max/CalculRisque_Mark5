"""
Validation QUALITATIVE, par LLM (Gemini), du signal quantitatif de
valorisation (07_calcul_dcf.py / 06b_calcul_valorisation_combinee.py) :
pour chaque (symbol, période) déjà valorisé, récupère le texte du 10-K/10-Q
DE CE DÉPÔT PRÉCIS (jamais un filing plus récent) et demande au modèle si le
narratif de l'entreprise à cette date (risques, perspectives) est cohérent
avec, à surveiller, ou contradictoire vis-à-vis de l'écart de valorisation
calculé (gap_pct). Un DCF peut être arithmétiquement correct tout en reposant
sur des hypothèses que l'entreprise elle-même contredit dans son propre
texte (ex: dépréciation d'actif annoncée, procédure judiciaire matérielle,
guidance revue à la baisse) -- ce script sert de garde-fou qualitatif, pas de
remplacement du signal quantitatif.

Contrainte anti-anticipation (point-in-time)
----------------------------------------------
Le modèle ne reçoit QUE le texte du filing déposé À la filed_date de la
période évaluée -- jamais un filing postérieur, jamais un résumé agrégé de
plusieurs dépôts. Cette garantie est structurelle, pas une simple consigne de
prompt : sec_filings_text.py::get_filing_text_asof ne peut physiquement
renvoyer que le document déposé à cette date exacte (voir sa docstring). Le
verdict produit pour une période N est donc exactement ce qu'un lecteur du
10-K/10-Q de l'époque aurait pu établir à l'époque -- aucune connaissance
d'événements survenus depuis n'est jamais transmise au modèle.

Au moindre coût en requêtes et en jetons
----------------------------------------
Au palier gratuit de Gemini, c'est le nombre de REQUÊTES par jour qui borne
un run. Trois économies :
  - une MÉMOIRE des verdicts (cache_qualitative.jsonl) : un filing est figé, et
    un verdict rendu pour lui -- et pour le même sens d'écart de valorisation
    -- sert à tous les runs suivants, sans requête ni téléchargement. Avant,
    chaque run renvoyait toutes les périodes à Gemini (2 182 dans le dépôt) ;
  - des LOTS (classer_en_attente) : jusqu'à PERIODES_PAR_REQUETE périodes par
    requête, toutes déposées le même jour -- aucun filing plus récent n'éclaire
    le jugement d'un plus ancien ;
  - un extrait plus court (MAX_CARACTERES_MODELE).
Chaque lot réunit les extraits, une consigne (CONSIGNE_TEMPLATE) et le format
de la réponse (SCHEMA_VERDICT, un schéma JSON, un verdict par période) :
Gemini ne peut répondre qu'un verdict parmi coherent, a_surveiller et
contradictoire -- ceux que lit le filtre qualitatif du backtest --, une phrase
de justification et au plus cinq risques cités ; la réponse est revérifiée
avant d'être gardée.

Ce module ne crée volontairement PAS de mécanisme générique séparé : la
logique de recherche/téléchargement/extraction de texte SEC et l'appel LLM
générique vivent dans sec_filings_text.py (réutilisé aussi par
04c_recuperation_8k.py, qui applique la même contrainte anti-anticipation à
la détection d'événements matériels).

Prérequis :
    pip install requests beautifulsoup4
    GEMINI_API_KEY=ta_cle dans .env (modèle : .env.example) -- sans cette
    variable, le script sert les verdicts déjà en mémoire, journalise les
    autres périodes comme "non_evalue_pas_de_cle_api" et n'appelle jamais le
    modèle, plutôt que de planter

Usage :
    python 07b_validation_qualitative.py
    python 07b_validation_qualitative.py --limit 10
    python 07b_validation_qualitative.py --resume
    python 07b_validation_qualitative.py --no-llm-cache    # rejuger, mémoire ignorée
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

import config
import ecriture_atomique
import reprise_jsonl
import sec_filings_text as sft

logger = logging.getLogger("validation_qualitative")

CHECKPOINT_EVERY = 10
FORM_BY_PERIOD_TYPE = {"FY": ("10-K",), "TTM": ("10-Q", "10-K")}  # TTM peut être "complété" par un 10-K si le dernier trimestre du TTM est un Q4 (voir 04b)

# Verdicts possibles. Le filtre qualitatif du backtest exclut ceux de
# config.QUALITATIVE_GATE_EXCLUDED_VERDICTS : ils doivent figurer ici, sinon le
# modèle ne pourrait jamais les rendre et le filtre ne filtrerait plus rien.
VERDICTS = ("coherent", "a_surveiller", "contradictoire")

# Périodes par requête, toutes déposées le MÊME jour (voir classer_en_attente).
# Un extrait fait jusqu'à MAX_CARACTERES_MODELE caractères : cinq tiennent
# largement dans le quota de jetons par minute du palier gratuit.
PERIODES_PAR_REQUETE = 5
# Extrait transmis à Gemini par période (sections de risque, ou début du
# document), au lieu des 15 000 caractères d'avant.
MAX_CARACTERES_MODELE = 8_000
# Budget de RÉPONSE par période : un verdict, une phrase, cinq risques courts.
JETONS_PAR_VERDICT = 200
MEMOIRE_FILENAME = "cache_qualitative.jsonl"

# Ce que contient l'extrait transmis, selon sec_filings_text.fetch_filing_text.
# L'ancien prompt annonçait toujours « le début du document », y compris quand
# le texte était fait des sections de risque.
_CONTENU_EXTRAIT = {
    "sections": "sections facteurs de risque, procédures judiciaires et analyse de la direction, "
                "intitulés entre crochets",
    "debut_document": "début du document",
}

# La CONSIGNE donnée à Gemini (instruction système), une fois par lot. Les
# extraits partent à côté, et le format de la réponse dans SCHEMA_VERDICT : la
# consigne ne répète ni les champs ni un exemple de JSON -- Google le
# déconseille, la qualité baisse.
CONSIGNE_TEMPLATE = """Tu es un analyste financier. Tu reçois des extraits de 10-K ou de 10-Q déposés le {filed_date}, chacun entre des balises <document id="...">, précédé de l'entreprise, de la forme, de ce que contient l'extrait et de l'écart de valorisation qu'un modèle quantitatif (DCF/multiples) calcule à cette date (positif : entreprise jugée sous-évaluée ; négatif : jugée survalorisée). Juge chaque document séparément, uniquement à partir de son propre texte : ignore les autres documents du lot et tout ce que tu pourrais savoir par ailleurs, en particulier après cette date.

Pour chacun, rends un verdict : coherent si rien dans le document ne remet l'écart en cause, a_surveiller s'il signale des risques à suivre, contradictoire s'il le contredit (dépréciation d'actifs, procédure judiciaire matérielle, révision de guidance, doute sur la continuité d'exploitation...). Justifie-le en une phrase courte, en français, et relève les principaux risques que le document cite, en quelques mots chacun."""

# Le FORMAT de la réponse pour UNE période ; sec_filings_text.analyser_documents
# en exige un par document du lot. Gemini génère sous sa contrainte : un
# verdict de la liste, une phrase, au plus cinq risques -- et rien d'autre.
# Sans description : répétée pour chaque période du lot, elle coûterait des
# jetons à chaque requête, et la consigne dit déjà ce que chaque champ attend.
SCHEMA_VERDICT = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "justification": {"type": "string"},
        "risques_cites": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    },
    "required": ["verdict", "justification", "risques_cites"],
}


def build_consigne(filed_date: str) -> str:
    return CONSIGNE_TEMPLATE.format(filed_date=filed_date)


def load_signal_periods(limit: Optional[int] = None) -> pd.DataFrame:
    """Une ligne par (symbol, period_type, fiscal_year, fiscal_quarter) déjà
    valorisée : DCF_HISTORY_FILE (07) en priorité (toujours présent),
    complété par VALORISATION_COMBINEE_FILE (06b, gap_pct par multiples
    éventuellement différent) si disponible -- dédupliqué sur la clé de
    période, la valeur de gap_pct utilisée n'a pas besoin d'être identique
    entre les deux (l'objectif ici est juste de couvrir les périodes déjà
    signalées comme intéressantes par AU MOINS une des deux méthodes)."""
    if not config.DCF_HISTORY_FILE.exists():
        raise FileNotFoundError(f"{config.DCF_HISTORY_FILE} introuvable. Lance d'abord 07_calcul_dcf.py.")
    dcf = pd.read_parquet(config.DCF_HISTORY_FILE)
    dcf = dcf.dropna(subset=["gap_pct", "cik"]) if "cik" in dcf.columns else dcf.dropna(subset=["gap_pct"])

    keep_cols = ["symbol", "cik", "period_type", "fiscal_year", "fiscal_quarter", "filed_date", "gap_pct"]
    keep_cols = [c for c in keep_cols if c in dcf.columns]
    df = dcf[keep_cols].copy()

    if "cik" not in df.columns:
        # DCF_HISTORY_FILE ne porte pas le CIK (ajouté par 04/04b, mais
        # perdu en route dans build_input_table -- cf. 07 : le CIK n'est
        # utile qu'ici) : impossible de retrouver le filing sans lui.
        raise ValueError(
            f"{config.DCF_HISTORY_FILE} n'a pas de colonne 'cik' : relance 04_recuperation_10k.py "
            "--force-refresh puis 07_calcul_dcf.py pour régénérer un historique avec le CIK."
        )

    df = df.dropna(subset=["cik", "filed_date"]).drop_duplicates(subset=["symbol", "period_type", "fiscal_year", "fiscal_quarter"])
    df = df.sort_values("filed_date", ascending=False)
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Mémoire des verdicts (cache_qualitative.jsonl)
# ----------------------------------------------------------------------------

def memoire_path(output_dir: Path) -> Path:
    return output_dir / MEMOIRE_FILENAME


def cle_memoire(accession_number: str, gap_pct: float) -> str:
    """Un verdict vaut pour UN filing et le SENS de l'écart de valorisation :
    le texte d'un filing ne change plus, mais le verdict juge sa cohérence avec
    l'écart -- une sous-évaluation et une survalorisation ne se jugent pas
    pareil. L'ampleur de l'écart, qui bouge à chaque recalcul, ne compte pas :
    la prendre en compte renverrait chaque période à Gemini à chaque run."""
    return f"{accession_number}|{'positif' if gap_pct >= 0 else 'negatif'}"


def charger_memoire(output_dir: Path) -> Dict[str, dict]:
    """Verdicts déjà rendus, par cle_memoire -- la dernière écriture gagne.
    Tolérante aux lignes tronquées par un run interrompu."""
    lignes, illisibles = reprise_jsonl.lire_lignes(memoire_path(output_dir))
    if illisibles:
        logger.warning("%d ligne(s) illisible(s) ignorée(s) dans %s.", illisibles, memoire_path(output_dir))
    memoire = {ligne["cle"]: ligne for ligne in lignes
               if ligne.get("cle") and ligne.get("verdict") in VERDICTS}
    logger.info("Mémoire des verdicts : %d filing(s) déjà jugé(s) dans %s.", len(memoire), memoire_path(output_dir))
    return memoire


def memoriser(output_dir: Path, entree: dict) -> None:
    """Écriture IMMÉDIATE : la requête vient d'être payée."""
    with memoire_path(output_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entree, default=str, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# Une période : verdict immédiat, ou mise en attente de son lot
# ----------------------------------------------------------------------------

@dataclass
class PeriodeEnAttente:
    """Une période dont le filing est téléchargé, qui attend son lot pour Gemini."""
    row: pd.Series
    filing: dict           # form, accession_number, extraction_mode...
    texte: str


def _sans_verdict(verdict: str, justification: str, filing: Optional[dict] = None) -> dict:
    resultat = {"verdict": verdict, "justification": justification, "risques_cites": None}
    if filing is not None:
        resultat.update(accession_number=filing["accession_number"], form=filing["form"],
                        extraction_mode=filing.get("extraction_mode"))
    return resultat


def preparer_periode(row: pd.Series, memoire: Optional[Dict[str, dict]] = None
                     ) -> Tuple[Optional[dict], Optional[PeriodeEnAttente]]:
    """(verdict, None) quand la période se règle sans requête -- filing
    introuvable, pas de clé, verdict déjà en mémoire --, sinon
    (None, période en attente) avec l'extrait de son filing, téléchargé.

    La mémoire est lue AVANT le téléchargement : un filing déjà jugé ne coûte
    ni requête Gemini, ni requête SEC pour son document. Elle sert aussi sans
    clé : un verdict déjà rendu reste valable."""
    cik = str(row["cik"])
    period_type = row.get("period_type") or "FY"
    filed_date = str(row["filed_date"])[:10]
    forms = FORM_BY_PERIOD_TYPE.get(period_type, ("10-K", "10-Q"))

    filing = sft.find_filing_asof(cik, filed_date, forms=forms)
    if filing is None:
        # Distingué de "pas de clé API" ci-dessous : les deux produisaient le
        # même "non_evalue", si bien qu'un trou de couverture SEC et une
        # absence de configuration se lisaient pareil dans le parquet de
        # sortie -- impossible de savoir laquelle des deux corriger.
        return _sans_verdict("non_evalue_filing_introuvable",
                             "Aucun 10-K/10-Q trouvé à cette date de dépôt exacte."), None

    connu = memoire.get(cle_memoire(filing["accession_number"], float(row["gap_pct"]))) if memoire else None
    if connu is not None:
        return {
            "verdict": connu["verdict"], "justification": connu.get("justification"),
            "risques_cites": connu.get("risques_cites"), "accession_number": filing["accession_number"],
            "form": filing["form"], "extraction_mode": connu.get("extraction_mode"),
            "modele": connu.get("modele"), "from_cache": True,
        }, None

    if not sft.llm_disponible():
        return _sans_verdict("non_evalue_pas_de_cle_api",
                             f"{sft.GEMINI_API_KEY_ENV} non définie : aucun appel au modèle.", filing), None

    url = sft.filing_document_url(cik, filing["accession_number"], filing["primary_document"])
    extrait = sft.fetch_filing_text(url, max_chars=MAX_CARACTERES_MODELE, form=filing.get("form"))
    if extrait is None:
        return _sans_verdict("non_evalue_filing_introuvable",
                             "Document du filing impossible à télécharger.", filing), None
    texte, extraction_mode = extrait
    return None, PeriodeEnAttente(row=row, filing={**filing, "extraction_mode": extraction_mode}, texte=texte)


def _document_pour_le_modele(attente: PeriodeEnAttente) -> str:
    contenu = _CONTENU_EXTRAIT.get(attente.filing.get("extraction_mode"), "extrait du document")
    return (f"Entreprise : {attente.row['symbol']}. Forme : {attente.filing['form']}. Extrait : {contenu}. "
            f"Écart de valorisation : {float(attente.row['gap_pct']):+.1f} %.\n{attente.texte}")


def classer_en_attente(
    en_attente: List[PeriodeEnAttente], memoire: Optional[Dict[str, dict]] = None,
    output_dir: Optional[Path] = None, par_requete: int = PERIODES_PAR_REQUETE,
) -> Iterator[Tuple[PeriodeEnAttente, dict]]:
    """Soumet à Gemini les périodes en attente, par lots d'au plus
    `par_requete` filings déposés le MÊME JOUR -- jamais un filing plus récent
    à côté d'un plus ancien, qu'il pourrait éclairer (anticipation) --, les
    dates les plus récentes d'abord. Rend (période, verdict) au fil des lots ;
    chaque verdict du modèle est mémorisé aussitôt. Une période sans réponse
    -- son lot entier, ou elle seule quand elle fait échouer le lot (voir
    sec_filings_text.analyser_documents) -- sort en non_evalue_reponse_invalide."""
    par_date: Dict[str, List[PeriodeEnAttente]] = {}
    for attente in en_attente:
        par_date.setdefault(str(attente.row["filed_date"])[:10], []).append(attente)
    lots = [
        (date, par_date[date][debut:debut + par_requete])
        for date in sorted(par_date, reverse=True)
        for debut in range(0, len(par_date[date]), par_requete)
    ]
    if lots:
        logger.info(
            "Gemini : %d période(s) à juger, en %d requête(s) -- au plus %d filings déposés le "
            "même jour par requête, les plus récents d'abord.", len(en_attente), len(lots), par_requete)

    for date, lot in lots:
        reponses = None
        if not sft.llm_coupe_pour_ce_run():
            documents = {f"d{rang}": _document_pour_le_modele(a) for rang, a in enumerate(lot, start=1)}
            reponses = sft.analyser_documents(documents, build_consigne(date), SCHEMA_VERDICT,
                                              max_tokens_par_document=JETONS_PAR_VERDICT)
        for rang, attente in enumerate(lot, start=1):
            verdict = reponses.get(f"d{rang}") if reponses else None
            if verdict is None:
                yield attente, _sans_verdict("non_evalue_reponse_invalide",
                                             "Réponse de Gemini indisponible, ou hors du format demandé.",
                                             attente.filing)
                continue
            resultat = {
                "verdict": verdict["verdict"], "justification": verdict["justification"],
                "risques_cites": json.dumps(verdict["risques_cites"], ensure_ascii=False),
                "accession_number": attente.filing["accession_number"], "form": attente.filing["form"],
                # Qualité de l'extraction, remontée jusqu'au parquet : un
                # verdict rendu sur le début du document ne porte pas sur la
                # même chose qu'un verdict rendu sur les Items 1A/3/7.
                "extraction_mode": attente.filing.get("extraction_mode"),
                "modele": sft.dernier_modele_utilise(), "from_cache": False,
            }
            if memoire is not None:
                entree = {"cle": cle_memoire(attente.filing["accession_number"], float(attente.row["gap_pct"])),
                          **resultat, "evaluated_at": datetime.now().isoformat(timespec="seconds")}
                memoire[entree["cle"]] = entree
                if output_dir is not None:
                    memoriser(output_dir, entree)
            yield attente, resultat


# ----------------------------------------------------------------------------
# Checkpoint/reprise façon 08_recuperation_options.py
# ----------------------------------------------------------------------------

def _progress_path(output_dir: Path) -> Path:
    return output_dir / "progress_qualitative.json"


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint_qualitative.jsonl"


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


def append_checkpoint(output_dir: Path, row: dict) -> None:
    with _checkpoint_path(output_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")


def load_checkpoint_rows(output_dir: Path) -> list:
    """Une ligne par période : une période refaite après une reprise
    (--resume) écrit son verdict une seconde fois (cf. reprise_jsonl)."""
    return reprise_jsonl.lire_sans_doublons(
        _checkpoint_path(output_dir),
        cle=lambda row: (row.get("symbol"), row.get("period_type"),
                         row.get("fiscal_year"), row.get("fiscal_quarter")),
        journal=logger)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=config.DIR_DCF, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-llm-cache", action="store_true",
        help="Ignore " + MEMOIRE_FILENAME + " et rejuge chaque période (à réserver à un changement "
             "de consigne ou de modèle : chaque requête compte dans le quota).",
    )
    parser.add_argument(
        "--par-requete", type=int, default=PERIODES_PAR_REQUETE,
        help="Filings déposés le même jour envoyés à Gemini dans une seule requête (défaut: "
             "%(default)s).",
    )
    args = parser.parse_args()
    if args.par_requete < 1:
        parser.error("--par-requete doit valoir au moins 1.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not sft.llm_disponible():
        logger.warning(
            "Aucune clé Gemini (%s) : les périodes sans verdict en mémoire seront journalisées "
            "comme 'non_evalue_pas_de_cle_api' (pas d'appel au modèle). %s",
            sft.GEMINI_API_KEY_ENV, sft.aide_cle_absente(),
        )
    else:
        logger.info("Validation qualitative par %s.", sft.description_llm())

    periods = load_signal_periods(limit=args.limit)
    if periods.empty:
        logger.warning("Aucune période à évaluer (DCF_HISTORY_FILE vide ou sans gap_pct exploitable).")
        return

    processed_keys: set = set()
    if args.resume:
        processed_keys = load_progress(args.output_dir)
        logger.info("Reprise : %d périodes déjà traitées.", len(processed_keys))
    else:
        _progress_path(args.output_dir).unlink(missing_ok=True)
        _checkpoint_path(args.output_dir).unlink(missing_ok=True)

    def _key(row: pd.Series) -> str:
        return f"{row['symbol']}|{row.get('period_type')}|{row.get('fiscal_year')}|{row.get('fiscal_quarter')}"

    to_process = [row for _, row in periods.iterrows() if _key(row) not in processed_keys]
    logger.info("%d/%d périodes à évaluer (%d déjà traitées).", len(to_process), len(periods), len(processed_keys))

    memoire: Optional[Dict[str, dict]] = None
    if args.no_llm_cache:
        logger.warning("--no-llm-cache : les périodes déjà jugées seront renvoyées à Gemini.")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        memoire = charger_memoire(args.output_dir)

    since_checkpoint = 0

    def ecrire(row: pd.Series, result: dict, traitee: bool = True) -> None:
        nonlocal since_checkpoint
        append_checkpoint(args.output_dir, {
            "symbol": row["symbol"], "period_type": row.get("period_type"),
            "fiscal_year": row.get("fiscal_year"), "fiscal_quarter": row.get("fiscal_quarter"),
            "filed_date": row["filed_date"], "gap_pct": row["gap_pct"],
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            **result,
        })
        # Une période sans verdict faute de réponse reste à faire : un
        # --resume la redonnera à Gemini.
        if traitee:
            processed_keys.add(_key(row))
        since_checkpoint += 1
        if since_checkpoint >= CHECKPOINT_EVERY:
            save_progress(args.output_dir, processed_keys)
            since_checkpoint = 0

    en_attente: List[PeriodeEnAttente] = []
    try:
        for i, row in enumerate(to_process, start=1):
            key = _key(row)
            # pd.notna et non la seule valeur de vérité : fiscal_quarter est
            # NaN sur les lignes annuelles, et un NaN est "vrai" en Python --
            # le log affichait donc "INTU FY 2016 nan".
            quarter = row.get("fiscal_quarter")
            suffixe = f" {quarter}" if pd.notna(quarter) and quarter else ""
            logger.info(
                "[%d/%d] %s %s %s%s...", i, len(to_process),
                row["symbol"], row.get("period_type"), row.get("fiscal_year"), suffixe,
            )
            try:
                result, attente = preparer_periode(row, memoire)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  -> ECHEC pour %s : %s (période ignorée, on continue)", key, exc)
                result, attente = {"verdict": "non_evalue", "justification": str(exc), "risques_cites": None}, None
            if attente is not None:
                en_attente.append(attente)
            else:
                ecrire(row, result)

        # Les périodes à juger partent ensemble, par lots de même date.
        for attente, result in classer_en_attente(en_attente, memoire, args.output_dir, args.par_requete):
            ecrire(attente.row, result, traitee=result["verdict"] in VERDICTS)
    finally:
        save_progress(args.output_dir, processed_keys)

    if sft.llm_disponible():
        logger.info("Modèle : %s", sft.bilan_llm())

    rows = load_checkpoint_rows(args.output_dir)
    if not rows:
        logger.warning("Aucun résultat produit, pas de fichier de sortie généré.")
        return

    df = pd.DataFrame(rows)
    if "from_cache" in df.columns:
        # Les lignes sans verdict n'ont pas la colonne : sans normalisation, le
        # mélange bool/NaN part en colonne "object" et pyarrow refuse d'inférer
        # un type.
        df["from_cache"] = df["from_cache"].fillna(False).astype(bool)
    if args.limit:
        # Run PARTIEL : il ne remplace que ses propres périodes dans le fichier
        # complet, que le filtre qualitatif du backtest lit (cf. reprise_jsonl).
        df, conservees = reprise_jsonl.fusionner_run_partiel(
            df, config.QUALITATIVE_VALIDATION_FILE,
            ["symbol", "period_type", "fiscal_year", "fiscal_quarter"])
        logger.info(
            "Run partiel (--limit) : %d période(s) de ce run fusionnée(s) dans %s, %d autres "
            "conservées telles quelles.", len(rows), config.QUALITATIVE_VALIDATION_FILE, conservees)
    config.QUALITATIVE_VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.QUALITATIVE_VALIDATION_FILE, index=False, engine="pyarrow")
    logger.info("Validation qualitative sauvegardée : %s (%d lignes).", config.QUALITATIVE_VALIDATION_FILE, len(df))
    logger.info("Répartition des verdicts : %s", df["verdict"].value_counts().to_dict())


if __name__ == "__main__":
    main()
