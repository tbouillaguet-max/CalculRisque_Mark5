"""
Extrait les données financières TRIMESTRIELLES (10-Q, + 10-K pour le 4e
trimestre implicite) de l'univers US, via la même API XBRL "companyfacts" de
la SEC que 04_recuperation_10k.py (mêmes tags, même endpoint -- un 10-K
dépose AUSSI ses valeurs annuelles dans ce même JSON, réutilisées ici pour
compléter le 4e trimestre, qui n'existe jamais comme 10-Q à part).

Objectif : que 07_calcul_dcf.py (via 05/06b) puisse recalculer la
valorisation chaque trimestre plutôt qu'une fois par an, sans que la
saisonnalité d'un seul trimestre ne fausse les multiples (voir "TTM vs
trimestre brut" ci-dessous) et sans jamais utiliser une donnée avant sa date
réelle de publication (voir "Point-in-time" ci-dessous).

TTM vs trimestre brut
----------------------
Un trimestre BRUT (ex: Q4 seul) peut être trompeur pour un multiple de
valorisation : un retailer fait l'essentiel de son EBITDA au Q4 (fêtes de
fin d'année), une entreprise cyclique peut avoir un trimestre exceptionnel ou
franchement mauvais sans que ça reflète sa rentabilité "normale". Le TTM
("Trailing Twelve Months", somme glissante des 4 derniers trimestres
disponibles) lisse cette saisonnalité tout en restant à jour bien plus vite
qu'un exercice annuel complet -- c'est ce que produit ce script en sortie
principale (config.FINANCIALS_TTM_FILE), le détail par trimestre brut étant
gardé à part (config.FINANCIALS_QUARTERLY_FILE) pour audit/debug.

Le TTM ne s'applique de la même façon à TOUTES les métriques :
    - Métriques de FLUX (revenue, ebit, net_income, capex, da,
      income_tax_expense, interest_expense) : le TTM est leur SOMME sur les
      4 derniers trimestres (ce sont des montants "sur une période").
    - Métriques de STOCK/bilan (cash, total_assets, total_liabilities,
      short_term_debt, long_term_debt, shares_outstanding, current_assets,
      current_liabilities) : le TTM est la DERNIÈRE valeur connue (le solde
      le plus récent), PAS une somme -- ce sont des montants "à une date",
      additionner 4 trimestres de "Cash" n'aurait aucun sens.

Reconstruction du trimestre discret (le vrai casse-tête XBRL)
---------------------------------------------------------------
XBRL companyfacts ne distingue pas explicitement "ce trimestre seul" de
"cumul depuis le 1er janvier" : les deux coexistent sous le même tag, avec
des dates start/end différentes (beaucoup d'entreprises présentent leur
compte de résultat en colonnes "3 mois" ET "6/9 mois" dans un même 10-Q,
tout tagué pareil). Règle utilisée ici (standard chez les fournisseurs de
données fondamentales) :
    1. Si une entrée de durée ~90 jours existe pour ce (année fiscale,
       trimestre) -- le trimestre déjà présenté seul -- on la prend telle
       quelle.
    2. Sinon (fréquent pour le tableau de flux de trésorerie -- capex, D&A --
       jamais présenté en 3 mois discrets dans un 10-Q, seulement en cumul) :
       delta = cumul de ce trimestre moins cumul du trimestre précédent DÉJÀ
       résolu de la même année fiscale.
    3. Le 4e trimestre (jamais déposé comme 10-Q à part) = valeur annuelle du
       10-K (fp="FY") moins (Q1+Q2+Q3) discrets déjà résolus.
Les métriques de stock n'ont pas besoin de cette discrétisation : ce sont
déjà des soldes à une date (fp="FY" du 10-K sert directement de valeur Q4).

Point-in-time
-------------
Chaque ligne (trimestre brut ou TTM) porte un filed_date = date de DÉPÔT SEC
réelle (jamais la date de fin de période) : il y a un délai de ~30 à 45 jours
entre la fin d'un trimestre et la publication du 10-Q correspondant. Pour un
TTM, filed_date = filed_date du PLUS RÉCENT des 4 trimestres qui le
composent (le TTM n'est "connaissable" qu'à partir de ce dépôt-là, même s'il
inclut des trimestres plus anciens déjà publiés). Un backtest qui utilise
DCF_HISTORY_FILE/VALORISATION_COMBINEE_FILE (07/06b) ne doit donc jamais
utiliser une ligne dont filed_date est postérieure à la date simulée -- même
principe déjà en place pour l'annuel (04_recuperation_10k.py).

Prérequis / usage : identiques à 04_recuperation_10k.py (voir sa docstring).
    python 04b_recuperation_10q.py
    python 04b_recuperation_10q.py --limit 10
    python 04b_recuperation_10q.py --ticker AAPL
    python 04b_recuperation_10q.py --resume       # reprendre un run interrompu
    python 04b_recuperation_10q.py --refresh-days 7
    python 04b_recuperation_10q.py --force-refresh
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import config
import sec_http
import sec_xbrl

# ----------------------------------------------------------------------------
# Fetch SEC : dupliqué à l'identique de 04_recuperation_10k.py (même pattern
# que 03b duplique le sien plutôt que d'importer 03) -- cadence de refresh
# différente (trimestrielle vs annuelle), scripts indépendants par conception
# (voir l'orchestrateur, run_pipeline_quarterly.py, qui les lance séparément).
# ----------------------------------------------------------------------------
FETCH_STATE_FILE = config.DIR_FINANCIALS / "fetch_state_10q.json"
REFRESH_DAYS_DEFAULT = 7

TICKERS_JSON_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Tags et classification flux/stock : définis dans sec_xbrl, partagés avec 04.
# Les deux scripts en avaient une copie que ce commentaire demandait de
# "garder en synchronisation" à la main -- exactement le genre d'invariant qui
# se perd au premier tag ajouté d'un seul côté.
XBRL_TAGS: Dict[str, List[str]] = sec_xbrl.XBRL_TAGS

# Flux (montant SUR la période -> le TTM en fait la SOMME) vs stock/bilan
# (montant À une date -> le TTM en garde la DERNIÈRE valeur connue). Voir
# section "TTM vs trimestre brut" de la docstring. Définis dans sec_xbrl
# (module partagé avec 04) plutôt qu'ici : 04 a besoin de la même
# classification depuis le passage au rattachement par `end`, et deux copies
# divergeraient au premier tag ajouté d'un seul côté.
FLOW_METRICS = set(sec_xbrl.FLOW_METRICS)
STOCK_METRICS = set(sec_xbrl.STOCK_METRICS)
assert FLOW_METRICS | STOCK_METRICS == set(XBRL_TAGS), "Chaque tag XBRL doit être classé flux XOR stock."

QUARTER_SEQUENCE = ("Q1", "Q2", "Q3", "Q4")
TTM_WINDOW = 4  # nombre de trimestres sommés/considérés pour un TTM

CHECKPOINT_EVERY = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# compute_derived encode les formules dérivées (net_debt, ebitda, fcf,
# cost_of_debt, tax_rate) : importé depuis 04 plutôt que dupliqué, pour ne
# JAMAIS diverger d'un correctif futur fait uniquement côté 04 (cf. le bug de
# double-comptage du cash déjà corrigé une fois côté 07_calcul_dcf.py --
# exactement le genre de correctif qui doit s'appliquer identiquement ici).
_module_10k = importlib.import_module("04_recuperation_10k")
compute_derived = _module_10k.compute_derived


def _get(url: str) -> Optional[dict]:
    """Passe désormais par sec_http : limiteur de débit GLOBAL partagé avec
    04/04c (ce script envoyait ses requêtes sans throttle commun, si bien que
    deux scripts lancés en parallèle doublaient le débit vers la SEC) et
    réessais sur erreur transitoire (il n'en avait aucun : un 503 passager
    faisait perdre le ticker pour tout le run)."""
    return sec_http.get_json_or_none(url)


def get_cik_map() -> Dict[str, str]:
    data = _get(TICKERS_JSON_URL)
    if not data:
        return {}
    return {entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in data.values()}


# ----------------------------------------------------------------------------
# Discrétisation trimestrielle (voir docstring)
# ----------------------------------------------------------------------------

def collect_period_entries(facts: dict, tags: List[str]) -> Dict[Tuple[int, str], List[dict]]:
    """Pour une métrique (plusieurs tags candidats, ordre de préférence),
    regroupe TOUTES les entrées XBRL par (année fiscale, trimestre fiscal),
    form in {10-Q, 10-K}, fp in {Q1,Q2,Q3,FY} -- y compris plusieurs entrées
    pour la même période si le filing présente plusieurs durées (ex: "3 mois"
    ET "6 mois" sous le même tag). Le premier tag (dans l'ordre de la liste)
    à fournir au moins une entrée pour une période donnée en devient
    "propriétaire" pour cette période -- même règle de priorité par tag que
    04_recuperation_10k.py::extract_annual_values, étendue ici période par
    période plutôt qu'année par année."""
    all_facts = facts.get("facts", {})
    owner_priority: Dict[Tuple[int, str], int] = {}
    entries_by_period: Dict[Tuple[int, str], List[dict]] = {}

    for priority, tag_spec in enumerate(tags):
        namespace, _, tag = tag_spec.rpartition(":") if ":" in tag_spec else ("us-gaap", "", tag_spec)
        entry = all_facts.get(namespace, {}).get(tag)
        if not entry:
            continue
        for unit_values in entry.get("units", {}).values():
            for v in unit_values:
                if v.get("form") not in ("10-Q", "10-K"):
                    continue
                fp = v.get("fp")
                if fp not in ("Q1", "Q2", "Q3", "FY"):
                    continue
                fy, filed, val = v.get("fy"), v.get("filed", ""), v.get("val")
                if fy is None or val is None:
                    continue

                key = (fy, fp)
                current_owner = owner_priority.get(key)
                if current_owner is not None and priority > current_owner:
                    continue  # un tag mieux classé possède déjà cette période
                owner_priority.setdefault(key, priority)

                start, end = v.get("start"), v.get("end")
                entries_by_period.setdefault(key, []).append({
                    "val": val, "filed": filed, "end": end,
                    "duration_days": sec_xbrl.duration_days(start, end),
                })

    return entries_by_period


def resolve_stock_value(entries: List[dict]) -> Optional[Tuple[float, str, Optional[str]]]:
    """Métrique de bilan : pas de discrétisation, juste la valeur (dépôt le
    plus récent en cas de doublon, ex: restatement) -- même tie-break que 04,
    désormais littéralement la même fonction (sec_xbrl.best_entry)."""
    best = sec_xbrl.best_entry(entries)
    if best is None:
        return None
    return best["val"], best["filed"], best.get("end")


def resolve_flow_quarter_or_annual(entries: List[dict]) -> Optional[Tuple[float, str, str, Optional[str]]]:
    """Résout la MEILLEURE entrée disponible pour un (fy, fp) donné : une
    valeur "trimestre seul" (~90 jours) si disponible telle quelle, sinon la
    meilleure entrée cumulative (peu importe sa durée -- 6 mois, 9 mois, ou
    l'année complète pour fp="FY"). Retourne (valeur, filed, kind, end) où
    kind = "direct" (déjà discret, à utiliser tel quel) ou "cumulative" (à
    charge de l'appelant de soustraire le(s) trimestre(s) précédent(s)).

    La fenêtre de durée trimestrielle et le tie-break "dépôt le plus récent"
    viennent de sec_xbrl, partagés avec 04 (qui applique la même mécanique à
    sa fenêtre annuelle) -- ils étaient écrits ici en premier."""
    if not entries:
        return None
    best = sec_xbrl.best_entry_in_duration_window(entries, sec_xbrl.QUARTER_DURATION_DAYS)
    if best is not None:
        return best["val"], best["filed"], "direct", best.get("end")
    best = sec_xbrl.best_entry(entries)
    return best["val"], best["filed"], "cumulative", best.get("end")


def _resolve_discrete_quarter(
    entries_by_period: Dict[Tuple[int, str], List[dict]], fy: int, fp: str,
    resolved_so_far: Dict[str, Tuple[float, str]],
) -> Optional[Tuple[float, str, Optional[str]]]:
    """Valeur DISCRÈTE (3 mois) d'un trimestre pour une métrique de flux.
    resolved_so_far : trimestres déjà résolus de la MÊME année fiscale et de
    la MÊME métrique (Q1 avant Q2 avant Q3 avant Q4 -- l'appelant doit
    respecter cet ordre, cf. build_quarterly_table)."""
    if fp == "Q4":
        fy_resolved = resolve_flow_quarter_or_annual(entries_by_period.get((fy, "FY"), []))
        if fy_resolved is None:
            return None
        fy_val, fy_filed, _kind, fy_end = fy_resolved
        parts = [resolved_so_far.get(q) for q in ("Q1", "Q2", "Q3")]
        if any(p is None for p in parts):
            return None  # Q4 non calculable sans les 3 trimestres précédents déjà résolus
        q_sum = sum(p[0] for p in parts)
        q4_filed = max([fy_filed] + [p[1] for p in parts])
        return fy_val - q_sum, q4_filed, fy_end

    resolved = resolve_flow_quarter_or_annual(entries_by_period.get((fy, fp), []))
    if resolved is None:
        return None
    val, filed, kind, end = resolved
    if kind == "direct":
        return val, filed, end

    prior_fps = {"Q2": ("Q1",), "Q3": ("Q1", "Q2")}.get(fp, ())
    prior = [resolved_so_far.get(p) for p in prior_fps]
    if any(p is None for p in prior):
        return None  # cumul reçu mais trimestre(s) précédent(s) pas encore résolu(s)
    prior_sum = sum(p[0] for p in prior)
    discrete_filed = max([filed] + [p[1] for p in prior])
    return val - prior_sum, discrete_filed, end


def build_quarterly_table(facts: dict, symbol: str, cik: str) -> pd.DataFrame:
    """Une ligne par (année fiscale, trimestre fiscal Q1-Q4) réellement
    résolue, avec les colonnes canoniques (mêmes noms que 04), filed_date, et
    period_end (date de fin de trimestre, dérivée des métriques de bilan --
    None si seules des métriques de flux ont pu être résolues ce trimestre)."""
    flow_entries = {m: collect_period_entries(facts, XBRL_TAGS[m]) for m in FLOW_METRICS}
    stock_entries = {m: collect_period_entries(facts, XBRL_TAGS[m]) for m in STOCK_METRICS}

    fiscal_years = sorted({
        fy for entries in list(flow_entries.values()) + list(stock_entries.values())
        for (fy, _fp) in entries
    })

    rows = []
    for fy in fiscal_years:
        resolved_flow: Dict[str, Dict[str, Tuple[float, str]]] = {m: {} for m in FLOW_METRICS}
        for fp in QUARTER_SEQUENCE:
            row = {"symbol": symbol, "cik": cik, "fiscal_year": fy, "fiscal_quarter": fp}
            filed_dates: List[str] = []
            period_ends: List[str] = []

            for metric in FLOW_METRICS:
                resolved = _resolve_discrete_quarter(flow_entries[metric], fy, fp, resolved_flow[metric])
                if resolved is None:
                    row[metric] = None
                    continue
                val, filed, end = resolved
                resolved_flow[metric][fp] = (val, filed)
                row[metric] = val
                filed_dates.append(filed)
                if end:
                    period_ends.append(end)

            stock_fp = "FY" if fp == "Q4" else fp
            for metric in STOCK_METRICS:
                resolved = resolve_stock_value(stock_entries[metric].get((fy, stock_fp), []))
                if resolved is None:
                    row[metric] = None
                    continue
                val, filed, end = resolved
                row[metric] = val
                filed_dates.append(filed)
                if end:
                    period_ends.append(end)

            if not filed_dates:
                continue  # rien de résolu pour ce trimestre : pas de ligne (plutôt qu'une ligne 100% vide)

            row["filed_date"] = max(filed_dates)
            row["period_end"] = max(period_ends) if period_ends else None
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    # compute_derived prend le DataFrame ENTIER (il est vectorisé côté 04).
    # L'appel d'origine était `df.apply(compute_derived, axis=1)`, qui lui
    # passait chaque LIGNE sous forme de Series : `series.columns` n'existe
    # pas, donc AttributeError -- et comme extract_quarterly_for_ticker est
    # enveloppé dans un try/except par ticker, l'échec se contentait
    # d'afficher "ECHEC pour X" et TOUS les tickers échouaient. Résultat :
    # financials_ttm.parquet n'était jamais produit, et tout le pipeline
    # retombait sur l'annuel seul.
    return compute_derived(pd.DataFrame(rows))


def compute_ttm(quarterly: pd.DataFrame) -> pd.DataFrame:
    """TTM = somme glissante des TTM_WINDOW derniers trimestres discrets
    (métriques de flux) + dernière valeur connue (métriques de stock), pour
    chaque symbole -- voir "TTM vs trimestre brut" dans la docstring.
    filed_date du TTM = filed_date du plus récent des trimestres qui le
    composent (le TTM n'est connaissable qu'à partir de ce dépôt-là)."""
    if quarterly.empty:
        return pd.DataFrame()

    ttm_rows = []
    for symbol, group in quarterly.sort_values(["symbol", "period_end", "fiscal_year", "fiscal_quarter"]).groupby("symbol"):
        group = group.reset_index(drop=True)
        for i in range(TTM_WINDOW - 1, len(group)):
            window = group.iloc[i - TTM_WINDOW + 1: i + 1]
            latest = window.iloc[-1]
            row = {
                "symbol": symbol, "cik": latest["cik"],
                "fiscal_year": latest["fiscal_year"], "fiscal_quarter": latest["fiscal_quarter"],
                "period_end": latest["period_end"],
            }
            for metric in FLOW_METRICS:
                vals = window[metric]
                row[metric] = vals.sum(min_count=TTM_WINDOW)  # None si un seul trimestre manque cette métrique
            for metric in STOCK_METRICS:
                row[metric] = latest[metric]
            # Provenance de l'EBIT : reprise du dernier trimestre de la
            # fenêtre. Ce n'est ni un flux (rien à sommer) ni un stock, mais
            # elle doit suivre l'agrégat -- sinon compute_derived, revoyant un
            # ebit déjà renseigné, l'étiquetterait "operating_income" alors
            # qu'il a pu être reconstruit trimestre par trimestre.
            if "ebit_source" in window.columns:
                sources = window["ebit_source"].dropna()
                row["ebit_source"] = sources.iloc[-1] if len(sources) else None
            row["filed_date"] = window["filed_date"].max()
            row["ttm_quarters_used"] = ", ".join(f"{y}{q}" for y, q in zip(window["fiscal_year"], window["fiscal_quarter"]))
            ttm_rows.append(row)

    if not ttm_rows:
        return pd.DataFrame()

    # Même correctif que build_quarterly_table : le DataFrame entier, pas
    # ligne par ligne. Les métriques dérivées sont RErecalculées sur les
    # agrégats TTM (et non sommées trimestre par trimestre), ce qui est le
    # comportement voulu pour des ratios comme tax_rate ou cost_of_debt.
    return compute_derived(pd.DataFrame(ttm_rows))


# ----------------------------------------------------------------------------
# Orchestration par ticker (throttle + checkpoint/reprise façon 08)
# ----------------------------------------------------------------------------

def extract_quarterly_for_ticker(ric: str, cik: str) -> pd.DataFrame:
    facts = _get(COMPANYFACTS_URL.format(cik=cik))
    if not facts:
        logger.warning("Aucune donnée companyfacts pour %s (CIK %s)", ric, cik)
        return pd.DataFrame()
    symbol = config.to_ib_symbol(ric)
    df = build_quarterly_table(facts, symbol, cik)
    if df.empty:
        logger.warning("Aucun trimestre 10-Q/10-K exploitable pour %s", ric)
    return df


def load_fetch_state(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("État de suivi illisible (%s), on repart de zéro.", e)
        return {}


def save_fetch_state(path: Path, state: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def should_skip(symbol: str, existing: pd.DataFrame, state: Dict[str, str], refresh_days: int) -> bool:
    if symbol not in state:
        return False
    if existing.empty or symbol not in set(existing["symbol"]):
        return False
    last_fetched = datetime.fromisoformat(state[symbol])
    return (datetime.now() - last_fetched).days <= refresh_days


def _progress_path(output_dir: Path) -> Path:
    return output_dir / "progress_10q.json"


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint_10q.jsonl"


def load_progress(output_dir: Path) -> set[str]:
    path = _progress_path(output_dir)
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")).get("processed", []))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Fichier de progression illisible (%s), on repart de zéro.", e)
        return set()


def save_progress(output_dir: Path, processed: set[str]) -> None:
    path = _progress_path(output_dir)
    tmp = path.with_suffix(".json.tmp")
    payload = {"processed": sorted(processed), "updated_at": datetime.now().isoformat(timespec="seconds")}
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_checkpoint(output_dir: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with _checkpoint_path(output_dir).open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")


def load_checkpoint_rows(output_dir: Path) -> List[dict]:
    path = _checkpoint_path(output_dir)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", type=Path, default=config.UNIVERSE_FILE)
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--output-dir", default=config.DIR_FINANCIALS, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refresh-days", type=int, default=REFRESH_DAYS_DEFAULT)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    if sec_http.require_contact_email(logger) is None:
        sys.exit(1)

    if args.ticker:
        symbols = [args.ticker.upper()]
    else:
        universe = pd.read_csv(args.tickers, encoding="utf-8-sig")
        symbols = universe["RIC"].dropna().unique().tolist()
        if args.limit:
            symbols = symbols[: args.limit]

    existing_quarterly = pd.read_parquet(config.FINANCIALS_QUARTERLY_FILE) if config.FINANCIALS_QUARTERLY_FILE.exists() else pd.DataFrame()

    state = load_fetch_state(FETCH_STATE_FILE)
    processed_keys: set[str] = set()
    if args.resume:
        processed_keys = load_progress(args.output_dir)
        logger.info("Reprise : %d tickers déjà traités dans ce run interrompu.", len(processed_keys))
    else:
        _progress_path(args.output_dir).unlink(missing_ok=True)
        _checkpoint_path(args.output_dir).unlink(missing_ok=True)

    to_query = [
        t for t in symbols
        if config.to_ib_symbol(t) not in processed_keys
        and (args.force_refresh or not should_skip(config.to_ib_symbol(t), existing_quarterly, state, args.refresh_days))
    ]
    skip_count = len(symbols) - len(to_query) - len(processed_keys & {config.to_ib_symbol(t) for t in symbols})

    if not to_query:
        logger.info("Tous les tickers demandés sont déjà à jour ou déjà traités dans ce run : rien à faire.")
    else:
        logger.info("%d/%d tickers à interroger (%d déjà à jour, %d déjà traités).", len(to_query), len(symbols), skip_count, len(processed_keys))
        logger.info("Récupération des CIK SEC...")
        cik_map = get_cik_map()

        ok_count, fail_count = 0, 0
        since_checkpoint = 0
        now_iso = datetime.now().isoformat(timespec="seconds")

        try:
            for i, ticker in enumerate(to_query, start=1):
                symbol = config.to_ib_symbol(ticker)
                cik = cik_map.get(ticker.upper())
                if not cik:
                    logger.warning("[%d/%d] CIK introuvable pour %s, ignoré.", i, len(to_query), ticker)
                    fail_count += 1
                    processed_keys.add(symbol)
                    continue

                logger.info("[%d/%d] %s (CIK %s)...", i, len(to_query), ticker, cik)
                # Une erreur sur CE ticker (réseau, JSON malformé...) ne doit
                # jamais faire perdre la progression déjà acquise sur les
                # précédents -- capturée et journalisée ici, comme
                # 08_recuperation_options.py le fait par ticker (le script
                # continue avec le suivant plutôt que de tout arrêter).
                try:
                    df_ticker = extract_quarterly_for_ticker(ticker, cik)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("  -> ECHEC pour %s : %s (ticker ignoré, on continue)", ticker, exc)
                    df_ticker = pd.DataFrame()

                state[symbol] = now_iso
                processed_keys.add(symbol)
                if df_ticker.empty:
                    fail_count += 1
                else:
                    append_checkpoint(args.output_dir, df_ticker.to_dict("records"))
                    ok_count += 1

                since_checkpoint += 1
                if since_checkpoint >= CHECKPOINT_EVERY:
                    save_progress(args.output_dir, processed_keys)
                    save_fetch_state(FETCH_STATE_FILE, state)
                    since_checkpoint = 0
        finally:
            # Toujours exécuté, même sur une interruption inattendue
            # (Ctrl+C, erreur non prévue par le try/except ci-dessus) :
            # --resume doit pouvoir repartir de l'état réel, pas d'un
            # checkpoint périodique potentiellement dépassé.
            save_progress(args.output_dir, processed_keys)
            save_fetch_state(FETCH_STATE_FILE, state)

        logger.info("Terminé. OK: %d | Échecs: %d", ok_count, fail_count)

    new_rows = load_checkpoint_rows(args.output_dir)
    if not new_rows:
        if existing_quarterly.empty:
            logger.warning("Aucune donnée trimestrielle récupérée, pas de fichier de sortie généré.")
        else:
            logger.info("Rien de nouveau : fichiers existants conservés tels quels.")
        return

    df_new = pd.DataFrame(new_rows)
    quarterly = pd.concat([existing_quarterly, df_new], ignore_index=True) if not existing_quarterly.empty else df_new
    quarterly = (
        quarterly.sort_values(["symbol", "fiscal_year", "fiscal_quarter"])
        .drop_duplicates(subset=["symbol", "fiscal_year", "fiscal_quarter"], keep="last")
        .reset_index(drop=True)
    )
    quarterly["period_type"] = "Q"
    config.FINANCIALS_QUARTERLY_FILE.parent.mkdir(parents=True, exist_ok=True)
    quarterly.to_parquet(config.FINANCIALS_QUARTERLY_FILE, index=False, engine="pyarrow")
    logger.info("Trimestres bruts sauvegardés : %s (%d lignes, %d entreprises).",
                config.FINANCIALS_QUARTERLY_FILE, len(quarterly), quarterly["symbol"].nunique())

    ttm = compute_ttm(quarterly.drop(columns=["period_type"]))
    if ttm.empty:
        logger.warning("Aucun TTM calculable (moins de %d trimestres consécutifs disponibles par entreprise).", TTM_WINDOW)
        return
    ttm["period_type"] = "TTM"
    config.FINANCIALS_TTM_FILE.parent.mkdir(parents=True, exist_ok=True)
    ttm.to_parquet(config.FINANCIALS_TTM_FILE, index=False, engine="pyarrow")
    logger.info("TTM sauvegardés : %s (%d lignes, %d entreprises).",
                config.FINANCIALS_TTM_FILE, len(ttm), ttm["symbol"].nunique())


if __name__ == "__main__":
    main()
