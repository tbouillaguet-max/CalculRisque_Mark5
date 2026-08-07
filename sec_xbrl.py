"""
Briques communes d'interprétation des faits XBRL "companyfacts", partagées
entre 04_recuperation_10k.py (annuel) et 04b_recuperation_10q.py
(trimestriel). Factorisées ici parce que les deux scripts appliquent
exactement le même raisonnement à deux fenêtres de durée différentes ; les
garder dupliqués revenait à corriger un bug d'un seul côté (c'est déjà la
raison pour laquelle 04b importe compute_derived de 04 plutôt que de le
recopier).

Pourquoi la PÉRIODE d'un fait se lit sur start/end, jamais sur fy/fp
---------------------------------------------------------------------
Dans companyfacts, `fy` et `fp` décrivent LE RAPPORT dans lequel le fait
apparaît, pas la période que le fait mesure. Un 10-K FY2023 publie ses
comparatifs FY2022 et FY2021 : les trois valeurs sortent du JSON avec
fy=2023, fp="FY", form="10-K" ET la même date `filed`. Classer par `fy`
mettait donc trois exercices différents dans le même panier, et le départage
"dépôt le plus récent gagne" était inopérant (comparaison de deux `filed`
strictement égaux) -- c'était le premier fait rencontré dans l'ordre du JSON
qui l'emportait, souvent la période la plus ancienne.

Seuls `start` et `end` décrivent la période réellement mesurée :
    - un poste de FLUX (montant SUR une période : chiffre d'affaires, EBIT,
      capex...) porte les deux, et sa durée dit s'il s'agit d'un trimestre,
      d'un cumul semestriel ou d'un exercice complet ;
    - un poste de BILAN (montant À une date : trésorerie, dette, actions en
      circulation) ne porte que `end`.
L'exercice de rattachement se déduit alors de `end` seul (voir
fiscal_year_of_end) : le rattachement reste COHÉRENT au sein d'une même ligne,
flux et bilan de la même période tombant sur la même étiquette -- c'est ce
dont le DCF a besoin.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

# Flux (montant SUR la période) vs stock/bilan (montant À une date). Cette
# classification pilote deux choses : la fenêtre de durée applicable au fait
# XBRL (un poste de bilan n'a pas de durée), et la façon dont 04b agrège un
# TTM (somme pour un flux, dernière valeur connue pour un stock).
FLOW_METRICS = frozenset({
    "revenue", "ebit", "net_income", "capex", "da", "income_tax_expense", "interest_expense",
})
STOCK_METRICS = frozenset({
    "total_assets", "total_liabilities", "cash", "short_term_debt", "long_term_debt",
    "shares_outstanding", "current_assets", "current_liabilities",
})

# Fenêtres de durée acceptées pour un poste de flux, en jours.
#   - trimestre : 60-100 jours (un "3 mois" réel va de ~89 à ~92 jours ; la
#     fenêtre large absorbe les exercices 52/53 semaines et les décalages de
#     calendrier fiscal) ;
#   - exercice : 330-400 jours (même raisonnement autour de 365, plus les
#     exercices de transition légèrement raccourcis/rallongés).
QUARTER_DURATION_DAYS: Tuple[int, int] = (60, 100)
ANNUAL_DURATION_DAYS: Tuple[int, int] = (330, 400)

# Mois à partir duquel une clôture étiquette l'exercice de son ANNÉE civile ;
# en dessous, l'exercice porte l'année précédente. Voir fiscal_year_of_end.
FISCAL_YEAR_LABEL_MONTH_CUTOFF = 6


def duration_days(start: Optional[str], end: Optional[str]) -> Optional[int]:
    """Durée d'un fait XBRL en jours, ou None si l'une des bornes manque
    (poste de bilan) ou n'est pas une date ISO exploitable."""
    if not start or not end:
        return None
    try:
        return (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
    except ValueError:
        return None


def fiscal_year_of_end(end: Optional[str]) -> Optional[int]:
    """Exercice de rattachement d'un fait, déduit de sa date de FIN de période.

    Règle : année civile de `end`, MOINS UN quand la clôture tombe avant
    juillet.

    Pourquoi pas simplement l'année de `end`. Beaucoup d'émetteurs suivent un
    calendrier 52/53 semaines dont la clôture oscille autour du 31 décembre.
    Johnson & Johnson clôt "le dimanche le plus proche de fin décembre" : son
    exercice 2022 se termine le 1er janvier 2023 et son exercice 2023 le
    31 décembre 2023. Les rattacher tous deux à "2023" les ferait entrer en
    COLLISION -- le départage "dépôt le plus récent gagne" en garderait un
    seul et un exercice entier disparaîtrait sans bruit du parquet. Le
    décalage sous le seuil de juillet sépare les deux (2022 et 2023) et
    redonne au passage l'étiquette que l'entreprise emploie elle-même, y
    compris pour les exercices clos en janvier de la grande distribution.

    Deux clôtures annuelles consécutives étant séparées d'environ 365 jours,
    cette règle produit toujours des étiquettes distinctes et croissantes :
    aucune collision possible.

    Limite connue : un émetteur qui clôt en mai ou juin et appelle cet
    exercice par l'année de sa clôture (Microsoft, dont "FY2023" se termine le
    30 juin 2023) reste étiqueté par cette règle... correctement pour juin
    (seuil inclusif), d'un an trop tôt pour mai. L'étiquette reste unique et
    l'ordre chronologique exact ; seule sa comparaison avec le millésime des
    pairs (06b) en souffre marginalement.

    None si `end` est absent ou malformé."""
    if not end or len(end) < 7:
        return None
    try:
        year, month = int(end[:4]), int(end[5:7])
    except ValueError:
        return None
    return year if month >= FISCAL_YEAR_LABEL_MONTH_CUTOFF else year - 1


def in_duration_window(days: Optional[int], window: Tuple[int, int]) -> bool:
    """Une durée inconnue (None) n'est JAMAIS dans la fenêtre : un fait sans
    start est un poste de bilan, il n'a pas à passer pour un flux annuel."""
    if days is None:
        return False
    low, high = window
    return low <= days <= high


def entries_in_duration_window(entries: Sequence[dict], window: Tuple[int, int]) -> List[dict]:
    """Sous-ensemble des entrées dont la durée tombe dans la fenêtre.
    Les entrées portent la clé "duration_days" (voir duration_days)."""
    return [e for e in entries if in_duration_window(e.get("duration_days"), window)]


def best_entry(entries: Sequence[dict]) -> Optional[dict]:
    """Entrée retenue quand plusieurs décrivent la même période : celle du
    dépôt le plus récent. C'est la règle "restatement gagnant" -- une valeur
    republiée dans un 10-K ultérieur remplace celle d'origine."""
    if not entries:
        return None
    return max(entries, key=lambda e: e.get("filed") or "")


def best_entry_in_duration_window(entries: Sequence[dict], window: Tuple[int, int]) -> Optional[dict]:
    """best_entry restreint aux entrées de la bonne durée. None si aucune
    entrée ne décrit une période de cette longueur -- l'appelant décide alors
    s'il sait reconstruire la valeur autrement (04b soustrait des cumuls) ou
    s'il abandonne la période (04, qui n'a pas de cumul annuel à retrancher)."""
    return best_entry(entries_in_duration_window(entries, window))


def split_tag_spec(tag_spec: str) -> Tuple[str, str]:
    """"dei:EntityCommonStockSharesOutstanding" -> ("dei", "EntityCommon...").
    Un tag sans préfixe est cherché dans "us-gaap", la taxonomie par défaut."""
    if ":" in tag_spec:
        namespace, _, tag = tag_spec.rpartition(":")
        return namespace, tag
    return "us-gaap", tag_spec


def facts_for_tag(facts: dict, tag_spec: str) -> List[dict]:
    """Toutes les entrées brutes d'un tag, unités confondues. Liste vide si
    l'entreprise ne tague jamais cette métrique."""
    entry = facts.get("facts", {}).get(split_tag_spec(tag_spec)[0], {}).get(split_tag_spec(tag_spec)[1])
    if not entry:
        return []
    return [v for unit_values in entry.get("units", {}).values() for v in unit_values]


def normalize_fact(v: dict) -> Optional[dict]:
    """Fait XBRL brut -> {"val", "filed", "start", "end", "duration_days"}.
    None si la valeur ou la date de fin manque : sans `end`, le fait ne peut
    être rattaché à aucune période."""
    val, end = v.get("val"), v.get("end")
    if val is None or not end:
        return None
    start = v.get("start")
    return {
        "val": val,
        "filed": v.get("filed", ""),
        "start": start,
        "end": end,
        "duration_days": duration_days(start, end),
    }


# Tags de PAGE DE GARDE : leur `end` n'est pas une fin de période comptable
# mais la date à laquelle l'émetteur arrête le compte (typiquement quelques
# semaines AVANT le dépôt, donc APRÈS la clôture de l'exercice décrit). Les
# rattacher à l'année de leur `end` les enverrait sur l'exercice suivant --
# pour un 10-K FY2023 déposé en février 2024, la page de garde est datée de
# février 2024. Ils sont donc "recalés" sur la dernière fin d'exercice
# réellement observée avant cette date (voir snap_to_period_end).
COVER_PAGE_TAGS = frozenset({"dei:EntityCommonStockSharesOutstanding"})


def snap_to_period_end(end: Optional[str], period_ends: Sequence[str]) -> Optional[str]:
    """Fin d'exercice observée la plus récente qui soit <= `end`.

    Recale une date de page de garde sur l'exercice qu'elle décrit vraiment.
    Comme la cible est une date de fin RÉELLEMENT extraite des autres
    métriques, le rattachement reste cohérent avec elles quelle que soit la
    convention de calendrier fiscal de l'entreprise -- y compris pour un
    exercice clos début janvier, où l'année de `end` et l'année que
    l'entreprise appelle "son exercice" diffèrent."""
    if not end:
        return None
    candidates = [p for p in period_ends if p and p <= end]
    return max(candidates) if candidates else None


def collect_period_end_dates(facts: dict, tags_by_metric: Dict[str, Sequence[str]]) -> List[str]:
    """Dates de fin d'exercice réellement observées pour cette entreprise,
    tous tags confondus SAUF ceux de page de garde (qui sont précisément ceux
    qu'on cherche à recaler dessus)."""
    ends = set()
    for metric, tags in tags_by_metric.items():
        is_flow = metric in FLOW_METRICS
        for tag_spec in tags:
            if tag_spec in COVER_PAGE_TAGS:
                continue
            for entries in collect_annual_entries(facts, tag_spec, is_flow).values():
                ends.update(e["end"] for e in entries if e["end"])
    return sorted(ends)


def collect_cover_page_entries(
    facts: dict, tag_spec: str, period_ends: Sequence[str], forms: Sequence[str] = ("10-K",),
) -> Dict[int, List[dict]]:
    """{année de l'exercice recalé: [faits normalisés]} pour un tag de page de
    garde. Aucune durée à filtrer (c'est un instant), et aucun risque de
    comparatif : un filing ne porte qu'UNE valeur de page de garde.

    Un fait qu'on ne sait recaler sur aucune fin d'exercice connue est
    ignoré : mieux vaut une colonne shares_outstanding vide (la ligne sera
    écartée par 07 avec un avertissement) qu'une valeur posée sur le mauvais
    exercice."""
    by_year: Dict[int, List[dict]] = {}
    for v in facts_for_tag(facts, tag_spec):
        if v.get("form") not in forms:
            continue
        normalized = normalize_fact(v)
        if normalized is None:
            continue
        snapped = snap_to_period_end(normalized["end"], period_ends)
        year = fiscal_year_of_end(snapped)
        if year is None:
            continue
        by_year.setdefault(year, []).append(normalized)
    return by_year


def collect_annual_entries(
    facts: dict, tag_spec: str, metric_is_flow: bool, forms: Sequence[str] = ("10-K",),
) -> Dict[int, List[dict]]:
    """{année de `end`: [faits normalisés]} pour UN tag, en ne gardant que ce
    qui décrit réellement un exercice complet :
        - poste de FLUX : durée dans ANNUAL_DURATION_DAYS (un cumul 6 ou 9
          mois, ou un trimestre isolé, est écarté) ;
        - poste de BILAN : pas de durée à vérifier, `end` suffit.

    `forms` restreint aux faits publiés dans un rapport ANNUEL. `fp` n'est
    volontairement PAS testé : c'est une propriété du rapport, pas du fait
    (voir la docstring du module) -- c'est précisément ce test qui laissait
    passer trois exercices sous la même clé."""
    by_year: Dict[int, List[dict]] = {}
    for v in facts_for_tag(facts, tag_spec):
        if v.get("form") not in forms:
            continue
        normalized = normalize_fact(v)
        if normalized is None:
            continue
        if metric_is_flow and not in_duration_window(normalized["duration_days"], ANNUAL_DURATION_DAYS):
            continue
        year = fiscal_year_of_end(normalized["end"])
        if year is None:
            continue
        by_year.setdefault(year, []).append(normalized)
    return by_year
