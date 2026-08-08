"""
Génère les fixtures companyfacts réduites utilisées par
tests/test_extraction_xbrl.py.

Pourquoi un fixture plutôt qu'un appel réseau : le test doit tourner dans
n'importe quel environnement (CI hors ligne, poste sans accès à data.sec.gov,
et notamment celui où ce correctif a été écrit -- l'accès sortant y était
bloqué). Le JSON produit reproduit la STRUCTURE exacte de l'API companyfacts,
en particulier le piège que corrige A3 : un 10-K publie ses deux exercices
comparatifs sous le MÊME fy, le MÊME fp et la MÊME date `filed` que
l'exercice courant.

Provenance des montants : comptes consolidés publiés (10-K, "Consolidated
Statements of Operations" / "Consolidated Balance Sheets"), recopiés à la
main. Ils n'ont PAS pu être re-vérifiés automatiquement dans l'environnement
de rédaction (data.sec.gov inaccessible) -- recoupe-les si un chiffre te
paraît douteux. Ce que le test vérifie n'est de toute façon pas leur
exactitude comptable mais le RATTACHEMENT : que chaque valeur atterrisse sur
l'exercice qui l'a produite plutôt que sur un autre.

Usage (à relancer seulement si tu modifies les fixtures) :
    python tests/fixtures/build_companyfacts_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Description compacte des trois entreprises.
#
# "exercices" : (fin_de_periode, debut_de_periode, date_de_depot) par exercice.
#   AAPL clôt fin septembre, XOM au 31 décembre, JNJ le dimanche le plus
#   proche du 31 décembre -- ce dernier tombe donc DEUX FOIS SUR TROIS début
#   janvier de l'année suivante, ce qui est exactement le cas limite que la
#   docstring de sec_xbrl documente (l'exercice que l'entreprise appelle 2021
#   se rattache à l'année civile 2022, celle de sa date de fin).
# ---------------------------------------------------------------------------
COMPANIES = {
    "AAPL": {
        "cik": 320193,
        "entity_name": "Apple Inc.",
        "revenue_tag": "RevenueFromContractWithCustomerExcludingAssessedTax",
        "periods": {
            2021: {"start": "2020-09-27", "end": "2021-09-25", "filed": "2021-10-29"},
            2022: {"start": "2021-09-26", "end": "2022-09-24", "filed": "2022-10-28"},
            2023: {"start": "2022-09-25", "end": "2023-09-30", "filed": "2023-11-03"},
        },
        "flows": {
            "revenue":    {2021: 365_817e6, 2022: 394_328e6, 2023: 383_285e6},
            "ebit":       {2021: 108_949e6, 2022: 119_437e6, 2023: 114_301e6},
            "net_income": {2021:  94_680e6, 2022:  99_803e6, 2023:  96_995e6},
        },
        "instants": {
            "total_assets": {2021: 351_002e6, 2022: 352_755e6, 2023: 352_583e6},
            "cash":         {2021:  34_940e6, 2022:  23_646e6, 2023:  29_965e6},
        },
        # Page de garde : date d'arrêté du compte d'actions, postérieure à la
        # clôture de l'exercice décrit.
        "cover_shares": {
            2021: {"as_of": "2021-10-15", "val": 16_406_397_000},
            2022: {"as_of": "2022-10-14", "val": 15_908_118_000},
            2023: {"as_of": "2023-10-20", "val": 15_552_752_000},
        },
    },
    "JNJ": {
        "cik": 200406,
        "entity_name": "Johnson & Johnson",
        "revenue_tag": "Revenues",
        "periods": {
            2021: {"start": "2021-01-04", "end": "2022-01-02", "filed": "2022-02-17"},
            2022: {"start": "2022-01-03", "end": "2023-01-01", "filed": "2023-02-16"},
            2023: {"start": "2023-01-02", "end": "2023-12-31", "filed": "2024-02-16"},
        },
        "flows": {
            "revenue": {2021: 93_775e6, 2022: 94_943e6, 2023: 85_159e6},
        },
        "instants": {
            "total_assets": {2021: 182_018e6, 2022: 187_378e6, 2023: 167_558e6},
        },
        "cover_shares": {
            2021: {"as_of": "2022-02-10", "val": 2_631_000_000},
            2022: {"as_of": "2023-02-09", "val": 2_594_000_000},
            2023: {"as_of": "2024-02-09", "val": 2_407_000_000},
        },
    },
    "XOM": {
        "cik": 34088,
        "entity_name": "Exxon Mobil Corporation",
        "revenue_tag": "Revenues",
        "periods": {
            2021: {"start": "2021-01-01", "end": "2021-12-31", "filed": "2022-02-23"},
            2022: {"start": "2022-01-01", "end": "2022-12-31", "filed": "2023-02-22"},
            2023: {"start": "2023-01-01", "end": "2023-12-31", "filed": "2024-02-28"},
        },
        "flows": {
            "revenue":    {2021: 285_640e6, 2022: 413_680e6, 2023: 344_582e6},
            "net_income": {2021:  23_040e6, 2022:  55_740e6, 2023:  36_010e6},
        },
        "instants": {
            "total_assets": {2021: 338_923e6, 2022: 369_067e6, 2023: 376_317e6},
        },
        "cover_shares": {
            2021: {"as_of": "2022-01-31", "val": 4_236_000_000},
            2022: {"as_of": "2023-01-31", "val": 4_099_000_000},
            2023: {"as_of": "2024-01-31", "val": 3_953_000_000},
        },
    },
}

# Tag us-gaap correspondant à chaque métrique du schéma canonique de 04.
US_GAAP_TAG = {
    "ebit": "OperatingIncomeLoss",
    "net_income": "NetIncomeLoss",
    "total_assets": "Assets",
    "cash": "CashAndCashEquivalentsAtCarryingValue",
}
COVER_TAG = "EntityCommonStockSharesOutstanding"

# Un 10-K reprend les DEUX exercices précédents au compte de résultat et le
# précédent au bilan -- c'est de là que vient tout le problème corrigé par A3.
COMPARATIVE_INCOME_YEARS = 3
COMPARATIVE_BALANCE_YEARS = 2


def _fact(val, filed, fy, end, start=None, form="10-K", fp="FY") -> dict:
    fact = {"end": end, "val": val, "accn": f"{fy}-{form}", "fy": fy, "fp": fp, "form": form, "filed": filed}
    if start:
        fact["start"] = start
    return fact


def build_company(spec: dict) -> dict:
    periods = spec["periods"]
    years = sorted(periods)
    us_gaap: dict[str, dict] = {}

    def add(tag: str, unit: str, fact: dict) -> None:
        us_gaap.setdefault(tag, {"units": {}})["units"].setdefault(unit, []).append(fact)

    for report_year in years:
        filed = periods[report_year]["filed"]

        # Compte de résultat : l'exercice du rapport + ses comparatifs, tous
        # tagués fy=report_year, fp="FY", même `filed`.
        for offset in range(COMPARATIVE_INCOME_YEARS):
            described = report_year - offset
            if described not in periods:
                continue
            for metric, by_year in spec["flows"].items():
                if described not in by_year:
                    continue
                tag = spec["revenue_tag"] if metric == "revenue" else US_GAAP_TAG[metric]
                add(tag, "USD", _fact(
                    by_year[described], filed, report_year,
                    periods[described]["end"], periods[described]["start"],
                ))

        # Bilan : instants (pas de `start`), exercice courant + comparatif.
        for offset in range(COMPARATIVE_BALANCE_YEARS):
            described = report_year - offset
            if described not in periods:
                continue
            for metric, by_year in spec["instants"].items():
                if described not in by_year:
                    continue
                add(US_GAAP_TAG[metric], "USD", _fact(
                    by_year[described], filed, report_year, periods[described]["end"],
                ))

        # Leurre 1 : un trimestre isolé publié DANS le 10-K (données
        # trimestrielles sélectionnées). Durée ~90 jours -> doit être écarté
        # par le filtre de durée annuelle, sinon un chiffre de Q4 se
        # retrouverait dans la colonne annuelle.
        current = periods[report_year]
        for metric, by_year in spec["flows"].items():
            tag = spec["revenue_tag"] if metric == "revenue" else US_GAAP_TAG[metric]
            add(tag, "USD", _fact(
                by_year[report_year] / 4, filed, report_year,
                current["end"], _minus_days(current["end"], 91),
            ))

        # Leurre 2 : un cumul 9 mois déposé en 10-Q, même fy. Doit être écarté
        # par le filtre de formulaire ET par celui de durée.
        for metric, by_year in spec["flows"].items():
            tag = spec["revenue_tag"] if metric == "revenue" else US_GAAP_TAG[metric]
            add(tag, "USD", _fact(
                by_year[report_year] * 0.72, filed, report_year,
                _minus_days(current["end"], 91), current["start"], form="10-Q", fp="Q3",
            ))

    dei = {COVER_TAG: {"units": {"shares": [
        _fact(cover["val"], periods[year]["filed"], year, cover["as_of"])
        for year, cover in sorted(spec["cover_shares"].items())
    ]}}}

    return {"cik": spec["cik"], "entityName": spec["entity_name"], "facts": {"us-gaap": us_gaap, "dei": dei}}


def _minus_days(iso_date: str, days: int) -> str:
    from datetime import datetime, timedelta
    return (datetime.strptime(iso_date, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")


def main() -> None:
    for symbol, spec in COMPANIES.items():
        path = FIXTURES_DIR / f"companyfacts_{symbol}.json"
        path.write_text(json.dumps(build_company(spec), indent=1, sort_keys=True), encoding="utf-8")
        print(f"écrit : {path}")


if __name__ == "__main__":
    main()
