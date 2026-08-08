"""A3 : l'extraction annuelle rattache chaque fait XBRL à la période qui l'a
réellement produite.

Les fixtures (tests/fixtures/companyfacts_*.json, cf. leur script de
génération) reproduisent la structure exacte de l'API companyfacts et donc le
piège corrigé ici : dans un 10-K FY2023, les comparatifs FY2022 et FY2021
sortent du JSON avec fy=2023, fp="FY", form="10-K" ET la même date `filed`.
L'ancienne implémentation classait par `fy` : les trois exercices tombaient
sous la même clé et le départage sur `filed` était inopérant (égalité
stricte), si bien que c'est le premier fait rencontré dans l'ordre du JSON qui
l'emportait.

Les montants des fixtures sont recopiés des comptes publiés ; ils n'ont pas pu
être re-vérifiés depuis le réseau dans l'environnement où ce test a été écrit
(data.sec.gov inaccessible). Ce que le test contrôle n'est de toute façon pas
leur exactitude comptable, mais le rattachement période par période.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

import sec_xbrl

module_04 = importlib.import_module("04_recuperation_10k")
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def facts_for(symbol: str) -> dict:
    return json.loads((FIXTURES / f"companyfacts_{symbol}.json").read_text(encoding="utf-8"))


def extract(symbol: str, metric: str) -> dict:
    facts = facts_for(symbol)
    period_ends = sec_xbrl.collect_period_end_dates(facts, module_04.XBRL_TAGS)
    return module_04.extract_annual_values(
        facts, module_04.XBRL_TAGS[metric], metric in sec_xbrl.FLOW_METRICS, period_ends,
    )


def values_only(extracted: dict) -> dict:
    return {year: value for year, (value, _filed) in extracted.items()}


# --------------------------------------------------------------------------- #
# Les trois entreprises, sur trois exercices, flux et bilan
# --------------------------------------------------------------------------- #

ATTENDU_AAPL = {
    # Clôture fin septembre : l'étiquette est l'année civile de la clôture.
    "revenue":      {2021: 365_817e6, 2022: 394_328e6, 2023: 383_285e6},
    "ebit":         {2021: 108_949e6, 2022: 119_437e6, 2023: 114_301e6},
    "net_income":   {2021:  94_680e6, 2022:  99_803e6, 2023:  96_995e6},
    "total_assets": {2021: 351_002e6, 2022: 352_755e6, 2023: 352_583e6},
    "cash":         {2021:  34_940e6, 2022:  23_646e6, 2023:  29_965e6},
}

ATTENDU_JNJ = {
    # Calendrier 52/53 semaines : l'exercice 2021 se clôt le 2 janvier 2022 et
    # l'exercice 2022 le 1er janvier 2023. Le seuil de juillet leur redonne
    # leur étiquette d'origine -- sans lui, 2022 et 2023 entreraient en
    # collision sur "2023" et un exercice entier disparaîtrait.
    "revenue":      {2021: 93_775e6, 2022: 94_943e6, 2023: 85_159e6},
    "total_assets": {2021: 182_018e6, 2022: 187_378e6, 2023: 167_558e6},
}

ATTENDU_XOM = {
    "revenue":      {2021: 285_640e6, 2022: 413_680e6, 2023: 344_582e6},
    "net_income":   {2021:  23_040e6, 2022:  55_740e6, 2023:  36_010e6},
    "total_assets": {2021: 338_923e6, 2022: 369_067e6, 2023: 376_317e6},
}

CAS = [
    pytest.param(symbol, metric, attendu, id=f"{symbol}-{metric}")
    for symbol, table in (("AAPL", ATTENDU_AAPL), ("JNJ", ATTENDU_JNJ), ("XOM", ATTENDU_XOM))
    for metric, attendu in table.items()
]


@pytest.mark.parametrize("symbol,metric,attendu", CAS)
def test_valeurs_rattachees_au_bon_exercice(symbol, metric, attendu):
    extracted = values_only(extract(symbol, metric))
    for year, expected in attendu.items():
        assert extracted.get(year) == pytest.approx(expected), (
            f"{symbol} {metric} {year} : {extracted.get(year)!r} au lieu de {expected!r}"
        )


@pytest.mark.parametrize("symbol", ["AAPL", "JNJ", "XOM"])
def test_les_trois_exercices_sont_distincts(symbol):
    """Le symptôme le plus direct du bug : trois exercices comparatifs
    écrasés les uns par les autres ne laissaient qu'une valeur."""
    revenue = values_only(extract(symbol, "revenue"))
    assert len(set(revenue.values())) == 3, f"{symbol} : {revenue}"


def test_filed_date_est_celle_de_la_premiere_publication():
    """Chaque exercice porte la date à laquelle il a été rendu public pour la
    PREMIÈRE fois, pas celle du dernier 10-K qui l'a repris en comparatif.

    Retenir le dépôt le plus récent -- ce que faisait la première version de
    ce correctif -- collait la date du 10-K 2023 aux exercices 2021 et 2022 :
    trois exercices devenaient connaissables le même jour, chacun jusqu'à deux
    ans trop tard, et 07b allait chercher le mauvais document."""
    extracted = extract("AAPL", "revenue")
    assert extracted[2021][1] == "2021-10-29"
    assert extracted[2022][1] == "2022-10-28"
    assert extracted[2023][1] == "2023-11-03"


@pytest.mark.parametrize("symbol", ["AAPL", "JNJ", "XOM"])
def test_les_dates_de_depot_sont_toutes_distinctes(symbol):
    """Symptôme direct de la régression : trois exercices partageant une seule
    date de dépôt."""
    dates = [filed for _val, filed in extract(symbol, "revenue").values()]
    assert len(set(dates)) == len(dates), f"{symbol} : dates de dépôt confondues {dates}"


@pytest.mark.parametrize("symbol", ["AAPL", "JNJ", "XOM"])
def test_le_depot_suit_toujours_la_cloture(symbol):
    """Une date de dépôt antérieure à la clôture qu'elle publie serait du
    look-ahead ; très postérieure, c'est un signal qui arrive en retard."""
    for year, (_val, filed) in extract(symbol, "revenue").items():
        assert filed[:4] in (str(year), str(year + 1)), (
            f"{symbol} exercice {year} déposé le {filed} : écart anormal"
        )


def test_trimestre_isole_du_10k_ecarte():
    """Le fixture glisse dans chaque 10-K un trimestre seul (~91 jours) sous
    le même tag. Sans filtre de durée, ce quart de chiffre d'affaires
    pouvait remonter dans la colonne annuelle."""
    revenue = values_only(extract("AAPL", "revenue"))
    assert revenue[2023] == pytest.approx(383_285e6)
    assert all(value > 300e9 for value in revenue.values())


def test_cumul_9_mois_en_10q_ecarte():
    """Un cumul 9 mois déposé en 10-Q ne doit jamais servir de valeur
    annuelle (filtre de formulaire ET de durée)."""
    revenue = values_only(extract("XOM", "revenue"))
    assert revenue[2022] == pytest.approx(413_680e6)


def test_page_de_garde_recalee_sur_l_exercice():
    """dei:EntityCommonStockSharesOutstanding est daté du jour d'arrêté du
    compte, POSTÉRIEUR à la clôture décrite : sans recalage, le compte
    d'actions du 10-K FY2023 de JNJ (arrêté au 9 février 2024) partirait sur
    l'exercice 2024, laissant 2023 sans nombre d'actions -- et 07 écarterait
    la ligne faute de dénominateur."""
    shares = values_only(extract("JNJ", "shares_outstanding"))
    assert shares == {2021: 2_631_000_000, 2022: 2_594_000_000, 2023: 2_407_000_000}


def test_page_de_garde_coherente_avec_les_autres_metriques():
    """Le recalage doit produire EXACTEMENT les mêmes étiquettes que les
    autres métriques, sinon la ligne de financials.parquet est incohérente."""
    for symbol in ("AAPL", "JNJ", "XOM"):
        shares = set(extract(symbol, "shares_outstanding"))
        assets = set(extract(symbol, "total_assets"))
        assert shares == assets, symbol


def test_extraction_complete_produit_une_ligne_par_exercice():
    """Bout en bout : le DataFrame de sortie de 04 doit porter trois lignes
    datées de la bonne filed_date."""
    facts = facts_for("XOM")
    df = module_04.extract_financials_for_ticker("XOM", "0000034088", session=None, facts=facts)
    assert sorted(df["year"]) == [2021, 2022, 2023]
    assert df.set_index("year").loc[2023, "filed_date"] == "2024-02-28"
    assert df.set_index("year").loc[2022, "revenue"] == pytest.approx(413_680e6)


# --------------------------------------------------------------------------- #
# Règles unitaires de sec_xbrl
# --------------------------------------------------------------------------- #

def test_etiquette_exercice_selon_le_mois_de_cloture():
    assert sec_xbrl.fiscal_year_of_end("2023-09-30") == 2023   # Apple
    assert sec_xbrl.fiscal_year_of_end("2023-12-31") == 2023   # calendrier civil
    assert sec_xbrl.fiscal_year_of_end("2023-01-01") == 2022   # JNJ, exercice 2022
    assert sec_xbrl.fiscal_year_of_end("2021-01-30") == 2020   # grande distribution
    assert sec_xbrl.fiscal_year_of_end("2023-06-30") == 2023   # clôture de juin, seuil inclusif
    assert sec_xbrl.fiscal_year_of_end(None) is None
    assert sec_xbrl.fiscal_year_of_end("pas-une-date") is None


def test_deux_clotures_consecutives_ne_collisionnent_jamais():
    assert sec_xbrl.fiscal_year_of_end("2022-01-02") != sec_xbrl.fiscal_year_of_end("2023-01-01")
    assert sec_xbrl.fiscal_year_of_end("2023-01-01") != sec_xbrl.fiscal_year_of_end("2023-12-31")


def test_duree_et_fenetres():
    assert sec_xbrl.duration_days("2022-09-25", "2023-09-30") == 370
    assert sec_xbrl.duration_days(None, "2023-09-30") is None
    assert sec_xbrl.in_duration_window(370, sec_xbrl.ANNUAL_DURATION_DAYS)
    assert not sec_xbrl.in_duration_window(91, sec_xbrl.ANNUAL_DURATION_DAYS)
    assert sec_xbrl.in_duration_window(91, sec_xbrl.QUARTER_DURATION_DAYS)
    # Une durée inconnue (poste de bilan) ne passe jamais pour un flux.
    assert not sec_xbrl.in_duration_window(None, sec_xbrl.ANNUAL_DURATION_DAYS)


def test_la_publication_d_origine_gagne_sur_le_comparatif():
    """Deux faits décrivant la MÊME période, publiés à deux dates : c'est le
    chiffre d'ORIGINE qui est retenu, avec SA date.

    Prendre la valeur révisée (111) tout en la datant de 2022-02-01 ferait
    lire au backtest une correction publiée un an plus tard ; la prendre avec
    sa vraie date (2023-02-01) retarderait l'exercice 2021 d'un an. Le seul
    couple cohérent est (valeur d'origine, date d'origine)."""
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2021-01-01", "end": "2021-12-31", "val": 100.0,
         "fy": 2021, "fp": "FY", "form": "10-K", "filed": "2022-02-01"},
        {"start": "2021-01-01", "end": "2021-12-31", "val": 111.0,
         "fy": 2022, "fp": "FY", "form": "10-K", "filed": "2023-02-01"},
    ]}}}}}
    extrait = module_04.extract_annual_values(facts, ["Revenues"], True)
    assert extrait == {2021: (100.0, "2022-02-01")}


def test_earliest_entry_ignore_les_dates_manquantes():
    """Une entrée sans date de dépôt ne doit pas gagner avec une chaîne vide."""
    entrees = [
        {"val": 1.0, "filed": ""},
        {"val": 2.0, "filed": "2022-02-01"},
    ]
    assert sec_xbrl.earliest_entry(entrees)["val"] == 2.0
    assert sec_xbrl.earliest_entry([]) is None


def test_priorite_entre_tags_candidats():
    """Un tag de repli ne remplace jamais un tag mieux classé pour la même
    année, mais couvre les années que le premier ne renseigne pas (changement
    de tag en cours d'historique, ex. ASC 606)."""
    def annual(tag, year, val, filed):
        return {"start": f"{year}-01-01", "end": f"{year}-12-31", "val": val,
                "fy": year, "fp": "FY", "form": "10-K", "filed": filed}

    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [annual("Revenues", 2017, 50.0, "2018-02-01")]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            annual("RFC", 2017, 999.0, "2019-02-01"),   # doit perdre : tag moins prioritaire
            annual("RFC", 2019, 70.0, "2020-02-01"),    # doit gagner : seul tag pour 2019
        ]}},
    }}}
    extracted = values_only(module_04.extract_annual_values(
        facts, module_04.XBRL_TAGS["revenue"], True,
    ))
    assert extracted == {2017: 50.0, 2019: 70.0}
