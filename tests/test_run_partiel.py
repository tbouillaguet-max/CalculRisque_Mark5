"""Un run partiel (--ticker, --limit) ne vide pas le fichier de sortie complet.

04c et 07b écrivaient leur sortie avec les SEULES lignes du run : un essai
`04c --ticker AAPL` réduisait les 99 147 8-K de material_events_8k.parquet à
ceux d'AAPL, et le filtre d'événements du backtest et du paper trading avec,
sans rien signaler. Même chose pour 07b et la validation qualitative.
"""

from __future__ import annotations

import importlib
import math
import sys

import pandas as pd
import pytest

import config
import reprise_jsonl

_04c = importlib.import_module("04c_recuperation_8k")
_07b = importlib.import_module("07b_validation_qualitative")


# --------------------------------------------------------------------------- #
# La fusion
# --------------------------------------------------------------------------- #
def test_les_lignes_refaites_remplacent_les_anciennes_et_les_autres_restent(tmp_path):
    chemin = tmp_path / "sortie.parquet"
    pd.DataFrame({"symbol": ["AAPL", "MSFT"], "accession_number": ["1", "2"], "v": ["ancien", "ancien"]}) \
        .to_parquet(chemin)
    nouveau = pd.DataFrame({"symbol": ["AAPL", "AAPL"], "accession_number": ["1", "3"], "v": ["neuf", "neuf"]})
    fusion, conservees = reprise_jsonl.fusionner_run_partiel(nouveau, chemin, ["symbol", "accession_number"])
    assert conservees == 1
    assert sorted(map(tuple, fusion[["symbol", "accession_number", "v"]].values)) == [
        ("AAPL", "1", "neuf"), ("AAPL", "3", "neuf"), ("MSFT", "2", "ancien")]


def test_une_periode_relue_du_json_reconnait_sa_version_parquet(tmp_path):
    """2023 (parquet, int64) et 2023.0 (JSON), NaN et None : même période --
    sinon la fusion garderait l'ancienne ET la nouvelle."""
    cle = ["symbol", "period_type", "fiscal_year", "fiscal_quarter"]
    chemin = tmp_path / "sortie.parquet"
    pd.DataFrame({"symbol": ["AAPL"], "period_type": ["10-K"], "fiscal_year": [2023],
                  "fiscal_quarter": [math.nan], "verdict": ["ancien"]}).to_parquet(chemin)
    nouveau = pd.DataFrame([{"symbol": "AAPL", "period_type": "10-K", "fiscal_year": 2023.0,
                             "fiscal_quarter": None, "verdict": "neuf"}])
    fusion, conservees = reprise_jsonl.fusionner_run_partiel(nouveau, chemin, cle)
    assert conservees == 0 and list(fusion["verdict"]) == ["neuf"]


def test_sans_fichier_existant_la_sortie_est_le_run(tmp_path):
    nouveau = pd.DataFrame({"symbol": ["AAPL"], "accession_number": ["1"]})
    fusion, conservees = reprise_jsonl.fusionner_run_partiel(nouveau, tmp_path / "absent.parquet", ["symbol"])
    assert conservees == 0 and fusion.equals(nouveau)


# --------------------------------------------------------------------------- #
# 04c, de bout en bout
# --------------------------------------------------------------------------- #
def test_04c_un_essai_sur_un_ticker_garde_les_8k_des_autres(tmp_path, monkeypatch):
    sortie = tmp_path / "material_events_8k.parquet"
    pd.DataFrame({
        "symbol": ["MSFT", "MSFT", "AAPL"], "cik": ["789", "789", "320193"],
        "filed_date": ["2020-03-01", "2020-06-01", "2020-05-01"],
        "accession_number": ["M-1", "M-2", "A-1"], "category": ["non_evalue"] * 3,
    }).to_parquet(sortie)
    ttm = tmp_path / "ttm.parquet"
    pd.DataFrame({"symbol": ["AAPL"], "cik": ["320193"]}).to_parquet(ttm)
    monkeypatch.setattr(config, "MATERIAL_EVENTS_8K_FILE", sortie)
    monkeypatch.setattr(config, "FINANCIALS_TTM_FILE", ttm)
    monkeypatch.setattr(_04c.sft.sec_http, "require_contact_email", lambda journal=None: "test@example.com")
    monkeypatch.setattr(_04c, "compute_search_windows", lambda ttm, symbol, today: [("2020-01-01", "2020-12-31")])
    monkeypatch.setattr(_04c.sft, "fetch_submissions_strict", lambda cik: [
        {"form": "8-K", "filing_date": "2020-05-01", "accession_number": "A-1", "primary_document": "a1.htm"},
        {"form": "8-K", "filing_date": "2020-08-01", "accession_number": "A-2", "primary_document": "a2.htm"},
    ])
    monkeypatch.setattr(_04c.sft, "fetch_filing_text",
                        lambda url, form=None: ("Item 8.01 Other Events. The company announced a dividend.",
                                                "debut_document"))
    monkeypatch.setattr(sys, "argv", ["04c", "--ticker", "AAPL", "--output-dir", str(tmp_path)])

    _04c.main()

    resultat = pd.read_parquet(sortie)
    assert sorted(resultat["accession_number"]) == ["A-1", "A-2", "M-1", "M-2"]
    refaits = resultat.set_index("accession_number")
    assert refaits.loc["A-1", "category"] != "non_evalue"          # remplacé par le run
    assert list(refaits.loc[["M-1", "M-2"], "category"]) == ["non_evalue"] * 2   # intacts


# --------------------------------------------------------------------------- #
# 07b, de bout en bout
# --------------------------------------------------------------------------- #
def test_07b_un_essai_limite_garde_les_autres_periodes(tmp_path, monkeypatch):
    sortie = tmp_path / "validation.parquet"
    pd.DataFrame({
        "symbol": ["AAPL", "MSFT"], "period_type": ["10-K", "10-K"], "fiscal_year": [2023, 2023],
        "fiscal_quarter": [math.nan, math.nan], "verdict": ["ancien", "ancien"],
    }).to_parquet(sortie)
    monkeypatch.setattr(config, "QUALITATIVE_VALIDATION_FILE", sortie)
    monkeypatch.setattr(_07b.sft.sec_http, "require_contact_email", lambda journal=None: "test@example.com")
    monkeypatch.setattr(_07b, "load_signal_periods", lambda limit=None: pd.DataFrame([{
        "symbol": "AAPL", "period_type": "10-K", "fiscal_year": 2023, "fiscal_quarter": math.nan,
        "filed_date": "2023-11-03", "gap_pct": 30.0, "cik": "320193",
    }]))
    monkeypatch.setattr(_07b, "evaluate_period",
                        lambda row: {"verdict": "coherent", "justification": "x", "risques_cites": None})
    monkeypatch.setattr(sys, "argv", ["07b", "--limit", "1", "--output-dir", str(tmp_path)])

    _07b.main()

    resultat = pd.read_parquet(sortie).set_index("symbol")
    assert list(resultat.index.sort_values()) == ["AAPL", "MSFT"]
    assert (resultat.loc["AAPL", "verdict"], resultat.loc["MSFT", "verdict"]) == ("coherent", "ancien")
