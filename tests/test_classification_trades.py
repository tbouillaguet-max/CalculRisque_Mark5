"""Classification des thèses (18_classification_trades.py) : ce qui doit
tenir pour que « les points communs des perdantes » veuillent dire quelque
chose.

    - une thèse = (symbole, date d'entrée), ses allègements regroupés, et une
      thèse encore ouverte en fin de run écartée (issue inconnue) ;
    - l'alpha compare la thèse à l'indice sur les dates de CHAQUE vente ;
    - aucune variable du modèle ne lit une information postérieure à la
      décision (signal publié après, 8-K déposé après) ;
    - les thèses à cheval sur la coupure n'appartiennent à aucun échantillon ;
    - sur des données où un effet est planté, le modèle et l'arbre le
      retrouvent ; sur du bruit pur, le test le dit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import classification_trades as ct

J = pd.Timestamp


def _trades(lignes):
    return pd.DataFrame(lignes, columns=["symbol", "entry_date", "exit_date", "shares", "entry_price",
                                         "exit_price", "pnl", "return_pct", "holding_days", "exit_reason"])


def test_une_these_regroupe_ses_ventes_et_ecarte_les_ouvertes():
    trades = _trades([
        # AAA : allégée deux fois puis soldée au stop -> UNE thèse perdante.
        ("AAA", J("2020-01-02"), J("2020-02-03"), 10, 100.0, 110.0, 100.0, 10.0, 32, "rebalance"),
        ("AAA", J("2020-01-02"), J("2020-03-02"), 10, 100.0, 105.0, 50.0, 5.0, 60, "rebalance"),
        ("AAA", J("2020-01-02"), J("2020-04-01"), 80, 100.0, 80.0, -1600.0, -20.0, 90, "stop_loss"),
        # BBB : allégée, mais toujours détenue au dernier jour -> écartée.
        ("BBB", J("2020-01-02"), J("2020-02-03"), 5, 50.0, 60.0, 50.0, 20.0, 32, "rebalance"),
        # AAA rouverte plus tard : thèse distincte.
        ("AAA", J("2020-06-01"), J("2020-07-01"), 10, 90.0, 99.0, 90.0, 10.0, 30, "take_profit"),
    ])
    positions = pd.DataFrame({"date": [J("2020-12-31")], "symbol": ["BBB"],
                              "entry_date": [J("2020-01-02")]})
    theses = ct.construire_theses(trades, positions)

    assert len(theses) == 2
    aaa = theses[theses["entry_date"] == J("2020-01-02")].iloc[0]
    assert aaa["pendant_nb_ventes"] == 3
    assert aaa["pendant_motif_sortie"] == "stop_loss"
    assert aaa["pnl"] == pytest.approx(-1450.0)
    # P&L / coût de revient des titres vendus : -1450 / (100 x 100).
    assert aaa["rendement_pct"] == pytest.approx(-14.5)
    assert aaa["exit_date"] == J("2020-04-01")


def test_l_alpha_compare_chaque_vente_a_l_indice_sur_ses_propres_dates():
    trades = _trades([
        ("AAA", J("2020-01-01"), J("2020-01-03"), 1, 100.0, 120.0, 20.0, 20.0, 2, "rebalance"),
        ("AAA", J("2020-01-01"), J("2020-01-05"), 3, 100.0, 100.0, 0.0, 0.0, 4, "stop_loss"),
    ])
    indice = pd.Series([100.0, 100.0, 110.0, 110.0, 130.0],
                       index=pd.date_range("2020-01-01", periods=5))
    theses = ct.ajouter_rendement_indice(ct.construire_theses(trades, pd.DataFrame()), trades, indice)
    # Indice : +10 % sur la première vente (coût 100), +30 % sur la seconde (coût 300).
    assert theses.loc[0, "rendement_indice_pct"] == pytest.approx((0.10 * 100 + 0.30 * 300) / 400 * 100)
    assert theses.loc[0, "alpha_pct"] == pytest.approx(5.0 - 25.0)


def test_etiquettes():
    df = pd.DataFrame({"alpha_pct": np.arange(-50, 50, dtype=float), "pnl": np.arange(-50, 50, dtype=float) + 10})
    assert ct.etiqueter(df, "alpha")["gagnante"].sum() == 49
    assert ct.etiqueter(df, "pnl")["gagnante"].sum() == 59
    extremes = ct.etiqueter(df, "extremes", 20)["gagnante"]
    assert extremes.notna().sum() == 40 and extremes.sum() == 20
    with pytest.raises(ValueError):
        ct.etiqueter(df, "inconnue")


def test_la_decision_est_la_cloture_de_la_seance_precedente():
    calendrier = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"])
    decision = ct.dates_de_decision(pd.Series([J("2020-01-06"), J("2020-01-03"), J("2020-01-02")]), calendrier)
    assert decision.iloc[0] == J("2020-01-03")
    assert decision.iloc[1] == J("2020-01-02")
    assert pd.isna(decision.iloc[2])


def test_aucun_signal_publie_apres_la_decision_n_est_lu():
    theses = pd.DataFrame({"symbol": pd.array(["AAA", "AAA", "BBB"], dtype="string"),
                           "date_decision": [J("2020-03-31"), J("2020-05-15"), J("2020-03-31")]})
    signaux = pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB"],
        "filed_date": [J("2020-02-01"), J("2020-04-01"), J("2020-04-01")],
        "sector": ["Technologie"] * 3, "period_type": ["TTM"] * 3, "close": [100.0, 100.0, 50.0],
        "gap_pct": [30.0, 80.0, 99.0], "source": ["multiples"] * 3,
        "valuation_theoretical_per_share": [130.0, 180.0, 99.5],
    })
    out = ct.variables_signal(theses, signaux, pd.Series([100.0, 120.0, 50.0]))
    assert out.loc[0, "ecart_pct"] == 30.0           # le dépôt du 1er avril n'est pas encore connu
    assert out.loc[0, "age_signal_jours"] == 59
    assert out.loc[1, "ecart_pct"] == 80.0
    assert out.loc[1, "ecart_a_la_decision_pct"] == pytest.approx(50.0)  # 180 / 120 - 1
    assert out.loc[1, "cours_depuis_depot_pct"] == pytest.approx(20.0)
    assert pd.isna(out.loc[2, "ecart_pct"])          # BBB n'a encore rien publié


def test_les_8k_sont_comptes_avant_la_decision_et_pendant_a_part():
    theses = pd.DataFrame({"symbol": ["AAA"], "date_decision": [J("2020-06-30")],
                           "entry_date": [J("2020-07-01")], "exit_date": [J("2020-12-31")]})
    evenements = pd.DataFrame({
        "symbol": ["AAA"] * 4,
        "filed_date": [J("2019-01-01"), J("2020-03-01"), J("2020-06-30"), J("2020-08-01")],
        "item_codes": [["Item 2.06"], ["Item 5.02", "Item 9.01"], ["Item 2.02"], ["Item 4.02"]],
    })
    out = ct.variables_8k(theses, evenements)
    assert out.loc[0, "nb_8k_180j"] == 2                 # 1er mars et 30 juin, pas 2019 ni août
    assert out.loc[0, "nb_8k_item_5.02_180j"] == 1
    assert out.loc[0, "nb_8k_materiels_1an"] == 0        # le 2.06 date de plus d'un an
    assert out.loc[0, "pendant_nb_8k_materiels"] == 1    # le 4.02 d'août, descriptif seulement


def test_les_theses_a_cheval_sur_la_coupure_sont_ecartees():
    df = pd.DataFrame({
        "entry_date": [J("2021-01-01"), J("2021-11-01"), J("2022-03-01"), J("2021-01-01")],
        "exit_date": [J("2021-06-01"), J("2022-02-01"), J("2022-06-01"), J("2021-03-01")],
        "gagnante": [1.0, 0.0, 1.0, np.nan],
    })
    app, test = ct.decouper(df, J("2022-01-01"))
    assert app.tolist() == [True, False, False, False]
    assert test.tolist() == [False, False, True, False]


def test_le_modele_ne_voit_ni_l_issue_ni_le_pendant():
    df = pd.DataFrame({
        "symbol": ["A", "B"], "entry_date": [J("2020-01-01")] * 2, "alpha_pct": [1.0, -1.0],
        "gagnante": [1.0, 0.0], "pendant_duree_jours": [10, 20], "ecart_pct": [10.0, 20.0],
        "constante": [1.0, 1.0], "secteur": ["Santé", "Banques"],
    })
    numeriques, categorielles = ct.colonnes_modele(df)
    assert numeriques == ["ecart_pct"]
    assert categorielles == ["secteur"]


def test_benjamini_hochberg():
    q = ct.benjamini_hochberg(pd.Series([0.01, 0.04, 0.03, 0.20]))
    # Triées : 0,01 0,03 0,04 0,20 -> p x 4 / rang = 0,04 0,06 0,0533 0,20, puis
    # minimum cumulé depuis la fin.
    assert q.tolist() == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.20])


def _theses_synthetiques(effet: bool, n: int = 1600, graine: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(graine)
    entree = J("2015-01-01") + pd.to_timedelta(rng.integers(0, 365 * 9, n), unit="D")
    df = pd.DataFrame({
        "symbol": [f"S{i}" for i in range(n)],
        "entry_date": entree, "exit_date": entree + pd.Timedelta(days=60),
        "ecart_pct": rng.normal(80, 40, n), "bruit_1": rng.normal(size=n), "bruit_2": rng.normal(size=n),
        "secteur": rng.choice(["Santé", "Banques", "Technologie"], n),
    })
    # Effet planté : un momentum très négatif fait perdre.
    df["momentum_12_1_pct"] = rng.normal(0, 30, n)
    logit = (-0.06 * np.clip(-df["momentum_12_1_pct"] - 20, 0, None)) if effet else np.zeros(n)
    df["alpha_pct"] = rng.normal(0, 10, n) + logit * 5
    df["rendement_pct"] = df["alpha_pct"] + 2.0
    df["pnl"] = df["rendement_pct"] * 100
    df["gagnante"] = (df["alpha_pct"] > 0).astype(float)
    return df


def test_un_effet_plante_est_retrouve_hors_echantillon():
    df = _theses_synthetiques(effet=True)
    app, test = ct.decouper(df, J("2021-01-01"))
    numeriques, categorielles = ct.colonnes_modele(df)

    univ = ct.analyse_univariee(df, numeriques, app, test)
    stables = univ.loc[univ["stable"], "variable"].tolist()
    assert "momentum_12_1_pct" in stables
    assert "bruit_1" not in stables and "bruit_2" not in stables

    res = ct.modele_principal(df, numeriques, categorielles, app, test, n_bootstrap=200)
    assert res.ic_test[0] > 0.5
    assert res.importance.iloc[0]["variable"] == "momentum_12_1_pct"

    profils = ct.regles(df, numeriques, categorielles, app, test, profondeur=2)
    assert "momentum_12_1_pct" in profils.iloc[0]["regle"]
    assert profils.iloc[0]["taux_gagnantes_test_pct"] < profils.iloc[-1]["taux_gagnantes_test_pct"]


def test_sur_du_bruit_le_test_ne_trouve_rien():
    df = _theses_synthetiques(effet=False, graine=7)
    app, test = ct.decouper(df, J("2021-01-01"))
    numeriques, categorielles = ct.colonnes_modele(df)
    univ = ct.analyse_univariee(df, numeriques, app, test)
    assert not univ["stable"].any()
    res = ct.modele_principal(df, numeriques, categorielles, app, test, n_bootstrap=200)
    assert res.ic_test[0] < 0.5 < res.ic_test[1]


def test_walk_forward_ne_predit_que_des_theses_futures():
    df = _theses_synthetiques(effet=True)
    numeriques, categorielles = ct.colonnes_modele(df)
    annuel, proba = ct.walk_forward(df, numeriques, categorielles, premiere_annee=2018)
    assert annuel["annee"].min() >= 2018
    assert df.loc[proba.index, "entry_date"].dt.year.min() >= 2018
    eco = ct.lecture_economique(df, proba)
    assert len(eco) == 5
    assert eco["alpha_moyen_pct"].iloc[-1] > eco["alpha_moyen_pct"].iloc[0]
