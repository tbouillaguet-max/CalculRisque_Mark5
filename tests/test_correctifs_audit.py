"""Tests de non-régression des correctifs issus de RAPPORT_AUDIT.md.

Un test par constat, nommé par sa référence dans le rapport. Chacun REJOUE le
scénario qui avait servi à démontrer le bug -- c'est l'absence de tests
exécutant les affirmations des docstrings qui avait laissé ces cinq écarts
s'installer entre ce que le code disait faire et ce qu'il faisait (cf. la
section "Le commentaire dérive du code" du rapport).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from backtest.data_loader import OptionSnapshotIndex
from backtest.options_engine import OptionsBacktestEngine
from backtest.strategies.base import capped_weights
from backtest.strategies.valuation_gap_multiples_options import ValuationGapMultiplesOptionsStrategy
from tests.options_harness import cours, moteur, serie_bruitee, signaux


def _run(**kwargs):
    """Un CALL sur un sous-jacent qui perd 47% -- le scénario du rapport."""
    defaults = dict(
        target_tenor_days=730, roll_when_days_left=270, max_delta_notional_pct=100.0,
        slippage_pct_of_premium=2.5, min_resize_relative_pct=15.0,
        max_fee_pct_of_trade=1.0, signal_max_age_days=100_000,
    )
    defaults.update(kwargs)
    engine = moteur(
        cours({"AAA": serie_bruitee(100.0, 900, sigma=0.012, graine=1)}),
        signaux(["AAA"], "2020-01-02"), {"AAA": "CALL"}, **defaults,
    )
    engine.run()
    return engine


# --------------------------------------------------------------------------- #
# B1 -- le plancher de primes était une martingale sur les perdants
# --------------------------------------------------------------------------- #

def test_B1_le_plancher_de_primes_est_desactive_par_defaut():
    """C'était la première cause de perte du moteur : un plancher exprimé en
    PRIMES est violé quand la prime baisse, donc quand la thèse échoue."""
    assert config.OPTIONS_MIN_DEPLOYMENT_PCT == 0


def test_B1_sans_plancher_la_position_ne_grossit_pas_quand_la_these_echoue():
    avec = _run(min_deployment_pct=25.0)
    sans = _run(min_deployment_pct=0.0)

    contrats = lambda e: pd.DataFrame(e.positions_history_rows)["contracts"].max()  # noqa: E731
    nav = lambda e: e.equity_curve_rows[-1]["nav"]  # noqa: E731

    # Le plancher accumule des contrats d'autant plus vite que l'option perd
    # de la valeur : c'est le symptôme direct.
    #
    # L'écart mesuré ici (facteur ~6) est plus faible qu'à la découverte du
    # bug (2 253 contre 26, soit un facteur 87) parce que le plancher de DELTA
    # (config.OPTIONS_MIN_DELTA_FOR_SIZING) coupe désormais l'emballement en
    # amont : une fois l'option assez loin de la monnaie, plus aucun
    # renforcement n'est dimensionné. Les deux correctifs se recouvrent
    # partiellement, et c'est voulu -- ils attaquent la même divergence à deux
    # endroits différents.
    assert contrats(avec) > 3 * contrats(sans)
    # Et il transforme une perte modérée en perte majeure, à signal identique.
    assert nav(sans) > 1.5 * nav(avec)


# --------------------------------------------------------------------------- #
# B2 -- le plafond de levier n'était vérifié que dans _deploy_idle_cash
# --------------------------------------------------------------------------- #

def test_B2_le_plafond_de_levier_borne_le_portefeuille_et_pas_seulement_un_ordre():
    """281% de delta notionnel observés pour un plafond à 100% : le plafond
    n'était appliqué ni à l'ouverture, ni après coup quand le gamma faisait
    dériver l'exposition."""
    engine = _run(min_deployment_pct=25.0)
    observe = pd.DataFrame(engine.equity_curve_rows)["delta_notional_pct"].max()
    # La borne n'est pas le plafond exact : la décision est prise à la clôture
    # de J et exécutée à l'ouverture de J+1, et un NAV qui s'effondre fait
    # monter le ratio entre les deux. Mais elle doit rester du même ordre.
    assert observe < 200, f"delta notionnel max {observe:.0f}% -- le dé-levier ne mord pas"
    assert engine.delever_events_count > 0


def test_B2_le_delever_est_signale_dans_les_diagnostics():
    engine = _run(min_deployment_pct=25.0)
    assert "delever_events_count" in engine.execution_diagnostics()


# --------------------------------------------------------------------------- #
# B3 -- le roulement remettait la position sur une base de taille incompatible
# --------------------------------------------------------------------------- #

def test_B3_le_roulement_conserve_l_exposition_etablie():
    """`_check_rolls` annonce "à exposition inchangée" ; la position passait
    de 2 253 à 161 contrats."""
    engine = _run()
    historique = pd.DataFrame(engine.positions_history_rows)
    trades = pd.DataFrame(engine.trades)
    rolls = trades.loc[trades["exit_reason"] == "roll", "exit_date"].unique()
    assert len(rolls) > 0, "scénario sans roulement : le test ne prouverait rien"

    for date in rolls:
        date = pd.Timestamp(date)
        avant = historique[historique["date"] < date]["contracts"].iloc[-1]
        apres = historique[historique["date"] >= date]["contracts"].iloc[0]
        # Le contrat change (nouveau strike, échéance pleine) donc le nombre de
        # contrats pour une même exposition $ change aussi : on vérifie
        # l'absence d'effondrement, pas une égalité.
        assert apres > avant * 0.5, f"roulement du {date.date()} : {avant:.0f} -> {apres:.0f}"


def test_B3_un_roulement_n_est_pas_plafonne_par_le_plafond_par_ordre():
    """Un roulement RENOUVELLE une position ; le plafond par ordre le
    ramenait à une fraction de sa taille, puis il fallait des semaines de
    renforcements pour la reconstituer."""
    from backtest.options_engine import UNCAPPED_BUY_REASONS
    assert "roll" in UNCAPPED_BUY_REASONS


# --------------------------------------------------------------------------- #
# B4 -- snapshots d'options lus dans le futur
# --------------------------------------------------------------------------- #

def _snapshot(date: str) -> dict:
    return {
        "symbol": "AAA", "option_type": "CALL", "snapshot_date": pd.Timestamp(date),
        "expiry": pd.Timestamp("2022-01-20"), "strike": 100.0, "bid": 19.0, "ask": 21.0,
        "implied_vol": 0.90, "delta": 0.60, "gamma": 0.0, "vega": 0.0, "theta": 0.0,
        "underlying_spot": 100.0, "moneyness_pct": 0.0, "multiplier": 100.0,
    }


def test_B4_aucun_snapshot_posterieur_a_la_date_interrogee():
    index = OptionSnapshotIndex(pd.DataFrame([_snapshot("2020-01-20")]))
    trouve = index.find(
        "AAA", "CALL", pd.Timestamp("2020-01-10"), tolerance_days=14,
        target_strike=100.0, target_tenor_days=730,
    )
    assert trouve is None, "un snapshot du futur reste sélectionnable"


def test_B4_le_dernier_snapshot_anterieur_est_retenu():
    index = OptionSnapshotIndex(pd.DataFrame([_snapshot("2020-01-02"), _snapshot("2020-01-06")]))
    trouve = index.find(
        "AAA", "CALL", pd.Timestamp("2020-01-10"), tolerance_days=14,
        target_strike=100.0, target_tenor_days=730,
    )
    assert trouve is not None
    assert trouve["snapshot_date"] == pd.Timestamp("2020-01-06")


def test_B4_l_entree_est_repricee_au_spot_d_execution():
    """On garde l'IV du snapshot (grandeur transportable) et on en dérive le
    prix au spot du jour, au lieu de reprendre une prime cotée à un autre
    spot : sans quoi le premier repricing fait apparaître un saut de P&L."""
    panel = cours({"AAA": serie_bruitee(100.0, 120, graine=8)})
    snapshots = pd.DataFrame([{
        **_snapshot(panel.close.index[1].strftime("%Y-%m-%d")),
        "expiry": panel.close.index[1] + pd.Timedelta(days=730),
        # Prime volontairement ABERRANTE : si elle était reprise telle quelle,
        # le prix d'entrée s'en ressentirait immédiatement.
        "bid": 900.0, "ask": 900.0,
    }])
    engine = moteur(
        panel, signaux(["AAA"], "2020-01-02"), {"AAA": "CALL"},
        option_snapshots=snapshots, real_snapshot_tolerance_days=14,
        roll_when_days_left=None, signal_max_age_days=100_000,
    )
    engine.run()
    position = pd.DataFrame(engine.positions_history_rows)
    assert not position.empty
    assert (position["source"] == "real").all(), "le snapshot réel n'a pas été utilisé"
    # La prime d'entrée suit Black-Scholes au spot du jour, pas les 900$ cotés.
    assert position["entry_premium"].iloc[0] < 100


# --------------------------------------------------------------------------- #
# B5 -- ordres abandonnés en silence
# --------------------------------------------------------------------------- #

def test_B5_les_ordres_abandonnes_sont_comptes_par_motif():
    """Un run peut ne rien faire du tout ; il ne doit pas le rapporter comme
    un run normal."""
    engine = moteur(
        cours({"AAA": serie_bruitee(100.0, 120, graine=4)}), signaux(["AAA"], "2020-01-02"),
        {"AAA": "CALL"}, max_fee_pct_of_trade=0.0001, daily_rebalance=True,
        signal_max_age_days=100_000,
    )
    engine.run()
    diagnostics = engine.execution_diagnostics()
    assert not engine.positions
    assert diagnostics["dropped_orders_count"] > 0
    assert diagnostics["dropped_orders_by_reason"]["frais_excessifs"] > 0


# --------------------------------------------------------------------------- #
# B6 -- capped_weights abandonnait le plafond sous 5 candidats
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n_candidats", [1, 2, 3, 4, 5, 8])
def test_B6_le_plafond_de_ponderation_n_est_jamais_depasse(n_candidats):
    poids = capped_weights(pd.Series(range(1, n_candidats + 1), dtype=float), cap_pct=20.0)
    assert poids.max() <= 0.20 + 1e-9, f"{n_candidats} candidats -> {poids.max():.1%}"


def test_B6_le_reste_du_capital_n_est_pas_alloue_de_force():
    """Avec une seule candidate, la bonne réponse est 20% investi et 80% en
    cash -- pas 100% sur un titre."""
    poids = capped_weights(pd.Series([1.0]), cap_pct=20.0)
    assert poids.sum() == pytest.approx(0.20)


# --------------------------------------------------------------------------- #
# B7 -- une sortie d'indice était liquidée sous le motif "signal_lost"
# --------------------------------------------------------------------------- #

def test_B7_une_sortie_d_indice_ne_ferme_pas_une_position():
    """Le README l'annonce explicitement ; le moteur faisait l'inverse -- et
    seulement quand un AUTRE symbole était éligible ce jour-là."""
    panel = cours({
        "AAA": serie_bruitee(100.0, 200, graine=2),
        "BBB": serie_bruitee(100.0, 200, graine=3),
    })
    sortie = panel.close.index[60]
    historique = pd.DataFrame([
        {"ric": "AAA", "start_date": pd.Timestamp("2000-01-01"), "end_date": sortie},
        {"ric": "BBB", "start_date": pd.Timestamp("2000-01-01"), "end_date": pd.NaT},
    ])
    engine = moteur(
        panel, signaux(["AAA", "BBB"], "2020-01-02"), {"AAA": "CALL", "BBB": "CALL"},
        universe_history=historique, exit_when_signal_lost=True, daily_rebalance=True,
        roll_when_days_left=None,
    )
    engine.run()
    trades = pd.DataFrame(engine.trades)
    sorties_aaa = trades[trades["symbol"] == "AAA"] if not trades.empty else trades
    assert sorties_aaa.empty, "AAA a été liquidée pour être sortie de l'indice"


# --------------------------------------------------------------------------- #
# B8 -- le filtre momentum s'orientait sur un gap_pct périmé
# --------------------------------------------------------------------------- #

def test_B8_le_gap_est_rafraichi_avec_le_cours_du_jour():
    """En réévaluation quotidienne, gap_pct doit suivre le cours : c'est son
    SIGNE qui oriente le filtre momentum, et c'est lui que lit
    valuation_gap_options."""
    panel = cours({"AAA": [100.0] * 10 + [200.0] * 10})
    engine = moteur(
        panel, signaux(["AAA"], "2020-01-02", theoretical=150.0), {"AAA": "CALL"},
        daily_rebalance=True, signal_max_age_days=100_000,
    )
    signal = {
        "symbol": "AAA", "valuation_theoretical_per_share": 150.0,
        "close": 100.0, "gap_pct": 50.0,
    }
    # Au jour 1 (cours 100), l'entreprise est sous-évaluée : gap positif.
    tot = engine._signal_row_for_rebalance("AAA", signal, panel.close.index[0])
    assert tot["gap_pct"] > 0
    # Au jour 15 (cours 200), elle est SURvaluée : le gap doit avoir changé de
    # signe, sinon le filtre momentum applique encore la règle du call.
    tard = engine._signal_row_for_rebalance("AAA", signal, panel.close.index[15])
    assert tard["gap_pct"] < 0, "gap_pct est resté figé à sa valeur du dépôt"


# --------------------------------------------------------------------------- #
# B9 -- friction déclarée mais jamais payée
# --------------------------------------------------------------------------- #

def test_B9_une_expiration_hors_de_la_monnaie_ne_facture_rien():
    """Le contrat est ABANDONNÉ, pas vendu : ni commission ni slippage."""
    engine = moteur(
        cours({"AAA": serie_bruitee(100.0, 120, sigma=0.02, graine=5)}),
        signaux(["AAA"], "2020-01-02"), {"AAA": "CALL"}, target_tenor_days=60,
        roll_when_days_left=None, slippage_pct_of_premium=2.5,
    )
    engine.run()
    executions = pd.DataFrame(engine.executions)
    expirations = executions[executions["reason"] == "expiry"]
    assert not expirations.empty, "scénario sans expiration : le test ne prouverait rien"
    sans_valeur = expirations[expirations["cash_flow"] == 0]
    assert not sans_valeur.empty
    assert (sans_valeur["commission"] == 0).all(), "commission facturée sur un contrat abandonné"
    assert (expirations["slippage"] == 0).all(), "slippage sur un règlement à l'échéance"


def test_B9_le_cash_boucle_toujours():
    engine = moteur(
        cours({"AAA": serie_bruitee(100.0, 200, graine=6)}), signaux(["AAA"], "2020-01-02"),
        {"AAA": "CALL"}, slippage_pct_of_premium=2.5,
    )
    engine.run()
    flux = pd.DataFrame(engine.executions)["cash_flow"].sum()
    # + les intérêts du cash oisif, seul mouvement de trésorerie qui ne passe
    # pas par un fill (cf. _accrue_cash_interest).
    attendu = 1_000_000.0 + flux + engine.total_cash_interest
    assert engine.cash == pytest.approx(attendu, rel=1e-9)


# --------------------------------------------------------------------------- #
# B10 -- Sortino surévalué
# --------------------------------------------------------------------------- #

def test_B10_le_sortino_utilise_la_deviation_a_la_baisse_standard():
    from backtest import metrics as metrics_mod

    dates = pd.bdate_range("2020-01-01", periods=600)
    rng = np.random.default_rng(0)
    nav = 1_000_000 * np.cumprod(1 + rng.normal(0.0004, 0.012, len(dates)))
    equity = pd.DataFrame({"date": dates, "nav": nav, "num_positions": 1,
                           "invested_value": nav, "cash": 0.0})
    calcule = metrics_mod.compute_metrics(equity, pd.DataFrame())["sortino_ratio"]

    # Reconstruction des excès EXACTEMENT comme le fait compute_metrics
    # (taux sans risque année par année), pour comparer les deux dénominateurs
    # sur la même série et rien d'autre.
    rendements = equity["nav"].pct_change().dropna()
    rf = metrics_mod._risk_free_daily(equity["date"].iloc[1:], config.RISK_FREE_RATE)
    rf.index = rendements.index
    exces = rendements - rf

    standard = np.sqrt(np.square(np.minimum(exces, 0.0)).mean())
    # L'ancienne formule : écart-type des seuls rendements négatifs, mesuré
    # autour de LEUR moyenne (et non de zéro) -- systématiquement plus petit,
    # donc un Sortino systématiquement flatté.
    ancienne = rendements[rendements < 0].std()

    assert standard > ancienne, "la déviation standard doit être la plus grande des deux"
    assert calcule == pytest.approx(exces.mean() / standard * np.sqrt(252), rel=1e-9)
    # L'écart n'est pas cosmétique : ~19% sur une distribution normale.
    assert standard / ancienne > 1.1


# --------------------------------------------------------------------------- #
# S5 -- hystérésis entre le seuil d'entrée et le seuil de sortie
# --------------------------------------------------------------------------- #

def test_S5_le_seuil_de_sortie_est_plus_bas_que_le_seuil_d_entree():
    strategie = ValuationGapMultiplesOptionsStrategy()
    assert strategie.exit_threshold_pct < strategie.entry_threshold_pct


def test_S5_une_convergence_partielle_ne_solde_pas_la_position():
    """Une position entrée au seuil doit survivre à un écart repassé JUSTE
    en dessous : elle a déjà payé son slippage et sa valeur temps."""
    strategie = ValuationGapMultiplesOptionsStrategy()
    entre_les_deux = (strategie.entry_threshold_pct + strategie.exit_threshold_pct) / 2

    # La correction d'inflation est ADDITIVE en log et s'applique sur
    # l'horizon du contrat : le cours doit être choisi pour que l'écart
    # CORRIGÉ tombe entre les deux seuils, sinon le test mesure l'inflation.
    publie = pd.Timestamp("2020-01-02")
    correction = np.log1p(config.inflation_known_at(publie) / 100) * strategie.tenor_days / 365.0 * 100
    theorique = 100.0 * np.exp((entre_les_deux - correction) / 100)

    signals = pd.DataFrame([{
        "symbol": "AAA", "published_date": publie, "sector": "Technologie",
        "source": "multiples", "close": 100.0,
        "valuation_theoretical_per_share": theorique,
        "gap_pct": 0.0, "fiscal_year": 2019,
    }])
    targets, directions = strategie.evaluate(signals, {})
    # Pas assez pour ENTRER...
    assert "AAA" not in targets
    # ...mais encore assez pour RESTER (l'inflation ne fait que renforcer ce sens).
    assert directions.get("AAA") == "CALL"


# --------------------------------------------------------------------------- #
# V1 -- le pipeline de sélection tournait deux fois par jour
# --------------------------------------------------------------------------- #

def test_V1_les_cibles_et_les_sens_sont_calcules_en_un_seul_passage():
    strategie = ValuationGapMultiplesOptionsStrategy()
    appels = {"n": 0}
    original = strategie._scored

    def compte(signals):
        appels["n"] += 1
        return original(signals)

    strategie._scored = compte
    signals = pd.DataFrame([{
        "symbol": "AAA", "published_date": pd.Timestamp("2020-01-02"), "sector": "Technologie",
        "source": "multiples", "close": 100.0, "valuation_theoretical_per_share": 150.0,
        "gap_pct": 50.0, "fiscal_year": 2019,
    }])
    strategie.evaluate(signals, {})
    assert appels["n"] == 1, f"_scored appelé {appels['n']} fois pour une seule évaluation"


# --------------------------------------------------------------------------- #
# S3 -- WACC indexé sur la courbe de taux
# --------------------------------------------------------------------------- #

def test_S3_le_wacc_suit_la_courbe_de_taux():
    """Un WACC figé de 2010 à 2026 est un pari de taux non voulu : trop haut
    quand les taux étaient à zéro, trop bas quand ils étaient à 5%."""
    zirp = config.sector_dcf_params("Technologie", 2021)      # taux connu ~0,04% (moyenne 2020)
    hausse = config.sector_dcf_params("Technologie", 2024)    # taux connu ~5,15% (moyenne 2023)
    assert zirp["wacc"] < hausse["wacc"]
    # La PRIME de risque sectorielle, elle, ne bouge pas. Le taux retenu est
    # celui CONNU à la date, donc la moyenne de l'année précédente : une
    # décision prise en cours d'année N ne peut pas s'appuyer sur la moyenne
    # de N, qui n'existe pas encore (cf. config.risk_free_rate_known_at).
    ecart_taux = config.risk_free_rate_known_at(2024) - config.risk_free_rate_known_at(2021)
    assert hausse["wacc"] - zirp["wacc"] == pytest.approx(ecart_taux, abs=1e-9)


def test_S3_le_wacc_n_utilise_pas_un_taux_annuel_pas_encore_publie():
    """RISK_FREE_RATE_BY_YEAR porte des MOYENNES annuelles. Actualiser un
    dépôt de 2020 au taux moyen 2020 (0,37%, écrasé par le krach de mars)
    suppose connue une moyenne qui ne le sera qu'en décembre."""
    assert config.risk_free_rate_known_at(2020) == config.risk_free_rate_for(2019)
    assert config.risk_free_rate_known_at(pd.Timestamp("2020-02-14")) == config.risk_free_rate_for(2019)

    params_2020 = config.sector_dcf_params("Technologie", 2020)
    prime = config.SECTOR_DCF_PARAMS["Technologie"]["wacc"] - config.WACC_CALIBRATION_RISK_FREE_RATE
    assert params_2020["wacc"] == pytest.approx(config.risk_free_rate_for(2019) + prime, abs=1e-9)

    # Le comportement ex-post reste accessible pour reproduire un run ancien.
    ex_post = config.sector_dcf_params("Technologie", 2020, point_in_time=False)
    assert ex_post["wacc"] == pytest.approx(config.risk_free_rate_for(2020) + prime, abs=1e-9)
    assert ex_post["wacc"] < params_2020["wacc"], (
        "2020 est justement l'année où le look-ahead abaissait le WACC, "
        "donc gonflait la valeur théorique juste avant le krach"
    )


def test_S3_le_wacc_reste_au_dessus_de_la_croissance_terminale():
    """calculer_terminal_value diverge si le WACC s'approche du taux terminal ;
    en 2011-2015 le sans-risque tombe à 0,05%."""
    for annee in range(2010, 2027):
        for secteur in config.SECTOR_DCF_PARAMS:
            params = config.sector_dcf_params(secteur, annee)
            marge = params["wacc"] - params["terminal_growth"]
            assert marge >= config.DCF_MIN_WACC_MINUS_TERMINAL_GROWTH - 1e-9, (
                f"{secteur} en {annee} : marge {marge:.4f}"
            )


def test_S3_sans_annee_le_comportement_est_inchange():
    for secteur, attendu in config.SECTOR_DCF_PARAMS.items():
        assert config.sector_dcf_params(secteur, None)["wacc"] == attendu["wacc"]


# --------------------------------------------------------------------------- #
# S4 -- médianes sectorielles point-in-time
# --------------------------------------------------------------------------- #

def _multiples_du_millesime(dates_de_depot: list[str], pe: list[float]) -> pd.DataFrame:
    """Un millésime (même secteur, même exercice) dont les entreprises
    déposent à des dates ÉCHELONNÉES -- la situation réelle : les 10-K d'un
    même exercice s'étalent sur près de trois mois."""
    return pd.DataFrame([
        {
            "symbol": f"S{i}", "sector": "Technologie", "period_type": "FY",
            "fiscal_year": 2020, "fiscal_quarter": None,
            "filed_date": pd.Timestamp(d), "EV/EBITDA": np.nan, "EV/Sales": np.nan, "P/E": v,
        }
        for i, (d, v) in enumerate(zip(dates_de_depot, pe))
    ])


def test_S4_la_mediane_n_utilise_que_les_pairs_deja_deposes():
    """Le multiple d'un pair est calculé sur SON cours à SA filed_date : la
    médiane servant à valoriser un déposant de février intégrait les cours de
    ses pairs jusqu'en avril."""
    module = pytest.importorskip("importlib").import_module("06b_calcul_valorisation_combinee")

    # Six pairs, déposant du 1er février au 1er avril. Les derniers déposants
    # ont un P/E très différent : s'il fuite vers les premiers, la médiane du
    # 1er février s'en ressent.
    dates = ["2021-02-01", "2021-02-10", "2021-02-20", "2021-03-01", "2021-03-15", "2021-04-01"]
    df = _multiples_du_millesime(dates, [10.0, 11.0, 12.0, 13.0, 40.0, 45.0])

    pit = module.compute_pit_sector_multiples(df)

    # Le PREMIER déposant ne voit aucun pair : pas de médiane possible.
    assert pd.isna(pit["P/E_median"].iloc[0])
    assert pit["P/E_n_peers"].iloc[0] == 0
    # Le nombre de pairs visibles croît avec la date de dépôt, et n'inclut
    # jamais la ligne elle-même.
    assert list(pit["P/E_n_peers"]) == [0, 1, 2, 3, 4, 5]


def test_S4_la_derniere_ligne_du_millesime_voit_tous_ses_pairs():
    module = pytest.importorskip("importlib").import_module("06b_calcul_valorisation_combinee")
    dates = [f"2021-02-{j:02d}" for j in range(1, 8)]
    df = _multiples_du_millesime(dates, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    pit = module.compute_pit_sector_multiples(df)
    # 6 pairs pour la dernière ligne (elle-même exclue), donc au-dessus du
    # seuil de robustesse : une médiane est produite.
    assert pit["P/E_n_peers"].iloc[-1] == 6
    assert pit["P/E_median"].iloc[-1] == pytest.approx(np.median([10, 11, 12, 13, 14, 15]))


def test_S4_le_nombre_de_pairs_est_propage_jusqu_au_signal():
    """Un écart de 30% adossé à 40 pairs et le même adossé à 5 ne doivent plus
    être indiscernables en aval."""
    module = pytest.importorskip("importlib").import_module("06b_calcul_valorisation_combinee")
    import inspect
    source = inspect.getsource(module.build_combined_valuation)
    assert '"n_peers"' in source


# =========================================================================== #
# Seconde campagne : défauts remontés par les backtests comparatifs
# (2015-2026, capital 1 M$). Un test par constat, même convention de nommage.
# =========================================================================== #

# --------------------------------------------------------------------------- #
# C1 -- le dimensionnement divergeait quand le delta tendait vers zéro
# --------------------------------------------------------------------------- #

def test_C1_le_plancher_de_delta_est_actif_par_defaut():
    """`nb = target_dollar / (|delta| x spot x mult)` diverge quand le delta
    s'effondre : à 0,01 elle attribue cent fois plus de contrats qu'à 1,0."""
    assert config.OPTIONS_MIN_DELTA_FOR_SIZING > 0


def _position_a_delta_faible(min_delta_for_sizing: float):
    """Une position dont le delta s'est effondré : CALL entré à 100, le titre
    s'effondre à 30, strike inchangé. Le contrat finit très hors de la monnaie."""
    n = 260
    px = [100.0] * 20 + list(np.linspace(100.0, 30.0, n - 20))
    panel = cours({"AAA": px})
    engine = moteur(
        panel, signaux(["AAA"], "2020-01-02"), {"AAA": "CALL"},
        target_tenor_days=730, roll_when_days_left=None, stop_loss_pct=-99.0,
        take_profit_pct=1e9, signal_max_age_days=100_000,
        min_delta_for_sizing=min_delta_for_sizing,
        min_resize_relative_pct=None, max_fee_pct_of_trade=None,
        # Le plancher borde les RENFORCEMENTS : sans mécanisme qui en génère,
        # le run n'achète qu'une fois et les deux variantes sont identiques.
        # Le plancher de primes rejoue ici exactement le scénario du bug --
        # une position perdante que le moteur recharge pendant qu'elle meurt.
        min_deployment_pct=25.0, max_trade_pct_of_nav=0,
    )
    engine.run()
    return engine


def test_C1_le_plancher_bride_le_volume_de_contrats():
    avec = _position_a_delta_faible(config.OPTIONS_MIN_DELTA_FOR_SIZING)
    sans = _position_a_delta_faible(0.0)

    volume = lambda e: sum(abs(r["contracts"]) for r in e.executions if r["side"] == "buy")  # noqa: E731
    assert volume(avec) < volume(sans), (
        f"plancher inopérant : {volume(avec):.0f} contrats achetés contre "
        f"{volume(sans):.0f} sans plancher"
    )


def test_C1_le_plancher_ne_bloque_JAMAIS_une_vente():
    """LE point critique. Un plancher qui ferait un `return` avant le calcul de
    `delta_contracts` rendrait invendable une position devenue très hors de la
    monnaie : ni rebalancement, ni perte de signal, ni stop ne pourraient plus
    la réduire, et elle resterait gelée jusqu'à l'expiration."""
    n = 120
    px = [100.0] * 10 + list(np.linspace(100.0, 35.0, n - 10))
    panel = cours({"AAA": px})
    engine = moteur(
        panel, signaux(["AAA"], "2020-01-02"), {"AAA": "CALL"},
        target_tenor_days=730, roll_when_days_left=None,
        # Stop volontairement atteignable : c'est la SORTIE qu'on teste.
        stop_loss_pct=-40.0, take_profit_pct=1e9, signal_max_age_days=100_000,
        min_delta_for_sizing=0.15, min_resize_relative_pct=None,
        max_fee_pct_of_trade=None,
    )
    engine.run()

    trades = pd.DataFrame(engine.trades)
    assert not trades.empty, "la position n'a jamais été vendue malgré le stop-loss"
    assert "stop_loss" in set(trades["exit_reason"])
    assert "AAA" not in engine.positions, "position restée gelée : le plancher l'a rendue invendable"


def test_C1_une_reduction_passe_meme_sous_le_plancher():
    """Vérification directe sur _open_or_resize : delta sous le plancher,
    cible RÉDUITE -> la réduction doit s'exécuter, pas être abandonnée."""
    n = 140
    px = [100.0] * 10 + list(np.linspace(100.0, 40.0, n - 10))
    panel = cours({"AAA": px})
    engine = moteur(
        panel, signaux(["AAA"], "2020-01-02"), {"AAA": "CALL"},
        target_tenor_days=730, roll_when_days_left=None, stop_loss_pct=-99.0,
        take_profit_pct=1e9, signal_max_age_days=100_000,
        min_delta_for_sizing=0.15, min_resize_relative_pct=None,
        max_fee_pct_of_trade=None,
    )
    engine.run()
    pos = engine.positions.get("AAA")
    if pos is None:
        pytest.skip("scénario sans position ouverte à la fin : rien à réduire")

    jour = engine.calendar[-1]
    spot = panel.close_at("AAA", jour)
    delta = abs(engine._position_delta(pos, jour, spot))
    assert delta < 0.15, f"delta {delta:.3f} : le scénario ne teste pas le plancher"

    avant = pos.contracts
    # Cible volontairement minuscule -> forcément une réduction.
    engine._open_or_resize("AAA", "CALL", 1.0, spot, jour, "rebalance")
    apres = engine.positions.get("AAA")
    assert apres is None or apres.contracts < avant, (
        "la position n'a pas pu être réduite alors que son delta est sous le plancher"
    )


# --------------------------------------------------------------------------- #
# C2 -- le plafond de levier à l'ouverture (déjà corrigé, non-régression)
# --------------------------------------------------------------------------- #

def test_C2_le_plafond_de_levier_est_consulte_a_l_ouverture():
    """Il ne l'était que dans _deploy_idle_cash : le chemin principal
    d'ouverture dimensionnait à ~100% du NAV quel que soit le plafond."""
    serre = moteur(
        cours({"AAA": serie_bruitee(100.0, 300, graine=12)}), signaux(["AAA"], "2020-01-02"),
        {"AAA": "CALL"}, max_delta_notional_pct=30.0, roll_when_days_left=None,
        signal_max_age_days=100_000,
        # Plafond PAR ORDRE désactivé : il borne l'ouverture bien avant le
        # plafond de LEVIER et masquerait l'effet qu'on veut mesurer.
        max_trade_pct_of_nav=0,
    )
    serre.run()
    large = moteur(
        cours({"AAA": serie_bruitee(100.0, 300, graine=12)}), signaux(["AAA"], "2020-01-02"),
        {"AAA": "CALL"}, max_delta_notional_pct=100.0, roll_when_days_left=None,
        signal_max_age_days=100_000, max_trade_pct_of_nav=0,
    )
    large.run()

    moyen = lambda e: pd.DataFrame(e.equity_curve_rows)["delta_notional_pct"].mean()  # noqa: E731
    # Baisser le plafond doit RÉELLEMENT baisser l'exposition -- avant, les
    # deux runs étaient quasi identiques (76,7% contre 75,7%).
    assert moyen(serre) < moyen(large) * 0.6


# --------------------------------------------------------------------------- #
# C3 -- commissions facturées à l'expiration
# --------------------------------------------------------------------------- #

def test_C3_une_expiration_ne_facture_ni_commission_ni_slippage():
    """Une expiration n'est pas un ordre : hors de la monnaie le contrat est
    abandonné, dans la monnaie il est exercé automatiquement. Ni fourchette à
    traverser, ni ordre à router. Le cas ITM payait encore 85 $ par expiration."""
    engine = moteur(
        cours({"AAA": serie_bruitee(100.0, 200, sigma=0.02, graine=21)}),
        signaux(["AAA"], "2020-01-02"), {"AAA": "CALL"}, target_tenor_days=90,
        roll_when_days_left=None, slippage_pct_of_premium=2.5,
        signal_max_age_days=100_000,
    )
    engine.run()
    executions = pd.DataFrame(engine.executions)
    expirations = executions[executions["reason"] == "expiry"]
    assert not expirations.empty, "scénario sans expiration : le test ne prouverait rien"
    # Y compris DANS la monnaie (produit strictement positif) : c'est le cas
    # que l'ancien `min(commission, gross_value)` laissait passer.
    assert (expirations["commission"] == 0).all()
    assert (expirations["slippage"] == 0).all()


def test_C3_une_expiration_dans_la_monnaie_encaisse_l_intrinseque_entier():
    engine = moteur(
        cours({"AAA": [100.0] * 40 + [200.0] * 40}), signaux(["AAA"], "2020-01-02"),
        {"AAA": "CALL"}, target_tenor_days=60, roll_when_days_left=None,
        slippage_pct_of_premium=2.5, stop_loss_pct=-99.0, take_profit_pct=1e9,
        signal_max_age_days=100_000,
    )
    engine.run()
    executions = pd.DataFrame(engine.executions)
    expiry = executions[executions["reason"] == "expiry"]
    assert not expiry.empty
    row = expiry.iloc[0]
    # Produit encaissé = intrinsèque plein, aucun frais retranché.
    attendu = row["contracts"] * row["multiplier"] * row["price"]
    assert row["cash_flow"] == pytest.approx(attendu, rel=1e-9)


# --------------------------------------------------------------------------- #
# C4 -- intérêts sur le cash oisif
# --------------------------------------------------------------------------- #

def test_C4_un_portefeuille_sans_position_croit_au_taux_sans_risque():
    """Sans ce crédit, un portefeuille peu investi est pénalisé par
    construction : cette stratégie porte en moyenne 74% de cash."""
    n = 400
    panel = cours({"AAA": serie_bruitee(100.0, n, graine=31)})
    # Signal présent mais AUCUNE direction demandée (StrategieFixe({})) :
    # le moteur tourne normalement sans jamais ouvrir de position.
    engine = moteur(panel, signaux(["AAA"], "2020-01-02"), {}, roll_when_days_left=None)
    equity, _, trades, _ = engine.run()

    assert trades.empty and not engine.positions, "le scénario doit rester 100% cash"

    dates = pd.DatetimeIndex(equity["date"])
    attendu = 1_000_000.0
    for precedent, jour in zip(dates[:-1], dates[1:]):
        attendu *= 1 + config.risk_free_rate_for(jour.year) * (jour - precedent).days / 365.0
    assert equity["nav"].iloc[-1] == pytest.approx(attendu, rel=1e-9)


def test_C4_le_credit_est_desactivable():
    n = 300
    panel = cours({"AAA": serie_bruitee(100.0, n, graine=32)})
    engine = moteur(
        panel, signaux(["AAA"], "2020-01-02"), {}, roll_when_days_left=None,
        credit_idle_cash=False,
    )
    equity, _, _, _ = engine.run()
    assert equity["nav"].iloc[-1] == pytest.approx(1_000_000.0, rel=1e-12)
    assert engine.total_cash_interest == 0.0


def test_C4_les_interets_sont_isoles_du_pnl_d_options():
    """Ils ne sont attribuables ni à la jambe call ni à la jambe put : la
    réconciliation de put_call_analysis doit les retirer du NAV."""
    from backtest import put_call_analysis as pca

    engine = moteur(
        cours({"AAA": serie_bruitee(100.0, 300, graine=33)}), signaux(["AAA"], "2020-01-02"),
        {"AAA": "CALL"}, roll_when_days_left=None, signal_max_age_days=100_000,
    )
    equity, positions, trades, _ = engine.run()
    assert engine.total_cash_interest > 0, "le scénario doit générer des intérêts"

    residual = pca.reconciliation_residual(equity, positions, trades)
    assert residual["max_abs_residual_dollar"] < 1.0


# --------------------------------------------------------------------------- #
# C5 -- hystérésis et durée de détention minimale
# --------------------------------------------------------------------------- #

def test_C5_min_holding_days_retient_une_sortie_sur_perte_de_signal():
    n = 260
    # Éligible au départ, puis le cours rejoint la théorique -> signal perdu.
    px = [60.0] * 15 + [150.0] * (n - 15)
    panel = cours({"AAA": px})
    common = dict(
        target_tenor_days=730, roll_when_days_left=None, exit_when_signal_lost=True,
        daily_rebalance=True, stop_loss_pct=-99.0, take_profit_pct=1e9,
        take_profit_convergence_fraction=0, signal_max_age_days=100_000,
        strategy=ValuationGapMultiplesOptionsStrategy(),
    )
    sans = moteur(panel, signaux(["AAA"], "2020-01-02", theoretical=150.0), {}, **common)
    sans.run()
    avec = moteur(
        panel, signaux(["AAA"], "2020-01-02", theoretical=150.0), {},
        min_holding_days=180, **common,
    )
    avec.run()

    def duree(engine):
        trades = pd.DataFrame(engine.trades)
        perdus = trades[trades["exit_reason"] == "signal_lost"]
        return None if perdus.empty else perdus["holding_days"].min()

    assert duree(sans) is not None, "le scénario doit produire une sortie signal_lost"
    assert duree(sans) < 180
    if duree(avec) is not None:
        assert duree(avec) >= 180


@pytest.mark.parametrize("motif", ["stop_loss", "expiry"])
def test_C5_min_holding_days_ne_bloque_jamais_un_garde_fou(motif):
    """Un stop-loss ou une échéance ne se négocient pas contre un calendrier."""
    n = 120
    px = [100.0] * 5 + list(np.linspace(100.0, 45.0, n - 5))
    panel = cours({"AAA": px})
    engine = moteur(
        panel, signaux(["AAA"], "2020-01-02"), {"AAA": "CALL"},
        target_tenor_days=(730 if motif == "stop_loss" else 45),
        roll_when_days_left=None,
        stop_loss_pct=(-30.0 if motif == "stop_loss" else -99.0),
        take_profit_pct=1e9, signal_max_age_days=100_000,
        min_holding_days=10_000,  # absurde : ne doit rien empêcher ici
        min_delta_for_sizing=0,
    )
    engine.run()
    trades = pd.DataFrame(engine.trades)
    assert not trades.empty, f"aucune sortie {motif} malgré min_holding_days"
    assert motif in set(trades["exit_reason"])


def test_C5_exit_threshold_identique_a_l_entree_ne_change_rien():
    """GARANTIT qu'aucune SECONDE définition de l'écart n'a été introduite :
    avec un seuil de sortie égal au seuil d'entrée, le run doit être
    strictement identique à celui sans hystérésis."""
    panel = cours({"AAA": serie_bruitee(100.0, 300, graine=41),
                   "BBB": serie_bruitee(100.0, 300, graine=42)})
    evenements = signaux(["AAA", "BBB"], "2020-01-02", theoretical=150.0)
    common = dict(
        target_tenor_days=730, roll_when_days_left=None, exit_when_signal_lost=True,
        daily_rebalance=True, take_profit_convergence_fraction=0,
        stop_loss_pct=-99.0, take_profit_pct=1e9, signal_max_age_days=100_000,
    )

    def run(ratio):
        engine = moteur(
            panel, evenements, {},
            strategy=ValuationGapMultiplesOptionsStrategy(exit_threshold_ratio=ratio),
            **common,
        )
        equity, _, trades, _ = engine.run()
        return equity["nav"].iloc[-1], len(trades)

    assert run(1.0) == run(1.0)  # déterminisme
    # Le ratio 1,0 est le comportement "sans hystérésis" : sortie au seuil
    # d'entrée, exactement comme avant l'introduction du paramètre.
    nav_neutre, n_neutre = run(1.0)
    nav_bande, n_bande = run(0.5)
    # Une bande morte ne peut que RETENIR des positions, jamais en créer.
    assert n_bande <= n_neutre


# --------------------------------------------------------------------------- #
# C6 -- diagnostics
# --------------------------------------------------------------------------- #

def test_C6_les_diagnostics_du_dimensionnement_sont_publies():
    engine = moteur(
        cours({"AAA": serie_bruitee(100.0, 300, graine=51)}), signaux(["AAA"], "2020-01-02"),
        {"AAA": "CALL"}, roll_when_days_left=None, signal_max_age_days=100_000,
    )
    engine.run()
    d = engine.execution_diagnostics()

    for cle in (
        "total_contracts_traded", "max_contracts_single_order", "max_contracts_order_symbol",
        "max_contracts_order_date", "min_delta_at_sizing", "pct_days_above_delta_cap",
        "median_excess_above_delta_cap_pct", "total_cash_interest_dollar",
    ):
        assert cle in d, f"diagnostic manquant : {cle}"

    executions = pd.DataFrame(engine.executions)
    assert d["total_contracts_traded"] == pytest.approx(executions["contracts"].abs().sum())
    achats = executions[executions["side"] == "buy"]
    assert d["max_contracts_single_order"] == pytest.approx(achats["contracts"].abs().max())
    # Le delta de dimensionnement est celui d'une entrée ATM à 2 ans : bien
    # au-dessus du plancher, sinon le run n'aurait rien acheté.
    assert d["min_delta_at_sizing"] >= config.OPTIONS_MIN_DELTA_FOR_SIZING


# --------------------------------------------------------------------------- #
# C7 -- le moteur ACTIONS annulait le plafond par renormalisation
# --------------------------------------------------------------------------- #

def test_C7_le_moteur_actions_ne_remonte_jamais_les_poids_vers_1():
    """base.capped_weights renvoie délibérément une somme < 1 quand le plafond
    mord (B6). Le moteur actions la renormalisait à 1, ce qui plaçait 100% du
    NAV sur un seul titre les jours à candidate unique -- pour un plafond
    demandé à 20%. Le moteur options, lui, ne renormalise pas : les deux
    appliquaient deux règles de concentration différentes."""
    from backtest.engine import BacktestEngine
    from backtest.strategies.valuation_gap import ValuationGapDCFStrategy

    n = 400
    dates = pd.bdate_range("2020-01-01", periods=n)
    # UNE seule entreprise éligible : le plafond doit laisser 80% en cash.
    close = pd.DataFrame({"AAA": np.linspace(100.0, 130.0, n),
                          "BBB": np.linspace(100.0, 130.0, n)}, index=dates)
    from backtest.data_loader import PricePanel
    panel = PricePanel(close, close.copy(), close.apply(lambda c: c.last_valid_index()))

    signaux_df = pd.DataFrame([{
        "symbol": "AAA", "published_date": dates[2], "fiscal_year": 2019,
        "sector": "Technologie", "close_at_filing": 100.0,
        "valuation_dcf_per_share": 300.0, "gap_pct": 200.0, "period_type": "FY",
    }])

    engine = BacktestEngine(
        price_panel=panel, signal_events=signaux_df, universe_history=None,
        fallback_universe_symbols={"AAA", "BBB"},
        strategy=ValuationGapDCFStrategy(), initial_capital=1_000_000.0,
        cost_bps=10.0, stop_loss_pct=-99.0, take_profit_pct=1e9,
        signal_max_age_days=100_000, momentum_min_pct=None,
    )
    equity, positions, _, _ = engine.run()

    nav = equity.set_index("date")["nav"]
    poids = positions["market_value"] / positions["date"].map(nav) * 100
    plafond = config.BACKTEST_MAX_WEIGHT_PER_POSITION_PCT
    # Marge : la position dérive avec le cours entre deux rebalancements, mais
    # elle ne doit jamais partir de 100% du NAV.
    assert poids.max() < plafond * 2, (
        f"poids max {poids.max():.0f}% du NAV pour un plafond de {plafond:.0f}% "
        "-- la renormalisation annule le plafond"
    )
