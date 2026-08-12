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
    assert contrats(avec) > 10 * contrats(sans)
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
    assert engine.cash == pytest.approx(1_000_000.0 + flux, rel=1e-9)


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
    zirp = config.sector_dcf_params("Technologie", 2021)      # taux ~0,04%
    hausse = config.sector_dcf_params("Technologie", 2023)    # taux ~5,15%
    assert zirp["wacc"] < hausse["wacc"]
    # La PRIME de risque sectorielle, elle, ne bouge pas.
    ecart_taux = config.risk_free_rate_for(2023) - config.risk_free_rate_for(2021)
    assert hausse["wacc"] - zirp["wacc"] == pytest.approx(ecart_taux, abs=1e-9)


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
