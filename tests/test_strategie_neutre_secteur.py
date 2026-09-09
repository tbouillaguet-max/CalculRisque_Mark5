"""Stratégie valuation_gap_sector_neutral.

Ce qu'elle corrige : SECTOR_DCF_PARAMS prête des hypothèses inégales aux
secteurs (techno 7% de croissance et 3% de terminal, agro-alimentaire 3% et
2%). À flux identique, la techno ressort structurellement mieux valorisée --
non parce que le marché s'y trompe davantage, mais parce que la table le dit.
Classer sur gap_pct brut revient donc pour partie à classer la table, et ces
paramètres ont été écrits en connaissant l'histoire boursière.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from backtest.strategies import STRATEGY_REGISTRY
from backtest.strategies.valuation_gap_sector_neutral import (
    MIN_PEERS_PER_SECTOR, ValuationGapSectorNeutralStrategy,
)

PUBLIEE = pd.Timestamp("2020-03-02")


def _signaux(par_secteur: dict[str, list[float]]) -> pd.DataFrame:
    lignes = [
        {"symbol": f"{secteur[:3].upper()}{i}", "sector": secteur, "gap_pct": gap}
        for secteur, gaps in par_secteur.items()
        for i, gap in enumerate(gaps)
    ]
    df = pd.DataFrame(lignes).assign(published_date=PUBLIEE)
    assert df["symbol"].is_unique, "symboles en collision dans le jeu de test"
    return df


def _poids_par_secteur(weights: dict[str, float], signaux: pd.DataFrame) -> dict[str, float]:
    secteur_de = signaux.set_index("symbol")["sector"]
    par_secteur: dict[str, float] = {}
    for symbol, poids in weights.items():
        par_secteur[secteur_de[symbol]] = par_secteur.get(secteur_de[symbol], 0.0) + poids
    return par_secteur


# --------------------------------------------------------------------------- #
# Le coeur : l'écart se lit par rapport aux pairs, pas dans l'absolu
# --------------------------------------------------------------------------- #
def test_un_secteur_entierement_bon_marche_ne_donne_pas_que_des_candidates():
    """Huit technos entre +40% et +47% : elles sont toutes "bon marché" au
    sens absolu, aucune ne l'est PAR RAPPORT À SES PAIRES. La stratégie
    d'origine les achèterait toutes."""
    signaux = _signaux({"Technologie": [40.0 + i for i in range(8)]})
    retenues = ValuationGapSectorNeutralStrategy().generate_target_weights(signaux, set())
    assert retenues == {}

    origine = STRATEGY_REGISTRY["valuation_gap_dcf"]().generate_target_weights(signaux, set())
    assert len(origine) == 8, "le scénario ne contraste avec rien"


def test_l_anomalie_est_trouvee_dans_le_secteur_le_moins_bien_note():
    """Une entreprise agro-alimentaire à +45% quand ses pairs sont à ~+7% est
    l'anomalie du jour -- mais elle est invisible à un classement brut, où
    huit technos à +40% passent devant."""
    signaux = _signaux({
        "Technologie": [40.0 + i for i in range(8)],
        "Agro-alimentaire et boissons": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 45.0],
    })
    retenues = ValuationGapSectorNeutralStrategy().generate_target_weights(signaux, set())
    assert list(retenues) == ["AGR6"]


def test_le_niveau_absolu_d_un_secteur_ne_biaise_plus_l_allocation():
    """Même dispersion INTERNE dans les deux secteurs, seul leur niveau
    diffère -- exactement ce que produit SECTOR_DCF_PARAMS. L'allocation doit
    être symétrique."""
    dispersion = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    signaux = _signaux({
        "Technologie": [35.0 + d for d in dispersion],
        "Agro-alimentaire et boissons": [8.0 + d for d in dispersion],
    })
    poids = _poids_par_secteur(
        ValuationGapSectorNeutralStrategy().generate_target_weights(signaux, set()), signaux,
    )
    assert poids["Technologie"] == pytest.approx(poids["Agro-alimentaire et boissons"], rel=1e-6)

    biaisee = _poids_par_secteur(
        STRATEGY_REGISTRY["valuation_gap_dcf"]().generate_target_weights(signaux, set()), signaux,
    )
    assert biaisee["Technologie"] > 2 * biaisee["Agro-alimentaire et boissons"], (
        "le scénario ne reproduit pas le biais de niveau"
    )


# --------------------------------------------------------------------------- #
# Garde-fous
# --------------------------------------------------------------------------- #
def test_un_secteur_entierement_survalorise_ne_produit_aucune_candidate():
    """La neutralité sectorielle sert à CLASSER, pas à absoudre : la "moins
    pire" d'un secteur cher reste chère."""
    signaux = _signaux({"Technologie": [-40.0, -35.0, -30.0, -25.0, -20.0, -15.0, 5.0]})
    assert ValuationGapSectorNeutralStrategy().generate_target_weights(signaux, set()) == {}


def test_le_seuil_absolu_est_reglable():
    signaux = _signaux({"Technologie": [-40.0, -35.0, -30.0, -25.0, -20.0, -15.0, 5.0]})
    permissive = ValuationGapSectorNeutralStrategy(min_absolute_gap_pct=-100.0)
    assert permissive.generate_target_weights(signaux, set())


def test_un_secteur_trop_peu_peuple_retombe_sur_la_mediane_de_l_univers():
    """Une médiane sur deux titres n'est pas une norme sectorielle, c'est du
    bruit qu'on prendrait pour un signal."""
    assert MIN_PEERS_PER_SECTOR > 2
    signaux = _signaux({
        "Technologie": [10.0] * 10,
        # Deux titres seulement : leur propre médiane (50) annulerait l'écart
        # du plus cher. Avec le repli sur la médiane de l'univers (10), il
        # ressort nettement.
        "Chimie": [50.0, 90.0],
    })
    retenues = ValuationGapSectorNeutralStrategy().generate_target_weights(signaux, set())
    assert set(retenues) == {"CHI0", "CHI1"}


def test_le_plafond_par_secteur_borne_l_allocation_cumulee():
    """Vingt technos à 4% chacune font 80% du portefeuille sur un seul secteur
    sans qu'aucune ligne ne dépasse son plafond individuel : neutraliser le
    secteur dans le SCORE sans le borner dans l'ALLOCATION le laisserait
    revenir par la fenêtre."""
    # La MAJORITÉ du secteur doit être bon marché pour que la médiane reste
    # basse : avec 20 titres à +60% sur 26, la médiane vaut 60 et plus aucun
    # d'entre eux n'est en excès sur ses pairs.
    signaux = _signaux({
        "Technologie": [10.0] * 20 + [60.0] * 6,
        "Santé": [10.0] * 20 + [60.0] * 3,
    })
    strategie = ValuationGapSectorNeutralStrategy(max_weight_per_sector_pct=30.0)
    poids = _poids_par_secteur(strategie.generate_target_weights(signaux, set()), signaux)
    assert poids["Technologie"] <= 0.30 + 1e-9
    assert sum(poids.values()) < 1.0, "l'excédent ne doit pas être redistribué"


def test_le_plafond_par_secteur_est_desactivable():
    signaux = _signaux({"Technologie": [10.0] * 20 + [60.0] * 6})
    sans_cap = ValuationGapSectorNeutralStrategy(max_weight_per_sector_pct=0)
    poids = _poids_par_secteur(sans_cap.generate_target_weights(signaux, set()), signaux)
    assert poids["Technologie"] > 0.30


def test_le_plafond_par_position_reste_applique():
    signaux = _signaux({"Technologie": [10.0] * 6 + [500.0]})
    retenues = ValuationGapSectorNeutralStrategy().generate_target_weights(signaux, set())
    assert max(retenues.values()) <= config.BACKTEST_MAX_WEIGHT_PER_POSITION_PCT / 100 + 1e-9


# --------------------------------------------------------------------------- #
# Robustesse
# --------------------------------------------------------------------------- #
def test_aucun_signal_ne_produit_aucune_candidate():
    vide = pd.DataFrame(columns=["symbol", "sector", "gap_pct", "published_date"])
    assert ValuationGapSectorNeutralStrategy().generate_target_weights(vide, set()) == {}


def test_un_secteur_manquant_ne_fait_pas_planter():
    """Les entreprises radiées n'ont pas toujours de secteur (02 ne les
    classe que si l'univers complet lui a été passé)."""
    signaux = _signaux({"Technologie": [10.0] * 6})
    signaux.loc[0, "sector"] = None
    signaux.loc[1, "sector"] = np.nan
    retenues = ValuationGapSectorNeutralStrategy().generate_target_weights(signaux, set())
    assert isinstance(retenues, dict)


def test_la_mediane_resiste_a_un_ecart_aberrant():
    """L'historique contient des écarts à plusieurs milliers de pour cent,
    produits par une valeur théorique proche de zéro. Une MOYENNE sectorielle
    s'y déplacerait d'un facteur dix sur un seul titre."""
    signaux = _signaux({"Technologie": [10.0] * 8 + [50.0, 9000.0]})
    retenues = ValuationGapSectorNeutralStrategy().generate_target_weights(signaux, set())
    assert "TEC8" in retenues, "l'aberration a emporté la référence du secteur"


def test_un_seuil_negatif_ne_divise_pas_par_zero():
    """capped_weights normalise par la somme des convictions : des convictions
    négatives la feraient s'annuler."""
    signaux = _signaux({"Technologie": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]})
    strategie = ValuationGapSectorNeutralStrategy(
        entry_threshold_pct=-10.0, min_absolute_gap_pct=-100.0,
    )
    retenues = strategie.generate_target_weights(signaux, set())
    assert retenues
    assert all(np.isfinite(p) and p >= 0 for p in retenues.values())


def test_les_parametres_sont_exposes_pour_la_reproductibilite():
    """run_config.json doit porter de quoi rejouer le run à l'identique."""
    strategie = ValuationGapSectorNeutralStrategy(
        entry_threshold_pct=7.0, min_absolute_gap_pct=3.0, max_weight_per_sector_pct=25.0,
    )
    assert strategie.params == {
        "entry_threshold_pct": 7.0, "min_absolute_gap_pct": 3.0,
        "max_weight_per_sector_pct": 25.0,
    }


def test_la_strategie_est_enregistree():
    assert STRATEGY_REGISTRY["valuation_gap_sector_neutral"] is ValuationGapSectorNeutralStrategy


# --------------------------------------------------------------------------- #
# Câblage CLI : chaque stratégie doit garder SON seuil par défaut
# --------------------------------------------------------------------------- #
def test_la_cli_n_ecrase_pas_le_seuil_propre_a_la_strategie(tmp_path, monkeypatch):
    """09_backtest.py passait `entry_threshold_pct` systématiquement, avec le
    défaut de valuation_gap_dcf (20%). Or les deux seuils ne mesurent pas la
    même chose : 20 points d'excès sur la MÉDIANE DE SON SECTEUR est bien plus
    sélectif que 20% d'écart au cours. La stratégie neutre tournait donc deux
    fois trop strictement, en silence."""
    import importlib
    import json
    import sys

    dates = pd.bdate_range("2015-01-02", periods=300)
    symbols = [f"S{i}" for i in range(8)]
    prix = pd.concat([
        pd.DataFrame({"symbol": s, "date": dates, "close": 100.0,
                      "open": 100.0, "high": 100.0, "low": 100.0, "volume": 1e6})
        for s in symbols
    ], ignore_index=True)
    prix.to_parquet(tmp_path / "daily.parquet", index=False)
    pd.DataFrame([
        {"symbol": s, "filed_date": dates[5], "year": 2014, "fiscal_year": 2014,
         "sector": "Technologie", "close": 100.0, "period_type": "FY",
         "valuation_dcf_per_share": 150.0, "gap_pct": 30.0}
        for s in symbols
    ]).to_parquet(tmp_path / "dcf.parquet", index=False)
    pd.DataFrame({"RIC": symbols}).to_csv(tmp_path / "univ.csv", index=False, encoding="utf-8-sig")

    monkeypatch.setattr(config, "DAILY_PRICES_FILE", tmp_path / "daily.parquet")
    monkeypatch.setattr(config, "DCF_HISTORY_FILE", tmp_path / "dcf.parquet")
    monkeypatch.setattr(config, "UNIVERSE_FILE", tmp_path / "univ.csv")
    monkeypatch.setattr(config, "UNIVERSE_HISTORY_FILE", tmp_path / "absent.parquet")
    monkeypatch.setattr(config, "QUALITATIVE_VALIDATION_FILE", tmp_path / "absent.parquet")
    monkeypatch.setattr(config, "MATERIAL_EVENTS_8K_FILE", tmp_path / "absent.parquet")
    monkeypatch.setattr(config, "DIR_BACKTEST", tmp_path / "bt")

    module_09 = importlib.import_module("09_backtest")
    monkeypatch.setattr(sys, "argv", [
        "09_backtest.py", "--strategy", "valuation_gap_sector_neutral", "--run-id", "run",
    ])
    module_09.main()

    run_config = json.loads((tmp_path / "bt" / "run" / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["strategy_params"]["entry_threshold_pct"] == (
        config.BACKTEST_SECTOR_NEUTRAL_ENTRY_THRESHOLD_PCT
    )
    # run_config.json doit porter les valeurs EFFECTIVES, défauts compris,
    # sinon le run n'est pas rejouable dès qu'un défaut bouge.
    assert run_config["strategy_params"]["max_weight_per_sector_pct"] == (
        config.BACKTEST_MAX_WEIGHT_PER_SECTOR_PCT
    )


def test_la_cli_respecte_un_seuil_explicite(tmp_path, monkeypatch):
    """Contrôle négatif : --entry-threshold-pct doit toujours fonctionner."""
    import importlib
    import sys

    module_09 = importlib.import_module("09_backtest")
    parser_args = module_09.parse_strategy_params(["min_absolute_gap_pct=3"])
    assert parser_args == {"min_absolute_gap_pct": 3}
