"""Stratégie actions sur la VALORISATION COMBINÉE (06b) plutôt que sur le DCF
seul (07).

CE QUE CETTE STRATÉGIE CORRIGE. Les deux moteurs du dépôt ne lisaient pas la
même valorisation, et rien ne le disait : `09_backtest.py` chargeait
`dcf_historique.parquet`, `10_backtest_options.py`
`valorisation_combinee_historique.parquet`. Le côté actions se privait du
signal que le projet décrit lui-même comme le meilleur -- multiples sectoriels
point-in-time en priorité, DCF en repli -- sans qu'aucune décision ne l'ait
jamais tranché.

L'ARGUMENT N'EST PAS LE SHARPE. Mesuré sur 2015-2026, l'écart de Sharpe est de
+0,05, et le test apparié le donne non significatif (p = 0,25). Ce qui
distingue vraiment les deux signaux est ailleurs, et ne relève pas de la
statistique : la COUVERTURE de l'univers passe de 77% à 94%. Un DCF n'existe
pas pour une entreprise à flux de trésorerie négatifs ni pour un métier de
bilan ; un multiple sectoriel, si. Or le moteur mesure son alpha contre un
indice qui porte 100% de ses membres : choisir parmi 77% pendant qu'on est
jugé sur 100% surestime l'alpha, et c'est le biais que le README documente
comme le plus flatteur pour un signal *value*.
"""

from __future__ import annotations

import pandas as pd
import pytest

import config
from backtest import data_loader
from backtest.strategies import STRATEGY_REGISTRY
from backtest.strategies.valuation_gap_combined import ValuationGapCombinedStrategy


def _signaux(gaps: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([
        {"symbol": s, "gap_pct": g, "sector": "Technologie",
         "published_date": pd.Timestamp("2020-06-01"), "fiscal_year": 2019,
         "close_at_filing": 100.0, "valuation_dcf_per_share": 100.0 * (1 + g / 100)}
        for s, g in gaps.items()
    ])


def test_la_strategie_est_enregistree_et_declare_sa_source():
    assert STRATEGY_REGISTRY["valuation_gap_combined"] is ValuationGapCombinedStrategy
    assert ValuationGapCombinedStrategy.signal_source == "combinee"
    # Contrôle de séparation : les deux autres stratégies actions gardent le DCF.
    assert STRATEGY_REGISTRY["valuation_gap_dcf"].signal_source == "dcf"
    assert STRATEGY_REGISTRY["valuation_gap_sector_neutral"].signal_source == "dcf"


def test_le_seuil_se_lit_comme_celui_de_valuation_gap_dcf():
    """Les deux signaux produisent la même grandeur -- un écart au cours en
    pourcentage -- donc le seuil est directement comparable. C'est ce qui rend
    la comparaison entre les deux stratégies interprétable, contrairement à
    celui de valuation_gap_sector_neutral (écart à la médiane du secteur)."""
    assert (ValuationGapCombinedStrategy().entry_threshold_pct
            == STRATEGY_REGISTRY["valuation_gap_dcf"]().entry_threshold_pct)


def test_seules_les_candidates_au_dessus_du_seuil_sont_retenues():
    strategie = ValuationGapCombinedStrategy(entry_threshold_pct=20.0)
    poids = strategie.generate_target_weights(
        _signaux({"AAA": 50.0, "BBB": 25.0, "CCC": 5.0, "DDD": -30.0}), set())

    assert set(poids) == {"AAA", "BBB"}, "une ligne sous le seuil a été retenue"


def test_les_poids_suivent_l_ampleur_de_l_ecart():
    """Plafond volontairement DÉSACTIVÉ pour ce test. Avec le plafond par
    défaut et deux candidates seulement, capped_weights applique sa règle
    documentée -- plafond inatteignable, donc chaque ligne prend le plafond et
    le reste va en cash -- et les poids ressortent égaux. On mesurerait alors
    le plafond, pas la pondération."""
    strategie = ValuationGapCombinedStrategy(entry_threshold_pct=20.0, max_weight_pct=0)
    poids = strategie.generate_target_weights(
        _signaux({"AAA": 60.0, "BBB": 30.0, "CCC": 5.0}), set())

    assert set(poids) == {"AAA", "BBB"}
    # L'ORDRE, pas le rapport exact : la correction d'inflation est
    # multiplicative ((1+g)(1+pi)^T - 1), donc elle ne conserve pas un rapport
    # de 2 entre deux écarts. Exiger 2,000 testerait la formule d'inflation,
    # pas la pondération.
    assert poids["AAA"] > poids["BBB"]
    assert poids["AAA"] + poids["BBB"] == pytest.approx(1.0, rel=1e-9)


def test_le_plafond_par_position_s_applique():
    strategie = ValuationGapCombinedStrategy(entry_threshold_pct=0.0, max_weight_pct=5.0)
    poids = strategie.generate_target_weights(
        _signaux({f"S{i}": 10.0 * (i + 1) for i in range(30)}), set())
    assert max(poids.values()) <= 0.05 + 1e-9


def test_aucune_candidate_ne_produit_aucun_poids():
    strategie = ValuationGapCombinedStrategy(entry_threshold_pct=500.0)
    assert strategie.generate_target_weights(_signaux({"AAA": 10.0}), set()) == {}


# --------------------------------------------------------------------------- #
# Construction du signal
# --------------------------------------------------------------------------- #
def test_le_signal_combine_porte_le_schema_attendu_par_le_moteur():
    """Le moteur et les stratégies lisent `valuation_dcf_per_share` : la
    colonne garde ce nom et reçoit la valeur théorique COMBINÉE. La renommer
    partout aurait cassé la relecture des runs archivés pour un gain
    cosmétique."""
    brut = pd.DataFrame([{
        "symbol": "AAA", "sector": "Technologie", "period_type": "FY",
        "fiscal_year": 2019, "filed_date": pd.Timestamp("2020-06-01"),
        "close": 100.0, "valuation_theoretical_per_share": 150.0,
        "valuation_dcf_per_share": 999.0,  # doit être IGNORÉ au profit du combiné
        "gap_pct": 50.0, "source": "multiples",
    }])
    evenements = data_loader.build_combined_signal_events(brut)

    for colonne in ("symbol", "published_date", "fiscal_year", "sector",
                    "close_at_filing", "valuation_dcf_per_share", "gap_pct"):
        assert colonne in evenements.columns
    assert evenements["valuation_dcf_per_share"].iloc[0] == 150.0, (
        "la valeur théorique combinée doit remplacer le DCF, pas coexister avec lui"
    )


def test_une_ligne_sans_ecart_est_ecartee():
    brut = pd.DataFrame([{
        "symbol": "AAA", "sector": "Technologie", "fiscal_year": 2019,
        "filed_date": pd.Timestamp("2020-06-01"), "close": 100.0,
        "valuation_theoretical_per_share": None, "gap_pct": None,
    }])
    assert data_loader.build_combined_signal_events(brut).empty


def test_le_dispatch_par_source_refuse_une_valeur_inconnue():
    """Le point d'entrée unique doit échouer bruyamment : une source inconnue
    qui retomberait silencieusement sur le DCF ferait tourner une stratégie
    sur un signal qui n'est pas le sien."""
    with pytest.raises(ValueError, match="Source de signal inconnue"):
        data_loader.build_strategy_signal_events("multiples_maison")


def _contenu_lfs_disponible(*chemins) -> bool:
    """Vrai seulement si ces fichiers portent leur CONTENU et non un pointeur.

    Tester `.exists()` ne suffit pas : un dépôt cloné sans `git lfs pull` a bien
    les fichiers, mais ils contiennent 130 octets de pointeur, et pandas échoue
    dessus par une erreur peu parlante (« Parquet magic bytes not found »).
    C'est l'état normal d'un clone frais, pas une panne -- le test doit donc
    être SAUTÉ, pas rouge."""
    for chemin in chemins:
        if not chemin.exists():
            return False
        with open(chemin, "rb") as f:
            if f.read(40).startswith(b"version https://git-lfs"):
                return False
    return True


@pytest.mark.skipif(
    not _contenu_lfs_disponible(config.VALORISATION_COMBINEE_FILE, config.DCF_HISTORY_FILE),
    reason="contenu LFS non rapatrié (dépôt cloné sans `git lfs pull`)",
)
def test_le_signal_combine_couvre_plus_d_entreprises_que_le_DCF_seul():
    """LE fait qui justifie cette stratégie, vérifié sur les données réelles :
    un DCF n'existe pas pour une entreprise à flux négatifs ni pour un métier
    de bilan, un multiple sectoriel si."""
    combine = data_loader.build_strategy_signal_events("combinee")
    dcf = data_loader.build_strategy_signal_events("dcf")
    assert combine["symbol"].nunique() >= dcf["symbol"].nunique()
    assert len(combine) > len(dcf) * 0.9
