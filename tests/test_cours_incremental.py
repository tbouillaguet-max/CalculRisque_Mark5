"""03_recuperation_cours.py : ne pas retélécharger ce qui est déjà en cache,
et ne pas perdre le travail déjà fait en cas d'interruption."""

from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest

module_03 = importlib.import_module("03_recuperation_cours")
decider = module_03.determine_fetch_start_year


def cache(symbol: str, annees, derniere_date: str | None = None) -> pd.DataFrame:
    lignes = [
        {"symbol": symbol, "year": annee, "date": pd.Timestamp(f"{annee}-12-31"), "close": 100.0}
        for annee in annees
    ]
    if derniere_date and lignes:
        lignes[-1]["date"] = pd.Timestamp(derniere_date)
    return pd.DataFrame(lignes)


VIDE = pd.DataFrame(columns=["symbol", "year", "date", "close"])


# --------------------------------------------------------------------------- #
# Le bug : une année qui n'existera jamais forçait un retéléchargement complet
# --------------------------------------------------------------------------- #

def test_une_ipo_recente_ne_force_plus_un_retelechargement_complet():
    """Une entreprise entrée en bourse en 2014 n'aura JAMAIS de cours
    2010-2013. L'ancienne version les voyait "manquants" à chaque run et
    repartait de 2010 : 17 années retéléchargées, indéfiniment."""
    complet = cache("X", range(2014, 2027), derniere_date="2026-12-31")
    assert decider(complet, "X", 2010, 2026, 1, requested_from=2010) is None


def test_seule_la_nouvelle_annee_est_reclamee():
    """Le cache s'arrête en 2025, on est en 2026 : une seule année à
    récupérer, pas seize."""
    assert decider(cache("X", range(2014, 2026)), "X", 2010, 2026, 1, requested_from=2010) == 2026


def test_l_annee_en_cours_est_rafraichie_si_elle_a_vieilli():
    perime = cache("X", range(2014, 2027), derniere_date="2026-01-15")
    assert decider(perime, "X", 2010, 2026, 30, requested_from=2010) == 2026


def test_l_annee_en_cours_fraiche_ne_declenche_rien():
    frais = cache("X", range(2014, 2027), derniere_date="2026-12-31")
    assert decider(frais, "X", 2010, 2026, 30, requested_from=2010) is None


def test_un_symbole_absent_du_cache_est_recupere_entierement():
    assert decider(VIDE, "NOUVEAU", 2010, 2026, 1) == 2010


def test_abaisser_start_year_redemande_l_historique():
    """Seul cas où redescendre en arrière a un sens : l'utilisateur élargit
    la fenêtre, ces années n'ont jamais été réclamées."""
    complet = cache("X", range(2014, 2027), derniere_date="2026-12-31")
    assert decider(complet, "X", 2005, 2026, 1, requested_from=2010) == 2005


def test_sans_etat_de_suivi_on_ne_retelecharge_pas_l_historique_ancien():
    """Cache antérieur à l'introduction de l'état de suivi : on ne peut pas
    savoir si les années absentes ont déjà été réclamées. On s'abstient --
    quitte à manquer quelques années anciennes, plutôt que de retélécharger
    tout l'univers à chaque run. --force-refresh reste disponible."""
    complet = cache("X", range(2014, 2027), derniere_date="2026-12-31")
    assert decider(complet, "X", 2010, 2026, 1, requested_from=None) is None


def test_un_trou_au_milieu_du_cache_n_est_pas_recomble():
    """Une année absente au milieu d'une plage déjà réclamée a été demandée
    et IBKR n'avait rien : la redemander à chaque run ne changera rien.

    L'année en cours est fraîche ici, pour isoler le trou de 2016 du
    rafraîchissement du millésime courant."""
    annees = [a for a in range(2014, 2027) if a != 2016]
    troue = cache("X", annees, derniere_date="2026-12-31")
    assert decider(troue, "X", 2010, 2026, 30, requested_from=2010) is None


# --------------------------------------------------------------------------- #
# État de suivi
# --------------------------------------------------------------------------- #

def test_l_etat_survit_a_un_aller_retour_disque(tmp_path):
    etat = {"AAPL": {"requested_from": 2010, "last_attempt": "2026-08-09T12:00:00"}}
    module_03.save_fetch_state(tmp_path, etat)
    assert module_03.load_fetch_state(tmp_path) == etat


def test_un_etat_illisible_ne_bloque_pas_le_run(tmp_path, caplog):
    module_03.fetch_state_path(tmp_path).write_text("{ pas du json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert module_03.load_fetch_state(tmp_path) == {}
    assert any("illisible" in m for m in caplog.messages)


def test_etat_absent_donne_un_dictionnaire_vide(tmp_path):
    assert module_03.load_fetch_state(tmp_path) == {}


# --------------------------------------------------------------------------- #
# Sauvegarde incrémentale
# --------------------------------------------------------------------------- #

def _lignes(symbol: str, annees, close: float = 100.0) -> list[dict]:
    return [
        {"symbol": symbol, "ric": symbol, "company_name": symbol, "country": "United States",
         "currency": "USD", "exchange": "SMART", "year": a, "date": f"{a}-12-31", "close": close}
        for a in annees
    ]


def test_la_sauvegarde_ecrit_reellement_le_fichier(tmp_path):
    fichier = tmp_path / "year_end_prices.parquet"
    combine = module_03.merge_and_save(VIDE, _lignes("AAA", [2024, 2025]), fichier)

    assert fichier.exists()
    assert len(combine) == 2
    assert len(pd.read_parquet(fichier)) == 2


def test_les_sauvegardes_successives_s_accumulent(tmp_path):
    """C'est ce qui permet de reprendre après une interruption : chaque point
    de sauvegarde contient tout ce qui précède."""
    fichier = tmp_path / "year_end_prices.parquet"
    etape1 = module_03.merge_and_save(VIDE, _lignes("AAA", [2024, 2025]), fichier)
    etape2 = module_03.merge_and_save(etape1, _lignes("BBB", [2024, 2025]), fichier)

    sur_disque = pd.read_parquet(fichier)
    assert len(etape2) == 4
    assert set(sur_disque["symbol"]) == {"AAA", "BBB"}


def test_une_nouvelle_valeur_remplace_l_ancienne(tmp_path):
    """Le rafraîchissement de l'année en cours doit écraser le point
    provisoire, pas le dupliquer."""
    fichier = tmp_path / "year_end_prices.parquet"
    etape1 = module_03.merge_and_save(VIDE, _lignes("AAA", [2026], close=100.0), fichier)
    etape2 = module_03.merge_and_save(etape1, _lignes("AAA", [2026], close=142.0), fichier)

    assert len(etape2) == 1
    assert etape2["close"].iloc[0] == 142.0


def test_sans_nouvelle_ligne_le_cache_est_rendu_tel_quel(tmp_path):
    fichier = tmp_path / "year_end_prices.parquet"
    existant = pd.DataFrame(_lignes("AAA", [2024]))
    assert module_03.merge_and_save(existant, [], fichier) is existant
    assert not fichier.exists(), "aucune écriture inutile"
