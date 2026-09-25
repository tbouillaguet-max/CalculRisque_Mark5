"""Les fichiers de reprise ne gardent pas de doublons, et une ligne tronquée ne fait rien planter.

Un run interrompu puis repris (--resume) refait jusqu'à neuf unités -- la liste
des éléments traités n'est sauvegardée que toutes les dix -- et réécrit leurs
lignes. 04c, 07b et 08 recopiaient ensuite leur checkpoint tel quel dans leur
fichier de sortie. La mémoire des 8-K de 04c, elle, gardait chaque verdict
remplacé. Voir reprise_jsonl.py.
"""

from __future__ import annotations

import importlib
import json
import random

import pytest

import reprise_jsonl

_04c = importlib.import_module("04c_recuperation_8k")
_07b = importlib.import_module("07b_validation_qualitative")
_08 = importlib.import_module("08_recuperation_options")


def _ecrire(chemin, lignes, fin=""):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text("".join(json.dumps(l) + "\n" for l in lignes) + fin, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Le module commun
# --------------------------------------------------------------------------- #
def test_la_derniere_ecriture_gagne_a_la_place_de_la_premiere():
    lignes = [{"k": "a", "v": 1}, {"k": "b", "v": 1}, {"k": "a", "v": 2}]
    uniques, doublons = reprise_jsonl.dedoublonner(lignes, lambda l: l["k"])
    assert uniques == [{"k": "a", "v": 2}, {"k": "b", "v": 1}] and doublons == 1


def test_une_ligne_tronquee_est_ignoree_pas_fatale(tmp_path):
    chemin = tmp_path / "ckpt.jsonl"
    _ecrire(chemin, [{"k": "a"}, [1, 2]], fin='{"k": "b", "v"')   # liste, puis ligne coupée
    assert reprise_jsonl.lire_lignes(chemin) == ([{"k": "a"}], 2)


# --------------------------------------------------------------------------- #
# Les trois checkpoints
# --------------------------------------------------------------------------- #
def test_04c_un_8k_refait_apres_reprise_ne_sort_qu_une_fois(tmp_path):
    _ecrire(_04c._checkpoint_path(tmp_path), [
        {"symbol": "AAPL", "accession_number": "1", "category": "autre_materiel"},
        {"symbol": "AAPL", "accession_number": "2", "category": "rachat_actions"},
        {"symbol": "AAPL", "accession_number": "1", "category": "autre_materiel"},
    ])
    assert [r["accession_number"] for r in _04c.load_checkpoint_rows(tmp_path)] == ["1", "2"]


def test_07b_une_periode_refaite_ne_sort_qu_une_fois_meme_apres_un_crash(tmp_path):
    """Le cas complet : doublons ET dernière ligne tronquée. Avant, json.loads
    levait et le --resume plantait au moment d'écrire la sortie."""
    periode = {"symbol": "AAPL", "period_type": "10-K", "fiscal_year": 2023, "fiscal_quarter": None}
    _ecrire(_07b._checkpoint_path(tmp_path), [
        {**periode, "verdict": "coherent"},
        {**periode, "fiscal_year": 2024, "verdict": "coherent"},
        {**periode, "verdict": "incoherent"},
    ], fin='{"symbol": "MSFT", "verd')
    lignes = _07b.load_checkpoint_rows(tmp_path)
    assert [(l["fiscal_year"], l["verdict"]) for l in lignes] == [(2023, "incoherent"), (2024, "coherent")]


def test_08_un_contrat_refait_garde_sa_version_la_plus_fraiche(tmp_path):
    contrat = {"symbol": "AAPL", "expiry": "20270115", "strike": 200.0, "option_type": "CALL",
               "trading_class": "AAPL", "con_id": 111}
    _ecrire(_08._checkpoint_path(tmp_path), [
        {**contrat, "bid": 10.0},
        {**contrat, "con_id": 222, "strike": 210.0, "bid": 7.0},   # un autre contrat
        {**contrat, "bid": 10.5},
    ])
    lignes = _08.load_checkpoint_rows(tmp_path)
    assert [(l["con_id"], l["bid"]) for l in lignes] == [(111, 10.5), (222, 7.0)]


# --------------------------------------------------------------------------- #
# La mémoire des 8-K de 04c
# --------------------------------------------------------------------------- #
def _verdict(sym, acc, source, n=0):
    return {"symbol": sym, "accession_number": acc, "classification_source": source,
            "category": f"{source}-{n}"}


def _reference(entrees, llm):
    """La sémantique de load_llm_cache AVANT nettoyage : dernière écriture
    gagnante, verdicts par règles ignorés quand un modèle est disponible."""
    cache = {}
    for e in entrees:
        if llm and e["classification_source"] == "regles_document":
            continue
        cache[_04c.cache_key(e["symbol"], e["accession_number"])] = e
    return cache


def _charger(tmp_path, monkeypatch, llm):
    monkeypatch.setattr(_04c.sft, "llm_disponible", lambda: llm)
    return _04c.load_llm_cache(tmp_path)


def test_la_memoire_des_8k_est_nettoyee_sans_rien_changer_a_ce_qu_elle_rend(tmp_path, monkeypatch):
    entrees = [
        _verdict("A", "1", "gemini", 1),
        _verdict("A", "1", "gemini", 2),          # ré-analyse (--no-llm-cache) : remplace
        _verdict("B", "1", "regles_document"),
        _verdict("B", "1", "gemini"),             # repris par le modèle quand la clé est arrivée
        _verdict("C", "1", "mistral"),
        _verdict("C", "1", "regles_document"),    # ré-analyse SANS clé : plus récent que le modèle
        _verdict("D", "1", "regles_document"),
    ]
    chemin = _04c.llm_cache_path(tmp_path)
    _ecrire(chemin, entrees + [{"symbol": "E"}], fin='{"symbol": "F", "acc')

    sans_cle = _charger(tmp_path, monkeypatch, llm=False)
    assert sans_cle == _reference(entrees, llm=False)
    nettoye, illisibles = reprise_jsonl.lire_lignes(chemin)
    assert illisibles == 0
    assert [(e["symbol"], e["classification_source"]) for e in nettoye] == [
        ("A", "gemini"), ("B", "gemini"), ("C", "mistral"), ("C", "regles_document"),
        ("D", "regles_document")]

    # Le verdict du modèle de C a survécu : avec une clé, c'est lui qui sert.
    assert _charger(tmp_path, monkeypatch, llm=True) == _reference(entrees, llm=True)


def test_une_memoire_sans_doublon_n_est_pas_reecrite(tmp_path, monkeypatch):
    chemin = _04c.llm_cache_path(tmp_path)
    _ecrire(chemin, [_verdict("A", "1", "gemini"), _verdict("B", "1", "regles_document")])
    avant = chemin.read_bytes()
    _charger(tmp_path, monkeypatch, llm=False)
    assert chemin.read_bytes() == avant


@pytest.mark.parametrize("graine", range(20))
def test_le_nettoyage_preserve_la_memoire_dans_les_deux_modes(tmp_path, monkeypatch, graine):
    """Sur des historiques tirés au hasard -- ré-analyses, reprises par le
    modèle, retours aux règles --, la mémoire rendue après nettoyage est
    celle d'avant, avec comme sans clé d'API."""
    tirage = random.Random(graine)
    entrees = [
        _verdict(tirage.choice("ABCD"), tirage.choice("12"),
                 tirage.choice(["gemini", "mistral", "regles_document"]), n)
        for n in range(tirage.randint(1, 25))
    ]
    _ecrire(_04c.llm_cache_path(tmp_path), entrees)
    for llm in (False, True, False):          # le fichier est nettoyé au premier chargement
        assert _charger(tmp_path, monkeypatch, llm) == _reference(entrees, llm)
