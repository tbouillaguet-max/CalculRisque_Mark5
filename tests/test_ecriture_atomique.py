"""Remplacement atomique qui survit aux verrous transitoires de Windows.

CE QUI L'A MOTIVÉ. Run quotidien du 2026-09-05 : `07b_validation_qualitative.py`
tombe sur `PermissionError: [WinError 5] Access is denied` en renommant
`progress_qualitative.json.tmp` vers `progress_qualitative.json`. Sur Windows, ce
renommage échoue dès qu'un autre processus tient la cible ouverte -- antivirus,
indexeur, OneDrive (qui synchronise le Bureau, où vit le dépôt). Le verrou dure
quelques millisecondes ; il suffisait à tuer l'étape.

LE TEST QUI COMPTE est `test_aucun_script_ne_renomme_encore_sans_reprise` : le même
motif fragile existait à NEUF endroits du dépôt, et le prochain fichier d'état
ajouté le reproduirait sans ce garde-fou.
"""

from __future__ import annotations

import os
import pathlib
import re

import pytest

import ecriture_atomique

RACINE = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _sans_attente(monkeypatch):
    """Les pauses réelles (jusqu'à 3 s) n'apportent rien à ces tests."""
    monkeypatch.setattr(ecriture_atomique.time, "sleep", lambda s: None)


def _verrou_transitoire(monkeypatch, echecs: int, erreur=PermissionError):
    """os.replace qui échoue `echecs` fois, puis réussit."""
    vrai = os.replace
    appels = {"n": 0}

    def faux(src, dst):
        appels["n"] += 1
        if appels["n"] <= echecs:
            raise erreur("[WinError 5] Access is denied")
        return vrai(src, dst)

    monkeypatch.setattr(ecriture_atomique.os, "replace", faux)
    return appels


def test_un_verrou_transitoire_ne_fait_plus_tomber_l_ecriture(tmp_path, monkeypatch):
    """LE cas du 2026-09-05 : quelques refus, puis la cible se libère."""
    appels = _verrou_transitoire(monkeypatch, echecs=3)
    ecriture_atomique.ecrire_texte(tmp_path / "progress.json", '{"ok": true}')

    assert (tmp_path / "progress.json").read_text(encoding="utf-8") == '{"ok": true}'
    assert appels["n"] == 4
    assert not (tmp_path / "progress.json.tmp").exists()


def test_un_verrou_qui_persiste_remonte_au_lieu_d_etre_masque(tmp_path, monkeypatch):
    """Au-delà de trois secondes, le fichier est vraiment tenu ouvert. Avaler
    l'erreur ferait croire à une sauvegarde qui n'a pas eu lieu."""
    appels = _verrou_transitoire(monkeypatch, echecs=99)
    with pytest.raises(PermissionError):
        ecriture_atomique.ecrire_texte(tmp_path / "progress.json", "x")
    assert appels["n"] == ecriture_atomique.TENTATIVES


def test_seule_une_permissionerror_est_reessayee(tmp_path, monkeypatch):
    """Un disque plein ou une source absente ne se résolvent pas en attendant :
    les réessayer retarderait une erreur qui doit remonter tout de suite."""
    appels = _verrou_transitoire(monkeypatch, echecs=99, erreur=OSError)
    with pytest.raises(OSError):
        ecriture_atomique.ecrire_texte(tmp_path / "progress.json", "x")
    assert appels["n"] == 1


def test_l_attente_double_a_chaque_essai(tmp_path, monkeypatch):
    """0,1 + 0,2 + 0,4 + 0,8 + 1,6 s : un peu plus de trois secondes au total.
    Assez pour un scan d'antivirus ou une synchronisation OneDrive, pas assez
    pour qu'un vrai blocage passe inaperçu."""
    attentes = []
    monkeypatch.setattr(ecriture_atomique.time, "sleep", attentes.append)
    _verrou_transitoire(monkeypatch, echecs=99)
    with pytest.raises(PermissionError):
        ecriture_atomique.ecrire_texte(tmp_path / "x.json", "x")
    assert attentes == pytest.approx([0.1, 0.2, 0.4, 0.8, 1.6])
    assert sum(attentes) < 3.5


def test_le_fichier_temporaire_garde_le_nom_complet(tmp_path):
    """`x.json` -> `x.json.tmp`, comme le faisaient déjà les scripts : changer
    ce nom laisserait traîner les anciens temporaires au prochain run."""
    cible = tmp_path / "progress_qualitative.json"
    ecriture_atomique.ecrire_texte(cible, "{}")
    assert cible.exists()
    assert not (tmp_path / "progress_qualitative.json.tmp").exists()


# --------------------------------------------------------------------------- #
# Le garde-fou
# --------------------------------------------------------------------------- #
def test_aucun_script_ne_renomme_encore_sans_reprise():
    """LE point. Le même `tmp.replace(path)` existait à neuf endroits : 03, 04
    (x2), 04b (x2), 04c, 07b, 08 et le rapport de l'orchestrateur. Tous passent
    désormais par ecriture_atomique ; un fichier d'état ajouté demain doit
    faire de même, et ce test le lui rappellera."""
    motif = re.compile(r"\b\w+\.replace\((path|target|cible|destination)\)")
    fautifs = []
    for fichier in sorted(RACINE.glob("*.py")) + sorted((RACINE / "backtest").rglob("*.py")):
        if fichier.name == "ecriture_atomique.py":
            continue
        for n, ligne in enumerate(fichier.read_text(encoding="utf-8").splitlines(), 1):
            if motif.search(ligne) and "str" not in ligne.split(".replace")[0][-6:]:
                fautifs.append(f"{fichier.relative_to(RACINE)}:{n}: {ligne.strip()}")
    assert not fautifs, (
        "Renommage atomique sans reprise -- sur Windows, un antivirus ou OneDrive suffit à "
        "le faire échouer. Utilise ecriture_atomique.remplacer(tmp, path) :\n" + "\n".join(fautifs)
    )


def test_la_sauvegarde_de_07b_survit_a_un_verrou(tmp_path, monkeypatch):
    """Le fichier et la fonction exacts qui sont tombés le 2026-09-05."""
    import importlib
    mod = importlib.import_module("07b_validation_qualitative")
    _verrou_transitoire(monkeypatch, echecs=2)

    mod.save_progress(tmp_path, {"AAA|FY|2024", "BBB|FY|2024"})

    assert mod.load_progress(tmp_path) == {"AAA|FY|2024", "BBB|FY|2024"}
