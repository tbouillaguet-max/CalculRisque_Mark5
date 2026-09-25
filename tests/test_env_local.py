"""Une clé posée dans `.env` est vue par tous les scripts, d'où qu'ils soient lancés.

Constaté : « Aucune clé LLM » alors que la clé avait bien été « mise ». Les
scripts ne lisaient que l'environnement du processus, et sous Windows `setx`
n'agit que sur les terminaux ouverts APRÈS lui, `$env:` que sur la fenêtre
courante, `set` de cmd pas du tout sous PowerShell. Voir env_local.py.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

import env_local
import sec_filings_text as sft

RACINE = pathlib.Path(__file__).resolve().parent.parent


def _env(tmp_path, texte: str, bom: bool = False) -> pathlib.Path:
    chemin = tmp_path / ".env"
    chemin.write_text(("﻿" if bom else "") + texte, encoding="utf-8")
    return chemin


def test_la_syntaxe_tolere_les_formes_courantes(tmp_path):
    chemin = _env(tmp_path, "\n".join([
        "# commentaire",
        "GEMINI_API_KEY=AIza-sans-guillemets",
        'MISTRAL_API_KEY="entre-guillemets"',
        "export SEC_CONTACT_EMAIL = moi@exemple.fr",
        "GEMINI_MODEL=gemini-2.5-flash # commentaire en fin de ligne",
        "ligne sans egal",
        "",
    ]), bom=True)
    assert env_local.lire(chemin) == {
        "GEMINI_API_KEY": "AIza-sans-guillemets", "MISTRAL_API_KEY": "entre-guillemets",
        "SEC_CONTACT_EMAIL": "moi@exemple.fr", "GEMINI_MODEL": "gemini-2.5-flash",
    }


def test_seules_les_variables_autorisees_sont_chargees(tmp_path, monkeypatch):
    """Le mot de passe IB du même fichier n'a rien à faire dans l'environnement
    de tous les scripts et de leurs sous-processus."""
    for nom in ("GEMINI_API_KEY", "IB_PASSWORD"):
        monkeypatch.delenv(nom, raising=False)
    chemin = _env(tmp_path, "GEMINI_API_KEY=cle-du-fichier\nIB_PASSWORD=secret\n")
    assert env_local.charger(chemin) == ["GEMINI_API_KEY"]
    assert os.environ["GEMINI_API_KEY"] == "cle-du-fichier"
    assert "IB_PASSWORD" not in os.environ
    monkeypatch.delenv("GEMINI_API_KEY")


def test_l_environnement_l_emporte_sur_le_fichier(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "cle-de-l-environnement")
    assert env_local.charger(_env(tmp_path, "GEMINI_API_KEY=cle-du-fichier\n")) == []
    assert os.environ["GEMINI_API_KEY"] == "cle-de-l-environnement"


def test_une_valeur_vide_ne_masque_rien(tmp_path, monkeypatch):
    """`GEMINI_API_KEY=` (le modèle tel quel) ne doit pas passer pour une clé."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert env_local.charger(_env(tmp_path, "GEMINI_API_KEY=\n")) == []
    assert not sft.llm_disponible()


def test_un_script_lance_ailleurs_voit_la_cle_du_fichier(tmp_path):
    """De bout en bout, comme sous Windows : un processus neuf, sans la
    variable dans son environnement, doit voir la clé posée dans .env --
    parce que config.py charge le fichier à l'import."""
    code = (
        "import env_local, pathlib, os;"
        "env_local.FICHIER = pathlib.Path(os.environ['FICHIER_ENV_TEST']);"
        "import config, sec_filings_text as s;"
        "print(s.description_llm())"
    )
    chemin = _env(tmp_path, "GEMINI_API_KEY=cle-du-fichier\n")
    env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "MISTRAL_API_KEY", "LLM_PROVIDER")}
    # config.py appelle env_local.charger() sans argument, qui lit FICHIER :
    # redirigé avant l'import de config, pour ne pas dépendre d'un vrai .env.
    resultat = subprocess.run(
        ["python", "-c", code], cwd=tmp_path, capture_output=True, text=True,
        env={**env, "PYTHONPATH": str(RACINE), "FICHIER_ENV_TEST": str(chemin)})
    assert resultat.returncode == 0, resultat.stderr[-500:]
    assert resultat.stdout.strip().startswith("Gemini"), resultat.stdout


def test_l_aide_montre_le_bon_chemin_et_detecte_une_cle_collee_dans_le_code(monkeypatch):
    assert ".env" in sft.aide_cle_absente() and "$env:GEMINI_API_KEY" in sft.aide_cle_absente()
    monkeypatch.setattr(sft, "GEMINI_API_KEY_ENV", "AIzaSyExempleDeCleCollee")
    assert "sec_filings_text.py a été modifié" in sft.aide_cle_absente()


def test_les_constantes_sont_des_noms_de_variables_pas_des_cles():
    """Coller la clé à la place du nom fait chercher une variable qui porte ce
    nom -- elle n'existe pas, et le LLM est jugé indisponible."""
    assert (sft.GEMINI_API_KEY_ENV, sft.MISTRAL_API_KEY_ENV) == ("GEMINI_API_KEY", "MISTRAL_API_KEY")


def test_le_fichier_env_ne_part_jamais_sur_git_et_le_modele_est_vide():
    ignore = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=RACINE)
    assert ignore.returncode == 0, ".env n'est plus ignoré par git"
    modele = env_local.lire(RACINE / ".env.example")
    assert modele["GEMINI_API_KEY"] == "" and "@" in modele["SEC_CONTACT_EMAIL"]
    texte = (RACINE / ".env.example").read_text(encoding="utf-8")
    assert [nom for nom in env_local.LISTE_BLANCHE if nom not in texte] == [], \
        "une variable lue dans .env n'est pas documentée dans .env.example"
