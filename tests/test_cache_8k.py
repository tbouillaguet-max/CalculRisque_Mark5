"""Mémoire des 8-K déjà classifiés (04c_recuperation_8k.py).

Un 8-K est un document figé : une fois classifié avec succès, il ne doit plus
jamais être re-téléchargé ni re-soumis au LLM.
"""

from __future__ import annotations

import importlib
import json

import pytest

module_04c = importlib.import_module("04c_recuperation_8k")


FILING = {
    "form": "8-K", "filing_date": "2023-05-04",
    "accession_number": "0000066740-23-000042", "primary_document": "d8k.htm",
}
VERDICT = {
    "item_codes": ["Item 5.02"], "category": "depart_dirigeant",
    "materiality": True, "summary": "Départ du directeur financier.",
    "classification_source": "gemini",
}


@pytest.fixture
def sec(monkeypatch):
    """Remplace la couche SEC + LLM de 04c, et compte ce qui a réellement été
    téléchargé et classifié par Gemini (en 8-K, pas en requêtes)."""
    compteurs = {"telechargements": 0, "classifications": 0}
    sft = module_04c.sft

    monkeypatch.setattr(sft, "fetch_submissions_strict", lambda cik: [FILING])
    monkeypatch.setattr(sft, "filter_filings", lambda subs, **kw: list(subs))
    monkeypatch.setattr(sft, "filing_document_url", lambda *a: "https://sec.example/d8k.htm")

    def faux_fetch(url, form=None):
        compteurs["telechargements"] += 1
        return ("Item 5.02 Departure of Officers.", "brut")

    def faux_lot(documents, consigne, schema_element, max_tokens_par_document=150):
        compteurs["classifications"] += len(documents)
        return {ident: {"category": "depart_dirigeant", "materiality": True,
                        "summary": "Départ du directeur financier."} for ident in documents}

    monkeypatch.setattr(sft, "fetch_filing_text", faux_fetch)
    monkeypatch.setattr(sft, "texte_piece_jointe", lambda *a, **k: None)
    monkeypatch.setattr(sft, "analyser_documents", faux_lot)
    return compteurs


def _run(cache, tmp_path):
    """Les deux temps de 04c pour MMM : classement par règles et mise en
    attente, puis le lot pour Gemini. Rend (lignes finales, 8-K servis par la
    mémoire) ; la ligne de Gemini remplace celle des règles, comme dans le
    fichier de sortie."""
    attente = []
    rows, hits = module_04c.process_ticker_8k(
        "MMM", "0000066740", [("2023-01-01", "2023-12-31")], cache, tmp_path, None, attente)
    finales = {row["accession_number"]: row for row in rows}
    finales.update({ligne["accession_number"]: ligne
                    for ligne in module_04c.classer_en_attente(attente, cache, tmp_path)})
    return list(finales.values()), hits


def test_un_8k_deja_classifie_n_est_ni_retelecharge_ni_reanalyse(sec, tmp_path):
    rows, hits = _run({}, tmp_path)
    assert hits == 0
    assert sec == {"telechargements": 1, "classifications": 1}
    assert (rows[0]["category"], rows[0]["classification_source"]) == ("depart_dirigeant", "gemini")

    # Second run : le cache relu depuis le disque, comme au démarrage réel.
    relu = module_04c.load_llm_cache(tmp_path)
    rows2, hits2 = _run(relu, tmp_path)

    assert hits2 == 1
    assert sec == {"telechargements": 1, "classifications": 1}, "aucun nouvel appel attendu"
    assert rows2[0]["category"] == "depart_dirigeant"
    assert rows2[0]["materiality"] is True
    assert rows2[0]["from_cache"] is True
    assert rows2[0]["classification_source"] == "gemini"


def test_un_verdict_manquant_n_est_pas_memorise(sec, tmp_path, monkeypatch):
    """Pas de verdict du modèle (quota épuisé, clé absente) : le 8-K garde son
    verdict par règles, qui est mémorisé, car déterministe."""
    monkeypatch.setattr(module_04c.sft, "analyser_documents", lambda *a, **k: None)

    cache = {}
    rows, _ = _run(cache, tmp_path)

    assert rows[0]["category"] != "non_evalue"
    assert rows[0]["classification_source"] == "regles_document"
    # Mémorisé, car déterministe : c'est le TÉLÉCHARGEMENT qu'on évite de
    # repayer, pas le calcul de la règle.
    assert len(cache) == 1
    assert module_04c.llm_cache_path(tmp_path).exists()


def test_no_llm_cache_force_la_reanalyse(sec, tmp_path):
    """llm_cache=None (--no-llm-cache) : ni lecture ni écriture de la mémoire."""
    module_04c.append_llm_cache(tmp_path, {"symbol": "MMM", **FILING, **VERDICT})

    rows, hits = _run(None, tmp_path)

    assert hits == 0
    assert sec == {"telechargements": 1, "classifications": 1}
    assert rows[0]["from_cache"] is False


def test_le_cache_est_ecrit_au_fil_de_l_eau(sec, tmp_path):
    """Un appel payé doit survivre à un Ctrl-C immédiat : l'écriture ne peut
    pas attendre la fin du run. Le verdict par règles d'abord, dès le
    téléchargement ; celui de Gemini dès la réponse à son lot."""
    attente = []
    module_04c.process_ticker_8k(
        "MMM", "0000066740", [("2023-01-01", "2023-12-31")], {}, tmp_path, None, attente)

    chemin = module_04c.llm_cache_path(tmp_path)
    lignes = chemin.read_text(encoding="utf-8").strip().splitlines()
    assert len(lignes) == 1
    entree = json.loads(lignes[0])
    assert entree["accession_number"] == FILING["accession_number"]
    assert entree["symbol"] == "MMM"
    assert entree["classification_source"] == "regles_document"

    module_04c.classer_en_attente(attente, {}, tmp_path)
    lignes = chemin.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(l)["classification_source"] for l in lignes] == ["regles_document", "gemini"]


def test_une_ligne_corrompue_ne_perd_pas_tout_le_cache(tmp_path, caplog):
    path = module_04c.llm_cache_path(tmp_path)
    path.write_text(
        json.dumps({"symbol": "MMM", **FILING, **VERDICT}) + "\n"
        + '{"symbol": "AAPL", "accession_num'  # ligne tronquée par un kill
        + "\n" + json.dumps({"symbol": "IBM", **FILING, **VERDICT}) + "\n",
        encoding="utf-8",
    )
    cache = module_04c.load_llm_cache(tmp_path)
    assert set(cache) == {
        module_04c.cache_key("MMM", FILING["accession_number"]),
        module_04c.cache_key("IBM", FILING["accession_number"]),
    }


def test_la_derniere_ecriture_gagne(tmp_path):
    """Une ré-analyse (--no-llm-cache) doit primer sur l'ancien verdict sans
    qu'il faille réécrire le fichier."""
    path = module_04c.llm_cache_path(tmp_path)
    path.write_text(
        json.dumps({"symbol": "MMM", **FILING, **VERDICT}) + "\n"
        + json.dumps({"symbol": "MMM", **FILING, **{**VERDICT, "category": "non_materiel"}}) + "\n",
        encoding="utf-8",
    )
    cache = module_04c.load_llm_cache(tmp_path)
    assert cache[module_04c.cache_key("MMM", FILING["accession_number"])]["category"] == "non_materiel"


@pytest.mark.parametrize("categorie, memorisable", [
    ("depart_dirigeant", True), ("non_materiel", True),
    ("non_evalue", False), (None, False), ("", False),
])
def test_seuls_les_verdicts_reels_sont_memorisables(categorie, memorisable):
    assert module_04c.is_cacheable({"category": categorie}) is memorisable


# --------------------------------------------------------------------------- #
# Les verdicts de Mistral, l'ancien fournisseur
# --------------------------------------------------------------------------- #

def _entree(symbole, accession, **champs):
    return {"symbol": symbole, **FILING, "accession_number": accession, **VERDICT, **champs}


def test_les_verdicts_de_mistral_sont_ecartes_et_effaces(tmp_path):
    """Écartés à la demande : 1 647 verdicts de Mistral dans la mémoire du
    dépôt, 77 % jugés matériels, dont une catégorie inventée. Une entrée sans
    source date d'avant la colonne classification_source, donc de Mistral."""
    mistral_sans_source = _entree("ADBE", "1", category="aut_materiel")
    del mistral_sans_source["classification_source"]
    path = module_04c.llm_cache_path(tmp_path)
    path.write_text("".join(json.dumps(e) + "\n" for e in [
        mistral_sans_source,
        _entree("MMM", "2", classification_source="mistral"),
        _entree("ABT", "3"),
        _entree("AMD", "4", classification_source="regles_document",
                version_regles=module_04c.VERSION_REGLES),
    ]), encoding="utf-8")

    cache = module_04c.load_llm_cache(tmp_path)

    assert set(cache) == {"ABT:3", "AMD:4"}
    relues = [json.loads(ligne) for ligne in path.read_text(encoding="utf-8").splitlines()]
    assert [e["accession_number"] for e in relues] == ["3", "4"], "effacés du fichier, pas seulement ignorés"


def test_un_8k_classe_par_mistral_est_reclasse(sec, tmp_path, monkeypatch):
    """Le 8-K de Mistral repasse par le téléchargement et la classification,
    comme un neuf -- ici par Gemini, qui répond."""
    monkeypatch.setenv(module_04c.sft.GEMINI_API_KEY_ENV, "une-cle")
    ancien = {"symbol": "MMM", **FILING, **VERDICT, "classification_source": "mistral",
              "category": "aut_materiel"}
    module_04c.llm_cache_path(tmp_path).write_text(json.dumps(ancien) + "\n", encoding="utf-8")

    cache = module_04c.load_llm_cache(tmp_path)
    rows, hits = _run(cache, tmp_path)

    assert hits == 0
    assert sec == {"telechargements": 1, "classifications": 1}
    assert rows[0]["category"] == "depart_dirigeant"
    assert rows[0]["classification_source"] == "gemini"


def test_l_ancienne_memoire_est_renommee(tmp_path):
    ancien = tmp_path / module_04c.ANCIEN_LLM_CACHE_FILENAME
    ancien.write_text(json.dumps(_entree("MMM", "1")) + "\n", encoding="utf-8")

    cache = module_04c.load_llm_cache(tmp_path)

    assert not ancien.exists()
    assert module_04c.llm_cache_path(tmp_path).exists()
    assert set(cache) == {"MMM:1"}


def test_les_deux_memoires_sont_fusionnees_la_plus_recente_gagne(tmp_path):
    """Un run de l'ancienne version après un run de la nouvelle : rien n'est
    perdu, et le verdict le plus récent -- celui du nouveau fichier -- gagne."""
    ancien = tmp_path / module_04c.ANCIEN_LLM_CACHE_FILENAME
    ancien.write_text(json.dumps(_entree("MMM", "1", category="non_materiel"))   # sans \n final
                      , encoding="utf-8")
    nouveau = module_04c.llm_cache_path(tmp_path)
    nouveau.write_text(json.dumps(_entree("MMM", "1", category="rachat_actions")) + "\n"
                       + json.dumps(_entree("IBM", "2")) + "\n", encoding="utf-8")

    cache = module_04c.load_llm_cache(tmp_path)

    assert not ancien.exists()
    assert cache["MMM:1"]["category"] == "rachat_actions"
    assert set(cache) == {"MMM:1", "IBM:2"}
