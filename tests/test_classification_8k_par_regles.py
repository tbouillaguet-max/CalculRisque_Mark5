"""Classification des 8-K À PARTIR DU DOCUMENT, sans modèle de langage.

CE QUE CES TESTS PROTÈGENT. Sans MISTRAL_API_KEY, `classify_8k` renvoyait
`non_evalue` et jetait le texte qu'il venait de télécharger. Mesuré sur
l'archive du dépôt : 99 147 dépôts, `category` à `non_evalue` sur 100% des
lignes, `materiality` et `summary` vides partout -- le filtre d'événements
matériels du moteur ne s'appliquait donc à RIEN.

Le repli documentaire lit deux choses, toutes deux dans le texte : les codes
d'item que le déposant déclare lui-même, et des formulations caractéristiques
qui tranchent là où le code seul est ambigu. Les tests couvrent surtout ce
second point -- c'est là que la règle apporte quelque chose qu'une table de
codes ne donne pas, et c'est là qu'elle peut se tromper.
"""

from __future__ import annotations

import importlib

import pytest

_c8k = importlib.import_module("04c_recuperation_8k")


def classer(text: str) -> dict:
    """Passe par extract_item_codes, comme le fait classify_8k : les codes
    viennent du DOCUMENT, pas d'une liste fournie à côté."""
    return _c8k.classify_8k_par_regles(_c8k.extract_item_codes(text), text)


# --------------------------------------------------------------------------- #
# Ce que le code seul suffit à trancher
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code, attendue", [
    ("Item 4.02", "autre_materiel"),        # non-fiabilité des états financiers
    ("Item 2.06", "autre_materiel"),        # dépréciation
    ("Item 1.03", "procedure_judiciaire"),  # faillite
    ("Item 2.01", "fusion_acquisition"),    # acquisition réalisée
    ("Item 3.01", "autre_materiel"),        # radiation
])
def test_les_codes_non_ambigus_tranchent_seuls(code, attendue):
    """Ces codes sont matériels par définition SEC : aucune lecture requise."""
    resultat = classer(f"{code}\nThe registrant hereby furnishes the following.")
    assert resultat["materiality"] is True
    assert resultat["category"] == attendue
    assert resultat["classification_source"] == "regles_document"


@pytest.mark.parametrize("texte", [
    "Item 9.01 Financial Statements and Exhibits\nExhibit 99.1 is furnished herewith.",
    "Item 5.07 Submission of Matters to a Vote of Security Holders\n"
    "At the annual meeting, stockholders elected each of the nominees.",
    "Item 7.01 Regulation FD Disclosure\nThe company will present at a conference.",
])
def test_les_codes_administratifs_ne_sont_pas_materiels(texte):
    """9.01 (pièces jointes), 5.07 (vote en assemblée) et 7.01 (Regulation FD)
    sont de la routine. Les compter matériels périmerait les signaux en
    permanence : c'est le premier risque de ce genre de règle."""
    resultat = classer(texte)
    assert resultat["materiality"] is False
    assert resultat["category"] == "non_materiel"


# --------------------------------------------------------------------------- #
# Ce que seul le TEXTE peut trancher -- la raison d'être du repli
# --------------------------------------------------------------------------- #
def test_item_5_02_depart_du_directeur_general_est_materiel():
    resultat = classer(
        "Item 5.02 Departure of Directors or Certain Officers\n"
        "On May 3, 2021, the Company announced that its Chief Executive Officer "
        "notified the Board of his decision to resign, effective June 30, 2021."
    )
    assert resultat["materiality"] is True
    assert resultat["category"] == "depart_dirigeant"
    assert resultat["summary"], "un résumé tiré du document est attendu"


def test_item_5_02_election_routiniere_d_administrateur_ne_l_est_pas():
    """LE cas qui justifie de lire le document : l'Item 5.02 couvre aussi bien
    un départ de directeur général qu'une élection d'administrateur. Une table
    de codes les confond nécessairement ; le texte les sépare."""
    resultat = classer(
        "Item 5.02 Election of Directors\n"
        "On April 2, 2021, the Board of Directors appointed Jane Doe as a director, "
        "to serve until the next annual meeting of stockholders."
    )
    assert resultat["materiality"] is False


def test_item_8_01_annoncant_une_fusion_est_materiel():
    """8.01 est le fourre-tour « autres événements » : inerte par son code,
    décisif par son contenu."""
    resultat = classer(
        "Item 8.01 Other Events\n"
        "The Company entered into an Agreement and Plan of Merger pursuant to which "
        "it will acquire all outstanding shares of the target for $42.00 per share in cash."
    )
    assert resultat["materiality"] is True
    assert resultat["category"] == "fusion_acquisition"


def test_item_8_01_communication_ordinaire_ne_l_est_pas():
    resultat = classer(
        "Item 8.01 Other Events\n"
        "The Company issued a press release announcing the date of its annual meeting."
    )
    assert resultat["materiality"] is False


def test_revision_de_guidance_est_materielle():
    resultat = classer(
        "Item 2.02 Results of Operations and Financial Condition\n"
        "The Company lowered its full-year guidance for fiscal 2022 and now expects "
        "revenue to decline in the mid-single digits."
    )
    assert resultat["materiality"] is True
    assert resultat["category"] == "changement_guidance"


def test_resultats_trimestriels_ordinaires_ne_sont_pas_materiels():
    """Item 2.02 est le deuxième code le plus fréquent de l'archive (21 455
    dépôts). Le traiter matériel en bloc périmerait tous les signaux chaque
    trimestre -- ce serait supprimer la stratégie, pas la protéger."""
    resultat = classer(
        "Item 2.02 Results of Operations and Financial Condition\n"
        "On February 1, 2021, the Company issued a press release announcing its "
        "financial results for the quarter ended December 31, 2020."
    )
    assert resultat["materiality"] is False


def test_depreciation_et_restructuration_sont_materielles():
    resultat = classer(
        "Item 2.05 Costs Associated with Exit or Disposal Activities\n"
        "The Company approved a restructuring plan and expects to record a goodwill "
        "impairment charge of approximately $1.2 billion."
    )
    assert resultat["materiality"] is True
    assert resultat["category"] == "autre_materiel"


def test_une_mention_de_passage_dans_un_depot_administratif_ne_compte_pas():
    """Contrôle négatif du risque principal d'une règle textuelle : un mot-clé
    trouvé dans un dépôt qui ne déclare QUE des codes administratifs est très
    probablement une mention de passage, pas l'objet du dépôt."""
    resultat = classer(
        "Item 9.01 Financial Statements and Exhibits\n"
        "Exhibit 99.1 contains the historical financial statements required in "
        "connection with the merger agreement previously reported."
    )
    assert resultat["materiality"] is False


# --------------------------------------------------------------------------- #
# Intégration : le repli s'enclenche, et n'écrase pas le modèle
# --------------------------------------------------------------------------- #
def test_classify_8k_se_replie_sur_les_regles_sans_modele(monkeypatch):
    """Sans clé d'API, `analyser_texte_llm` rend None : le document doit
    alors être classé par règles au lieu de repartir en `non_evalue`."""
    monkeypatch.setattr(_c8k.sft, "analyser_texte_llm", lambda *a, **k: None)
    resultat = _c8k.classify_8k("AAPL", "2021-05-03", "Item 2.06 Material Impairments\nThe Company recorded an impairment charge.")

    assert resultat["category"] != "non_evalue"
    assert resultat["materiality"] is True
    assert resultat["classification_source"] == "regles_document"


@pytest.mark.parametrize("fournisseur", ["gemini", "mistral"])
def test_le_modele_reste_prioritaire_quand_il_repond(monkeypatch, fournisseur):
    """La règle est un repli, pas un remplacement : un verdict du modèle ne
    doit jamais être écrasé par elle. La source enregistrée est le fournisseur
    qui a répondu, pas un « mistral » figé."""
    cle = {"gemini": _c8k.sft.GEMINI_API_KEY_ENV, "mistral": _c8k.sft.MISTRAL_API_KEY_ENV}[fournisseur]
    monkeypatch.setenv(cle, "une-cle")
    monkeypatch.setattr(
        _c8k.sft, "analyser_texte_llm",
        lambda *a, **k: {"category": "rachat_actions", "materiality": True, "summary": "verdict du modèle"},
    )
    resultat = _c8k.classify_8k("AAPL", "2021-05-03", "Item 2.06 Material Impairments\nimpairment charge")

    assert resultat["category"] == "rachat_actions"
    assert resultat["summary"] == "verdict du modèle"
    assert resultat["classification_source"] == fournisseur


@pytest.mark.parametrize("cle_env", ["GEMINI_API_KEY", "MISTRAL_API_KEY"])
def test_un_verdict_par_regles_est_remis_en_jeu_quand_une_cle_arrive(tmp_path, monkeypatch, cle_env):
    """Un repli mémorisé ne doit pas devenir un plafond : dès qu'une clé d'API
    est disponible -- Gemini comme Mistral -- les lignes classées par règles
    repartent au modèle."""
    import json

    cache = _c8k.llm_cache_path(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"symbol": "AAPL", "accession_number": "0000-1", "category": "autre_materiel",
                    "classification_source": "regles_document"}) + "\n"
        + json.dumps({"symbol": "AAPL", "accession_number": "0000-2", "category": "rachat_actions",
                      "classification_source": "mistral"}) + "\n",
        encoding="utf-8",
    )

    assert len(_c8k.load_llm_cache(tmp_path)) == 2, "sans clé, le repli mémorisé doit être réutilisé"

    monkeypatch.setenv(cle_env, "une-cle")
    avec_cle = _c8k.load_llm_cache(tmp_path)
    assert len(avec_cle) == 1
    assert _c8k.cache_key("AAPL", "0000-2") in avec_cle
