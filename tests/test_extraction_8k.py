"""Extraction des faits chiffrés d'un 8-K par Gemini (04d_extraction_8k.py).

CE QUE CES TESTS PROTÈGENT. Un chiffre extrait par le modèle finit dans une
valorisation. La vérification est donc le cœur du script : une citation qui
n'est pas dans le texte, ou un nombre qui n'est pas dans la citation, doit
faire rejeter le fait -- jamais le laisser passer, jamais le corriger. Le
périmètre compte aussi : il décide de ce qu'on paie au modèle.
"""

from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest

import config
import recalcul_8k
import sec_filings_text as sft

x8k = importlib.import_module("04d_extraction_8k")

TEXTE = (
    "Item 2.02 Results of Operations. ACME Corp. reported third quarter revenue of $1.2 billion, "
    "compared with $1,050.3 million in the prior-year quarter. Operating income was $(45) million. "
    "The Company issued 12,000,000 shares at $25.00 per share. "
    "The agreement has a term of five years. Diluted EPS guidance for fiscal 2025 is $3.10 to $3.30."
)


def fait(type_, valeur, citation):
    return {"type": type_, "valeur": valeur, "citation": citation}


# --------------------------------------------------------------------------- #
# Vérification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fait_extrait", [
    fait("ca_trimestre", 1200.0, "third quarter revenue of $1.2 billion"),
    fait("ca_trimestre_n1", 1050.3, "compared with $1,050.3 million in the prior-year quarter"),
    fait("ebit_trimestre", -45.0, "Operating income was $(45) million"),
    fait("actions_emises", 12.0, "The Company issued 12,000,000 shares"),
    fait("contrat_duree_annees", 5.0, "The agreement has a term of five years"),
    fait("guidance_bpa_bas", 3.10, "Diluted EPS guidance for fiscal 2025 is $3.10 to $3.30"),
])
def test_un_fait_cite_et_chiffre_correctement_est_accepte(fait_extrait):
    verifies, rejetes = x8k.verifier_faits([fait_extrait], x8k.normaliser(TEXTE))
    assert rejetes == [] and len(verifies) == 1


def test_la_citation_tolere_casse_espaces_et_guillemets():
    texte = "Revenue was “record” at   $1.2 billion."
    verifies, _ = x8k.verifier_faits([fait("ca_trimestre", 1200.0, 'revenue was "record" at $1.2 billion')],
                                     x8k.normaliser(texte))
    assert len(verifies) == 1


@pytest.mark.parametrize("fait_extrait, raison", [
    (fait("ca_trimestre", 1300.0, "third quarter revenue of $1.3 billion"), "citation introuvable dans le texte"),
    (fait("ca_trimestre", 1300.0, "third quarter revenue of $1.2 billion"), "nombre absent de la citation"),
    (fait("ca_trimestre", 1213.0, "third quarter revenue of $1.2 billion"), "nombre absent de la citation"),
    (fait("ca_trimestre", 1200.0, "$1.2"), "citation trop courte"),
    (fait("chiffre_invente", 1200.0, "third quarter revenue of $1.2 billion"), "type inconnu"),
    (fait("ca_trimestre", "beaucoup", "third quarter revenue of $1.2 billion"), "valeur non numérique"),
    # Un BPA ne se met pas à l'échelle : 3 100 n'est pas 3,10.
    (fait("guidance_bpa_bas", 3100.0, "Diluted EPS guidance for fiscal 2025 is $3.10 to $3.30"),
     "nombre absent de la citation"),
])
def test_un_fait_non_verifiable_est_rejete_avec_sa_raison(fait_extrait, raison):
    verifies, rejetes = x8k.verifier_faits([fait_extrait], x8k.normaliser(TEXTE))
    assert verifies == []
    assert rejetes[0]["raison"] == raison


def test_l_entree_ne_garde_que_les_faits_verifies():
    reponse = {
        "faits": [fait("ca_trimestre", 1200.0, "third quarter revenue of $1.2 billion"),
                  fait("ca_trimestre_n1", 999.0, "invented sentence about revenue")],
        "fin_periode_resultats": "2025-09-30", "direction": "positive", "prime_risque": "aucune",
        "veto": False, "resume": "Résultats en hausse.",
    }
    evenement = {"symbol": "ACME", "cik": "0000000001", "accession_number": "a-1",
                 "filed_date": pd.Timestamp("2025-10-20"), "item_codes": ["Item 2.02"]}
    entree = x8k.entree_depuis_reponse(evenement, reponse, TEXTE)
    assert [f["type"] for f in json.loads(entree["faits"])] == ["ca_trimestre"]
    assert [f["type"] for f in json.loads(entree["faits_rejetes"])] == ["ca_trimestre_n1"]
    assert entree["chiffrable"] is True
    assert entree["fin_periode_resultats"] == "2025-09-30"
    assert entree["version_extraction"] == x8k.VERSION_EXTRACTION
    # Ce que recalcul_8k en relit.
    assert recalcul_8k.faits_par_type(recalcul_8k.charger_faits(entree["faits"])) == {"ca_trimestre": 1200.0}


def test_une_date_de_cloture_illisible_est_videe():
    reponse = {"faits": [], "fin_periode_resultats": "troisième trimestre", "direction": "neutre",
               "prime_risque": "aucune", "veto": False, "resume": ""}
    evenement = {"symbol": "ACME", "accession_number": "a-1", "filed_date": "2025-10-20"}
    assert x8k.entree_depuis_reponse(evenement, reponse, TEXTE)["fin_periode_resultats"] == ""


# --------------------------------------------------------------------------- #
# Consigne et format
# --------------------------------------------------------------------------- #
def test_la_consigne_definit_chaque_type_de_fait_du_schema():
    consigne = x8k.build_consigne("2025-10-20")
    for type_fait in recalcul_8k.TYPES_FAITS:
        assert type_fait in consigne
    assert "2025-10-20" in consigne


def test_une_reponse_conforme_passe_le_controle_de_format():
    reponse = {"faits": [fait("ca_trimestre", 1200.0, "x")], "fin_periode_resultats": "",
               "direction": "neutre", "prime_risque": "faible", "veto": False, "resume": "r"}
    assert sft._ecart_au_schema(reponse, x8k.SCHEMA_EXTRACTION) is None
    assert sft._ecart_au_schema({**reponse, "prime_risque": "enorme"}, x8k.SCHEMA_EXTRACTION) is not None


def test_les_primes_de_risque_du_schema_sont_celles_de_la_config():
    assert x8k.SCHEMA_EXTRACTION["properties"]["prime_risque"]["enum"] == list(config.PRIME_RISQUE_8K_BPS)


# --------------------------------------------------------------------------- #
# Périmètre
# --------------------------------------------------------------------------- #
def evenement(symbol="ACME", date="2025-10-20", items=("Item 2.02", "Item 9.01"), accession=None):
    return {"symbol": symbol, "cik": "0000000001", "filed_date": date,
            "accession_number": accession or f"{symbol}-{date}", "item_codes": list(items)}


def signal(symbol="ACME", date="2025-08-01", gap=35.0, period_type="TTM"):
    return {"symbol": symbol, "filed_date": pd.Timestamp(date), "gap_pct": gap, "period_type": period_type}


def eligibles(evenements, signaux, positions=frozenset(), depuis="2025-01-01"):
    return x8k.huit_k_eligibles(pd.DataFrame(evenements), pd.DataFrame(signaux), set(positions),
                                pd.Timestamp(depuis))


def test_un_8k_sur_un_signal_actif_est_extrait():
    assert len(eligibles([evenement()], [signal()])) == 1


@pytest.mark.parametrize("signaux, raison", [
    ([signal(gap=5.0)], "écart sous le seuil"),
    ([signal(date="2025-01-02")], "signal TTM périmé (plus de 120 jours)"),
    ([signal(gap=900.0)], "écart absurde"),
    ([signal(date="2025-10-20")], "signal publié le jour même"),
    ([], "aucun signal"),
])
def test_un_8k_hors_signal_actif_n_est_pas_extrait(signaux, raison):
    assert eligibles([evenement()], signaux).empty, raison


def test_un_8k_d_une_position_ouverte_est_extrait_meme_sans_signal_actif():
    assert len(eligibles([evenement()], [signal(gap=5.0)], positions={"ACME"})) == 1


def test_seuls_les_items_porteurs_de_chiffres_sont_extraits():
    administratif = evenement(items=("Item 5.07", "Item 9.01"))
    veto = evenement(items=("Item 4.02",), accession="v")
    assert eligibles([administratif, veto], [signal()]).empty


def test_les_8k_anterieurs_a_la_fenetre_sont_ignores():
    assert eligibles([evenement(date="2024-12-15")], [signal(date="2024-11-01")]).empty


def test_les_positions_ouvertes_viennent_du_dernier_run_paper(tmp_path):
    chemin = tmp_path / "dernier_run.json"
    chemin.write_text(json.dumps({"cibles": {"AAA": {"poids": 0.1}, "BBB": {"poids": 0.0}}}))
    assert x8k.positions_ouvertes(chemin) == {"AAA"}
    assert x8k.positions_ouvertes(tmp_path / "absent.json") == set()


# --------------------------------------------------------------------------- #
# Envoi au modèle et mémoire
# --------------------------------------------------------------------------- #
def test_les_lots_sont_par_date_et_chaque_extraction_est_memorisee(tmp_path, monkeypatch):
    appels = []

    def faux_modele(documents, consigne, schema, jetons):
        appels.append((consigne, sorted(documents)))
        return {ident: {"faits": [fait("ca_trimestre", 1200.0, "third quarter revenue of $1.2 billion")],
                        "fin_periode_resultats": "2025-09-30", "direction": "positive",
                        "prime_risque": "aucune", "veto": False, "resume": "r"} for ident in documents}

    monkeypatch.setattr(sft, "analyser_documents", faux_modele)
    a_extraire = [x8k.AExtraire(evenement(symbol=s, date=d), TEXTE)
                  for s, d in (("A", "2025-10-20"), ("B", "2025-10-20"), ("C", "2025-10-20"), ("D", "2025-10-21"))]
    entrees = x8k.extraire(a_extraire, tmp_path, par_requete=2)

    assert len(entrees) == 4
    # Le plus récent d'abord ; jamais deux dates dans une requête.
    assert ["2025-10-21" in c for c, _ in appels] == [True, False, False]
    assert [docs for _, docs in appels] == [["d1"], ["d1", "d2"], ["d1"]]
    memoire = x8k.charger_memoire(tmp_path)
    assert set(memoire) == {x8k.cle(s, f"{s}-{d}") for s, d in
                            (("A", "2025-10-20"), ("B", "2025-10-20"), ("C", "2025-10-20"), ("D", "2025-10-21"))}

    sortie = x8k.ecrire_sortie(memoire, tmp_path / "extractions_8k.parquet")
    assert sortie["chiffrable"].all()
    relu = pd.read_parquet(tmp_path / "extractions_8k.parquet")
    assert list(relu["symbol"]) == ["A", "B", "C", "D"]


def test_une_extraction_d_une_autre_version_n_est_plus_servie(tmp_path):
    x8k.memoriser(tmp_path, {"symbol": "A", "accession_number": "1", "version_extraction": 0})
    x8k.memoriser(tmp_path, {"symbol": "B", "accession_number": "2",
                             "version_extraction": x8k.VERSION_EXTRACTION})
    assert set(x8k.charger_memoire(tmp_path)) == {"B:2"}


def test_sans_reponse_rien_n_est_memorise(tmp_path, monkeypatch):
    monkeypatch.setattr(sft, "analyser_documents", lambda *a, **k: None)
    monkeypatch.setattr(sft, "llm_coupe_pour_ce_run", lambda: False)
    assert x8k.extraire([x8k.AExtraire(evenement(), TEXTE)], tmp_path) == []
    assert x8k.charger_memoire(tmp_path) == {}


def test_le_texte_envoye_va_du_premier_item_aux_signatures_puis_au_communique():
    texte = x8k.texte_pour_extraction("COVER PAGE Item 2.02 Results.  Body text. SIGNATURES John Doe",
                                      "Press   release  text")
    assert texte.startswith("Item 2.02 Results. Body text.")
    assert "John Doe" not in texte and "COVER" not in texte
    assert texte.endswith("[Communiqué joint (Exhibit 99)]\nPress release text")
