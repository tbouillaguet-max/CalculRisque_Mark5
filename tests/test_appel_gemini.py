"""Appel de Gemini (sec_filings_text) : un document, une consigne et un format de réponse.

Ce que ces tests protègent : les trois partent SÉPARÉMENT (contents,
systemInstruction, generationConfig.responseJsonSchema), la requête a la forme
de celle du SDK officiel, et une réponse hors format n'est jamais rendue -- le
mode JSON seul, sans schéma, laissait passer une catégorie inventée
(« aut_materiel »).
"""

from __future__ import annotations

from json import dumps

import pytest
import requests

import sec_filings_text as sft


class _ReponseGemini:
    """Réponse de l'API Gemini : texte dans candidates[0].content.parts, ou
    corps d'erreur {"error": {...}} quand status_code >= 400."""

    def __init__(self, texte="", status_code=200, corps=None):
        self.status_code = status_code
        self.headers = {}
        self._corps = corps if corps is not None else {
            "candidates": [{"content": {"role": "model", "parts": [{"text": texte}]},
                            "finishReason": "STOP"}],
        }
        self.text = str(self._corps)

    def json(self):
        return self._corps


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setenv(sft.GEMINI_API_KEY_ENV, "cle-gemini-de-test")
    monkeypatch.setattr(sft.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sft, "GEMINI_RATE_LIMITER", sft.AdaptiveRateLimiter(1000.0))

    appels = []

    def poser(reponses):
        def faux_post(url, headers=None, json=None, timeout=None):
            appels.append({"url": url, "headers": headers, "json": json})
            item = reponses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item if isinstance(item, _ReponseGemini) else _ReponseGemini(item)
        monkeypatch.setattr(sft.requests, "post", faux_post)
        return appels

    return poser


SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["a", "b"], "description": "Catégorie."},
        "materiality": {"type": "boolean"},
    },
    "required": ["category", "materiality"],
}

# Corps relevé en interceptant, hors ligne, la requête du SDK officiel
# google-genai 2.25.0 : client.models.generate_content(model=
# "gemini-3.5-flash-lite", contents="LE DOCUMENT", config=GenerateContentConfig(
# system_instruction="LA CONSIGNE", response_mime_type="application/json",
# response_json_schema=SCHEMA, max_output_tokens=500 + 8192,
# thinking_config=ThinkingConfig(thinking_level="minimal"))).
# Un schéma qui accepte tout objet, pour les tests qui ne portent pas sur le
# format de la réponse.
OBJET = {"type": "object"}


def _analyser(max_tokens: int = 500):
    return sft.analyser_document("document", "consigne", OBJET, max_tokens=max_tokens)


CORPS_SDK = {
    "contents": [{"parts": [{"text": "LE DOCUMENT"}], "role": "user"}],
    "systemInstruction": {"parts": [{"text": "LA CONSIGNE"}], "role": "user"},
    "generationConfig": {
        "maxOutputTokens": 8692,
        "responseMimeType": "application/json",
        "responseJsonSchema": SCHEMA,
        "thinkingConfig": {"thinking_level": "MINIMAL"},
    },
}


# --------------------------------------------------------------------------- #
# Clé
# --------------------------------------------------------------------------- #

def test_sans_cle_pas_de_llm():
    assert not sft.llm_disponible()
    assert sft.description_llm() == "aucun LLM (GEMINI_API_KEY à définir)"


@pytest.mark.parametrize("valeur", ["", "   "])
def test_une_cle_vide_compte_comme_absente(monkeypatch, valeur):
    monkeypatch.setenv(sft.GEMINI_API_KEY_ENV, valeur)
    assert not sft.llm_disponible()


def test_la_cle_est_nettoyee_de_ses_espaces(gemini, monkeypatch):
    """Une clé recopiée avec un espace ou un retour à la ligne partait telle
    quelle dans l'en-tête, et Gemini répondait « API key not valid »."""
    monkeypatch.setenv(sft.GEMINI_API_KEY_ENV, "  cle-propre\n")
    appels = gemini(['{"ok": true}'])
    _analyser()
    assert appels[0]["headers"]["x-goog-api-key"] == "cle-propre"


# --------------------------------------------------------------------------- #
# Requête : un document, une consigne, un format
# --------------------------------------------------------------------------- #

def test_le_corps_est_celui_du_sdk_officiel(gemini):
    appels = gemini(['{"category": "a", "materiality": true}'])
    assert sft.analyser_document("LE DOCUMENT", "LA CONSIGNE", SCHEMA) == {"category": "a", "materiality": True}
    assert appels[0]["url"] == sft.GEMINI_URL.format(model=sft.GEMINI_DEFAULT_MODEL)
    assert appels[0]["json"] == CORPS_SDK


def test_document_consigne_et_format_partent_separement(gemini):
    appels = gemini(['{"category": "b", "materiality": false}'])
    sft.analyser_document("LE DOCUMENT", "LA CONSIGNE", SCHEMA)
    corps = appels[0]["json"]
    assert corps["contents"][0]["parts"] == [{"text": "LE DOCUMENT"}]
    assert corps["systemInstruction"]["parts"] == [{"text": "LA CONSIGNE"}]
    assert corps["generationConfig"]["responseJsonSchema"] == SCHEMA
    assert "LA CONSIGNE" not in str(corps["contents"]) and "LE DOCUMENT" not in str(corps["systemInstruction"])


def test_la_cle_part_en_en_tete_et_la_marge_s_ajoute_au_budget(gemini):
    appels = gemini(['{"category": "non_materiel"}'])
    assert _analyser(max_tokens=321) == {"category": "non_materiel"}

    appel = appels[0]
    # Clé dans l'en-tête, jamais dans l'URL (qui finit dans les journaux).
    assert appel["headers"]["x-goog-api-key"] == "cle-gemini-de-test"
    assert "cle-gemini-de-test" not in appel["url"]
    assert appel["json"]["generationConfig"]["maxOutputTokens"] == 321 + sft.GEMINI_THINKING_HEADROOM_TOKENS


def test_aucun_reglage_retire_par_google_n_est_envoye(gemini):
    """Google demande de retirer temperature, top_p et top_k des requêtes aux
    modèles 3.8 ; l'ancien code envoyait temperature=0.1."""
    appels = gemini(['{"ok": true}'])
    _analyser()
    generation = appels[0]["json"]["generationConfig"]
    assert not {"temperature", "topP", "topK"} & set(generation)


@pytest.mark.parametrize("valeur, envoye", [
    (None, "MINIMAL"), ("high", "HIGH"), ("LOW", "LOW"), (" medium ", "MEDIUM"),
])
def test_le_niveau_de_reflexion_se_regle(gemini, monkeypatch, valeur, envoye):
    if valeur is not None:
        monkeypatch.setenv(sft.GEMINI_THINKING_LEVEL_ENV, valeur)
    appels = gemini(['{"ok": true}'])
    _analyser()
    assert appels[0]["json"]["generationConfig"]["thinkingConfig"] == {"thinking_level": envoye}


def test_un_niveau_inconnu_revient_au_defaut_et_se_signale(gemini, monkeypatch, caplog):
    sft._niveau_valide.cache_clear()
    monkeypatch.setenv(sft.GEMINI_THINKING_LEVEL_ENV, "turbo-inconnu")
    appels = gemini(['{"ok": true}', '{"ok": true}'])
    with caplog.at_level("WARNING", logger="sec_filings_text"):
        _analyser()
        _analyser()
    assert appels[1]["json"]["generationConfig"]["thinkingConfig"] == {"thinking_level": "MINIMAL"}
    assert caplog.text.count("turbo-inconnu") == 1, "signalé une fois par run, pas à chaque document"


def test_gemini_2_5_flash_garde_sa_reflexion_coupee(gemini, monkeypatch):
    """Toujours choisissable par GEMINI_MODEL (clé ancienne) : réflexion à
    budget nul, tout le budget de jetons va à la réponse, et pas de niveau de
    réflexion, que la génération 2 refuserait."""
    monkeypatch.setenv(sft.GEMINI_MODEL_ENV, "gemini-2.5-flash")
    appels = gemini(['{"ok": true}'])
    _analyser(max_tokens=321)
    generation = appels[0]["json"]["generationConfig"]
    assert generation["maxOutputTokens"] == 321
    assert generation["thinkingConfig"] == {"thinkingBudget": 0}


def test_un_autre_modele_2_x_recoit_une_marge_sans_niveau(gemini, monkeypatch):
    monkeypatch.setenv(sft.GEMINI_MODEL_ENV, "gemini-2.5-pro")
    appels = gemini(['{"ok": true}'])
    _analyser(max_tokens=500)
    assert "/models/gemini-2.5-pro:generateContent" in appels[0]["url"]
    generation = appels[0]["json"]["generationConfig"]
    assert "thinkingConfig" not in generation
    assert generation["maxOutputTokens"] == 500 + sft.GEMINI_THINKING_HEADROOM_TOKENS


def test_le_prefixe_models_recopie_de_google_est_tolere(gemini, monkeypatch):
    monkeypatch.setenv(sft.GEMINI_MODEL_ENV, "models/gemini-3.8-flash")
    appels = gemini(['{"ok": true}'])
    _analyser()
    assert appels[0]["url"] == sft.GEMINI_URL.format(model="gemini-3.8-flash")


def test_modele_choisi_par_variable_d_environnement(gemini, monkeypatch):
    monkeypatch.setenv(sft.GEMINI_MODEL_ENV, "gemini-3.1-pro-preview")
    appels = gemini(['{"ok": true}'])
    _analyser()
    assert "/models/gemini-3.1-pro-preview:generateContent" in appels[0]["url"]
    assert appels[0]["json"]["generationConfig"]["thinkingConfig"] == {"thinking_level": "MINIMAL"}
    assert sft.description_llm() == "Gemini (gemini-3.1-pro-preview, réflexion minimal)"


# --------------------------------------------------------------------------- #
# Réponse : conforme au format, ou rien
# --------------------------------------------------------------------------- #

def test_une_reponse_hors_liste_est_redemandee(gemini):
    appels = gemini(['{"category": "aut_materiel", "materiality": true}', '{"category": "a", "materiality": true}'])
    assert sft.analyser_document("d", "c", SCHEMA) == {"category": "a", "materiality": True}
    assert len(appels) == 2


@pytest.mark.parametrize("reponse", [
    '{"category": "aut_materiel", "materiality": true}',      # catégorie inventée
    '{"category": "a", "materiality": "false"}',              # booléen écrit en texte
    '{"category": "a"}',                                      # champ obligatoire absent
    '["a", true]',                                            # pas un objet
])
def test_une_reponse_hors_format_n_est_jamais_rendue(gemini, caplog, reponse):
    appels = gemini([reponse, reponse, reponse])
    with caplog.at_level("WARNING", logger="sec_filings_text"):
        assert sft.analyser_document("d", "c", SCHEMA) is None
    assert len(appels) == sft.GEMINI_MAX_PARSE_RETRIES, "une reprise, pas davantage"
    assert "hors format" in caplog.text


def test_les_parties_de_reflexion_sont_ignorees(gemini):
    corps = {"candidates": [{"content": {"parts": [
        {"text": "je réfléchis...", "thought": True},
        {"text": '{"verdict": "coherent"}', "thoughtSignature": "c2lnbmF0dXJl"},
    ]}, "finishReason": "STOP"}]}
    gemini([_ReponseGemini(corps=corps)])
    assert _analyser() == {"verdict": "coherent"}


def test_prompt_bloque_ne_plante_pas(gemini):
    """Un prompt bloqué par les filtres renvoie un 200 SANS candidates."""
    gemini([_ReponseGemini(corps={"promptFeedback": {"blockReason": "SAFETY"}})])
    assert _analyser() is None


def test_reponse_tronquee_sans_texte_ne_plante_pas(gemini):
    gemini([_ReponseGemini(corps={"candidates": [{"finishReason": "MAX_TOKENS", "content": {}}]})])
    assert _analyser() is None


def test_une_reponse_coupee_par_la_limite_de_jetons_se_dit(gemini, caplog):
    """Un JSON coupé ne se relit pas : le journal doit dire pourquoi, et quoi
    régler, plutôt que « non parsable »."""
    corps = {"candidates": [{"finishReason": "MAX_TOKENS",
                             "content": {"parts": [{"text": '{"category": "a", "materi'}]}}]}
    gemini([_ReponseGemini(corps=corps)])
    with caplog.at_level("ERROR", logger="sec_filings_text"):
        assert sft.analyser_document("d", "c", SCHEMA) is None
    assert "MAX_TOKENS" in caplog.text and sft.GEMINI_THINKING_LEVEL_ENV in caplog.text


def test_le_message_d_erreur_de_l_api_est_journalise(gemini, caplog):
    """Le seul code HTTP (403) ne dit pas POURQUOI l'appel est refusé ; le
    message de l'API, si."""
    erreur = {"error": {"code": 403, "message": "Method doesn't allow unregistered callers",
                        "status": "PERMISSION_DENIED"}}
    appels = gemini([_ReponseGemini(status_code=403, corps=erreur), '{"ok": true}'])
    with caplog.at_level("ERROR", logger="sec_filings_text"):
        assert _analyser() is None
    assert len(appels) == 1
    assert "unregistered callers" in caplog.text
    assert "Gemini" in caplog.text


def test_le_retry_delay_du_corps_est_respecte(gemini, monkeypatch):
    """Gemini annonce son délai de reprise dans le corps d'un 429, pas dans
    un en-tête Retry-After."""
    attentes = []
    monkeypatch.setattr(sft.time, "sleep", attentes.append)
    quota = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "36s"},
    ]}}
    gemini([_ReponseGemini(status_code=429, corps=quota), '{"ok": true}'])
    assert _analyser() == {"ok": True}
    assert any(36 <= a <= 37 for a in attentes), attentes


def test_erreur_reseau_reessayee(gemini):
    appels = gemini([requests.exceptions.ConnectionError("coupé"), '{"ok": true}'])
    assert _analyser() == {"ok": True}
    assert len(appels) == 2


# --------------------------------------------------------------------------- #
# Vérification contre le schéma
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("valeur, schema", [
    ({"a": True}, {"type": "object", "properties": {"a": {"type": "boolean"}}, "required": ["a"]}),
    ({"n": 3}, {"type": "OBJECT", "properties": {"n": {"type": "INTEGER"}}}),    # forme OpenAPI
    ([{"x": "b"}], {"type": "array", "items": {"type": "object", "properties": {"x": {"enum": ["a", "b"]}}}}),
    ({"autre": 1}, {"type": "object", "properties": {"a": {"type": "string"}}}),  # champ facultatif absent
    (2.5, {"type": "number"}),
])
def test_une_valeur_conforme_passe(valeur, schema):
    assert sft._ecart_au_schema(valeur, schema) is None


@pytest.mark.parametrize("valeur, schema, attendu", [
    ({"a": 1}, {"type": "object", "properties": {"a": {"type": "boolean"}}}, "réponse.a : boolean attendu"),
    ({"n": True}, {"type": "object", "properties": {"n": {"type": "integer"}}}, "réponse.n : integer attendu"),
    ({}, {"type": "object", "required": ["a"]}, "champ obligatoire « a » absent"),
    ("c", {"type": "string", "enum": ["a", "b"]}, "hors de la liste autorisée"),
    ([{"x": "c"}], {"type": "array", "items": {"properties": {"x": {"enum": ["a"]}}}}, "réponse[0].x"),
])
def test_le_premier_ecart_est_nomme(valeur, schema, attendu):
    assert attendu in sft._ecart_au_schema(valeur, schema)


# --------------------------------------------------------------------------- #
# Liste de modèles : chacun son quota du jour
# --------------------------------------------------------------------------- #

def _quota_du_jour(modele):
    """Le 429 que Gemini rend quand le quota QUOTIDIEN d'un modèle est épuisé."""
    return _ReponseGemini(status_code=429, corps={"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED",
        "message": "You exceeded your current quota. Quota exceeded for metric: "
                   "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 500",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{
                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                "quotaDimensions": {"location": "global", "model": modele}, "quotaValue": "500"}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "21600s"},
        ]}})


def _modeles_appeles(appels):
    return [appel["url"].split("/models/")[1].split(":")[0] for appel in appels]


def test_la_liste_par_defaut_commence_par_les_flash_lite():
    """Au palier gratuit, un modèle Flash n'a qu'une vingtaine de requêtes par
    jour ; les Flash-Lite, des centaines."""
    assert sft._modeles_configures() == list(sft.GEMINI_DEFAULT_MODELS)
    assert sft.GEMINI_DEFAULT_MODELS[0].endswith("flash-lite")


def test_gemini_model_accepte_une_liste(monkeypatch):
    monkeypatch.setenv(sft.GEMINI_MODEL_ENV, " models/gemini-a , gemini-b,, gemini-a ")
    assert sft._modeles_configures() == ["gemini-a", "gemini-b"]
    assert sft.description_llm() == "aucun LLM (GEMINI_API_KEY à définir)"
    monkeypatch.setenv(sft.GEMINI_API_KEY_ENV, "k")
    assert sft.description_llm() == "Gemini (gemini-a > gemini-b, réflexion minimal)"


def test_quota_du_jour_epuise_le_suivant_reprend_la_meme_requete(gemini, caplog):
    """Attendre ne servirait à rien avant le lendemain : on passe au modèle
    suivant tout de suite, et il garde la main pour la suite du run."""
    premier, second = sft.GEMINI_DEFAULT_MODELS[:2]
    appels = gemini([_quota_du_jour(premier), '{"ok": true}', '{"ok": true}'])
    with caplog.at_level("WARNING", logger="sec_filings_text"):
        assert _analyser() == {"ok": True}
        assert _analyser() == {"ok": True}
    assert _modeles_appeles(appels) == [premier, second, second]
    assert sft.dernier_modele_utilise() == second
    assert f"{premier} écarté pour ce run (quota du jour épuisé)" in caplog.text
    assert not sft.llm_en_pause() and not sft.llm_coupe_pour_ce_run()


def test_quotas_du_jour_tous_epuises_coupe_pour_le_run(gemini, monkeypatch, caplog):
    monkeypatch.setenv(sft.GEMINI_MODEL_ENV, "gemini-a,gemini-b")
    appels = gemini([_quota_du_jour("gemini-a"), _quota_du_jour("gemini-b"), '{"ok": true}'])
    with caplog.at_level("ERROR", logger="sec_filings_text"):
        assert _analyser() is None
    assert sft.llm_coupe_pour_ce_run()
    assert _analyser() is None
    assert len(appels) == 2, "plus aucune requête une fois tous les quotas du jour épuisés"
    assert "quota du jour épuisé pour chaque modèle" in caplog.text


def test_un_429_par_minute_attend_sans_changer_de_modele(gemini, monkeypatch):
    """Le quota PAR MINUTE revient vite : réessayer le même modèle."""
    attentes = []
    monkeypatch.setattr(sft.time, "sleep", attentes.append)
    minute = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
            {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"}]}}
    appels = gemini([_ReponseGemini(status_code=429, corps=minute), '{"ok": true}'])
    assert _analyser() == {"ok": True}
    assert len(set(_modeles_appeles(appels))) == 1
    assert any(12 <= a <= 13 for a in attentes), attentes


def test_un_modele_inconnu_de_la_cle_passe_la_main(gemini, caplog):
    premier, second = sft.GEMINI_DEFAULT_MODELS[:2]
    appels = gemini([(_ReponseGemini(status_code=404, corps={"error": {
        "code": 404, "message": f"models/{premier} is not found for API version v1beta"}})), '{"ok": true}'])
    assert _analyser() == {"ok": True}
    assert _modeles_appeles(appels) == [premier, second]
    assert not sft.llm_coupe_pour_ce_run()


def test_une_cle_refusee_coupe_sans_essayer_les_autres_modeles(gemini):
    appels = gemini([_ReponseGemini(status_code=403, corps={"error": {
        "code": 403, "message": "Method doesn't allow unregistered callers"}}), '{"ok": true}'])
    assert _analyser() is None
    assert sft.llm_coupe_pour_ce_run()
    assert len(appels) == 1


# --------------------------------------------------------------------------- #
# Plusieurs documents par requête
# --------------------------------------------------------------------------- #

ELEMENT = {"type": "object", "properties": {"c": {"type": "string", "enum": ["a", "b"]}}, "required": ["c"]}


def test_plusieurs_documents_une_seule_requete(gemini):
    appels = gemini(['{"d1": {"c": "a"}, "d2": {"c": "b"}}'])
    reponses = sft.analyser_documents({"d1": "PREMIER", "d2": "SECOND"}, "LA CONSIGNE", ELEMENT,
                                      max_tokens_par_document=100)

    assert reponses == {"d1": {"c": "a"}, "d2": {"c": "b"}}
    assert len(appels) == 1
    corps = appels[0]["json"]
    texte = corps["contents"][0]["parts"][0]["text"]
    assert texte == '<document id="d1">\nPREMIER\n</document>\n\n<document id="d2">\nSECOND\n</document>'
    assert corps["systemInstruction"]["parts"][0]["text"] == "LA CONSIGNE"
    schema = corps["generationConfig"]["responseJsonSchema"]
    assert schema["required"] == ["d1", "d2"] and schema["properties"] == {"d1": ELEMENT, "d2": ELEMENT}
    assert corps["generationConfig"]["maxOutputTokens"] == 200 + sft.GEMINI_THINKING_HEADROOM_TOKENS


def test_un_document_oublie_fait_redemander_le_lot(gemini):
    """Chaque identifiant est un champ obligatoire : une réponse qui en oublie
    un est hors format."""
    appels = gemini(['{"d1": {"c": "a"}}', '{"d1": {"c": "a"}, "d2": {"c": "a"}}'])
    assert sft.analyser_documents({"d1": "x", "d2": "y"}, "c", ELEMENT) == {"d1": {"c": "a"}, "d2": {"c": "a"}}
    assert len(appels) == 2


def test_le_bilan_compte_des_documents_et_des_requetes(gemini):
    gemini(['{"d1": {"c": "a"}, "d2": {"c": "a"}, "d3": {"c": "b"}}'])
    sft.analyser_documents({"d1": "x", "d2": "y", "d3": "z"}, "c", ELEMENT)
    bilan = sft.bilan_llm()
    assert "3 verdict(s)" in bilan and "1 requête(s) envoyée(s)" in bilan


def test_un_lot_vide_ne_coute_rien(gemini):
    appels = gemini([])
    assert sft.analyser_documents({}, "c", ELEMENT) == {}
    assert appels == []


@pytest.fixture
def gemini_selon_le_lot(monkeypatch):
    """Gemini refuse (400) toute requête qui contient un document « POISON »,
    et répond pour chaque document des autres ; rend les lots envoyés."""
    monkeypatch.setenv(sft.GEMINI_API_KEY_ENV, "cle-gemini-de-test")
    monkeypatch.setattr(sft.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sft, "GEMINI_RATE_LIMITER", sft.AdaptiveRateLimiter(1000.0))
    lots = []

    def faux_post(url, headers=None, json=None, timeout=None):
        idents = json["generationConfig"]["responseJsonSchema"]["required"]
        lots.append(idents)
        if "POISON" in json["contents"][0]["parts"][0]["text"]:
            return _ReponseGemini(status_code=400, corps={"error": {
                "code": 400, "message": "Request contains an invalid argument."}})
        return _ReponseGemini(dumps({ident: {"c": "a"} for ident in idents}))
    monkeypatch.setattr(sft.requests, "post", faux_post)
    return lots


def test_un_document_qui_fait_echouer_son_lot_est_isole(gemini_selon_le_lot):
    """Sans ça, le lot entier échouerait de nouveau, à l'identique, à chaque run."""
    lots = gemini_selon_le_lot
    documents = {"d1": "x", "d2": "y", "d3": "POISON", "d4": "z"}

    reponses = sft.analyser_documents(documents, "c", ELEMENT)

    assert reponses == {"d1": {"c": "a"}, "d2": {"c": "a"}, "d4": {"c": "a"}}
    assert lots == [["d1", "d2", "d3", "d4"], ["d1", "d2"], ["d3", "d4"], ["d3"], ["d4"]]
    bilan = sft.bilan_llm()
    assert "3 verdict(s)" in bilan and "1 réponse(s) inexploitable(s)" in bilan
    assert "5 requête(s)" in bilan


def test_deux_moities_en_echec_arretent_la_recherche(gemini_selon_le_lot):
    """La cause n'est pas un document isolé : descendre document par document
    multiplierait les requêtes pour rien."""
    lots = gemini_selon_le_lot
    documents = {"d1": "POISON", "d2": "y", "d3": "z", "d4": "POISON"}

    assert sft.analyser_documents(documents, "c", ELEMENT) is None
    assert lots == [["d1", "d2", "d3", "d4"], ["d1", "d2"], ["d3", "d4"]]
    assert "4 réponse(s) inexploitable(s)" in sft.bilan_llm()


def test_un_document_seul_en_echec_n_est_pas_recoupe(gemini_selon_le_lot):
    assert sft.analyser_documents({"d1": "POISON"}, "c", ELEMENT) is None
    assert gemini_selon_le_lot == [["d1"]]
    assert "1 réponse(s) inexploitable(s)" in sft.bilan_llm()


def test_un_schema_trop_lourd_fait_recouper_le_lot_sans_ecarter_le_modele(monkeypatch):
    """« too many states » : le schéma de CE lot est trop lourd, pas le modèle.
    Lu comme un refus du modèle, il écartait chaque modèle de la liste l'un
    après l'autre, et coupait Gemini pour le run."""
    monkeypatch.setenv(sft.GEMINI_API_KEY_ENV, "cle-gemini-de-test")
    monkeypatch.setattr(sft.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sft, "GEMINI_RATE_LIMITER", sft.AdaptiveRateLimiter(1000.0))
    lots = []

    def faux_post(url, headers=None, json=None, timeout=None):
        idents = json["generationConfig"]["responseJsonSchema"]["required"]
        lots.append(idents)
        if len(idents) > 2:
            return _ReponseGemini(status_code=400, corps={"error": {"code": 400, "message": (
                "The specified schema produces a constraint that has too many states for serving. "
                "Typical causes of this error are schemas with lots of text.")}})
        return _ReponseGemini(dumps({ident: {"c": "a"} for ident in idents}))
    monkeypatch.setattr(sft.requests, "post", faux_post)

    reponses = sft.analyser_documents({f"d{i}": "x" for i in range(1, 5)}, "c", ELEMENT)

    assert set(reponses) == {"d1", "d2", "d3", "d4"}
    assert lots == [["d1", "d2", "d3", "d4"], ["d1", "d2"], ["d3", "d4"]]
    assert not sft.llm_coupe_pour_ce_run() and "écarté" not in sft.bilan_llm()


def test_un_lot_sans_reponse_n_est_pas_recoupe(gemini):
    """Une panne ne tient pas au contenu : recouper le lot ne ferait que
    multiplier les requêtes."""
    appels = gemini([requests.exceptions.ConnectionError("coupé")] * sft.GEMINI_MAX_RETRIES)
    assert sft.analyser_documents({"d1": "x", "d2": "y"}, "c", ELEMENT) is None
    assert len(appels) == sft.GEMINI_MAX_RETRIES
    assert all(len(appel["json"]["generationConfig"]["responseJsonSchema"]["required"]) == 2
               for appel in appels)
