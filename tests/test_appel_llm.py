"""D7 : robustesse du parsing et des reprises de l'appel LLM."""

from __future__ import annotations

import json

import pytest
import requests

import sec_filings_text as sft

ATTENDU = {"verdict": "coherent", "justification": "rien à signaler"}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("reponse", [
    '{"verdict": "coherent", "justification": "rien à signaler"}',
    '```json\n{"verdict": "coherent", "justification": "rien à signaler"}\n```',
    '```\n{"verdict": "coherent", "justification": "rien à signaler"}\n```',
    '  \n {"verdict": "coherent", "justification": "rien à signaler"}  \n ',
    'Voici mon analyse :\n{"verdict": "coherent", "justification": "rien à signaler"}\nFin.',
])
def test_les_enrobages_courants_sont_acceptes(reponse):
    """L'exigence `content.startswith("{")` rejetait tout ça, alors que le
    JSON attendu était bien présent."""
    assert sft._parse_json_reponse(reponse) == ATTENDU


@pytest.mark.parametrize("reponse", [
    "", "   ", "Je ne peux pas répondre.", "[1, 2, 3]", '{"incomplet": ', None, 42,
])
def test_les_reponses_reellement_inexploitables_sont_refusees(reponse):
    """La tolérance ne doit pas aller jusqu'à accepter autre chose qu'un
    objet JSON complet."""
    assert sft._parse_json_reponse(reponse) is None


# --------------------------------------------------------------------------- #
# Reprises
# --------------------------------------------------------------------------- #

class _ReponseHttp:
    def __init__(self, contenu, status_code=200, headers=None):
        self._contenu = contenu
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return {"choices": [{"message": {"content": self._contenu}}]}


def _erreur_http(status_code, headers=None):
    """Réponse d'erreur telle que la renvoie l'API (429 de quota, 401...)."""
    return _ReponseHttp("", status_code=status_code, headers=headers)


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv(sft.MISTRAL_API_KEY_ENV, "cle-de-test")
    monkeypatch.setattr(sft.time, "sleep", lambda _s: None)
    # Limiteur neuf par test : l'état (intervalle élargi par un 429) est
    # global au processus et fuiterait d'un test à l'autre.
    monkeypatch.setattr(sft, "MISTRAL_RATE_LIMITER", sft.AdaptiveRateLimiter(1000.0))

    appels = []

    def poser(reponses):
        def faux_post(url, headers=None, json=None, timeout=None):
            appels.append(json)
            item = reponses.pop(0)
            if isinstance(item, Exception):
                raise item
            # Une réponse déjà construite (code HTTP d'erreur) passe telle
            # quelle ; une chaîne est le contenu d'une réponse 200.
            return item if isinstance(item, _ReponseHttp) else _ReponseHttp(item)
        monkeypatch.setattr(sft.requests, "post", faux_post)
        return appels

    return poser


def test_le_mode_json_natif_est_demande(api):
    appels = api(['{"verdict": "coherent"}'])
    sft.analyser_texte_mistral("prompt")
    assert appels[0]["response_format"] == {"type": "json_object"}


def test_une_reprise_sur_reponse_non_parsable(api):
    """L'ancienne version abandonnait immédiatement ; on retente UNE fois."""
    appels = api(["pas du json", '{"verdict": "coherent"}'])
    assert sft.analyser_texte_mistral("prompt") == {"verdict": "coherent"}
    assert len(appels) == 2


def test_pas_plus_d_une_reprise_sur_reponse_non_parsable(api):
    """Insister sur un modèle hors format coûte des appels payants pour rien."""
    appels = api(["pas du json", "toujours pas", "encore moins"])
    assert sft.analyser_texte_mistral("prompt") is None
    assert len(appels) == 2


def test_les_erreurs_reseau_gardent_leurs_trois_tentatives(api):
    appels = api([
        requests.exceptions.ConnectionError("coupé"),
        requests.exceptions.ConnectionError("encore"),
        '{"verdict": "coherent"}',
    ])
    assert sft.analyser_texte_mistral("prompt") == {"verdict": "coherent"}
    assert len(appels) == 3


# --------------------------------------------------------------------------- #
# Quota (429)
# --------------------------------------------------------------------------- #

def test_un_429_est_reessaye_puis_finit_par_passer(api):
    """Le symptôme d'origine : trois tentatives seulement, et 04c abandonnait
    la classification (category="non_evalue") dès la première rafale de 429."""
    appels = api([
        _erreur_http(429), _erreur_http(429), _erreur_http(429), _erreur_http(429),
        '{"verdict": "coherent"}',
    ])
    assert sft.analyser_texte_mistral("prompt") == {"verdict": "coherent"}
    assert len(appels) == 5


def test_un_429_ralentit_le_debit_sortant(api, monkeypatch):
    """Réessayer ne suffit pas : sans élargir l'intervalle, l'appel suivant
    repart au même rythme et se fait refuser de nouveau."""
    limiteur = sft.AdaptiveRateLimiter(1.0)
    monkeypatch.setattr(sft, "MISTRAL_RATE_LIMITER", limiteur)
    avant = limiteur.interval

    api([_erreur_http(429), '{"verdict": "coherent"}'])
    sft.analyser_texte_mistral("prompt")

    assert limiteur.interval > avant


def test_le_retry_after_du_serveur_est_respecte(api, monkeypatch):
    """Quand Mistral annonce lui-même le délai, on l'applique au lieu du
    backoff maison."""
    attentes = []
    monkeypatch.setattr(sft.time, "sleep", attentes.append)

    api([_erreur_http(429, headers={"Retry-After": "30"}), '{"verdict": "coherent"}'])
    assert sft.analyser_texte_mistral("prompt") == {"verdict": "coherent"}
    assert any(30 <= a <= 31 for a in attentes), attentes


def test_une_erreur_definitive_n_est_pas_reessayee(api):
    """Une clé invalide (401) ne deviendra pas valide à la troisième tentative :
    insister ne fait que retarder le diagnostic."""
    appels = api([_erreur_http(401), '{"verdict": "coherent"}'])
    assert sft.analyser_texte_mistral("prompt") is None
    assert len(appels) == 1


def test_le_debit_se_resserre_apres_une_serie_de_succes():
    """L'auto-freinage doit être réversible : un 429 isolé ne doit pas
    condamner tout le reste du run à ramper."""
    limiteur = sft.AdaptiveRateLimiter(1.0, successes_before_speedup=2)
    nominal = limiteur.interval
    limiteur.penalize()
    limiteur.penalize()
    penalise = limiteur.interval
    assert penalise > nominal

    for _ in range(2):
        limiteur.reward()
    assert limiteur.interval == pytest.approx(penalise / 2)

    for _ in range(10):
        limiteur.reward()
    assert limiteur.interval == pytest.approx(nominal)


def test_le_debit_penalise_reste_borne():
    limiteur = sft.AdaptiveRateLimiter(1.0, max_interval=8.0)
    for _ in range(20):
        limiteur.penalize()
    assert limiteur.interval == 8.0


def test_sans_cle_api_aucun_appel(monkeypatch):
    monkeypatch.delenv(sft.MISTRAL_API_KEY_ENV, raising=False)

    def interdit(*a, **k):
        raise AssertionError("aucun appel réseau ne doit être tenté sans clé")

    monkeypatch.setattr(sft.requests, "post", interdit)
    assert sft.analyser_texte_mistral("prompt") is None


def test_enveloppe_inattendue_ne_plante_pas(monkeypatch):
    """Une réponse HTTP 200 dont le corps n'a pas la forme attendue est un
    échec définitif, pas une réponse à reparser indéfiniment."""
    monkeypatch.setenv(sft.MISTRAL_API_KEY_ENV, "cle-de-test")
    monkeypatch.setattr(sft.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sft, "MISTRAL_RATE_LIMITER", sft.AdaptiveRateLimiter(1000.0))

    class _SansChoices:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            pass

        def json(self):
            return {"pas_de_choices": True}

    monkeypatch.setattr(
        sft.requests, "post", lambda url, headers=None, json=None, timeout=None: _SansChoices(),
    )
    assert sft.analyser_texte_mistral("prompt") is None
