"""E1 et D4 : identité SEC obligatoire, et distinction 404 / panne réseau."""

from __future__ import annotations

import pytest
import requests

import sec_http


# --------------------------------------------------------------------------- #
# E1 : SEC_CONTACT_EMAIL en variable d'environnement
# --------------------------------------------------------------------------- #

def test_adresse_absente_leve_une_erreur_explicite(monkeypatch):
    monkeypatch.delenv(sec_http.SEC_CONTACT_EMAIL_ENV, raising=False)
    with pytest.raises(sec_http.SecContactEmailMissing) as exc:
        sec_http.contact_email()
    assert sec_http.SEC_CONTACT_EMAIL_ENV in str(exc.value)
    assert "export" in str(exc.value)


def test_adresse_vide_traitee_comme_absente(monkeypatch):
    """Une variable définie mais vide ne doit pas produire un User-Agent
    "OptionsPipeline/1.0 ()" que la SEC rejetterait."""
    monkeypatch.setenv(sec_http.SEC_CONTACT_EMAIL_ENV, "   ")
    with pytest.raises(sec_http.SecContactEmailMissing):
        sec_http.contact_email()


def test_aucun_repli_silencieux(monkeypatch):
    """Le point du correctif : pas d'adresse par défaut committée dans le
    dépôt, et pas de dégradation muette vers un User-Agent générique."""
    monkeypatch.delenv(sec_http.SEC_CONTACT_EMAIL_ENV, raising=False)
    assert sec_http.require_contact_email() is None
    with pytest.raises(sec_http.SecContactEmailMissing):
        sec_http.headers()


def test_user_agent_identifie_le_contact(monkeypatch):
    monkeypatch.setenv(sec_http.SEC_CONTACT_EMAIL_ENV, "test@exemple.fr")
    assert sec_http.headers()["User-Agent"] == "OptionsPipeline/1.0 (test@exemple.fr)"


# --------------------------------------------------------------------------- #
# D4 : 404 vs panne réseau
# --------------------------------------------------------------------------- #

class _Reponse:
    def __init__(self, status_code=200, payload=None, body=b"", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.content = body
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("pas du JSON")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}", response=self)


class _Session:
    """Rejoue une suite de réponses/exceptions et compte les appels."""

    def __init__(self, reponses):
        self.reponses = list(reponses)
        self.appels = 0

    def get(self, url, timeout=None, stream=False):
        self.appels += 1
        item = self.reponses.pop(0) if self.reponses else self.reponses
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _sans_attente(monkeypatch):
    """Neutralise le backoff : les tests ne doivent pas dormir 14 secondes."""
    monkeypatch.setattr(sec_http.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sec_http.RATE_LIMITER, "acquire", lambda: None)


def test_404_leve_secnotfound_sans_reessai():
    """Un CIK inexistant est une réponse DÉFINITIVE : la réessayer trois fois
    ne fait que consommer le quota SEC."""
    session = _Session([_Reponse(status_code=404)])
    with pytest.raises(sec_http.SecNotFound):
        sec_http.request("https://data.sec.gov/x", session=session)
    assert session.appels == 1


def test_503_est_reessaye_puis_leve_secunavailable():
    session = _Session([_Reponse(status_code=503) for _ in range(sec_http.MAX_ATTEMPTS)])
    with pytest.raises(sec_http.SecUnavailable):
        sec_http.request("https://data.sec.gov/x", session=session)
    assert session.appels == sec_http.MAX_ATTEMPTS


def test_503_transitoire_finit_par_reussir():
    session = _Session([_Reponse(status_code=503), _Reponse(payload={"ok": True})])
    assert sec_http.get_json("https://data.sec.gov/x", session=session) == {"ok": True}
    assert session.appels == 2


def test_panne_reseau_est_reessayee():
    session = _Session([
        requests.exceptions.ConnectionError("réseau coupé"),
        _Reponse(payload={"ok": True}),
    ])
    assert sec_http.get_json("https://data.sec.gov/x", session=session) == {"ok": True}


def test_json_illisible_est_une_indisponibilite_pas_une_absence():
    """Une réponse tronquée par la SEC n'est pas "cette entreprise n'a pas de
    données" : la confondre avec un 404 créerait un trou silencieux."""
    session = _Session([_Reponse(payload=None)])
    with pytest.raises(sec_http.SecUnavailable):
        sec_http.get_json("https://data.sec.gov/x", session=session)


def test_variante_tolerante_renvoie_none_dans_les_deux_cas():
    assert sec_http.get_json_or_none("https://x", session=_Session([_Reponse(status_code=404)])) is None
    assert sec_http.get_json_or_none(
        "https://x", session=_Session([_Reponse(status_code=503)] * sec_http.MAX_ATTEMPTS),
    ) is None


def test_retry_after_est_respecte():
    reponse = _Reponse(status_code=429, headers={"Retry-After": "7"})
    assert sec_http._retry_delay(reponse, attempt=0) == 7.0
    assert sec_http._retry_delay(None, attempt=0) == 2.0
    assert sec_http._retry_delay(None, attempt=1) == 4.0
