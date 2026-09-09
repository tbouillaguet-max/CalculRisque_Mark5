"""
Accès HTTP à la SEC, partagé par 04_recuperation_10k.py,
04b_recuperation_10q.py, 04c_recuperation_8k.py et sec_filings_text.py.

Centralise deux choses qui étaient recopiées (ou absentes) d'un script à
l'autre :

    - L'IDENTITÉ de l'appelant. La politique d'accès équitable de la SEC
      (https://www.sec.gov/os/webmaster-faq#developers) exige un User-Agent
      identifiant un contact réel. Elle était renseignée en dur dans trois
      fichiers, avec une adresse personnelle committée dans le dépôt : elle
      vient désormais de la variable d'environnement SEC_CONTACT_EMAIL, sans
      valeur par défaut. Pas de repli silencieux : un User-Agent générique se
      fait bloquer (403/429) et le pipeline échouerait plus loin, de façon
      bien plus difficile à diagnostiquer qu'un message au démarrage.

    - Le LIMITEUR DE DÉBIT et la POLITIQUE DE RÉESSAI, écrits pour 04 et dont
      04b/04c/sec_filings_text n'avaient aucun équivalent -- ces derniers
      envoyaient leurs requêtes sans throttle global et traitaient un 404
      (CIK inexistant) exactement comme un 503 (SEC surchargée) : un None dans
      les deux cas, sans réessai. Une panne passagère se traduisait donc par
      des trous SILENCIEUX dans les parquets de sortie.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger("sec_http")

SEC_CONTACT_EMAIL_ENV = "SEC_CONTACT_EMAIL"
USER_AGENT_TEMPLATE = "OptionsPipeline/1.0 ({email})"

# Politique d'accès équitable de la SEC : ~10 requêtes/seconde avec un
# User-Agent identifiant un contact réel. On reste juste en dessous, tous
# threads confondus (voir RateLimiter).
MAX_REQUESTS_PER_SECOND = 8.0
DEFAULT_WORKERS = 8

RETRYABLE_STATUS = frozenset((429, 500, 502, 503, 504))
MAX_ATTEMPTS = 4
REQUEST_TIMEOUT = 30

MISSING_EMAIL_MESSAGE = (
    f"La variable d'environnement {SEC_CONTACT_EMAIL_ENV} n'est pas définie.\n"
    "La SEC exige un User-Agent identifiant un contact réel pour ses API "
    "(https://www.sec.gov/os/webmaster-faq#developers) : sans elle, les requêtes "
    "sont bloquées (403/429).\n"
    '    export SEC_CONTACT_EMAIL="ton.adresse@exemple.fr"'
)


class SecContactEmailMissing(RuntimeError):
    """SEC_CONTACT_EMAIL absente de l'environnement."""


class SecNotFound(Exception):
    """La ressource n'existe pas (404). Réponse DÉFINITIVE de la SEC : le CIK
    est inconnu, ou n'a aucun document à cette adresse. À distinguer d'un
    échec réseau -- il n'y a rien à réessayer, et rien à signaler comme
    incident."""


class SecUnavailable(Exception):
    """Échec réseau ou erreur serveur persistante après tous les réessais.
    Le document existe peut-être : ce qui manque, c'est la réponse. Un
    appelant qui écrit un fichier de sortie doit COMPTER ces cas et refuser de
    présenter le résultat comme complet."""


def contact_email() -> str:
    email = os.environ.get(SEC_CONTACT_EMAIL_ENV, "").strip()
    if not email:
        raise SecContactEmailMissing(MISSING_EMAIL_MESSAGE)
    return email


def require_contact_email(logger_: Optional[logging.Logger] = None) -> Optional[str]:
    """À appeler au DÉMARRAGE d'un script qui interroge la SEC : échoue tout
    de suite avec un message actionnable, plutôt qu'au milieu d'un run de
    plusieurs heures. Retourne l'adresse, ou None après avoir journalisé
    l'erreur (à l'appelant de sortir)."""
    try:
        return contact_email()
    except SecContactEmailMissing as exc:
        (logger_ or logger).error("%s", exc)
        return None


def headers() -> dict:
    return {"User-Agent": USER_AGENT_TEMPLATE.format(email=contact_email()), "Accept": "application/json"}


class RateLimiter:
    """Espace les requêtes d'AU MOINS 1/rate seconde, tous threads confondus.

    L'attente a lieu à l'intérieur du verrou : les threads se mettent
    naturellement en file et le débit global reste borné, quel que soit leur
    nombre -- c'est cette garantie qui permet de paralléliser sans dépasser
    la limite d'accès équitable de la SEC."""

    def __init__(self, rate_per_second: float):
        self._min_interval = 1.0 / rate_per_second
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait > 0:
                time.sleep(wait)
                now = self._next_slot
            self._next_slot = now + self._min_interval


# Limiteur GLOBAL au processus : tous les appelants le partagent, sinon deux
# modules interrogeant la SEC en parallèle doubleraient le débit réel.
RATE_LIMITER = RateLimiter(MAX_REQUESTS_PER_SECOND)

_session_local = threading.local()


def build_session(pool_size: int = DEFAULT_WORKERS) -> requests.Session:
    """Session HTTP persistante : la connexion TLS vers data.sec.gov est
    négociée une fois puis réutilisée.

    Les réessais sont gérés ici et NON par l'adaptateur (max_retries=0) : un
    réessai interne à urllib3 ne repasserait pas par le limiteur de débit, et
    rejouerait donc une requête hors quota -- précisément au moment où la SEC
    vient de nous signaler qu'on en envoie trop (429)."""
    session = requests.Session()
    session.headers.update(headers())
    adapter = HTTPAdapter(max_retries=0, pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("https://", adapter)
    return session


def _thread_session() -> requests.Session:
    session = getattr(_session_local, "session", None)
    if session is None:
        session = _session_local.session = build_session()
    return session


def _retry_delay(response: Optional[requests.Response], attempt: int) -> float:
    """Backoff exponentiel (2s, 4s, 8s), sauf si la SEC indique elle-même
    combien de temps attendre via Retry-After."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return float(retry_after)
    return 2.0 ** (attempt + 1)


def request(
    url: str, session: Optional[requests.Session] = None, stream: bool = False,
    timeout: int = REQUEST_TIMEOUT,
) -> requests.Response:
    """Réponse HTTP de la SEC, sous limiteur de débit et avec réessais.

    Lève SecNotFound sur 404 (définitif, aucun réessai) et SecUnavailable
    après épuisement des tentatives -- les deux cas que les appelants
    confondaient en renvoyant None."""
    session = session or _thread_session()
    last_error: Optional[Exception] = None

    for attempt in range(MAX_ATTEMPTS):
        RATE_LIMITER.acquire()
        response = None
        try:
            response = session.get(url, timeout=timeout, stream=stream)
            if response.status_code == 404:
                raise SecNotFound(url)
            if response.status_code in RETRYABLE_STATUS:
                raise requests.exceptions.HTTPError(f"HTTP {response.status_code}", response=response)
            response.raise_for_status()
            return response
        except SecNotFound:
            raise
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt == MAX_ATTEMPTS - 1:
                break
            delay = _retry_delay(response, attempt)
            logger.warning("Requête %s en échec (%s), nouvelle tentative dans %.0fs.", url, exc, delay)
            time.sleep(delay)

    logger.error("Échec de la requête %s (%d tentatives) : %s", url, MAX_ATTEMPTS, last_error)
    raise SecUnavailable(f"{url} : {last_error}")


def get_json(url: str, session: Optional[requests.Session] = None) -> dict:
    """JSON de la SEC. Lève SecNotFound / SecUnavailable (voir request), ou
    SecUnavailable si le corps n'est pas du JSON exploitable -- une réponse
    tronquée par la SEC n'est pas une absence de données."""
    response = request(url, session=session)
    try:
        return response.json()
    except ValueError as exc:
        raise SecUnavailable(f"{url} : réponse non-JSON ({exc})") from exc


def get_json_or_none(url: str, session: Optional[requests.Session] = None) -> Optional[dict]:
    """Variante tolérante, pour les appelants qui n'ont pas encore de
    politique d'erreur : None dans TOUS les cas d'échec. À éviter dans du code
    neuf -- c'est précisément cette confusion entre "pas de données" et "pas de
    réponse" que ce module sert à lever."""
    try:
        return get_json(url, session=session)
    except SecNotFound:
        return None
    except SecUnavailable:
        return None
