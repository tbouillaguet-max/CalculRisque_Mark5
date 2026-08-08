"""
Connexion à IB Gateway / TWS, partagée par 03, 03b et 08.

POURQUOI CE MODULE : `IB.connect()` échoue là où le pipeline n'a besoin de rien
---------------------------------------------------------------------------
`ib_insync.IB.connect()` ne se contente pas d'ouvrir la session API : il
enchaîne une phase de SYNCHRONISATION DE COMPTE (positions, ordres ouverts,
ordres exécutés, mises à jour de compte, puis exécutions). Symptôme observé :

    [INFO] Connected
    [INFO] Logged on to server version 176
    [INFO] API connection ready
    ... 20 secondes plus tard ...
    TimeoutError

La connexion elle-même a réussi -- c'est la synchronisation qui bloque. En
regardant la source d'ib_insync 0.9.86, toutes ces requêtes d'initialisation
sont lancées avec `asyncio.gather(..., return_exceptions=True)` : un
dépassement de délai s'y contente d'un message d'erreur. TOUTES SAUF LA
DERNIÈRE :

    # the request for executions must come after all orders are in
    await asyncio.wait_for(self.reqExecutionsAsync(), timeout)

Celle-ci n'est ni protégée par `return_exceptions`, ni conditionnée par
`readonly` (contrairement aux ordres ouverts et exécutés juste au-dessus). Si
la passerelle ne renvoie jamais son `execDetailsEnd` -- ce qui arrive avec
l'API en lecture seule côté Gateway, sur certains comptes papier, ou selon la
version de la passerelle --, `connect()` lève TimeoutError, se déconnecte, et
le script s'arrête.

Or AUCUN script de ce pipeline ne passe d'ordre : 03 et 03b ne demandent que
des barres historiques, 08 des détails de contrat et des données de marché.
Positions, ordres et exécutions ne servent à rien ici.

STRATÉGIE : on tente d'abord la connexion normale (rien ne change quand elle
fonctionne), et sur dépassement de délai on se rabat sur la connexion API
SEULE (`Client.connect`), qui fait la poignée de main et s'arrête là. Le repli
est journalisé explicitement : une connexion dégradée ne doit pas passer pour
une connexion normale.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger("ib_connect")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_TIMEOUT_SEC = 30

# Le pipeline ne passe JAMAIS d'ordre. En lecture seule, ib_insync saute déjà
# les requêtes d'ordres ouverts et exécutés -- ça ne suffit pas à éviter le
# blocage sur les exécutions (voir la docstring du module), mais ça réduit
# d'autant la phase de synchronisation, et c'est la posture correcte.
DEFAULT_READONLY = True


def connect(
    port: int,
    client_id: int,
    host: str = DEFAULT_HOST,
    timeout: int = DEFAULT_TIMEOUT_SEC,
    readonly: bool = DEFAULT_READONLY,
    request_timeout: Optional[float] = None,
    ib=None,
):
    """Retourne un `IB` connecté, ou lève la dernière erreur rencontrée.

    `ib` : instance à (re)connecter EN PLACE. Indispensable aux appelants qui
    gardent la référence et y ont attaché des gestionnaires d'événements
    (08_recuperation_options.py et son `errorEvent`) -- rendre un objet neuf
    leur ferait perdre ces branchements sans bruit. Par défaut, une instance
    est créée.

    `request_timeout` renseigne `IB.RequestTimeout` (délai des requêtes
    ultérieures), distinct du délai de connexion."""
    from ib_insync import IB

    if ib is None:
        ib = IB()
    if request_timeout is not None:
        ib.RequestTimeout = request_timeout

    try:
        ib.connect(host, port, clientId=client_id, timeout=timeout, readonly=readonly)
        logger.info("Connecté à IBKR sur %s:%s (clientId=%s, readonly=%s).", host, port, client_id, readonly)
        return ib
    except (asyncio.TimeoutError, TimeoutError) as exc:
        logger.warning(
            "La synchronisation de compte d'ib_insync a dépassé %ds (%s). La session API "
            "elle-même s'était pourtant ouverte : c'est la requête d'exécutions finale qui "
            "reste sans réponse (voir la docstring de ib_connect.py). Nouvelle tentative en "
            "connexion API SEULE -- suffisante pour ce pipeline, qui ne passe aucun ordre.",
            timeout, type(exc).__name__,
        )

    # `IB.connect` s'est déjà déconnecté avant de relever l'erreur (son
    # `except BaseException: self.disconnect(); raise`), donc l'instance est
    # dans un état propre et réutilisable -- c'est d'ailleurs ce que fait déjà
    # la boucle de reconnexion de 08.
    ib.wrapper.clientId = client_id      # posé par IB.connectAsync, que l'on court-circuite
    ib.client.connect(host, port, client_id, timeout)

    if not ib.isConnected():
        raise ConnectionError(
            f"Connexion à IBKR impossible sur {host}:{port} (clientId={client_id}). "
            "Vérifie que IB Gateway/TWS tourne, que le port de l'API correspond "
            "(4002 = Gateway papier, 4001 = Gateway réel, 7497/7496 = TWS), et que "
            "127.0.0.1 est autorisé dans Configuration > API > Settings."
        )

    logger.info(
        "Connecté à IBKR sur %s:%s (clientId=%s) en mode API seule : positions, ordres et "
        "exécutions NE sont pas synchronisés. Sans effet sur ce script, qui ne lit que des "
        "données de marché.", host, port, client_id,
    )
    return ib
