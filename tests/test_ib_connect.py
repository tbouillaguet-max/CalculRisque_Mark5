"""Connexion IBKR : repli sur une connexion API seule quand la
synchronisation de compte d'ib_insync dépasse son délai.

Le symptôme corrigé : la session API s'ouvre ("API connection ready" dans les
logs), puis `IB.connect()` lève TimeoutError 20 secondes plus tard. En cause,
la dernière requête d'initialisation d'ib_insync -- `reqExecutionsAsync` --
qui n'est ni protégée par `return_exceptions`, ni conditionnée par `readonly`,
contrairement à toutes les autres."""

from __future__ import annotations

import asyncio

import pytest

import ib_connect


class _Client:
    """Client bas niveau : c'est lui qui fait la poignée de main API, sans
    aucune synchronisation de compte."""

    def __init__(self, ib):
        self._ib = ib
        self.appels = []

    def connect(self, host, port, client_id, timeout):
        self.appels.append((host, port, client_id, timeout))
        self._ib.connecte = True


class _Wrapper:
    clientId = None


class _IBFactice:
    """Reproduit le comportement d'ib_insync : IB.connect() se déconnecte
    avant de relever son erreur."""

    def __init__(self, echoue_sur_sync: bool = False, echoue_toujours: bool = False):
        self.client = _Client(self)
        self.wrapper = _Wrapper()
        self.connecte = False
        self.RequestTimeout = None
        self._echoue_sur_sync = echoue_sur_sync
        self._echoue_toujours = echoue_toujours
        self.tentatives_haut_niveau = 0

    def connect(self, host, port, clientId=None, timeout=None, readonly=None):
        self.tentatives_haut_niveau += 1
        if self._echoue_sur_sync or self._echoue_toujours:
            self.connecte = False          # IB.connect fait self.disconnect() puis raise
            raise asyncio.TimeoutError()
        self.connecte = True

    def isConnected(self):
        return self.connecte and not self._echoue_toujours


def test_connexion_normale_inchangee():
    """Quand la synchronisation fonctionne, rien ne change : aucun repli."""
    ib = _IBFactice()
    resultat = ib_connect.connect(port=4002, client_id=3, ib=ib)

    assert resultat is ib
    assert ib.tentatives_haut_niveau == 1
    assert ib.client.appels == [], "le repli API seule n'aurait pas dû être utilisé"


def test_repli_sur_timeout_de_synchronisation(caplog):
    """Le cas observé : la session s'ouvre, puis reqExecutions ne répond pas."""
    ib = _IBFactice(echoue_sur_sync=True)
    with caplog.at_level("WARNING"):
        resultat = ib_connect.connect(port=4002, client_id=3, ib=ib)

    assert resultat is ib
    assert ib.isConnected()
    assert ib.client.appels == [("127.0.0.1", 4002, 3, ib_connect.DEFAULT_TIMEOUT_SEC)]
    assert any("connexion API SEULE" in m for m in caplog.messages)


def test_le_repli_pose_le_client_id_sur_le_wrapper():
    """Posé par IB.connectAsync, que le repli court-circuite : l'oublier
    casserait l'association des réponses aux requêtes."""
    ib = _IBFactice(echoue_sur_sync=True)
    ib_connect.connect(port=4002, client_id=7, ib=ib)
    assert ib.wrapper.clientId == 7


def test_l_instance_est_reconnectee_en_place():
    """08_recuperation_options.py attache errorEvent AVANT de connecter et
    garde la référence : rendre une instance neuve perdrait ce branchement."""
    ib = _IBFactice(echoue_sur_sync=True)
    marqueur = object()
    ib.errorEvent = marqueur

    resultat = ib_connect.connect(port=4002, client_id=1, ib=ib)
    assert resultat is ib
    assert resultat.errorEvent is marqueur


def test_echec_complet_donne_un_message_actionnable():
    ib = _IBFactice(echoue_toujours=True)
    with pytest.raises(ConnectionError) as exc:
        ib_connect.connect(port=4002, client_id=3, ib=ib)

    message = str(exc.value)
    assert "4002" in message
    assert "Gateway" in message and "API" in message


def test_request_timeout_est_applique():
    ib = _IBFactice()
    ib_connect.connect(port=4002, client_id=3, request_timeout=20, ib=ib)
    assert ib.RequestTimeout == 20


def test_lecture_seule_par_defaut():
    """Aucun script du pipeline ne passe d'ordre."""
    assert ib_connect.DEFAULT_READONLY is True


def test_timeout_par_defaut_plus_genereux_que_l_ancien():
    """L'ancien appel codait 20s en dur, ce qui était juste pour une passerelle
    qui vient de démarrer."""
    assert ib_connect.DEFAULT_TIMEOUT_SEC >= 20


@pytest.mark.parametrize("erreur", [asyncio.TimeoutError, TimeoutError])
def test_les_deux_types_de_timeout_declenchent_le_repli(erreur):
    """Selon la version de Python, asyncio.TimeoutError et TimeoutError sont
    le même objet ou non -- les deux doivent être rattrapés."""
    class _IB(_IBFactice):
        def connect(self, *a, **k):
            self.tentatives_haut_niveau += 1
            raise erreur()

    ib = _IB()
    ib_connect.connect(port=4002, client_id=3, ib=ib)
    assert ib.isConnected()
