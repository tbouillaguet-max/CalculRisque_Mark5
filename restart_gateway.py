"""
Redémarrage automatique d'IB Gateway via IBC (https://github.com/IbcAlpha/IBC).

IBC est un outil tiers qui pilote la fenêtre de login d'IB Gateway pour
pouvoir le lancer sans intervention manuelle (identifiants injectés au
démarrage). Ce module ne réinvente pas cette mécanique : il se contente
d'appeler le script de lancement d'IBC, en lui passant les identifiants lus
depuis .env (jamais committé, voir .env.example).

PRÉREQUIS avant d'utiliser ce module :
1. Installer IBC : https://github.com/IbcAlpha/IBC/releases
2. Copier .env.example en .env et le remplir (IB_USERNAME, IB_PASSWORD,
   IB_TRADING_MODE, IBC_PATH, IB_GATEWAY_PORT).
3. `pip install python-dotenv psutil`
4. Vérifier la configuration d'IBC lui-même (config.ini dans son dossier
   d'installation) selon la documentation officielle du projet -- ce module
   suppose une installation IBC standard, mais ne peut pas garantir le
   comportement exact selon ta version/config.

LIMITE IMPORTANTE -- DOUBLE AUTHENTIFICATION (2FA) :
Si le compte IBKR utilisé a la 2FA activée (notification mobile), un
redémarrage automatique ne pourra PAS se reconnecter tout seul sans
confirmation sur le téléphone, quelle que soit la configuration d'IBC.
Deux options : configurer IBKR pour ce cas précis (2FA "readonly" ou
équivalent selon ta config de compte), ou accepter qu'un redémarrage
automatique nécessite une confirmation manuelle sur le téléphone quand la
2FA est active -- dans ce cas, ce module attend simplement que le port API
redevienne joignable (donc fonctionne aussi en 2FA, juste pas "sans aucune
intervention").
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

WAIT_FOR_PORT_TIMEOUT_SEC = 180  # temps max attendu pour qu'IB Gateway soit de nouveau joignable
WAIT_FOR_PORT_POLL_SEC = 5
POST_KILL_PAUSE_SEC = 5


def _load_env() -> dict:
    """Charge .env (à la racine du dépôt) sans dépendance obligatoire à
    python-dotenv : si le paquet est installé on l'utilise (gère les cas
    limites proprement), sinon on fait un parsing minimal fait main."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        raise FileNotFoundError(
            f"{env_path} introuvable. Copie .env.example en .env et remplis tes identifiants."
        )

    try:
        from dotenv import dotenv_values
        values = dict(dotenv_values(env_path))
    except ImportError:
        values = {}
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip()

    required = ["IB_USERNAME", "IB_PASSWORD", "IB_TRADING_MODE", "IBC_PATH", "IB_GATEWAY_PORT"]
    missing = [k for k in required if not values.get(k)]
    if missing:
        raise ValueError(f"Variables manquantes dans .env : {missing}")
    return values


def _is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


GATEWAY_PROCESS_NAMES = {"javaw.exe", "javaw", "ibgateway.exe", "ibgateway"}
GATEWAY_CMDLINE_TOKENS = {"ibgateway", "ibcstart", "ibcstart.bat", "ibcstart.sh", "ibcalpha.ibc"}


def kill_ib_gateway() -> int:
    """Ferme les process IB Gateway (javaw.exe lancé pour ibgateway) et IBC
    encore actifs, pour repartir d'un état propre avant le relancement.
    Correspondance STRICTE (nom de process exact + token de ligne de
    commande exact, jamais une simple sous-chaîne) pour ne jamais fermer un
    autre process par coïncidence -- une recherche par sous-chaîne sur "ibc"
    matcherait par exemple n'importe quel chemin contenant "libcairo" ou
    "libcurl"."""
    try:
        import psutil
    except ImportError:
        logger.warning("psutil non installé (pip install psutil) : fermeture manuelle d'IB Gateway nécessaire.")
        return 0

    killed = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = str(proc.info.get("name") or "").lower()
            if name not in GATEWAY_PROCESS_NAMES:
                continue
            cmdline_parts = [str(c).lower() for c in (proc.info["cmdline"] or [])]
            is_gateway = any(
                part in GATEWAY_CMDLINE_TOKENS or Path(part).name in GATEWAY_CMDLINE_TOKENS
                for part in cmdline_parts
            )
            if not is_gateway:
                continue
            logger.warning("Fermeture d'IB Gateway/IBC (PID %d) : %s", proc.info["pid"], " ".join(cmdline_parts))
            proc.terminate()
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if killed:
        time.sleep(POST_KILL_PAUSE_SEC)
    return killed


def start_ib_gateway(env: dict) -> None:
    """Lance IB Gateway via le script de démarrage d'IBC, identifiants
    injectés par variables d'environnement (jamais en argument de ligne de
    commande, pour ne pas les exposer dans la liste des process)."""
    ibc_path = Path(env["IBC_PATH"])
    start_script = ibc_path / "StartGateway.bat"
    if not start_script.exists():
        raise FileNotFoundError(
            f"{start_script} introuvable. Vérifie IBC_PATH dans .env et l'installation d'IBC "
            "(https://github.com/IbcAlpha/IBC)."
        )

    proc_env = os.environ.copy()
    proc_env["TWSUSERID"] = env["IB_USERNAME"]
    proc_env["TWSPASSWORD"] = env["IB_PASSWORD"]
    proc_env["TRADING_MODE"] = env["IB_TRADING_MODE"]

    logger.info("Lancement d'IB Gateway via IBC (%s)...", start_script)
    # DETACHED_PROCESS : ne bloque pas ce script en attendant qu'IB Gateway
    # se termine (il doit continuer à tourner en arrière-plan).
    subprocess.Popen(
        [str(start_script)],
        env=proc_env,
        cwd=str(ibc_path),
        creationflags=subprocess.DETACHED_PROCESS if hasattr(subprocess, "DETACHED_PROCESS") else 0,
    )


def wait_for_gateway(host: str, port: int, timeout: float = WAIT_FOR_PORT_TIMEOUT_SEC) -> bool:
    """Attend que le port API d'IB Gateway soit joignable. Si la 2FA est
    active, ça peut nécessiter une confirmation manuelle sur le téléphone
    avant que le port ne s'ouvre -- ce délai (jusqu'à `timeout`) en tient
    compte, mais reste borné pour ne pas bloquer indéfiniment."""
    elapsed = 0.0
    while elapsed < timeout:
        if _is_port_open(host, port):
            return True
        time.sleep(WAIT_FOR_PORT_POLL_SEC)
        elapsed += WAIT_FOR_PORT_POLL_SEC
    return False


def restart_ib_gateway() -> bool:
    """Point d'entrée principal : ferme IB Gateway/IBC s'ils tournent, les
    relance via IBC avec les identifiants de .env, et attend que le port API
    soit de nouveau joignable. Retourne True si le redémarrage a réussi
    (port joignable avant timeout), False sinon -- ne lève jamais d'exception
    pour un échec de reconnexion (seulement pour une config manquante/
    invalide), pour que l'appelant puisse décider comment réagir."""
    env = _load_env()
    port = int(env["IB_GATEWAY_PORT"])

    kill_ib_gateway()
    start_ib_gateway(env)

    logger.info("Attente qu'IB Gateway soit joignable sur le port %d (jusqu'à %ds)...", port, WAIT_FOR_PORT_TIMEOUT_SEC)
    ok = wait_for_gateway("127.0.0.1", port)
    if ok:
        logger.info("IB Gateway de nouveau joignable sur le port %d.", port)
    else:
        logger.error(
            "IB Gateway ne répond toujours pas sur le port %d après %ds. "
            "Si la 2FA est active, une confirmation manuelle sur le téléphone était peut-être "
            "nécessaire. Vérifie l'état d'IB Gateway manuellement.",
            port, WAIT_FOR_PORT_TIMEOUT_SEC,
        )
    return ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    success = restart_ib_gateway()
    raise SystemExit(0 if success else 1)
