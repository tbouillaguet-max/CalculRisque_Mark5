"""
Récupération des chaînes d'options (ITM / ATM / OTM) pour l'univers américain
(config.UNIVERSE_FILE) via l'API IBKR (ib_insync).

Filtres appliqués :
    - expiration à plus de 9 mois (aucun plafond en haut)
    - strikes compris entre -30% et +30% du dernier prix de clôture du sous-jacent

Conçu pour un compte SANS abonnement de données de marché payant : le script
utilise des données DIFFÉRÉES-FIGÉES (delayed-frozen, market data type 4) et
est prévu pour être lancé APRÈS la clôture des marchés US.

Corrections / changements par rapport à RecuperationOptionMark9 :
    - Exchange map réduite aux places US (config.EXCHANGE_MAP). L'univers US
      étant homogène (contrairement au STOXX 600 multi-pays), toute valeur
      non mappée est routée en SMART par défaut plutôt qu'ignorée : la
      logique de blacklist d'exchange (v9, conçue pour les abonnements
      manquants par pays en Europe) est conservée pour les vrais refus de
      permission IBKR, mais ne sert plus à filtrer des pays entiers.
    - Bug corrigé : le fichier de sortie était écrit en dur dans
      /option/calcul/... (ignorant --output-dir). Écrit maintenant dans
      config.OPTIONS_FILE.
    - Lit config.UNIVERSE_FILE par défaut (au lieu d'un chemin CSV STOXX 600 en dur).

Prérequis :
    pip install ib_insync pandas pyarrow

Usage :
    python 04_recuperation_options.py
    python 04_recuperation_options.py --resume       # reprendre un run interrompu
    python 04_recuperation_options.py --limit 10      # test rapide
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from ib_insync import IB, Stock, Option, Contract, util

import config

IB_HOST = "127.0.0.1"
IB_PORT = 4002
IB_CLIENT_ID = 1

MIN_DAYS_TO_EXPIRY = 9 * 30
STRIKE_BAND_PCT = 0.30

BATCH_SIZE = 25
BATCH_PAUSE_SEC = 0.5
SNAPSHOT_MAX_WAIT_SEC = 3.0   # délai MAXIMUM (voir _wait_for_batch : sort dès que les données arrivent)
SNAPSHOT_POLL_SEC = 0.25
HIST_REQUEST_PAUSE_SEC = 0.4

RETRY_PER_TICKER = 1
RETRY_PAUSE_SEC = 1.5

CHECKPOINT_EVERY = 10

MARKET_DATA_TYPE = 4  # 1=live, 2=frozen, 3=delayed, 4=delayed-frozen

# Timeout par appel bloquant IBKR (reqContractDetails, reqHistoricalData,
# reqSecDefOptParams...). Était à 360s ("Ajout après une erreur temps
# infini") : un compte sans abonnement data adapté fait bloquer certains
# appels, et avec RETRY_PER_TICKER=1 un seul ticker à problème pouvait
# coûter plusieurs fois 6 minutes -> plusieurs heures pour 10 tickers.
# 20s suffit largement pour un appel qui va aboutir ; au-delà, il vaut mieux
# échouer vite (et retenter/passer au ticker suivant) que rester bloqué.
IB_REQUEST_TIMEOUT_SEC = 20

REQUIRED_COLUMNS = [
    "symbol", "company_name", "expiry", "strike", "option_type",
    "last_price", "bid", "ask", "volume", "open_interest", "iv_source",
]

logger = logging.getLogger("us_options")

NOISY_ERROR_CODES = {300, 2103, 2104, 2105, 2106, 2107, 2108, 2119, 2157, 2158}
NOISY_MESSAGE_MARKERS = ("delayed market data is available",)


class NoisyIBFilter(logging.Filter):
    """Supprime du log les messages ib_insync répétitifs sans intérêt pratique."""

    _pattern = re.compile(r"\b(?:Error|Warning) (\d+)")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        m = self._pattern.search(msg)
        if m and int(m.group(1)) in NOISY_ERROR_CODES:
            return False
        low = msg.lower()
        return not any(marker in low for marker in NOISY_MESSAGE_MARKERS)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"us_options_{datetime.now():%Y%m%d_%H%M%S}.log"
    handlers = [logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    noisy_filter = NoisyIBFilter()
    for h in handlers:
        h.addFilter(noisy_filter)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)
    logger.info("Log file: %s", log_path)


class PermanentTickerError(RuntimeError):
    """Erreur structurelle : un nouvel essai ne changera jamais le résultat."""


PERMISSION_ERROR_CODES = {162, 354, 10089, 10197}
_EXCHANGE_FROM_162 = re.compile(r"for ([A-Z0-9.]+) STK")
_EXCHANGE_FROM_STREAM = re.compile(r"[A-Z0-9]+\s+([A-Z0-9.]+)/TOP/ALL")

BLACKLISTED_EXCHANGES: set[str] = set()

# Erreurs qui indiquent un état de SESSION corrompu (pas propre au ticker en
# cours) : typiquement une souscription IBKR fantôme (ex: reqAccountSummary)
# laissée ouverte par un ancien process de ce script tué/planté sans
# déconnexion propre. Contrairement aux PERMISSION_ERROR_CODES, retenter le
# même ticker ne sert à rien tant que la session elle-même n'est pas
# nettoyée -- voir recover_from_session_poisoning().
SESSION_ERROR_CODES = {322}
MAX_SESSION_RECOVERIES_PER_TICKER = 3
_session_poisoned = threading.Event()


def _extract_exchange(error_string: str, contract) -> Optional[str]:
    m = _EXCHANGE_FROM_162.search(error_string)
    if m:
        return m.group(1)
    m = _EXCHANGE_FROM_STREAM.search(error_string)
    if m:
        return m.group(1)
    if contract is not None and getattr(contract, "exchange", None):
        return contract.exchange
    return None


def _on_ib_error(reqId, errorCode, errorString, contract=None) -> None:
    if errorCode in SESSION_ERROR_CODES:
        logger.error(
            "Erreur de session IBKR détectée (code %d: %s). Probable souscription fantôme "
            "issue d'un ancien process non déconnecté proprement -- récupération automatique...",
            errorCode, errorString,
        )
        _session_poisoned.set()
        return

    if errorCode not in PERMISSION_ERROR_CODES:
        return
    low = errorString.lower()
    if any(marker in low for marker in NOISY_MESSAGE_MARKERS):
        return
    exch = _extract_exchange(errorString, contract)
    if exch and exch not in BLACKLISTED_EXCHANGES:
        BLACKLISTED_EXCHANGES.add(exch)
        logger.warning(
            "Exchange '%s' BLACKLISTÉE (pas d'abonnement data, erreur %d: %s) "
            "-> tickers suivants sur cet exchange ignorés sans requête réseau.",
            exch, errorCode, errorString,
        )


def kill_stray_pipeline_processes() -> int:
    """Termine toute AUTRE instance de CE script encore en cours d'exécution
    (processus zombie issu d'un run précédent qui a planté/été tué sans se
    déconnecter proprement d'IBKR -- cause typique de l'erreur 322). Ne
    touche jamais au processus courant, ni à TWS/IB Gateway lui-même : s'il
    est dans un état incohérent, il faut le redémarrer manuellement.
    Nécessite `pip install psutil` ; sans lui, ce nettoyage est simplement
    sauté (avec un avertissement) plutôt que de faire planter le script."""
    try:
        import psutil
    except ImportError:
        logger.warning(
            "psutil n'est pas installé (pip install psutil) : impossible de fermer "
            "automatiquement les processus fantômes. Vérifie-le manuellement."
        )
        return 0

    this_script = Path(__file__).name
    current_pid = os.getpid()
    killed = 0
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if proc.info["pid"] == current_pid:
                continue
            cmdline = proc.info["cmdline"] or []
            if not cmdline:
                continue
            # Exige (1) un interpréteur python en premier argument ET (2) un
            # argument qui correspond EXACTEMENT (ou en suffixe de chemin) au
            # nom de ce script -- une simple sous-chaîne dans la ligne de
            # commande ne suffit pas (ça tuerait par erreur n'importe quel
            # processus qui mentionne juste ce nom de fichier, ex: un éditeur
            # de texte ou une commande git).
            if "python" not in str(cmdline[0]).lower():
                continue
            if not any(str(part).endswith(this_script) for part in cmdline[1:]):
                continue
            joined = " ".join(str(part) for part in cmdline)
            logger.warning("Processus fantôme détecté (PID %d) : %s -- fermeture.", proc.info["pid"], joined)
            proc.terminate()
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if killed:
        time.sleep(3)  # laisse le temps à IBKR de détecter la déconnexion et libérer les souscriptions
    return killed


def recover_from_session_poisoning(ib: IB, auto_restart_gateway: bool = False, escalate: bool = False) -> None:
    """Erreur 322 (et assimilées) : ferme toute autre instance de ce script
    encore active, se déconnecte proprement, attend, puis se reconnecte. Le
    ticker en cours n'est jamais marqué comme traité tant que cette
    récupération n'a pas réussi -- il est retenté juste après (voir la
    boucle principale dans main()).

    Si `escalate` est vrai (dernière tentative pour ce ticker) et que
    `auto_restart_gateway` est activé, tente en plus un redémarrage complet
    d'IB Gateway via IBC (restart_gateway.py) AVANT de se reconnecter --
    utile quand le problème dépasse notre propre process (ex: TWS/Gateway
    lui-même dans un état bloqué)."""
    killed = kill_stray_pipeline_processes()
    if killed:
        logger.warning("%d processus fantôme(s) fermé(s).", killed)

    if escalate and auto_restart_gateway:
        logger.warning("Récupération légère insuffisante -> tentative de redémarrage complet d'IB Gateway...")
        try:
            from restart_gateway import restart_ib_gateway
            ok = restart_ib_gateway()
            if not ok:
                logger.error("Le redémarrage d'IB Gateway n'a pas abouti (voir les logs de restart_gateway).")
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Redémarrage automatique d'IB Gateway impossible (%s). Vérifie .env et "
                "l'installation d'IBC (voir restart_gateway.py).", exc,
            )

    try:
        ib.disconnect()
    except Exception:  # noqa: BLE001
        pass

    time.sleep(5)
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=20)
    ib.reqMarketDataType(MARKET_DATA_TYPE)
    logger.info("Reconnecté après récupération de session.")
    _session_poisoned.clear()


def load_universe(csv_path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"RIC", "Instrument_Name", "Country", "Currency", "Exchange"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

    df = df.dropna(subset=["RIC"]).drop_duplicates(subset=["RIC"]).reset_index(drop=True)
    df["ib_symbol"] = df["RIC"].apply(config.to_ib_symbol)
    df["ib_exchange"] = df["Exchange"].map(config.EXCHANGE_MAP).fillna(config.DEFAULT_EXCHANGE)

    if limit:
        df = df.head(limit)

    logger.info("Univers chargé : %d tickers.", len(df))
    return df


def connect_ib() -> IB:
    ib = IB()
    ib.RequestTimeout = IB_REQUEST_TIMEOUT_SEC
    ib.errorEvent += _on_ib_error
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=20)
    ib.reqMarketDataType(MARKET_DATA_TYPE)
    logger.info("Connecté à IBKR sur %s:%s (clientId=%s), marketDataType=%s", IB_HOST, IB_PORT, IB_CLIENT_ID, MARKET_DATA_TYPE)
    return ib


def ensure_connected(ib: IB, max_attempts: int = 6) -> None:
    if ib.isConnected():
        return
    # Force un état local propre avant de retenter : si la connexion a été
    # coupée de façon inattendue, il peut rester des traces locales
    # (souscriptions, event handlers) qui perturbent la reconnexion.
    try:
        ib.disconnect()
    except Exception:  # noqa: BLE001
        pass
    attempt = 0
    while not ib.isConnected():
        attempt += 1
        if attempt > max_attempts:
            raise ConnectionError(f"Impossible de se reconnecter à IBKR après {max_attempts} tentatives.")
        wait = min(60, 5 * attempt)
        logger.warning("Connexion IBKR perdue. Reconnexion tentative %d/%d dans %ds...", attempt, max_attempts, wait)
        time.sleep(wait)
        try:
            ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=20)
            ib.reqMarketDataType(MARKET_DATA_TYPE)
            logger.info("Reconnecté à IBKR.")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tentative de reconnexion échouée: %s", exc)


def resolve_stock_contract(ib: IB, symbol: str, exchange: str, currency: str) -> Optional[Contract]:
    if exchange == "SMART":
        candidates = [Stock(symbol, "SMART", currency)]
    else:
        candidates = [Stock(symbol, exchange, currency), Stock(symbol, "SMART", currency, primaryExchange=exchange)]
    for contract in candidates:
        try:
            details = ib.reqContractDetails(contract)
        except Exception as exc:  # noqa: BLE001
            logger.debug("reqContractDetails a échoué pour %s: %s", contract, exc)
            details = []
        if details:
            return details[0].contract
    return None


def _has_price(value) -> bool:
    if value is None:
        return False
    try:
        return not math.isnan(value)
    except TypeError:
        return True


def _wait_for_data(ib: IB, tickers, max_wait: float, poll_interval: float = SNAPSHOT_POLL_SEC) -> None:
    """Attend que TOUS les tickers passés aient au moins une donnée de prix
    (bid, ask ou last) ou que max_wait soit écoulé — en sortant dès que
    possible plutôt que d'attendre systématiquement le délai plein. Gain de
    temps important quand IBKR répond vite (cas majoritaire), et ne coûte
    jamais plus que max_wait quand ça traîne."""
    elapsed = 0.0
    while elapsed < max_wait:
        ib.sleep(poll_interval)
        elapsed += poll_interval
        if all(_has_price(t.bid) or _has_price(t.ask) or _has_price(t.last) for t in tickers):
            return


def get_spot_price(ib: IB, contract: Contract) -> Optional[float]:
    try:
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr="5 D", barSizeSetting="1 day",
            whatToShow="TRADES", useRTH=True, formatDate=1,
        )
        time.sleep(HIST_REQUEST_PAUSE_SEC)
        if bars:
            return float(bars[-1].close)
    except Exception as exc:  # noqa: BLE001
        logger.debug("reqHistoricalData a échoué pour %s: %s", contract.symbol, exc)

    if contract.exchange in BLACKLISTED_EXCHANGES:
        return None

    try:
        ticker = ib.reqMktData(contract, "", True, False)
        _wait_for_data(ib, [ticker], SNAPSHOT_MAX_WAIT_SEC)
        price = ticker.close or ticker.last or ticker.marketPrice()
        ib.cancelMktData(contract)
        if price and price == price:  # filtre NaN
            return float(price)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Fallback snapshot spot a échoué pour %s: %s", contract.symbol, exc)

    return None


def get_option_chain(ib: IB, contract: Contract):
    chains = ib.reqSecDefOptParams(contract.symbol, "", contract.secType, contract.conId)
    if not chains:
        return None
    same_class_non_smart = [c for c in chains if c.tradingClass == contract.symbol and c.exchange != "SMART"]
    non_smart = [c for c in chains if c.exchange != "SMART"]
    if same_class_non_smart:
        return same_class_non_smart[0]
    if non_smart:
        return non_smart[0]
    return chains[0]


def _bs_price(spot: float, strike: float, t: float, vol: float, option_type: str, r: float = config.RISK_FREE_RATE) -> float:
    """Prix Black-Scholes (pas de dividende, taux constant)."""
    if t <= 0 or vol <= 0:
        intrinsic = (spot - strike) if option_type == "CALL" else (strike - spot)
        return max(0.0, intrinsic)
    d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    if option_type == "CALL":
        return spot * norm.cdf(d1) - strike * np.exp(-r * t) * norm.cdf(d2)
    return strike * np.exp(-r * t) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def _bs_greeks(spot: float, strike: float, t: float, vol: float, option_type: str, r: float = config.RISK_FREE_RATE) -> dict:
    d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    pdf_d1 = norm.pdf(d1)
    gamma = pdf_d1 / (spot * vol * np.sqrt(t))
    vega = spot * pdf_d1 * np.sqrt(t) / 100  # pour +1 point de vol (convention IBKR)
    if option_type == "CALL":
        delta = norm.cdf(d1)
        theta = (-spot * pdf_d1 * vol / (2 * np.sqrt(t)) - r * strike * np.exp(-r * t) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (-spot * pdf_d1 * vol / (2 * np.sqrt(t)) + r * strike * np.exp(-r * t) * norm.cdf(-d2)) / 365
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def estimate_iv_and_greeks(mid_price, spot, strike, days_to_expiry, option_type: str) -> Optional[dict]:
    """
    Volatilité implicite et greeks calculés localement (Black-Scholes +
    solveur numérique `scipy.optimize.brentq`), utilisés en repli quand IBKR
    ne renvoie pas de modelGreeks.

    C'est le cas systématique avec MARKET_DATA_TYPE=4 (délayé-figé, cf. plus
    haut) : IBKR ne calcule quasiment jamais les greeks côté serveur sur ce
    type de flux (ça nécessite un abonnement temps réel ou figé), donc sans
    ce repli, implied_vol/delta/gamma/vega/theta restent NULL pour TOUS les
    contrats, et la nappe de volatilité du rapport n'a rien à afficher.

    Approximation : taux sans risque constant (config.RISK_FREE_RATE), pas
    de dividende. Retourne None si les données d'entrée sont insuffisantes
    ou incohérentes (prix < valeur intrinsèque, résolution impossible du
    solveur, etc.) plutôt que de forcer une valeur non fiable.
    """
    if mid_price is None or mid_price <= 0:
        return None
    if spot is None or spot <= 0 or strike is None or strike <= 0:
        return None
    if days_to_expiry is None or days_to_expiry <= 0:
        return None

    t = days_to_expiry / 365.0
    intrinsic = max(0.0, (spot - strike) if option_type == "CALL" else (strike - spot))
    if mid_price < intrinsic:
        return None  # prix incohérent avec la valeur intrinsèque (quote figée/aberrante)

    try:
        iv = brentq(lambda v: _bs_price(spot, strike, t, v, option_type) - mid_price, 1e-4, 5.0, maxiter=100)
    except ValueError:
        return None  # pas de racine dans [0.01%, 500%] de vol : prix trop aberrant pour être inversé

    return {"implied_vol": iv, **_bs_greeks(spot, strike, t, iv, option_type)}


def fetch_market_data(ib: IB, contracts: list[Option]) -> dict:
    data = {}
    for i in range(0, len(contracts), BATCH_SIZE):
        batch = contracts[i:i + BATCH_SIZE]
        tickers = []
        for c in batch:
            try:
                # Tick "106" (greeks calculés par IBKR) retiré : on calcule
                # nous-mêmes l'IV/greeks localement (estimate_iv_and_greeks),
                # donc ce tick n'apporte rien et sollicite un calcul serveur
                # par contrat qui peut être lent/bloquant sur un compte sans
                # abonnement de données d'options temps réel/figé.
                t = ib.reqMktData(c, "101", False, False)
                tickers.append((c, t))
            except Exception as exc:  # noqa: BLE001
                logger.debug("reqMktData a échoué pour %s: %s", c, exc)

        _wait_for_data(ib, [t for _, t in tickers], SNAPSHOT_MAX_WAIT_SEC)

        for c, t in tickers:
            open_interest = t.callOpenInterest if c.right == "C" else t.putOpenInterest
            greeks = t.modelGreeks
            data[c.conId] = {
                "last_price": _clean(t.last), "bid": _clean(t.bid), "ask": _clean(t.ask),
                "close": _clean(t.close), "high": _clean(t.high), "low": _clean(t.low),
                "volume": _clean(t.volume), "open_interest": _clean(open_interest),
                "implied_vol": _clean(getattr(greeks, "impliedVol", None)) if greeks else None,
                "delta": _clean(getattr(greeks, "delta", None)) if greeks else None,
                "gamma": _clean(getattr(greeks, "gamma", None)) if greeks else None,
                "vega": _clean(getattr(greeks, "vega", None)) if greeks else None,
                "theta": _clean(getattr(greeks, "theta", None)) if greeks else None,
            }

        for c, _ in tickers:
            try:
                ib.cancelMktData(c)
            except Exception:  # noqa: BLE001
                pass

        ib.sleep(BATCH_PAUSE_SEC)

    return data


def _clean(value):
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except TypeError:
        return value
    if isinstance(value, (int, float)) and value in (-1.0, -1):
        return None
    return value


def build_candidate_contracts(stock: Contract, chain, spot: float, min_days: int, band_pct: float) -> list[Option]:
    """
    Construit la liste des contrats CANDIDATS (strike x échéance x C/P)
    directement à partir de chain.strikes / chain.expirations -- des listes
    déjà en mémoire (obtenues via reqSecDefOptParams dans get_option_chain,
    UN SEUL appel léger), filtrées aux mêmes critères qu'avant (>270 jours,
    ±30% du spot).

    Avant : fetch_chain_contracts demandait reqContractDetails() en mode
    wildcard, qui renvoie TOUS les contrats du sous-jacent (potentiellement
    plusieurs milliers pour un nom liquide type AAPL, toutes échéances et
    tous strikes confondus) -- puis filter_contracts() en jetait l'immense
    majorité. Ici on ne construit QUE les combinaisons qui nous intéressent
    déjà, avant même de contacter IBKR pour les détails -- même couverture
    finale, beaucoup moins de contrats à qualifier ensuite.
    """
    cutoff = datetime.now() + timedelta(days=min_days)
    valid_expiries = []
    for e in chain.expirations:
        try:
            if datetime.strptime(e, "%Y%m%d") > cutoff:
                valid_expiries.append(e)
        except (ValueError, TypeError):
            continue

    lo, hi = spot * (1 - band_pct), spot * (1 + band_pct)
    valid_strikes = [s for s in chain.strikes if lo <= s <= hi]

    contracts = []
    for expiry in valid_expiries:
        for strike in valid_strikes:
            for right in ("C", "P"):
                contracts.append(Option(
                    symbol=stock.symbol, lastTradeDateOrContractMonth=expiry, strike=strike,
                    right=right, exchange=chain.exchange, currency=stock.currency,
                    tradingClass=chain.tradingClass, multiplier=chain.multiplier,
                ))
    return contracts


def qualify_contracts_in_batches(ib: IB, contracts: list[Option], batch_size: int = 100) -> list[Option]:
    """Résout (qualifie : conId, exchange réel, etc.) les contrats candidats
    par lots. Certaines combinaisons de la grille théorique strike x échéance
    ne sont pas réellement cotées (pas toutes les échéances n'ont pas tous
    les strikes) : ib_insync les laisse simplement de côté sans lever
    d'erreur bloquante, ce qui remplace naturellement l'ancien
    filter_contracts (déjà appliqué en amont, dans build_candidate_contracts)."""
    qualified: list[Option] = []
    for i in range(0, len(contracts), batch_size):
        batch = contracts[i:i + batch_size]
        try:
            qualified.extend(c for c in ib.qualifyContracts(*batch) if c.conId)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Qualification du lot %d-%d échouée (%s), on continue.", i, i + len(batch), exc)
    return qualified


def process_ticker(ib: IB, row: pd.Series) -> list[dict]:
    symbol = row["ib_symbol"]
    exchange = row["ib_exchange"]
    currency = row["Currency"]
    company_name = row["Instrument_Name"]

    stock = resolve_stock_contract(ib, symbol, exchange, currency)
    if stock is None:
        raise PermanentTickerError(f"Impossible de résoudre le contrat action pour {symbol} ({exchange})")

    spot = get_spot_price(ib, stock)
    if spot is None:
        if exchange in BLACKLISTED_EXCHANGES:
            raise PermanentTickerError(f"Spot indisponible pour {symbol} : exchange {exchange} sans abonnement")
        raise RuntimeError(f"Impossible d'obtenir le spot pour {symbol}")

    chain = get_option_chain(ib, stock)
    if chain is None:
        raise PermanentTickerError(f"Chaîne d'options indisponible pour {symbol}")

    candidates = build_candidate_contracts(stock, chain, spot, MIN_DAYS_TO_EXPIRY, STRIKE_BAND_PCT)
    if not candidates:
        raise PermanentTickerError(
            f"Aucun contrat candidat pour {symbol} (spot={spot}, {len(chain.strikes)} strikes / "
            f"{len(chain.expirations)} échéances dispo côté IBKR)"
        )

    selected = qualify_contracts_in_batches(ib, candidates)
    if not selected:
        raise PermanentTickerError(
            f"Aucun des {len(candidates)} contrats candidats n'a pu être qualifié pour {symbol}"
        )

    md = fetch_market_data(ib, selected)

    rows = []
    now_dt = datetime.now()
    now = now_dt.isoformat(timespec="seconds")
    for c in selected:
        vals = md.get(c.conId, {})
        implied_vol = vals.get("implied_vol")
        delta, gamma, vega, theta = vals.get("delta"), vals.get("gamma"), vals.get("vega"), vals.get("theta")

        if implied_vol is not None:
            iv_source = "ibkr"
        else:
            # IBKR n'a pas fourni de modelGreeks (cas systématique en
            # MARKET_DATA_TYPE=4) : on estime l'IV et les greeks nous-mêmes
            # à partir du prix de l'option (cf. estimate_iv_and_greeks).
            bid, ask = vals.get("bid"), vals.get("ask")
            mid_price = (bid + ask) / 2 if bid is not None and ask is not None else (vals.get("last_price") or vals.get("close"))
            try:
                expiry_dt = datetime.strptime(c.lastTradeDateOrContractMonth, "%Y%m%d")
                days_to_expiry = (expiry_dt - now_dt).days
            except (ValueError, TypeError):
                days_to_expiry = None
            option_type = "CALL" if c.right == "C" else "PUT"
            estimate = estimate_iv_and_greeks(mid_price, spot, c.strike, days_to_expiry, option_type)
            if estimate:
                implied_vol = estimate["implied_vol"]
                delta, gamma, vega, theta = estimate["delta"], estimate["gamma"], estimate["vega"], estimate["theta"]
                iv_source = "estime_local"
            else:
                iv_source = "indisponible"

        rows.append({
            "symbol": symbol, "company_name": company_name, "expiry": c.lastTradeDateOrContractMonth,
            "strike": c.strike, "option_type": "CALL" if c.right == "C" else "PUT",
            "last_price": vals.get("last_price"), "bid": vals.get("bid"), "ask": vals.get("ask"),
            "volume": vals.get("volume"), "open_interest": vals.get("open_interest"),
            "close": vals.get("close"), "high": vals.get("high"), "low": vals.get("low"),
            "implied_vol": implied_vol, "delta": delta, "gamma": gamma,
            "vega": vega, "theta": theta, "iv_source": iv_source, "underlying_spot": spot,
            "moneyness_pct": round((c.strike / spot - 1) * 100, 2), "currency": currency,
            "exchange": c.exchange, "trading_class": c.tradingClass, "multiplier": c.multiplier,
            "con_id": c.conId, "fetch_timestamp": now,
        })
    return rows


def _progress_path(output_dir: Path) -> Path:
    return output_dir / "progress.json"


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint.jsonl"


def load_progress(output_dir: Path) -> tuple[set[str], set[str]]:
    path = _progress_path(output_dir)
    if not path.exists():
        return set(), set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return set(payload.get("processed", [])), set(payload.get("blacklisted_exchanges", []))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Fichier de progression illisible (%s), on repart de zéro.", exc)
        return set(), set()


def save_progress(output_dir: Path, processed: set[str]) -> None:
    path = _progress_path(output_dir)
    tmp = path.with_suffix(".json.tmp")
    payload = {
        "processed": sorted(processed),
        "blacklisted_exchanges": sorted(BLACKLISTED_EXCHANGES),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_checkpoint(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with _checkpoint_path(output_dir).open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")


def load_checkpoint_rows(output_dir: Path) -> list[dict]:
    path = _checkpoint_path(output_dir)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", type=Path, default=config.UNIVERSE_FILE)
    parser.add_argument("--output-dir", default=config.DIR_OPTIONS, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--auto-restart-gateway", action="store_true",
        help="Si la récupération légère (fermeture des process fantômes + reconnexion) échoue "
             "3 fois de suite, tente un redémarrage complet d'IB Gateway via IBC (nécessite .env "
             "configuré, voir .env.example et restart_gateway.py).",
    )
    args = parser.parse_args()

    setup_logging(args.output_dir)
    universe = load_universe(args.tickers, limit=args.limit)

    processed_keys: set[str] = set()
    if args.resume:
        processed_keys, seeded_blacklist = load_progress(args.output_dir)
        BLACKLISTED_EXCHANGES.update(seeded_blacklist)
        logger.info("Reprise : %d tickers déjà traités, exchanges blacklistés: %s", len(processed_keys), sorted(BLACKLISTED_EXCHANGES))
    else:
        _progress_path(args.output_dir).unlink(missing_ok=True)
        _checkpoint_path(args.output_dir).unlink(missing_ok=True)

    ib = connect_ib()
    ok_count, fail_count, skip_count = 0, 0, 0
    since_last_checkpoint = 0

    try:
        for idx, row in universe.iterrows():
            symbol = row["ib_symbol"]
            exchange = row["ib_exchange"]
            key = f"{symbol}|{exchange}"

            if key in processed_keys:
                continue

            if exchange in BLACKLISTED_EXCHANGES:
                skip_count += 1
                processed_keys.add(key)
                logger.info("[%d/%d] %s -> SKIP : exchange %s sans abonnement de données.", idx + 1, len(universe), symbol, exchange)
                continue

            ensure_connected(ib)

            logger.info("[%d/%d] %s (%s)...", idx + 1, len(universe), symbol, row["Instrument_Name"])
            last_exc, rows = None, None
            attempt = 0
            session_recoveries = 0

            while attempt <= RETRY_PER_TICKER:
                try:
                    rows = process_ticker(ib, row)
                    if _session_poisoned.is_set():
                        # L'erreur peut arriver de façon asynchrone (callback
                        # errorEvent) sans forcément faire échouer l'appel en
                        # cours -- on vérifie explicitement après coup.
                        raise RuntimeError("Session IBKR corrompue détectée pendant le traitement.")
                    break
                except PermanentTickerError as exc:
                    last_exc = exc
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc

                    if _session_poisoned.is_set():
                        if session_recoveries >= MAX_SESSION_RECOVERIES_PER_TICKER:
                            logger.error(
                                "Erreur de session IBKR toujours présente après %d tentatives de "
                                "récupération automatique pour %s. Abandon -- redémarre TWS/IB "
                                "Gateway manuellement, puis relance avec --resume.",
                                MAX_SESSION_RECOVERIES_PER_TICKER, symbol,
                            )
                            break
                        session_recoveries += 1
                        logger.warning("Tentative de récupération de session %d/%d pour %s...",
                                        session_recoveries, MAX_SESSION_RECOVERIES_PER_TICKER, symbol)
                        try:
                            recover_from_session_poisoning(
                                ib,
                                auto_restart_gateway=args.auto_restart_gateway,
                                escalate=(session_recoveries >= MAX_SESSION_RECOVERIES_PER_TICKER),
                            )
                        except Exception as recover_exc:  # noqa: BLE001
                            logger.error("Échec de la récupération automatique de session : %s", recover_exc)
                            break
                        continue  # retente CE ticker, sans consommer de tentative RETRY_PER_TICKER

                    if exchange in BLACKLISTED_EXCHANGES:
                        break
                    if attempt < RETRY_PER_TICKER:
                        logger.debug("  -> tentative %d échouée pour %s (%s), nouvel essai...", attempt + 1, symbol, exc)
                        ib.sleep(RETRY_PAUSE_SEC)
                attempt += 1

            processed_keys.add(key)
            if rows is not None:
                append_checkpoint(args.output_dir, rows)
                ok_count += 1
                logger.info("  -> %d contrats récupérés pour %s", len(rows), symbol)
            else:
                fail_count += 1
                logger.warning("  -> ECHEC pour %s : %s (ticker ignoré, on continue)", symbol, last_exc)

            since_last_checkpoint += 1
            if since_last_checkpoint >= CHECKPOINT_EVERY:
                save_progress(args.output_dir, processed_keys)
                since_last_checkpoint = 0
    finally:
        save_progress(args.output_dir, processed_keys)
        ib.disconnect()
        logger.info("Déconnecté d'IBKR.")

    logger.info("Terminé. OK: %d | Echecs: %d | Skippés: %d | Exchanges blacklistés: %s", ok_count, fail_count, skip_count, sorted(BLACKLISTED_EXCHANGES))

    all_rows = load_checkpoint_rows(args.output_dir)
    if not all_rows:
        logger.warning("Aucune donnée récupérée, pas de fichier de sortie généré.")
        return

    df = pd.DataFrame(all_rows)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Horodatage unique pour tout le run (une ligne = une date de snapshot),
    # pratique pour regrouper/filtrer les archives par date côté rapport.
    run_started_at = datetime.now()
    df["snapshot_date"] = run_started_at.date().isoformat()
    df["snapshot_datetime"] = run_started_at.isoformat(timespec="seconds")

    # --output-dir pilote désormais réellement l'emplacement du fichier de
    # sortie (avant : seuls logs/checkpoint en tenaient compte, le fichier de
    # données allait toujours dans config.OPTIONS_FILE quel que soit l'argument).
    output_file = args.output_dir / config.OPTIONS_FILE.name
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False, engine="pyarrow")
    logger.info("Fichier écrit (dernier snapshot) : %s (%d lignes)", output_file, len(df))

    # Archive en plus un fichier horodaté, JAMAIS écrasé, pour construire une
    # nappe de volatilité qui évolue dans le temps (le fichier ci-dessus,
    # lui, est toujours remplacé au run suivant). Un fichier par run.
    config.DIR_OPTIONS_HISTORY.mkdir(parents=True, exist_ok=True)
    history_file = config.DIR_OPTIONS_HISTORY / f"option_chains_{run_started_at:%Y%m%d_%H%M%S}.parquet"
    df.to_parquet(history_file, index=False, engine="pyarrow")
    logger.info("Fichier archivé (historique) : %s (%d lignes)", history_file, len(df))


if __name__ == "__main__":
    main()
