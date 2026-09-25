"""
Paper trading de la stratégie actions : le compte paper IBKR RÉPLIQUE le
portefeuille que le backtest détient aujourd'hui.

LE PRINCIPE
-----------
Chaque run rejoue le moteur actions (backtest/engine.py) depuis la date de
départ jusqu'à la dernière clôture des données, avec EXACTEMENT la
configuration de 09_backtest.py (backtest/construction_moteur.py). Le moteur
décide à la clôture et exécute à l'ouverture suivante : à la fin du run, il
détient un portefeuille ET a mis en attente les ordres de demain matin.
Ensemble, ils forment le portefeuille CIBLE de l'ouverture suivante
(BacktestEngine.cibles_ouverture_suivante). On le traduit en POIDS du NAV, on
rapporte ces poids au NAV réel du compte paper, et on envoie la différence en
ordres « au marché à l'ouverture » -- l'hypothèse d'exécution du backtest.

POURQUOI REJOUER TOUT L'HISTORIQUE plutôt que tenir un état local. Les sorties
dépendent du passé de chaque position : référence du stop figée à l'entrée,
plus haut du stop suiveur, durée de détention ; et le ciblage de volatilité lit
la courbe de NAV des 60 dernières séances. Tenir tout cela dans un fichier,
c'est réécrire la comptabilité du moteur et diverger de lui au premier cas
limite. Le rejouer coûte une trentaine de secondes et garantit que le compte
trade ce que le backtest aurait tradé : ni plus, ni autre chose.

CE QUI PASSE D'UN PORTEFEUILLE À L'AUTRE : LES POIDS, PAS LES MONTANTS. Le
moteur a démarré en 2015 avec son propre capital, le compte a le sien ; seule
la répartition est répliquée.

LES TROIS RÈGLES DE RÉCONCILIATION (planifier_ordres)
  1. Une ligne que le moteur trade à l'ouverture suivante est amenée à sa
     cible, sous réserve du même plancher de taille que le moteur -- sauf une
     liquidation, qui passe toujours (même règle que le moteur).
  2. Un écart de STRUCTURE est toujours corrigé : une ligne que le compte
     détient et pas le moteur est liquidée, une ligne que le moteur détient et
     pas le compte est achetée. Premier run, ordre refusé, run manqué : c'est
     ainsi que le compte rattrape le moteur.
  3. Une ligne détenue des deux côtés, que le moteur ne touche pas, n'est
     corrigée qu'au-delà de la tolérance. Les deux portefeuilles bougent avec
     les mêmes cours, leurs poids restent alignés à l'écart d'exécution près ;
     corriger cet écart chaque jour recréerait la rotation que la stratégie
     ancrée existe pour supprimer.

LES GARDE-FOUS
  - Compte PAPER uniquement. IBKR numérote ses comptes papier « D… » (DU… pour
    un compte individuel) et ses comptes réels « U… » : tout autre compte est
    refusé, en simulation comme en envoi.
  - Rien n'est envoyé sans --transmettre. Par défaut, le plan est calculé,
    affiché et journalisé ; c'est tout.
  - Données trop anciennes, ordre démesuré face au NAV : envoi refusé.
  - Jamais de levier : les achats sont ramenés au cash disponible, produit des
    ventes du jour compris. Le moteur n'est pas margé, le compte ne doit pas
    l'être.
  - Seuls NOS ordres encore ouverts (étiquette PAPER_TRADING_ORDER_REF) sont
    annulés avant un nouvel envoi, et une position hors du champ de la
    stratégie (autre classe d'actifs, autre devise, titre inconnu du pipeline)
    n'est jamais touchée. Le compte doit néanmoins être DÉDIÉ à la stratégie :
    toute action US dotée d'un signal que le moteur ne détient pas y est
    vendue (cf. symboles_geres).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

import config
import ecriture_atomique

logger = logging.getLogger("paper_trading")

# IBKR : « DU1234567 » = compte papier individuel, « U1234567 » = compte réel.
PREFIXE_COMPTE_PAPER = "D"

# Statuts IB d'un ordre encore vivant, donc annulable.
STATUTS_OUVERTS = frozenset({"ApiPending", "PendingSubmit", "PreSubmitted", "Submitted"})

# Part du cash tenue en réserve : les ordres sont dimensionnés sur la clôture et
# exécutés à l'ouverture suivante, qui peut avoir bougé.
MARGE_CASH = 0.01

# Type d'ordre -> durée de validité IB. « moo » (market-on-open, tif OPG) est
# l'hypothèse d'exécution du backtest ; « marche » part au marché pendant la
# séance, pour un run lancé en journée.
TYPES_ORDRE = {"moo": "OPG", "marche": "DAY"}

RAISON_HORS_MOTEUR = "hors_portefeuille_moteur"
RAISON_ACHAT_MANQUANT = "absente_du_compte"
RAISON_DERIVE = "derive_au_dela_de_la_tolerance"


class RefusEnvoi(RuntimeError):
    """Une condition interdit de continuer. Le message dit laquelle, et quoi faire."""


# --------------------------------------------------------------------------- #
# Ce que le moteur veut détenir
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LigneCible:
    symbol: str
    poids: float      # part du NAV visée à l'ouverture suivante (0 = liquidation)
    cours: float      # dernière clôture : sert au dimensionnement
    raison: str       # raison de l'ordre du moteur ; vide si la ligne est conservée telle quelle

    @property
    def ordre_du_moteur(self) -> bool:
        return bool(self.raison)


@dataclass(frozen=True)
class PortefeuilleCible:
    date: pd.Timestamp
    nav_moteur: float
    lignes: dict

    @property
    def exposition(self) -> float:
        return sum(ligne.poids for ligne in self.lignes.values())


def symboles_geres(engine) -> set:
    """Les titres que la stratégie peut détenir : ceux qu'un signal de
    valorisation lui a fait connaître, plus ceux qu'elle détient. PAS tout le
    panel de cours, qui porte aussi l'indice de référence (SPY) : une position
    prise à la main sur un titre sans signal n'est pas la sienne, elle n'a pas
    à la vendre."""
    return set(engine.known_signals) | set(engine.positions) | set(engine.pending_orders)


def portefeuille_cible(engine) -> PortefeuilleCible:
    """Le portefeuille de l'ouverture suivante, en poids du NAV du moteur."""
    date_signal, nav, cibles = engine.cibles_ouverture_suivante()
    if not nav > 0:
        raise RefusEnvoi(f"NAV du moteur non positif au {date_signal.date()} ({nav}) : rien à répliquer.")
    lignes = {
        symbol: LigneCible(symbol, valeur / nav, cours, raison)
        for symbol, (valeur, cours, raison) in cibles.items()
    }
    return PortefeuilleCible(date_signal, nav, lignes)


# --------------------------------------------------------------------------- #
# Ce que le compte détient
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EtatCompte:
    compte: str
    nav: float                 # en dollars US
    cash: float                # en dollars US, toutes devises converties
    detentions: dict = field(default_factory=dict)      # symbole -> actions US détenues
    hors_strategie: list = field(default_factory=list)  # positions que la stratégie ne gère pas
    devise_base: str = "USD"


def compte_hors_ligne(capital: float) -> EtatCompte:
    """Compte fictif et vide, pour calculer un plan sans IB Gateway : c'est le
    plan d'un PREMIER run, celui qui construit tout le portefeuille."""
    return EtatCompte("HORS-LIGNE", capital, capital)


def valeur_detentions(compte: EtatCompte, cours_de: Callable[[str], Optional[float]]) -> float:
    return sum(
        actions * cours for symbol, actions in compte.detentions.items()
        if (cours := cours_de(symbol)) is not None
    )


def appliquer_capital(
    compte: EtatCompte, capital: float, cours_de: Callable[[str], Optional[float]],
) -> EtatCompte:
    """Alloue à la stratégie un capital FIXE, en dollars, au lieu du NAV entier
    du compte : la stratégie gère alors une poche, dont le cash est le capital
    moins la valeur des actions qu'elle détient déjà."""
    if not capital > 0:
        raise RefusEnvoi(f"--capital doit être positif (reçu {capital}).")
    if compte.nav > 0 and capital > compte.nav * 1.0001:
        raise RefusEnvoi(
            f"--capital ({capital:,.0f} $) dépasse le NAV du compte ({compte.nav:,.0f} $) : "
            "ce serait du levier, que ni le moteur ni ce script ne prennent.")
    cash = capital - valeur_detentions(compte, cours_de)
    # La poche ne dispose jamais de plus que le cash réel du compte -- quand on
    # le connaît : sans taux de change, il n'est pas convertible en dollars.
    if math.isfinite(compte.cash):
        cash = min(cash, compte.cash)
    return dataclasses.replace(compte, nav=capital, cash=cash)


# --------------------------------------------------------------------------- #
# Le plan d'ordres
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Ordre:
    symbol: str
    quantite: int          # signée : > 0 achat, < 0 vente
    cours: float
    raison: str
    detenu: float
    cible: int
    poids_cible: float
    poids_actuel: float

    @property
    def sens(self) -> str:
        return "BUY" if self.quantite > 0 else "SELL"

    @property
    def montant(self) -> float:
        return abs(self.quantite) * self.cours


def planifier_ordres(
    cible: PortefeuilleCible,
    compte: EtatCompte,
    montant_minimal: float,
    tolerance_pct: float = config.PAPER_TRADING_TOLERANCE_PCT,
    cours_de: Callable[[str], Optional[float]] = lambda symbol: None,
) -> tuple[list[Ordre], list[str]]:
    """Les ordres qui amènent le compte au portefeuille cible -- ventes d'abord,
    puis achats -- et les remarques à afficher avec eux (cf. les trois règles
    de la docstring du module).

    `montant_minimal` : le plancher de taille du MOTEUR, rapporté au NAV du
    compte (BacktestEngine._montant_minimal). `cours_de` chiffre les lignes du
    compte que le moteur ne détient pas ; il n'en faut pas pour les vendre."""
    nav = compte.nav
    if not nav > 0:
        raise RefusEnvoi(f"NAV du compte {compte.compte} non positif ({nav}).")
    tolerance = tolerance_pct / 100.0 * nav
    ordres: list[Ordre] = []
    remarques: list[str] = []

    for symbol in sorted(set(cible.lignes) | set(compte.detentions)):
        detenu = compte.detentions.get(symbol, 0)
        ligne = cible.lignes.get(symbol)

        if ligne is None:
            # Règle 2 : le moteur ne détient pas cette ligne.
            quantite = -int(round(detenu))
            if not quantite:
                continue
            cours = cours_de(symbol)
            cours = float("nan") if cours is None else cours
            ordres.append(Ordre(
                symbol, quantite, cours, RAISON_HORS_MOTEUR, detenu, 0, 0.0,
                detenu * cours / nav if cours == cours else float("nan"),
            ))
            continue

        if not ligne.cours > 0:
            remarques.append(f"{symbol} : cours inexploitable ({ligne.cours}) -- ligne ignorée.")
            continue
        cible_actions = math.floor(ligne.poids * nav / ligne.cours + 1e-9) if ligne.poids > 0 else 0
        if ligne.poids > 0 and cible_actions == 0:
            # Actions ENTIÈRES : une cible sous le prix d'une action tombe à
            # zéro. Rien de faux, mais la ligne manquera au compte -- à dire.
            remarques.append(
                f"{symbol} : cible de {ligne.poids * nav:,.0f} $ inférieure au prix d'une action "
                f"({ligne.cours:,.2f} $) -- ligne absente du compte.")
        delta = int(round(cible_actions - detenu))
        if delta == 0:
            continue
        montant = abs(delta) * ligne.cours

        if ligne.poids <= 0:
            # Liquidation décidée par le moteur : passe toujours, comme dans
            # le moteur -- un plancher emprisonnerait la ligne, stop-loss compris.
            raison = ligne.raison or RAISON_HORS_MOTEUR
        elif ligne.ordre_du_moteur:                        # règle 1
            if montant < montant_minimal:
                remarques.append(
                    f"{symbol} : ordre du moteur ({ligne.raison}) de {montant:,.0f} $ sous le "
                    f"plancher de {montant_minimal:,.0f} $ -- non passé.")
                continue
            raison = ligne.raison
        elif not detenu:                                   # règle 2
            if montant < montant_minimal:
                remarques.append(
                    f"{symbol} : cible de {montant:,.0f} $ sous le plancher de "
                    f"{montant_minimal:,.0f} $ -- ligne non achetée.")
                continue
            raison = RAISON_ACHAT_MANQUANT
        else:                                              # règle 3
            if montant < max(montant_minimal, tolerance):
                continue
            raison = RAISON_DERIVE

        ordres.append(Ordre(
            symbol, delta, ligne.cours, raison, detenu, cible_actions,
            ligne.poids, detenu * ligne.cours / nav,
        ))

    ventes = [o for o in ordres if o.quantite < 0]
    achats = [o for o in ordres if o.quantite > 0]
    achats, remarque_cash = _ramener_au_cash(ventes, achats, compte.cash, montant_minimal)
    if remarque_cash:
        remarques.append(remarque_cash)
    return ventes + achats, remarques


def _ramener_au_cash(
    ventes: list[Ordre], achats: list[Ordre], cash: float, montant_minimal: float,
) -> tuple[list[Ordre], Optional[str]]:
    """Pas de levier : le total des achats tient dans le cash plus le produit
    des ventes du même jour, moins une marge pour l'écart clôture/ouverture.
    Tous les achats sont réduits dans la même proportion, comme le fait le
    moteur (_execute_buys)."""
    produit = sum(o.montant for o in ventes if o.montant == o.montant)
    disponible = max((cash + produit) * (1 - MARGE_CASH), 0.0)
    besoin = sum(o.montant for o in achats)
    if not achats or besoin <= disponible:
        return achats, None
    ratio = disponible / besoin
    reduits = []
    for o in achats:
        quantite = math.floor(o.quantite * ratio)
        if quantite <= 0 or quantite * o.cours < montant_minimal:
            continue
        reduits.append(dataclasses.replace(o, quantite=quantite, cible=int(round(o.detenu + quantite))))
    return reduits, (
        f"Cash insuffisant : achats ramenés à {ratio:.1%} ({besoin:,.0f} $ demandés, "
        f"{disponible:,.0f} $ disponibles, marge de {MARGE_CASH:.0%} comprise).")


def verifier_envoi(
    ordres: list[Ordre],
    compte: EtatCompte,
    date_donnees: pd.Timestamp,
    aujourd_hui: pd.Timestamp,
    max_age_jours: int = config.PAPER_TRADING_MAX_DATA_AGE_DAYS,
    plafond_pct: float = config.PAPER_TRADING_MAX_ORDER_PCT_OF_NAV,
) -> None:
    """Lève RefusEnvoi si l'une des conditions d'envoi manque."""
    age = (pd.Timestamp(aujourd_hui).normalize() - pd.Timestamp(date_donnees).normalize()).days
    if age > max_age_jours:
        raise RefusEnvoi(
            f"Dernière clôture des données : {pd.Timestamp(date_donnees).date()}, il y a {age} "
            f"jours (plus de {max_age_jours}). Relance d'abord le pipeline quotidien "
            "(run_pipeline_daily.py) : un plan calculé sur ces cours vaudrait pour une autre date.")
    plafond = plafond_pct / 100.0 * compte.nav
    trop = [o for o in ordres if o.montant == o.montant and o.montant > plafond]
    if trop:
        detail = ", ".join(f"{o.symbol} {o.montant:,.0f} $" for o in trop)
        raise RefusEnvoi(
            f"Ordre(s) au-delà de {plafond_pct:g} % du NAV ({plafond:,.0f} $) : {detail}. La "
            "stratégie plafonne ses lignes bien en deçà : c'est le signe d'une erreur "
            "d'échelle (NAV, devise, cours), pas d'une décision. Rien n'est envoyé.")


# --------------------------------------------------------------------------- #
# IB Gateway
# --------------------------------------------------------------------------- #
def comptes_geres(ib) -> list[str]:
    comptes = list(ib.managedAccounts() or [])
    if not comptes:
        # Connexion « API seule » (ib_connect) : la liste arrive à la poignée
        # de main, côté client, sans passer par la synchronisation.
        try:
            comptes = list(ib.client.getAccounts())
        except Exception:  # noqa: BLE001 -- pas de liste : choisir_compte_paper le dira
            comptes = []
    return comptes


def choisir_compte_paper(comptes: list[str], demande: Optional[str] = None) -> str:
    """Le compte à utiliser -- et le refus net de tout compte qui n'est pas
    un compte paper."""
    if demande:
        if demande not in comptes:
            raise RefusEnvoi(
                f"Compte {demande} inconnu de cette session IB (comptes : {', '.join(comptes) or 'aucun'}).")
        compte = demande
    elif len(comptes) == 1:
        compte = comptes[0]
    elif not comptes:
        raise RefusEnvoi("La session IB ne déclare aucun compte.")
    else:
        raise RefusEnvoi(f"Plusieurs comptes gérés ({', '.join(comptes)}) : précise --compte.")
    if not compte.startswith(PREFIXE_COMPTE_PAPER):
        raise RefusEnvoi(
            f"{compte} n'est PAS un compte paper : IBKR numérote ses comptes papier « D… » "
            "(DU1234567) et ses comptes réels « U… ». Ce script refuse de trader un compte "
            "réel. Connecte IB Gateway en mode Paper Trading (port 4002 par défaut).")
    return compte


def _valeur_resume(resume, tag: str, devise: Optional[str] = None):
    for valeur in resume:
        if valeur.tag == tag and (devise is None or valeur.currency == devise):
            return valeur
    return None


def lire_compte(ib, compte: str, symboles_geres: set, capital_fourni: bool = False) -> EtatCompte:
    """NAV et cash du compte, convertis en dollars, et ses positions en
    actions US que la stratégie gère.

    POURQUOI LA CONVERSION. La stratégie trade des actions américaines, en
    dollars, mais un compte paper IBKR hérite de la devise de base du compte
    réel -- l'euro pour un compte ouvert en France. Le NAV publié est alors en
    euros ; le taux vient du même résumé de compte (tag ExchangeRate, valeur
    d'un dollar en devise de base). Sans lui, et sans --capital, on refuse :
    dimensionner des ordres en dollars sur un NAV en euros les fausserait de
    tout l'écart de change."""
    resume = ib.accountSummary(compte)
    nav_brut = _valeur_resume(resume, "NetLiquidation")
    if nav_brut is None:
        raise RefusEnvoi(f"Le résumé du compte {compte} ne donne pas de NetLiquidation.")
    devise = nav_brut.currency or "USD"
    taux = 1.0
    if devise != "USD":
        ligne_taux = _valeur_resume(resume, "ExchangeRate", "USD")
        taux = float(ligne_taux.value) if ligne_taux is not None else float("nan")
        if not taux > 0:
            if not capital_fourni:
                raise RefusEnvoi(
                    f"Compte {compte} en {devise}, sans taux de change USD dans son résumé : "
                    "passe --capital (montant en dollars alloué à la stratégie).")
            taux = float("nan")
    cash_brut = _valeur_resume(resume, "TotalCashValue")
    nav = float(nav_brut.value) / taux
    cash = float(cash_brut.value) / taux if cash_brut is not None else 0.0

    detentions: dict = {}
    hors_strategie: list = []
    for position in ib.reqPositions():
        if position.account != compte or not position.position:
            continue
        contrat = position.contract
        if contrat.secType == "STK" and contrat.currency == "USD" and contrat.symbol in symboles_geres:
            detentions[contrat.symbol] = detentions.get(contrat.symbol, 0) + float(position.position)
        else:
            hors_strategie.append(
                f"{contrat.symbol} ({contrat.secType} {contrat.currency}) x {float(position.position):g}")
    return EtatCompte(compte, nav, cash, detentions, hors_strategie, devise)


def ordre_ib(ordre: Ordre, compte: str, type_ordre: str = "moo"):
    from ib_insync import MarketOrder

    order = MarketOrder(ordre.sens, abs(ordre.quantite))
    order.tif = TYPES_ORDRE[type_ordre]
    order.account = compte
    order.orderRef = config.PAPER_TRADING_ORDER_REF
    return order


def annuler_nos_ordres_ouverts(ib, attente: float = 2.0) -> list[str]:
    """Annule les ordres encore ouverts que CE script a passés (étiquette
    PAPER_TRADING_ORDER_REF) : relancer le script le même soir REMPLACE le plan
    précédent au lieu de le doubler. Les autres ordres du compte ne sont pas
    touchés. IBKR ne rend à un client que ses propres ordres : d'où
    l'identifiant client fixe (config.PAPER_TRADING_CLIENT_ID)."""
    nos_ordres = [
        trade for trade in ib.reqOpenOrders()
        if trade.order.orderRef == config.PAPER_TRADING_ORDER_REF
        and trade.orderStatus.status in STATUTS_OUVERTS
    ]
    for trade in nos_ordres:
        ib.cancelOrder(trade.order)
    if nos_ordres:
        ib.sleep(attente)
    return [f"{t.contract.symbol} {t.order.action} {t.order.totalQuantity:g}" for t in nos_ordres]


def transmettre(ib, ordres: list[Ordre], compte: str, type_ordre: str = "moo",
                attente: float = 3.0) -> list[dict]:
    """Envoie les ordres (ventes d'abord, dans l'ordre du plan) et rend leur
    statut IB après `attente` secondes, messages d'erreur compris."""
    from ib_insync import Stock

    contrats = {o.symbol: Stock(o.symbol, "SMART", "USD") for o in ordres}
    if contrats:
        ib.qualifyContracts(*contrats.values())
    envoyes, resultats = [], []
    for o in ordres:
        contrat = contrats[o.symbol]
        if not getattr(contrat, "conId", 0):
            resultats.append({"symbol": o.symbol, "statut": "non_envoye",
                              "message": "contrat introuvable chez IBKR"})
            continue
        envoyes.append((o, ib.placeOrder(contrat, ordre_ib(o, compte, type_ordre))))
    if envoyes:
        ib.sleep(attente)
    for o, trade in envoyes:
        messages = [entree.message for entree in getattr(trade, "log", []) if getattr(entree, "message", "")]
        resultats.append({"symbol": o.symbol, "statut": trade.orderStatus.status,
                          "message": " | ".join(messages)})
    return resultats


# --------------------------------------------------------------------------- #
# Journal
# --------------------------------------------------------------------------- #
def tableau_ordres(ordres: list[Ordre], resultats: Optional[list[dict]] = None) -> pd.DataFrame:
    statuts = {r["symbol"]: r for r in resultats or []}
    return pd.DataFrame([{
        "symbol": o.symbol, "sens": o.sens, "quantite": abs(o.quantite),
        "cours_reference": round(o.cours, 4), "montant": round(o.montant, 2),
        "raison": o.raison, "detenu": o.detenu, "cible": o.cible,
        "poids_cible_pct": round(100 * o.poids_cible, 3),
        "poids_actuel_pct": round(100 * o.poids_actuel, 3),
        "statut": statuts.get(o.symbol, {}).get("statut", ""),
        "message": statuts.get(o.symbol, {}).get("message", ""),
    } for o in ordres], columns=[
        "symbol", "sens", "quantite", "cours_reference", "montant", "raison", "detenu", "cible",
        "poids_cible_pct", "poids_actuel_pct", "statut", "message",
    ])


def _ajouter_csv(chemin: Path, lignes: pd.DataFrame) -> None:
    if lignes.empty:
        return
    lignes.to_csv(chemin, mode="a", header=not chemin.exists(), index=False, encoding="utf-8")


def journaliser(
    repertoire: Path,
    horodatage: str,
    mode: str,
    compte: EtatCompte,
    cible: PortefeuilleCible,
    ordres: list[Ordre],
    remarques: list[str],
    exposition_compte: float,
    resultats: Optional[list[dict]] = None,
    parametres: Optional[dict] = None,
) -> None:
    """Complète ordres.csv et compte.csv, réécrit dernier_run.json. Les deux
    CSV s'allongent d'un run à l'autre : c'est l'historique du compte paper,
    à comparer à la courbe du backtest."""
    repertoire.mkdir(parents=True, exist_ok=True)
    tableau = tableau_ordres(ordres, resultats)
    if mode == "simulation":
        tableau["statut"] = "simulation"
    entete = {
        "horodatage": horodatage, "date_signal": cible.date.date().isoformat(),
        "compte": compte.compte, "mode": mode,
    }
    _ajouter_csv(repertoire / "ordres.csv", pd.concat(
        [pd.DataFrame([entete] * len(tableau)), tableau], axis=1))
    _ajouter_csv(repertoire / "compte.csv", pd.DataFrame([{
        **entete,
        "nav_compte_usd": round(compte.nav, 2), "cash_compte_usd": round(compte.cash, 2),
        "exposition_compte_pct": round(100 * exposition_compte, 3),
        "nav_moteur": round(cible.nav_moteur, 2),
        "exposition_moteur_pct": round(100 * cible.exposition, 3),
        "lignes_moteur": sum(1 for ligne in cible.lignes.values() if ligne.poids > 0),
        "lignes_compte": len(compte.detentions),
        "ordres": len(ordres),
    }]))
    ecriture_atomique.ecrire_texte(repertoire / "dernier_run.json", json.dumps({
        **entete,
        "parametres": parametres or {},
        "nav_compte_usd": compte.nav, "cash_compte_usd": compte.cash,
        "devise_base": compte.devise_base, "positions_hors_strategie": compte.hors_strategie,
        "nav_moteur": cible.nav_moteur, "exposition_moteur": cible.exposition,
        "cibles": {
            s: {"poids": ligne.poids, "cours": ligne.cours, "raison": ligne.raison}
            for s, ligne in sorted(cible.lignes.items())
        },
        "ordres": tableau.to_dict("records"),
        "remarques": remarques,
    }, indent=2, ensure_ascii=False, default=str))
