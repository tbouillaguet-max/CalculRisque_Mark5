"""
Paper trading de la stratégie actions sur IB Gateway -- compte PAPER uniquement.

Rejoue le backtest de la stratégie (par défaut valuation_gap_combined_ancre,
depuis 2015, configuration de 09_backtest.py) jusqu'à la dernière clôture, lit
le portefeuille qu'il détiendra à l'ouverture suivante, et amène le compte
paper à ces mêmes poids par des ordres « au marché à l'ouverture ». Le
principe, les règles de réconciliation et les garde-fous sont décrits en tête
de paper_trading.py.

Prérequis : IB Gateway (ou TWS) connecté en mode Paper Trading, API activée
en écriture (Configure > Settings > API > Settings : « Enable ActiveX and
Socket Clients » coché, « Read-Only API » DÉCOCHÉ), et des données à jour --
lance-le après run_pipeline_daily.py, après la clôture américaine.

Usage :
    python 17_paper_trading.py                  # simulation : calcule et affiche le plan, n'envoie rien
    python 17_paper_trading.py --transmettre    # envoie les ordres au compte paper
    python 17_paper_trading.py --hors-ligne     # sans Gateway : plan d'un premier run sur un compte vide
    python 17_paper_trading.py --strategy valuation_gap_combined --transmettre
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

import pandas as pd

import config
import paper_trading
from backtest.construction_moteur import (
    ajouter_options_moteur, charger_donnees, construire_moteur, construire_strategie,
)
from backtest.strategies import STRATEGY_REGISTRY

logger = logging.getLogger("paper_trading.cli")


def _maintenant() -> pd.Timestamp:
    """L'heure du run -- isolée pour que les tests puissent la fixer."""
    return pd.Timestamp.now()


def _port_par_defaut() -> int:
    # La même source que l'orchestrateur : .env, repli sur 4002 (Gateway papier).
    from run_pipeline_quarterly import _gateway_port
    return _gateway_port()


def construire_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--strategy", default=config.PAPER_TRADING_STRATEGY,
        help="Stratégie actions rejouée (défaut: %(default)s).")
    parser.add_argument(
        "--start-date", default=config.PAPER_TRADING_START_DATE,
        help="Départ du backtest rejoué (défaut: %(default)s, celui des runs de référence). "
             "Le compte réplique le portefeuille que CE backtest détient aujourd'hui.")
    # Les réglages du moteur, avec les défauts de 09_backtest.py : le compte
    # paper teste la configuration mesurée, pas une variante.
    ajouter_options_moteur(parser)
    parser.add_argument(
        "--transmettre", action="store_true",
        help="Envoie les ordres au compte paper. Sans cette option : simulation (rien n'est envoyé).")
    parser.add_argument(
        "--hors-ligne", action="store_true",
        help="Sans IB Gateway : plan calculé contre un compte VIDE de --capital dollars "
             "(défaut 1 000 000) -- le plan d'un premier run. Incompatible avec --transmettre.")
    parser.add_argument("--port", type=int, default=None, help="Port API d'IB Gateway (défaut : .env, sinon 4002).")
    parser.add_argument(
        "--client-id", type=int, default=config.PAPER_TRADING_CLIENT_ID,
        help="Identifiant client API (défaut: %(default)s). Garde-le FIXE : IBKR ne laisse "
             "annuler un ordre qu'au client qui l'a passé.")
    parser.add_argument("--compte", default=None, help="Compte paper à utiliser, si la session en gère plusieurs.")
    parser.add_argument(
        "--type-ordre", choices=sorted(paper_trading.TYPES_ORDRE), default="moo",
        help="moo : au marché à l'OUVERTURE suivante, l'hypothèse du backtest (défaut). "
             "marche : au marché tout de suite, pour un run lancé pendant la séance.")
    parser.add_argument(
        "--capital", type=float, default=None,
        help="Capital en dollars alloué à la stratégie (défaut : tout le NAV du compte). "
             "Indispensable si le compte est dans une autre devise sans taux USD publié.")
    parser.add_argument(
        "--tolerance-pct", type=float, default=config.PAPER_TRADING_TOLERANCE_PCT,
        help="Écart, en points de NAV, au-delà duquel une ligne que le moteur ne trade pas est "
             "tout de même recalée sur sa cible (défaut: %(default)s).")
    parser.add_argument(
        "--max-data-age-days", type=int, default=config.PAPER_TRADING_MAX_DATA_AGE_DAYS,
        help="Âge maximal de la dernière clôture pour envoyer des ordres (défaut: %(default)s jours).")
    parser.add_argument(
        "--max-order-pct", type=float, default=config.PAPER_TRADING_MAX_ORDER_PCT_OF_NAV,
        help="Aucun ordre au-delà de cette part du NAV n'est envoyé (défaut: %(default)s %%).")
    parser.add_argument(
        "--sortie", default=None, help="Écrit aussi le plan dans ce fichier CSV.")
    return parser


def rejouer_strategie(args: argparse.Namespace) -> tuple[paper_trading.PortefeuilleCible, object]:
    logger.info("Chargement des données...")
    donnees = charger_donnees(args.strategy)
    strategy = construire_strategie(args.strategy, args)
    engine = construire_moteur(args, donnees, strategy, start_date=pd.Timestamp(args.start_date))
    logger.info(
        "Rejoue '%s' du %s au %s (%d jours de bourse)...",
        args.strategy, engine.calendar[0].date(), engine.calendar[-1].date(), len(engine.calendar))
    engine.run()
    return paper_trading.portefeuille_cible(engine), engine


def afficher_plan(cible, compte, ordres, remarques, exposition_compte) -> None:
    logger.info(
        "Portefeuille cible du moteur au %s : %d lignes, exposition %.1f %% (NAV moteur %s $).",
        cible.date.date(), sum(1 for ligne in cible.lignes.values() if ligne.poids > 0),
        100 * cible.exposition, f"{cible.nav_moteur:,.0f}")
    logger.info(
        "Compte %s : NAV %s $, cash %s $, %d lignes, exposition %.1f %%.",
        compte.compte, f"{compte.nav:,.0f}", f"{compte.cash:,.0f}", len(compte.detentions),
        100 * exposition_compte)
    for position in compte.hors_strategie:
        logger.info("Hors stratégie, jamais touchée : %s", position)
    if not ordres:
        logger.info("Aucun ordre : le compte est aligné sur le moteur.")
    else:
        tableau = paper_trading.tableau_ordres(ordres)[[
            "symbol", "sens", "quantite", "cours_reference", "montant", "raison",
            "poids_actuel_pct", "poids_cible_pct"]]
        logger.info("Plan (%d ordres) :\n%s", len(ordres), tableau.to_string(index=False))
        achats = [o for o in ordres if o.quantite > 0]
        ventes = [o for o in ordres if o.quantite < 0]
        logger.info(
            "%d achats pour %s $, %d ventes pour %s $.",
            len(achats), f"{sum(o.montant for o in achats):,.0f}",
            len(ventes), f"{sum(o.montant for o in ventes if o.montant == o.montant):,.0f}")
    for remarque in remarques:
        logger.warning("%s", remarque)


def main() -> None:
    args = construire_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.strategy not in STRATEGY_REGISTRY:
        logger.error("Stratégie inconnue : %s. Disponibles : %s", args.strategy, sorted(STRATEGY_REGISTRY))
        sys.exit(1)
    if args.hors_ligne and args.transmettre:
        logger.error("--hors-ligne calcule un plan sans compte : rien ne peut être transmis.")
        sys.exit(1)

    horodatage = datetime.now().isoformat(timespec="seconds")
    cible, engine = rejouer_strategie(args)

    def cours_de(symbol):
        return engine.prices.close_at(symbol, cible.date)

    ib = None
    try:
        if args.hors_ligne:
            compte = paper_trading.compte_hors_ligne(args.capital or config.BACKTEST_INITIAL_CAPITAL)
        else:
            import ib_connect

            port = args.port or _port_par_defaut()
            # readonly=False : ce script est le SEUL du dépôt qui passe des ordres.
            ib = ib_connect.connect(port, args.client_id, readonly=False)
            nom = paper_trading.choisir_compte_paper(paper_trading.comptes_geres(ib), args.compte)
            compte = paper_trading.lire_compte(
                ib, nom, paper_trading.symboles_geres(engine), capital_fourni=args.capital is not None)
            if args.capital is not None:
                compte = paper_trading.appliquer_capital(compte, args.capital, cours_de)

        montant_minimal = engine._montant_minimal(compte.nav)  # le plancher du moteur, au NAV du compte
        ordres, remarques = paper_trading.planifier_ordres(
            cible, compte, montant_minimal, args.tolerance_pct, cours_de)
        exposition = paper_trading.valeur_detentions(compte, cours_de) / compte.nav
        afficher_plan(cible, compte, ordres, remarques, exposition)
        if args.sortie:
            paper_trading.tableau_ordres(ordres).to_csv(args.sortie, index=False, encoding="utf-8")

        if args.hors_ligne:
            logger.info("Hors ligne : rien n'est journalisé ni envoyé.")
            return

        resultats = None
        mode = "simulation"
        if args.transmettre and ordres:
            paper_trading.verifier_envoi(
                ordres, compte, cible.date, _maintenant(),
                args.max_data_age_days, args.max_order_pct)
            annules = paper_trading.annuler_nos_ordres_ouverts(ib)
            if annules:
                logger.info("Nos ordres encore ouverts, annulés et remplacés : %s", ", ".join(annules))
            resultats = paper_trading.transmettre(ib, ordres, compte.compte, args.type_ordre)
            mode = "transmis"
            for resultat in resultats:
                niveau = logging.INFO if resultat["statut"] in paper_trading.STATUTS_OUVERTS | {"Filled"} else logging.WARNING
                logger.log(niveau, "%s : %s %s", resultat["symbol"], resultat["statut"], resultat["message"])
        elif ordres:
            logger.info("Simulation : rien n'est envoyé. Relance avec --transmettre pour passer ces ordres.")

        paper_trading.journaliser(
            config.DIR_PAPER_TRADING, horodatage, mode, compte, cible, ordres, remarques, exposition,
            resultats, parametres={
                "strategy": args.strategy, "start_date": args.start_date,
                "type_ordre": args.type_ordre, "tolerance_pct": args.tolerance_pct,
                "capital": args.capital, "montant_minimal": montant_minimal,
            })
        logger.info("Journal : %s", config.DIR_PAPER_TRADING)
    except paper_trading.RefusEnvoi as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except (ConnectionError, OSError, TimeoutError) as exc:
        logger.error("IB Gateway injoignable : %s", exc)
        sys.exit(1)
    finally:
        if ib is not None:
            ib.disconnect()


if __name__ == "__main__":
    main()
