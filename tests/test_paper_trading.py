"""Paper trading : le compte réplique le portefeuille du moteur, et rien ne part par erreur.

Trois familles de propriétés, dans l'ordre de ce qui coûterait le plus cher :

1. LES GARDE-FOUS. Un compte réel est refusé ; rien n'est envoyé sans
   --transmettre ; des données périmées ou un ordre démesuré bloquent l'envoi ;
   seuls NOS ordres ouverts sont annulés ; jamais de levier.
2. LA FIDÉLITÉ AU MOTEUR. Les cibles lues sont exactement ce que le moteur
   exécute à l'ouverture suivante -- vérifié en faisant tourner le moteur un
   jour de plus.
3. LES RÈGLES DE RÉCONCILIATION (paper_trading.planifier_ordres).

IB Gateway est remplacé par un faux qui expose la même API qu'ib_insync ; les
ordres, eux, sont de vrais objets ib_insync (MarketOrder, Stock).
"""

from __future__ import annotations

import importlib
import json
import math
import pathlib
import sys
from typing import NamedTuple

import pandas as pd
import pytest

import config
import paper_trading as pt
from backtest.data_loader import PricePanel
from backtest.engine import BacktestEngine
from backtest.strategies.base import Strategy

RACINE = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Un faux IB Gateway, avec la forme des objets d'ib_insync
# --------------------------------------------------------------------------- #
class _Valeur(NamedTuple):
    account: str
    tag: str
    value: str
    currency: str
    modelCode: str = ""


class _Contrat:
    def __init__(self, symbol, secType="STK", currency="USD"):
        self.symbol, self.secType, self.currency = symbol, secType, currency


class _Position(NamedTuple):
    account: str
    contract: _Contrat
    position: float
    avgCost: float = 0.0


class _Statut:
    def __init__(self, status):
        self.status = status


class _Trade:
    def __init__(self, contract, order, status="PreSubmitted"):
        self.contract, self.order, self.orderStatus, self.log = contract, order, _Statut(status), []


class _Ordre:
    def __init__(self, orderRef, action="BUY", totalQuantity=10):
        self.orderRef, self.action, self.totalQuantity = orderRef, action, totalQuantity


def _resume(compte="DU123", nav=1_000_000.0, cash=1_000_000.0, devise="USD", taux_usd=None):
    valeurs = [_Valeur(compte, "NetLiquidation", str(nav), devise),
               _Valeur(compte, "TotalCashValue", str(cash), devise)]
    if taux_usd is not None:
        valeurs.append(_Valeur(compte, "ExchangeRate", str(taux_usd), "USD"))
    return valeurs


class FauxIB:
    def __init__(self, comptes=("DU123",), resume=None, positions=(), ordres_ouverts=(), inconnus=()):
        self.comptes = list(comptes)
        self.resume = resume if resume is not None else _resume()
        self.positions = list(positions)
        self.ordres_ouverts = list(ordres_ouverts)
        self.inconnus = set(inconnus)
        self.places: list = []
        self.annules: list = []
        self.deconnecte = False

    def managedAccounts(self):
        return self.comptes

    def accountSummary(self, account=""):
        return [v for v in self.resume if not account or v.account == account]

    def reqPositions(self):
        return list(self.positions)

    def reqOpenOrders(self):
        return list(self.ordres_ouverts)

    def cancelOrder(self, order):
        self.annules.append(order)

    def qualifyContracts(self, *contrats):
        for i, contrat in enumerate(contrats):
            contrat.conId = 0 if contrat.symbol in self.inconnus else 1000 + i
        return [c for c in contrats if c.conId]

    def placeOrder(self, contract, order):
        trade = _Trade(contract, order)
        self.places.append(trade)
        return trade

    def sleep(self, secondes):
        pass

    def disconnect(self):
        self.deconnecte = True


# --------------------------------------------------------------------------- #
# Un moteur synthétique, qui a des ordres en attente à sa dernière clôture
# --------------------------------------------------------------------------- #
class _ToutAcheter(Strategy):
    def generate_target_weights(self, signals, current_positions):
        symbols = list(signals["symbol"])
        return {s: 1.0 / len(symbols) for s in symbols} if symbols else {}


def _evt(symbol, published, gap_pct=100.0):
    return {"symbol": symbol, "published_date": published, "fiscal_year": 2019,
            "sector": "Technologie", "close_at_filing": 100.0,
            "valuation_dcf_per_share": 200.0, "gap_pct": gap_pct, "period_type": None}


DATES = pd.bdate_range("2020-01-01", periods=20)


def _panel(ccc=None) -> PricePanel:
    close = pd.DataFrame({
        "AAA": [100.0 + i for i in range(20)],
        "BBB": [50.0 + 0.5 * i for i in range(20)],
        "CCC": ccc or [200.0 - i for i in range(20)],
        "DDD": [30.0] * 20,
    }, index=DATES)
    # L'OUVERTURE DE J VAUT LA CLÔTURE DE J-1 : le moteur exécute alors ses
    # ordres exactement aux cours qui ont servi à les décider, ce qui rend la
    # comparaison cible/exécution exacte.
    ouverture = close.shift(1).fillna(close)
    return PricePanel(close, ouverture, close.apply(lambda c: c.last_valid_index()))


def _moteur(fin: pd.Timestamp, stop_loss_pct=-99.0, evenements=None, panel=None) -> BacktestEngine:
    if evenements is None:
        evenements = pd.DataFrame([
            _evt("AAA", DATES[2]), _evt("BBB", DATES[2]), _evt("DDD", DATES[2]),
            _evt("CCC", DATES[18]),        # candidate neuve à la veille de la dernière séance
        ])
    return BacktestEngine(
        price_panel=panel or _panel(), signal_events=evenements, universe_history=None,
        fallback_universe_symbols={"AAA", "BBB", "CCC", "DDD"}, strategy=_ToutAcheter(),
        initial_capital=1_000_000.0, cost_bps=0.0, stop_loss_pct=stop_loss_pct,
        take_profit_pct=1e6, momentum_min_pct=None, rebalance_band_pct=0.0,
        trailing_stop_pct=None, max_holding_days=None, exit_gap_threshold_pct=None,
        impact_coefficient_bps=0.0, min_commission_dollar=0.0, min_trade_pct_of_nav=0.0,
        max_fee_pct_of_trade=0.0, vol_target_pct=None, end_date=fin,
    )


# --------------------------------------------------------------------------- #
# 2. Fidélité au moteur
# --------------------------------------------------------------------------- #
def test_les_cibles_sont_ce_que_le_moteur_execute_le_lendemain():
    """LA propriété dont tout dépend. On lit les cibles à la veille, puis on
    laisse le moteur jouer la séance suivante : ce qu'il détient alors doit
    être, ligne par ligne, la cible lue -- sinon le compte paper répliquerait
    autre chose que le backtest."""
    veille = _moteur(DATES[18])
    veille.run()
    assert veille.pending_orders, "le scénario doit laisser des ordres en attente"
    date_signal, nav, cibles = veille.cibles_ouverture_suivante()
    assert date_signal == DATES[18]

    lendemain = _moteur(DATES[19])
    lendemain.run()
    for symbol, (valeur, cours, _raison) in cibles.items():
        detenues = lendemain.positions[symbol].shares if symbol in lendemain.positions else 0.0
        assert detenues * cours == pytest.approx(valeur, rel=1e-9, abs=1e-6), symbol
    assert set(lendemain.positions) == {s for s, (v, _, _) in cibles.items() if v > 0}
    assert nav == pytest.approx(veille._current_nav(DATES[18]))


def test_une_liquidation_du_moteur_se_lit_comme_une_cible_nulle():
    """Un stop-loss déclenché à la clôture sort à l'ouverture suivante : la
    cible lue doit valoir zéro, avec la raison du moteur."""
    # CCC, achetée au début, décroche de 10 % à la DERNIÈRE clôture : le stop
    # à -5 % se déclenche ce soir-là et sort à l'ouverture suivante.
    moteur = _moteur(
        DATES[18], stop_loss_pct=-5.0,
        evenements=pd.DataFrame([_evt("AAA", DATES[2]), _evt("CCC", DATES[2])]),
        panel=_panel(ccc=[200.0] * 18 + [180.0, 180.0]))
    moteur.run()
    cible = pt.portefeuille_cible(moteur)
    liquidations = {s: l for s, l in cible.lignes.items() if l.poids == 0}
    assert liquidations and all(l.raison == "stop_loss" for l in liquidations.values())


# --------------------------------------------------------------------------- #
# 3. Règles de réconciliation
# --------------------------------------------------------------------------- #
def _cible(**lignes) -> pt.PortefeuilleCible:
    """_cible(AAA=(poids, cours, raison), ...)"""
    return pt.PortefeuilleCible(pd.Timestamp("2026-09-24"), 5_000_000.0, {
        s: pt.LigneCible(s, *valeurs) for s, valeurs in lignes.items()})


def _compte(nav=100_000.0, cash=None, **detentions) -> pt.EtatCompte:
    return pt.EtatCompte("DU123", nav, nav if cash is None else cash, dict(detentions))


def _plan(cible, compte, minimum=50.0, tolerance=1.0, cours=None):
    return pt.planifier_ordres(cible, compte, minimum, tolerance, (cours or {}).get)


def test_premier_run_achete_chaque_ligne_en_actions_entieres():
    ordres, _ = _plan(_cible(AAA=(0.10, 99.0, ""), BBB=(0.05, 30.0, "rebalance")), _compte())
    assert {(o.symbol, o.quantite) for o in ordres} == {("AAA", 101), ("BBB", 166)}
    assert {o.raison for o in ordres} == {pt.RAISON_ACHAT_MANQUANT, "rebalance"}


def test_une_liquidation_du_moteur_passe_meme_sous_le_plancher():
    """Même règle que le moteur : un plancher emprisonnerait la ligne."""
    ordres, _ = _plan(_cible(AAA=(0.0, 10.0, "stop_loss")), _compte(AAA=2), minimum=1_000.0)
    assert [(o.symbol, o.quantite, o.raison) for o in ordres] == [("AAA", -2, "stop_loss")]


def test_une_ligne_que_le_moteur_ne_detient_pas_est_liquidee_meme_sans_cours():
    ordres, _ = _plan(_cible(), _compte(ZZZ=7))
    assert [(o.symbol, o.quantite, o.raison) for o in ordres] == [("ZZZ", -7, pt.RAISON_HORS_MOTEUR)]
    assert math.isnan(ordres[0].montant)


def test_la_derive_sous_la_tolerance_n_est_pas_corrigee():
    """Règle 3 : ligne détenue des deux côtés, sans ordre du moteur. 10 % visés,
    10,5 % détenus : 0,5 point d'écart, sous la tolérance de 1 point."""
    cible = _cible(AAA=(0.10, 100.0, ""))
    assert _plan(cible, _compte(AAA=105))[0] == []
    ordres, _ = _plan(cible, _compte(AAA=115))           # 1,5 point : corrigé
    assert [(o.quantite, o.raison) for o in ordres] == [(-15, pt.RAISON_DERIVE)]


def test_un_ordre_du_moteur_est_suivi_meme_sous_la_tolerance():
    """Règle 1 : la tolérance ne filtre que la dérive, pas une décision."""
    ordres, _ = _plan(_cible(AAA=(0.10, 100.0, "rebalance")), _compte(AAA=105))
    assert [(o.quantite, o.raison) for o in ordres] == [(-5, "rebalance")]


def test_un_ordre_du_moteur_sous_le_plancher_n_est_pas_passe_et_le_plan_le_dit():
    ordres, remarques = _plan(_cible(AAA=(0.10, 100.0, "rebalance")), _compte(AAA=99), minimum=500.0)
    assert ordres == [] and any("sous le plancher" in r for r in remarques)


def test_une_cible_sous_le_prix_d_une_action_est_signalee():
    ordres, remarques = _plan(_cible(NVR=(0.004, 6_300.0, "")), _compte())
    assert ordres == [] and any("inférieure au prix d'une action" in r for r in remarques)


def test_une_fraction_d_action_hors_moteur_ne_donne_pas_d_ordre_de_zero():
    assert _plan(_cible(), _compte(ZZZ=0.4))[0] == []


def test_jamais_de_levier_les_achats_sont_ramenes_au_cash():
    """40 000 $ de cash, 10 000 $ de ventes, 90 000 $ d'achats demandés : les
    achats sont réduits dans la même proportion, les ventes restent entières."""
    cible = _cible(AAA=(0.45, 100.0, "rebalance"), BBB=(0.45, 100.0, "rebalance"),
                   CCC=(0.0, 100.0, "stop_loss"))
    ordres, remarques = _plan(cible, _compte(cash=40_000.0, CCC=100))
    ventes = [o for o in ordres if o.quantite < 0]
    achats = [o for o in ordres if o.quantite > 0]
    assert [(o.symbol, o.quantite) for o in ventes] == [("CCC", -100)]
    assert sum(o.montant for o in achats) <= (40_000.0 + 10_000.0) * (1 - pt.MARGE_CASH)
    assert achats[0].quantite == achats[1].quantite > 0
    assert any("Cash insuffisant" in r for r in remarques)


def test_les_ventes_passent_avant_les_achats():
    cible = _cible(AAA=(0.2, 100.0, "rebalance"), ZZZ=(0.0, 100.0, "stop_loss"))
    ordres, _ = _plan(cible, _compte(ZZZ=10))
    assert [o.sens for o in ordres] == ["SELL", "BUY"]


# --------------------------------------------------------------------------- #
# 1. Garde-fous
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("comptes,demande", [(["U1234567"], None), (["DU1", "U2"], "U2")])
def test_un_compte_reel_est_refuse(comptes, demande):
    with pytest.raises(pt.RefusEnvoi, match="PAS un compte paper"):
        pt.choisir_compte_paper(comptes, demande)


def test_le_compte_paper_est_retenu_et_l_ambiguite_refusee():
    assert pt.choisir_compte_paper(["DU1234567"]) == "DU1234567"
    with pytest.raises(pt.RefusEnvoi, match="précise --compte"):
        pt.choisir_compte_paper(["DU1", "DU2"])
    with pytest.raises(pt.RefusEnvoi, match="inconnu"):
        pt.choisir_compte_paper(["DU1"], "DU9")


def test_des_donnees_perimees_bloquent_l_envoi():
    ordres, _ = _plan(_cible(AAA=(0.1, 100.0, "")), _compte())
    aujourd_hui = pd.Timestamp("2026-09-24")
    pt.verifier_envoi(ordres, _compte(), pd.Timestamp("2026-09-21"), aujourd_hui, 4, 25.0)
    with pytest.raises(pt.RefusEnvoi, match="il y a 5 jours"):
        pt.verifier_envoi(ordres, _compte(), pd.Timestamp("2026-09-19"), aujourd_hui, 4, 25.0)


def test_un_ordre_demesure_bloque_l_envoi():
    """Le signe d'une erreur d'échelle (NAV lu dans la mauvaise devise...)."""
    ordres, _ = _plan(_cible(AAA=(0.4, 100.0, "rebalance")), _compte())
    with pytest.raises(pt.RefusEnvoi, match="au-delà de 25 % du NAV"):
        pt.verifier_envoi(ordres, _compte(), pd.Timestamp("2026-09-24"), pd.Timestamp("2026-09-24"), 4, 25.0)


def test_seuls_nos_ordres_ouverts_sont_annules():
    a_nous = _Trade(_Contrat("AAA"), _Ordre(config.PAPER_TRADING_ORDER_REF))
    deja_execute = _Trade(_Contrat("BBB"), _Ordre(config.PAPER_TRADING_ORDER_REF), status="Filled")
    manuel = _Trade(_Contrat("CCC"), _Ordre(""))
    ib = FauxIB(ordres_ouverts=[a_nous, deja_execute, manuel])
    assert pt.annuler_nos_ordres_ouverts(ib) == ["AAA BUY 10"]
    assert ib.annules == [a_nous.order]


def test_le_capital_alloue_ne_peut_pas_depasser_le_nav():
    compte = _compte(nav=100_000.0, AAA=100)
    with pytest.raises(pt.RefusEnvoi, match="levier"):
        pt.appliquer_capital(compte, 150_000.0, {"AAA": 100.0}.get)
    poche = pt.appliquer_capital(compte, 50_000.0, {"AAA": 100.0}.get)
    assert (poche.nav, poche.cash) == (50_000.0, 40_000.0)


# --------------------------------------------------------------------------- #
# Lecture du compte et envoi, contre le faux Gateway
# --------------------------------------------------------------------------- #
def test_le_compte_ne_garde_que_les_actions_us_du_pipeline():
    ib = FauxIB(positions=[
        _Position("DU123", _Contrat("AAA"), 10),
        _Position("DU123", _Contrat("AAA", secType="OPT"), 2),       # option : hors stratégie
        _Position("DU123", _Contrat("MC", currency="EUR"), 5),        # autre devise
        _Position("DU123", _Contrat("XYZ"), 3),                       # inconnue du pipeline
        _Position("DU999", _Contrat("BBB"), 7),                       # autre compte
    ])
    compte = pt.lire_compte(ib, "DU123", {"AAA", "BBB"})
    assert compte.detentions == {"AAA": 10.0}
    assert len(compte.hors_strategie) == 3
    assert (compte.nav, compte.cash) == (1_000_000.0, 1_000_000.0)


def test_un_compte_en_euros_est_converti_en_dollars():
    """1 USD = 0,8 EUR : 800 000 EUR de NAV font 1 000 000 USD."""
    ib = FauxIB(resume=_resume(nav=800_000.0, cash=400_000.0, devise="EUR", taux_usd=0.8))
    compte = pt.lire_compte(ib, "DU123", set())
    assert compte.nav == pytest.approx(1_000_000.0)
    assert compte.cash == pytest.approx(500_000.0)
    assert compte.devise_base == "EUR"


def test_sans_taux_de_change_il_faut_un_capital():
    ib = FauxIB(resume=_resume(devise="EUR"))
    with pytest.raises(pt.RefusEnvoi, match="--capital"):
        pt.lire_compte(ib, "DU123", set())
    pt.lire_compte(ib, "DU123", set(), capital_fourni=True)


def test_les_ordres_sont_au_marche_a_l_ouverture_etiquetes_et_dans_l_ordre():
    ordres, _ = _plan(_cible(AAA=(0.1, 100.0, "rebalance"), ZZZ=(0.0, 50.0, "stop_loss")),
                      _compte(ZZZ=4))
    ib = FauxIB(inconnus={"AAA"})
    resultats = pt.transmettre(ib, ordres, "DU123", "moo")
    assert [(t.contract.symbol, t.order.action, t.order.totalQuantity) for t in ib.places] == [("ZZZ", "SELL", 4)]
    ordre = ib.places[0].order
    assert (ordre.orderType, ordre.tif, ordre.account, ordre.orderRef) == (
        "MKT", "OPG", "DU123", config.PAPER_TRADING_ORDER_REF)
    assert {r["symbol"]: r["statut"] for r in resultats} == {"ZZZ": "PreSubmitted", "AAA": "non_envoye"}


# --------------------------------------------------------------------------- #
# Le script, de bout en bout
# --------------------------------------------------------------------------- #
@pytest.fixture
def script(tmp_path, monkeypatch):
    """17_paper_trading.py branché sur le moteur synthétique et un faux IB."""
    module = importlib.import_module("17_paper_trading")
    moteur = _moteur(DATES[18])
    moteur.run()
    monkeypatch.setattr(module, "rejouer_strategie", lambda args: (pt.portefeuille_cible(moteur), moteur))
    monkeypatch.setattr(module, "_port_par_defaut", lambda: 4002)
    monkeypatch.setattr(module, "_maintenant", lambda: DATES[19])
    monkeypatch.setattr(config, "DIR_PAPER_TRADING", tmp_path / "paper")
    ib = FauxIB(resume=_resume(nav=100_000.0, cash=100_000.0))
    import ib_connect
    monkeypatch.setattr(ib_connect, "connect", lambda *a, **k: ib)

    def lancer(*arguments):
        monkeypatch.setattr(sys, "argv", ["17_paper_trading.py", *arguments])
        module.main()
    return lancer, ib, tmp_path / "paper"


def test_sans_transmettre_rien_ne_part_mais_tout_est_journalise(script):
    lancer, ib, journal = script
    lancer()
    assert ib.places == [] and ib.annules == []
    ordres = pd.read_csv(journal / "ordres.csv")
    assert len(ordres) > 0 and set(ordres["statut"]) == {"simulation"}
    assert set(ordres["compte"]) == {"DU123"}
    compte = pd.read_csv(journal / "compte.csv")
    assert list(compte["mode"]) == ["simulation"] and compte["nav_compte_usd"].iloc[0] == 100_000.0
    assert json.loads((journal / "dernier_run.json").read_text(encoding="utf-8"))["mode"] == "simulation"
    assert ib.deconnecte


def test_transmettre_envoie_le_plan_et_journalise_les_statuts(script):
    lancer, ib, journal = script
    lancer("--transmettre")
    assert ib.places and all(t.order.tif == "OPG" for t in ib.places)
    ordres = pd.read_csv(journal / "ordres.csv")
    assert set(ordres["statut"]) == {"PreSubmitted"}
    assert list(pd.read_csv(journal / "compte.csv")["mode"]) == ["transmis"]


def test_un_compte_reel_arrete_le_script_avant_tout_ordre(script):
    lancer, ib, journal = script
    ib.comptes = ["U1234567"]
    with pytest.raises(SystemExit) as sortie:
        lancer("--transmettre")
    assert sortie.value.code == 1
    assert ib.places == [] and ib.annules == [] and not journal.exists()


def test_des_donnees_perimees_arretent_l_envoi(script, monkeypatch):
    lancer, ib, journal = script
    module = importlib.import_module("17_paper_trading")
    monkeypatch.setattr(module, "_maintenant", lambda: DATES[18] + pd.Timedelta(days=10))
    with pytest.raises(SystemExit) as sortie:
        lancer("--transmettre")
    assert sortie.value.code == 1
    assert ib.places == [] and ib.annules == []


def test_hors_ligne_et_transmettre_sont_incompatibles(script):
    lancer, ib, _ = script
    with pytest.raises(SystemExit) as sortie:
        lancer("--hors-ligne", "--transmettre")
    assert sortie.value.code == 1 and ib.places == []


# --------------------------------------------------------------------------- #
# Une seule définition des réglages du moteur
# --------------------------------------------------------------------------- #
def test_les_reglages_du_moteur_ne_sont_declares_qu_a_un_endroit():
    """Le compte paper doit tester la configuration mesurée par 09. Si un
    script redéclarait ses propres options, les deux pourraient diverger sans
    que rien ne le signale."""
    declarants = sorted(
        str(p.relative_to(RACINE)) for p in RACINE.rglob("*.py")
        if "tests" not in p.parts and '"--vol-target-pct"' in p.read_text(encoding="utf-8", errors="ignore"))
    assert declarants == ["backtest/construction_moteur.py"]
    for script in ("09_backtest.py", "17_paper_trading.py"):
        assert "ajouter_options_moteur(parser)" in (RACINE / script).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Le run quotidien, sur demande
# --------------------------------------------------------------------------- #
def test_le_run_quotidien_ne_trade_que_sur_demande_et_au_bon_moment():
    """Absent par défaut ; avec --paper-trading, optionnel (un Gateway fermé ne
    doit pas coûter la valorisation du jour), après le signal et son filtre
    qualitatif, et AVANT 08, qui peut durer plus d'une heure."""
    import run_pipeline_daily as daily

    assert "17_paper_trading.py" not in [s.script for s in daily.daily_steps(7)]
    etapes = daily.daily_steps(7, paper_trading=True)
    noms = [s.script for s in etapes]
    paper = etapes[noms.index("17_paper_trading.py")]
    assert (paper.required, paper.needs_gateway, paper.extra_args) == (False, True, ("--transmettre",))
    assert noms.index("06b_calcul_valorisation_combinee.py") < noms.index("17_paper_trading.py")
    assert noms.index("07b_validation_qualitative.py") < noms.index("17_paper_trading.py")
    assert noms.index("17_paper_trading.py") < noms.index("08_recuperation_options.py")
    assert "17_paper_trading.py" in daily.PRICES_ONLY


def test_l_indice_de_reference_n_est_pas_un_titre_de_la_strategie():
    """Le panel de cours porte aussi SPY (indice de référence) : un SPY pris à
    la main sur le compte ne doit pas être vendu au motif que le moteur ne le
    détient pas. Seuls les titres dotés d'un signal sont gérés."""
    moteur = _moteur(DATES[18])
    moteur.run()
    geres = pt.symboles_geres(moteur)
    assert geres == {"AAA", "BBB", "CCC", "DDD"}
    ib = FauxIB(positions=[_Position("DU123", _Contrat("SPY"), 50), _Position("DU123", _Contrat("AAA"), 5)])
    compte = pt.lire_compte(ib, "DU123", geres)
    assert compte.detentions == {"AAA": 5.0} and "SPY" in compte.hors_strategie[0]
