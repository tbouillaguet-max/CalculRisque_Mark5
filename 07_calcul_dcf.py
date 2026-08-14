"""
Calcule la valorisation DCF (Discounted Cash Flow) pour chaque entreprise à
partir des données financières (04) et des cours (03), et compare la valeur
intrinsèque par action au cours actuel : c'est la "valeur théorique" de
l'entreprise, et l'écart calculé ici (Écart_DCF_vs_Cours_%) est le filtre
utilisé par 08_recuperation_options.py pour décider quelles entreprises
méritent qu'on aille chercher leurs chaînes d'options (écart >= ±20% par
défaut, voir config.VALUATION_GAP_THRESHOLD_PCT).

Corrections par rapport à CalculDCFMark1 :
    - Le script lisait un fichier fixe "donnees_entreprises.xlsx" que
      personne dans le pipeline ne produisait. Construit maintenant lui-même
      son entrée en fusionnant 05 (financials) + 03 (cours) + 02 (secteur).
    - Colonnes attendues en entrée ("EBIT"/"CapEx"/"BFR"/"Dette net"/
      "Cash et Cash Equivalents"/"Nombre d'action"/"Taux d'imposition")
      étaient un mélange incohérent avec les noms produits par les autres
      scripts d'origine (ex: "SharesOutstanding" vs "Nombre d'action") :
      aucune valeur ne matchait jamais, tout tombait sur les valeurs par
      défaut. Utilise maintenant les colonnes canoniques
      (ebit, capex, working_capital, net_debt, cash, shares_outstanding,
      tax_rate) produites par 05.
    - On utilise l'exercice financier le plus récent disponible par
      entreprise, mis en correspondance avec le cours de la même année (pas
      systématiquement la dernière année de cours si les 10-K ont du retard).
    - NOUVEAU : build_input_table empile désormais l'annuel (04) et le TTM
      trimestriel (04b, si disponible) via 05_calcul_multiples.py::
      load_financials_with_periods/match_price_asof (réutilisées, pas
      dupliquées) -- le cours utilisé est le cours quotidien connu à
      filed_date (03b) plutôt que systématiquement la clôture annuelle.
      DCF_HISTORY_FILE porte désormais period_type ("FY"/"TTM") et
      fiscal_quarter, pour que 06b_calcul_valorisation_combinee.py compare
      les pairs au sein du même "millésime de publication".
    - CORRIGÉ : calculer_fcf ne réintégrait pas la D&A (charge non
      décaissée) et retranchait des intérêts qu'aucun appelant ne
      renseignait -- voir la docstring de calculer_fcf. Le FCF, la valeur
      terminale et la valeur par action étaient systématiquement
      sous-estimés. Ce correctif invalide dcf_historique.parquet ET
      valorisation_combinee_historique.parquet : relance 07 puis 06b.

Usage :
    python 07_calcul_dcf.py
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import config

# Réutilise le chargement combiné FY+TTM et l'association de cours au plus
# proche de filed_date déjà écrits pour 05_calcul_multiples.py, plutôt que de
# les dupliquer (même raisonnement que 04b réutilisant compute_derived de 04 :
# une seule source de vérité pour cette logique).
_module_05 = importlib.import_module("05_calcul_multiples")
load_financials_with_periods = _module_05.load_financials_with_periods
match_price_asof = _module_05.match_price_asof

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HYPOTHESES_DEFAUT = {
    "periode_prevision": 5,
    "taux_croissance_fcf": 0.05,
    "taux_croissance_terminal": 0.02,
    "taux_actualisation": 0.10,
    "taux_imposition_defaut": 0.21,  # taux fédéral US ; utilisé seulement si tax_rate manquant
}


def calculer_fcf(ebit: float, taux_imposition: float, capex: float, variation_bfr: float, da: float = 0.0) -> float:
    """FCFF = EBIT x (1 - t) + D&A - CapEx - ΔBFR.

    ANCIEN COMPORTEMENT (corrigé) : "EBIT x (1-t) - CapEx - ΔBFR - Intérêts x
    (1-t)", qui souffrait de deux erreurs cumulées.

    1. La D&A n'était pas réintégrée. C'est une charge NON DÉCAISSÉE déjà
       déduite de l'EBIT : sans elle, le FCF est sous-estimé d'un montant
       proche de la D&A elle-même (30 à 60% du FCF pour une industrielle), et
       la sous-estimation se propage à la valeur terminale puis à la valeur
       par action. Effet observable : quasiment toutes les entreprises
       ressortaient survalorisées, la stratégie actions ne trouvait plus de
       candidat au-dessus de son seuil, et la stratégie options basculait
       structurellement du côté PUT.

    2. Le paramètre `interets` est supprimé plutôt que branché. Il n'était
       renseigné par AUCUN appelant (il valait donc toujours 0 : un paramètre
       mort, et donc un piège), mais surtout le retrancher aurait été faux
       ici : dans un FCFF, le coût de la dette est porté par le WACC utilisé
       à l'actualisation (config.SECTOR_DCF_PARAMS), pas par le flux. Le
       soustraire aussi du flux le compterait deux fois."""
    return (ebit * (1 - taux_imposition)) + da - capex - variation_bfr


def calculer_fcf_futurs(fcf_actuel: float, taux_croissance: float, periode: int) -> np.ndarray:
    return fcf_actuel * (1 + taux_croissance) ** np.arange(1, periode + 1)


def calculer_terminal_value(fcf_final: float, taux_croissance_terminal: float, taux_actualisation: float) -> float:
    if taux_actualisation <= taux_croissance_terminal:
        raise ValueError("Le taux d'actualisation doit être > au taux de croissance terminal.")
    return fcf_final * (1 + taux_croissance_terminal) / (taux_actualisation - taux_croissance_terminal)


def calculer_dcf(
    fcf_actuel: float, taux_croissance_fcf: float, taux_croissance_terminal: float,
    taux_actualisation: float, periode_prevision: int, dette_nette: float = 0,
) -> Tuple[float, Dict]:
    fcf_futurs = calculer_fcf_futurs(fcf_actuel, taux_croissance_fcf, periode_prevision)
    tv = calculer_terminal_value(fcf_futurs[-1], taux_croissance_terminal, taux_actualisation)
    valeur_actualisee_fcf = np.sum(fcf_futurs / (1 + taux_actualisation) ** np.arange(1, periode_prevision + 1))
    valeur_actualisee_tv = tv / (1 + taux_actualisation) ** periode_prevision
    ev = valeur_actualisee_fcf + valeur_actualisee_tv
    # dette_nette (04_recuperation_10k.py) est DÉJÀ nette de cash (dette
    # brute - cash) : "ev - dette_nette + cash" comptait le cash deux fois
    # (bug corrigé -- la valeur DCF était surestimée pour les entreprises
    # avec beaucoup de trésorerie nette).
    equity_value = ev - dette_nette

    details = {
        "FCF_futurs": fcf_futurs.tolist(), "Terminal_Value": tv,
        "Valeur_actualisee_FCF": valeur_actualisee_fcf, "Valeur_actualisee_TV": valeur_actualisee_tv,
        "Enterprise_Value": ev, "Equity_Value": equity_value,
    }
    return equity_value, details


def build_input_table(latest_only: bool = True) -> pd.DataFrame:
    """Empile financials annuel (04, period_type="FY") + TTM trimestriel
    (04b, period_type="TTM", si config.FINANCIALS_TTM_FILE existe -- sinon
    annuel seul, comportement identique à avant) via
    05_calcul_multiples.py::load_financials_with_periods, associe le cours
    (quotidien au plus proche de filed_date si disponible, sinon annuel --
    match_price_asof) et le secteur (02), calcule la variation du BFR (ΔBFR :
    BFR de la période N moins BFR de la période N-1 CHRONOLOGIQUEMENT
    précédente -- pas seulement "année N-1", une ligne TTM et une ligne FY
    peuvent partager la même 'year').

    latest_only=True (comportement historique, utilisé par le rapport Excel
    de main()) : ne garde que la période la plus récente disponible par
    entreprise (FY ou TTM, la plus récente des deux).
    latest_only=False (utilisé par 09_backtest.py via DCF_HISTORY_FILE) :
    garde TOUTES les périodes, avec leur 'filed_date', pour permettre de
    rejouer l'écart de valorisation tel qu'il était connu à chaque date
    passée plutôt que seulement à la date de la dernière période disponible
    aujourd'hui."""
    financials = load_financials_with_periods()
    financials = match_price_asof(financials)
    # BIAIS CONNU (non corrigé, cf. README) : le secteur vient de
    # config.UNIVERSE_FILE, soit la classification GICS d'AUJOURD'HUI,
    # appliquée rétroactivement à des exercices de 2012. Il pilote le WACC et
    # les deux taux de croissance du DCF (hypotheses_pour_secteur), donc une
    # entreprise reclassée depuis est valorisée sur toute sa partie ancienne
    # avec les hypothèses du mauvais secteur. L'historique GICS point-in-time
    # n'est pas disponible gratuitement : biais assumé, pas un oubli.
    universe = pd.read_csv(config.UNIVERSE_FILE, encoding="utf-8-sig")

    # 'working_capital' (produit par 04/04b) est un NIVEAU (current_assets -
    # current_liabilities) pour une période donnée, pas une variation. On
    # calcule ici la vraie variation ΔBFR, en se basant sur l'historique
    # multi-périodes déjà présent -- AVANT de filtrer sur la période la plus
    # récente uniquement, sinon la période précédente nécessaire au calcul du
    # delta serait déjà perdue. Tri par filed_date (ordre RÉEL de
    # publication), pas par 'year' seul : year=fiscal_year est partagé par
    # une ligne FY et plusieurs lignes TTM de la même année civile.
    financials = financials.sort_values(["symbol", "filed_date"])
    financials["working_capital_change"] = financials.groupby("symbol")["working_capital"].diff()

    df = financials.dropna(subset=["close"]).copy()
    if "sector" in universe.columns:
        universe = universe.copy()
        universe["symbol"] = universe["RIC"].apply(config.to_ib_symbol)
        df = df.merge(universe[["symbol", "sector"]], on="symbol", how="left")

    df = df.sort_values(["symbol", "filed_date"])
    if latest_only:
        df = df.groupby("symbol", as_index=False).tail(1)
    return df.reset_index(drop=True)


def hypotheses_pour_secteur(sector, hypotheses: Dict = HYPOTHESES_DEFAUT, year=None) -> Dict:
    """Hypothèses DCF de `hypotheses`, dont le WACC et les deux taux de
    croissance sont remplacés par ceux du secteur (config.sector_dcf_params).

    Un taux d'actualisation unique pour tout l'univers est un biais
    systématique : à 10%, une utility régulée (WACC réel ~6,5%) ressort
    mécaniquement sous-évaluée et une techno surévaluée, quelle que soit sa
    situation réelle.

    `year` : exercice valorisé. Le WACC y est indexé sur la courbe de taux de
    l'année (config.DCF_WACC_FOLLOWS_RATE_CURVE) -- un WACC figé de 2010 à
    2026 est un pari de taux non voulu, et systématiquement à contretemps :
    trop haut quand les taux étaient à zéro (valeur sous-estimée, excès de
    PUT), trop bas quand ils étaient à 5% (excès de CALL). year=None
    reproduit exactement le comportement d'avant."""
    params = config.sector_dcf_params(sector, year)
    return {
        **hypotheses,
        "taux_actualisation": params["wacc"],
        "taux_croissance_fcf": params["fcf_growth"],
        "taux_croissance_terminal": params["terminal_growth"],
    }


def calculer_dcf_par_entreprise(df: pd.DataFrame, hypotheses: Dict = HYPOTHESES_DEFAUT) -> pd.DataFrame:
    results = []
    secteurs_inconnus = set()
    # Entreprises dont AUCUNE période n'a de D&A tagué : signalées une fois en
    # fin de run plutôt qu'une fois par ligne (une entreprise a jusqu'à ~40
    # périodes FY+TTM dans l'historique).
    symboles_sans_da = set()
    # Lignes écartées parce que leur secteur ne se valorise pas par un FCFF
    # (cf. config.SECTORS_SANS_DCF) : comptées par secteur pour que le volume
    # écarté soit visible, et non deviné à la baisse du nombre de lignes.
    secteurs_sans_dcf = set(getattr(config, "SECTORS_SANS_DCF", ()) or ())
    ecartees_par_secteur: Dict[str, int] = {}
    # Raisons d'exclusion comptées : le script n'émettait qu'un avertissement
    # PAR LIGNE, noyé dans des milliers d'autres. Savoir que 3 000 lignes
    # tombent sur "EBIT manquant" et 40 sur "actions manquantes" ne se lisait
    # nulle part, alors que c'est ce qui dit où porter l'effort.
    raisons: Dict[str, int] = {}

    def ecarter(raison: str) -> None:
        raisons[raison] = raisons.get(raison, 0) + 1

    for _, row in df.iterrows():
        symbol = row.get("symbol", "?")
        try:
            secteur_ligne = row.get("sector")
            if isinstance(secteur_ligne, str) and secteur_ligne in secteurs_sans_dcf:
                # Un DCF FCFF sur une banque ou une foncière n'a pas de sens :
                # l'EBIT n'y est pas une mesure opérationnelle pertinente et la
                # dette est un intrant du métier, pas un financement à
                # retrancher. Ces entreprises restent valorisées par 06b via
                # les multiples sectoriels de config.SECTOR_MULTIPLES.
                ecartees_par_secteur[secteur_ligne] = ecartees_par_secteur.get(secteur_ligne, 0) + 1
                ecarter("secteur sans DCF (FCFF non pertinent)")
                continue

            ebit = row.get("ebit")
            if pd.isna(ebit) or ebit is None or ebit <= 0:
                ecarter("EBIT manquant" if pd.isna(ebit) or ebit is None else "EBIT negatif ou nul")
                logger.debug("EBIT manquant/invalide pour %s. DCF non calculé.", symbol)
                continue

            capex = row.get("capex") or 0
            # ΔBFR réel (variation N vs N-1), pas le niveau du BFR (cf.
            # erreur 1.3 du rapport). Si l'historique ne contient qu'un seul
            # exercice pour cette entreprise, la variation est inconnue :
            # on la traite comme 0 tout en le signalant, plutôt que de
            # soustraire par erreur le niveau absolu du BFR au FCF.
            bfr = row.get("working_capital_change")
            if pd.isna(bfr):
                logger.warning(
                    "Variation de BFR indisponible pour %s (un seul exercice "
                    "dans l'historique) : traitée comme 0 dans le calcul du FCF.",
                    symbol,
                )
                bfr = 0
            # D&A (colonne produite par 04_recuperation_10k.py, XBRL_TAGS["da"]) :
            # réintégrée au FCF comme charge non décaissée. Absente pour les
            # entreprises qui ne taguent ni DepreciationDepletionAndAmortization
            # ni DepreciationAmortizationAndAccretionNet -- traitée comme 0 en le
            # SIGNALANT : le DCF est alors CONSERVATEUR (FCF sous-estimé du
            # montant de la D&A réelle, donc valeur par action sous-estimée et
            # écart de valorisation biaisé du côté "survalorisée").
            da = row.get("da")
            if da is None or pd.isna(da):
                symboles_sans_da.add(symbol)
                da = 0.0
            dette_nette = row.get("net_debt") or 0
            shares_outstanding = row.get("shares_outstanding")
            taux_imposition = row.get("tax_rate")
            if pd.isna(taux_imposition) or taux_imposition is None or not (0 <= taux_imposition < 1):
                taux_imposition = hypotheses["taux_imposition_defaut"]

            if pd.isna(shares_outstanding) or not shares_outstanding or shares_outstanding <= 0:
                ecarter("nombre d'actions manquant")
                logger.debug("Nombre d'actions manquant pour %s. DCF non calculé.", symbol)
                continue

            fcf_actuel = calculer_fcf(
                ebit=ebit, taux_imposition=taux_imposition, capex=capex, variation_bfr=bfr, da=da,
            )
            if fcf_actuel <= 0:
                ecarter("FCF <= 0")
                logger.debug("FCF actuel <= 0 pour %s. DCF non calculé.", symbol)
                continue

            secteur = row.get("sector")
            if isinstance(secteur, str) and secteur not in config.SECTOR_DCF_PARAMS:
                secteurs_inconnus.add(secteur)
            # WACC indexé sur la courbe de taux de l'exercice valorisé.
            # L'année de DÉPÔT (filed_date) plutôt que l'exercice comptable :
            # c'est au moment où l'information devient publique que le marché
            # actualise, et c'est cette date qui pilote tout le reste du
            # pipeline. Repli sur l'exercice si filed_date manque.
            filed = pd.to_datetime(row.get("filed_date"), errors="coerce")
            annee_actualisation = filed.year if pd.notna(filed) else row.get("year")
            hyp = hypotheses_pour_secteur(
                secteur, hypotheses,
                year=int(annee_actualisation) if pd.notna(annee_actualisation) else None,
            )

            equity_value, details = calculer_dcf(
                fcf_actuel=fcf_actuel,
                taux_croissance_fcf=hyp["taux_croissance_fcf"],
                taux_croissance_terminal=hyp["taux_croissance_terminal"],
                taux_actualisation=hyp["taux_actualisation"],
                periode_prevision=hyp["periode_prevision"],
                dette_nette=dette_nette,
            )

            valeur_par_action = equity_value / shares_outstanding
            cours_actuel = row.get("close")
            ecart_pct = (
                (valeur_par_action - cours_actuel) / cours_actuel * 100
                if pd.notna(cours_actuel) and cours_actuel else None
            )

            results.append({
                "Ticker": symbol, "Secteur": row.get("sector"), "Année": row.get("year"),
                "cik": row.get("cik"), "WACC": hyp["taux_actualisation"],
                "period_type": row.get("period_type"), "fiscal_year": row.get("fiscal_year"),
                "fiscal_quarter": row.get("fiscal_quarter"), "filed_date": row.get("filed_date"),
                "FCF_actuel": fcf_actuel, "Enterprise_Value": details["Enterprise_Value"],
                "Equity_Value": equity_value, "Valeur_par_action_DCF": valeur_par_action,
                "Cours_actuel": cours_actuel, "Écart_DCF_vs_Cours_%": ecart_pct,
                **{k: v for k, v in details.items() if k != "FCF_futurs"},
            })
        except Exception as e:  # noqa: BLE001
            ecarter(f"erreur de calcul ({type(e).__name__})")
            logger.error("Erreur pour %s: %s", symbol, e)
            continue

    if raisons:
        total = sum(raisons.values())
        logger.info(
            "--- %d/%d lignes écartées du DCF (%d calculées) ---",
            total, total + len(results), len(results),
        )
        for raison, compte in sorted(raisons.items(), key=lambda kv: kv[1], reverse=True):
            logger.info("  %-42s %6d (%4.1f%%)", raison, compte, compte / (total + len(results)) * 100)
        if raisons.get("EBIT manquant"):
            logger.warning(
                "%d lignes sans EBIT : ces entreprises ne taguent ni OperatingIncomeLoss ni "
                "aucun des postes permettant de le reconstruire (résultat avant impôt, marge "
                "brute, coûts totaux -- cf. 04_recuperation_10k.py::_reconstruct_ebit). "
                "Vérifie le résumé de couverture affiché en fin de run de 04 ; si le taux y est "
                "bas, c'est l'extraction qu'il faut compléter, pas ce script.",
                raisons["EBIT manquant"],
            )

    if ecartees_par_secteur:
        logger.info(
            "%d ligne(s) écartées du DCF, secteur non valorisable par un FCFF "
            "(config.SECTORS_SANS_DCF) : %s. Ces entreprises restent valorisées par "
            "06b_calcul_valorisation_combinee.py via les multiples sectoriels.",
            sum(ecartees_par_secteur.values()),
            ", ".join(f"{s} ({n})" for s, n in sorted(ecartees_par_secteur.items())),
        )

    if symboles_sans_da:
        apercu = ", ".join(sorted(symboles_sans_da)[:15])
        logger.warning(
            "D&A absente pour %d entreprise(s) (aucun tag XBRL DepreciationDepletionAndAmortization "
            "ni DepreciationAmortizationAndAccretionNet) : traitée comme 0, le DCF de ces "
            "entreprises est donc CONSERVATEUR (FCF, valeur terminale et valeur par action "
            "sous-estimés d'autant, écart de valorisation biaisé vers 'survalorisée'). %s%s",
            len(symboles_sans_da), apercu, "..." if len(symboles_sans_da) > 15 else "",
        )

    if secteurs_inconnus:
        logger.warning(
            "Secteurs absents de config.SECTOR_DCF_PARAMS (hypothèses par défaut "
            "appliquées, WACC %.1f%%) : %s. Ajoute-les à la table si le paramétrage "
            "sectoriel doit s'y appliquer.",
            config.SECTOR_DCF_PARAMS["_default"]["wacc"] * 100, ", ".join(sorted(secteurs_inconnus)),
        )
    return pd.DataFrame(results)


def main() -> None:
    if not (config.FINANCIALS_FILE.exists() and config.PRICES_FILE.exists()):
        logger.error("Fichiers manquants. Lance d'abord 03_recuperation_cours.py et 04_recuperation_10k.py.")
        return

    # Calculé UNE SEULE FOIS sur tout l'historique (latest_only=False), puis
    # dérivé en deux sorties : le rapport Excel (dernier exercice seulement,
    # comportement historique inchangé) et l'historique complet pour le
    # backtest (DCF_HISTORY_FILE), sans dupliquer le calcul DCF.
    df_input_full = build_input_table(latest_only=False)
    df_dcf_full = calculer_dcf_par_entreprise(df_input_full, HYPOTHESES_DEFAUT)

    if df_dcf_full.empty:
        logger.error("Aucun résultat DCF calculé. Vérifiez les données d'entrée.")
        return

    history = df_dcf_full.rename(columns={
        "Ticker": "symbol", "Secteur": "sector", "Année": "year",
        "Valeur_par_action_DCF": "valuation_dcf_per_share", "Cours_actuel": "close",
        "Écart_DCF_vs_Cours_%": "gap_pct",
    })[[
        "symbol", "cik", "sector", "period_type", "year", "fiscal_year", "fiscal_quarter",
        "filed_date", "close", "valuation_dcf_per_share", "gap_pct",
    ]]
    config.DCF_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(config.DCF_HISTORY_FILE, index=False, engine="pyarrow")
    logger.info(
        "Historique DCF (tous exercices, pour le backtest) sauvegardé : %s (%d lignes, %d entreprises).",
        config.DCF_HISTORY_FILE, len(history), history["symbol"].nunique(),
    )

    # Tri par filed_date (ordre RÉEL de publication), pas par année civile
    # seule : une ligne FY et une ou plusieurs lignes TTM (04b) peuvent
    # partager la même 'year'/'Année' -- seul filed_date départage laquelle
    # est effectivement la plus récente.
    df_dcf = df_dcf_full.sort_values(["Ticker", "filed_date"]).groupby("Ticker", as_index=False).tail(1)
    df_input = df_input_full.sort_values(["symbol", "filed_date"]).groupby("symbol", as_index=False).tail(1)

    config.DCF_FILE.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(config.DCF_FILE, engine="openpyxl") as writer:
        df_dcf.to_excel(writer, sheet_name="DCF", index=False)
        df_input.to_excel(writer, sheet_name="Données d'entrée", index=False)
    logger.info("Résultats DCF sauvegardés : %s", config.DCF_FILE)

    logger.info("Résumé des résultats DCF:")
    for _, row in df_dcf.iterrows():
        cours = row["Cours_actuel"]
        cours_str = f"{cours:.2f}$" if pd.notna(cours) else "n/a"
        ecart = row["Écart_DCF_vs_Cours_%"]
        ecart_str = f"{ecart:.1f}%" if pd.notna(ecart) else "n/a"
        logger.info(
            "%s (%s): Valeur DCF/action = %.2f$, Cours = %s, Écart = %s",
            row["Ticker"], row["Secteur"], row["Valeur_par_action_DCF"], cours_str, ecart_str,
        )


if __name__ == "__main__":
    main()