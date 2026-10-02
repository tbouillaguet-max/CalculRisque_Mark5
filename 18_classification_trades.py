"""Qu'ont en commun les thèses PERDANTES d'une stratégie, et qu'ont en commun
les GAGNANTES ? Classification par apprentissage automatique sur les sorties
d'un run, sans le relancer.

Méthode détaillée en tête de backtest/classification_trades.py. En bref :

    - une ligne par THÈSE (symbole, date d'entrée), pas par vente ;
    - gagnante = alpha > 0 : rendement de la thèse moins celui de l'indice de
      référence sur les mêmes dates (--label pnl ou extremes sinon) ;
    - seules les variables connues à la DÉCISION entrent dans le modèle --
      signal, fondamentaux, prix, marché, 8-K, contexte du portefeuille ; ce
      qui se passe pendant la détention est décrit à part ;
    - apprentissage sur les thèses closes avant --date-coupure, test sur
      celles ouvertes après, plus un réapprentissage annuel (walk-forward).

Cinq sections :

    1. Variable par variable : ce qui distingue gagnantes et perdantes, et si
       l'écart tient sur la période de test.
    2. Profils : les règles d'un arbre peu profond, des plus perdantes aux
       plus gagnantes, avec leur tenue au test.
    3. Modèle complet : AUC hors échantillon contre l'écart de valorisation
       seul, et les variables dont il dépend.
    4. Année par année, et lecture ÉCONOMIQUE : les thèses que le modèle juge
       perdantes rapportent-elles vraiment moins ?
    5. Pendant la détention : ce que vivent les deux groupes (descriptif).

Usage :
    python 18_classification_trades.py                       # dernier run actions
    python 18_classification_trades.py --run-id ancre_ref
    python 18_classification_trades.py --run-dir data/backtest_options/<run>
    python 18_classification_trades.py --label extremes --extremes-pct 25

Les tables sont écrites dans <run>/classification/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import classification_trades as ct

logger = logging.getLogger("classification_trades")


def construire_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default=None, help="Sous-dossier de data/backtest (ou data/backtest_options).")
    parser.add_argument("--run-dir", type=Path, default=None, help="Chemin complet, prioritaire sur --run-id.")
    parser.add_argument("--label", choices=("alpha", "pnl", "extremes"), default="alpha",
                        help="Définition de la thèse gagnante (défaut: %(default)s).")
    parser.add_argument("--extremes-pct", type=float, default=30.0,
                        help="Avec --label extremes : part gardée de chaque côté, en %% (défaut: %(default)s).")
    parser.add_argument("--date-coupure", default=ct.DATE_COUPURE_DEFAUT,
                        help="Fin de l'apprentissage, début du test (défaut: %(default)s, celle de 16).")
    parser.add_argument("--premiere-annee-walk-forward", type=int, default=2018,
                        help="Première année prédite par le réapprentissage annuel (défaut: %(default)s).")
    parser.add_argument("--profondeur-arbre", type=int, default=3,
                        help="Profondeur de l'arbre des profils : 3 donne au plus 8 règles (défaut: %(default)s).")
    parser.add_argument("--top", type=int, default=15, help="Lignes affichées par tableau (défaut: %(default)s).")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Tirages des intervalles de confiance.")
    parser.add_argument("--graine", type=int, default=0)
    return parser


def _titre(texte: str) -> None:
    print(f"\n--- {texte} ---")


def main() -> None:
    args = construire_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

    try:
        dossier = ct.resoudre_run(args.run_id, args.run_dir)
        run = ct.charger_run(dossier)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Chargement des données du pipeline...")
    donnees = ct.charger_donnees(run.run_config.get("benchmark_symbol"))
    if donnees.indice is None and args.label != "pnl":
        logger.error("Aucun indice de référence : l'alpha n'est pas calculable. Relance avec --label pnl.")
        sys.exit(1)

    theses = ct.construire_theses(run.trades, run.positions_history)
    if donnees.indice is not None:
        theses = ct.ajouter_rendement_indice(theses, run.trades, donnees.indice)
    else:
        theses["rendement_indice_pct"] = np.nan
        theses["alpha_pct"] = np.nan
    theses = ct.etiqueter(theses, args.label, args.extremes_pct)
    df = ct.construire_variables(theses, run, donnees)

    coupure = pd.Timestamp(args.date_coupure)
    apprentissage, test = ct.decouper(df, coupure)
    numeriques, categorielles = ct.colonnes_modele(df)

    print(f"\n=== Classification des thèses de {dossier} ===")
    print(f"Stratégie : {run.run_config.get('strategy', '?')}  |  étiquette : {args.label}"
          f"  |  thèses closes : {len(df)}  |  variables à la décision : "
          f"{len(numeriques)} numériques, {len(categorielles)} catégorielles")
    print(f"Apprentissage : {int(apprentissage.sum())} thèses closes avant {coupure.date()} "
          f"({df.loc[apprentissage, 'gagnante'].mean() * 100:.1f} % gagnantes)  |  "
          f"test : {int(test.sum())} thèses ouvertes depuis "
          f"({df.loc[test, 'gagnante'].mean() * 100:.1f} % gagnantes)  |  "
          f"écartées (à cheval ou non étiquetées) : {len(df) - int(apprentissage.sum()) - int(test.sum())}")
    if apprentissage.sum() < 100 or test.sum() < 30:
        logger.error("Trop peu de thèses de part et d'autre de la coupure pour conclure quoi que ce soit.")
        sys.exit(1)

    sortie = dossier / "classification"
    sortie.mkdir(parents=True, exist_ok=True)

    # 1 ------------------------------------------------------------------- #
    _titre("1. Variable par variable (AUC de la variable seule : >0,5 = plus haute chez les gagnantes)")
    univ = ct.analyse_univariee(df, numeriques, apprentissage, test)
    univ.to_csv(sortie / "univarie.csv", index=False)
    predictives = univ[~univ["descriptive"]]
    colonnes = ["variable", "mediane_gagnantes", "mediane_perdantes", "auc_apprentissage",
                "auc_test", "q_valeur", "p_valeur_test", "stable", "couverture_pct"]
    print(predictives[colonnes].head(args.top).to_string(index=False))
    stables = predictives[predictives["stable"]]
    print(f"\n{len(stables)} variable(s) STABLE(S) sur {len(predictives)} : significatives après "
          "correction des tests multiples sur l'apprentissage, puis confirmées au test dans le même sens.")

    tableaux_quintiles = []
    for variable in stables["variable"].head(5):
        q = ct.quintiles(df, variable, apprentissage, test)
        if not q.empty:
            tableaux_quintiles.append(q)
            print(f"\n  {variable} par quintile (bornes de l'apprentissage) :")
            print(q[["quintile", "borne_basse", "borne_haute", "n_apprentissage",
                     "taux_gagnantes_apprentissage_pct", "alpha_moyen_apprentissage_pct", "n_test",
                     "taux_gagnantes_test_pct", "alpha_moyen_test_pct"]].to_string(index=False))
    if tableaux_quintiles:
        pd.concat(tableaux_quintiles).to_csv(sortie / "quintiles.csv", index=False)

    modalites = [ct.taux_par_modalite(df, c, apprentissage, test) for c in categorielles]
    modalites = [m for m in modalites if not m.empty]
    if modalites:
        pd.concat(modalites).to_csv(sortie / "modalites.csv", index=False)
        for m in modalites:
            print(f"\n  {m['variable'].iloc[0]} :")
            print(m.drop(columns="variable").head(args.top).to_string(index=False))

    # 2 ------------------------------------------------------------------- #
    _titre(f"2. Profils (arbre de profondeur {args.profondeur_arbre}, du plus perdant au plus gagnant)")
    profils = ct.regles(df, numeriques, categorielles, apprentissage, test,
                        args.profondeur_arbre, args.graine)
    profils.to_csv(sortie / "regles.csv", index=False)
    for _, ligne in profils.iterrows():
        print(f"\n  [{ligne['taux_gagnantes_apprentissage_pct']:5.1f} % gagnantes, alpha "
              f"{ligne['alpha_moyen_apprentissage_pct']:+.1f} %, n={ligne['n_apprentissage']}]"
              f"  -> test : {ligne['taux_gagnantes_test_pct']:5.1f} %, alpha "
              f"{ligne['alpha_moyen_test_pct']:+.1f} %, n={ligne['n_test']}")
        print(f"    {ligne['regle']}")
        if ligne["n_test"] == 0:
            print("    (aucune thèse du test dans cette feuille : la règle porte sur une variable qui "
                  "dérive avec le temps -- taille du portefeuille, régime -- et non sur le titre)")

    # 3 ------------------------------------------------------------------- #
    _titre("3. Modèle complet (gradient boosting), mesuré sur le test")
    res = ct.modele_principal(df, numeriques, categorielles, apprentissage, test,
                              args.graine, args.bootstrap)
    res.importance.to_csv(sortie / "importance.csv", index=False)
    print(f"AUC test du modèle        : {res.auc_test:.3f}  IC95 [{res.ic_test[0]:.3f}, {res.ic_test[1]:.3f}]")
    print(f"AUC test de l'écart seul  : {res.auc_reference_ecart:.3f}  "
          f"IC95 [{res.ic_reference[0]:.3f}, {res.ic_reference[1]:.3f}]")
    print(f"AUC test, logistique      : {res.auc_logistique:.3f}  "
          f"IC95 [{res.ic_logistique[0]:.3f}, {res.ic_logistique[1]:.3f}]")
    print("(0,5 = hasard. IC tirés par mois d'entrée : les thèses d'un même mois ne sont pas indépendantes.)")
    if not (res.ic_test[0] > 0.5):
        print(">>> L'intervalle du modèle contient 0,5 : hors échantillon, il ne distingue pas les "
              "gagnantes des perdantes mieux que le hasard. Les profils ci-dessus sont à lire comme "
              "des descriptions de la période d'apprentissage, pas comme des règles.")
    print("\nVariables dont le modèle dépend le plus (baisse d'AUC quand on les brouille, sur le test) :")
    print(res.importance.head(args.top).to_string(index=False))

    # 4 ------------------------------------------------------------------- #
    _titre("4. Réapprentissage annuel (walk-forward) et lecture économique")
    annuel, proba_oos = ct.walk_forward(df, numeriques, categorielles,
                                        args.premiere_annee_walk_forward, args.graine)
    annuel.to_csv(sortie / "walk_forward.csv", index=False)
    if annuel.empty:
        print("historique trop court pour un réapprentissage annuel.")
    else:
        print(annuel.to_string(index=False))
        etiq = df.loc[proba_oos.index, "gagnante"]
        print(f"\nAUC hors échantillon, toutes années confondues : {ct.auc(etiq, proba_oos):.3f} "
              f"sur {len(proba_oos)} thèses ; AUC > 0,5 sur "
              f"{int((annuel['auc'] > 0.5).sum())}/{len(annuel)} années.")
        eco = ct.lecture_economique(df, proba_oos)
        eco.to_csv(sortie / "lecture_economique.csv", index=False)
        print("\nThèses classées par probabilité hors échantillon d'être gagnantes (1 = jugées les plus perdantes) :")
        print(eco.to_string(index=False))
        if len(eco) >= 2:
            ecart = eco["alpha_moyen_pct"].iloc[-1] - eco["alpha_moyen_pct"].iloc[0]
            print(f"\nÉcart d'alpha moyen entre le quintile le plus « gagnant » et le plus « perdant » : "
                  f"{ecart:+.2f} points. C'est l'ordre de grandeur de ce qu'un filtre d'entrée pourrait "
                  "rapporter AVANT tout effet de second ordre (cash redistribué, rotation) : seul un "
                  "A/B apparié du backtest le tranchera.")

    # 5 ------------------------------------------------------------------- #
    _titre("5. Pendant la détention (descriptif, jamais dans le modèle)")
    descriptives = univ[univ["descriptive"]]
    if not descriptives.empty:
        print(descriptives[["variable", "mediane_gagnantes", "mediane_perdantes", "auc_apprentissage",
                            "auc_test"]].to_string(index=False))
    etiquetees = df["gagnante"].notna()
    motifs = pd.crosstab(df.loc[etiquetees, "pendant_motif_sortie"],
                         df.loc[etiquetees, "gagnante"].map({1.0: "gagnantes", 0.0: "perdantes"}),
                         normalize="columns") * 100
    print("\nMotif de la dernière vente, en % de chaque groupe :")
    print(motifs.round(1).to_string())

    # Sorties ------------------------------------------------------------- #
    df = df.assign(proba_gagnante_test=res.proba_test, proba_gagnante_walk_forward=proba_oos,
                   echantillon=np.select([apprentissage, test], ["apprentissage", "test"], "ecartee"))
    df.to_parquet(sortie / "theses.parquet", index=False)
    (sortie / "resume.json").write_text(json.dumps({
        "run": str(dossier), "strategie": run.run_config.get("strategy"), "label": args.label,
        "date_coupure": str(coupure.date()), "n_theses": int(len(df)),
        "n_apprentissage": res.n_apprentissage, "n_test": res.n_test,
        "auc_test": res.auc_test, "ic_auc_test": list(res.ic_test),
        "auc_test_ecart_seul": res.auc_reference_ecart, "auc_test_logistique": res.auc_logistique,
        "ic_auc_test_logistique": list(res.ic_logistique),
        "auc_walk_forward": ct.auc(df.loc[proba_oos.index, "gagnante"], proba_oos) if len(proba_oos) else None,
        "variables_stables": stables["variable"].tolist(),
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nTables écrites dans {sortie}")


if __name__ == "__main__":
    main()
