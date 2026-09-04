"""
Mise à jour QUOTIDIENNE du pipeline, en une commande.

    python run_pipeline_daily.py

POURQUOI UN RUN QUOTIDIEN A UN SENS ALORS QUE LES COMPTES SONT TRIMESTRIELS
---------------------------------------------------------------------------
Le signal de la stratégie options est un ÉCART entre deux grandeurs :

    écart = 100 x ln( valeur théorique / cours )

La valeur théorique ne bouge qu'au dépôt d'un 10-Q/10-K. Le COURS, lui, bouge
tous les jours -- et la stratégie multiples est configurée en
`daily_rebalance=True` précisément pour cela (cf. backtest/options_engine.py).
Une entreprise peut donc franchir le seuil d'entrée, ou repasser sous le seuil
de sortie, par le seul mouvement du titre, sans qu'aucun compte n'ait été
publié. Sans run quotidien, ces franchissements ne sont vus qu'au trimestre
suivant.

C'est la différence avec run_pipeline_quarterly.py : celui-ci va CHERCHER les
nouveaux comptes (04b/04c, coûteux, inutile tous les jours) ; celui-là
rafraîchit les COURS et recalcule l'écart avec la valeur théorique déjà
connue. Les étapes de calcul (05/06/06b/07) sont communes aux deux parce
qu'elles sont locales et bon marché : les rejouer chaque jour garantit que le
signal lu par le dashboard et par le backtest est celui des cours du jour.

CE QUE FAIT LE RUN, DANS L'ORDRE
---------------------------------
    03b   cours quotidiens (incrémental : seuls les jours manquants sont
          téléchargés). IBKR, avec repli Stooq automatique si le Gateway est
          fermé -- l'étape est DÉGRADÉE, jamais sautée : c'est la seule qui
          apporte réellement de l'information nouvelle un jour ordinaire.
    04    10-K annuels, --refresh-days 7 : ne réinterroge la SEC que pour les
          entreprises non vues depuis une semaine. Presque toujours un no-op,
          mais c'est ce qui fait entrer un 10-K le jour de son dépôt sans
          attendre le run trimestriel.
    04b   10-Q + TTM, même logique incrémentale.
    04c   8-K matériels (optionnel : MISTRAL_API_KEY).
    05    multiples par entreprise
    06    multiples sectoriels moyens
    06b   valorisation combinée (multiples, repli DCF)   <- LE SIGNAL
    07    DCF
    07b   validation qualitative LLM (optionnel)
    08    chaînes d'options (optionnel : IB Gateway)

04/04b/04c/07b/08 sont OPTIONNELLES : leur échec est journalisé et le run
continue en statut "partial". 03b et 05/06/06b/07 sont requises -- sans elles
le signal du jour est soit absent, soit incohérent avec les cours.

TOUT LE RESTE EST REPRIS DE run_pipeline_quarterly.py (réessais avec backoff,
délai par étape, journal JSON par run, --resume, redémarrage automatique d'IB
Gateway) : ce fichier ne redéfinit que la LISTE des étapes et leurs arguments,
pas la mécanique d'exécution.

Cron (jours de bourse, après la clôture US = 22h00 CET / 16h00 ET) :

    30 22 * * 1-5  cd /chemin/vers/CalculRisque_Mark5 && \
        /usr/bin/python3 run_pipeline_daily.py >> logs/daily.log 2>&1

Usage :
    python run_pipeline_daily.py                  # run complet
    python run_pipeline_daily.py --skip-options   # sans 08 (pas besoin d'IBKR)
    python run_pipeline_daily.py --prices-only    # 03b + recalcul du signal
    python run_pipeline_daily.py --resume         # reprend un run interrompu
    python run_pipeline_daily.py --limit 10       # test rapide
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from typing import List, Optional

import config
from run_pipeline_quarterly import (
    RETRIES_DEFAULT,
    RunReport,
    Step,
    find_interrupted_report,
    resolve_gateway_args,
    run_step,
    skip_step,
    succeeded_steps,
)

logger = logging.getLogger("run_pipeline_daily")

# Plus court que les 2 h du run trimestriel : un run quotidien qui déborde sur
# la séance suivante n'a plus d'intérêt, et le cron du lendemain le remplacera.
STEP_TIMEOUT_DEFAULT = 3600

# Fenêtre de fraîcheur des dépôts SEC. 7 jours plutôt que les 30 du run
# trimestriel : on veut qu'un 10-K/10-Q déposé aujourd'hui entre dans le signal
# en quelques jours, sans pour autant réinterroger toute la SEC chaque matin
# (cf. 04_recuperation_10k.py::should_skip).
FILINGS_REFRESH_DAYS = 7


def daily_steps(filings_refresh_days: int) -> List[Step]:
    """Étapes du run quotidien.

    Construites par fonction plutôt que déclarées en constante : les arguments
    de fraîcheur (--refresh-days) dépendent de la ligne de commande, et les
    figer dans une constante obligerait à les réécrire à l'exécution."""
    refresh = ("--refresh-days", str(filings_refresh_days))
    return [
        # La seule étape qui apporte de l'information nouvelle un jour
        # ordinaire -- donc requise, et dégradée sur Stooq plutôt que sautée
        # si le Gateway est fermé (cf. Step.degraded_args).
        Step(
            "03b_recuperation_cours_quotidiens.py",
            needs_gateway=True, degraded_args=("--skip-ibkr",), accepts_limit=True,
        ),
        Step("04_recuperation_10k.py", required=False, accepts_limit=True, extra_args=refresh),
        Step("04b_recuperation_10q.py", required=False, accepts_limit=True, extra_args=refresh),
        Step("04c_recuperation_8k.py", required=False, accepts_limit=True),
        Step("05_calcul_multiples.py"),
        Step("06_calcul_multiples_moyens.py"),
        Step("06b_calcul_valorisation_combinee.py"),
        Step("07_calcul_dcf.py"),
        Step("07b_validation_qualitative.py", required=False, accepts_limit=True),
        Step("08_recuperation_options.py", required=False, needs_gateway=True),
    ]


# Étapes conservées par --prices-only : rafraîchir les cours et recalculer
# l'écart, sans aucun appel SEC ni LLM. C'est le run le plus court qui met
# encore le signal à jour.
PRICES_ONLY = {
    "03b_recuperation_cours_quotidiens.py",
    "05_calcul_multiples.py", "06_calcul_multiples_moyens.py",
    "06b_calcul_valorisation_combinee.py", "07_calcul_dcf.py",
}


def run_daily(
    report: RunReport, limit: Optional[int], skip_options: bool, prices_only: bool,
    retries: int, timeout: int, already_done: set[str], filings_refresh_days: int,
) -> None:
    for step in daily_steps(filings_refresh_days):
        if step.script in already_done:
            skip_step(step, report, "déjà réussie lors du run repris (--resume)")
            continue
        if prices_only and step.script not in PRICES_ONLY:
            skip_step(step, report, "--prices-only")
            continue
        if skip_options and step.script == "08_recuperation_options.py":
            skip_step(step, report, "--skip-options")
            continue
        gateway_args = resolve_gateway_args(step, report)
        if gateway_args is None:
            continue

        extra_args = ["--limit", str(limit)] if (limit and step.accepts_limit) else []
        run_step(
            step, [*step.extra_args, *extra_args, *gateway_args],
            report, retries, timeout,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--skip-options", action="store_true",
        help="Saute l'étape 08 (chaînes d'options) -- pas besoin d'IB Gateway.")
    parser.add_argument(
        "--prices-only", action="store_true",
        help="Cours + recalcul du signal uniquement (aucun appel SEC ni LLM). Le run le plus court "
             "qui met encore le signal à jour.")
    parser.add_argument(
        "--filings-refresh-days", type=int, default=FILINGS_REFRESH_DAYS,
        help="Fenêtre de fraîcheur des dépôts SEC transmise à 04/04b (défaut: %(default)s).")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Transmis aux étapes qui bouclent sur l'univers, pour un test rapide.")
    parser.add_argument(
        "--retries", type=int, default=RETRIES_DEFAULT,
        help="Réessais par étape, avec backoff exponentiel (défaut: %(default)s).")
    parser.add_argument(
        "--step-timeout", type=int, default=STEP_TIMEOUT_DEFAULT,
        help="Durée maximale d'une étape, en secondes (défaut: %(default)s).")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reprend le dernier run quotidien interrompu en sautant les étapes déjà réussies.")
    parser.add_argument("--run-id", default=None, help="Nom du sous-dossier de journal.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    # Mode distinct de "live"/"replay" : --resume ne doit reprendre qu'un run
    # QUOTIDIEN interrompu, jamais un run trimestriel laissé en plan (leurs
    # listes d'étapes diffèrent, les noms d'étapes réussies ne sont pas
    # interchangeables).
    report = RunReport(run_id=run_id, mode="daily", directory=config.DIR_PIPELINE_RUNS / run_id)
    report.save()

    already_done: set[str] = set()
    if args.resume:
        previous = find_interrupted_report("daily", run_id)
        already_done = succeeded_steps(previous)
        if already_done:
            logger.info(
                "--resume : %d étape(s) déjà réussies au run %s seront sautées.",
                len(already_done), previous.get("run_id"))
        else:
            logger.info("--resume : aucun run quotidien interrompu à reprendre, run complet.")

    start = time.monotonic()
    exit_code = 0
    try:
        run_daily(
            report, args.limit, args.skip_options, args.prices_only,
            args.retries, args.step_timeout, already_done, args.filings_refresh_days,
        )
        failed_optional = [s["script"] for s in report.steps if s["status"] == "failed"]
        report.status = "partial" if failed_optional else "success"
        if failed_optional:
            logger.warning(
                "Run terminé en mode dégradé : étape(s) optionnelle(s) en échec -> %s",
                ", ".join(failed_optional))
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        report.status = "failed"
        logger.error("Run quotidien arrêté : %s", exc)
        logger.error("Relance avec --resume pour repartir des étapes restantes.")
        exit_code = 1

    report.finished_at = datetime.now().isoformat(timespec="seconds")
    report.duration_seconds = round(time.monotonic() - start, 1)
    report.save()

    logger.info(
        "Run quotidien terminé (%s) en %.0fs. Journal : %s",
        report.status, report.duration_seconds,
        report.directory / config.PIPELINE_RUN_REPORT_NAME)
    if report.status != "failed":
        logger.info("Signal du jour : %s", config.VALORISATION_COMBINEE_FILE)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
