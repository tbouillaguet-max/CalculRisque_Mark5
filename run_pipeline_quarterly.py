"""
Orchestrateur du pipeline sur une base TRIMESTRIELLE plutôt qu'annuelle.

Deux modes, mutuellement exclusifs :

MODE LIVE (défaut, sans --as-of-date)
---------------------------------------
Enchaîne les étapes en conditions réelles (nouveaux appels réseau SEC/IBKR/
Mistral), dans l'ordre :
    04b (10-Q + TTM) -> 04c (8-K entre trimestres) -> 05 (multiples) ->
    06 (multiples moyens) -> 06b (valorisation combinée) -> 07 (DCF) ->
    07b (validation qualitative) -> 08 (options, si le filtre d'écart de
    valorisation retient des entreprises)
Chaque étape est un sous-processus indépendant (aucun script numéroté
n'expose de main(args) appelable en mémoire -- ils lisent tous sys.argv via
argparse).

Étapes REQUISES et étapes OPTIONNELLES
-----------------------------------------
Toutes les étapes ne se valent pas. 04b/05/06/06b/07 produisent la
valorisation elle-même : leur échec rend tout ce qui suit incohérent, donc il
arrête le pipeline. 04c/07b (verdicts LLM, nécessitent MISTRAL_API_KEY) et 08
(chaînes d'options, nécessite IB Gateway) sont des ENRICHISSEMENTS : leur
échec est journalisé et le pipeline continue, plutôt que de perdre une
valorisation trimestrielle correcte parce qu'une clé d'API a expiré ou qu'IB
Gateway était fermé. Le run se termine alors en statut "partial".

Chaque étape est réessayée jusqu'à --retries fois avec un backoff exponentiel
(échec transitoire : coupure réseau, 429, Gateway pas encore prêt), et bornée
par --step-timeout pour qu'un blocage réseau ne fige pas un run de cron
jusqu'au suivant.

Avant l'étape 08, le port d'IB Gateway est testé et, s'il ne répond pas,
restart_gateway.py est appelé pour le relancer (voir ses PRÉREQUIS et sa
limite sur la 2FA) -- l'étape n'est tentée que si le port répond ensuite.

Journal de run
-----------------
Chaque run écrit data/pipeline_runs/<run_id>/report.json (statut, durée,
nombre de tentatives et code de retour de chaque étape) et un fichier de log
par étape. La page Streamlit "Pipeline" lit ce journal ; --resume s'en sert
pour reprendre un run interrompu là où il s'était arrêté, sans refaire les
étapes déjà réussies.

MODE REPLAY (--as-of-date AAAA-MM-JJ)
----------------------------------------
Ne fait AUCUN appel réseau. Reconstitue ce que 05/06/06b/07 auraient produit
avec UNIQUEMENT les données déjà connues (filed_date <= as_of_date) à cette
date passée -- utile pour une inspection point-in-time ponctuelle hors du
moteur de backtest complet (qui, lui, fait déjà ce filtrage nativement au
jour le jour, voir backtest/engine.py).

Aucune écriture dans data/ réel : les fichiers sources (financials, TTM,
cours) sont filtrés puis copiés dans un répertoire de travail temporaire, et
05/06/06b/07 y sont exécutés avec ce répertoire comme working directory --
config.py résout BASE_DIR="./data" relativement au répertoire d'exécution
(voir README.md), donc un sous-processus lancé avec cwd=<temp> lit/écrit
entièrement isolé de data/, sans monkey-patch ni variable d'environnement.
Les résultats (DCF_FILE, VALORISATION_COMBINEE_FILE) sont recopiés à côté du
répertoire temporaire pour inspection, puis le répertoire est journalisé
(pas supprimé automatiquement, pour recopie/debug manuel si besoin).

Cron (production, pas de daemon Python custom)
--------------------------------------------------
Un 10-Q est déposé ~45 jours après la fin de trimestre (délai SEC habituel) :
exemple de crontab déclenchant ce script début-mai/août/novembre/février
(après la fenêtre de dépôt la plus probable) :

    0 6 5 2,5,8,11 *  cd /chemin/vers/CalculRisque_Mark3 && \
        /usr/bin/python3 run_pipeline_quarterly.py >> logs/quarterly.log 2>&1

(08 se relance IB Gateway toute seule ; --skip-options si tu préfères la
piloter séparément.)

Usage :
    python run_pipeline_quarterly.py                       # mode live, toutes les étapes
    python run_pipeline_quarterly.py --skip-options         # mode live, sans 08 (pas d'IBKR)
    python run_pipeline_quarterly.py --limit 10             # test rapide (04b/04c/07b limités)
    python run_pipeline_quarterly.py --resume               # reprend le dernier run interrompu
    python run_pipeline_quarterly.py --as-of-date 2024-06-30  # mode replay, aucun appel réseau
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

import config

logger = logging.getLogger("run_pipeline_quarterly")

REPO_ROOT = Path(__file__).resolve().parent

RETRIES_DEFAULT = 2
STEP_TIMEOUT_DEFAULT = 7200  # 2 h : large pour un run complet, borné pour un cron
RETRY_BACKOFF_SEC = 30


@dataclass
class Step:
    """Une étape du pipeline.

    required     : son échec arrête le pipeline (sinon simple avertissement).
    accepts_limit: --limit lui est transmis (étapes qui bouclent sur
                   l'univers/les périodes ; 05/06/06b/07 calculent en une
                   passe sur le cache existant, --limit n'y aurait pas de sens).
    needs_gateway: IB Gateway doit répondre avant de la lancer.
    degraded_args: arguments à utiliser AU LIEU DE SAUTER l'étape quand IB
                   Gateway est indisponible, pour une étape qui sait travailler
                   depuis une autre source (03b se replie sur Stooq via
                   --skip-ibkr). Vide = comportement inchangé, l'étape est
                   sautée. La distinction compte pour un run quotidien : sauter
                   la récupération des cours parce que le Gateway est fermé
                   laisse le signal du jour figé sur les cours de la veille,
                   alors que la source de repli, elle, était disponible.
    extra_args   : arguments TOUJOURS passés à l'étape, quel que soit le mode
                   (le run quotidien y met --refresh-days pour resserrer la
                   fenêtre de fraîcheur SEC). Vides pour toutes les étapes du
                   run trimestriel, qui gardent leurs défauts.
    """
    script: str
    required: bool = True
    accepts_limit: bool = False
    needs_gateway: bool = False
    degraded_args: tuple = ()
    extra_args: tuple = ()


LIVE_STEPS: List[Step] = [
    Step("04b_recuperation_10q.py", accepts_limit=True),
    Step("04c_recuperation_8k.py", required=False, accepts_limit=True),
    Step("05_calcul_multiples.py"),
    Step("06_calcul_multiples_moyens.py"),
    Step("06b_calcul_valorisation_combinee.py"),
    Step("07_calcul_dcf.py"),
    Step("07b_validation_qualitative.py", required=False, accepts_limit=True),
    Step("08_recuperation_options.py", required=False, needs_gateway=True),
]

REPLAY_STEPS: List[Step] = [
    Step("05_calcul_multiples.py"), Step("06_calcul_multiples_moyens.py"),
    Step("06b_calcul_valorisation_combinee.py"), Step("07_calcul_dcf.py"),
]


# ----------------------------------------------------------------------------
# Journal de run
# ----------------------------------------------------------------------------

@dataclass
class RunReport:
    run_id: str
    mode: str
    directory: Path
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    steps: List[dict] = field(default_factory=list)
    status: str = "running"
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    replay_workspace: Optional[str] = None

    def record(self, entry: dict) -> None:
        self.steps.append(entry)
        self.save()

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id, "mode": self.mode, "status": self.status,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds, "steps": self.steps,
        }
        if self.replay_workspace:
            payload["replay_workspace"] = self.replay_workspace
        path = self.directory / config.PIPELINE_RUN_REPORT_NAME
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


RESUMABLE_STATUSES = ("failed", "running")


def find_interrupted_report(mode: str, current_run_id: str) -> Optional[dict]:
    """Rapport du run INTERROMPU le plus récent dans ce mode, ou None.

    Le run courant est exclu (son rapport est déjà sur disque quand --resume
    est évalué), et seuls les runs réellement interrompus sont candidats :
    reprendre un run terminé sauterait toutes ses étapes et ne ferait donc
    rien du tout -- typiquement le cas d'un cron trimestriel qui passerait
    --resume en permanence."""
    if not config.DIR_PIPELINE_RUNS.exists():
        return None
    for run_dir in sorted(config.DIR_PIPELINE_RUNS.iterdir(), reverse=True):
        if run_dir.name == current_run_id:
            continue
        path = run_dir / config.PIPELINE_RUN_REPORT_NAME
        if not path.exists():
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if report.get("mode") == mode:
            return report if report.get("status") in RESUMABLE_STATUSES else None
    return None


def succeeded_steps(report: Optional[dict]) -> set[str]:
    if not report:
        return set()
    return {s["script"] for s in report.get("steps", []) if s.get("status") == "success"}


# ----------------------------------------------------------------------------
# Exécution d'une étape
# ----------------------------------------------------------------------------

def _stream_subprocess(cmd: List[str], cwd: Optional[Path], log_path: Path, timeout: int) -> int:
    """Lance la commande en diffusant sa sortie à la fois sur la console (run
    interactif) et dans log_path (run de cron, et consultation a posteriori
    depuis la page Streamlit). Tue le processus au-delà de timeout secondes.

    La sortie est drainée par un thread dédié et le délai est surveillé par
    `process.wait(timeout=...)` : une étape figée sans rien écrire (attente
    réseau, IB Gateway muet) ne produit aucune ligne, donc un contrôle du
    délai dans la boucle de lecture ne s'exécuterait jamais."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )

        def pump() -> None:
            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()

        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            returncode = process.wait()

        reader.join(timeout=30)
        if timed_out:
            message = f"[orchestrateur] étape interrompue : dépassement du délai de {timeout}s.\n"
            sys.stdout.write(message)
            log_file.write(message)
        return returncode


def run_step(step: Step, extra_args: List[str], report: RunReport, retries: int, timeout: int, cwd: Optional[Path] = None) -> bool:
    """Exécute une étape, avec réessais et backoff. Retourne True si elle a
    réussi. Une étape requise en échec lève RuntimeError (le pipeline
    s'arrête) ; une étape optionnelle est simplement journalisée."""
    cmd = [sys.executable, str(REPO_ROOT / step.script), *extra_args]
    entry = {
        "script": step.script, "required": step.required, "args": extra_args,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "attempts": 0, "status": "failed", "returncode": None,
        "log": str((report.directory / f"{step.script}.log").relative_to(config.BASE_DIR)),
    }
    started = time.monotonic()

    returncode = None
    for attempt in range(1, retries + 2):
        entry["attempts"] = attempt
        logger.info("=== %s %s (tentative %d/%d, cwd=%s) ===", step.script, " ".join(extra_args), attempt, retries + 1, cwd or REPO_ROOT)
        returncode = _stream_subprocess(cmd, cwd, report.directory / f"{step.script}.log", timeout)
        if returncode == 0:
            entry["status"] = "success"
            break
        if attempt <= retries:
            delay = RETRY_BACKOFF_SEC * (2 ** (attempt - 1))
            logger.warning("%s a échoué (code %s) -- nouvelle tentative dans %ds.", step.script, returncode, delay)
            time.sleep(delay)

    entry["returncode"] = returncode
    entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
    entry["duration_seconds"] = round(time.monotonic() - started, 1)
    report.record(entry)

    if entry["status"] == "success":
        return True
    if step.required:
        raise RuntimeError(f"{step.script} a échoué (code {returncode}) après {entry['attempts']} tentative(s) -- pipeline arrêté.")
    logger.warning(
        "%s a échoué (code %s) mais n'est pas bloquante : le pipeline continue "
        "(run marqué 'partial'). Détail : %s",
        step.script, returncode, report.directory / f"{step.script}.log",
    )
    return False


def skip_step(step: Step, report: RunReport, reason: str) -> None:
    logger.info("%s sautée : %s", step.script, reason)
    report.record({
        "script": step.script, "required": step.required, "status": "skipped",
        "reason": reason, "attempts": 0,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    })


# ----------------------------------------------------------------------------
# IB Gateway (étape 08)
# ----------------------------------------------------------------------------

def _gateway_port() -> int:
    """Port API d'IB Gateway, lu depuis .env comme le fait restart_gateway.py.
    Repli sur le port paper trading par défaut d'IBKR si .env est absent."""
    try:
        import restart_gateway
        return int(restart_gateway._load_env()["IB_GATEWAY_PORT"])
    except Exception:  # noqa: BLE001 -- .env absent/incomplet : on teste quand même le port usuel
        return 4002


def ensure_gateway_available() -> bool:
    """True si le port d'IB Gateway répond, au besoin après l'avoir relancé.
    Ne lève jamais : l'étape 08 étant optionnelle, un Gateway indisponible
    doit dégrader le run, pas l'interrompre."""
    port = _gateway_port()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=3):
            return True
    except OSError:
        pass

    logger.warning("IB Gateway ne répond pas sur le port %d : tentative de redémarrage automatique...", port)
    try:
        import restart_gateway
        return restart_gateway.restart_ib_gateway()
    except Exception as exc:  # noqa: BLE001 -- config .env/IBC manquante ou incomplète
        logger.warning("Redémarrage automatique d'IB Gateway impossible (%s).", exc)
        return False


# ----------------------------------------------------------------------------
# Mode live
# ----------------------------------------------------------------------------

def resolve_gateway_args(step: Step, report: RunReport) -> Optional[List[str]]:
    """Arguments supplémentaires imposés par l'état d'IB Gateway, ou None si
    l'étape doit être sautée.

    Trois cas : l'étape n'a pas besoin du Gateway (aucun argument), le Gateway
    répond (aucun argument), ou il ne répond pas -- et c'est alors
    `degraded_args` qui décide entre repli et saut (cf. Step)."""
    if not step.needs_gateway or ensure_gateway_available():
        return []
    if step.degraded_args:
        logger.warning(
            "IB Gateway indisponible : %s est lancée en mode dégradé (%s) plutôt que sautée.",
            step.script, " ".join(step.degraded_args),
        )
        return list(step.degraded_args)
    skip_step(step, report, "IB Gateway indisponible (et non relançable automatiquement)")
    return None


def run_live(report: RunReport, limit: Optional[int], skip_options: bool, retries: int, timeout: int, already_done: set[str]) -> None:
    for step in LIVE_STEPS:
        if step.script in already_done:
            skip_step(step, report, "déjà réussie lors du run repris (--resume)")
            continue
        if skip_options and step.needs_gateway:
            skip_step(step, report, "--skip-options")
            continue
        gateway_args = resolve_gateway_args(step, report)
        if gateway_args is None:
            continue

        extra_args = ["--limit", str(limit)] if (limit and step.accepts_limit) else []
        run_step(step, [*extra_args, *gateway_args], report, retries, timeout)


# ----------------------------------------------------------------------------
# Mode replay : reconstitution point-in-time, aucun appel réseau
# ----------------------------------------------------------------------------

def _filter_by_filed_date(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if "filed_date" not in df.columns or df.empty:
        return df
    filed = pd.to_datetime(df["filed_date"], errors="coerce")
    return df[filed <= as_of].copy()


def _prepare_replay_workspace(as_of: pd.Timestamp) -> Path:
    """Copie FILTRÉE (filed_date <= as_of) des fichiers sources nécessaires
    à 05/06/06b/07, dans une arborescence data/ isolée sous un répertoire
    temporaire -- jamais d'écriture dans data/ réel (voir docstring)."""
    scratch = Path(tempfile.mkdtemp(prefix="quarterly_replay_"))
    scratch_data = scratch / "data"

    for d in (config.DIR_UNIVERSE, config.DIR_PRICES, config.DIR_FINANCIALS, config.DIR_MULTIPLES, config.DIR_DCF):
        rel = d.relative_to(config.BASE_DIR)
        (scratch_data / rel).mkdir(parents=True, exist_ok=True)

    def _dest(path: Path) -> Path:
        return scratch_data / path.relative_to(config.BASE_DIR)

    if config.UNIVERSE_FILE.exists():
        shutil.copy(config.UNIVERSE_FILE, _dest(config.UNIVERSE_FILE))
    else:
        raise FileNotFoundError(f"{config.UNIVERSE_FILE} introuvable -- lance d'abord 01_build_universe.py.")

    if config.FINANCIALS_FILE.exists():
        annual = _filter_by_filed_date(pd.read_parquet(config.FINANCIALS_FILE), as_of)
        annual.to_parquet(_dest(config.FINANCIALS_FILE), index=False)
    else:
        raise FileNotFoundError(f"{config.FINANCIALS_FILE} introuvable -- lance d'abord 04_recuperation_10k.py.")

    if config.FINANCIALS_TTM_FILE.exists():
        ttm = _filter_by_filed_date(pd.read_parquet(config.FINANCIALS_TTM_FILE), as_of)
        if not ttm.empty:
            ttm.to_parquet(_dest(config.FINANCIALS_TTM_FILE), index=False)

    if config.PRICES_FILE.exists():
        prices = pd.read_parquet(config.PRICES_FILE)
        # Un cours de clôture de fin d'année N est réputé connu dès le
        # 31/12/N (pas de délai de dépôt SEC pour un cours de marché,
        # contrairement aux données financières) -- filtré sur cette base.
        year_end_known = pd.to_datetime(prices["year"].astype(int).astype(str) + "-12-31")
        prices[year_end_known <= as_of].to_parquet(_dest(config.PRICES_FILE), index=False)

    if config.DAILY_PRICES_FILE.exists():
        daily = pd.read_parquet(config.DAILY_PRICES_FILE)
        daily = daily[pd.to_datetime(daily["date"]) <= as_of]
        (scratch_data / config.DAILY_PRICES_FILE.relative_to(config.BASE_DIR)).parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(_dest(config.DAILY_PRICES_FILE), index=False)

    return scratch


def run_replay(report: RunReport, as_of_date: str, retries: int, timeout: int) -> None:
    try:
        as_of = pd.Timestamp(as_of_date)
    except ValueError as exc:
        raise ValueError(f"--as-of-date invalide (attendu AAAA-MM-JJ) : {as_of_date}") from exc

    logger.info("Mode replay point-in-time : reconstitution au %s (aucun appel réseau).", as_of.date())
    scratch = _prepare_replay_workspace(as_of)
    report.replay_workspace = str(scratch)
    report.save()
    logger.info("Espace de travail isolé : %s", scratch)

    for step in REPLAY_STEPS:
        run_step(step, [], report, retries, timeout, cwd=scratch)

    scratch_dcf = scratch / "data" / config.DCF_FILE.relative_to(config.BASE_DIR)
    scratch_combined = scratch / "data" / config.VALORISATION_COMBINEE_FILE.relative_to(config.BASE_DIR)
    logger.info("Terminé. Résultats point-in-time disponibles sous :")
    if scratch_dcf.exists():
        logger.info("  - DCF : %s", scratch_dcf)
    if scratch_combined.exists():
        logger.info("  - Valorisation combinée : %s", scratch_combined)
    logger.info("Répertoire temporaire conservé (pas de suppression automatique) pour inspection/recopie manuelle.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of-date", type=str, default=None, metavar="YYYY-MM-DD", help="Mode replay point-in-time (aucun appel réseau) au lieu du mode live.")
    parser.add_argument("--limit", type=int, default=None, help="Transmis à 04b/04c/07b (mode live uniquement) pour un test rapide.")
    parser.add_argument("--skip-options", action="store_true", help="Saute l'étape 08 (chaînes d'options) au lieu de tenter de relancer IB Gateway -- mode live uniquement.")
    parser.add_argument("--retries", type=int, default=RETRIES_DEFAULT, help="Réessais par étape en cas d'échec, avec backoff exponentiel (défaut: %(default)s).")
    parser.add_argument("--step-timeout", type=int, default=STEP_TIMEOUT_DEFAULT, help="Durée maximale d'une étape, en secondes (défaut: %(default)s).")
    parser.add_argument("--resume", action="store_true", help="Reprend le dernier run du même mode en sautant les étapes déjà réussies.")
    parser.add_argument("--run-id", default=None, help="Nom du sous-dossier de journal (défaut: horodatage).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    mode = "replay" if args.as_of_date else "live"
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    report = RunReport(run_id=run_id, mode=mode, directory=config.DIR_PIPELINE_RUNS / run_id)
    report.save()

    already_done: set[str] = set()
    if args.resume:
        previous = find_interrupted_report(mode, run_id)
        already_done = succeeded_steps(previous)
        if already_done:
            logger.info("--resume : %d étape(s) déjà réussies au run %s seront sautées.", len(already_done), previous.get("run_id"))
        else:
            logger.info("--resume : aucun run interrompu à reprendre dans ce mode, run complet.")

    start = time.monotonic()
    exit_code = 0
    try:
        if args.as_of_date:
            run_replay(report, args.as_of_date, args.retries, args.step_timeout)
        else:
            run_live(report, args.limit, args.skip_options, args.retries, args.step_timeout, already_done)
        failed_optional = [s["script"] for s in report.steps if s["status"] == "failed"]
        report.status = "partial" if failed_optional else "success"
        if failed_optional:
            logger.warning("Run terminé en mode dégradé : étape(s) optionnelle(s) en échec -> %s", ", ".join(failed_optional))
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        report.status = "failed"
        logger.error("Pipeline arrêté : %s", exc)
        logger.error("Relance avec --resume pour repartir des étapes restantes.")
        exit_code = 1

    report.finished_at = datetime.now().isoformat(timespec="seconds")
    report.duration_seconds = round(time.monotonic() - start, 1)
    report.save()

    logger.info("Pipeline terminé (%s) en %.0fs. Journal : %s",
                report.status, report.duration_seconds, report.directory / config.PIPELINE_RUN_REPORT_NAME)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
