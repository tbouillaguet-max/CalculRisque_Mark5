"""
Stratégie options "convergence vers les multiples sectoriels", à horizon long.

Principe : comparer la valorisation THÉORIQUE issue de la méthode des
multiples (06b_calcul_valorisation_combinee.py) à la valorisation BOURSIÈRE,
et parier sur la convergence de la seconde vers la première.

    valorisation boursière   = nb d'actions x cours de l'action
    valorisation théorique   = nb d'actions x valeur théorique par action

Le rapport entre les deux ne dépend pas du nombre d'actions (identique des
deux côtés) : l'écart se calcule donc directement par action, sur les
colonnes déjà produites par 06b, sans reconstruire de capitalisation.

L'échéance 2 ans, le roulement à 9 mois et les stops mesurés sur le cours du
sous-jacent, propres à cette stratégie à l'origine, sont devenus le
comportement par défaut du moteur (config.OPTIONS_TARGET_TENOR_DAYS,
OPTIONS_ROLL_WHEN_DAYS_LEFT, OPTIONS_STOP_BASIS) : les engine_defaults
ci-dessous les redéclarent pour que la thèse reste explicite, mais ce ne sont
plus des différences. Ce qui la sépare de valuation_gap_options :

    1. MULTIPLES SEULS. 06b se rabat sur le DCF quand un secteur compte trop
       peu de pairs pour une médiane robuste ; ces lignes (source="dcf_fallback")
       sont écartées ici -- la thèse porte sur un écart aux COMPARABLES
       sectoriels, pas sur un écart à une actualisation de flux.
    2. ÉCART EN LOG, symétrique : 100 x ln(théorique / cours)
       (gap_basis="log"). "Moitié prix" et "double prix" comptent pour la même
       conviction au signe près, et le seuil filtre les deux sens à identité.
       Les deux conventions en pourcentage conservées pour rejouer un run
       ancien ("theoretical", "close") sont asymétriques en miroir l'une de
       l'autre : chacune borne un côté et laisse l'autre libre, ce qui
       déformait la sélection ET le classement en faveur d'une direction (voir
       config.OPTIONS_MULTIPLES_GAP_BASIS pour le détail chiffré).
    3. STRIKE à mi-chemin entre valeur théorique et cours, au lieu d'ATM. Il
       est donc systématiquement hors de la monnaie, de la moitié de l'écart :
       l'option ne devient gagnante que si le titre parcourt au moins la moitié
       du chemin vers sa valeur théorique. C'est délibéré -- une convergence
       partielle suffit, mais un simple bruit de marché ne suffit pas. Cette
       fragilité supplémentaire à un mouvement adverse justifie un stop-loss
       plus resserré (-25% contre -20%).
       Le TAKE-PROFIT, lui, n'est pas un seuil fixe : la position est prise en
       gain à OPTIONS_TAKE_PROFIT_CONVERGENCE_FRACTION du chemin vers la valeur
       théorique (80% par défaut). Un seuil fixe n'était pas atteignable de la
       même façon des deux côtés -- côté put, la convergence complète ne vaut
       que -16,7% de cours au seuil d'entrée, si bien qu'un take-profit à +25%
       ou +30% exigeait un dépassement de la cible et ne se déclenchait
       pratiquement jamais (cf. config.OPTIONS_TAKE_PROFIT_CONVERGENCE_FRACTION).
    4. SORTIES pilotées par le signal (le moteur est configuré en
       exit_when_signal_lost, cf. engine_defaults) : une position dont l'écart
       est repassé sous le seuil est VENDUE au rebalancement suivant, sans
       attendre le réexamen de roulement qui la clôturerait de toute façon à 9
       mois de l'échéance dans l'autre stratégie.
    5. RÉÉVALUATION QUOTIDIENNE (daily_rebalance=True, cf. engine_defaults et
       la docstring de options_engine.py) : l'éligibilité est recalculée
       chaque jour de bourse avec le cours DU JOUR, pas seulement aux dates
       de dépôt SEC -- une entreprise dont l'écart franchit le seuil entre
       deux publications (mouvement de cours pur, sans nouveau 10-Q/10-K) est
       donc détectée sans attendre le trimestre suivant. La valeur théorique,
       elle, ne bouge qu'au prochain dépôt (comme avant) : seul le cours
       comparé à cette valeur est mis à jour au jour le jour.

Le signal étant recalculé à chaque publication trimestrielle (10-Q via
04b_recuperation_10q.py, 10-K via 04) et daté de sa date de dépôt SEC réelle,
chaque nouveau dépôt met à jour la valorisation théorique de l'entreprise
concernée ; entre deux dépôts, c'est le rebalancement quotidien (point 5
ci-dessus) qui capte les mouvements de cours purs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
from backtest.strategies.base import (
    capped_weights, inflation_adjusted_gap, inflation_adjusted_log_gap,
)
from backtest.strategies.options_base import OptionsStrategy, register_options_strategy

logger = logging.getLogger("backtest.strategies.valuation_gap_multiples_options")


@register_options_strategy("valuation_gap_multiples_options")
class ValuationGapMultiplesOptionsStrategy(OptionsStrategy):
    # Le strike vise la valeur théorique (cf. generate_option_targets) : c'est
    # la fraction de convergence qui déclenche la prise de gain, pas
    # take_profit_pct (cf. OptionsStrategy.targets_convergence).
    targets_convergence = True

    # Réglages moteur que cette stratégie suppose (voir la docstring de
    # backtest/options_engine.py). Appliqués par 10_backtest_options.py sauf
    # si l'option correspondante est passée explicitement en ligne de commande.
    engine_defaults = {
        "stop_loss_pct": config.OPTIONS_MULTIPLES_STOP_LOSS_PCT,
        "take_profit_pct": config.OPTIONS_MULTIPLES_TAKE_PROFIT_PCT,
        "target_tenor_days": config.OPTIONS_MULTIPLES_TENOR_DAYS,
        "stop_basis": "underlying",
        "exit_when_signal_lost": True,
        "roll_when_days_left": config.OPTIONS_MULTIPLES_ROLL_WHEN_DAYS_LEFT,
        # Réévalue l'éligibilité tous les jours de bourse (avec le cours du
        # jour), pas seulement aux dates de dépôt SEC : une opportunité (ou
        # une sortie de seuil) créée par le seul mouvement du titre entre
        # deux publications trimestrielles est sinon invisible jusqu'au
        # prochain 10-Q/10-K -- voir la docstring d'options_engine.py.
        "daily_rebalance": True,
        # Volatilité de repricing suivie au jour le jour plutôt que figée à
        # l'entrée : sur une échéance 2 ans, figer la volatilité pendant toute
        # la vie de la position est une approximation nettement plus forte que
        # sur 9 mois (voir options_engine._repricing_vol).
        "vol_mode": "rolling",
    }

    def __init__(
        self,
        entry_threshold_pct: float = config.OPTIONS_MULTIPLES_ENTRY_THRESHOLD_PCT,
        tenor_days: int = config.OPTIONS_MULTIPLES_TENOR_DAYS,
        gap_basis: str = config.OPTIONS_MULTIPLES_GAP_BASIS,
        multiples_only: bool = True,
        weight_cap_pct: float | None = config.OPTIONS_MULTIPLES_WEIGHT_CAP_PCT,
        exit_threshold_ratio: float = config.OPTIONS_EXIT_THRESHOLD_RATIO,
        **kwargs,
    ):
        super().__init__(
            entry_threshold_pct=entry_threshold_pct,
            tenor_days=tenor_days, gap_basis=gap_basis, multiples_only=multiples_only,
            weight_cap_pct=weight_cap_pct, exit_threshold_ratio=exit_threshold_ratio, **kwargs,
        )
        if gap_basis not in ("log", "theoretical", "close"):
            raise ValueError(f"gap_basis attend 'log', 'theoretical' ou 'close', reçu {gap_basis!r}.")
        self.entry_threshold_pct = entry_threshold_pct
        self.tenor_days = int(tenor_days)
        self.gap_basis = gap_basis
        self.multiples_only = bool(multiples_only)
        # 0 ou négatif -> pas de plafond (équivaut à None), pour pouvoir le
        # désactiver depuis la ligne de commande, qui ne transmet que des nombres.
        self.weight_cap_pct = None if not weight_cap_pct or weight_cap_pct <= 0 else float(weight_cap_pct)
        # Hystérésis : le seuil au-dessous duquel une position DÉJÀ OUVERTE
        # cesse d'être justifiée, plus bas que le seuil d'entrée (cf.
        # config.OPTIONS_EXIT_THRESHOLD_RATIO). Ratio à 1 -> pas d'hystérésis.
        self.exit_threshold_ratio = float(exit_threshold_ratio)
        self.exit_threshold_pct = entry_threshold_pct * self.exit_threshold_ratio

    # ------------------------------------------------------------------ #
    def _scored(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Toutes les lignes exploitables, avec leur écart (`_gap`), sa valeur
        absolue (`_abs_gap`) et le score de conviction plafonné
        (`_conviction`) -- SANS filtrage par seuil.

        Séparé de _candidates parce que les deux seuils (entrée et sortie,
        cf. exit_threshold_pct) s'appliquent au MÊME calcul : c'est la partie
        coûteuse, et elle n'a aucune raison d'être refaite deux fois par jour
        de bourse."""
        df = signals
        if self.multiples_only:
            if "source" not in df.columns:
                logger.warning(
                    "Signal sans colonne 'source' : impossible d'écarter les lignes valorisées "
                    "par repli DCF, toutes sont donc retenues. Relance 06b_calcul_valorisation_combinee.py "
                    "pour un signal conforme à la stratégie (multiples sectoriels seuls)."
                )
            else:
                df = df[df["source"] == "multiples"]
        df = df.dropna(subset=["valuation_theoretical_per_share", "close"])
        df = df[(df["valuation_theoretical_per_share"] > 0) & (df["close"] > 0)]
        if df.empty:
            return df.assign(_gap=[], _abs_gap=[], _conviction=[])

        theoretical, close = df["valuation_theoretical_per_share"], df["close"]
        # Écart corrigé de l'inflation sur l'horizon du contrat (2 ans par
        # défaut, donc un effet marqué) : la valeur théorique étant nominale,
        # la convergence se fait vers une valeur inflatée -- ce qui aide un
        # call et pénalise un put.
        if self.gap_basis == "log":
            # 100 x ln(V/P) : symétrique par construction (cf.
            # config.OPTIONS_MULTIPLES_GAP_BASIS). La correction d'inflation
            # devient additive en log (cf. base.inflation_adjusted_log_gap).
            df = df.assign(_gap=inflation_adjusted_log_gap(
                np.log(theoretical / close) * 100, df["published_date"],
                self.tenor_days / 365.0,
            ))
        else:
            base = theoretical if self.gap_basis == "theoretical" else close
            df = df.assign(_gap=inflation_adjusted_gap(
                (theoretical - close) / base * 100, df["published_date"],
                self.tenor_days / 365.0,
            ))
        df = df.assign(_abs_gap=df["_gap"].abs())
        # Score de conviction plafonné en ENTRÉE, pour qu'une seule ligne
        # aberrante ne domine pas le calcul du poids, sans toucher à l'ordre
        # de sélection (fait sur l'écart brut). En base "log" le plafond mord
        # symétriquement dans les deux sens (100 points = un rapport de 2,72x) ;
        # dans les bases en pourcentage, il ne bornait en pratique qu'un seul
        # côté -- l'autre étant déjà borné par construction.
        # Ce plafond sur le score d'ENTRÉE ne borne pas à lui seul le poids
        # final (clipper puis normaliser ne garantit aucun pourcentage
        # précis) -- c'est capped_weights, plus bas, qui plafonne réellement
        # le poids résultant à BACKTEST_MAX_WEIGHT_PER_POSITION_PCT, comme
        # pour valuation_gap_options.
        conviction = df["_abs_gap"] if self.weight_cap_pct is None else df["_abs_gap"].clip(upper=self.weight_cap_pct)
        return df.assign(_conviction=conviction)

    def _candidates(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Entreprises dont l'écart justifie une NOUVELLE position (seuil
        d'entrée). Aucun plafond sur le nombre de positions simultanées :
        toutes les candidates retenues ici sont ouvertes."""
        scored = self._scored(signals)
        return scored[scored["_abs_gap"] >= self.entry_threshold_pct]

    @staticmethod
    def _direction(gap: float) -> str:
        """Théorique au-dessus du cours (écart positif) = sous-évalué = CALL ;
        en dessous = survalorisé = PUT."""
        return "CALL" if gap > 0 else "PUT"

    @staticmethod
    def _directions_of(df: pd.DataFrame) -> dict[str, str]:
        """{symbole: sens} construit par zip sur les colonnes numpy plutôt que
        par iterrows : ce dernier matérialise une Series pandas par ligne, ce
        qui représentait à lui seul ~22% du temps de run du moteur."""
        return {
            symbol: ("CALL" if gap > 0 else "PUT")
            for symbol, gap in zip(df["symbol"].to_numpy(), df["_gap"].to_numpy())
        }

    def _targets_from(self, candidates: pd.DataFrame) -> dict[str, dict]:
        if candidates.empty or candidates["_conviction"].sum() <= 0:
            return {}

        # Classement par l'écart BRUT (conviction), dimensionnement par
        # l'écart PLAFONNÉ (cf. _scored). Ordre décroissant conservé pour la
        # lisibilité des sorties, mais aucune troncature : toutes les
        # candidates sont retenues.
        candidates = candidates.sort_values("_abs_gap", ascending=False)

        # Poids réellement plafonné à BACKTEST_MAX_WEIGHT_PER_POSITION_PCT
        # (cf. base.capped_weights) : sans ce second plafond -- sur le poids
        # lui-même, pas seulement sur le score d'entrée -- une seule ligne
        # pouvait encore capter la majorité du capital (mesuré : 92% du
        # portefeuille pour une théorique à 5$ contre un cours à 100$),
        # exactement le scénario de concentration à l'origine des drawdowns
        # impossibles corrigés ailleurs (cf. options_engine._affordable).
        weights = capped_weights(candidates["_conviction"])

        # zip sur les colonnes numpy plutôt qu'iterrows (cf. _directions_of).
        return {
            symbol: {
                "option_type": "CALL" if gap > 0 else "PUT",
                "weight": weight,
                # Strike à mi-chemin théorique/cours : c'est le moteur qui fait
                # la moyenne, avec le spot du jour d'EXÉCUTION plutôt que le
                # cours du signal (les deux diffèrent après un roulement, où le
                # signal peut dater de plusieurs mois).
                "strike_reference_price": float(theoretical),
                "tenor_days": self.tenor_days,
            }
            for symbol, gap, weight, theoretical in zip(
                candidates["symbol"].to_numpy(),
                candidates["_gap"].to_numpy(),
                weights.to_numpy(),
                candidates["valuation_theoretical_per_share"].to_numpy(),
            )
        }

    def generate_option_targets(self, signals: pd.DataFrame, current_positions: dict[str, str]) -> dict[str, dict]:
        return self._targets_from(self._candidates(signals))

    def eligible_directions(self, signals: pd.DataFrame) -> dict[str, str]:
        """Sens encore justifiés au seuil de SORTIE, plus bas que le seuil
        d'entrée (hystérésis, cf. config.OPTIONS_EXIT_THRESHOLD_RATIO).

        C'est cette méthode que le moteur interroge pour décider s'il VEND une
        position (exit_when_signal_lost) ou s'il la ROULE : une position déjà
        engagée, qui a déjà payé son slippage et sa valeur temps, n'a pas à
        être soldée au premier retour sous le seuil qui l'a fait entrer. Une
        convergence partielle est le début de ce qu'on attendait, pas un signal
        perdu."""
        scored = self._scored(signals)
        if scored.empty:
            return {}
        return self._directions_of(scored[scored["_abs_gap"] >= self.exit_threshold_pct])

    def evaluate(
        self, signals: pd.DataFrame, current_positions: dict[str, str],
    ) -> tuple[dict[str, dict], dict[str, str]]:
        """Cibles (seuil d'ENTRÉE) et sens encore justifiés (seuil de SORTIE)
        à partir d'un SEUL calcul d'écart -- cf. OptionsStrategy.evaluate."""
        scored = self._scored(signals)
        if scored.empty:
            return {}, {}
        abs_gap = scored["_abs_gap"]
        targets = self._targets_from(scored[abs_gap >= self.entry_threshold_pct])
        directions = self._directions_of(scored[abs_gap >= self.exit_threshold_pct])
        return targets, directions
