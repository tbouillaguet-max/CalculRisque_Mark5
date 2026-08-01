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

Différences assumées avec valuation_gap_options (l'autre stratégie options,
inchangée) :

    1. MULTIPLES SEULS. 06b se rabat sur le DCF quand un secteur compte trop
       peu de pairs pour une médiane robuste ; ces lignes (source="dcf_fallback")
       sont écartées ici -- la thèse porte sur un écart aux COMPARABLES
       sectoriels, pas sur un écart à une actualisation de flux.
    2. ÉCART RAPPORTÉ AU THÉORIQUE, pas au cours (gap_basis="theoretical").
       Les deux conventions ne retiennent pas les mêmes entreprises : à 20%,
       théorique 120 / cours 100 donne +16,7% en base théorique (écarté) contre
       +20,0% en base cours (retenu).
    3. CONTRAT VISÉ : strike à mi-chemin entre valeur théorique et cours,
       échéance 2 ans. Le strike est donc systématiquement hors de la monnaie,
       de la moitié de l'écart : l'option ne devient gagnante que si le titre
       parcourt au moins la moitié du chemin vers sa valeur théorique. C'est
       délibéré -- une convergence partielle suffit, mais un simple bruit de
       marché ne suffit pas.
    4. SORTIES pilotées par le signal (le moteur est configuré en
       exit_when_signal_lost + roulement, cf. engine_defaults) : une position
       dont l'écart est repassé sous le seuil est VENDUE au trimestre suivant,
       contrairement au comportement "position gelée" de l'autre stratégie.

Le signal étant recalculé à chaque publication trimestrielle (10-Q via
04b_recuperation_10q.py, 10-K via 04) et daté de sa date de dépôt SEC réelle,
la réévaluation demandée "tous les trimestres" est automatique : chaque
nouveau dépôt produit un événement de signal, qui déclenche à la fois les
achats des entreprises passant le seuil et les ventes de celles qui le
repassent en sens inverse.
"""

from __future__ import annotations

import logging

import pandas as pd

import config
from backtest.strategies.options_base import OptionsStrategy, register_options_strategy

logger = logging.getLogger("backtest.strategies.valuation_gap_multiples_options")


@register_options_strategy("valuation_gap_multiples_options")
class ValuationGapMultiplesOptionsStrategy(OptionsStrategy):
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
        # Volatilité de repricing suivie au jour le jour plutôt que figée à
        # l'entrée : sur une échéance 2 ans, figer la volatilité pendant toute
        # la vie de la position est une approximation nettement plus forte que
        # sur 9 mois (voir options_engine._repricing_vol).
        "vol_mode": "rolling",
    }

    def __init__(
        self,
        entry_threshold_pct: float = config.OPTIONS_MULTIPLES_ENTRY_THRESHOLD_PCT,
        max_positions: int = config.OPTIONS_MAX_POSITIONS,
        tenor_days: int = config.OPTIONS_MULTIPLES_TENOR_DAYS,
        gap_basis: str = config.OPTIONS_MULTIPLES_GAP_BASIS,
        multiples_only: bool = True,
        weight_cap_pct: float | None = config.OPTIONS_MULTIPLES_WEIGHT_CAP_PCT,
        **kwargs,
    ):
        super().__init__(
            entry_threshold_pct=entry_threshold_pct, max_positions=max_positions,
            tenor_days=tenor_days, gap_basis=gap_basis, multiples_only=multiples_only,
            weight_cap_pct=weight_cap_pct, **kwargs,
        )
        if gap_basis not in ("theoretical", "close"):
            raise ValueError(f"gap_basis attend 'theoretical' ou 'close', reçu {gap_basis!r}.")
        self.entry_threshold_pct = entry_threshold_pct
        self.max_positions = max_positions
        self.tenor_days = int(tenor_days)
        self.gap_basis = gap_basis
        self.multiples_only = bool(multiples_only)
        # 0 ou négatif -> pas de plafond (équivaut à None), pour pouvoir le
        # désactiver depuis la ligne de commande, qui ne transmet que des nombres.
        self.weight_cap_pct = None if not weight_cap_pct or weight_cap_pct <= 0 else float(weight_cap_pct)

    # ------------------------------------------------------------------ #
    def _candidates(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Entreprises dont l'écart justifie une position, sans plafonnement
        du nombre de positions (celui-ci est appliqué par les cibles, pas par
        l'éligibilité -- voir OptionsStrategy.eligible_directions)."""
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
        base = theoretical if self.gap_basis == "theoretical" else close
        df = df.assign(_gap=(theoretical - close) / base * 100)
        df = df.assign(_abs_gap=df["_gap"].abs())
        # Poids plafonné : en base "theoretical", l'écart est BORNÉ à +100% du
        # côté sous-évalué (le cours ne peut pas descendre sous zéro) mais NON
        # BORNÉ du côté survalorisé -- une valeur théorique proche de zéro
        # donne un écart de plusieurs milliers de %. Sans plafond, une seule
        # ligne survalorisée capte l'essentiel du capital (mesuré : 92% du
        # portefeuille pour une théorique à 5$ contre un cours à 100$) alors
        # que le classement, lui, reste légitime -- une survalorisation
        # extrême est bien une forte conviction. On plafonne donc ce qui SERT
        # AU DIMENSIONNEMENT, sans toucher à l'ordre de sélection.
        conviction = df["_abs_gap"] if self.weight_cap_pct is None else df["_abs_gap"].clip(upper=self.weight_cap_pct)
        df = df.assign(_conviction=conviction)
        return df[df["_abs_gap"] >= self.entry_threshold_pct]

    @staticmethod
    def _direction(gap: float) -> str:
        """Théorique au-dessus du cours (écart positif) = sous-évalué = CALL ;
        en dessous = survalorisé = PUT."""
        return "CALL" if gap > 0 else "PUT"

    def generate_option_targets(self, signals: pd.DataFrame, current_positions: dict[str, str]) -> dict[str, dict]:
        candidates = self._candidates(signals)
        if candidates.empty:
            return {}

        # Classement par l'écart BRUT (conviction), dimensionnement par
        # l'écart PLAFONNÉ (cf. _candidates).
        candidates = candidates.sort_values("_abs_gap", ascending=False).head(self.max_positions)
        total_conviction = candidates["_conviction"].sum()
        if total_conviction <= 0:
            return {}

        return {
            row["symbol"]: {
                "option_type": self._direction(row["_gap"]),
                "weight": row["_conviction"] / total_conviction,
                # Strike à mi-chemin théorique/cours : c'est le moteur qui fait
                # la moyenne, avec le spot du jour d'EXÉCUTION plutôt que le
                # cours du signal (les deux diffèrent après un roulement, où le
                # signal peut dater de plusieurs mois).
                "strike_reference_price": float(row["valuation_theoretical_per_share"]),
                "tenor_days": self.tenor_days,
            }
            for _, row in candidates.iterrows()
        }

    def eligible_directions(self, signals: pd.DataFrame) -> dict[str, str]:
        candidates = self._candidates(signals)
        if candidates.empty:
            return {}
        return {row["symbol"]: self._direction(row["_gap"]) for _, row in candidates.iterrows()}
