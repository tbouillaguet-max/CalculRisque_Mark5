"""Stratégie "écart de valorisation NEUTRE AU SECTEUR" : achète les
entreprises dont l'écart DCF est large PAR RAPPORT À CELUI DE LEURS PAIRS
sectoriels à la même date, et non par rapport à un seuil absolu commun à tout
l'univers.

LE PROBLÈME QU'ELLE RÉSOUT. `config.SECTOR_DCF_PARAMS` fixe un WACC et deux
taux de croissance PAR SECTEUR, choisis à la main aujourd'hui. Ces
hypothèses ne sont pas neutres entre elles :

    Technologie                   wacc 10,0%  croissance 7%  terminal 3,0%
    Agro-alimentaire et boissons  wacc  7,0%  croissance 3%  terminal 2,0%

À flux de trésorerie identique, la techno ressort structurellement mieux
valorisée -- non parce que le marché s'y trompe davantage, mais parce que la
table le dit. Classer les candidates sur `gap_pct` brut revient donc pour
partie à classer la table de configuration, et à surpondérer en permanence
les secteurs auxquels on a prêté les hypothèses les plus généreuses. Comme
ces hypothèses ont été écrites en connaissant l'histoire boursière de
2010-2026, c'est un biais de rétrospection qui entre par la porte de service :
aucune date n'est violée, mais le CHOIX des paramètres, lui, connaît la suite.

CE QU'ELLE FAIT À LA PLACE. L'écart de chaque entreprise est mesuré en excès
de la MÉDIANE de son propre secteur, à la date courante, sur les seuls
signaux déjà publiés :

    score = gap_pct - mediane(gap_pct des pairs du secteur, connus à cette date)

Une techno n'est retenue que si elle est bon marché POUR UNE TECHNO. Ce qui
reste après cette soustraction est ce que la table de config n'explique pas --
c'est-à-dire ce qui ressemble le plus à une vraie anomalie de valorisation.

POINT-IN-TIME PAR CONSTRUCTION. La médiane porte sur `signals`, que l'engine
restreint déjà aux derniers signaux CONNUS à la date courante (voir
Strategy.generate_target_weights) : aucun dépôt futur n'y entre. Et la colonne
`sector` porte le secteur D'ÉPOQUE depuis la correction de 07 -- Alphabet est
comparée aux technos jusqu'en septembre 2018 puis aux médias, comme l'aurait
fait quelqu'un à l'époque.

GROUPE DE PAIRS. Le découpage fin de GICS "Financials"
(`config.GICS_SUB_INDUSTRY_TO_SECTEUR`) compte doublement ici : il rend au DCF
les métiers de commissions (Visa, S&P Global, Aon), et il leur donne un groupe
de pairs qui a un sens. Comparer Visa à JPMorgan n'en avait aucun.

Ni sortie sur convergence ni stop-loss/take-profit ici : l'engine les gère
uniformément pour toutes les stratégies (voir backtest/strategies/base.py).
"""

from __future__ import annotations

import pandas as pd

import config
from backtest.strategies.base import Strategy, capped_weights, inflation_adjusted_gap, register_strategy

# En dessous de ce nombre de pairs, une médiane sectorielle ne mesure plus une
# norme mais un ou deux titres. Le secteur retombe alors sur la médiane de
# TOUT l'univers connu à cette date : moins précise, mais elle reste une
# référence, là où une médiane sur deux points serait du bruit qu'on prendrait
# pour un signal. Même garde-fou que MIN_PEERS_PER_SECTOR_YEAR côté 06b.
MIN_PEERS_PER_SECTOR = 5


@register_strategy("valuation_gap_sector_neutral")
class ValuationGapSectorNeutralStrategy(Strategy):
    """`entry_threshold_pct` ne se lit PAS comme celui de
    `valuation_gap_dcf` : il porte sur l'écart À LA MÉDIANE DU SECTEUR, pas
    sur l'écart au cours. 10 points d'excès sur sa médiane sectorielle est
    une candidate nettement plus rare qu'un écart brut de 10% -- d'où un
    défaut plus bas que BACKTEST_ENTRY_THRESHOLD_PCT (20%).

    `min_absolute_gap_pct` : garde-fou de bon sens. Sans lui, un secteur
    entièrement survalorisé fournirait quand même des candidates -- les
    "moins pires" -- et la stratégie achèterait des titres qu'elle juge
    elle-même trop chers, simplement parce que leurs voisins le sont
    davantage. La neutralité sectorielle sert à CLASSER, pas à absoudre.
    """

    def __init__(
        self,
        entry_threshold_pct: float = config.BACKTEST_SECTOR_NEUTRAL_ENTRY_THRESHOLD_PCT,
        min_absolute_gap_pct: float = config.BACKTEST_SECTOR_NEUTRAL_MIN_ABSOLUTE_GAP_PCT,
        max_weight_per_sector_pct: float = config.BACKTEST_MAX_WEIGHT_PER_SECTOR_PCT,
        **kwargs,
    ):
        super().__init__(
            entry_threshold_pct=entry_threshold_pct,
            min_absolute_gap_pct=min_absolute_gap_pct,
            max_weight_per_sector_pct=max_weight_per_sector_pct,
            **kwargs,
        )
        self.entry_threshold_pct = entry_threshold_pct
        self.min_absolute_gap_pct = min_absolute_gap_pct
        self.max_weight_per_sector_pct = max_weight_per_sector_pct

    # ------------------------------------------------------------------ #
    def _sector_baseline(self, signals: pd.DataFrame) -> pd.Series:
        """Médiane de l'écart par secteur, alignée sur chaque ligne.

        La MÉDIANE et non la moyenne : l'historique contient des écarts
        aberrants à plusieurs milliers de pour cent, produits par une valeur
        théorique proche de zéro (cf. base.capped_weights). Une moyenne s'y
        déplace d'un facteur dix sur un seul titre, et la référence du secteur
        entier devient ce titre-là."""
        secteur = signals["sector"].where(signals["sector"].notna(), "_inconnu")
        par_secteur = signals.groupby(secteur)["gap_pct"]

        mediane = par_secteur.transform("median")
        # Secteurs trop peu peuplés à cette date : repli sur la médiane de
        # tout l'univers connu. `transform("size")` compte les pairs ligne à
        # ligne, sans second groupby.
        assez_de_pairs = par_secteur.transform("size") >= MIN_PEERS_PER_SECTOR
        return mediane.where(assez_de_pairs, signals["gap_pct"].median())

    def generate_target_weights(self, signals: pd.DataFrame, current_positions: set[str]) -> dict[str, float]:
        if signals.empty:
            return {}

        # Correction d'inflation appliquée AVANT la neutralisation, comme dans
        # valuation_gap_dcf. Elle est de toute façon quasi neutre ici : le
        # décalage est uniforme sur toutes les candidates d'une même date, donc
        # il disparaît presque entièrement dans la soustraction de la médiane.
        # On la garde pour que `min_absolute_gap_pct`, lui, porte bien sur un
        # écart comparable à celui de l'autre stratégie.
        signals = signals.assign(gap_pct=inflation_adjusted_gap(
            signals["gap_pct"], signals["published_date"],
            config.INFLATION_HORIZON_YEARS_STOCKS,
        ))

        excess = signals["gap_pct"] - self._sector_baseline(signals)
        candidates = signals[
            (excess >= self.entry_threshold_pct)
            & (signals["gap_pct"] >= self.min_absolute_gap_pct)
        ]
        if candidates.empty:
            return {}

        # Conviction = l'excès sectoriel, pas l'écart brut : c'est lui le
        # signal une fois la table de config neutralisée. Borné à zéro par
        # construction (les candidates ont franchi un seuil positif ou nul),
        # sauf si l'utilisateur passe un seuil négatif -- auquel cas on
        # ramène la conviction dans le positif, sans quoi capped_weights
        # diviserait par une somme qui peut s'annuler.
        conviction = excess.loc[candidates.index]
        if (conviction <= 0).any():
            conviction = conviction - conviction.min() + 1e-9

        weights = capped_weights(conviction)
        weights = self._cap_per_sector(weights, candidates["sector"])
        return dict(zip(candidates["symbol"], weights))

    # ------------------------------------------------------------------ #
    def _cap_per_sector(self, weights: pd.Series, sectors: pd.Series) -> pd.Series:
        """Plafond de poids CUMULÉ par secteur.

        Le plafond par position (`capped_weights`) ne borne rien au niveau du
        secteur : vingt technos à 4% chacune font 80% du portefeuille sur un
        seul secteur sans qu'aucune ligne ne dépasse son plafond individuel.
        Neutraliser le secteur dans le SCORE sans le borner dans l'ALLOCATION
        laisserait donc revenir par l'allocation ce qu'on vient d'écarter du
        score -- un secteur peut être massivement représenté un jour donné.

        L'excédent d'un secteur plafonné n'est PAS redistribué : la somme des
        poids descend, et l'engine laisse le reste en cash (il ne force jamais
        la somme à 1, cf. base.capped_weights). Redistribuer reviendrait à
        concentrer davantage sur les secteurs restants -- l'inverse du but."""
        cap = self.max_weight_per_sector_pct
        if not cap or cap <= 0:
            return weights

        cap = cap / 100
        secteur = sectors.where(sectors.notna(), "_inconnu")
        total_par_secteur = weights.groupby(secteur.values).transform("sum")
        # Facteur de réduction par secteur, 1 quand le plafond ne mord pas.
        facteur = (cap / total_par_secteur).clip(upper=1.0)
        return weights * facteur
