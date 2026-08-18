"""Importer un module ici l'enregistre dans STRATEGY_REGISTRY (voir base.py)
et le rend disponible via --strategy <nom> sur 09_backtest.py.

Pour ajouter une nouvelle stratégie : créer un fichier dans ce dossier,
définir une classe héritant de Strategy et décorée par @register_strategy,
puis l'importer ci-dessous."""

from backtest.strategies.base import STRATEGY_REGISTRY, Strategy, register_strategy
from backtest.strategies.valuation_gap import ValuationGapDCFStrategy
from backtest.strategies.valuation_gap_sector_neutral import ValuationGapSectorNeutralStrategy
from backtest.strategies.options_base import OPTIONS_STRATEGY_REGISTRY, OptionsStrategy, register_options_strategy
from backtest.strategies.valuation_gap_options import ValuationGapOptionsStrategy
from backtest.strategies.valuation_gap_multiples_options import ValuationGapMultiplesOptionsStrategy
from backtest.strategies.valuation_gap_expected_value_options import ValuationGapExpectedValueOptionsStrategy

__all__ = [
    "STRATEGY_REGISTRY", "Strategy", "register_strategy", "ValuationGapDCFStrategy",
    "ValuationGapSectorNeutralStrategy",
    "OPTIONS_STRATEGY_REGISTRY", "OptionsStrategy", "register_options_strategy",
    "ValuationGapOptionsStrategy", "ValuationGapMultiplesOptionsStrategy",
    "ValuationGapExpectedValueOptionsStrategy",
]
