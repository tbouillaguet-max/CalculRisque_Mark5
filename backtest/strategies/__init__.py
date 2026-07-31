"""Importer un module ici l'enregistre dans STRATEGY_REGISTRY (voir base.py)
et le rend disponible via --strategy <nom> sur 09_backtest.py.

Pour ajouter une nouvelle stratégie : créer un fichier dans ce dossier,
définir une classe héritant de Strategy et décorée par @register_strategy,
puis l'importer ci-dessous."""

from backtest.strategies.base import STRATEGY_REGISTRY, Strategy, register_strategy
from backtest.strategies.valuation_gap import ValuationGapDCFStrategy
from backtest.strategies.options_base import OPTIONS_STRATEGY_REGISTRY, OptionsStrategy, register_options_strategy
from backtest.strategies.valuation_gap_options import ValuationGapOptionsStrategy

__all__ = [
    "STRATEGY_REGISTRY", "Strategy", "register_strategy", "ValuationGapDCFStrategy",
    "OPTIONS_STRATEGY_REGISTRY", "OptionsStrategy", "register_options_strategy", "ValuationGapOptionsStrategy",
]
