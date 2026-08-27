"""_DATA doit atteindre les workers d'un ProcessPoolExecutor, sur fork ET sur
spawn -- pas seulement sur la plateforme de développement.

Sur Windows (et macOS depuis Python 3.8), ProcessPoolExecutor n'utilise PAS
fork() : chaque worker est un interpréteur NEUF qui réimporte le module de
zéro, et un `_DATA` global rempli seulement dans le process parent y est
donc vide. `initializer=_pool_initializer, initargs=(_DATA,)` répare cet
écart -- mesuré en conditions réelles : sur Windows, sans ce correctif,
_run_combo/_run_one échouaient avec `KeyError: 'price_panel'` sur 100% des
combinaisons d'une grille.

Le symptôme était trompeur : la grille se terminait « normalement », écrivait
son CSV, et n'annonçait qu'un « toutes les combinaisons ont échoué ».

Ces tests forcent explicitement le contexte "spawn" pour être discriminants
même lancés sur Linux (où fork() masquerait le bug : _DATA hérité par
copy-on-write fonctionnerait même SANS l'initializer). Ils couvrent TOUS les
grid-search : ils partagent le même schéma « charger une fois dans le
parent, lire _DATA dans le worker », donc le même défaut."""

from __future__ import annotations

import importlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import pytest

# Les quatre scripts d'optimisation, par nom de module (les chiffres de tête
# n'en font pas des identifiants Python valides : import_module les accepte,
# une clause `import` non).
SCRIPTS = [
    "11_optimize_options_stops",
    "11b_optimize_rebalance_threshold",
    "11c_optimize_convergence_fraction",
    "11d_optimize_entry_threshold",
    "11e_optimize_strategy_param",
]

_opt = importlib.import_module("11_optimize_options_stops")


def _read_from_data(module_name: str, key: str) -> object:
    """Fonction de module (picklable) : lit `_DATA` du process WORKER --
    jamais celui qui a lancé le test. Une valeur lue avec succès prouve que
    l'initializer a bien tourné dans CE process-là."""
    import importlib as _il
    return _il.import_module(module_name)._DATA.get(key)


def _spawn_context_or_skip():
    try:
        return mp.get_context("spawn")
    except ValueError:
        pytest.skip("contexte 'spawn' indisponible sur cette plateforme")


@pytest.mark.parametrize("module_name", SCRIPTS)
def test_pool_initializer_peuple_data_sur_spawn(module_name):
    """LE test qui aurait détecté le bug avant un run Windows : sans
    l'initializer, un worker spawné ne voit jamais le _DATA du parent."""
    ctx = _spawn_context_or_skip()
    init_fn = importlib.import_module(module_name)._pool_initializer

    payload = {"price_panel": "objet-panel-factice", "signal_events": "objet-signaux-factice"}
    with ProcessPoolExecutor(
        max_workers=1, mp_context=ctx, initializer=init_fn, initargs=(payload,),
    ) as pool:
        vu = pool.submit(_read_from_data, module_name, "price_panel").result(timeout=120)

    assert vu == "objet-panel-factice", (
        "le worker spawné n'a pas reçu _DATA -- c'est exactement le "
        "KeyError: 'price_panel' observé sur Windows"
    )


@pytest.mark.parametrize("module_name", SCRIPTS)
def test_sans_initializer_le_worker_spawne_ne_voit_rien(module_name):
    """Non-régression INVERSE : documente que le bug est réel en son
    absence, pour qu'un futur refactor qui retirerait l'initializer soit
    intercepté par un échec explicite plutôt qu'un succès accidentel."""
    ctx = _spawn_context_or_skip()
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        vu = pool.submit(_read_from_data, module_name, "price_panel").result(timeout=120)
    assert vu is None


@pytest.mark.parametrize("module_name", SCRIPTS)
def test_le_pool_declare_bien_l_initializer(module_name):
    """Garde-fou statique, léger : si `main()` recrée un jour le pool sans y
    repasser l'initializer (régression de refactor), ce test échoue sans
    avoir besoin de lancer tout le grid-search."""
    import inspect
    source = inspect.getsource(importlib.import_module(module_name).main)
    assert "initializer=_pool_initializer" in source
    assert "initargs=(_DATA,)" in source


def test_rank_results_ne_plante_plus_quand_tout_a_echoue():
    """Le second traceback du rapport : `usable.empty` était vérifié APRÈS
    l'appel à rank_results(), qui plantait lui-même en tentant de trier une
    colonne "sharpe_ratio" totalement absente -- les dict de résultat d'une
    combinaison en échec s'arrêtent à {stop_loss_pct, take_profit, error},
    la boucle qui ajoute les métriques n'étant jamais atteinte."""
    import pandas as pd

    results = pd.DataFrame([
        {"stop_loss_pct": -30.0, "take_profit": 0.6, "error": "boom"},
        {"stop_loss_pct": -25.0, "take_profit": 0.8, "error": "boom"},
    ])
    ranked = _opt.rank_results(results, "sharpe_ratio", min_trades=15)
    assert ranked.empty


def test_rank_results_fonctionne_toujours_avec_des_resultats_valides():
    """Non-régression du chemin normal : le correctif ne doit pas casser le
    classement quand des combinaisons ont bien abouti."""
    import pandas as pd

    results = pd.DataFrame([
        {"stop_loss_pct": -30.0, "take_profit": 0.6, "error": None,
         "sharpe_ratio": 0.5, "num_trades": 50},
        {"stop_loss_pct": -25.0, "take_profit": 0.8, "error": None,
         "sharpe_ratio": 1.2, "num_trades": 60},
        {"stop_loss_pct": -20.0, "take_profit": 1.0, "error": "boom",
         "num_trades": None},
    ])
    ranked = _opt.rank_results(results, "sharpe_ratio", min_trades=15)
    assert len(ranked) == 2
    assert ranked.iloc[0]["sharpe_ratio"] == 1.2
