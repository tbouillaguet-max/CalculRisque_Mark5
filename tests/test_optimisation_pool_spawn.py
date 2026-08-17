"""Les quatre scripts d'optimisation doivent fonctionner sous SPAWN.

Contexte. Ces scripts chargent les données une fois dans le parent, puis
comptaient sur fork() pour que les workers en héritent par copy-on-write.
C'est vrai sous Linux, FAUX sous Windows et sous macOS depuis 3.8, où le pool
démarre les process par spawn : le worker ré-importe le module à neuf, `_DATA`
y vaut {}, et CHAQUE tâche échoue en `KeyError: 'price_panel'`.

Le symptôme observé était trompeur : la grille se terminait « normalement »,
écrivait son CSV, et n'annonçait qu'un « toutes les combinaisons ont échoué ».

Ces tests vérifient le remède (_init_worker) sans toucher aux données réelles.
"""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
from pathlib import Path

import pytest

_RACINE = Path(__file__).resolve().parent.parent

SCRIPTS = [
    "11_optimize_options_stops.py",
    "11b_optimize_rebalance_threshold.py",
    "11c_optimize_convergence_fraction.py",
    "11d_optimize_entry_threshold.py",
]


def _charger(nom: str):
    spec = importlib.util.spec_from_file_location(nom.replace(".py", ""), _RACINE / nom)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _data_est_vide_a_l_import(nom: str) -> bool:
    """Exécuté DANS un process spawné : reproduit ce que voit un worker Windows."""
    return not _charger(nom)._DATA


@pytest.mark.parametrize("nom", SCRIPTS)
def test_un_worker_spawne_importe_le_module_avec_des_donnees_vides(nom):
    """La prémisse du bug. Sous spawn, le worker repart d'un module neuf : sans
    initializer, il n'a AUCUNE chance de voir le _DATA du parent."""
    with mp.get_context("spawn").Pool(1) as pool:
        assert pool.apply(_data_est_vide_a_l_import, (nom,)) is True


@pytest.mark.parametrize("nom", SCRIPTS)
def test_chaque_script_expose_un_initializer_pour_le_pool(nom):
    module = _charger(nom)
    assert callable(getattr(module, "_init_worker", None)), (
        f"{nom} n'expose pas _init_worker : son pool retombe sur l'hypothèse fork."
    )


@pytest.mark.parametrize("nom", SCRIPTS)
def test_le_pool_est_construit_avec_cet_initializer(nom):
    """Garde-fou contre une régression silencieuse : _init_worker pourrait
    exister sans être branché, et le bug reviendrait à l'identique."""
    source = (_RACINE / nom).read_text(encoding="utf-8")
    assert "initializer=_init_worker" in source, (
        f"{nom} construit son ProcessPoolExecutor sans initializer=_init_worker."
    )


@pytest.mark.parametrize("nom", SCRIPTS)
def test_l_initializer_ne_recharge_rien_quand_le_fork_a_deja_fourni_les_donnees(nom):
    """Chemin Linux : le worker hérite de _DATA, l'initializer doit être un
    no-op -- sinon on paierait un rechargement disque par worker, pour rien."""
    module = _charger(nom)
    appels = []
    module._load_data = lambda *a, **k: appels.append(1) or {"price_panel": "recharge"}
    module._DATA = {"price_panel": "herite du parent"}

    module._init_worker(None)

    assert appels == [], f"{nom} recharge les données alors qu'elles sont déjà là."
    assert module._DATA == {"price_panel": "herite du parent"}


@pytest.mark.parametrize("nom", SCRIPTS)
def test_l_initializer_charge_les_donnees_quand_le_module_est_neuf(nom):
    """Chemin Windows : _DATA vide à l'import, l'initializer doit le remplir --
    c'est ce qui empêche le KeyError: 'price_panel' sur chaque tâche."""
    module = _charger(nom)
    assert not module._DATA
    module._load_data = lambda *a, **k: {"price_panel": "charge par le worker"}

    module._init_worker(None)

    assert module._DATA == {"price_panel": "charge par le worker"}


def test_le_classement_ne_plante_pas_quand_tout_a_echoue():
    """Second défaut du même run : 11 levait un KeyError sur 'sharpe_ratio' au
    lieu du message « aucune combinaison exploitable », parce qu'un run
    intégralement en échec n'écrit AUCUNE colonne de métrique."""
    import pandas as pd

    module = _charger("11_optimize_options_stops.py")
    echecs = pd.DataFrame([
        {"stop_loss_pct": -25.0, "take_profit": 0.8, "error": "KeyError: 'price_panel'"},
        {"stop_loss_pct": -20.0, "take_profit": 1.0, "error": "KeyError: 'price_panel'"},
    ])

    classement = module.rank_results(echecs, "sharpe_ratio", min_trades=10)

    assert classement.empty
