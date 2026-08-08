"""Paquet de tests.

`__init__.py` explicite (et non un simple paquet-espace de noms) : ib_insync,
qui est dans requirements.txt, installe un paquet `tests` de PREMIER NIVEAU
dans site-packages. Sans ce fichier, `from tests.options_harness import ...`
résout vers celui d'ib_insync et la collecte échoue avec un
ModuleNotFoundError trompeur.
"""
