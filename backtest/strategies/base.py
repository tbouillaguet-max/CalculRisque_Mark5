"""
Interface commune à toutes les stratégies de backtest, + registre par nom
pour pouvoir en ajouter une nouvelle sans toucher à 09_backtest.py ni à
engine.py : il suffit de créer un fichier dans backtest/strategies/, y
définir une classe héritant de Strategy et décorée par @register_strategy,
puis de l'importer dans backtest/strategies/__init__.py.

Séparation des responsabilités (important pour comprendre pourquoi une
stratégie n'a PAS à gérer stop-loss/take-profit elle-même) :
    - La STRATÉGIE dit uniquement "parmi les entreprises actuellement
      éligibles, lesquelles acheter et avec quel poids relatif ?"
      (generate_target_weights).
    - Le MOTEUR (engine.py) gère tout le reste : dimensionnement réel du
      capital disponible, stop-loss/take-profit, coûts de transaction,
      positions "gelées" (une position ouverte n'est jamais liquidée juste
      parce qu'elle sort du panier éligible -- seul un stop-loss/take-profit
      ou une disparition des données de prix ferme une position, cf.
      engine.py). Le nombre de positions simultanées n'est PAS plafonné :
      toutes les candidates retenues par la stratégie sont ouvertes.
Ça permet à une nouvelle stratégie de ne se soucier que du signal, pas de la
gestion du risque ni de l'exécution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import config

STRATEGY_REGISTRY: dict[str, type["Strategy"]] = {}


def inflation_adjusted_gap(
    gap_pct: pd.Series,
    published_date: pd.Series,
    horizon_years: float,
    enabled: bool | None = None,
) -> pd.Series:
    """Écart de valorisation corrigé de l'inflation attendue sur l'horizon de
    convergence.

    RAISONNEMENT. L'écart g = (théorique - cours)/cours est un RATIO : à
    première vue l'inflation ne l'érode pas, et lui soustraire l'inflation
    serait faux. Mais la valeur théorique est une grandeur NOMINALE (chiffre
    d'affaires, résultats et flux futurs sont libellés en monnaie courante) :
    elle croît donc mécaniquement avec l'inflation. La convergence ne se fait
    pas vers V mais vers V x (1+pi)^T, et le mouvement NOMINAL attendu du
    titre devient :

        mouvement = (1 + g) x (1 + pi)^T - 1   ~=   g + pi x T

    L'inflation s'AJOUTE au mouvement attendu. C'est une asymétrie, pas un
    décalage uniforme, parce que les deux sens de position n'attendent pas le
    même mouvement :

        sous-évaluée (CALL) g=+20%, pi=5%, T=1 an  ->  +26%  (plus attractive)
        survalorisée (PUT)  g=-20%, pi=5%, T=1 an  ->  -16%  (moins attractive)

    Une entreprise survalorisée de 20% dans un régime à 5% d'inflation exige
    donc que le titre baisse de 20% alors que la dérive nominale le pousse à
    la hausse : sa thèse est plus fragile qu'un écart brut de -20% ne le
    laisse croire. C'est exactement le cas visé.

    NOTE pour une stratégie LONG-ONLY : n'ayant que des positions acheteuses,
    elle subit un décalage UNIFORME (+pi x T sur toutes ses candidates). Le
    classement est donc inchangé ; seul le franchissement du seuil d'entrée
    bouge. L'effet de re-classement n'existe que pour une stratégie
    directionnelle (call ET put).

    L'inflation retenue est celle CONNUE à la date de publication du signal
    (config.inflation_known_at), pas celle de l'année en cours : la moyenne
    annuelle n'est publiée qu'une fois l'année terminée."""
    if enabled is None:
        enabled = config.INFLATION_ADJUST_GAP
    if not enabled or horizon_years <= 0:
        return gap_pct

    inflation = published_date.map(config.inflation_known_at) / 100
    return ((1 + gap_pct / 100) * (1 + inflation) ** horizon_years - 1) * 100


def inflation_adjusted_log_gap(
    log_gap: pd.Series,
    published_date: pd.Series,
    horizon_years: float,
    enabled: bool | None = None,
) -> pd.Series:
    """Équivalent de inflation_adjusted_gap pour un écart exprimé en POINTS DE
    LOG (100 x ln(V/P)), tel que le produit valuation_gap_multiples_options.

    La correction ne peut pas être la même fonction : inflation_adjusted_gap
    applique `(1 + g/100) x (1+pi)^T - 1`, une formule qui suppose que
    `1 + g/100` vaut le rapport V/P. En log, ce n'est pas le cas -- l'appliquer
    telle quelle traiterait 18,23 points de log comme un écart de 18,23%.

    La correction devient ADDITIVE, ce qui est la forme naturelle en log. La
    convergence se fait vers V x (1+pi)^T, donc :

        ln( V x (1+pi)^T / P ) = ln(V/P) + T x ln(1+pi)

    Même conclusion économique que la version en pourcentage, et pour la même
    raison : le terme s'ajoute au score, ce qui rapproche du seuil une
    entreprise sous-évaluée (call) et l'en éloigne une survalorisée (put). La
    dérive nominale du titre joue contre la thèse baissière -- c'est
    précisément ce que la correction doit faire apparaître."""
    if enabled is None:
        enabled = config.INFLATION_ADJUST_GAP
    if not enabled or horizon_years <= 0:
        return log_gap

    inflation = published_date.map(config.inflation_known_at) / 100
    return log_gap + np.log1p(inflation) * horizon_years * 100


def risk_adjusted_conviction(
    conviction: pd.Series,
    realized_vol: pd.Series | None,
    exponent: float,
) -> pd.Series:
    """Conviction divisée par la volatilité réalisée, élevée à `exponent`.

    POURQUOI. Les poids ne portaient aucun terme de risque : deux entreprises
    au même écart de valorisation recevaient le même capital, que l'une bouge
    de 15% par an et l'autre de 60%. Le portefeuille concentrait donc son
    RISQUE là où la conviction n'était pas plus forte -- seulement plus
    volatile. Diviser par la volatilité est la correction standard, et à
    exposant 1 elle égalise la contribution de chaque ligne à la variance
    (parité de risque).

    Ce n'est pas un réglage ajusté aux données : à 0 comme à 1, l'exposant
    applique un raisonnement, pas un ajustement. C'est ce qui le distingue des
    autres axes de l'étude et le rend peu coûteux en degrés de liberté.

    UNE VOLATILITÉ MANQUANTE NE FAIT PAS SORTIR LA LIGNE. Un titre trop
    récemment coté n'a pas d'historique suffisant, et l'écarter pour cela
    reviendrait à faire du filtre de risque un filtre de signal -- la même
    erreur que la zone de non-négociation a déjà failli commettre sur les
    entrées neuves. La ligne garde sa conviction brute, comme si l'exposant
    valait 0 pour elle seule : c'est l'hypothèse neutre."""
    if not exponent or realized_vol is None:
        return conviction
    vol = pd.to_numeric(realized_vol, errors="coerce").reindex(conviction.index)
    # Une volatilité nulle ou négative n'a pas de sens et ferait exploser la
    # division : traitée comme manquante.
    vol = vol.where(vol > 0)
    facteur = vol.pow(exponent)
    return conviction / facteur.where(facteur.notna(), 1.0)


def capped_weights(conviction: pd.Series, cap_pct: float | None = None, max_iter: int = 20) -> pd.Series:
    """Poids proportionnels à `conviction`, aucun ne dépassant cap_pct % du
    portefeuille (config.BACKTEST_MAX_WEIGHT_PER_POSITION_PCT par défaut).

    Pondérer au prorata de l'écart de valorisation SANS plafond laisse une
    seule ligne capter la quasi-totalité du capital dès qu'un écart est
    aberrant -- et l'historique en contient (écarts à plusieurs milliers de %
    produits par une valeur théorique proche de zéro).

    L'excédent des lignes plafonnées est redistribué aux autres au prorata,
    en répétant l'opération : une simple renormalisation après écrêtage
    remonterait mécaniquement certaines lignes au-dessus du plafond.

    LA SOMME DES POIDS PEUT ÊTRE INFÉRIEURE À 1, et c'est voulu. Quand il y a
    trop peu de candidats pour que le plafond soit atteignable (moins de
    1/cap lignes), la version précédente renvoyait l'ÉQUIPONDÉRATION -- donc
    100% du portefeuille sur une seule ligne quand une seule candidate passait
    le seuil, 50% sur chacune quand il y en avait deux, pour un plafond
    pourtant demandé à 20%. Un plafond n'est pas une cible d'allocation, c'est
    une LIMITE DE RISQUE : avec une seule candidate, la bonne réponse est 20%
    investi et 80% en cash, pas tout le capital sur un titre. L'engine sait
    déjà gérer un budget partiellement alloué (il ne force jamais la somme des
    poids à 1, cf. options_engine._queue_isolated_order)."""
    cap = config.BACKTEST_MAX_WEIGHT_PER_POSITION_PCT if cap_pct is None else cap_pct
    total = conviction.sum()
    if total <= 0:
        return conviction
    weights = conviction / total
    if not cap or cap <= 0:
        return weights

    cap = cap / 100
    # Plafond inatteignable (trop peu de candidats) : chaque ligne prend le
    # plafond, et le reste du capital n'est simplement pas alloué.
    if cap * len(weights) <= 1:
        return pd.Series(cap, index=weights.index)

    # ITÉRATION EN NUMPY, pas en pandas. Le point fixe est identique -- mêmes
    # opérations, même ordre, mêmes arrondis flottants -- mais chaque tour
    # construisait auparavant trois Series intermédiaires via `.where()`.
    # Mesuré au profileur sur un run complet : `capped_weights` pesait 15,7 s
    # sur 45 s, soit 35% du temps du backtest actions, pour une fonction
    # appelée à chaque dépôt SEC (2624 jours sur 2936). C'est le plafond de
    # taille de toute étude un peu large, d'où la réécriture.
    valeurs = weights.to_numpy(dtype=float, copy=True)
    for _ in range(max_iter):
        au_dessus = valeurs > cap
        if not au_dessus.any():
            break
        excedent = float((valeurs[au_dessus] - cap).sum())
        valeurs[au_dessus] = cap
        en_dessous = ~au_dessus
        place = float(valeurs[en_dessous].sum())
        if place <= 0:
            break
        valeurs[en_dessous] += excedent * valeurs[en_dessous] / place
    return pd.Series(valeurs, index=weights.index, name=weights.name)


def top_n_candidates(candidates: pd.DataFrame, conviction: pd.Series, max_positions: int | None) -> pd.Index:
    """Index des `max_positions` meilleures convictions, ou tout l'index si le
    plafond est absent ou inatteignable.

    POURQUOI UN PLAFOND DE NOMBRE. Le moteur ne borne pas le nombre de lignes :
    toutes les candidates au-dessus du seuil sont ouvertes, et seul le plafond
    par position limite la concentration. Un portefeuille de 200 lignes n'est
    pas plus diversifié qu'un de 60 -- au-delà d'un certain point on n'ajoute
    plus que du coût de transaction et des convictions marginales, puisque les
    lignes entrent par ordre décroissant d'écart.

    Ce plafond est donc l'exact symétrique du plafond par position : l'un borne
    ce qu'une ligne peut peser, l'autre combien de lignes peuvent exister."""
    if not max_positions or max_positions <= 0 or len(candidates) <= max_positions:
        return candidates.index
    return conviction.nlargest(max_positions).index


def cap_per_sector(weights: pd.Series, sectors: pd.Series, cap_pct: float | None) -> pd.Series:
    """Plafond de poids CUMULÉ par secteur.

    Le plafond par position ne borne rien à ce niveau : vingt technos à 4%
    chacune font 80% du portefeuille sur un seul secteur sans qu'aucune ligne
    ne dépasse son plafond individuel.

    L'excédent d'un secteur plafonné n'est PAS redistribué : la somme des poids
    descend et l'engine laisse le reste en cash (il ne force jamais la somme à
    1). Redistribuer reviendrait à concentrer davantage sur les secteurs
    restants -- l'inverse du but.

    Extrait de valuation_gap_sector_neutral, où il vivait seul : le besoin
    n'avait rien de propre à la neutralité sectorielle, et `valuation_gap_dcf`
    n'avait aucun garde-fou de ce niveau."""
    if not cap_pct or cap_pct <= 0:
        return weights
    cap = cap_pct / 100
    secteur = sectors.where(sectors.notna(), "_inconnu")
    total_par_secteur = weights.groupby(secteur.values).transform("sum")
    facteur = (cap / total_par_secteur).clip(upper=1.0)
    return weights * facteur


def construire_poids(
    candidates: pd.DataFrame,
    conviction: pd.Series,
    *,
    max_weight_pct: float | None = None,
    max_positions: int | None = None,
    max_weight_per_sector_pct: float | None = None,
    vol_weight_exponent: float = 0.0,
    rank_weighting: bool = False,
) -> dict[str, float]:
    """Étapes de construction de portefeuille communes aux stratégies ACTIONS,
    dans l'ordre où elles doivent s'appliquer.

    L'ORDRE N'EST PAS ARBITRAIRE :
      1. la correction par le risque modifie la CONVICTION, donc elle doit
         précéder toute sélection -- sinon on garderait les N plus fortes
         convictions brutes puis on les repondérerait, ce qui n'est pas la même
         chose que garder les N meilleures une fois le risque pris en compte ;
      2. le plafond de NOMBRE réduit l'ensemble avant la normalisation, sans
         quoi les lignes écartées auraient déjà consommé une part du total ;
      3. le plafond par POSITION normalise et écrête ;
      4. le plafond par SECTEUR vient en dernier et ne redistribue rien --
         l'excédent va en cash.

    Factorisé ici parce que les trois stratégies actions ne diffèrent que par
    la façon d'établir leur conviction, pas par la façon de la transformer en
    portefeuille."""
    conviction = risk_adjusted_conviction(
        conviction, candidates.get("realized_vol"), vol_weight_exponent)

    if rank_weighting:
        # PONDÉRATION PAR RANG : la conviction devient la place dans le
        # classement, pas son ampleur. Un écart de 400% ne pèse alors que d'un
        # cran de plus qu'un écart de 300%, là où l'ampleur brute lui donnerait
        # 33% de capital en plus.
        #
        # L'intérêt est la robustesse aux valeurs extrêmes : une valorisation
        # erronée reste une erreur de RANG (elle passe devant), pas une erreur
        # d'AMPLEUR (elle capte tout). Le filtre de plausibilité du moteur
        # (config.BACKTEST_MAX_PLAUSIBLE_GAP_PCT) traite déjà le gros du
        # problème en amont ; le rang est la ceinture après les bretelles, et
        # il coûte l'information contenue dans l'ampleur -- d'où un réglage,
        # pas un défaut.
        conviction = conviction.rank(method="average", ascending=True)

    retenues = top_n_candidates(candidates, conviction, max_positions)
    candidates, conviction = candidates.loc[retenues], conviction.loc[retenues]
    if candidates.empty:
        return {}

    weights = capped_weights(conviction, cap_pct=max_weight_pct)
    if "sector" in candidates.columns:
        weights = cap_per_sector(weights, candidates["sector"], max_weight_per_sector_pct)
    return dict(zip(candidates["symbol"], weights))


def register_strategy(name: str):
    def decorator(cls: type["Strategy"]) -> type["Strategy"]:
        if name in STRATEGY_REGISTRY:
            raise ValueError(f"Stratégie '{name}' déjà enregistrée par {STRATEGY_REGISTRY[name]}.")
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


class Strategy(ABC):
    """params : hyperparamètres de la stratégie, exposés tels quels dans
    run_config.json (09_backtest.py) pour la reproductibilité d'un run."""

    # Table de valorisation dont cette stratégie tire son signal :
    #   "dcf"      -> dcf_historique.parquet (07), DCF seul ;
    #   "combinee" -> valorisation_combinee_historique.parquet (06b), multiples
    #                 sectoriels par année en priorité, DCF en repli.
    # Déclaré par la STRATÉGIE et non choisi par la CLI : c'est une propriété
    # de la thèse, pas une option d'exécution. Sans cet attribut, 09_backtest
    # devrait tester le nom de la stratégie en dur, et toute stratégie ajoutée
    # ensuite exigerait de le modifier -- exactement ce que le registre sert à
    # éviter.
    signal_source: str = "dcf"

    # Une candidate neuve doit-elle forcer le repesage de TOUT le portefeuille,
    # quelle que soit la zone de non-négociation ?
    #
    # True par défaut, et c'est le comportement historique : les poids sont
    # proportionnels à l'écart RAPPORTÉ À LA SOMME des écarts, donc l'arrivée
    # d'une candidate déplace réellement les cibles de toutes les lignes --
    # les repeser n'est pas un caprice. Le coupe-circuit répare aussi un défaut
    # latent : sans lui, une candidate SEULE dont la cible reste sous le seuil
    # ne serait jamais achetée (cf. engine._drift_is_material).
    #
    # Le coût de ce choix est mesuré : des dépôts SEC tombent 2624 séances sur
    # 2936, donc la zone est court-circuitée presque tous les jours et devient
    # inerte au-delà de 15 points -- élargir la bande de 15 à l'infini ne
    # change que 22 ventes sur 39044. Une stratégie qui veut vraiment moins
    # négocier doit donc lever ce coupe-circuit, et assumer de corriger le
    # défaut latent autrement.
    entree_neuve_force_repesage: bool = True

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate_target_weights(self, signals: pd.DataFrame, current_positions: set[str]) -> dict[str, float]:
        """
        signals : signaux CONNUS à la date courante (dernière valeur publiée
            par entreprise, pas seulement ceux publiés aujourd'hui),
            déjà restreints aux entreprises actuellement membres du S&P 500
            (voir data_loader.UniverseResolver). Colonnes : symbol, gap_pct,
            sector, valuation_dcf_per_share, close_at_filing, fiscal_year,
            published_date.
        current_positions : symboles actuellement en portefeuille (permet à
            une stratégie de favoriser la continuité si besoin -- non utilisé
            par ValuationGapDCFStrategy).

        Retourne {symbol: poids} pour le sous-ensemble de "candidats actifs"
        que la stratégie souhaite acheter/renforcer. Les poids sont relatifs
        entre eux (pas nécessairement normalisés à 1 : l'engine les
        renormalise si leur somme dépasse 1, et les alloue sur le capital
        RESTANT après les positions gelées -- voir docstring du module).
        Un symbole actuellement en position mais absent du résultat n'est
        PAS vendu : il devient une position "gelée" (voir engine.py).
        """
        raise NotImplementedError
