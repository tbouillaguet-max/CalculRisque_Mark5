"""
Stratégie options "espérance de gain" : même signal que
valuation_gap_multiples_options, mais le STRIKE n'est plus posé a priori -- il
est choisi en maximisant le taux de croissance log-optimal (Kelly) du contrat.

Les deux stratégies existantes posent leur strike par convention :
valuation_gap_options prend l'ATM, valuation_gap_multiples_options prend le
mi-chemin entre cours et valeur théorique. Ces choix sont défendables, mais ils
ne consultent jamais la distribution des issues : à conviction égale, ils
retiennent le même contrat que le titre soit calme ou nerveux, l'échéance
proche ou lointaine. Or c'est précisément ce que la volatilité et la maturité
changent -- un écart de 30 % est presque acquis sur un titre à 60 % de vol à
deux ans, et très improbable sur un titre à 12 % à six mois.


CE QUE FAIT LA STRATÉGIE
------------------------
Pour chaque candidate retenue par le signal (héritage direct de la stratégie
multiples : mêmes filtres, même écart en log corrigé de l'inflation, mêmes
poids plafonnés), elle :

    1. traduit la thèse de valorisation en une DÉRIVE annualisée
           mu = fraction x ln(V / S0) / T
    2. balaie une grille de strikes candidats adaptative en sigma x racine(T),
       restreinte aux strikes RÉELLEMENT COTÉS si un snapshot est disponible,
    3. retient celui qui maximise g* = max_f E[log(1 + f x R)], où
       R = payoff/prime - 1 est le rendement du contrat,
    4. n'ouvre RIEN si aucun candidat n'a d'espérance nette positive.

Les formules sont dans backtest/expected_value.py, module de maths pur et
testé isolément (tests/test_expected_value.py).


POURQUOI KELLY, ET PAS UN RATIO GAIN/RISQUE
--------------------------------------------
C'est le point le plus important, et il a été MESURÉ avant d'être retenu.

Un critère en ratio -- Sharpe du contrat, ou Sortino sur le risque baissier --
a un optimum DÉGÉNÉRÉ : il ne choisit jamais un strike intérieur, il se colle à
un bord de grille. La raison est commune aux deux, et structurelle : un ratio
est invariant d'échelle, or dans la queue de distribution le rapport
espérance/risque est gouverné par le rapport de vraisemblance entre mesure
physique et mesure risque-neutre, qui est monotone en K.

    - Sharpe (écart-type symétrique) : très dans la monnaie, le payoff vaut
      S_T - K, donc son écart-type est celui du sous-jacent, CONSTANT en K,
      pendant que l'espérance nette croît quand K baisse. Mesuré : optimum
      collé à la borne basse pour toute grille de +/-2 à +/-8 sigma, quelle que
      soit la dérive, et convergeant vers le Sharpe de l'action seule. Le
      critère dit « n'achète pas d'option » -- cohérent, mais il ne
      sélectionne rien.
    - Sortino (semi-écart-type sous le seuil de rentabilité) : le risque
      baissier est PLAFONNÉ PAR LA PRIME, donc le ratio se comporte comme
      espérance/prime et diverge vers le hors-la-monnaie. Mesuré : ratio de
      20,3 sur un contrat qui ne paie que dans 0,1 % des scénarios.

Un plancher de probabilité de gain ne corrige pas cela : il colle l'optimum à
la contrainte, et le fait BASCULER d'un extrême à l'autre (à mu = 20 %, un
plancher de 30 % choisit K = 1,53 x S0 ; un plancher de 40 % choisit
K = 0,28 x S0). Le seuil déciderait du strike.

Kelly n'a pas ce défaut, et pas par réglage. La perte totale a une probabilité
STRICTEMENT POSITIVE et log(1 - f) diverge : le critère refuse donc de charger
un contrat qui ne paie presque jamais, ce qui borne le côté hors-la-monnaie ;
le levier atteignable borne l'autre côté. Résultat mesuré : le strike optimal
se déplace CONTINÛMENT avec la conviction (à sigma = 20 % : K*/S0 = 0,43 pour
mu = 6 %, 0,99 pour mu = 20 %, 1,42 pour mu = 35 %).

Les cas où Kelly retient malgré tout un strike de bord (edge faible devant la
volatilité) ne sont pas une panne : ils DISENT quelque chose, à savoir que
l'avantage est trop mince pour payer de la convexité et qu'il vaut mieux du
delta. Le Sharpe, lui, disait cela toujours, y compris à conviction forte.


POURQUOI UNE CONVERGENCE PARTIELLE
-----------------------------------
`fraction = 1` supposerait que le cours atteint EXACTEMENT sa valeur théorique
à l'échéance, ce que rien n'étaye et qui transformerait chaque écart de
valorisation en gain certain. Le défaut de 0,5 reprend l'hypothèse déjà
implicite dans valuation_gap_multiples_options -- dont le strike à mi-chemin
suppose exactement la moitié du chemin parcourue -- mais la rend EXPLICITE,
donc mesurable et optimisable (11c_optimize_convergence_fraction.py).


POURQUOI UNE GRILLE ADAPTATIVE, ET CONTRAINTE PAR LES COTÉS
------------------------------------------------------------
La grille va de S0 x exp(-3 sigma racine(T)) à S0 x exp(+3 sigma racine(T)) :
sa largeur suit la volatilité et la maturité, donc elle couvre toujours la même
portion de la distribution du sous-jacent. Une grille en pourcentage fixe du
spot serait trop large sur un titre calme à trois mois et trop étroite sur un
titre nerveux à deux ans -- dans les deux cas, l'optimum se retrouverait sur un
bord, c'est-à-dire non optimal.

Quand un snapshot réel est disponible, la grille est restreinte aux strikes
effectivement cotés. Sans cela, l'optimisation retiendrait un strike que
personne ne cotait, et le moteur se rabattrait silencieusement sur le contrat
coté le plus proche (options_engine._select_contract n'est contraignant sur
aucun critère) : on mesurerait donc la performance d'un contrat qu'on n'a pas
choisi. Si aucun strike coté ne tombe dans la fenêtre théorique, on retombe sur
la grille théorique avec un avertissement -- mieux vaut un contrat simulé dans
la fenêtre qu'un contrat réel manifestement hors sujet.


POURQUOI L'ESPÉRANCE NÉGATIVE EST INTERDITE
--------------------------------------------
Ce n'est pas un filtre ajouté par-dessus Kelly, c'est sa CONDITION
D'EXISTENCE : la mise log-optimale est non nulle si et seulement si E[R] > 0,
c'est-à-dire E[payoff] > prime. Une ligne dont on calcule soi-même qu'elle perd
en moyenne n'a aucune raison d'être prise ; elle est écartée du portefeuille du
jour et comptée dans `dropped_expected_value_negative_count`.


LIMITE FONDAMENTALE
-------------------
Tout ce que produit cette stratégie est la traduction en dollars de `mu`, et
`mu` est ESTIMÉ -- à partir d'une valeur théorique issue de multiples
sectoriels, et d'une hypothèse de convergence partielle que rien ne garantit.
L'espérance calculée n'est pas une prédiction : c'est ce que vaudrait le
contrat SI la thèse était juste. Un `mu` faux donne une espérance fausse avec
la même précision apparente à la douzième décimale, et le raffinement du
critère de sélection n'y change rien. C'est aussi pourquoi la comparaison avec
les deux stratégies à strike ad hoc (compare_options_strategies.py) importe
autant que la théorie : elle seule dit si l'optimisation apporte quelque chose
une fois passée par un signal imparfait.
"""

from __future__ import annotations

import logging

import pandas as pd

import config
from backtest import expected_value, options_pricing
from backtest.strategies.options_base import register_options_strategy
from backtest.strategies.valuation_gap_multiples_options import (
    ValuationGapMultiplesOptionsStrategy,
)

logger = logging.getLogger("backtest.strategies.valuation_gap_expected_value_options")


@register_options_strategy("valuation_gap_expected_value_options")
class ValuationGapExpectedValueOptionsStrategy(ValuationGapMultiplesOptionsStrategy):
    """Hérite de la stratégie multiples pour n'en changer QUE le choix du
    strike.

    Tout ce qui précède ce choix -- filtrage des lignes valorisées par repli
    DCF, écart en log corrigé de l'inflation, seuils d'entrée et de sortie avec
    hystérésis, conviction plafonnée, poids plafonnés, calcul unique partagé
    entre cibles et sens éligibles -- est rigoureusement le même. C'est
    délibéré : la comparaison entre les deux stratégies ne mesure alors QUE
    l'effet de la sélection de strike, et pas un empilement de différences.
    Seul `_targets_from` est redéfini."""

    # Le strike reste rattaché à la valeur théorique : la prise de gain se fait
    # à une fraction du chemin parcouru vers elle, pas à un seuil fixe.
    targets_convergence = True

    # Le choix du strike a besoin de la volatilité et des strikes cotés, que la
    # trame de signaux ne porte pas (cf. options_base.MarketContext).
    needs_market_context = True

    engine_defaults = dict(ValuationGapMultiplesOptionsStrategy.engine_defaults)

    def __init__(
        self,
        convergence_fraction: float = config.OPTIONS_EV_CONVERGENCE_FRACTION_DEFAULT,
        strike_grid_n_sigma: float = config.OPTIONS_EV_STRIKE_GRID_N_SIGMA,
        strike_grid_step_sigma: float = config.OPTIONS_EV_STRIKE_GRID_STEP_SIGMA,
        **kwargs,
    ):
        super().__init__(
            convergence_fraction=convergence_fraction,
            strike_grid_n_sigma=strike_grid_n_sigma,
            strike_grid_step_sigma=strike_grid_step_sigma,
            **kwargs,
        )
        fraction = float(convergence_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"convergence_fraction attend une valeur dans ]0, 1], reçu {convergence_fraction!r}. "
                "0 supprimerait toute dérive (aucune thèse), au-delà de 1 supposerait un "
                "dépassement de la valeur théorique."
            )
        self.convergence_fraction = fraction
        self.strike_grid_n_sigma = float(strike_grid_n_sigma)
        self.strike_grid_step_sigma = float(strike_grid_step_sigma)

        # Compteurs agrégés, remontés dans metrics.json par le moteur (cf.
        # options_engine._strategy_diagnostics). Cumulés sur tout le run : ils
        # comptent des ÉVALUATIONS (une par candidate et par jour de
        # rebalancement), pas des positions.
        self._n_evaluations = 0
        self._n_negative_edge = 0
        self._n_implied_vol = 0
        self._n_realized_vol = 0
        self._n_fallback_vol = 0
        self._n_quoted_grid = 0
        self._n_theoretical_grid = 0
        self._logged_grid_fallback = False

    # ------------------------------------------------------------------ #
    def diagnostics(self) -> dict:
        """Compteurs exposés dans metrics.json.

        `dropped_expected_value_negative_count` est le chiffre qui dit si le
        critère MORD : à zéro, la contrainte d'espérance positive n'a jamais
        rien écarté et la stratégie se réduit à sa grille ; très élevé, le
        signal produit surtout des thèses que le marché facture déjà."""
        vol_total = self._n_implied_vol + self._n_realized_vol
        return {
            "expected_value_evaluations_count": self._n_evaluations,
            "dropped_expected_value_negative_count": self._n_negative_edge,
            "expected_value_implied_vol_count": self._n_implied_vol,
            "expected_value_realized_vol_count": self._n_realized_vol,
            "expected_value_implied_vol_pct": (
                float(self._n_implied_vol / vol_total * 100) if vol_total else None
            ),
            "expected_value_fallback_vol_count": self._n_fallback_vol,
            "expected_value_quoted_grid_count": self._n_quoted_grid,
            "expected_value_theoretical_grid_count": self._n_theoretical_grid,
        }

    # ------------------------------------------------------------------ #
    def _volatility(self, symbol: str, option_type: str, tenor_days: int) -> float:
        """Volatilité de sélection : IMPLICITE réellement cotée si disponible,
        RÉALISÉE sinon, repli partagé avec le moteur en dernier recours.

        L'implicite est préférée parce que c'est elle qui fixe le PRIX auquel
        on achètera : sélectionner sur une réalisée de 25 % un contrat que le
        marché price à 45 % d'implicite reviendrait à calculer une espérance
        sur une distribution que le vendeur ne partage pas -- et à trouver
        systématiquement des « bonnes affaires » là où le marché anticipe
        simplement plus de mouvement.

        À défaut, la réalisée est celle du REPRICING (même fenêtre, mêmes
        bornes que options_engine._repricing_vol) : choisir un contrat sur une
        volatilité que le repricing du lendemain ne reprendrait pas ferait
        apparaître un saut de P&L sans qu'aucun prix n'ait bougé."""
        contexte = self.market_context
        if contexte is None:
            self._n_fallback_vol += 1
            return config.OPTIONS_FALLBACK_VOL

        implicite = contexte.implied_vol(symbol, option_type, float(tenor_days))
        if implicite is not None and implicite > 0:
            self._n_implied_vol += 1
            return implicite

        realisee = contexte.realized_vol(symbol)
        if realisee is not None and realisee > 0:
            self._n_realized_vol += 1
            return realisee

        # Ni implicite cotée, ni historique assez long : on se replie sur la
        # MÊME volatilité que le moteur appliquera au pricing d'entrée
        # (config.OPTIONS_FALLBACK_VOL). Écarter la ligne serait plus prudent
        # en apparence, mais introduirait un biais de sélection vers les seuls
        # titres à long historique -- alors que le moteur, lui, sait
        # parfaitement ouvrir la position, au même prix.
        self._n_fallback_vol += 1
        return config.OPTIONS_FALLBACK_VOL

    def _strikes(
        self, symbol: str, option_type: str, spot: float, sigma: float, t_years: float,
        tenor_days: int,
    ) -> list:
        """Strikes candidats : ceux réellement cotés s'ils tombent dans la
        fenêtre théorique, la grille théorique sinon.

        Le repli est délibérément silencieux au-delà du premier avertissement :
        sur un run de plusieurs milliers de jours de bourse, l'absence de
        snapshot est la situation NORMALE sur la plus grande partie de la
        période (08_recuperation_options.py ne collecte que depuis peu), et un
        avertissement par évaluation noierait le journal."""
        theorique = expected_value.strike_grid(
            spot, sigma, t_years,
            n_sigma=self.strike_grid_n_sigma, step_sigma=self.strike_grid_step_sigma,
        )
        contexte = self.market_context
        if contexte is None or not theorique:
            self._n_theoretical_grid += 1
            return theorique

        cotes = contexte.quoted_strikes(symbol, option_type, float(tenor_days))
        # Fenêtre théorique conservée comme BORNE : un strike coté très loin de
        # la distribution plausible du sous-jacent n'est pas un candidat, c'est
        # une ligne de carnet sans rapport avec la thèse.
        dans_fenetre = [k for k in cotes if theorique[0] <= k <= theorique[-1]]
        if dans_fenetre:
            self._n_quoted_grid += 1
            return dans_fenetre

        if cotes and not self._logged_grid_fallback:
            self._logged_grid_fallback = True
            logger.warning(
                "%s : aucun des %d strikes cotés ne tombe dans la fenêtre théorique "
                "[%.2f, %.2f] -- repli sur la grille théorique. Avertissement émis une "
                "seule fois pour tout le run.",
                symbol, len(cotes), theorique[0], theorique[-1],
            )
        self._n_theoretical_grid += 1
        return theorique

    def _optimal_strike(
        self, symbol: str, option_type: str, spot: float, theoretical: float,
    ) -> float | None:
        """Strike qui maximise le taux de croissance log-optimal, ou None si la
        ligne doit être écartée."""
        tenor_days = self.tenor_days
        t_years = tenor_days / 365.0

        sigma = self._volatility(symbol, option_type, tenor_days)
        mu = expected_value.convergence_drift(spot, theoretical, t_years, self.convergence_fraction)
        if mu is None:
            return None

        strikes = self._strikes(symbol, option_type, spot, sigma, t_years, tenor_days)
        if not strikes:
            return None

        # La prime est un prix de MARCHÉ : Black-Scholes au taux sans risque et
        # au rendement de dividende sectoriel, exactement le pricing que le
        # moteur appliquera (options_engine._pricing_context). Jamais recalculée
        # sous `mu`, sinon l'espérance nette serait nulle par construction.
        contexte = self.market_context
        r, q = contexte.pricing_context(symbol) if contexte is not None else (config.RISK_FREE_RATE, 0.0)

        self._n_evaluations += 1
        resultat = expected_value.optimal_strike(
            spot, mu, sigma, t_years, option_type, strikes, r=r, q=q, criterion="kelly",
            premium_fn=lambda strike: options_pricing.bs_price(
                spot, strike, t_years, sigma, option_type, r=r, q=q),
        )
        if resultat is None:
            self._n_negative_edge += 1
            return None
        return float(resultat["strike"])

    # ------------------------------------------------------------------ #
    def _targets_from(self, candidates: pd.DataFrame) -> dict[str, dict]:
        """Mêmes candidates et mêmes poids que la stratégie multiples, mais le
        strike de chaque ligne est optimisé plutôt que posé à mi-chemin.

        LE STRIKE EST TRANSMIS EN MONEYNESS (K* / spot), pas en prix de
        référence. La première version inversait la moyenne du moteur en lui
        passant 2K* - spot ; c'était faux sur trois points, tous constatés en
        run réel :

          1. Le prix de référence est REJOUÉ TEL QUEL au roulement, des mois
             plus tard, contre un spot qui a bougé. Sur un titre qui a baissé,
             la moyenne (référence + spot) / 2 devenait négative et faisait
             échouer le logarithme de Black-Scholes en plein backtest.
          2. Quand un signal frais existait, le moteur ÉCRASAIT la référence
             par la valeur théorique (_roll_position) : le contrat renouvelé
             repartait sur le strike à mi-chemin de la stratégie multiples, et
             l'optimisation était silencieusement perdue à chaque roulement.
          3. La prise de gain par convergence lit strike_reference_price comme
             une valeur théorique (_convergence_fraction) : lui donner
             2K* - spot faisait viser 80 % du chemin vers une grandeur qui
             n'avait aucun sens économique.

        Une moneyness n'a aucun de ces défauts : elle est relative au spot,
        donc valide à toute date, et c'est déjà ainsi que la grille de
        candidats est définie (multiplicative en spot). `strike_reference_price`
        retrouve du même coup son sens documenté -- la valeur théorique -- et
        pilote correctement la prise de gain par convergence et le
        rafraîchissement au roulement.

        Le roulement reconduit la moneyness sans réoptimiser : le contrat est
        recentré sur le cours du jour, mais la volatilité et l'écart de
        valorisation ayant pu changer, ce n'est plus exactement le strike que
        Kelly choisirait aujourd'hui. C'est une approximation assumée -- le
        moteur ne sait pas redemander une optimisation en cours de route, et la
        reconduction relative est de loin plus proche du choix initial que le
        retour au mi-chemin.
        """
        base = super()._targets_from(candidates)
        if not base:
            return base

        theoriques = dict(zip(
            candidates["symbol"].to_numpy(),
            candidates["valuation_theoretical_per_share"].to_numpy(),
        ))
        spots = dict(zip(candidates["symbol"].to_numpy(), candidates["close"].to_numpy()))

        cibles: dict[str, dict] = {}
        for symbol, cible in base.items():
            spot, theoretical = float(spots[symbol]), float(theoriques[symbol])
            if spot <= 0 or theoretical <= 0:
                continue

            strike = self._optimal_strike(symbol, cible["option_type"], spot, theoretical)
            if strike is None:
                continue  # espérance nette négative, ou volatilité indisponible

            cibles[symbol] = {
                **cible,
                "strike_moneyness": strike / spot,
                # Conservée telle quelle (et NON inversée) : c'est elle que
                # lisent la prise de gain par convergence et le roulement.
                "strike_reference_price": theoretical,
            }
        return cibles
