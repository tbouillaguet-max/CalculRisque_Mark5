# Rapport d'audit — code et stratégie

Audit complet du dépôt à `f8c0281` : simplicité, efficacité, conformité aux
intentions déclarées, vitesse, puis la stratégie elle-même.

**Périmètre vérifié** : 21 500 lignes, 277 tests (tous verts), profilage du
moteur d'options à échelle réaliste, et reproduction chiffrée de chaque bug
listé ci-dessous. Aucun bug n'est signalé sans un scénario qui le montre.

**Verdict en une phrase** : le pipeline de données et la discipline
point-in-time sont sérieux ; le **dimensionnement** du moteur d'options ne
l'est pas, et c'est là que passe l'essentiel de l'argent.

---

## Table des pertes, par ordre de gravité

| # | Ce qui se passe | Coût mesuré |
|---|---|---|
| **B1** | `_deploy_idle_cash` renforce les positions **perdantes** en boucle | **−56 % de NAV au lieu de −2,3 %** sur le même chemin de cours |
| **B2** | Le plafond de levier n'est pas appliqué hors de `_deploy_idle_cash` | **281 %** de delta notionnel observé pour un plafond à 100 % |
| **B3** | Le roulement remet la position à une base de taille incompatible | **2 253 → 161 contrats**, aller-retour complet payé pour rien |
| **B4** | Snapshots d'options réels lus **dans le futur** | jusqu'à **14 jours** de look-ahead sur prime, IV et greeks |
| **B5** | Ordres d'entrée abandonnés en silence, re-tentés chaque jour | position jamais prise, **aucun compteur** ne le dit |
| **B6** | `capped_weights` abandonne le plafond sous 5 candidats | **100 %** du portefeuille sur une ligne |
| **B7** | Sortie d'indice = clôture forcée étiquetée `signal_lost` | contredit la doc, et dépend d'un **autre** symbole |
| **B8** | Filtre momentum orienté par un `gap_pct` périmé | le filtre échoue **exactement** sur le cas qu'il vise |
| **B9** | Friction sur-déclarée à l'expiration | commission comptée, jamais payée |
| **B10** | Sortino surévalué | **×1,19** sur une distribution normale |

Vitesse : `_current_targets` consomme **80 %** du temps de run, dont la
**moitié est un recalcul à l'identique**.

---

# Bloc B — Bugs

## B1 — `_deploy_idle_cash` est une martingale sur les positions perdantes

**C'est de loin la première cause de perte du système.**

`config.OPTIONS_MIN_DEPLOYMENT_PCT = 25.0` demande que 25 % du NAV soit
investi **en primes**. Or une prime baisse quand la thèse échoue. Le plancher
se retrouve donc violé *précisément* quand la position perd — et le moteur
rachète. Moins l'option vaut cher, plus 1 $ achète de contrats : le
renforcement s'accélère à mesure que la thèse se dégrade.

Une position gagnante, elle, voit sa prime monter, dépasse le plancher, et
n'est jamais renforcée. Le mécanisme est donc **strictement asymétrique** :
il ne moyenne qu'à la baisse.

Reproduction — un CALL, un sous-jacent qui perd 47 %, 900 jours :

```
sous-jacent : 100,4 -> 53,2 (-47%)

nominal                    NAV   437 074 $ | friction 30 214 $ | contrats max  2 253
sans plancher de primes    NAV   976 766 $ | friction  1 593 $ | contrats max     26
```

**Le plancher de primes transforme une perte de 2,3 % en une perte de 56 %**,
sur exactement le même signal et le même chemin de cours. Il investit
866 728 $ de primes là où le dimensionnement par delta n'en demandait
43 459 $.

Évolution de la position pendant la chute (le tableau se lit de haut en bas) :

```
date        nav        invested  delta_not.%  contrats  prime
2020-06-17  893 250 $   208 160        99 %       198   10,5 $
2020-12-02  676 474 $    95 823        94 %       440    2,2 $
2021-11-03  646 104 $   217 786       168 %       189   11,5 $
2022-10-05  465 090 $    79 874       110 %       272    2,9 $
2023-03-22  354 691 $    29 636        89 %     1 074    0,3 $
```

1 074 contrats d'une option à 0,30 $. C'est un billet de loterie acheté en
volume parce qu'il est devenu bon marché.

**Correctif.** Le plancher doit porter sur ce que le moteur dimensionne
réellement, c'est-à-dire l'**exposition delta**, pas la prime décaissée. Deux
options :

1. Supprimer `OPTIONS_MIN_DEPLOYMENT_PCT` et laisser le dimensionnement par
   delta décider seul (c'est le comportement « sans plancher » ci-dessus).
2. Le conserver mais l'exprimer en delta notionnel, et interdire tout
   renforcement d'une ligne dont la prime a baissé depuis l'entrée
   (`premium < stop_reference_premium`) — un renforcement doit venir d'un
   signal, jamais d'une baisse de prix.

L'option 1 est la plus sûre : le commentaire de `config.py` justifie le
plancher par « le cash dort », mais du cash qui dort coûte 0 %, alors que ce
mécanisme coûte 54 points de NAV.

## B2 — `OPTIONS_MAX_DELTA_NOTIONAL_PCT` n'est pas un plafond de portefeuille

Le plafond de levier n'est vérifié **que dans `_deploy_idle_cash`**
(`options_engine.py:772-793`). Il n'est appliqué :

- ni dans `_open_or_resize`, le chemin de dimensionnement principal ;
- ni après coup, quand le marché a bougé.

Résultat sur le run ci-dessus : **281 % de delta notionnel observé pour un
plafond déclaré à 100 %**. Rien ne réduit jamais l'exposition.

Pire, le plafond est **structurellement incapable de mordre** sur les
positions où B1 est le plus agressif : une option très hors de la monnaie a
un delta minuscule, donc 2 253 contrats ne pèsent que ~600 k$ de notionnel
delta — sous le plafond — alors qu'ils représentent tout le capital investi.

`execution_diagnostics()` publie déjà `max_delta_notional_pct_observed` : le
dépassement est **visible dans les sorties**, mais aucun code ne le lit.

**Correctif.** Vérifier le plafond dans `_open_or_resize` (avant
`_cap_order_size`), et ajouter une passe de dé-levier quotidienne dans
`run()` qui réduit au prorata quand le notionnel dépasse le plafond.

## B3 — Le roulement remet la position à une base de taille incompatible

`_roll_position` rejoue `pos.target_dollar`, qui est une **exposition delta
notionnelle** figée à l'ouverture. Mais entre-temps `_deploy_idle_cash` a
redimensionné la position sur une **base de prime**, sans jamais mettre
`target_dollar` à jour.

Les deux bases sont incommensurables, donc le roulement ne « renouvelle » pas
la position : il la remplace par une autre, d'un ordre de grandeur différent.

Réglages de production (échéance 730 j, roulement à 270 j) :

```
roll 2021-04-08 :  2 253 ->  161 contrats
roll 2022-07-13 :    189 ->  139 contrats
```

93 % de la position liquidée au premier roulement, puis reconstruite en
**32 ordres `deploy_idle_cash`** successifs, chacun payant 2,5 % de slippage
et la commission minimum. La docstring de `_check_rolls` annonce pourtant
« à exposition inchangée ».

**Correctif.** Une seule base de dimensionnement dans tout le moteur. Si
`_deploy_idle_cash` est conservé, il doit mettre `target_dollar` à jour ; s'il
est supprimé (recommandé, cf. B1), le problème disparaît de lui-même.

## B4 — Look-ahead de 14 jours sur les snapshots d'options réels

`OptionSnapshotIndex.find` retient le snapshot **le plus proche** de `as_of`,
en regardant des deux côtés :

```python
candidates = [i for i in (insert_at - 1, insert_at) if 0 <= i < len(dates)]
best_date = min(candidates, key=lambda i: abs(dates[i] - moment))
```

`insert_at` désigne une date **postérieure** à `as_of`. Vérifié :

```
interrogé au 2020-01-10 -> snapshot retenu du 2020-01-20 (prime 20.0, IV 0.9)
  ==> 10 jours DANS LE FUTUR
```

La prime d'entrée, l'IV, le delta et le `underlying_spot` peuvent donc venir
d'une date que le moteur n'a pas encore atteinte. C'est le seul endroit du
dépôt où la discipline point-in-time — tenue partout ailleurs, jusqu'au
`filed_date` du 10-K — est rompue.

Deux conséquences distinctes :

1. **Fuite d'information.** 14 jours d'IV future, soit exactement la fenêtre
   où un choc de volatilité se produit. Le biais n'est pas centré : en entrée
   de krach, l'IV future est plus haute, la prime plus chère, et le backtest
   *sous*-estime la performance ; en sortie de krach, l'inverse. Le signe
   dépend du régime, ce qui est pire qu'un biais constant.
2. **Incohérence prime / spot.** La prime vient du snapshot, le spot de
   dimensionnement vient du jour courant. Si le titre a bougé entre les deux,
   la position est valorisée dès le lendemain par Black-Scholes au spot du
   jour : un saut de P&L fantôme apparaît le premier jour. Le même problème
   existe même à date égale, puisque `08` peut associer une **IV Alpha
   Vantage** à une **prime bid/ask IBKR** (`08_recuperation_options.py:890-935`)
   — les deux ne sont pas cohérentes entre elles.

**Correctif.** Restreindre `find` au passé (`direction="backward"`), comme le
fait déjà `05_calcul_multiples.match_price_asof` pour les cours. Et repricer
l'entrée par Black-Scholes au spot d'exécution en utilisant l'**IV** du
snapshot plutôt que sa prime — l'IV est la grandeur transportable, pas le prix.

## B5 — Ordres d'entrée abandonnés en silence, re-tentés chaque jour

`_open_or_resize` compte **six sorties anticipées** sans journalisation :
conflit de sens, `|delta| < 1e-6`, `target_contracts <= 0`, `_fee_ratio_ok`
faux, `cost < MIN_TRADE_DOLLAR`, `_cap_order_size` → 0.

Seules les deux dernières incrémentent un compteur. En particulier
`_fee_ratio_ok` — qui refuse un ordre dont les frais dépassent
`OPTIONS_MAX_FEE_PCT_OF_TRADE = 1 %` — n'apparaît **nulle part** : ni dans
`buy_orders_count` (incrémenté plus loin, dans `_affordable`), ni dans les
logs, ni dans `metrics.json`.

Reproduction — la stratégie redemande la même position **tous les jours de
bourse pendant 460 jours**, et échoue tous les jours :

```
QUEUE 2020-03-26 AAA rebalance_daily poids=1.000
  !! OPEN ÉCHOUÉ 2020-03-27 AAA cible=990 686$ spot=103.01
QUEUE 2020-03-27 AAA rebalance_daily poids=1.000
  !! OPEN ÉCHOUÉ 2020-03-30 AAA cible=990 686$ spot=102.71
  ... (répété ~200 fois)
```

Cause dans ce cas : prime trop faible → frais à 101 % de la valeur de l'ordre
→ `_fee_ratio_ok` refuse. Comportement correct **en soi**, mais le portefeuille
reste intégralement en cash et `metrics.json` ne porte aucune trace du fait
que la stratégie voulait être investie.

C'est le point le plus important de la question « où est-ce que je perds de
l'**information** » : un run peut ne rien faire du tout et le rapporter comme
un run normal.

**Correctif.** Un compteur par motif d'abandon
(`dropped_orders_by_reason: dict[str, int]`), publié dans
`execution_diagnostics()`, plus un `logger.warning` de synthèse en fin de run
quand le taux d'abandon dépasse un seuil.

## B6 — `capped_weights` abandonne le plafond sous 5 candidats

`backtest/strategies/base.py` : quand `cap × len(weights) <= 1`, la fonction
renvoie l'équipondération, ce qui **dépasse le plafond demandé** :

```
1 candidats -> poids max 100,0%  (plafond demandé 20%)
2 candidats -> poids max  50,0%  (plafond demandé 20%)
3 candidats -> poids max  33,3%  (plafond demandé 20%)
4 candidats -> poids max  25,0%  (plafond demandé 20%)
5 candidats -> poids max  20,0%  (plafond demandé 20%)
```

Le commentaire dit « plafond inatteignable, donc équipondération ». Mais le
plafond n'est pas une cible d'allocation, c'est une **limite de risque** : s'il
n'y a qu'un candidat, la bonne réponse est 20 % investi et 80 % en cash, pas
100 % sur une ligne. Combiné à B1 et B2, une seule journée à candidat unique
suffit à concentrer tout le portefeuille.

**Correctif.** Ne jamais dépasser `cap` ; laisser la somme des poids être
inférieure à 1 (l'engine gère déjà un budget partiellement alloué).

## B7 — Sortie d'indice = clôture forcée, mal étiquetée, et non déterministe

Le README affirme : « Les positions déjà ouvertes ne sont PAS clôturées de
force si l'entreprise sort de l'indice ». Avec `exit_when_signal_lost=True`
(le défaut de `valuation_gap_multiples_options`), c'est faux.

`_current_targets` filtre `eligible_signals` sur `sym in universe_today`. Un
symbole sorti de l'indice disparaît donc de `eligible_directions`, et
`_close_lost_signals` le liquide :

```
AAA sort du S&P 500 le 2020-03-25 ; BBB reste éligible.
symbol  exit_date exit_reason  contracts
   AAA 2020-03-26 signal_lost        9.0
```

Trois problèmes :

1. La position est fermée alors que son écart de valorisation est **inchangé**.
2. Le motif enregistré est `signal_lost`, ce qui **fausse le diagnostic** :
   `exits_by_reason` attribue à la stratégie une décision qui vient de
   l'indice.
3. Le comportement dépend d'un **autre symbole** : si AAA est la seule
   position et que rien d'autre n'est éligible ce jour-là, `eligible_signals`
   est vide, `_rebalance_daily` sort tôt, et AAA survit. La sortie de AAA
   dépend donc de l'éligibilité de BBB.

**Correctif.** Traiter l'appartenance à l'univers comme un filtre d'**entrée**
seulement (c'est déjà l'intention : `sym in self.positions` court-circuite les
autres filtres), et donner un motif distinct (`index_exit`) si la clôture est
malgré tout voulue.

## B8 — Le filtre momentum est orienté par un `gap_pct` périmé

`_momentum_ok(signal, today)` oriente le filtre avec `signal["gap_pct"]`,
c'est-à-dire l'écart calculé **à la date de dépôt SEC**. En mode
`daily_rebalance`, `_signal_row_for_rebalance` rafraîchit `close` mais **pas**
`gap_pct` — le filtre travaille donc sur une direction potentiellement
obsolète.

Le cas qui casse est exactement celui que le filtre existe pour attraper :
une entreprise sous-évaluée au dépôt (gap > 0), dont le titre s'envole de 40 %
et devient survalorisée. La stratégie veut désormais un PUT ; le filtre, lui,
applique encore la règle CALL (« écarter si le titre chute ») et laisse
passer. Le garde-fou « ne pas vendre à découvert une fusée » ne se déclenche
jamais dans le seul cas où il compte.

**Correctif.** Recalculer le sens depuis la ligne rafraîchie, ou faire porter
l'orientation par la stratégie (`eligible_directions`) plutôt que par
`gap_pct`.

## B9 — Friction sur-déclarée à l'expiration

Dans `_reduce_position`, une expiration hors de la monnaie donne
`proceeds = max(gross_value - commission, 0.0) = 0` : la commission n'est
**pas** payée (correct — un contrat sans valeur est abandonné). Mais
`_record_fill` reçoit quand même `commission=commission` et l'ajoute à
`total_commission` :

```
      date side  contracts  price  cash_flow  commission  slippage
2020-03-03 sell       31.0    0.0        0.0     9.34774       0.0
```

L'invariant de caisse tient (`cash final == initial + Σ cash_flow`, vérifié),
donc rien de faux dans le NAV — mais `total_friction_dollar`, qui est
justement le chiffre censé arbitrer « thèse contre friction » dans
`13_diagnostic_friction.py`, est gonflé.

Même famille : le slippage de 2,5 % est appliqué au **règlement à la valeur
intrinsèque** d'une option ITM à l'échéance. Un exercice n'a pas de fourchette
bid/ask.

**Correctif.** Ne comptabiliser dans `_record_fill` que la friction
réellement décaissée, et exempter `expiry` du slippage.

## B10 — Sortino surévalué d'environ 19 %

`metrics.py` : `sortino = excess.mean() / downside.std()` avec
`downside = daily_returns[daily_returns < 0]`.

Deux écarts à la définition : le numérateur porte sur les excès, le
dénominateur sur les rendements bruts ; et surtout `.std()` mesure la
dispersion **autour de la moyenne des négatifs**, pas autour de 0. Comme cette
moyenne est négative, l'écart est systématiquement rétréci.

```
denominateur metrics.py         : 0.007061
déviation à la baisse standard  : 0.008406
-> Sortino surévalué d'un facteur 1.19x
   ex. Sortino affiché 1.50 -> Sortino réel 1.26
```

**Correctif** : `np.sqrt((np.minimum(excess, 0.0) ** 2).mean())`.

---

# Bloc V — Vitesse

Profilage à échelle réaliste (120 symboles, 500 jours, `daily_rebalance`,
stratégie de production) : **22,6 s**.

```
ncalls  cumtime  fonction
     1   22.598  run
   490   18.095  _current_targets            <- 80 % du run
   980    9.230  _candidates                 <- 2 appels par jour, identiques
   490    7.591  eligible_directions
 65346    5.050  pandas iterrows             <- 22 % du run
  1646    1.905  _delta_notional
```

## V1 — `_candidates` est calculé exactement deux fois par jour

`_current_targets` appelle `generate_option_targets` (qui appelle
`_candidates`), puis `eligible_directions` (qui rappelle `_candidates` sur le
**même** DataFrame). 980 appels pour 490 jours.

C'est ~4,6 s sur 22,6 s, soit **20 % du temps de run en recalcul pur**.

**Correctif** : calculer `_candidates` une fois dans `_current_targets` et
passer le résultat aux deux consommateurs (ou mémoïser sur `id(signals)`).

## V2 — `iterrows` dans les deux compréhensions de dictionnaire

`valuation_gap_multiples_options.py:222` et `:240` construisent leurs
dictionnaires par `for _, row in candidates.iterrows()`. 65 346 appels, 5 s.

**Correctif** : `zip()` sur les colonnes numpy
(`candidates["symbol"].to_numpy()`, etc.). Gain quasi total sur ce poste.

## V3 — `_delta_notional` est en O(n²) dans `_deploy_idle_cash`

`_deploy_idle_cash` appelle `self._delta_notional(today)` **à l'intérieur de
la boucle sur les symboles** (`options_engine.py:806`), et `_delta_notional`
parcourt lui-même toutes les positions en appelant `bs_greeks`. Avec 8 passes
possibles, le coût est de 8·n² appels Black-Scholes par jour de bourse.

Peu visible ici (42 positions), dominant sur un portefeuille de plusieurs
centaines de lignes.

**Correctif** : tenir un accumulateur de delta notionnel mis à jour
incrémentalement à chaque renforcement, au lieu de tout recalculer.

**Gain global attendu** de V1+V2 seuls : run divisé par ~3.

---

# Bloc S — Stratégie

## S1 — Le coût structurel du portage est de ~18 % de NAV par 2 ans

Mesuré, sous-jacent quasi plat sur 500 jours, sans aucun churn
(1 renforcement) : NAV 1 000 000 $ → 814 481 $, dont seulement 6 189 $ de
friction. **Le reste est de la valeur temps.**

C'est le seuil que la thèse doit battre avant de gagner quoi que ce soit :
acheter des options à 2 ans, c'est payer ~9 % de NAV par an pour le droit
d'avoir raison. Aucune métrique du dépôt n'exprime ce seuil, alors que c'est
la première chose à comparer à l'alpha attendu du signal.

**À ajouter** : une ligne `theta_cost_pct_of_nav` dans `metrics.json`, et le
rendement du signal *sous-jacent* (le même écart joué en actions, via
`09_backtest.py`) comme point de comparaison — si la version actions ne bat
pas 9 %/an, la version options ne peut pas gagner.

## S2 — Le vega n'est pas modélisé, et le biais a un signe

C'est documenté (avertissement au lancement de `10_backtest_options.py`,
docstring de `options_pricing.py`), mais la conséquence directionnelle ne
l'est pas : la stratégie est **longue de vega des deux côtés** (elle achète
CALL *et* PUT). Elle est donc structurellement longue de volatilité.

Sur 2015-2026, la volatilité implicite a une tendance de fond baissière
ponctuée de pics. Un backtest qui ignore le vega surestime donc la
performance sur toute la période **hors chocs**, c'est-à-dire sur la majorité
des jours. L'avertissement dit « borne optimiste » — c'est correct, mais la
borne peut être très loin.

## S3 — Le WACC du DCF est gelé depuis 2010, alors que les taux ne le sont pas

`config.SECTOR_DCF_PARAMS` fixe un WACC par secteur (10 % pour la tech, 6,5 %
pour les utilities…) appliqué **identiquement de 2010 à 2026**. Or le dépôt
sait pertinemment que les taux ont bougé : `RISK_FREE_RATE_BY_YEAR` va de
0,05 % à 5,3 %, et sert déjà au pricing des options et au Sharpe.

Conséquence : la valorisation théorique ne réagit pas du tout au cycle de
taux, alors que c'est le premier déterminant d'un DCF.

- **2020-2021 (taux ~0)** : WACC de 10 % trop élevé → valeur théorique
  sous-estimée → écarts négatifs → **excès de PUT** au pire moment possible.
- **2023-2024 (taux ~5 %)** : WACC de 10 % trop bas → valeur théorique
  surestimée → **excès de CALL**.

Le signal porte donc un pari de taux non voulu, systématiquement à contretemps.
`INFLATION_ADJUST_GAP` corrige déjà une dérive nominale bien plus petite ;
celle-ci est plus grosse et n'est pas corrigée.

**Correctif** : `wacc = risk_free_rate_for(année) + prime_de_risque_sectorielle`,
en gardant les valeurs actuelles comme calibration de la prime.

## S4 — Look-ahead intra-millésime dans les médianes sectorielles

`06b` calcule les médianes par `(secteur, period_type, fiscal_year,
fiscal_quarter)` et sa docstring conclut que cela « supprime bien le
look-ahead temporel ». Cela supprime le mélange **entre** millésimes, pas le
décalage **à l'intérieur** d'un millésime.

Le multiple d'un pair est calculé sur son cours **à sa propre `filed_date`**
(`05_calcul_multiples.match_price_asof`). Les 10-K d'un même exercice
s'étalent sur ~3 mois. La médiane sectorielle utilisée pour valoriser une
entreprise qui dépose en février intègre donc les cours de ses pairs jusqu'en
avril — **jusqu'à 3 mois d'information future**, y compris les cours de
bourse.

C'est le second look-ahead du dépôt après B4, et il touche le signal lui-même
plutôt que son exécution.

**Correctif** : ne retenir dans la médiane que les pairs dont `filed_date`
est **antérieure ou égale** à celle de la ligne valorisée. Cela réduit le
nombre de pairs des premiers déposants — ce qui est la réalité de
l'information disponible à cette date.

## S5 — Aucune hystérésis entre le seuil d'entrée et le seuil de sortie

Avec `daily_rebalance=True` et `exit_when_signal_lost=True` (les deux défauts
de la stratégie principale), l'entrée et la sortie utilisent **le même seuil**,
réévalué avec le cours du jour. Une position s'ouvre à
`|écart| ≥ 18,23` et se ferme à `|écart| < 18,23`.

Le filtre ε (`OPTIONS_REBALANCE_LOG_GAP_THRESHOLD`) et
`min_resize_relative_pct` protègent tous deux le **redimensionnement**, jamais
la décision d'ouvrir/fermer. Un aller-retour coûte pourtant bien plus cher
qu'un redimensionnement : 2 × 2,5 % de slippage, deux commissions minimum, et
l'abandon de toute la valeur temps déjà payée.

Une bande de sortie plus basse que la bande d'entrée (par exemple sortie à
0,7 × seuil) est le correctif standard, et elle est cohérente avec la thèse :
une convergence de 30 % n'est pas une raison de solder un pari à 2 ans.

## S6 — Les optimiseurs sont in-sample

`11_optimize_options_stops.py` balaie 64 combinaisons (stop × take-profit) et
`11b` une grille de ε, tous deux classés par Sharpe **sur l'historique
complet**, sans découpage apprentissage/test ni walk-forward.

Les garde-fous présents sont réels et bien pensés — `--min-trades` écarte les
combinaisons trop peu tradées, et un avertissement signale un optimum en bord
de grille. Mais aucun des deux ne traite le sur-apprentissage : avec 64
combinaisons sur une seule période, le meilleur Sharpe est en grande partie
du bruit sélectionné.

**Correctif** : optimiser sur 2015-2020, rapporter la performance sur
2021-2026, et publier les deux dans le CSV. Une combinaison qui ne survit pas
à cette séparation ne doit pas être retenue.

## S7 — Une « médiane » sur 5 pairs

`MIN_PEERS_PER_SECTOR_YEAR = 5`. Le commentaire du code défend correctement le
passage de 3 à 5, mais 5 reste très faible : la valeur théorique — et donc le
strike, la direction et le poids de chaque position — repose sur la médiane de
5 observations, après écrêtage par `MULTIPLE_PLAUSIBLE_RANGE`.

Il serait utile de propager `*_n_peers` jusque dans `signals_history.parquet`
et de pondérer la conviction par la robustesse de la médiane : un écart de
30 % contre 40 pairs et un écart de 30 % contre 5 pairs ne méritent pas la
même taille de position.

## S8 — Biais de survivance (connu, non corrigé)

Documenté honnêtement en tête de `06b` : les médianes sectorielles sont
calculées sur `multiples.parquet`, qui ne contient que l'univers **actuel**.
Le sens de l'erreur est correctement identifié (médianes surestimées, donc
écarts biaisés vers le CALL).

Ce n'est pas un bug — c'est la limite la plus coûteuse à lever du dépôt, et
elle interagit avec S4 : les deux poussent la valeur théorique dans le même
sens.

---

# Bloc C — Simplicité et conformité

## C1 — `options_engine.py` : 1 909 lignes, une seule classe

Le fichier concentre le calendrier, l'exécution, le pricing, le
dimensionnement, la gestion du risque, le rebalancement, la comptabilité et
les diagnostics. Trois responsabilités en sortiraient sans effort et avec un
vrai gain de testabilité :

- **le carnet d'ordres** (`_execution_order`, `_execute_pending_orders`,
  `_affordable`, `_cap_order_size`) ;
- **la tarification du courtier** (`_order_commission`, `_commission_rate`,
  `_record_contract_volume`, `_slippage_amount`) — aujourd'hui la grille IBKR
  est répartie entre `config.py` et quatre méthodes ;
- **le dimensionnement** (`_size_contracts`, `_round_contracts`,
  `_contracts_under_delta_cap`, `_deploy_idle_cash`) — c'est précisément le
  bloc où B1, B2 et B3 cohabitent, et son isolement rendrait l'incohérence des
  deux bases de taille immédiatement visible.

## C2 — Le commentaire dérive du code

La densité de commentaire est inhabituellement élevée (souvent 3 à 5 lignes
d'explication par ligne de code) et la qualité de raisonnement y est réelle —
c'est une des forces du dépôt. Mais le commentaire y est traité comme un
journal de décisions, pas comme une description du comportement actuel, et
plusieurs docstrings décrivent maintenant une intention que le code ne tient
plus :

| Affirmation | Réalité |
|---|---|
| `_check_rolls` : « à exposition inchangée » | B3 : 2 253 → 161 contrats |
| README : sortie d'indice ne ferme pas une position | B7 : elle la ferme |
| `OPTIONS_MAX_DELTA_NOTIONAL_PCT` « plafonne l'exposition » | B2 : 281 % observé |
| `06b` : « supprime le look-ahead temporel » | S4 : ~3 mois subsistent |
| `find` : « le plus représentatif autour de as_of » | B4 : « autour » inclut le futur |

Ces cinq écarts se traiteraient au mieux par des tests qui **exécutent**
l'affirmation. `tests/options_harness.py` est excellent et rend chacun de ces
tests court — les reproductions de ce rapport font 20 lignes chacune.

## C3 — Duplication entre les deux moteurs

`engine.py` et `options_engine.py` partagent la boucle jour, la file d'ordres
en attente, la résolution d'univers, les positions gelées, le filtre 8-K, la
péremption du signal et `execution_diagnostics`. Les deux implémentations ont
déjà divergé (les stops mesurent la même chose de deux façons, les
diagnostics ne comptent pas les mêmes ordres).

Une classe de base commune pour la boucle et la file, laissant aux
sous-classes le seul pricing d'instrument, supprimerait ~300 lignes et
garantirait que les deux moteurs restent comparables.

## C4 — Points positifs, à préserver

Il faut le dire clairement, parce que cela conditionne la valeur du reste :

- **La discipline point-in-time** est réelle et rare : `filed_date` du 10-K
  plutôt que la clôture d'exercice, `close_history` borné à la date courante,
  `UniverseResolver` par spans d'appartenance, `merge_asof` en
  `direction="backward"`. B4 et S4 sont des exceptions à une règle par ailleurs
  bien tenue.
- **Les limites connues sont écrites**, pas cachées : biais de survivance,
  vega non modélisé, médianes sur les survivants. C'est le contraire de la
  pratique habituelle.
- **`PricePanel`** (numpy brut + dictionnaires d'index) et
  **`OptionSnapshotIndex`** sont bien optimisés, et pour les bonnes raisons.
- **La suite de tests** (277) couvre les correctifs passés avec un harnais
  réutilisable.
- **`13_diagnostic_friction.py`** — le plan 2×2 thèse/friction/churn — est
  exactement le bon outil ; il est simplement biaisé aujourd'hui par B9 et
  n'isole pas B1.

---

# Ordre d'attaque recommandé

Les trois premiers points valent, ensemble, plus que tout le reste.

**1. Neutraliser B1 immédiatement.** Relancer les backtests existants avec
`--min-deployment-pct 0`. C'est un changement d'une option de ligne de
commande, et il change le résultat de 54 points de NAV sur le scénario testé.
Tout chiffre de performance produit avant ce changement est à jeter.

**2. Corriger B2 et B3** — une seule base de dimensionnement, plafond de
levier appliqué au vrai chemin de code. B3 disparaît en grande partie si B1
est traité.

**3. Corriger B4 et S4** — les deux look-ahead. Ce sont eux qui déterminent
si les résultats du backtest veulent dire quelque chose.

**4. Instrumenter B5** avant tout autre réglage : tant qu'un ordre peut
disparaître sans trace, aucune optimisation de paramètre n'est interprétable.

**5. V1 + V2** — 20 lignes de code, run divisé par ~3. À faire avant les
grid-searches, pas après.

**6. S3, S5, S6** — le WACC dynamique, l'hystérésis de sortie, la séparation
apprentissage/test. Ce sont des changements de thèse, à évaluer une fois que
le moteur ne fausse plus la mesure.

**7. B6 à B10** — corrections de justesse, sans effet sur la thèse mais
nécessaires pour que les métriques disent la vérité.

Un point de méthode pour finir : les correctifs 1 à 4 vont **dégrader** les
performances affichées, parfois beaucoup. C'est le signe qu'ils marchent.
