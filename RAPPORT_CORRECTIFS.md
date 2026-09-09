# Rapport de correction — pipeline de valorisation et backtest

Un correctif = un commit. Ce document récapitule, pour chacun : ce qui a
changé, quels fichiers de données sont invalidés, et dans quel ordre les
régénérer.

**Suite de tests** : 236 tests, tous verts (`python -m pytest`). Au moins un
test par correctif du bloc A, et un banc d'essai réutilisable pour le moteur
d'options (`tests/options_harness.py`) — c'est son absence qui expliquait
qu'aucun des bugs du bloc C n'était couvert.

---

## Ordre de régénération des données

Les correctifs invalident des caches en cascade. **Ordre complet** :

```bash
export SEC_CONTACT_EMAIL="ton.adresse@exemple.fr"    # désormais obligatoire (E1)

python 04_recuperation_10k.py --force-refresh        # A3 : financials.parquet
python 04b_recuperation_10q.py --force-refresh       # A3 (mécanique partagée)
python 05_calcul_multiples.py                        # B1 : multiples.parquet
python 07_calcul_dcf.py                              # A2, B4 : dcf_historique.parquet
python 06b_calcul_valorisation_combinee.py           # valorisation_combinee_historique.parquet
python 07b_validation_qualitative.py                 # D1 : validation_qualitative.parquet (sans --resume)
python 04c_recuperation_8k.py                        # D2/D3/D4 : material_events_8k.parquet
python 09_backtest.py
python 10_backtest_options.py
```

`04 --force-refresh` est **indispensable** : le throttle par ticker
n'aurait rien retéléchargé et l'ancien schéma serait resté en place.

`07b` doit tourner **sans `--resume`** : les verdicts existants ont été rendus
sur le début des documents, pas sur les sections de risque.

---

## Bloc A — bugs qui plantent ou calculent faux

### A1 — `NameError` dans `OptionSnapshotIndex.find`

`np.zeros(width)` référençait une variable inexistante. Déclenché dès qu'un
appelant demandait un `target_strike` **sans** `target_tenor_days` et qu'un
snapshot réel existait pour ce couple (symbole, type) — exactement le cas de
`valuation_gap_multiples_options`. Remplacé par `np.zeros(hi - lo)`.

*Données invalidées* : aucune.

### A2 — la D&A n'était pas réintégrée au FCF

`calculer_fcf` implémentait `EBIT × (1−t) − CapEx − ΔBFR − Intérêts × (1−t)`
au lieu du FCFF `EBIT × (1−t) + D&A − CapEx − ΔBFR`.

Le paramètre `interets` est **supprimé**, pas branché : aucun appelant ne le
renseignait (vérifié), et dans un FCFF le coût de la dette est déjà porté par
le WACC d'actualisation — le retrancher du flux le compterait deux fois.

Quand `da` est absent, il vaut 0 **avec un avertissement de synthèse** : le
DCF est alors conservateur, et le log le dit.

*Données invalidées* : `dcf_historique.parquet`, puis
`valorisation_combinee_historique.parquet`. → relancer **07**, **06b**.

### A3 — mauvaise clé d'année dans l'extraction XBRL

`fy`/`fp` décrivent le **rapport**, pas la période mesurée : les comparatifs
FY2022 et FY2021 d'un 10-K FY2023 sortent tous avec `fy=2023`, `fp="FY"` et la
**même** date `filed`. Trois exercices tombaient sous la même clé et le
départage sur `filed` était inopérant (égalité stricte) — c'est le premier
fait rencontré dans l'ordre du JSON qui l'emportait.

Rattachement désormais sur `end` / `start` : filtre de durée 330–400 jours
pour les postes de flux, `end` seul pour les postes de bilan. Priorité entre
tags et règle « dépôt le plus récent gagne » conservées, appliquées **après**
ce filtre — et enfin opérantes.

**Deux écarts assumés par rapport à la lettre du brief**, tous deux parce que
l'appliquer littéralement cassait quelque chose :

1. **Étiquette d'exercice**. « Année de `end` » fait entrer en collision les
   calendriers 52/53 semaines : l'exercice 2022 de Johnson & Johnson se clôt
   le 1er janvier 2023 et son exercice 2023 le 31 décembre 2023 — les deux
   tomberaient sur « 2023 » et le tie-break en **supprimerait un**. La règle
   retenue est « année de `end`, moins un si la clôture tombe avant juillet » :
   deux clôtures consécutives étant séparées d'environ 365 jours, aucune
   collision n'est possible, et l'étiquette redevient celle que l'entreprise
   emploie elle-même.
2. **Page de garde**. `dei:EntityCommonStockSharesOutstanding` est daté du
   jour d'arrêté du compte d'actions, **postérieur** à la clôture décrite : le
   rattacher à l'année de son `end` l'aurait envoyé sur l'exercice suivant,
   privant de dénominateur les entreprises qui ne taguent que la version `dei`
   (V, VLO…) — 07 aurait écarté leurs lignes. Ces tags sont recalés sur la
   dernière fin d'exercice réellement observée.

La logique de durée et le tie-break, déjà écrits dans `04b`, sont factorisés
dans `sec_xbrl.py` plutôt que dupliqués.

*Tests* : fixtures `companyfacts` réduites pour AAPL, JNJ et XOM
(`tests/fixtures/`), reproduisant la structure réelle de l'API — comparatifs
partageant `fy`/`fp`/`filed`, leurres « trimestre isolé » et « cumul 9 mois ».
**Le réseau était inaccessible** dans l'environnement de rédaction
(`data.sec.gov` bloqué par le proxy) : les montants sont recopiés à la main
des comptes publiés et le script de génération le documente. Ce que le test
vérifie n'est pas leur exactitude comptable mais le **rattachement**.

*Données invalidées* : **`financials.parquet`**. → relancer
**04 `--force-refresh`**, puis **05**, **07**, **06b**.

---

## Bloc B — signal de valorisation

### B1 — cours périmé dans le repli annuel

Le repli joignait `PRICES_FILE` sur `year` : on récupérait la clôture du 31/12
de l'exercice décrit, alors que `filed_date` tombe deux à trois mois plus
tard — un look-ahead franc. Remplacé par un `merge_asof` **en arrière** sur la
date de cotation réelle : pour un 10-K déposé en mars, c'est le 31/12 de
l'année précédente. Plus ancien, mais point-in-time correct.

Une colonne `price_source` trace l'origine de chaque cours, un log donne la
répartition, et au-delà de 20 % de lignes en repli un avertissement invite à
lancer `03b`.

Atténue au passage un effet de bord d'A3 : l'étiquette d'exercice ne sert plus
de clé de jointure pour les cours.

*Données invalidées* : `multiples.parquet`. → relancer **05**, **06b**.

### B2 — médianes sectorielles sur les survivants *(documenté, non corrigé)*

Limite documentée en tête de `06b` **avec le sens de l'erreur** (les disparues
étant moins bien valorisées, les médianes historiques sont probablement
surestimées). Le nombre de pairs par (secteur, millésime) est journalisé, et
un avertissement liste les tickers radiés absents quand `01b` a tourné. La
procédure de backfill complet est dans le README.

### B3 — secteur GICS rétroactif *(documenté, non corrigé)*

Documenté aux deux points d'usage (05 et 07) et au README, avec la liste de ce
qu'il pilote : WACC et taux de croissance du DCF, multiples pertinents,
rendement du dividende du pricing, exclusion du DCF.

### B4 — financières et foncières valorisées par DCF

`config.SECTORS_SANS_DCF` (Banques, Assurance, Services financiers,
Immobilier) ; `calculer_dcf_par_entreprise` les saute avec un décompte par
secteur. Le repli multiples de `06b` est vérifié par test (une banque sans
ligne DCF garde une valorisation par le P/E sectoriel), ainsi que la
cohérence « tout secteur exclu du DCF a au moins un multiple pertinent ».

*Effet de bord à connaître* : ces entreprises disparaissent de
`dcf_historique.parquet`, donc du périmètre de `07b` et du filtre de `08`.

*Données invalidées* : `dcf_historique.parquet`. → relancer **07**, **06b**.

---

## Bloc C — moteurs de backtest

### C1 — le déploiement forcé annulait le dimensionnement par delta

`OPTIONS_MIN_DEPLOYMENT_PCT = 90` forçait 90 % de la NAV en primes ; une
option ATM à 9 mois valant ~8–12 % du spot, le levier delta effectif atteignait
8–10× la NAV.

- `config.OPTIONS_MAX_DELTA_NOTIONAL_PCT` (défaut 100 %) borne l'exposition,
  vérifié avant chaque passe **et** re-vérifié à chaque renforcement ; le
  plafond prime sur le plancher de primes.
- `OPTIONS_MIN_DEPLOYMENT_PCT` passe de **90 à 25**, avec le commentaire de
  l'arbitrage cash / theta / levier.
- `--min-deployment-pct` et `--max-delta-notional-pct` en ligne de commande.
- `equity_curve` porte `delta_notional` et `delta_notional_pct`.

**Mesuré** sur un jeu synthétique de 30 titres × 10 ans :

| réglages | delta moyen | delta max | drawdown max |
|---|---|---|---|
| avant (pas de plafond, deploy 90 %) | 205,7 % | 1194,2 % | −99,87 % |
| après (plafond 100 %, deploy 25 %) | **123,3 %** | **220,4 %** | **−68,99 %** |

**Résidu signalé** : 123 % en moyenne reste au-dessus de la cible de 100 %. Le
plafond contraint l'**ordre**, pas la position dans la durée — et surtout,
`_rebalance` déduit du budget la **valeur en primes** des positions gelées,
alors que celles-ci portent une exposition delta bien supérieure. Étendre le
plafond à `_open_or_resize` corrigerait ce résidu, mais changerait le
dimensionnement voulu par les stratégies : c'est une décision de conception,
laissée hors du périmètre demandé. Les champs `avg_delta_notional_pct` et
`max_delta_notional_pct_observed` de `metrics.json` la rendent mesurable.

### C2 — la référence du stop dérivait avec les renforts

En base `premium`, `entry_premium` était moyenné à la baisse et le seuil
descendait avec lui — le stop ne se déclenchait quasiment jamais tant qu'on
moyennait à la baisse. En base `underlying`, `entry_spot` n'était jamais mis à
jour. Les deux modes mesuraient donc deux choses différentes.

Sémantique unique : le stop mesure la perte depuis l'ouverture de la **thèse**.
`stop_reference_premium` / `stop_reference_spot` (options) et
`stop_reference_price` (actions) sont posées à la première ouverture et jamais
recalculées ; le prix de revient comptable continue d'être moyenné pour le P&L.

**Le moteur actions souffrait du même défaut sans qu'il soit listé** : corrigé
aussi, les deux moteurs partagent désormais la même règle.

### C3 — `_deploy_idle_cash` n'était appelé qu'avec des ordres en file

Sorti de `_execute_pending_orders` (qui sort tôt sur file vide) et placé dans
`run()`, **juste après** l'exécution des ordres — le renforcement s'exécutant
au prix d'ouverture, il ne doit pas être décidé après les étapes qui lisent la
clôture du jour, sous peine de look-ahead intra-journalier.

### C4 — date de clôture incohérente sur radiation

`_close_position(symbol, last_valid, "data_gap")` datait la sortie du dernier
cours connu et créditait le cash à une date **antérieure** à la journée
simulée. Aligné sur le moteur actions : valorisé à `last_valid`, **daté de
`today`**.

### C5 — sous-investissement silencieux

La règle des positions gelées n'est pas touchée (choix utilisateur assumé),
mais sa conséquence devient visible : `execution_diagnostics()` produit
`buy_orders_count`, `truncated_orders_count`, `truncated_orders_pct` et
`avg_cash_pct` dans `metrics.json`, plus un avertissement au-delà de 10 %.
Le moteur options reçoit les mêmes compteurs, plus l'exposition delta moyenne
et maximale.

*Vérifié sur le run synthétique* : l'avertissement se déclenche
(11,8 % d'ordres tronqués, 12,1 % de cash moyen).

### C6 — aucun benchmark dans les métriques

`compute_metrics` accepte `benchmark_prices` et produit
`benchmark_total_return_pct`, `benchmark_cagr_pct`, `alpha_pct`, `beta`,
`information_ratio`, `tracking_error_pct`. Source : `SPY` s'il est dans les
cours quotidiens, sinon un indice **équipondéré** de l'univers point-in-time —
le libellé retenu est stocké dans `run_config.json`, un équipondéré n'étant
pas le S&P 500 pondéré.

L'indice est réaligné sur le calendrier de l'`equity_curve` ; une série qui ne
couvre pas réellement la fenêtre est **refusée** plutôt que reportée en ligne
plate (un rendement de 0 % se lirait comme une surperformance imaginaire).

Deux corrections du même fichier :
- le Sharpe divisait la moyenne des excès par l'écart-type des rendements
  **bruts** ;
- `max_drawdown_pct not in (0, np.nan)` ne marchait que par accident de
  l'identité du singleton `np.nan`.

### C7 — trois approximations du pricing

1. **Dividendes** : `bs_price` / `bs_greeks` acceptent `q`
   (Black-Scholes-Merton). Sans lui, les calls sont surévalués et les puts
   sous-évalués — d'autant plus que le rendement est élevé, donc précisément
   sur les secteurs qu'un signal « value » sélectionne. Défaut sectoriel
   (`config.SECTOR_DIVIDEND_YIELD`), faute de donnée par titre. Parité
   call-put vérifiée par test.
2. **Taux sans risque** : `config.RISK_FREE_RATE_BY_YEAR` remplace la
   constante à 4 %, utilisée à la fois pour le pricing et pour les métriques
   (4 % appliqué à 2012 fabriquait une prime de risque négative).
3. **Vega** : non corrigé, faute de surface de volatilité historique.
   `10_backtest_options.py` l'annonce au démarrage de chaque run, en précisant
   que ses rendements sont une **borne optimiste** sur toute compression de
   volatilité. Repris au README.

**Vérifié** : `03` et `03b` demandent bien `whatToShow="TRADES"` à IBKR —
cours ajustés des splits mais **pas** des dividendes, ~2 %/an manquants.
Documenté aux deux points d'appel et au README.

---

## Bloc D — chaîne SEC

### D1 — la troncature vidait le verdict de 07b de son contenu

Les 15 000 premiers caractères d'un 10-K sont la page de garde, le sommaire et
le début de l'Item 1 (Business) ; les Items 1A, 3 et 7 — ce que le prompt
demande de repérer — se trouvent 30 000 à 80 000 caractères plus loin.

Extraction par section (regex tolérante à la casse, aux espaces insécables et
aux variantes `ITEM 1A.` / `Item 1A —` / `item 1a:`), budget **réparti entre
les sections trouvées**, **dernière** occurrence retenue (le sommaire cite les
mêmes intitulés). Repli sur le début du document, avec un champ
`extraction_mode` (`"sections"` / `"debut_document"`) remonté jusqu'au parquet
de sortie. 8-K inchangés.

*Données invalidées* : `validation_qualitative.parquet`. → relancer **07b sans
`--resume`**.

### D2 — `filings.recent` ne couvrait pas l'historique

`recent` compte ~1000 dépôts **tous formulaires confondus**, et les Form 3/4/5
des dirigeants d'une grande capitalisation en consomment plusieurs centaines
par an — `recent` ne couvrait parfois que trois à cinq ans. Les pages
`filings.files` sont désormais suivies, dédoublonnées par accession et triées.

07b distingue enfin `non_evalue_filing_introuvable`,
`non_evalue_pas_de_cle_api` et `non_evalue_reponse_invalide`, qui rendaient
tous le même `non_evalue`. Aucun n'est dans
`QUALITATIVE_GATE_EXCLUDED_VERDICTS` : une période non évaluée reste
conservée.

### D3 — une requête `submissions` par fenêtre

`04c` appelait `list_company_filings` une fois par fenêtre : 40 téléchargements
du même JSON pour une entreprise à 40 trimestres, ~20 000 requêtes pour ~500
nécessaires. `fetch_submissions(cik)` est séparée de `filter_filings(...)`,
avec cache mémoire + disque (TTL 24 h).

### D4 — gestion d'erreur trop grossière

`sec_http.py` factorise le limiteur de débit, la session persistante et la
politique de réessai (écrits pour 04, absents de 04b/04c/`sec_filings_text`) et
distingue **`SecNotFound`** (404, définitif, aucun réessai) de
**`SecUnavailable`** (échec réseau ou 5xx après épuisement). Une réponse
non-JSON est une indisponibilité, pas une absence de données.

`04c` tient trois compteurs distincts et **échoue sans rien écrire** au-delà de
`--max-failure-ratio` (10 %) : un `material_events_8k.parquet` à moitié rempli
est pire qu'absent — vide, le filtre reste sans effet et ça se voit ; à moitié
rempli, il passe pour complet.

### D5 — choix non déterministe du filing

`filings[0]` ignorait l'ordre de préférence de la tuple `forms`.
`filter_filings` trie par date puis par cet ordre ; les dates partagées par
plusieurs filings sont journalisées en debug.

### D6 — robustesse de l'extraction

`primary_document` vide détecté et journalisé (07b et 04c) ; téléchargement en
flux avec arrêt anticipé (`MAX_DOWNLOAD_BYTES`, un iXBRL fait 10–30 Mo) et
conservation de ce qui est arrivé en cas de coupure ; parser `lxml` avec repli
`html.parser`.

### D7 — robustesse de l'appel LLM

Mode JSON natif de l'API (`response_format`), retrait des délimiteurs de bloc
de code, isolation du JSON noyé dans de la prose en dernier recours, et **une**
reprise sur réponse non parsable (l'ancienne version abandonnait
immédiatement ; trois reprises coûteraient des appels payants pour rien).

---

## Bloc E — sécurité et hygiène

### E1 — adresse e-mail en dur

`SEC_CONTACT_EMAIL` était écrit dans trois fichiers. Désormais lu depuis
l'environnement **sans valeur par défaut** ; 04, 04b et 04c échouent au
démarrage avec un message actionnable. Audit du reste du dépôt : toutes les
autres clés (Mistral, AlphaVantage) sont déjà lues depuis l'environnement, et
`.gitignore` couvre `.env`.

### E2 — Wikipédia téléchargé trois fois

`_fetch_tables` mémoïsée (la page fait plus d'un mégaoctet).

### E3 — commentaire obsolète

Le commentaire de `compute_implied_valuations` décrivait un double-comptage du
cash dans 07 corrigé depuis. Mis à jour.

### E4 — commission non payée à la sortie

`max(gross_value - commission, 0.0)` sortait la position à zéro sans jamais
payer la commission : une vente était comptabilisée alors qu'elle n'avait pas
lieu. Le comportement décrit est maintenant implémenté — la position reste
**ouverte** jusqu'à l'échéance. Deux exceptions où il n'y a rien à attendre :
l'expiration (contrat abandonné sans frais) et la disparition des cours du
sous-jacent.

---

## Hors brief — colonne `fiscal_year` dupliquée (10)

Trouvé en rejouant `09` et `10` de bout en bout après coup : chaque run de
`10` émettait des dizaines de
`UserWarning: DataFrame columns are not unique`.

`build_options_signal_events` renommait `year` → `fiscal_year`
**inconditionnellement**, alors que `06b` propage les colonnes de
`multiples.parquet`, où `05_calcul_multiples.load_financials_with_periods`
pose **déjà** `fiscal_year`. Deux colonnes de même nom en sortaient, et
pandas en supprimait une **sans erreur** au premier `.to_dict("records")`
(`options_engine.py`). Son jumeau `build_signal_events` porte depuis
longtemps un commentaire sur ce piège exact et évite le renommage.

Une version antérieure de ce rapport le classait « aujourd'hui inoffensif —
`06b` produit `year` seul » : c'est faux, `05` écrit bien les deux colonnes.
Le renommage n'a désormais lieu que si `fiscal_year` est absente (caches
produits par une version de `05` antérieure au TTM). Aucun résultat n'était
faux — les deux colonnes sont égales par construction — mais la perte aurait
été silencieuse le jour où elles auraient divergé.

*Données invalidées* : aucune.

---

## Critères d'acceptation

| Critère | État |
|---|---|
| `python -m pytest` passe | ✅ 236 tests verts |
| ≥ 1 test par correctif du bloc A | ✅ A1 (7), A2 (6), A3 (24) |
| `09` et `10` s'exécutent sans exception | ✅ voir réserve ci-dessous |
| `metrics.json` contient les champs de benchmark | ✅ |
| drawdown maximal strictement > −100 % | ✅ −14,83 % (09) et −68,99 % (10) |
| README liste les biais non corrigés | ✅ section « Biais et limites connus » |
| Rapport correctif par correctif | ✅ ce document |

**Réserve sur les runs complets** : le dépôt ne contient aucun `data/`
(gitignoré, clone neuf), et l'accès réseau sortant vers `data.sec.gov` et IBKR
est bloqué dans cet environnement — **aucun run sur tes données réelles n'a
donc pu être fait**. `09` et `10` ont été exécutés de bout en bout sur un jeu
**synthétique** de 30 titres × 10 ans (2 609 jours de bourse, 1 710 signaux),
généré pour l'occasion : les deux terminent sans exception et produisent tous
les champs attendus. Les chiffres de performance cités dans ce rapport n'ont
aucune signification économique — seuls les rapports **avant/après** sur le
même jeu (levier, drawdown) sont informatifs.

À refaire chez toi après la régénération complète des données.

---

## Observations hors périmètre

Relevées en passant, **non corrigées** :

- `07b_validation_qualitative.py::load_signal_periods` ne lit que
  `DCF_HISTORY_FILE`, alors que sa docstring annonce un complément par
  `VALORISATION_COMBINEE_FILE`. Depuis B4, les financières et foncières
  sortent donc aussi du périmètre de la validation qualitative.
