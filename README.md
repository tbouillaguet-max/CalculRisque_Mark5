# Rapport dynamique — pipeline options US

Dashboard Streamlit à deux pages, lu directement depuis les fichiers produits
par le pipeline (`01_build_universe.py` à `08_recuperation_options.py`). Le
rapport ne relance jamais de collecte lui-même : il ne fait que lire `./data/`.

## Raccourcis (`make`)

`make` seul liste les cibles disponibles. Les trois utiles au quotidien :

```bash
make daily        # mise à jour QUOTIDIENNE complète (la cible du cron)
make daily-fast   # cours + recalcul du signal seulement, aucun appel SEC ni LLM
make quarterly    # rafraîchissement trimestriel (10-Q, 8-K, valorisation)
```

Le `Makefile` n'ajoute aucune logique : il assemble les invocations décrites
plus bas, et chaque cible reste lançable à la main avec d'autres options
(`make -n daily` affiche la commande sans l'exécuter).

## Les données dans git, via Git LFS

`data/` est versionné. Cloner le dépôt suffit donc pour relancer un backtest ou
le dashboard, sans repasser des heures sur les API SEC et IBKR.

Ce n'était pas possible en direct : git conserve **chaque version complète**
d'un fichier binaire dans son historique, pour toujours. Un
`daily_prices.parquet` de 90 Mo réécrit à chaque run ajoute 90 Mo au `.git` à
chaque commit, et plus personne ne clone. Git LFS ne met qu'un **pointeur de
130 octets** dans l'historique ; le contenu part sur le stockage LFS du serveur
et n'est rapatrié qu'au checkout. Mesuré sur un jeu de 432 Mo : `.git/objects`
pèse 1,1 Mo.

### Mise en place

```bash
# 1. vérifier que git-lfs est là (une fois par machine)
git lfs version

# 2. diagnostiquer et chiffrer AVANT de pousser -- ne modifie rien
python setup_lfs.py

# 3. activer LFS et indexer data/
python setup_lfs.py --apply
git commit -m "Donnees du pipeline"
git push
```

Si `git lfs version` répond `'lfs' is not a git command`, il faut l'installer :

| | |
|---|---|
| **Windows** | Déjà inclus dans Git for Windows en principe. Sinon, réinstalle-le depuis [git-scm.com](https://git-scm.com/download/win) en cochant *Git LFS*, ou prends l'installeur sur [git-lfs.com](https://git-lfs.com). |
| **macOS** | `brew install git-lfs` |
| **Debian/Ubuntu** | `sudo apt install git-lfs` |

`make lfs` et `make lfs-apply` sont des raccourcis vers ces deux commandes,
pour les machines qui ont `make` — ce qui n'est **pas** le cas de Git Bash sous
Windows, d'où l'invocation directe ci-dessus. Sous Windows, `python` est aussi
le bon nom de l'interpréteur (`python3` n'existe généralement pas) ; le
`Makefile` le prend en compte avec `make PYTHON=python lfs`.

`setup_lfs.py` sans argument ne modifie rien : il vérifie que git-lfs est là,
demande **à git lui-même** (`git check-attr`) quels fichiers sont réellement
couverts par les motifs de `.gitattributes`, puis chiffre le volume par
sous-dossier et le compare au palier gratuit GitHub. Exemple de sortie :

```
data : 55 fichiers,  431.3 Mo
   430.8 Mo     47 fichiers  -> Git LFS (pointeurs dans git)
     0.5 Mo      8 fichiers  -> git en direct

Par sous-dossier
      total   dont LFS  fichiers
   169.0 Mo   169.0 Mo         7  data/options
   152.0 Mo   152.0 Mo        34  data/financials
    92.3 Mo    91.9 Mo         4  data/prices
```

### Ce qui passe par LFS, et ce qui n'y passe pas

Les motifs sont dans `.gitattributes`. Y vont les fichiers **lourds ou
binaires** : `*.parquet`, `*.xlsx`, `*.jsonl`, et le cache des index SEC
(`data/financials/sec_submissions/`, visé par son chemin pour ne pas embarquer
les petits JSON d'état qui vivent ailleurs). Tout le reste — CSV d'univers,
JSON d'état, journaux — est du petit texte que git compresse et diffe très bien
tout seul ; le mettre en LFS consommerait du quota pour rien.

Rien n'est exclu du dépôt dans un cas comme dans l'autre : c'est le **mode de
stockage** qui diffère, pas ce qui est versionné.

### Le point à surveiller : le quota

**LFS garde une copie complète de chaque version.** Ce n'est pas le premier
push qui coûte, c'est l'accumulation : un run qui réécrit 270 Mo de parquet
consomme 270 Mo de stockage LFS **de plus**, à chaque fois. Le palier gratuit
GitHub est de 1 Go de stockage et 1 Go de bande passante par mois (au-delà,
data pack à 5 $/mois pour 50 Go de chaque).

`make lfs` le dit explicitement :

```
Quota
  Le premier push consomme 430.8 Mo des 1.00 Go du palier gratuit GitHub (593.2 Mo restants).
  Chaque version compte : LFS garde une copie COMPLÈTE par commit. Les fichiers
  réécrits à chaque run pèsent 276.8 Mo, soit autant de quota consommé à chaque
  `make daily` qui les modifie.
  À ce rythme, le palier gratuit tient environ 2 run(s) après le premier push.
```

Concrètement : **ne committe pas `data/` à chaque run.** Un commit ponctuel
(après un `make bootstrap`, ou quand tu veux figer un état de référence) tient
largement dans le palier gratuit ; un commit quotidien automatique l'épuise en
quelques jours.

Si le volume ne passe pas, le rapport par sous-dossier est là pour trancher :
exclure le plus gros suffit, et il n'y a rien d'autre à faire — aucun script
n'a besoin qu'un dossier soit dans git pour tourner, le pipeline le régénère.

```bash
echo "data/options/history/" >> .gitignore
```

### Sur les autres machines

git-lfs doit y être installé **avant** le clone. Sans lui, `git clone` ne
récupère que les pointeurs, et pandas échoue à ouvrir les parquet avec une
erreur peu parlante. Si le mal est fait, installe git-lfs (tableau plus haut)
puis rattrape le contenu sans recloner :

```bash
git lfs pull
```

Et **clone le dépôt** plutôt que de télécharger le ZIP de GitHub : un ZIP ne
contient ni l'historique git ni le contenu LFS (juste les pointeurs), et ne
peut rien pousser.

```bash
git clone https://github.com/tbouillaguet-max/CalculRisque_Mark5.git
```

## Lancer un backtest sur GitHub et récupérer ses résultats

Les sorties de backtest ne sont **pas** versionnées : ce sont des résultats,
pas des entrées, `10_backtest_options.py` les reproduit à l'identique, et elles
pesaient la moitié du dépôt (474 Mo sur 915). Elles voyagent donc par
**artefact** plutôt que par commit.

**1. Lancer.** Onglet *Actions* du dépôt → *Backtest* → *Run workflow*. Trois
champs : la stratégie, la date de début, et des options en plus si besoin
(`--entry-threshold-pct 25 --stop-loss-pct -40`). Déclenchement manuel
uniquement — un backtest dure des minutes et consomme du quota, le lancer à
chaque commit n'aurait aucun sens.

**2. Rapatrier**, une fois le run terminé :

```bash
python recuperer_backtest.py            # le dernier backtest en date
python recuperer_backtest.py --liste    # ce qui est disponible
python recuperer_backtest.py --run-id 123
```

Les résultats atterrissent dans `data/backtest_options/<run_id>/` (ou
`data/backtest/` pour une stratégie actions), c'est-à-dire **là où un run local
les aurait écrits** : `python 14_audit_backtest.py`, `make audit`,
`compare_options_strategies.py` et le dashboard les lisent sans rien changer.
Un fichier déjà présent n'est jamais écrasé sans `--ecraser`.

Le script affiche aussi la fiche du run (stratégie, date de début, options,
commit) : on sait ce qui a produit ces chiffres sans retourner sur GitHub.

### Authentification

L'API des artefacts exige un jeton, même sur un dépôt public. Le script en
cherche un dans `GITHUB_TOKEN`/`GH_TOKEN`, puis via `gh auth token`. Le plus
simple est d'installer le [CLI GitHub](https://cli.github.com) et de faire
`gh auth login` une fois. Sinon, un jeton créé sur
[github.com/settings/tokens](https://github.com/settings/tokens) (portée `repo`,
ou `actions:read` pour un jeton à portée fine) :

```bash
export GITHUB_TOKEN=ghp_...      # Git Bash / macOS / Linux
setx GITHUB_TOKEN ghp_...        # Windows, puis rouvre le terminal
```

### Ce qui rend ça soutenable : le trafic LFS

Un `checkout` avec `lfs: true` rapatrierait **tout** le contenu LFS du dépôt à
chaque run — cache des index SEC compris, dont aucun backtest n'a besoin — et
chaque téléchargement compte contre le quota de bande passante LFS (1 Go/mois
sur le palier gratuit, soit deux runs). Le workflow fait donc deux choses :

- il ne rapatrie que les fichiers que `09`/`10` lisent réellement
  (`daily_prices.parquet`, l'univers, la valorisation combinée, l'historique
  DCF, les snapshots d'options, les événements 8-K) ;
- il met en cache les objets LFS entre les runs, donc un second backtest sur
  les mêmes données ne consomme rien.

Si tu ajoutes un fichier au chargement d'un backtest, pense à l'ajouter à
`DONNEES_BACKTEST` dans `.github/workflows/backtest.yml` — sinon le run partira,
tournera, et échouera sur un pointeur LFS que pandas ne sait pas ouvrir.

## Mise à jour quotidienne (`run_pipeline_daily.py`)

**Pourquoi un run quotidien alors que les comptes sont trimestriels.** Le
signal est un ÉCART entre deux grandeurs : `100 × ln(valeur théorique / cours)`.
La valeur théorique ne bouge qu'au dépôt d'un 10-Q/10-K, mais le COURS bouge
tous les jours — et la stratégie multiples est configurée en
`daily_rebalance=True` exactement pour cela. Une entreprise peut donc franchir
le seuil d'entrée, ou repasser sous le seuil de sortie, par le seul mouvement
du titre. Sans run quotidien, ces franchissements ne sont vus qu'au trimestre
suivant.

```bash
python run_pipeline_daily.py                  # run complet
python run_pipeline_daily.py --skip-options   # sans 08 (pas besoin d'IB Gateway)
python run_pipeline_daily.py --prices-only    # cours + signal, hors ligne SEC/LLM
python run_pipeline_daily.py --resume         # reprend un run interrompu
python run_pipeline_daily.py --paper-trading  # + ordres au compte paper (voir « Paper trading »)
```

Étapes, dans l'ordre : `03b` (cours, incrémental) → `04`/`04b` (dépôts SEC,
`--refresh-days 7`) → `04c` (8-K) → `05` → `06` → `07` → `06b` → `07b` → `08`.
`03b` et `05/06/07/06b` sont **requises** (sans elles le signal du jour est
absent ou incohérent avec les cours) ; `04/04b/04c/07b/08` sont des
enrichissements dont l'échec est journalisé sans arrêter le run, qui se
termine alors en statut `partial`.

**Sans `SEC_CONTACT_EMAIL`, le run le dit.** Les étapes qui interrogent la
SEC (`04`, `04b`, `04c`, `07b`, marquées `needs_sec`) sont sautées d'emblée
quand la variable manque : c'est une erreur de configuration, qu'aucun
réessai ne corrige (six minutes perdues ainsi au run du 2026-09-05). Le run
termine alors en `partial` **avec un avertissement** qui nomme ce qui n'a pas
été rafraîchi : « Signal recalculé SANS dépôts SEC frais -- non rafraîchis :
comptes annuels (10-K), comptes trimestriels (10-Q)… ». Il est écrit dans
`report.json` (clé `avertissements`), répété sur la dernière ligne du journal
(« Signal du jour : … ») et affiché en tête de la page 🩺 État du pipeline du
tableau de bord. Avant, rien ne distinguait ce run d'un run complet : le
tableau de fraîcheur, fondé sur la date de modification des fichiers,
montrait le signal « à jour » alors qu'il venait d'être recalculé sur les
comptes de la veille. Même avertissement quand une étape SEC échoue pour une
autre raison ; aucun avec `--prices-only`, qui écarte la SEC par choix. En
trimestriel, `04b` est requise : sans la variable, le run s'arrête avant de
commencer, au lieu de recalculer tout le reste sur des comptes qu'il était
chargé de rafraîchir. Couvert par `tests/test_prerequis_sec.py`.

**`07` avant `06b`, et pas l'inverse.** `06b` lit `dcf_historique.parquet`,
qu'écrit `07`, pour son repli DCF (les lignes dont le secteur a trop peu de
pairs). Les deux orchestrateurs lançaient `06b` d'abord : en quotidien, ce
repli valorisait donc les comptes du run précédent. En replay
(`--as-of-date`), c'était pire : l'espace de travail part sans DCF, `06b`
sortait sur « Fichier manquant »… avec le code 0, et le replay se déclarait
réussi sans produire **aucune** valorisation combinée. Deux défauts, donc, et
le second cachait le premier : neuf `main()` (dans `02`, `04c`, `05`, `06`,
`06b` et `07`) journalisaient une erreur puis sortaient par un `return` nu —
code 0, que l'orchestrateur, qui ne juge une étape qu'à son code de sortie,
prenait pour un succès. Ils sortent désormais en erreur.
`tests/test_ordre_et_codes_de_sortie.py` déduit les dépendances **du code**
(qui écrit, qui lit quel `config.*_FILE`) et vérifie l'ordre des trois listes
d'étapes, plutôt que de les recopier dans une liste qui vieillirait.

**Écritures atomiques et Windows.** Les fichiers de reprise
(`progress_qualitative.json` de `07b`, fichiers de progression de `04`,
`04b`, `04c` et `08`, états de suivi de `03`, `04` et `04b`) et le
`report.json` des orchestrateurs s'écrivent dans un `.tmp` qui remplace
ensuite la cible par `os.replace`. Sous Windows, ce remplacement échoue en
`PermissionError` tant qu'un autre processus tient la cible ouverte —
antivirus, indexeur, synchronisation OneDrive d'un dossier Bureau, éditeur :
c'est ce qui a fait échouer `07b` le 2026-09-05. `ecriture_atomique.remplacer`
réessaie ce seul cas, avec un délai croissant (≈ 3 s au total), et relève
l'erreur si le verrou persiste. `tests/test_ecriture_atomique.py` refuse
tout nouveau `tmp.replace(...)` qui contournerait le module.

**Reprises sans doublons.** `04c`, `07b` et `08` écrivent chaque ligne dans
leur checkpoint dès qu'elle est produite, mais ne sauvegardent la liste des
éléments traités que toutes les dix unités. Un run interrompu puis repris
(`--resume`) refait donc jusqu'à neuf unités et réécrit leurs lignes, que les
trois scripts recopiaient telles quelles dans leur fichier de sortie. Ils
relisent désormais leur checkpoint par `reprise_jsonl` : une ligne par 8-K,
par période ou par contrat, la dernière écriture gagnant. Une dernière ligne
tronquée par l'interruption est ignorée au lieu de faire planter la reprise.
La mémoire des 8-K de `04c` (`cache_8k_mistral.jsonl`) est en outre nettoyée
à chaque démarrage : les verdicts remplacés en sont retirés, sans rien changer
à ce qu'elle rend, avec ou sans clé LLM. Mesuré au 2026-09-25 sur les fichiers
du dépôt : aucun doublon, aucun fichier réécrit.

**Mode dégradé plutôt que saut.** Si IB Gateway ne répond pas, `03b` est
relancée avec `--skip-ibkr` (source Stooq) au lieu d'être sautée : sauter la
récupération des cours laisserait le signal du jour calculé sur ceux de la
veille, silencieusement. `08`, qui n'a aucune source alternative, est bien
sautée.

Toute la mécanique d'exécution (réessais avec backoff, délai par étape,
journal JSON par run sous `data/pipeline_runs/`, `--resume`, redémarrage
automatique d'IB Gateway) est celle de `run_pipeline_quarterly.py`, réutilisée
telle quelle.

Cron (jours de bourse, après la clôture US) :

```
30 22 * * 1-5  cd /chemin/vers/CalculRisque_Mark5 && python3 run_pipeline_daily.py >> logs/daily.log 2>&1
```

## Paper trading sur IB Gateway (`17_paper_trading.py`)

Pour essayer la stratégie actions en conditions réelles d'exécution, sur un
compte **paper** IBKR. Stratégie par défaut : `valuation_gap_combined_ancre`.

**Le principe : le compte paper réplique le portefeuille du backtest.**
Chaque run rejoue le moteur depuis 2015 jusqu'à la dernière clôture, avec
exactement la configuration de `09_backtest.py` : les deux scripts partagent
`backtest/construction_moteur.py`, et un test refuse qu'un réglage soit
redéclaré ailleurs. Le moteur décide à la clôture et exécute à l'ouverture
suivante ; à la fin du run, ses positions plus ses ordres en attente forment le
portefeuille qu'il détiendra demain matin. Le script le traduit en **poids du
NAV**, le rapporte au NAV du compte, et envoie la différence en ordres au
marché **à l'ouverture** (MOO) — l'hypothèse d'exécution du backtest.

Rejouer tout l'historique plutôt que tenir un état local est délibéré. Les
sorties dépendent du passé de chaque position (référence du stop figée à
l'entrée, plus haut du stop suiveur, durée de détention), et le ciblage de
volatilité lit les 60 dernières séances de NAV. Recopier cette comptabilité
dans un fichier d'état, c'est diverger du moteur au premier cas limite. Le
rejeu coûte environ 30 secondes, et un test vérifie que les cibles lues sont
exactement ce que le moteur exécute le lendemain.

```bash
python 17_paper_trading.py                  # simulation : plan affiché et journalisé, RIEN n'est envoyé
python 17_paper_trading.py --transmettre    # envoie les ordres au compte paper
python 17_paper_trading.py --hors-ligne     # sans Gateway : plan d'un premier run sur un compte vide
python run_pipeline_daily.py --paper-trading   # le run quotidien, puis l'envoi (après 06b/07b, avant 08)
```

Raccourcis : `make paper`, `make paper-transmettre`, `make paper-hors-ligne`.

**Prérequis côté IB Gateway :**
- connexion en mode **Paper Trading** (port 4002 par défaut, ou `IB_GATEWAY_PORT` du `.env`) ;
- dans *Configure > Settings > API > Settings*, « Enable ActiveX and Socket
  Clients » coché, et **« Read-Only API » décoché** : sinon IBKR refuse les ordres.

Le script se connecte avec l'identifiant client fixe 17
(`PAPER_TRADING_CLIENT_ID`) : IBKR ne laisse annuler un ordre qu'au client qui
l'a passé.

**Les trois règles de réconciliation** (`paper_trading.planifier_ordres`) :
1. Une ligne que le moteur trade demain est amenée à sa cible, avec le même
   plancher de taille que le moteur. Une liquidation (stop, prise de gain…)
   passe toujours.
2. Un écart de structure est toujours corrigé. Une action dotée d'un signal
   que le compte détient sans que le moteur la détienne est vendue ; une ligne du
   moteur absente du compte est achetée. C'est ainsi que le premier run
   construit le portefeuille, et qu'un ordre refusé ou un run manqué se
   rattrapent.
3. Une ligne détenue des deux côtés, que le moteur ne touche pas, n'est
   recalée qu'au-delà de `--tolerance-pct` (1 point de NAV). Les deux
   portefeuilles bougent avec les mêmes cours ; corriger chaque jour l'écart
   d'exécution recréerait la rotation que la stratégie ancrée supprime.

**Les garde-fous :**

| Garde-fou | Comportement |
|---|---|
| Compte réel | Refusé : IBKR numérote ses comptes papier « D… » (`DU1234567`) et ses comptes réels « U… ». |
| Pas de `--transmettre` | Simulation : le plan est affiché et journalisé, aucun ordre n'est envoyé. |
| Données périmées | Envoi refusé si la dernière clôture date de plus de 4 jours (`--max-data-age-days`). |
| Ordre démesuré | Envoi refusé si un ordre dépasse 25 % du NAV : c'est le signe d'une erreur d'échelle (devise, NAV). |
| Levier | Jamais : les achats sont ramenés au cash disponible, ventes du jour comprises, moins 1 % de marge. |
| Relance le même soir | Seuls NOS ordres encore ouverts (étiquette `calculrisque-paper`) sont annulés, puis remplacés. |
| Autres positions | Options, autre devise, titre sans signal de la stratégie (SPY compris) : jamais touchés, seulement listés. |

**Le compte doit être dédié à la stratégie** : toute action US dotée d'un
signal que le moteur ne détient pas y est vendue (règle 2).

**Compte en euros.** Un compte paper hérite de la devise de base du compte réel.
Le NAV est alors converti en dollars avec le taux publié par IBKR dans le
résumé du compte. Si ce taux manque, le script s'arrête et demande `--capital`
(montant en dollars alloué à la stratégie). `--capital` sert aussi à ne confier
à la stratégie qu'une partie du compte.

**Le journal** (`data/paper_trading/`, versionné en texte simple, hors LFS) :
`ordres.csv` (chaque ordre, simulé ou transmis, avec son statut IBKR),
`compte.csv` (NAV du compte et du moteur, exposition des deux, à chaque run)
et `dernier_run.json` (le détail complet du dernier run). Pousser ce dossier
suffit pour comparer le compte paper au backtest depuis une autre machine.

**Premier run, mesuré hors ligne** (données au 2026-09-04, compte vide de
1 M$) : le moteur vise 98 lignes pour une exposition de 69,4 %, le ciblage de
volatilité réduisant l'exposition. Le plan compte **97 achats pour 683 k$**.
La 98e ligne, NVR, cote 6 299 $ pour une cible de 4 660 $ : moins d'une action
entière, elle est signalée et laissée de côté.

**Ce que le compte paper ne reproduit pas :**
- **Le prix d'exécution.** L'ouverture réelle remplace l'ouverture simulée,
  frais IBKR réels compris.
- **Le niveau du NAV.** Seuls les poids sont répliqués. Le rendement du compte
  se compare à celui du moteur à partir de la date du premier run.
- **Les fractions d'action.** Les ordres portent sur des actions entières,
  ce qui écarte les lignes plus petites qu'une action.

**Horaire.** Les ordres MOO doivent partir avant l'ouverture : lance le script
le soir, après le run quotidien. Pour un run en séance, `--type-ordre marche`
envoie des ordres au marché immédiats. Si le run quotidien signale des dépôts
SEC non rafraîchis, le paper trading tourne quand même : il trade le signal
tel qu'il est, avertissement compris.

Cron (paper trading compris) :

```
30 22 * * 1-5  cd /chemin/vers/CalculRisque_Mark5 && python3 run_pipeline_daily.py --paper-trading >> logs/daily.log 2>&1
```

## Ce que le backtest fait payer, et ce qu'il vaut

Trois correctifs de MESURE (ils ne changent aucune thèse, ils changent ce que
les chiffres veulent dire). Les trois vont dans le même sens : retirer un
optimisme qui n'était pas voulu.

### 1. On achète à l'implicite, pas à la réalisée

Faute de surface de volatilité historique, le moteur ouvrait ses positions
simulées au prix Black-Scholes calculé sur la volatilité **réalisée** du titre.
On n'achète jamais une option à la réalisée : on l'achète à l'**implicite
cotée**. `options_pricing.quoted_implied_vol` modélise cette dernière à partir
de la réalisée, par deux constantes documentées dans `config.py` :

| Réglage | Défaut | Ce qu'il représente |
|---|---|---|
| `OPTIONS_IMPLIED_VOL_SPREAD` | `0.02` | Écart implicite − réalisée à la monnaie (prime de risque de variance) |
| `OPTIONS_VOL_SKEW_SLOPE` | `0.025` | Supplément par écart-type de log-moneyness **sous** la monnaie (skew) |

Le skew n'est appliqué **qu'aux strikes sous la monnaie**, délibérément : un
skew réel décroît aussi du côté haut, mais le reproduire face à une loi de S_T
lognormale à volatilité unique fabriquerait un edge de bord de grille sans
rapport avec la thèse (constaté en test : une thèse à +0,5 % ouvrait une
position à 22 fois le cours). La correction ne peut ainsi que rendre les
options **plus** chères, jamais moins.

Ce sont des **hypothèses, pas des mesures** — meilleures que celle qu'elles
remplacent (écart nul, skew nul), mais à calibrer sur les snapshots réels dès
qu'il y en a assez : `make slippage` et les archives de `08`. Les mettre à `0`
reproduit exactement le comportement d'avant.

### 2. Deux volatilités, pas une

La stratégie « espérance de gain » distingue désormais :

- **σ_P** (volatilité réalisée) — la loi de S_T : espérance, variance, Kelly ;
- **σ_Q** (implicite cotée, ou modélisée) — le prix payé.

Les confondre, comme avant, revenait à supposer que le titre bougera d'autant
que le marché le facture : acheter de la volatilité devenait gratuit par
construction, et la prime de risque de variance disparaissait du calcul. Les
séparer la rend visible dans l'espérance nette, si bien qu'un contrat trop cher
est écarté par la seule condition d'existence de Kelly (`E[R] > 0`) — sans
filtre ajouté.

### 3. Le Sharpe déflaté du nombre d'essais

Un grid-search classé sur un unique chemin historique retient la combinaison
qui colle le mieux à *ce* chemin. Son Sharpe est celui d'un **maximum sur N
tirages**, et le maximum de N tirages n'est pas nul même quand la vraie
performance l'est.

| Essais | Sharpe « gratuit » sur 10 ans |
|---|---|
| 8 | 0,46 |
| 16 | 0,57 |
| 64 (la grille de `11`) | 0,75 |
| 200 | 0,87 |

`metrics.json` porte maintenant `n_trials`, `sharpe_noise_floor` (le plancher
ci-dessus, annualisé) et `deflated_sharpe_ratio` (la probabilité que la
performance soit réelle, corrigée de l'asymétrie et des queues). Les cinq
optimiseurs transmettent leur taille de grille automatiquement ; pour un run
isolé qui reprend le meilleur point d'une recherche :

```bash
python 10_backtest_options.py --strategy ... --n-trials 64
python 14_audit_backtest.py     # section 3c : Sharpe affiché vs plancher de bruit
```

## Le signal de valorisation

### Agrégation des multiples sectoriels

Deux choix, autrefois implicites, désormais explicites et commutables — même
mécanique que `OPTIONS_MULTIPLES_GAP_BASIS`, pour rejouer un run ancien et pour
rendre l'A/B possible sans toucher au code.

| Réglage | Défaut | Alternative |
|---|---|---|
| `SECTOR_MULTIPLE_AGGREGATOR` | `harmonic` | `median` (historique) |
| `MULTIPLE_COMBINATION` | `tiers` | `flat` (historique) |

**Moyenne harmonique.** Un multiple est un ratio ; la grandeur qu'on veut
moyenner est son inverse, le rendement. La moyenne harmonique des P/E d'un
secteur, c'est l'inverse du rendement bénéficiaire moyen — une quantité qui a un
sens, là où la moyenne des P/E est mécaniquement tirée vers le haut. Baker &
Ruback (1999) montrent que c'est l'estimateur de variance minimale d'un ratio
sous erreur multiplicative. C'est un **raffinement**, pas un correctif : la
médiane n'était pas fausse.

Contrepartie : les bornes basses de `MULTIPLE_PLAUSIBLE_RANGE` ne sont plus à
zéro. Un multiple minuscule devient un rendement gigantesque et tire toute la
moyenne harmonique, là où la médiane l'ignorait. Les planchers retenus
(EV/EBITDA ≥ 1, P/E ≥ 1, EV/Sales ≥ 0,05) n'écartent que ce qui est presque
certainement une erreur d'extraction pour une société de l'indice.

**Hiérarchie de fiabilité.** La médiane des trois prix implicites traitait
EV/EBITDA, P/E et EV/Sales comme également informatifs. Liu, Nissim & Thomas
(2002) mesurent le contraire : les multiples de résultats dominent nettement,
les multiples de chiffre d'affaires sont les moins précis, de loin. Dans une
médiane à trois, quand EV/EBITDA et P/E divergent, c'est donc EV/Sales — le
moins fiable — qui tranchait.

Le mode `tiers` n'utilise que le **meilleur rang disponible** : les multiples de
résultats quand il y en a, EV/Sales seulement à défaut. Une hiérarchie, pas une
pondération — pondérer laisserait EV/Sales départager dès qu'il tombe entre les
deux autres. Le repli fonctionne là où il doit : une entreprise en perte n'a ni
P/E ni souvent EBITDA positif, et c'est précisément le cas où un multiple de
chiffre d'affaires est la seule valorisation possible.

### Le multiple mérité : un test avant tout branchement

`warranted_multiple.py` estime, par régression en coupe sur les fondamentaux
(marge, croissance, levier, ROIC, taille — tous déjà dans le pipeline), le
multiple que ces fondamentaux **justifient** :

```
ln(M_i) = a + b·x_i + e_i        M_mérité_i = exp(a + b·x_i)
```

L'idée (Bhojraj & Lee, 2002) : une décote sur la médiane sectorielle est le plus
souvent *méritée*, et seul le **résidu** est un candidat à la mispricing. C'est
la même chose que dit Fama-French (2015) en trouvant HML redondant une fois la
rentabilité incluse.

**Le module n'est branché nulle part, délibérément.** Avant de compliquer `06b`,
il faut savoir si le multiple mérité prédit mieux — et cela se tranche **sans
backtest** :

```bash
make merite                                  # EV/EBITDA, coupe complète
make merite MULTIPLE=P/E GROUPING=secteur
```

`15_test_multiple_merite.py` compare, **hors échantillon**, l'erreur de
prédiction des deux méthodes sur les mêmes pairs point-in-time (la ligne évaluée
est exclue de ses propres pairs, comme `06b` l'exclut déjà de sa médiane). Il
rapporte l'erreur absolue médiane en log — la convention de Liu-Nissim-Thomas —
et conclut explicitement s'il faut brancher ou non.

Le regroupement est la vraie décision : `--grouping millesime` (défaut) ajuste
sur la coupe complète de l'indice avec indicatrices sectorielles, parce qu'une
régression à cinq régresseurs demande des dizaines d'observations et qu'un
secteur × millésime en compte rarement plus de vingt. `--grouping secteur`
reste disponible et journalise ce qu'il doit abandonner.

Si le test conclut **non**, il n'y a rien à brancher — et c'est une réponse,
pas un échec.

## L'extraction SEC ne se paie qu'une fois

Les quatre scripts qui interrogent la SEC (`04`, `04b`, `04c`, `07b`) sont
tous **reprenables** : ce qui a été récupéré est écrit au fil de l'eau, une
interruption ne fait perdre que les quelques tickers en cours, et un second run
n'appelle pas la SEC pour ce qui est déjà là.

```bash
python 04_recuperation_10k.py --tickers data/universe/sp500_universe_full.csv
# interrompu ? relance la même commande avec --resume
python 04_recuperation_10k.py --tickers data/universe/sp500_universe_full.csv --resume
```

**Deux mécanismes distincts, et il faut les distinguer :**

| | Ce qu'il protège | Fichier |
|---|---|---|
| **Reprise** (`--resume`) | Un run **interrompu** : on repart des tickers non traités | `progress_*.json` + `checkpoint_*.jsonl` |
| **Throttle** (`--refresh-days`, défaut 30) | Un run **terminé** : on ne réinterroge pas ce qui est récent | `fetch_state_*.json` |

Le premier couvre le Ctrl+C et la coupure réseau ; le second évite de repayer
un run complet le lendemain. `--force-refresh` ignore le throttle,
`--resume` ignore ce qui est déjà traité dans le run en cours.

**Ce qui déclenche quand même un nouvel appel**, et c'est voulu :

- un ticker en **échec** n'est jamais marqué « à jour » — un échec réseau doit
  être réessayé au prochain run complet, pas ignoré pendant 30 jours ;
- un ticker dont l'état dit « déjà interrogé » mais dont le parquet ne contient
  rien est réinterrogé — sans quoi une ligne manquante le resterait pour
  toujours ;
- au-delà de `--refresh-days`, pour récupérer les dépôts de l'année écoulée.

Le fichier de progression est écrit **atomiquement** (temporaire puis
`replace`) et sauvegardé dans un `finally` : un Ctrl+C sauvegarde l'état réel,
pas un point de contrôle périodique dépassé. Les lignes récupérées vont dans un
JSONL *append-only*, relisible même après une interruption brutale — là où un
parquet réécrit en bloc ne l'est pas.

`04c` ajoute un troisième niveau qui lui est propre : un **cache par dépôt**
(`cache_8k_mistral.jsonl`), qui évite de retélécharger ET de reclassifier un
8-K déjà vu, même entre deux runs complets.

> **Note historique.** `04` était le seul des quatre sans reprise : il
> accumulait tout en mémoire et n'écrivait qu'à la fin, si bien qu'une
> interruption perdait l'intégralité du run — et le suivant repartait de zéro,
> l'état de suivi n'ayant jamais été sauvegardé. C'était aussi le plus long
> (~500 entreprises sur l'univers complet, à quelques requêtes par seconde) :
> le seul run qu'on ne pouvait pas se permettre de perdre était le seul qu'on
> perdait. Couvert depuis par `tests/test_reprise_10k.py`.

## Configuration requise

```bash
export SEC_CONTACT_EMAIL="ton.adresse@exemple.fr"   # obligatoire pour 04, 04b, 04c, 07b
export GEMINI_API_KEY="ta_cle"                      # optionnel : 02, 04c, 07b (LLM)
export ALPHAVANTAGE_API_KEY="ta_cle"                # optionnel : 08 --av-backfill-dates
```

`SEC_CONTACT_EMAIL` n'a **pas** de valeur par défaut : la SEC exige un
User-Agent identifiant un contact réel, et un User-Agent générique se fait
bloquer (403/429). Les scripts qui interrogent la SEC échouent au démarrage
avec un message explicite si elle est absente, plutôt que de dégrader
silencieusement. Les orchestrateurs la vérifient **avant** de lancer ces
étapes et signalent un signal recalculé sans dépôts frais (voir « Mise à
jour quotidienne »). La variable doit être visible du processus qui lance le
run : une tâche planifiée Windows ou un cron ne lisent pas le profil du shell
interactif.

### Le LLM : Gemini ou Mistral

`02`, `04c` et `07b` appellent un LLM à travers une seule fonction,
`sec_filings_text.analyser_texte_llm`. Le fournisseur se choisit par
l'environnement, sans toucher au code :

| Variable | Rôle |
|---|---|
| `GEMINI_API_KEY` | Clé Google AI Studio. Définie, elle rend Gemini prioritaire. |
| `GEMINI_MODEL` | Modèle Gemini (défaut `gemini-2.5-flash`). |
| `MISTRAL_API_KEY` | Clé Mistral, utilisée quand aucune clé Gemini n'est définie. |
| `LLM_PROVIDER` | `gemini` ou `mistral`, pour forcer le choix quand les deux clés existent. |
| `MISTRAL_REQUESTS_PER_SECOND` | Débit sortant vers le LLM, quel que soit le fournisseur (défaut 1). |

Les clés se donnent **par variable d'environnement**, jamais dans le code :
les constantes `*_API_KEY_ENV` de `sec_filings_text.py` sont les *noms* des
variables à lire, pas les clés. Sous Windows :

```powershell
setx GEMINI_API_KEY "ta_cle"      # puis ouvrir un NOUVEAU terminal
```

Au démarrage, `04c` et `07b` affichent le fournisseur retenu
(`Classification par Gemini (gemini-2.5-flash)`). Un refus de l'API (403,
400…) est journalisé avec le message renvoyé par le fournisseur, qui en dit
la cause.

Sans aucune clé, `07b` journalise ses lignes en `non_evalue_pas_de_cle_api`
au lieu d'appeler le modèle, et `04c` classe chaque 8-K **par règles** à partir
de son texte. Le cache de `04c` (`cache_8k_mistral.jsonl`, nom conservé) sert
quel que soit le fournisseur : un 8-K déjà classé par le modèle n'est pas
re-soumis. Un 8-K classé par règles, lui, est repris par le modèle dès qu'une
clé est définie.

**Seuls les 8-K récents vont au modèle.** Un 8-K ne sert qu'à périmer un
signal encore actionnable. Au-delà de la plus longue durée de vie d'un signal
(400 jours, `config.LLM_8K_FENETRE_JOURS`, déduit des durées de
`BACKTEST_SIGNAL_MAX_AGE_DAYS*`), il ne touche plus aucune décision. `04c` ne
soumet donc au modèle que les 8-K déposés dans cette fenêtre, et classe les
plus anciens par règles, sans appel ; un ancien 8-K classé par règles n'est
pas non plus rendu au modèle quand une clé arrive. Au 2026-09-26 : 6 338 8-K
sur 99 147 (6,4 %) dans la fenêtre, au lieu de tout l'historique.
`--llm-depuis-jours N` change la fenêtre, `0` rend tout l'historique au
modèle. Le backtest historique s'appuie donc sur la classification par règles
pour tout ce qui est plus ancien.

**Le quota.** Même réduit à environ 6 300 appels, le premier run avec une clé
peut dépasser le quota quotidien du palier gratuit de Gemini. Un
**disjoncteur** coupe alors le modèle : après trois analyses de suite refusées
pour quota malgré leurs réessais, plus aucun appel jusqu'à la fin du run. Les
8-K restants sont classés par règles, et le modèle reprend les récents au run
suivant. Sans lui, chaque appel attendait ses six réessais, jusqu'à 90 s
chacun, et le run rampait. Sur une offre payante, relève
`MISTRAL_REQUESTS_PER_SECOND`.

**Essayer sur quelques entreprises sans risque.** `04c --ticker AAPL`,
`04c --limit 5` ou `07b --limit 5` ne remplacent, dans le fichier de sortie
complet, que les lignes qu'ils ont refaites. Avant, ils réécrivaient le
fichier avec leurs seules lignes : un essai sur AAPL réduisait les 99 147 8-K
à ceux d'AAPL, et le filtre d'événements du backtest et du paper trading avec.

## Rafraîchissement trimestriel (04b, 04c, 07b, run_pipeline_quarterly.py)

Le pipeline de base (04→07) est annuel (un 10-K par an). Ces scripts
permettent un rafraîchissement TRIMESTRIEL de la valorisation elle-même,
point-in-time (chaque donnée datée de son dépôt SEC réel) :

    04b_recuperation_10q.py       -> 10-Q + reconstruction TTM (voir sa
                                      docstring : TTM vs trimestre brut, et
                                      la discrétisation cumul YTD -> trimestre)
    04c_recuperation_8k.py        -> événements matériels (8-K) entre deux
                                      trimestres TTM connus, classifiés par LLM
    07b_validation_qualitative.py -> verdict LLM de cohérence qualitative
                                      (texte du 10-K/10-Q à sa date de dépôt)
                                      vs l'écart de valorisation quantitatif
    run_pipeline_quarterly.py     -> orchestre 04b→04c→05→06→07→06b→07b→08 en
                                      conditions réelles (mode live), ou
                                      reconstitue une valorisation point-in-time
                                      passée sans aucun appel réseau
                                      (--as-of-date, mode replay)

04c et 07b réutilisent `sec_filings_text.py` (recherche/téléchargement de
filings SEC + appel LLM générique) et nécessitent `GEMINI_API_KEY` ou
`MISTRAL_API_KEY` (voir « Configuration requise ») pour produire un verdict de
modèle. Sans clé, aucun des deux ne plante : 07b journalise "non_evalue", et
04c classe chaque 8-K PAR RÈGLES à partir de son texte, verdicts que le modèle
reprend dès qu'une clé est définie.

05/06b/07 consomment automatiquement le TTM (`FINANCIALS_TTM_FILE`) dès que
04b a tourné une fois, en plus de l'annuel -- sans régression : identique à
avant si 04b n'a jamais tourné.

```bash
python 04b_recuperation_10q.py
python 04c_recuperation_8k.py
python 05_calcul_multiples.py && python 06_calcul_multiples_moyens.py
python 07_calcul_dcf.py && python 06b_calcul_valorisation_combinee.py   # 07 d'abord : 06b lit son DCF
python 07b_validation_qualitative.py
# ou, en une commande :
python run_pipeline_quarterly.py --skip-options   # sans 08 (pas besoin d'IB Gateway)
```

Cron (exemple, peu après chaque fenêtre de dépôt 10-Q habituelle) :

```
0 6 5 2,5,8,11 *  cd /chemin/vers/CalculRisque_Mark3 && python3 run_pipeline_quarterly.py --skip-options >> logs/quarterly.log 2>&1
```

Reconstitution point-in-time (backtest manuel, aucun appel réseau) :

```bash
python run_pipeline_quarterly.py --as-of-date 2024-06-30
```

## Backtest (01b, 03b, 09)

Trois scripts complètent le pipeline pour permettre de backtester une
stratégie construite sur l'écart entre cours de bourse et valorisation DCF
(07_calcul_dcf.py), sans biais de survivance et sans look-ahead bias :

    01b_historique_univers_sp500.py   -> univers POINT-IN-TIME (composants
                                          actuels + radiés, avec dates
                                          d'entrée/sortie de l'indice)
    03b_recuperation_cours_quotidiens.py -> cours QUOTIDIENS (IBKR + repli
                                          Stooq gratuit pour les radiés,
                                          qu'IBKR ne résout plus)
    09_backtest.py                    -> moteur de backtest événementiel

Ordre de lancement pour un backtest complet (en plus de 01/02 déjà connus) :

```bash
python 01b_historique_univers_sp500.py
python 03b_recuperation_cours_quotidiens.py --tickers data/universe/sp500_universe_full.csv
python 04_recuperation_10k.py --tickers data/universe/sp500_universe_full.csv
python 07_calcul_dcf.py
python 09_backtest.py --strategy valuation_gap_dcf --start-date 2015-01-01
```

`04_recuperation_10k.py` et `03b` doivent recevoir l'univers COMPLET
(`sp500_universe_full.csv`, sortie de 01b) pour backfiller aussi les
entreprises sorties du S&P 500 -- sinon le backtest retombe sur l'univers
actuel appliqué rétroactivement (biais de survivance, signalé par un warning
au lancement de 09).

Hypothèses du moteur (`backtest/engine.py`), à garder en tête pour
interpréter des résultats :
    - Décision à la clôture du jour J, exécution à l'ouverture de J+1 (aucune
      information n'est utilisée avant sa date réelle de publication : le
      signal DCF utilise `filed_date`, la date de dépôt SEC du 10-K, pas la
      date de clôture d'exercice).
    - Une position n'est JAMAIS fermée simplement parce que son écart de
      valorisation s'est refermé : seuls un stop-loss, un take-profit, ou une
      disparition des données de prix (radiation non couverte par Stooq) la
      clôturent. Elle reste sinon "gelée" à taille inchangée.
    - Un signal DCF vieux de plus de `BACKTEST_SIGNAL_MAX_AGE_DAYS` (config.py,
      400 jours par défaut, ou `BACKTEST_SIGNAL_MAX_AGE_DAYS_BY_PERIOD` selon
      le type de période) n'est plus une base valable pour METTRE DU CAPITAL
      sur ce symbole -- ni première entrée, ni renforcement. Même chose pour un
      8-K matériel déposé depuis. La position déjà ouverte n'est pas vendue
      pour autant : elle devient GELÉE, conservée à taille inchangée jusqu'à
      son stop-loss/take-profit. Le filtre momentum, lui, ne concerne que les
      nouvelles entrées (voir « Correctifs de l'audit du moteur actions »).
    - Le nombre de positions simultanées n'est **pas** plafonné : toutes les
      entreprises retenues par la stratégie sont ouvertes. La concentration
      reste bornée par le seul plafond de pondération
      (`BACKTEST_MAX_WEIGHT_PER_POSITION_PCT`), qui limite la part du
      portefeuille d'UNE ligne sans limiter leur nombre.

Résultats sauvegardés intégralement sous `data/backtest/<run_id>/` :
`equity_curve.parquet`, `positions_history.parquet`, `trades.parquet`,
`signals_history.parquet`, `metrics.json`, `run_config.json`.

### Relire un run : `14_audit_backtest.py`

`metrics.json` répond à « combien ça a rapporté », pas à « de quoi ce chiffre
est-il fait ». `14_audit_backtest.py` relit les sorties d'un run **sans le
relancer** (quelques secondes) et pose les cinq questions qui peuvent
l'invalider :

```bash
python 14_audit_backtest.py                          # dernier run
python 14_audit_backtest.py --run-id 20260816_000429
```

1. **Couverture des signaux.** Quelle part de l'indice RÉEL de chaque année la
   stratégie pouvait-elle acheter, et surtout : la même mesure séparément pour
   les membres actuels et pour les entreprises sorties depuis. C'est le test
   décisif du biais de survivance résiduel (voir la section dédiée plus bas).
2. **Trades affichés contre thèses réelles.** `num_trades`, `win_rate_pct` et
   `profit_factor` comptent des EXÉCUTIONS, ventes partielles de rebalancement
   comprises. Le script recalcule les mêmes indicateurs par THÈSE.
3. **Alpha année par année et par sous-période glissante**, pour distinguer un
   alpha régulier d'un alpha gagné sur une seule fenêtre.
4. **Sensibilité à la date de départ** : le CAGR obtenu en décalant le début
   du run de 1, 2, 3, 5 ans.
5. **Sensibilité aux coûts** : ce que devient le résultat si l'exécution réelle
   coûte 20, 30 ou 50 bps par aller simple au lieu des 10 bps supposés.

### Correctifs de l'audit du moteur actions

L'audit du run `20260816_000429` (19,00 % de CAGR, +5,49 % d'alpha, **17,9 %
d'ordres d'achat tronqués**) a montré que la troncature n'était pas la
conséquence assumée de la règle des positions gelées. Sept défauts distincts,
tous corrigés, tous couverts par `tests/test_audit_moteur_actions.py` :

| | Défaut | Effet mesuré |
|---|---|---|
| A1 | La file d'exécution triait sur le SIGNE DE LA CIBLE, pas sur le sens réel de l'ordre. Une cible de 5 000 $ sur une ligne qui en vaut 8 000 est une vente : elle partait pourtant dans le paquet des achats. | Le résultat du run dépendait de l'ordre d'itération d'un dictionnaire. Scénario identique, seul l'ordre d'énumération changeant : 750 000 $ laissés en cash d'un côté, portefeuille complet de l'autre. |
| A2 | Une position détenue court-circuitait TOUS les filtres de péremption. Le commentaire du code affirmait qu'elle était « de toute façon gelée, pas rebalancée sur la base de ce vieux signal » : elle ne l'était pas. | Ligne renforcée **x3,88** sur un signal périmé et invalidé par un 8-K matériel. Générateur de value trap exactement là où les filtres devaient protéger. |
| A3 | `num_trades` / `win_rate_pct` / `profit_factor` comptaient les allègements de rebalancement comme autant de trades. | Une thèse unique perdante de −226 k$ s'affichait comme 32 trades à 91 % de réussite. |
| A4 | L'indice de référence équipondéré appelait `pct_change()` sans `fill_method=None`, ce qui reportait les valeurs manquantes SANS limite et annulait le forward-fill borné du panel. | Un titre absent 15 jours puis repris 40 % plus bas déversait toute sa baisse sur UNE séance (−20 % mesuré sur un indice à deux composantes). `beta`, `tracking_error_pct` et `information_ratio` étaient calculés sur cette série faussée. |
| A5 | Les signaux étaient indexés sur la date de dépôt EXACTE et retrouvés par égalité avec le jour de bourse simulé. | Un 10-K déposé un jour où le NYSE est fermé — le Vendredi saint, tous les ans, la SEC étant ouverte — n'était jamais vu. Le signal disparaissait sans trace. Il est maintenant connu à la première séance suivante. |
| A6 | **La cause du « 17,9 % d'ordres tronqués ».** `_rebalance` alloue une VALEUR DE POSITION égale au NAV, alors qu'acquérir cette valeur consomme en plus la commission et le slippage. | Un portefeuille pleinement investi est court d'exactement `cost_bps` à chaque rebalancement, et ne peut pas ne pas l'être. Ce manque de 0,1 % était compté comme une troncature : **100 % des ordres signalés « tronqués » sur un moteur qui faisait exactement son travail.** |
| A7 | `03b` prenait l'univers point-in-time par défaut, `04`/`04b`/`04c` l'univers ACTUEL — chacun sa règle, en dur. | Cours des entreprises radiées collectés **sans** leurs fondamentaux : biais de survivance sur les signaux, pas sur l'indice de référence. Les quatre partagent maintenant `config.default_universe_file()` (voir la section dédiée dans « Biais et limites connus »). |

Trois conséquences à retenir avant de comparer un run d'avant à un run
d'après :

- **Les chiffres changent.** A1, A2 et A5 modifient les positions réellement
  prises, donc la courbe de NAV. Un run antérieur n'est pas comparable ligne à
  ligne.
- **`truncated_orders_pct` ne mesure plus la même chose** et ne déclenche plus
  l'avertissement. Il compte désormais les lots d'achats réduits au prorata
  **au-delà de ce que les frais expliquent**.
- **Le chiffre à lire est `unfilled_dollar_pct`** : la part du montant d'achat
  DEMANDÉ qui n'a pas pu être investie. C'est lui qui déclenche
  l'avertissement, au-delà de 1 %. Un compte par ordres ne dit rien tant qu'on
  ne sait pas de combien : mesuré sur un run de référence, 53 % des ordres
  étaient réduits… pour 0,12 % du montant demandé, soit un portefeuille investi
  à 99,88 % de ce que la stratégie voulait.

### Stratégie `valuation_gap_sector_neutral`

```bash
python 09_backtest.py --strategy valuation_gap_sector_neutral
```

`config.SECTOR_DCF_PARAMS` fixe un WACC et deux taux de croissance **par
secteur**, choisis à la main :

| | WACC calibré | croissance FCF | croissance terminale |
|---|---|---|---|
| Technologie | 10,0 % | 7 % | 3,0 % |
| Agro-alimentaire et boissons | 7,0 % | 3 % | 2,0 % |

À flux de trésorerie identique, la techno ressort structurellement mieux
valorisée — non parce que le marché s'y trompe davantage, mais parce que la
table le dit. Classer les candidates sur `gap_pct` brut revient donc pour
partie à **classer la table de configuration**, et à surpondérer en permanence
les secteurs auxquels on a prêté les hypothèses les plus généreuses. Comme ces
hypothèses ont été écrites en connaissant l'histoire boursière de 2010-2026,
c'est un biais de rétrospection qui entre par la porte de service : aucune date
n'est violée, mais le *choix* des paramètres, lui, connaît la suite.

La stratégie mesure donc l'écart **en excès de la médiane de son propre
secteur**, à la date courante, sur les seuls signaux déjà publiés :

```
score = gap_pct - mediane(gap_pct des pairs du secteur connus à cette date)
```

Une techno n'est retenue que si elle est bon marché *pour une techno*. Mesuré
sur un univers synthétique où seul le NIVEAU diffère d'un secteur à l'autre :

| | candidates | Technologie | Santé | Agro-alim. | Utilities |
|---|---|---|---|---|---|
| `valuation_gap_dcf` | 123 | **48,5 %** | 31,7 % | 9,7 % | 1,5 % |
| `valuation_gap_sector_neutral` | 66 | 30,0 % | 30,0 % | 20,6 % | 4,5 % |

Trois garde-fous :

- **`min_absolute_gap_pct`** (10 %) : la neutralité sectorielle sert à
  *classer*, pas à absoudre. Sans lui, un secteur entièrement survalorisé
  fournirait quand même ses « moins pires ».
- **`max_weight_per_sector_pct`** (30 %) : le plafond par position ne borne
  rien au niveau du secteur — vingt technos à 4 % font 80 % du portefeuille
  sans qu'aucune ligne ne dépasse son plafond. L'excédent n'est **pas**
  redistribué (ce serait concentrer ailleurs) : la somme des poids descend et
  le reste va en cash.
- **`MIN_PEERS_PER_SECTOR`** (5) : en dessous, la médiane sectorielle ne mesure
  plus une norme mais un ou deux titres ; repli sur la médiane de l'univers.

Attention : `--entry-threshold-pct` **ne se lit pas pareil** d'une stratégie à
l'autre — écart au cours pour `valuation_gap_dcf` (20 %), écart à la médiane du
secteur pour celle-ci (10 %). Ne pas le préciser laisse chaque stratégie
appliquer le sien.

### Optimisation des réglages actions (`16_optimize_strategie_actions.py`)

```bash
make optimize-actions                                    # DCF, 108 combinaisons, ~12 min sur 4 cœurs
make optimize-actions STRATEGY_ACTIONS=valuation_gap_combined   # 432 : l'axe de hiérarchie s'ajoute
make optimize-actions STRATEGY_ACTIONS=valuation_gap_sector_neutral
python 16_optimize_strategie_actions.py --report-only data/backtest/<csv>   # relire sans relancer
python 16_optimize_strategie_actions.py --multiple-hierarchy-grid flat tiers    # hierarchies au choix
python 16_optimize_strategie_actions.py --max-holding-grid -1 120 180 270   # rebalayer l'horizon
python 16_optimize_strategie_actions.py --stop-loss-grid -15 -25 -40 \
    --max-weight-grid 10 20                              # rebalayer les deux axes retirés
```

Grid-search sur **cinq axes à la fois** — take-profit, seuil d'entrée, filtre
momentum, zone de non-négociation, et **hiérarchie des multiples**. Les quatre
optimiseurs options font varier un paramètre par run, ce qui suffit quand les
réglages sont séparables ; ici ils ne le sont pas (le seuil d'entrée déplace le
nombre de lignes donc l'effet de la bande, la prise de gain déplace la rotation
donc la friction), et une descente axe par axe trouverait un optimum de
coordonnée, pas un optimum.

Quatre de ces axes sont des réglages d'**exécution** : ils disent *comment on
négocie*. Deux axes de **signal** ont été ajoutés depuis — ils disent *ce qu'on
croit*, pas comment on l'exécute :

| Axe de signal | Ce qu'il décide | Statut |
|---|---|---|
| `max_holding_days` | combien de temps on croit à la thèse | mesuré, **négatif**, réduit à la production |
| `multiple_hierarchy` | lequel des trois multiples tranche | **balayé par défaut** |

Le classement porte sur la **seule fenêtre d'apprentissage** (2015-2021), avec
un plancher de rendement contre le SPY sur cette même fenêtre — maximiser un
ratio autorise sinon à l'améliorer en désinvestissant. `test_sharpe_ratio`
(2022-2026) est affiché à côté sans jamais entrer dans la sélection.

#### L'horizon de convergence, et pourquoi ses bornes ne sont pas rondes

Le moteur ne ferme une position que sur stop-loss, prise de gain, stop suiveur,
perte de signal ou repesage. Une thèse de convergence qui ne se réalise **jamais**
n'est donc fermée par rien : elle occupe du capital indéfiniment.
`BACKTEST_MAX_HOLDING_DAYS` est la règle qui y met un terme — implémentée de
longue date, mais désactivée et jamais balayée.

Les points de la grille viennent d'une mesure, pas d'un choix rond. Sur les
**13 141 sorties** de la configuration de référence :

| Durée de détention | Rendement moyen | **Annualisé** |
|---|---|---|
| < 30 j | +2,2 % | **+79 %/an** |
| 30–60 j | +4,8 % | +47 %/an |
| 60–90 j | +5,8 % | +32 %/an |
| 90–180 j | +5,6 % | +17 %/an |
| 180–270 j | +6,6 % | +11 %/an |
| 270–365 j | +7,1 % | +8,4 %/an |
| 365–545 j | +7,3 % | +6,3 %/an |
| 545–730 j | +7,0 % | **+4,0 %/an** |
| > 730 j | +9,4 % | +4,2 %/an |

Le rendement **absolu** est quasi plat pendant que la durée est multipliée par
60 : le gain s'accumule dans les premières semaines puis **s'arrête**. C'est la
signature d'un horizon de convergence.

**Le contrôle de biais compte plus que le tableau.** Une position qui converge
vite sort vite *par construction* — la prise de gain tronque les positions
rapides —, donc son rendement annualisé est mécaniquement élevé. Restreinte aux
seules sorties `rebalance` (80,5 % du total, et les moins liées au rendement de
la ligne), la décroissance est **plus raide encore** : +132 %/an sous 30 jours
contre +3,9 %/an au-delà de 545. Elle n'est donc pas un artefact.

Un biais résiduel subsiste, et il joue dans le bon sens : une position encore
vivante à 600 jours est une qui n'a pas été stoppée, ce qui **flatte** les
tranches longues. La décroissance mesurée est donc un minorant.

Bornes retenues : **90, 180, 365 jours, et aucun horizon** (la valeur en
production). Elles couvrent la partie raide et la partie plate. Repères de
distribution : médiane 77 j, p90 280 j, p95 366 j ; en capital-jours, 45 % sous
180 j et 80 % sous 365 j.

**Ce tableau ne conclut rien à lui seul** — il est conditionné à la façon dont
chaque position s'est terminée. Seul le backtest complet, qui rejoue tout
l'historique sous la contrainte, répond. C'est à ça que sert l'axe.

#### Ce que l'horizon a donné : le plus gros axe de la grille, et il dit non

L'axe est **de loin le plus discriminant** que cette grille ait jamais porté :

| Axe | η² | Étendue des moyennes |
|---|---|---|
| **`max_holding_days`** | **57,3 %** | **0,094** |
| `take_profit_pct` | 18,5 % | 0,045 |
| `momentum_min_pct` | 8,5 % | 0,029 |
| `entry_threshold_pct` | 1,7 % | 0,013 |
| `rebalance_band_pct` | 0,6 % | 0,008 |

La prémisse du levier 4 était donc juste : **un axe de signal bouge plus que
n'importe quel axe d'exécution** — deux fois plus que la prise de gain, qui
dominait la grille jusque-là.

Et il bouge dans le mauvais sens. Les autres axes fixés sur la production,
écart apparié contre « aucun horizon » :

| Horizon | Sharpe | Rotation | Positions fermées | Écart apparié | IC 95 % |
|---|---|---|---|---|---|
| aucun *(production)* | 0,930 | 760 % | 2 628 | — | — |
| 365 j | 0,932 | 781 % | 2 694 | **+0,002** | [−0,011, +0,016] |
| 180 j | 0,906 | 885 % | 3 080 | −0,025 | [−0,073, +0,019] |
| 90 j | 0,857 | **1 177 %** | **4 226** | **−0,073** | [−0,133, **−0,008**] |

**Le mécanisme est lisible dans la colonne rotation.** Forcer une sortie ne
supprime pas la thèse : elle est toujours là le lendemain, et le moteur la
rachète. Un horizon de 90 jours multiplie les fermetures par 1,6 et la rotation
par 1,55 — on paie la friction deux fois pour se retrouver dans la même
position. À 365 jours, l'horizon ne touche que 155 sorties sur 13 409 : il est
gratuit parce qu'il ne fait rien.

**La leçon, et elle vaut au-delà de cet axe.** Le tableau de décroissance était
une observation **conditionnelle** — les positions qui ont vécu longtemps ont
moins gagné par an. L'intervention est **causale** — les couper court
rapporte-t-il davantage ? Les deux ne se déduisent pas l'une de l'autre, et ici
elles se contredisent : la décroissance mesure *quelles positions survivent*,
pas *ce que durer coûte*. C'est exactement le piège que le backtest complet
sert à détecter, et la raison pour laquelle le tableau de décroissance ne
pouvait pas conclure seul.

**Rien n'est adopté.** `BACKTEST_MAX_HOLDING_DAYS` reste à `None`. La seule
combinaison établie meilleure avec un horizon (seuil d'entrée 30 %, bande 15,
horizon 365) vaut **+0,023** — contre **+0,022** pour la même sans horizon,
avec un intervalle *plus serré*. L'horizon n'y apporte rien : le gain est celui
du seuil d'entrée, déjà connu. Et le prix de l'axe est réel : le plancher de
bruit du Sharpe déflaté passe de 0,983 à **1,004** (1 858 essais cumulés),
pendant que le meilleur Sharpe plein échantillon reste à 0,925.

**L'axe est ensuite réduit à sa valeur de production**, comme `stop_loss_pct` et
`max_weight_pct` avant lui, et pour la même raison : une réponse connue et
négative ne vaut pas un facteur quatre sur la grille, d'autant qu'élargir relève
le plancher de bruit. Il reste balayable via `--max-holding-grid`.

#### La hiérarchie des multiples, et ce qu'elle a révélé en chemin

Trois multiples donnent trois valeurs théoriques pour la même action, et elles
divergent : sur les **13 240 lignes** où P/E et EV/EBITDA coexistent, l'écart
médian entre les deux vaut **19,5 points de cours** (45,9 au troisième
quartile). Le choix de celui qui tranche est un réglage de **signal**, et
jusqu'ici il n'avait jamais été mesuré — `MULTIPLE_COMBINATION` était fixé sur
un argument de littérature (Liu, Nissim & Thomas 2002), pas sur ces données.

| Hiérarchie | Qui tranche | Le pari |
|---|---|---|
| `flat` | la médiane des trois | aucun — EV/Sales, le moins fiable, départage dès qu'il tombe au milieu |
| `tiers` | P/E et EV/EBITDA à égalité, moyennés | les deux multiples de résultats font jeu égal |
| `pe_first` | P/E seul, EV/EBITDA en repli | le résultat net est la mesure la mieux arbitrée |
| `ebitda_first` | EV/EBITDA seul, P/E en repli | l'EBITDA compare mieux des pairs aux dettes différentes |

**La couverture est invariante** (87,06 % pour les quatre) : l'axe déplace la
valeur, jamais le nombre de lignes valorisées. C'est ce qui en fait un axe
propre — on ne mesure pas un effet de couverture déguisé en effet de multiple.

`06b` stocke déjà les trois prix implicites, donc la combinaison se **rejoue au
chargement** sans régénérer le parquet. La mécanique vit dans
`hierarchie_multiples.py`, que `06b` et `backtest.data_loader` appellent tous
les deux : deux implémentations finiraient par diverger, et l'écart ne se
verrait que dans les chiffres.

##### La hiérarchie n'avait jamais tourné, et c'était un bug de trois mots

En vérifiant que la recombinaison reproduisait bien le fichier de production,
elle ne l'a reproduit que sous `flat` — **exactement, à 0,000e+00 sur les 27 674
lignes** — alors que `config.MULTIPLE_COMBINATION` vaut `tiers`.

La première explication était un fichier périmé. Elle était fausse : **rejouer
`06b` reproduisait `flat` à l'identique**. La cause est dans le code, et elle
tient en trois noms de colonnes :

```python
implied = pd.concat([price_from_ebitda, price_from_sales, price_from_pe], axis=1)
# -> colonnes : "ev_ebitda_median", "ev_sales_median", "pe_median"
# MULTIPLE_RELIABILITY_TIERS est indexée sur : "EV/EBITDA", "EV/Sales", "P/E"
```

Le classement par rang faisait `tiers.get(colonne, rang_de_repli)`. Aucune des
trois colonnes n'étant dans la table, **les trois tombaient dans le même rang de
repli** — et la médiane d'un rang qui contient les trois multiples est
exactement la médiane à plat. `MULTIPLE_COMBINATION = "tiers"` **n'a donc jamais
rien fait**, depuis son introduction.

Ce qui l'a rendu invisible : aucune erreur, aucun avertissement, un résultat
parfaitement plausible. C'est le même motif que la leçon n° 2 ci-dessous — une
protection documentée qui ne tournait pas — et il ne s'est vu que parce que
l'instrumentation exigeait de reproduire le fichier au bit près.

**Deux corrections, pas une.** Nommer les colonnes comme les tables de config
répare le cas présent ; refuser une colonne inconnue au lieu de la replier
empêche la classe entière de se reproduire :

```python
inconnues = [c for c in implied.columns if c not in table]
if inconnues:
    raise ValueError(...)   # un repli silencieux n'est plus possible
```

##### À faire après un `git pull` : régénérer le signal

Le correctif change **ce que `06b` produit**, pas seulement son code. Le parquet
versionné porte encore l'ancien signal — une seule commande suffit :

```bash
python 06b_calcul_valorisation_combinee.py     # ~2 min
```

`tests/test_hierarchie_multiples.py::test_le_parquet_porte_bien_ce_que_la_config_annonce`
échoue tant que ce n'est pas fait, et dit quoi lancer. La régénération est
déterministe : elle reproduit exactement le fichier mesuré ci-dessous.

##### Ce que la correction change dans les chiffres

`06b` rejoué, le fichier de production porte désormais `tiers`. La
régénération est **exactement** la recombinaison `tiers` (0,000e+00 sur les
trois colonnes dérivées), et rien d'autre n'a bougé — prix implicites, DCF,
cours et `n_peers` sont identiques au bit près. Le changement est donc
strictement celui de la hiérarchie, sur **18 008 lignes** (écart de gap médian
9,9 points).

| Configuration de production | Avant (`flat`) | Après (`tiers`) | |
|---|---|---|---|
| Sharpe plein échantillon | 0,930 | **0,931** | +0,001 |
| Sharpe **hors échantillon** | 0,817 | **0,846** | **+0,029** |
| Sharpe apprentissage | 1,003 | 0,988 | −0,015 |
| Drawdown maximal | −26,62 % | **−25,06 %** | **+1,56 pt** |
| CAGR | 15,20 % | 15,18 % | −0,01 pt |
| Rotation annualisée | 760 % | 757 % | −3 pts |

**L'écart apparié reste +0,002 (IC [−0,074, +0,078]) : non établi.** Ce qui
change est donc réel mais minuscule sur le Sharpe, et un peu plus net sur le
drawdown. Le gain hors échantillon (+0,029) est du bon côté, et la perte en
apprentissage (−0,015) est ce qu'on attend d'un réglage qui n'a pas été choisi
sur cette fenêtre.

**Les chiffres `combinee` publiés dans ce README ont été recalculés sur ce
signal.** Ceux des versions antérieures portaient sur `flat`.

Le fichier alimente aussi **toute la partie options** (`10_backtest_options.py`
et les quatre optimiseurs `11*`), dont les chiffres changent donc également.
Mesuré sur `valuation_gap_multiples_options` :

| | Avant (`flat`) | Après (`tiers`) |
|---|---|---|
| Sharpe | −0,695 | −0,638 |
| CAGR | −4,42 % | −4,03 % |
| Drawdown maximal | −55,57 % | −56,03 % |

La stratégie reste **franchement perdante** dans les deux cas — la correction ne
change pas cette conclusion-là. Les quatre optimiseurs options, eux, n'ont pas
été rejoués : leurs réglages retenus ont été choisis sur `flat`, et les
rebalayer est un chantier en soi.

##### Ce que l'axe a donné : rien d'établi, et un ordre qui s'inverse

L'axe explique **6,9 %** de la variance (étendue des moyennes 0,019) — loin de
`take_profit_pct` (51,1 %), au niveau du seuil d'entrée. Les autres axes fixés
sur la production :

*(mesure faite contre `flat`, la hiérarchie qui tournait alors)*

| Hiérarchie | Sharpe test | Écart apparié | IC 95 % |
|---|---|---|---|
| `flat` *(référence d'alors)* | 0,817 | — | — |
| **`tiers`** *(production depuis)* | **0,846** | +0,002 | [−0,074, +0,078] |
| `pe_first` | 0,836 | +0,002 | [−0,088, +0,090] |
| `ebitda_first` | 0,782 | −0,012 | [−0,091, +0,066] |

**Aucune n'est distinguable de ce qui tourne.** Les intervalles sont d'ailleurs
deux fois plus larges que ceux des axes d'exécution (±0,08 contre ±0,014 pour
l'horizon) : la hiérarchie change le signal sur 13 000 à 18 000 lignes, donc les
courbes de NAV se décorrèlent (0,988 → 0,982) et le test apparié y perd
mécaniquement de la précision. C'est le prix d'un axe qui touche au signal.

**L'ordre s'inverse entre les deux fenêtres**, et c'est le plus parlant :

| Hiérarchie | Rang en apprentissage | Rang hors échantillon |
|---|---|---|
| `ebitda_first` | **1er** | **4e** |
| `pe_first` | 2e | 1er |
| `flat` | 3e | 3e |
| `tiers` | 4e | **2e** |

Une inversion quasi complète est la signature du bruit, pas d'un effet. À
retenir tout de même : **hors échantillon**, l'ordre obtenu
(`pe_first` > `tiers` > `flat` > `ebitda_first`) est celui que prédit Liu,
Nissim & Thomas — les multiples de résultats devant, P/E en tête. C'est la
fenêtre qui n'a rien choisi qui le dit, ce qui rend l'indication intéressante ;
elle reste non établie, et l'ordre inverse en apprentissage interdit d'en faire
plus qu'une note.

**Ce n'est pas la mesure qui a fait changer la production, c'est le bug.**
`tiers` n'est pas établi meilleur que `flat` — il ne l'aurait pas emporté sur
ces chiffres. Ce qui a tranché est qu'un réglage documenté, justifié et
configuré ne s'appliquait pas : le réparer fait tourner ce que la configuration
dit depuis toujours, et la mesure dit que le prix de cette mise en cohérence est
nul à l'incertitude près. Si la préférence était de garder le comportement
historique, la correction à faire serait `MULTIPLE_COMBINATION = "flat"` — pas
de laisser le code contredire la config.

**Un avertissement sur la sélection, au passage.** Le plateau atteint 94
combinaisons et le départage par rotation y a retenu `ebitda_first` avec prise
de gain désactivée : rotation 535 % contre 760 %, mais **−0,048 de Sharpe hors
échantillon** (apparié −0,033, IC [−0,134, +0,071]). Le départage ne regarde pas
la fenêtre de test — c'est voulu, elle ne vaut que tant qu'elle n'a rien choisi
— mais un plateau qui grossit lui donne plus d'occasions de mal tomber. Le
plancher de bruit, lui, est monté à **1,021** (2 290 essais) pour un meilleur
Sharpe plein échantillon de 0,900.

#### Deux axes retirés du défaut, et comment on l'a su

La décomposition de variance des Sharpe de la grille (η² par axe) mesure ce
que chaque axe explique. Deux d'entre eux n'expliquent rien :

| Axe | η² | Étendue des moyennes |
|---|---|---|
| `take_profit_pct` | 63,9 % | 0,052 |
| `momentum_min_pct` | 15,1 % | 0,022 |
| `entry_threshold_pct` | 7,2 % | 0,015 |
| `rebalance_band_pct` | 4,7 % | 0,014 |
| **`stop_loss_pct`** | **0,6 %** | **0,005** |
| **`max_weight_pct`** | **0,1 %** | **0,001** |

Les deux derniers multipliaient la grille par **huit** (4 × 2) pour 0,7 % de
l'information. Au défaut, chacun est réduit à sa valeur de production — ce qui
garde la configuration en place **dans** la grille, condition du test apparié
ci-dessous. La grille passe de 864 à **108** combinaisons, et le plancher de
bruit du Sharpe déflaté baisse d'autant, puisqu'il croît avec le nombre
d'essais. Les deux axes restent balayables à la demande : c'est le défaut qui
change, pas la capacité.

#### Ce que la grille a changé, et ce qu'elle a refusé de changer

| Réglage | Avant | Après | Pourquoi |
|---|---|---|---|
| `BACKTEST_STOCKS_MOMENTUM_MIN_PCT` | −10 % | **désactivé** | Unanime sur le plateau des **deux** stratégies, et gagne sur les **deux** fenêtres |
| `BACKTEST_REBALANCE_BAND_PCT` | (n'existait pas) | **15** points de NAV | Apprentissage plat, rotation en baisse, test confirme |
| `BACKTEST_STOP_LOSS_PCT` | −15 % | −15 % | La grille confirme la valeur en place |
| `BACKTEST_TAKE_PROFIT_PCT` | +30 % | +30 % | Unanime sur le plateau ; l'élargir gagne en test mais **perd** en apprentissage |
| `BACKTEST_MAX_WEIGHT_PER_POSITION_PCT` | 20 % | 20 % | Gain massif en apprentissage, **inversé** hors échantillon |
| `BACKTEST_SECTOR_NEUTRAL_ENTRY_THRESHOLD_PCT` | 10 | 10 | Axe **plat** : rien à optimiser |

Résultat, sur `--start-date 2015-01-01` :

| | `valuation_gap_dcf` | | `valuation_gap_sector_neutral` | |
|---|---|---|---|---|
| | avant | après | avant | après |
| **Sharpe hors échantillon** (2022-2026) | 0,698 | **0,795** | 0,630 | **0,742** |
| Sharpe apprentissage (2015-2021) | 0,837 | 0,924 | 0,831 | 0,907 |
| Sharpe plein échantillon | 0,782 | 0,872 | 0,751 | 0,838 |
| CAGR | 15,56 % | 18,05 % | 14,28 % | 16,55 % |
| Alpha vs SPY | +3,57 % | +6,07 % | +2,30 % | +4,56 % |
| Information ratio | 0,40 | 0,63 | 0,25 | 0,49 |

Le gain résiste au durcissement des hypothèses de coût, sans se creuser
(DCF, plein échantillon) : 0,78 → 0,87 à **10 bps par aller simple** (soit
20 bps l'aller-retour, l'hypothèse retenue), 0,70 → 0,79 à 30 bps (60 bps
l'aller-retour), 0,63 → 0,71 à 50 bps (100 bps l'aller-retour).

#### Le filtre momentum coûtait plus qu'il ne protégeait

C'est le résultat le plus inattendu, et le mieux établi. Un titre dont le cours
a chuté de plus de 10 % sur un an est **précisément celui dont l'écart de
valorisation vient de s'élargir** — la candidate la plus attrayante de la
thèse. Le garde-fou anti-*value trap* supprimait donc du signal en même temps
que du piège, alors que le moteur a déjà deux protections qui, elles, ne
coûtent pas de signal : la péremption du signal et le stop-loss.

À lui seul, le désactiver vaut **+0,081** de Sharpe hors échantillon côté DCF
et **+0,100** côté neutre au secteur.

#### Le classement ne départage rien, et il faut le dire

L'erreur-type d'un Sharpe estimé sur sept ans vaut **0,47** (Lo, 2002). Sur les
432 combinaisons de la grille, **les 432 sont à moins d'une erreur-type du
maximum**. Retenir le premier du classement, c'est retenir le tirage le plus
chanceux d'un ensemble statistiquement homogène.

Trois conséquences dans l'outil :

- le meilleur point est, parmi les combinaisons indiscernables à
  `--plateau-tolerance` près, celle qui **négocie le moins**. Le départage ne
  regarde pas la fenêtre de test — ce serait la consommer — mais la rotation,
  qui n'est pas une mesure de performance mais d'**exposition à une
  hypothèse** : tout le backtest suppose 10 bps par aller simple ;
- le rapport dit ce que la grille **établit** (un axe sur lequel tout le
  plateau s'accorde) par opposition à ce qu'elle **classe**. Sur les cinq axes
  balayés, **aucun n'est unanime** depuis l'ajout des axes de signal — le plateau
  est passé de 23 à 94 combinaisons, et la prise de gain, jusque-là unanime, ne
  l'est plus. Un axe réduit à un point est affiché **non balayé**, jamais
  « unanime » : il l'est par construction, et le lire comme un résultat serait
  une erreur ;
- chaque combinaison est **comparée à la configuration en production** par
  bootstrap apparié (`--paired-bootstrap`, 2 000 par défaut). C'est ce qui rend
  la grille capable de conclure — détail ci-dessous.

#### Ce que le test apparié départage, et que le classement ne départageait pas

L'erreur-type marginale ci-dessus vaut pour deux stratégies **indépendantes**.
Les 432 combinaisons sont des variantes du **même** backtest : leurs courbes de
NAV sont corrélées à **0,984**. Leur écart est apparié, et sa dispersion est
bien plus faible que celle de chacun de ses termes.

| | demi-largeur de l'intervalle |
|---|---|
| Erreur-type marginale (Sharpe 0,93 sur 11,6 ans) | 0,353 |
| Intervalle **apparié** (médiane des 432) | **0,097** |

Soit **quatre fois plus précis**, et la grille passe de « rien n'est
distinguable » à un résultat :

| | combinaisons |
|---|---|
| Établies **meilleures** que la production (IC au-dessus de 0) | **0** |
| Établies **pires** (IC au-dessous de 0) | 41 |
| Indistinguables | 391 |

Les deux demi-largeurs portent sur la **même fenêtre**, et c'est indispensable :
l'intervalle apparié est mesuré sur la courbe entière, et le comparer à
l'erreur-type de la seule fenêtre d'apprentissage gonflerait le rapport de
√(11,6/7) — 25 % de précision annoncée qui n'existerait pas.

**Aucune combinaison n'est établie meilleure que ce qui tourne.** C'est un
résultat, pas une absence de résultat : 432 points balayés, et pas un seul dont
l'intervalle exclue zéro. Ce que la grille établit, elle l'établit **contre** —
41 combinaisons sont mesurément pires.

##### Le seul gain que la grille avait établi n'a pas survécu à la correction

Sur le signal `flat`, le seuil d'entrée à 30 % au lieu de 20 était établi
meilleur : **+0,022** (IC [+0,009, +0,037]), gain retrouvé sur les deux
fenêtres. Il avait résisté à deux élargissements de grille.

Sur le signal `tiers`, le même réglage vaut **+0,006** (IC [−0,011, +0,023]) :
l'intervalle recouvre zéro, le résultat disparaît.

| Seuil d'entrée | Sharpe test | Écart apparié | IC 95 % |
|---|---|---|---|
| 15 % | 0,844 | +0,000 | [−0,011, +0,009] |
| **20 %** *(production)* | 0,846 | — | — |
| 30 % | 0,850 | +0,006 | [−0,011, +0,023] |

**C'est la deuxième fois dans ce dépôt qu'un résultat « établi » ne survit pas à
un changement de conditions** — après le −0,063 qui s'était inversé sur une
fenêtre décalée de vingt mois. La leçon est la même : un intervalle qui exclut
zéro dit que l'écart est réel *sur ces données-là*, pas qu'il est robuste au
changement de ce qui les produit. Corriger le signal a changé 18 008 lignes ;
un gain de +0,022 n'y a pas résisté.

**Lire l'intervalle, pas la p-value.** L'intervalle est ce qui établit ou non un
résultat ; la p-value du bootstrap plancherise à 1/N (0,0005 pour 2 000
rééchantillonnages) et ne peut donc pas se comparer à un seuil de Bonferroni du
même ordre (0,05/432 = 0,00012). C'est ce que dit `paired_sharpe_difference` :
un bootstrap par blocs est légèrement libéral, une p-value juste sous 0,05 ne
vaut pas une preuve, un intervalle franchement à droite de zéro, oui.

**Et rien n'est adopté.** Le Sharpe plein échantillon du meilleur point vaut
0,900 pour un **plancher de bruit de 1,034** à 2 722 essais cumulés : au niveau
du programme entier, il reste sous le seuil à partir duquel un résultat se
distingue de la sélection elle-même. Le réglage en place ne
bouge pas. Le plancher a d'ailleurs **monté** de 0,983 à 1,004, 1,021 puis 1,034 au
fil des élargissements : chaque essai supplémentaire relève la barre, et c'est
le prix à payer pour tout axe ajouté.

**Ce test ne sélectionne pas, et ne doit pas.** Il est mesuré sur la courbe
entière, fenêtre de test comprise ; l'y faire entrer consommerait la seule
fenêtre qui n'a rien choisi. Il se lit **après**, au même titre que
`test_sharpe_ratio`. La sélection reste le Sharpe d'apprentissage départagé par
la rotation.

#### Changer de métrique : l'information ratio (`--rank-metric`)

Le Sharpe d'une stratégie actions long-only est dominé par le facteur **marché**,
que toutes les combinaisons d'une grille portent ensemble : il les bruite toutes
sans en séparer aucune. L'**information ratio** est le Sharpe de l'écart *actif*
(stratégie moins indice) — le facteur commun disparaît.

```bash
python 16_optimize_strategie_actions.py --rank-metric information_ratio
```

Le classement, le test apparié, la colonne de test et l'erreur-type affichée
suivent tous la métrique choisie : classer sur l'IR en jugeant sur le Sharpe
reviendrait à choisir selon un critère et à conclure selon un autre.

**Mesuré, à grille identique** (432 combinaisons, même signal) :

| | Sharpe | Information ratio |
|---|---|---|
| Étendue de la grille | 0,138 | 0,280 |
| Demi-largeur appariée | 0,097 | 0,152 |
| **Étendue / demi-largeur** | **1,43** | **1,84** |
| Établies **pires** que la production | 41 | **68** |
| Établies **meilleures** | 0 | 0 |

**Le gain est réel mais modeste : ×1,29**, pas le ×1,9 annoncé au départ. Cette
première estimation se mesurait contre l'erreur-type *marginale* — la base
d'avant le test apparié. Sur cette base-là, l'IR vaut bien ×2,4 (0,94 contre
0,39). Mais l'appariement retire déjà un facteur commun, et **les deux gains ne
se multiplient pas**.

Le classement, lui, ne bouge presque pas : corrélation de rang **+0,904** entre
les deux, et **la même combinaison en tête**. Ce que l'IR améliore est la
*résolution* — quelles combinaisons sont établies différentes —, pas l'ordre.

##### Un piège d'échelle, qui a failli me faire conclure l'inverse

Les demi-largeurs de deux métriques **ne se comparent pas**. Dans un régime
dominé par le marché, l'IR d'une stratégie vaut plusieurs fois son Sharpe, et
son intervalle est plus large dans la même proportion — ici 0,152 contre 0,097.
Lus bruts, ces deux nombres disent que l'IR sépare *moins* bien. C'est faux :
seul le rapport sans dimension (étendue / demi-largeur) se compare, et il donne
l'inverse. Un test verrouille ce piège.

**Le mécanisme du gain résiduel n'est pas établi.** L'hypothèse naturelle — que
l'appariement n'annule le marché que si les variantes le portent à l'identique,
et que leurs bêtas diffèrent (0,66 à 0,70 sur la grille) — ne s'est pas
reproduite proprement en simulation. Les tests qui prétendaient l'isoler ont été
retirés plutôt qu'ajustés jusqu'à passer : ce qui est vérifié est la mécanique
du calcul, pas l'explication.

**Le Sharpe reste le défaut.** Basculer changerait rétroactivement le sens de
tous les réglages retenus jusqu'ici, pour un gain de résolution qui ne désigne
aucun gagnant nouveau — la grille classée sur l'IR établit elle aussi **zéro**
combinaison meilleure que la production.

**La comparaison se fait contre la production, pas entre combinaisons.** La
question qui décide d'un changement n'est pas « laquelle de ces 108 gagne ? »
mais « laquelle bat ce qui tourne déjà ? ». C'est aussi pourquoi retirer un axe
de la grille ne peut jamais en retirer la valeur de production : sans elle dans
la grille, il n'y a plus rien à comparer, et l'outil le signale au lieu de
produire un classement muet.

#### Deux gains d'apprentissage écartés, et pourquoi

Le plafond par ligne à 10 % et une prise de gain élargie **gagnent en
apprentissage et perdent en test**. C'est la signature du sur-ajustement, et
c'est exactement ce que la fenêtre de validation sert à intercepter : ils ne
sont pas retenus.

Le cas de la prise de gain mérite d'être noté, parce qu'il n'a pas l'air d'un
accident : de 30 % à 100 %, le Sharpe de test monte régulièrement (DCF 0,744 →
0,785 → 0,810 ; sectorielle 0,726 → 0,762 → 0,801) pendant que celui
d'apprentissage descend, et la rotation est divisée par deux. C'est trop
monotone et trop reproductible d'une stratégie à l'autre pour être du bruit.
Mais le retenir reviendrait à **choisir sur la fenêtre de test**, qui ne vaut
que tant qu'elle n'a rien choisi : elle serait consommée, et il ne resterait
plus rien pour juger. Le sujet mérite son étude propre, avec une fenêtre de
validation neuve.

#### Ce que la zone de non-négociation corrige

Les poids sont proportionnels à l'écart de valorisation **rapporté à la somme**
des écarts des candidates : un seul dépôt SEC change ce dénominateur, donc la
cible de **toutes** les lignes. Des dépôts tombent 2624 jours sur 2936 séances
entre 2015 et 2026 — le portefeuille était repesé en entier 9 séances sur 10,
pour 722 % de rotation annualisée et 62 836 exécutions au service de 1 934
thèses seulement.

Le seuil porte sur la **dérive totale** et non ligne à ligne, et ce point a
demandé une mesure. Une bande par ligne divise bien les exécutions par 15, mais
elle filtre du même coup les **allègements**, qui sont exactement ce qui
finance les achats du même jour : 52 % du montant d'achat demandé devenait
infinançable, contre 5 % sans bande. Ou bien on repèse tout le portefeuille —
et les ventes financent les achats —, ou bien on n'y touche pas.

Une **entrée neuve** n'est jamais filtrée, et ce n'est pas un détail : avec un
plafond par ligne à 10 % et une zone à 15 points, une candidate seule pèse 10
points de dérive, reste sous le seuil, et n'est donc jamais achetée — jamais,
pas « plus tard ». En portefeuille fourni le cas est invisible, ce qui en
faisait un défaut latent ; il est couvert par un test à candidate unique
(`tests/test_reglages_sharpe_actions.py`).

### Second programme : ce que dix-huit pistes ont donné

La première étude n'avait balayé que des réglages. Celle-ci a repris le
problème par la mesure, la qualité du signal, la construction de portefeuille,
les sorties, les coûts et le risque — dix-huit pistes. **Quatre changements
seulement en sont sortis**, et trois découvertes valent plus que les gains.

#### Les découvertes

**1. Un faux positif spectaculaire, et ce qui l'a démasqué.** Plafonner le
nombre de candidates faisait monter le Sharpe de façon *monotone sur les deux
fenêtres* (0,908 → 1,192), avec un test apparié significatif jusque hors
échantillon. Tout indiquait un vrai effet. La volatilité annualisée a tranché :
**18,7 % que le portefeuille tienne 115 lignes ou 12** — impossible pour une
vraie concentration. En réalité le réglage ne concentrait rien (poids maximal
13,4 % contre 13,8 %) ; il changeait *qui* entrait, en gardant les plus fortes
convictions. Or la distribution des écarts monte jusqu'à **+1 817 436 625 %** :
plus on restreignait aux « meilleures » convictions, plus le portefeuille était
piloté par des valorisations cassées (90ᵉ centile des lignes détenues : 2 331 %
en illimité, **102 654 %** à cinq lignes).

`BACKTEST_MAX_PLAUSIBLE_GAP_PCT` (500 %) les écarte au niveau du moteur. Le
plafond de pondération bornait leur *dimensionnement*, pas leur *classement* —
et c'est le classement qui décide qui entre. Après le filtre, l'effet disparaît
(p = 0,42) et le Sharpe de référence tombe de 0,908 à 0,867. **Un filtre honnête
baisse le chiffre affiché.**

**2. Deux protections documentées ne tournaient pas.** `04c` télécharge le
texte de chaque 8-K puis le *jette* faute de clé d'API : 99 147 dépôts, 100 %
en `non_evalue`. Le filtre d'événements matériels ne s'appliquait à rien. Il
est maintenant classé **à partir du document**, sans modèle de langage — codes
d'item SEC que le déposant déclare, plus des formulations cherchées dans le
corps du texte là où le code seul est ambigu (un Item 5.02 couvre aussi bien la
démission d'un PDG que l'élection routinière d'un administrateur).

**3. Le classement d'une grille ne départage rien.** L'erreur-type d'un Sharpe
sur sept ans vaut 0,47 ; les 432 combinaisons de la grille y tiennent toutes.
`metrics.paired_sharpe_difference` compare donc chaque combinaison à la
configuration **en production** par bootstrap **apparié** — leurs courbes sont
corrélées à 0,98, et les juger à l'aune de l'erreur-type marginale revient à
déclarer « non significatif » absolument tout. Branché sur la grille, le test
fait passer celle-ci de « rien n'est distinguable » à **41 combinaisons sur 432
établies PIRES** que ce qui tourne — et aucune établie meilleure.

**4. Une décroissance conditionnelle n'est pas un effet causal.** Le rendement
annualisé des positions décroît de +79 %/an sous 30 jours à +4 %/an au-delà de
545, et la décroissance survit à tous les contrôles de biais. Elle ne dit
pourtant **pas** que les couper court rapporte : forcer une sortie ne supprime
pas la thèse, le moteur la rachète, et on paie la friction deux fois. Mesuré,
un horizon de 90 jours coûte −0,073 de Sharpe pour +55 % de rotation. La
décroissance mesure *quelles positions survivent*, pas *ce que durer coûte*.

#### Ce qui a été retenu

| Changement | Effet mesuré |
|---|---|
| **Signal sur la valorisation combinée** (`valuation_gap_combined`) | Couverture de l'univers **77 % → 94 %** |
| **Filtre de plausibilité** des écarts (500 %) | Sharpe 0,908 → 0,867 — *il retire de la performance fictive* |
| **Filtre 8-K** rendu opérant (5 646 événements) | Sharpe −0,015 : c'est le prix d'une protection |
| **Stop suiveur** à −20 % | Test **+0,120** (IC [+0,025, +0,198], p = 0,008) |

Le signal combiné n'a **pas** été retenu pour son Sharpe (+0,05, p = 0,25, non
significatif) mais pour sa couverture : un DCF n'existe pas pour une entreprise
à flux négatifs ni pour un métier de bilan, un multiple sectoriel si. Choisir
parmi 77 % de l'indice en étant jugé contre 100 % surestime l'alpha ; à 94 %,
l'alpha de +6,96 % est plus **solide** que celui de +5,58 %, indépendamment de
leur écart.

#### Ce qui a été réfuté, dont deux de mes propres recommandations

| Piste | Attendu | Mesuré |
|---|---|---|
| Pondération par le risque (`gap/σ`) | « la meilleure idée non testée » | **0,877** (k=0,5), **0,846** (k=1) contre 0,908 |
| Pondération par rang | robustesse aux extrêmes | **0,798** contre 0,867, IC [−0,115, −0,026] |
| Prise de gain élargie | piste la plus prometteuse | **−0,037** à 60 %, **−0,072** à 100 % |
| Allonger l'historique à 2012 | +26 % d'observations | couverture 73 % → 67,6 % : **de la puissance payée en biais** |
| Seuil d'entrée, âge du signal | — | axes **plats**, rien à optimiser |

Sur une stratégie *value*, là où l'écart est le plus large est aussi là où la
volatilité est la plus forte : diviser par elle retire le signal en même temps
que le risque. Et la prise de gain élargie ne tenait qu'au signal DCF et aux
valorisations cassées — une fois les deux corrigés, elle est négative.

#### Deux résultats qui attendent une décision

**La sortie sur perte de signal est activée** (`BACKTEST_EXIT_GAP_THRESHOLD_PCT
= 0`), sur décision de l'utilisateur : elle renverse la **règle des positions
gelées**, qui était un choix explicite et non un défaut technique. Elle a donc
été implémentée, mesurée, puis laissée désactivée jusqu'à ce que la décision
soit prise. Voir « Le résultat final » plus bas pour ses chiffres, dont un
drawdown qui se dégrade.

#### Le multiple mérité : mieux prédire, moins bien investir

Le seul résultat de tout le programme où deux mesures rigoureuses se
contredisent — et il mérite qu'on s'y arrête.

`15_test_multiple_merite.py` est catégorique : la régression sur les
fondamentaux prédit le multiple observé **28,4 % mieux** que la médiane
sectorielle, hors échantillon, et gagne sur 62,3 % des 17 682 observations.
Branchée dans `06b`, elle change 73,2 % des lignes. Puis l'A/B du backtest :

| | signal médiane | signal mérité |
|---|---|---|
| Sharpe plein échantillon | **0,918** | 0,787 |
| Sharpe hors échantillon | **0,795** | 0,616 |

Écart apparié **−0,134** (IC [−0,235, −0,040], p = 0,996), dont **−0,184** hors
échantillon (p = 0,990). Significativement **pire**, sur les deux fenêtres.

L'explication est au cœur de l'idée de Bhojraj & Lee poussée jusqu'au bout : le
multiple mérité **explique** la décote par les fondamentaux et ne laisse comme
signal que le résidu. Or toute la thèse d'une stratégie *value* est qu'une
partie de cette décote est une erreur de marché — et il se trouve que c'est la
part **expliquée** qui prédisait les rendements. Retirer ce que les
fondamentaux justifient retire le signal avec l'explication.

**La régression est un meilleur modèle de multiple ; elle est un moins bon
signal.** C'est précisément ce que le test hors backtest ne pouvait pas
trancher seul — d'où sa conclusion « l'étape suivante est l'A/B ». Le défaut
reste `median` ; `--multiple-method warranted` produit toujours le signal
alternatif.

#### Capacité : jusqu'à quel encours

Le coût forfaitaire de 10 bps ne dépend pas de la taille de l'ordre, donc ne
peut pas poser la question. `--impact-coefficient-bps 100` ajoute un impact en
racine de la part de volume consommée :

| Encours | 1 M$ | 100 M$ | 1 Md$ | 5 Md$ | 20 Md$ |
|---|---|---|---|---|---|
| CAGR | 17,7 % | 16,9 % | 14,6 % | 11,2 % | 6,2 % |
| Sharpe | 0,914 | 0,875 | 0,762 | 0,583 | 0,319 |

**La stratégie cesse de battre l'indice (11,99 %) vers 2 à 3 milliards de
dollars.**

#### Le résultat final

Configuration complète, `--start-date 2015-01-01`, sortie sur perte de signal
comprise :

| | Sharpe | apprentissage | **test** | **max drawdown** | CAGR | Calmar | alpha |
|---|---|---|---|---|---|---|---|
| **Départ du programme** | 0,782 | 0,837 | 0,698 | **−32,49 %** | 15,56 % | 0,479 | +3,57 % |
| `valuation_gap_dcf` | 0,935 | 1,063 | 0,737 | −32,76 % | 18,45 % | 0,563 | +6,47 % |
| `valuation_gap_sector_neutral` | 0,906 | 1,039 | 0,701 | −32,30 % | 17,69 % | 0,548 | +5,70 % |
| **`valuation_gap_combined`** | **0,977** | 1,017 | **0,910** | **−36,10 %** | **19,48 %** | 0,540 | **+7,49 %** |

Apport marginal de la seule sortie sur perte de signal, contre la
configuration complète (stop suiveur compris) : **+0,055** en plein échantillon
(IC [+0,006, +0,109], p = 0,013) et **+0,105** hors échantillon
(IC [+0,015, +0,197], p = 0,012) sur la stratégie combinée. Non significatif
sur les deux autres (p = 0,20 et 0,15), même si la direction y est la même.

Ce chiffre est plus bas que le +0,162 mesuré d'abord, et la différence n'est
pas du bruit : la première mesure comparait à une configuration **sans stop
suiveur**. Les deux sorties se recouvrent, donc l'apport marginal de celle-ci
une fois l'autre en place est plus faible. C'est l'apport marginal qui compte,
puisque c'est celui qu'on obtient en l'activant.

**Le drawdown se dégrade** : −34,4 % → −36,1 % sur la combinée. Le Calmar
s'améliore malgré tout (0,516 → 0,540) parce que le CAGR monte davantage, mais
ce réglage achète du Sharpe, pas de la tranquillité.

**Hypothèses de toutes les lignes ci-dessus** : 10 bps par aller simple
(commission 5 + glissement 5, appliqués symétriquement à l'achat et à la vente,
soit 20 bps l'aller-retour), impact de marché désactivé, **ciblage de
volatilité désactivé**. C'est le régime dans lequel toute l'optimisation a été
conduite, puisque son critère était le Sharpe. La section suivante mesure
l'autre régime, qui est désormais celui par défaut.

#### Le ciblage de volatilité à 12 %, activé

`BACKTEST_VOL_TARGET_PCT = 12.0`. L'exposition est réduite quand la volatilité
réalisée des 60 dernières séances dépasse 12 % annualisés, jamais augmentée
au-delà de 100 % (le moteur n'est pas margé). **C'est un arbitrage assumé, pas
un gain** — ce n'est pas le réglage que l'optimisation du Sharpe aurait retenu.

| | Sharpe | appr. | **test** | **max DD** | CAGR | vol | Calmar | alpha | β | exposition |
|---|---|---|---|---|---|---|---|---|---|---|
| `valuation_gap_dcf` | 0,935 | 1,063 | 0,737 | −32,76 % | 18,45 % | 17,50 % | 0,563 | +6,47 % | 0,89 | 98,2 % |
| `…` **+ cible 12 %** | 0,850 | 1,016 | 0,615 | **−24,41 %** | 14,01 % | 14,11 % | 0,574 | +2,02 % | 0,69 | 88,1 % |
| `valuation_gap_sector_neutral` | 0,906 | 1,039 | 0,701 | −32,30 % | 17,69 % | 17,30 % | 0,548 | +5,70 % | 0,88 | 98,0 % |
| `…` **+ cible 12 %** | 0,832 | 0,997 | 0,597 | **−24,14 %** | 13,65 % | 14,03 % | 0,566 | +1,66 % | 0,69 | 88,2 % |
| **`valuation_gap_combined`** | **0,977** | 1,017 | **0,910** | −36,10 % | 19,48 % | 17,65 % | 0,540 | +7,49 % | 0,89 | 98,3 % |
| **`…` + cible 12 %** | 0,931 | 0,988 | 0,846 | **−25,06 %** | 15,18 % | 13,96 % | 0,606 | +3,20 % | 0,68 | 88,4 % |

La ligne `valuation_gap_combined` **+ cible 12 %** est la configuration en
production, remesurée sur le signal `tiers` (cf. « la hiérarchie n'avait jamais
tourné » plus haut) ; la ligne sans cible n'a pas été remesurée et porte encore
sur `flat`. Les deux autres stratégies lisent le DCF seul : la correction ne les
touche pas.

Ce que le tableau dit, dans l'ordre d'importance :

- **Le drawdown baisse de 8 à 10 points**, sur les trois stratégies. C'est
  l'effet recherché, et il est net : −36,1 % → −26,5 % sur la combinée.
- **Le Sharpe baisse un peu**, jamais de façon significative. Écart apparié :
  −0,014 (combinée, p = 0,62), −0,046 (neutre au secteur, p = 0,81), −0,056
  (DCF, p = 0,85). Les trois intervalles de confiance contiennent zéro — la
  dégradation est réelle en direction, indiscernable du bruit en amplitude.
- **Le Calmar s'améliore** partout (0,540 → 0,573 sur la combinée) : le
  drawdown recule plus vite que le rendement.
- **Le CAGR et l'alpha reculent nettement** : 19,48 % → 15,22 % et +7,49 % →
  +3,23 % sur la combinée. Le garde-fou fixé au départ — battre le CAGR du SPY
  (11,99 %) — tient toujours sur les trois, mais la marge se réduit beaucoup.

**Le contrôle qui compte**, parce que le réglage baisse l'exposition moyenne de
98 % à 88 % : est-ce autre chose que « détenir moins » ? On compare donc à un
désinvestissement **constant** calibré sur la même volatilité réalisée
(k ≈ 0,79 en actions, le reste au taux sans risque).

| combinée | brut | cible 12 % | statique de même volatilité |
|---|---|---|---|
| max drawdown | −36,10 % | **−26,54 %** | −29,51 % |
| Sharpe | 0,872 | 0,799 | 0,872 |

*(Sharpe recalculés ici à formule identique pour que les trois colonnes soient
comparables ; ils diffèrent donc de ceux du tableau ci-dessus.)*

Le ciblage **fait mieux que détenir moins** sur le drawdown — 3 points
d'avance — et **moins bien sur le Sharpe**, qu'un désinvestissement constant
laisse mathématiquement intact. Le mécanisme apporte donc quelque chose de
réel, mais modeste, et il le fait payer.

**Le drawdown maximal est le Covid dans les deux régimes** (février-mars 2020),
et c'est là que le ciblage agit le plus. Sur le marché baissier de 2022, plus
lent, il ne gagne que 2 à 3 points : −17,96 % → −15,20 % sur la combinée. Un
ciblage de volatilité protège d'un choc qui dure assez pour être vu, pas d'une
baisse régulière.

Pour revenir au régime optimisé sur le Sharpe, sans toucher à la
configuration : `09_backtest.py --vol-target-pct 0`.

#### `valuation_gap_combined_ancre` : 72 % de transactions en moins

**Le constat de départ.** La stratégie combinée passe 39 044 ventes pour
2 568 thèses : **93,4 % des ventes sont des allègements de rebalancement**, pas
des décisions. La zone de non-négociation était censée les filtrer. Le
balayage montre qu'elle ne le fait pas :

| bande | Sharpe | ventes | dont rebalancement |
|---|---|---|---|
| 0 (désactivée) | 0,970 | 63 886 | 61 318 |
| 15 (retenue) | 0,977 | 39 044 | 36 476 |
| 30 / 50 / 100 / aucune | 0,978 | 39 022 | 36 454 |

Élargir la bande de 15 à l'infini change **22 ventes sur 39 044**. Ce n'est pas
un réglage, c'est un plancher.

**La cause.** `engine._drift_is_material` contenait un coupe-circuit : toute
candidate encore absente du portefeuille dont la cible dépasse le trade minimum
renvoyait `True`, donc forçait le repesage intégral quelle que soit la bande.
Comme des dépôts SEC amènent des candidates neuves 2 624 séances sur 2 936, la
bande n'était consultée que les jours sans nouveauté.

**Le changement, réservé à cette stratégie.** `Strategy` déclare désormais
`entree_neuve_force_repesage`, sur le modèle de `signal_source` — une propriété
de la thèse, pas une option d'exécution. Les trois stratégies existantes la
laissent à `True` et sont **bit-identiques** (test de non-régression de bout en
bout). `valuation_gap_combined_ancre` la met à `False` : une candidate neuve
compte alors dans la dérive comme n'importe quel écart, mais ne décide plus
seule. Le défaut latent que le coupe-circuit corrigeait — une candidate seule
jamais achetée — est repris par un amorçage sur portefeuille vide.

| | combinée | **ancrée** |
|---|---|---|
| Exécutions | 77 774 | **21 689** (−72 %) |
| Ventes de rebalancement | 36 476 | **8 376** (−77 %) |
| Séances sans repesage | 44,7 % | **85,7 %** |
| Rotation annualisée | 892 % | 733 % |
| Sharpe plein échantillon | 0,977 | 0,954 |
| Sharpe apprentissage | 1,017 | 0,965 |
| **Sharpe hors échantillon** | 0,910 | **0,935** |
| Max drawdown | −36,10 % | −35,54 % |
| CAGR | 19,48 % | 18,44 % |
| Achats infinançables | 2,23 % | **0,79 %** |

**Ce que ça établit, et ce que ça n'établit pas.** Les transactions baissent de
72 % pour un écart de Sharpe apparié de −0,019 en plein échantillon
(IC [−0,068, +0,037], p = 0,74) : **indiscernable de zéro**. La direction est
intéressante — le Sharpe d'apprentissage baisse (1,017 → 0,965) pendant que
celui de test monte (0,910 → 0,935, p = 0,20), signature d'un mécanisme qui
sur-ajustait — mais rien de tout cela n'est significatif, et il ne faut pas le
présenter autrement.

La crainte qui avait fait rejeter la bande par ligne — affamer les achats — ne
se matérialise pas : la part de montant d'achat infinançable **baisse**, de
2,23 % à 0,79 %, parce que le portefeuille cesse de dépenser son cash en
allers-retours.

**Ce qui reste à faire.** La cause première est la renormalisation de
`base.capped_weights` (`poids = conviction / somme`), qui fait qu'un seul dépôt
déplace réellement les 82 cibles. Lever le coupe-circuit ne supprime que les
repesages dont la dérive agrégée reste sous le seuil. Une pondération qui ne se
renormalise pas est l'étape suivante.

#### Tarification réelle : commission minimum et planchers de taille

Le coût du moteur était purement proportionnel. Un ordre de 7 $ y payait
0,7 centime, ce qu'aucun courtier ne facture — et c'est ce qui faisait passer
un portefeuille de 1 000 $ pour viable. Trois réglages, **tous à 0 par défaut**
(le moteur se comporte exactement comme avant, et les chiffres de référence
ci-dessus restent ceux qu'ils sont), activables par run comme `--commission-bps` :

| réglage | rôle |
|---|---|
| `--min-commission-dollar 1` | coût d'un ordre = `max(notionnel × 10 bps, 1 $)`, soit 1 $ à l'aller et 1 $ au retour |
| `--min-trade-pct-of-nav 0.05` | plancher de taille relatif au NAV — le plancher absolu de 1 $ vaut 0,000036 % d'un NAV de 2,8 M$ et ne coupe rien |
| `--max-fee-pct-of-trade 1` | critère de **viabilité** : la commission minimum ne doit pas dépasser 1 % de l'ordre, donc rien sous 100 $ n'est passé |

**Jamais sur les liquidations.** Stop-loss, take-profit, stop suiveur, perte de
signal et symbole périmé visent une cible de zéro : leur opposer un plancher
emprisonnerait toute ligne devenue plus petite que lui — le stop-loss cesserait
de fonctionner sur exactement les positions qui en ont le plus besoin, celles
qui se sont effondrées. Testé explicitement.

| run | Sharpe | test | max DD | CAGR | exécutions | ordre moyen | lignes | NAV finale |
|---|---|---|---|---|---|---|---|---|
| ancrée 10 k$ · proportionnel | 0,960 | 0,935 | −35,54 % | 18,56 % | 20 748 | 108 $ | 78,2 | 72 925 $ |
| **ancrée 10 k$ · tarif réel** | 0,840 | 0,685 | −33,23 % | 15,71 % | 5 879 | 288 $ | 63,8 | 54 913 $ |
| **combinée 10 k$ · tarif réel** | **0,906** | **0,798** | −35,17 % | 17,36 % | 7 261 | 271 $ | 68,4 | **64 765 $** |
| ancrée 1 M$ · proportionnel | 0,954 | 0,935 | −35,54 % | 18,44 % | 21 689 | 10 218 $ | 78,2 | 7 209 567 $ |
| **ancrée 1 M$ · tarif réel** | **0,981** | 0,891 | −34,34 % | 18,94 % | 14 023 | 16 734 $ | 78,7 | **7 572 498 $** |

**À 1 M$, la tarification réaliste AMÉLIORE le résultat** : Sharpe 0,954 →
0,981, CAGR 18,44 % → 18,94 %, drawdown −35,54 % → −34,34 %, et 35 % d'ordres
en moins. Le plancher relatif coupe la poussière, qui ne rapportait rien ; la
commission de 1 $ est négligeable sur un ordre moyen de 16 734 $. L'écart
apparié est +0,027 (IC [−0,016, +0,070], p = 0,095) : la direction est bonne,
la significativité n'y est pas.

**À 10 000 $, elle coûte cher, et ce coût est significatif** : −0,115 de Sharpe
(IC [−0,190, −0,047], p = 1,00). Le portefeuille se concentre — 78 lignes à
64 — parce que les petites cibles ne sont plus achetables. La stratégie bat
encore largement le SPY (54 913 $ contre 37 492 $, alpha +3,72 %), mais elle
n'est plus la même.

**Sur 2015-2026, l'ancrage ressortait perdant à 10 000 $** : −0,063 de Sharpe
contre la combinée, IC [−0,115, −0,013], p = 0,994. **Ce résultat ne survit pas
au changement de fenêtre**, et il faut le dire avant de le citer.

#### Le même test sur 10 ans glissants, et pourquoi il faut se méfier

Fenêtre 2016-09-06 → 2026-09-04 (2 514 séances), tarification réelle des deux
côtés, seule la stratégie change :

| | Sharpe | Sortino | Calmar | max DD | CAGR | alpha | exécutions | ordre moyen | NAV finale |
|---|---|---|---|---|---|---|---|---|---|
| **ancrée · 10 000 $** | 0,972 | 1,406 | 0,562 | −33,35 % | 18,75 % | +5,34 % | 5 153 | 296 $ | 55 696 $ |
| combinée · 10 000 $ | 0,948 | 1,367 | 0,544 | −34,32 % | 18,67 % | +5,26 % | 5 716 | 264 $ | 55 336 $ |
| **ancrée · 1 M$** | 1,017 | 1,472 | 0,594 | **−34,36 %** | 20,41 % | +7,00 % | **12 348** | 15 304 $ | 6 398 866 $ |
| combinée · 1 M$ | 1,017 | 1,473 | 0,584 | −36,06 % | **21,05 %** | +7,64 % | 22 388 | 9 701 $ | **6 745 200 $** |

*SPY sur la même fenêtre : CAGR 13,41 % — 10 000 $ → 35 164 $, 1 M$ → 3 516 368 $.*

**Le signe s'inverse.** À 10 000 $, l'écart apparié passe de **−0,063
(p = 0,994)** sur 2015-2026 à **+0,028 (p = 0,12)** sur 2016-2026. Mêmes
stratégies, même tarification : vingt mois de données en moins suffisent à
retourner la conclusion. Ce n'était donc pas un effet de l'ancrage, c'était un
effet de 2015-2016.

**Ce que les deux fenêtres disent en commun, et qui tient** : l'ancrage
supprime 10 à 45 % des exécutions sans effet mesurable sur le Sharpe, dans un
sens ou dans l'autre (à 1 M$ sur 10 ans : 1,017 contre 1,017). C'est le seul
énoncé que les données soutiennent.

#### Pourquoi moins de transactions ne fait pas monter le NAV

La question est la bonne, et la réponse tient en une phrase : **la friction
suit les DOLLARS négociés, pas le nombre d'ordres.** Le moteur totalise
désormais ce qu'il facture (`total_friction_dollar`, `executions_count`), ce
qu'il ne faisait pas — le moteur options le fait depuis toujours.

| 1 M$, sur 10 ans | combinée | ancrée | écart |
|---|---|---|---|
| Exécutions | 22 388 | 12 348 | **−45 %** |
| Dollars négociés | 220,4 M$ | 189,3 M$ | **−14 %** |
| **Friction payée** | 226 141 $ | 194 453 $ | **−31 688 $** |
| Coût moyen par ordre | 10,10 $ | 15,75 $ | +56 % |
| NAV finale | 6 745 200 $ | 6 398 866 $ | −346 334 $ |

Supprimer 45 % des ordres n'économise que 14 % des dollars, parce que **les
ordres supprimés sont les petits**. Le coût moyen par exécution monte donc de
10,10 $ à 15,75 $ : ce qui reste est plus gros.

La décomposition de l'écart de NAV est sans appel :

    −346 334 $  =  +31 688 $ (frais économisés)  −378 022 $ (effet de SÉLECTION)

**L'effet de sélection pèse douze fois l'économie de frais.** Différer les
entrées fait rater des positions, et ce manque à gagner écrase de très loin ce
que la friction fait gagner. À 10 000 $ le rapport s'inverse — l'économie de
frais (555 $) dépasse le coût de sélection (195 $), parce que la commission
minimum de 1 $ y représente 5,6 % du capital sur dix ans contre 3,2 % à 1 M$ —
mais les deux montants y sont dérisoires devant un NAV de 55 000 $.

Autrement dit : **réduire le nombre de transactions est un gain opérationnel,
pas un gain de performance.** Il compte pour le passage à l'exécution réelle
(carnet d'ordres, temps de gestion, risque opérationnel), pas pour le rendement
du backtest.

#### La configuration de référence sous tarification réaliste

Les chiffres de référence du tableau plus haut datent du coût purement
proportionnel. Voici la même configuration — 2015-2026, 1 M$, cible de
volatilité 12 % — mesurée avec la commission minimum de 1 $, le plancher de
0,05 % du NAV et le seuil de viabilité à 1 % :

| | Sharpe | appr. | test | Sortino | Calmar | max DD | CAGR | alpha | exécutions | NAV finale |
|---|---|---|---|---|---|---|---|---|---|---|
| Combinée · proportionnel | 0,932 | 1,006 | 0,817 | 1,315 | 0,573 | −26,54 % | 15,22 % | +3,23 % | 77 756 | 5 223 958 $ |
| **Combinée · tarif réel** | **0,930** | 1,003 | 0,817 | 1,313 | 0,571 | −26,62 % | 15,20 % | +3,21 % | **25 402** | 5 213 554 $ |
| DCF · proportionnel | 0,850 | 1,016 | 0,615 | 1,195 | 0,574 | −24,41 % | 14,01 % | +2,02 % | 75 818 | 4 620 481 $ |
| **DCF · tarif réel** | **0,851** | 1,017 | 0,617 | 1,197 | 0,573 | −24,48 % | 14,04 % | +2,05 % | **25 703** | 4 633 040 $ |
| Neutre secteur · proportionnel | 0,832 | 0,997 | 0,597 | 1,170 | 0,566 | −24,14 % | 13,65 % | +1,66 % | 71 816 | 4 453 596 $ |
| **Neutre secteur · tarif réel** | **0,832** | 0,996 | 0,598 | 1,170 | 0,565 | −24,19 % | 13,66 % | +1,67 % | **32 278** | 4 454 855 $ |

**Les chiffres de référence survivent intacts.** Écart apparié de −0,002
(combinée, p = 0,79), +0,002 (DCF), −0,000 (neutre au secteur) : indiscernable
de zéro sur les trois, avec des intervalles de confiance larges de cinq
millièmes. Les valeurs finales bougent de moins de 0,3 %.

**Pour deux tiers d'exécutions en moins.** Et c'est là que le mécanisme se
voit :

| | exécutions | friction payée | économie |
|---|---|---|---|
| Combinée | 77 756 → 25 402 (**−67 %**) | 212 516 $ → 197 441 $ (**−7 %**) | 15 075 $, soit 1,5 % du capital |
| DCF | 75 818 → 25 703 (−66 %) | 206 735 $ → 197 605 $ (−4 %) | 9 130 $ |
| Neutre secteur | 71 816 → 32 278 (−55 %) | 220 149 $ → 210 014 $ (−5 %) | 10 135 $ |

**−67 % d'ordres pour −7 % de frais** : la démonstration arithmétique de ce qui
précède. Les ordres supprimés étaient de la poussière, et la preuve qu'ils
n'étaient que ça, c'est que les retirer ne déplace pas le Sharpe d'un
millième — ni dans un sens ni dans l'autre.

**La tarification réaliste est donc ACTIVÉE PAR DÉFAUT** depuis cette mesure :
`BACKTEST_MIN_COMMISSION_DOLLAR = 1.0`, `BACKTEST_MIN_TRADE_PCT_OF_NAV = 0.05`,
`BACKTEST_MAX_FEE_PCT_OF_TRADE = 1.0`. Les trois restent réglables par run
(`--min-commission-dollar 0 …` rend le comportement purement proportionnel, qui
a produit les chiffres historiques du dépôt).

#### La pondération ancrée : mesurée, et écartée

Le dernier levier identifié, et le seul qui attaquait la racine. Plutôt que de
supprimer des ordres, il réduit l'**amplitude** de ce que chaque dépôt déplace :
`poids_i = min(conviction_i / ancre, plafond)`, sans renormalisation, au lieu de
`conviction_i / SOMME(convictions)`. L'arrivée d'une candidate laisse alors les
autres cibles strictement inchangées et le solde va en cash.

L'ancre est une ÉCHELLE, pas une cible d'allocation, et elle se calibre sur
l'exposition obtenue — trop petite, la somme des poids dépasse 1, le moteur
renormalise, et le couplage revient sans qu'on s'en aperçoive :

| ancre | exposition | Sharpe | CAGR | max DD | exécutions |
|---|---|---|---|---|---|
| 1 000 | 86,4 % | 0,802 | 13,01 % | — | 14 928 |
| 4 000 | 86,4 % | 0,857 | 13,90 % | −28,12 % | 14 962 |
| **8 000** | 79,5 % | **0,863** | 13,21 % | −25,55 % | 12 558 |
| 12 000 | 58,9 % | 0,775 | 9,85 % | −20,73 % | 8 763 |
| 16 000 | 45,0 % | 0,631 | 7,05 % | −17,39 % | 7 243 |
| 40 000 | 15,1 % | 0,104 | 2,41 % | −6,20 % | 3 603 |

En dessous de 8 000 l'ancre ne mord pas : l'exposition reste plate à 86 %, la
somme dépasse 1 et le moteur renormalise. Au-delà, le portefeuille part en cash
et le rendement s'effondre. L'optimum est à 8 000, et il ne suffit pas :

| | Sharpe | appr. | test | max DD | CAGR | exécutions | friction |
|---|---|---|---|---|---|---|---|
| **Combinée (référence)** | **0,930** | 1,003 | **0,817** | −26,62 % | **15,20 %** | 25 402 | 197 441 $ |
| Ancrée, pondération historique | 0,875 | 0,977 | 0,722 | −26,03 % | 14,08 % | 14 181 | 163 666 $ |
| Ancrée + ancre 8 000 | 0,863 | 0,951 | 0,742 | −25,55 % | 13,21 % | **12 558** | **137 976 $** |

Écarts appariés contre la référence : −0,051 (ancrage seul, p = 0,95), −0,054
(ancre 8 000, p = 0,86), −0,071 (ancre 4 000, p = 0,93). **Aucun n'est
significatif — les trois intervalles contiennent zéro — mais les trois vont
dans le même sens, et le Sharpe hors échantillon baisse aussi** (0,817 → 0,742).
Ce n'est donc pas une histoire de sur-ajustement : la pondération ancrée retire
du signal.

`BACKTEST_CONVICTION_ANCHOR` reste donc à `None`. Le mécanisme est implémenté,
calibré et testé ; il n'est pas retenu, comme la pondération par le risque et le
multiple mérité avant lui.

**Ce que tout ce fil aura établi** : les trois leviers successifs — zone de
non-négociation élargie, levée du coupe-circuit, pondération ancrée — réduisent
les transactions de 45 à 70 % sans jamais améliorer le Sharpe. La friction
n'était pas ce qui limitait cette stratégie.

L'ancrage, lui, ne suit pas : sous la même tarification et avec la cible de
volatilité, `valuation_gap_combined_ancre` rend 0,875 de Sharpe contre 0,930
(Δ = −0,051, IC [−0,108, +0,010], p = 0,95) pour 14 181 exécutions au lieu de
25 402. À la limite de la significativité, et du mauvais côté.

#### Ce qui reste bloqué

`04c` et `07b` ont besoin d'un accès à EDGAR pour reconstruire leurs fichiers à
partir des documents. Le filtre 8-K fonctionne en attendant sur l'archive déjà
écrite, par les seuls codes d'item — il n'y retient que ceux qui sont matériels
**par définition**, un code ambigu sans son texte ne disant rien.

### Ajouter une nouvelle stratégie

Créer un fichier dans `backtest/strategies/`, y définir une classe héritant
de `Strategy` (`backtest/strategies/base.py`) et décorée par
`@register_strategy("mon_nom")`, puis l'importer dans
`backtest/strategies/__init__.py`. Elle devient disponible via
`python 09_backtest.py --strategy mon_nom` sans toucher au moteur : la
stratégie ne gère que le choix des candidats et leur pondération relative,
l'engine gère uniformément le capital, le stop-loss/take-profit et les coûts
de transaction pour toutes les stratégies.

## Backtest OPTIONS (06b, 10)

Stratégie distincte de `09_backtest.py` (actions) : achète des CALL sur les
entreprises sous-évaluées, des PUT sur les survalorisées, dimensionnés par
le delta pour une exposition $ cible ("hedge par les greeks").

    06b_calcul_valorisation_combinee.py -> valorisation théorique combinée :
                                          multiples sectoriels PAR ANNÉE en
                                          priorité (cross-sectionnel, pas
                                          blendé comme 06), DCF (07) en repli
                                          quand le secteur a trop peu de pairs
    10_backtest_options.py            -> moteur de backtest options
    11_optimize_options_stops.py      -> grid-search stop-loss/take-profit sur ce moteur
    12_analyse_put_call.py            -> décomposition CALL/PUT d'un run sauvegardé
    13_diagnostic_friction.py         -> plan 2x2 thèse / friction / churn
    11b_optimize_rebalance_threshold.py -> grid-search sur ε (rebalancement sur dépôt SEC)
    11c_optimize_convergence_fraction.py -> grid-search sur la fraction de convergence
                                          (stratégie « espérance de gain »)
    compare_options_strategies.py     -> comparaison côte à côte des trois stratégies

```bash
python 05_calcul_multiples.py
python 07_calcul_dcf.py                         # avant 06b : son repli DCF
python 06b_calcul_valorisation_combinee.py
python 10_backtest_options.py --strategy valuation_gap_options --start-date 2015-01-01
```

Hypothèses du moteur (`backtest/options_engine.py`) :
    - Entrée : cherche le DERNIER snapshot RÉEL archivé par
      `08_recuperation_options.py` **au plus tard à la date d'exécution**
      (fenêtre `OPTIONS_REAL_SNAPSHOT_TOLERANCE_DAYS`, 14 jours par défaut) ;
      sinon simule par Black-Scholes (strike ATM,
      échéance 2 ans, volatilité réalisée glissante en repli). **Lance
      régulièrement `08_recuperation_options.py` sur un compte paper trading
      pour accumuler des snapshots réels au fil du temps** : plus il y en a,
      moins le backtest s'appuie sur du Black-Scholes simulé.
      Depuis l'ajout d'Alpha Vantage (source gratuite, voir la docstring de
      `08_recuperation_options.py` et `ALPHAVANTAGE_API_KEY`), deux leviers
      supplémentaires réduisent cette dépendance : (1) l'IV/greeks de chaque
      snapshot IBKR viennent en priorité d'Alpha Vantage (calculés côté
      Alpha Vantage, pas par notre propre Black-Scholes) ; (2)
      `08_recuperation_options.py --av-backfill-dates 2024-01-15 2024-02-15 ...`
      reconstitue directement un VRAI historique d'options déjà expirées
      (impossible via IBKR seul, qui ne résout plus les contrats expirés),
      sans attendre l'accumulation de runs futurs.
      `08` ne collecte que les entreprises dont l'écart de la valorisation
      COMBINÉE (`06b`, celle que tradent les stratégies options) franchit
      le seuil d'entrée de `valuation_gap_multiples_options`, en log et
      symétrique : ratio théorique/cours ≥ 1,20 ou ≤ 1/1,20. Il filtrait
      auparavant sur l'écart du DCF (`07`) : les 109 entreprises de son
      univers sans DCF (banques, assureurs, foncières) n'étaient jamais
      collectées quel que soit leur écart — 2 seulement figurent dans
      l'historique de snapshots, et seulement dans les collectes de fin
      juillet —, 56 l'étaient sans signal combiné, et la
      bande de ratio 0,80-0,833 était tradée en put sans être collectée
      (384 entreprises retenues désormais, contre 323). Ce correctif vaut
      pour la collecte À VENIR : les backtests restent presque entièrement
      simulés parce que l'historique de snapshots réels ne couvre que
      cinq semaines (2026-07-29 → 2026-09-05) et que le moteur ne regarde
      qu'en arrière — sur trois runs 2015-2026, toutes les positions
      sauf une ou deux s'ouvrent avant le premier snapshot (2 732 sur
      2 734 pour `valuation_gap_multiples_options`). Les rares positions
      ouvertes après ont été simulées faute de chaîne : un PUT COIN, que
      l'ancien filtre écartait faute de DCF, et un PUT IP, qui a pourtant
      un DCF — le filtre n'explique donc pas tout.
      Du snapshot, le moteur retient le strike, l'échéance et **l'IV** — pas
      la prime : celle-ci a été cotée à un autre spot que celui d'exécution,
      et la reprendre telle quelle faisait apparaître un saut de P&L au
      premier repricing. L'IV est la grandeur transportable d'une date à
      l'autre, le prix ne l'est pas ; la prime d'entrée en est dérivée par
      Black-Scholes au spot du jour.
    - Repricing quotidien TOUJOURS par Black-Scholes (aucune source ne fournit
      un flux d'options continu), à strike et échéance fixés à l'entrée. La
      volatilité, elle, dépend de `--vol-mode` : `frozen` (défaut, comportement
      historique) la fige à l'entrée pour toute la vie de la position ;
      `rolling` la fait suivre la volatilité réalisée du jour, remise à
      l'échelle de l'entrée (le rapport implicite/réalisé constaté à l'entrée
      est conservé, donc aucun saut de prime le premier jour). Plus l'échéance
      est longue, plus figer la volatilité est une approximation forte : le
      mode `rolling` est donc l'implicite de
      `valuation_gap_multiples_options` (échéance 2 ans), et reste optionnel
      ailleurs. Le coût en temps est nul (volatilité glissante précalculée en
      une passe vectorisée sur tout le panel de cours).
    - **Échéance 2 ans à l'entrée** (`OPTIONS_TARGET_TENOR_DAYS`), avec un
      **point de décision à 9 mois de l'expiration**
      (`OPTIONS_ROLL_WHEN_DAYS_LEFT`) : la position y est réexaminée à l'aune du
      signal courant. Écart toujours au-dessus du seuil d'entrée → le contrat
      est roulé sur une nouvelle échéance pleine, à exposition inchangée ;
      écart repassé sous le seuil (ou retourné de sens) → la position est
      clôturée (`exit_reason` `signal_lost`). On ne porte donc jamais un
      contrat sur sa dernière année de vie, là où la valeur temps s'érode le
      plus vite. `--roll-when-days-left 0` désactive ce réexamen (les positions
      vont alors jusqu'à l'expiration).
    - Stop-loss/take-profit **sur le cours du SOUS-JACENT**
      (`OPTIONS_STOP_BASIS`, `OPTIONS_STOP_LOSS_PCT`/`OPTIONS_TAKE_PROFIT_PCT`,
      −20%/+80% par défaut), orientés dans le sens de la position (pour un PUT,
      une hausse du titre est la perte) ; puis expiration (réglée à la valeur
      intrinsèque) ou disparition des données du sous-jacent. Entre deux
      réexamens de roulement, un écart qui se referme ne ferme pas la position :
      elle reste gelée jusqu'à l'un de ces déclencheurs.
      Adossés à la **prime**, ces seuils seraient atteints par la seule érosion
      de la valeur temps — une ATM à 2 ans perd ~20% de sa prime en 15 mois à
      cours strictement inchangé — et l'effet de levier ferait correspondre
      −20% de prime à une baisse du titre de quelques points seulement.
      `--stop-basis premium` rétablit l'ancienne base.
    - Le nombre de positions simultanées n'est pas plafonné, comme pour le
      moteur actions : toutes les entreprises retenues par la stratégie sont
      ouvertes. Le levier reste borné par `OPTIONS_MAX_DELTA_NOTIONAL_PCT` et
      la concentration par le plafond de pondération.
    - **Rebalancement en DEUX mécanismes disjoints** (voir la section dédiée
      plus bas) : un mécanisme JOURNALIER qui n'ouvre que des positions
      neuves, et un mécanisme SUR DÉPÔT SEC filtré par ε
      (`--rebalance-log-gap-threshold`, `OPTIONS_REBALANCE_LOG_GAP_THRESHOLD`)
      qui ne redimensionne une position déjà détenue que si son écart en log
      a suffisamment bougé depuis le dernier trade réel dessus.
    - Une nouvelle stratégie options s'ajoute de la même façon que pour les
      actions : fichier dans `backtest/strategies/`, classe héritant de
      `OptionsStrategy` (`backtest/strategies/options_base.py`), décorée par
      `@register_options_strategy("mon_nom")`.

### Stratégie `valuation_gap_multiples_options` (convergence long terme)

Seconde stratégie options, à côté de `valuation_gap_options`. Elle compare la
valorisation théorique issue des **multiples sectoriels seuls** à la
valorisation boursière, et parie sur la convergence de la seconde vers la
première à horizon 2 ans. Les deux stratégies partagent désormais l'échéance
2 ans, le roulement à 9 mois et les stops sur le sous-jacent ; ce qui les
sépare tient au signal, au strike et au traitement d'un écart refermé :

```bash
python 10_backtest_options.py --strategy valuation_gap_multiples_options --start-date 2015-01-01
```

| | `valuation_gap_options` | `valuation_gap_multiples_options` | `valuation_gap_expected_value_options` |
|---|---|---|---|
| Signal | multiples, **DCF en repli** | **multiples seuls** (repli DCF écarté) | identique à multiples |
| Mesure de l'écart | ±20% rapporté au cours | **100 × ln(théorique/cours)**, symétrique | identique à multiples |
| Strike | ATM | à mi-chemin théorique/cours | **maximise la croissance log-optimale (Kelly)** |
| Échéance | 2 ans, roulée à 9 mois | 2 ans, roulée à 9 mois | 2 ans, roulée à 9 mois |
| Stop-loss | −20% du cours du sous-jacent | −25% du cours du sous-jacent | −25% (hérité de multiples) |
| Take-profit | +80% du cours du sous-jacent | 80% du chemin vers la théorique | 80% du chemin vers la théorique |
| Écart refermé | position gelée jusqu'au **roulement à 9 mois**, où elle est clôturée | vendue au trimestre suivant | vendue au trimestre suivant |
| Volatilité de repricing | figée à l'entrée | suivie au jour le jour | suivie au jour le jour |
| Ligne écartée si… | jamais (le seuil décide seul) | jamais | **espérance de gain nette ≤ 0** |

Comme la valorisation boursière (nb d'actions × cours) et la valorisation
théorique (nb d'actions × valeur théorique par action) portent sur le même
nombre d'actions, leur rapport se calcule directement par action : la
stratégie lit les colonnes de `06b_calcul_valorisation_combinee.py` sans
reconstruire de capitalisation.

**Pourquoi les stops portent sur le sous-jacent et non sur la prime** (le
raisonnement qui vaut maintenant pour les deux stratégies, cf.
`OPTIONS_STOP_BASIS`). Sur le cas type de cette stratégie (théorique 120,
cours 100 → strike 110, 2 ans, vol 30%), le levier effectif est de 3,5x : un
stop à −25% de la prime se
déclencherait sur une baisse de seulement −7,6% du titre. Surtout, **à cours
strictement inchangé, la seule érosion de la valeur temps fait perdre 29% à
la prime en 9 mois** — le stop se déclencherait donc tout seul avant que la
convergence visée ait le temps de se produire. Appliqués au cours du
sous-jacent, les seuils décrivent bien le scénario voulu, et sont orientés
dans le sens de la position (pour un PUT, une hausse du titre est la perte).

Le strike à mi-chemin est **volontairement hors de la monnaie**, de la moitié
de l'écart : l'option ne devient gagnante que si le titre parcourt au moins
la moitié du chemin vers sa valeur théorique — une convergence partielle
suffit, un simple bruit de marché non.

**Plafond de pondération** (`OPTIONS_MULTIPLES_WEIGHT_CAP_PCT`, 100% par
défaut). Rapporté à la valeur théorique, l'écart est borné à +100% du côté
sous-évalué (le cours ne peut pas passer sous zéro) mais **non borné du côté
survalorisé** : une valeur théorique proche de zéro produit un écart de
plusieurs milliers de %. Comme le poids est proportionnel à l'écart, une
seule ligne capterait alors l'essentiel du capital — mesuré à **92% du
portefeuille** pour une théorique à 5$ contre un cours à 100$, contre 38%
une fois plafonnée. Le plafond ne s'applique qu'au **dimensionnement** : le
classement des candidats reste fait sur l'écart brut, une survalorisation
extrême restant une forte conviction. `--strategy-param weight_cap_pct=0`
le désactive.

**Réévaluation trimestrielle** : elle est automatique, sans réglage
supplémentaire. Chaque publication (10-Q via `04b_recuperation_10q.py`, 10-K
via `04`) produit un signal daté de sa date de dépôt SEC réelle ; à chacun,
les entreprises passant le seuil sont achetées et celles qui le repassent en
sens inverse sont vendues. Lance `run_pipeline_quarterly.py` pour rafraîchir
ces signaux.

Ces réglages de moteur font partie de la thèse de la stratégie : elle les
déclare (attribut de classe `engine_defaults`) et `10_backtest_options.py` les
applique automatiquement, sauf si l'option correspondante est passée
explicitement en ligne de commande (`--stop-basis`, `--roll-when-days-left`,
`--no-exit-when-signal-lost`, `--target-tenor-days`...).

Côté moteur (`backtest/options_engine.py`), l'échéance 2 ans, le roulement
(`OPTIONS_ROLL_WHEN_DAYS_LEFT`) et les stops sur le sous-jacent
(`OPTIONS_STOP_BASIS`) sont actifs par défaut, donc pour les deux stratégies ;
la vente sur perte de signal et la réévaluation quotidienne restent
**optionnelles et désactivées par défaut**, propres à
`valuation_gap_multiples_options`.

Résultats sauvegardés sous `data/backtest_options/<run_id>/` (mêmes fichiers
que le backtest actions).

### Stratégie `valuation_gap_expected_value_options` (strike par espérance de gain)

Troisième stratégie options. Elle reprend **exactement** le signal de
`valuation_gap_multiples_options` (mêmes filtres, même écart en log corrigé de
l'inflation, mêmes seuils, mêmes poids plafonnés — elle en hérite directement)
et n'en change **qu'une chose** : le strike n'est plus posé par convention, il
est choisi en maximisant le taux de croissance log-optimal du contrat. La
comparaison entre les deux ne mesure donc que l'effet de cette sélection.

```bash
python 10_backtest_options.py --strategy valuation_gap_expected_value_options --start-date 2015-01-01
python 10_backtest_options.py --strategy valuation_gap_expected_value_options \
    --start-date 2015-01-01 --strategy-param convergence_fraction=0.7
```

Pour chaque candidate, la stratégie :

1. traduit la thèse en **dérive annualisée** `mu = fraction × ln(V / S0) / T` ;
2. balaie une grille de strikes adaptative (de `S0·exp(−3σ√T)` à `S0·exp(+3σ√T)`,
   par pas de `0,25·σ√T`), **restreinte aux strikes réellement cotés** quand un
   snapshot est disponible ;
3. retient celui qui maximise `g* = max_f E[log(1 + f·R)]`, où `R = payoff/prime − 1` ;
4. **n'ouvre rien** si aucun candidat n'a d'espérance nette positive.

Les formules sont dans `backtest/expected_value.py`, module de maths pur testé
isolément (`tests/test_expected_value.py`, 122 tests). Le test qui porte tout le
reste : sous `mu = r`, l'espérance actualisée du payoff **égale le prix
Black-Scholes** (écart mesuré ~1e−14).

**Pourquoi Kelly et pas un ratio gain/risque.** C'est le point décisif, et il a
été mesuré avant d'être retenu. Un critère en ratio — Sharpe du contrat, ou
Sortino sur le risque baissier — a un optimum **dégénéré** : il ne choisit
jamais un strike intérieur, il se colle à un bord de grille.

| Critère | Comportement mesuré | Cause |
|---|---|---|
| Sharpe (écart-type) | optimum collé au bord **dans la monnaie**, pour toute grille de ±2σ à ±8σ et quelle que soit la dérive | très ITM, le payoff vaut `S_T − K` : son écart-type est celui du sous-jacent, **constant en K**, pendant que l'espérance nette croît quand K baisse |
| Sortino (semi-écart-type) | ratio de **20,3** sur un contrat qui ne paie que dans **0,1%** des scénarios | le risque baissier est **plafonné par la prime**, donc le ratio se comporte comme espérance/prime et diverge |
| Kelly | strike **intérieur**, qui se déplace continûment avec la conviction | la perte totale a une probabilité strictement positive et `log(1−f)` diverge |

Un plancher de probabilité de gain ne corrige pas les deux premiers : il colle
l'optimum à la contrainte et le fait **basculer d'un extrême à l'autre** (à
`mu = 20%`, un plancher de 30% choisit `K = 1,53·S0` ; un plancher de 40%
choisit `K = 0,28·S0`). Le seuil déciderait du strike.

Kelly, lui, est borné des deux côtés par construction. Mesuré à σ = 20% :

| `mu` | 6% | 10% | 20% | 35% |
|---|---|---|---|---|
| `K*/S0` | 0,43 | 0,43 | 0,99 | 1,42 |
| `f*` | 0,64 | 0,99 | 0,79 | 0,84 |

Les cas où Kelly retient un strike de bord (avantage faible devant la
volatilité) ne sont pas une panne : ils disent que l'avantage est trop mince
pour payer de la convexité, et qu'il vaut mieux du delta. Le Sharpe, lui,
disait cela **toujours**, y compris à conviction forte.

`g*` n'a pas de primitive : il est évalué par **quadrature de Gauss-Legendre**
(`OPTIONS_EV_QUADRATURE_NODES`, 128 nœuds) sur la seule région lucrative,
l'atome de perte totale étant traité en forme fermée. L'intégrande y est
analytique, donc la convergence est géométrique — mesuré : 1e−13 dès 32 nœuds.
Une quadrature et non un Monte-Carlo, dont le bruit d'échantillonnage ferait
changer le strike d'un run à l'autre sans qu'aucune donnée n'ait bougé.

**Transmission du strike au moteur.** Le moteur ne sait pas recevoir un strike
absolu : `strike_reference_price` est *moyenné* avec le spot d'exécution. Cette
stratégie transmet donc son strike en **moneyness** (`strike_moneyness = K*/spot`),
et laisse `strike_reference_price` à sa valeur documentée — la valeur théorique,
qui pilote la prise de gain par convergence et le rafraîchissement au roulement.

Une première version inversait la moyenne (`2K* − spot`). C'était faux sur trois
points, tous constatés en run réel :

1. **Crash.** Le prix de référence est rejoué tel quel sur les ordres en attente
   et au roulement, contre un spot qui a bougé depuis. Sous un certain niveau, la
   moyenne devenait négative et `math.log(spot/strike)` levait `ValueError` en
   plein backtest. La chute nécessaire dépend de σ, qui fixe la borne basse de la
   grille : −44 % à σ = 30 %, mais −84 % à σ = 60 %.
2. **Perte silencieuse de l'optimisation.** Quand un signal frais existait,
   `_roll_position` écrasait la référence par la valeur théorique : le contrat
   renouvelé repartait sur le strike à mi-chemin de la stratégie multiples, donc
   l'optimisation était perdue à chaque roulement (tous les 9 mois).
3. **Take-profit incohérent.** `_convergence_fraction` lit
   `strike_reference_price` comme une valeur théorique ; lui donner `2K* − spot`
   faisait viser 80 % du chemin vers une grandeur sans sens économique.

Le roulement reconduit la moneyness **sans réoptimiser** : le contrat est
recentré sur le cours du jour, mais σ et l'écart de valorisation ayant pu
changer, ce n'est plus exactement le strike que Kelly choisirait aujourd'hui.
Approximation assumée — le moteur ne sait pas redemander une optimisation en
cours de route, et la reconduction relative reste bien plus proche du choix
initial que le retour au mi-chemin.

**Fraction de convergence** (`OPTIONS_EV_CONVERGENCE_FRACTION_DEFAULT`, 0,8).
`fraction = 1` supposerait que le cours atteint exactement sa valeur théorique
à l'échéance — hypothèse que rien n'étaye. 0,5 reprendrait l'hypothèse **déjà
implicite** dans `valuation_gap_multiples_options` (dont le strike à mi-chemin
suppose exactement la moitié du chemin), rendue explicite, donc optimisable.

Le défaut est pourtant 0,8, et ce README disait 0,5 : la valeur est là depuis
le premier commit, sans trace de son origine (aucune grille `11c` archivée).
0,8 suppose une thèse nettement plus forte — au seuil d'entrée (ratio 1,20),
une dérive de 7,3 %/an sur l'échéance de 730 jours, contre 4,6 %/an à 0,5.
Mesuré le 2026-09-24 sur 2015-2026 (1 M$, `06b` régénéré en `tiers`) :

| Fraction | CAGR | Sharpe | Sortino | Max DD | Trades | Exposition |
|---|---|---|---|---|---|---|
| 0,5 | −4,0 % | −0,69 | −1,16 | −50,9 % | 2 393 | 28,5 % |
| 0,8 | −2,4 % | −0,63 | −0,99 | −42,9 % | 2 451 | 22,9 % |

Les deux valeurs sont **indiscernables** : le test apparié sur les rendements
en excès donne +0,06 de Sharpe pour 0,8, intervalle à 95 % [−0,45 ; +0,49], et
aucune des deux moitiés ne tranche (2015-2020 : −0,11 ; 2021-2026 : +0,33,
intervalle [−0,12 ; +0,83]). 0,8 perd moins en CAGR et en drawdown, mais en
investissant moins : à Sharpe égal, ce n'est pas un avantage de thèse. Sur
l'ancien parquet `flat`, encore versionné, l'écart était de +0,25
[−0,04 ; +0,60] — pas établi non plus. La production reste donc à 0,8 : on ne
la remplace que par une variante établie meilleure, et 0,5 ne l'est pas.
Surtout, **la stratégie perd aux deux valeurs**, comme
`valuation_gap_multiples_options` sur la même période (CAGR −4,0 %, Sharpe
−0,64) : ce n'est pas la fraction qui la rend négative.

**Volatilité de sélection** : l'implicite réellement cotée si un snapshot est
disponible, la volatilité réalisée sinon, et en dernier recours
`OPTIONS_FALLBACK_VOL` — la **même** valeur que celle sur laquelle le moteur se
replie. Écarter ces lignes serait plus prudent en apparence, mais biaiserait la
sélection vers les seuls titres à long historique. La part des ouvertures faites
sur IV réelle est remontée dans `metrics.json`
(`expected_value_implied_vol_pct`), à côté du nombre de lignes écartées pour
espérance négative (`dropped_expected_value_negative_count`).

**Limite fondamentale.** Tout ce que produit cette stratégie est la traduction
en dollars de `mu`, et `mu` est **estimé** — à partir d'une valeur théorique
issue de multiples sectoriels, et d'une hypothèse de convergence que rien ne
garantit. L'espérance calculée n'est pas une prédiction : c'est ce que vaudrait
le contrat **si** la thèse était juste. Un `mu` faux donne une espérance fausse
avec la même précision apparente à la douzième décimale, et le raffinement du
critère de sélection n'y change rien.

### Comparaison des trois stratégies (compare_options_strategies.py)

Rejoue les trois stratégies sur la **même période et le même univers**, et
produit un tableau côte à côte des métriques clés lues dans `metrics.json`,
plus la décomposition du P&L par motif de sortie (même fonction que
`12_analyse_put_call.py`, donc chiffres identiques à la ligne près).

```bash
python compare_options_strategies.py --start-date 2015-01-01
python compare_options_strategies.py --start-date 2015-01-01 --end-date 2024-01-01
python compare_options_strategies.py --reuse-existing        # ne rejoue rien
```

**Chaque stratégie tourne avec ses propres `engine_defaults`**, et c'est le
point du script : stops, échéance, roulement et mode de volatilité font partie
de la thèse de chacune. Les uniformiser donnerait trois variantes d'un cadre
commun que personne n'a conçu. Ce qui est uniformisé et doit l'être : la
période, l'univers, le capital initial, les coûts d'exécution et le benchmark.

Sorties : console + `data/backtest/comparisons/comparison_<horodatage>.csv`
(et `..._by_exit_reason.csv`). Attention à la lecture : trois runs sur une même
période sont trois tirages d'une **même** histoire de marché, pas trois
échantillons indépendants — un écart de Sharpe faible ne départage rien.

### Optimisation de la fraction de convergence (11c_optimize_convergence_fraction.py)

Grid-search sur `convergence_fraction`, calqué sur
`11b_optimize_rebalance_threshold.py` : données chargées une seule fois, tous
les autres réglages fixés, la fraction est le seul paramètre qui varie.

```bash
python 11c_optimize_convergence_fraction.py --start-date 2015-01-01
python 11c_optimize_convergence_fraction.py --fraction-grid 0.3 0.5 0.7 1.0 --workers 4
```

Grille par défaut : 0,2 · 0,3 · 0,4 · 0,5 · 0,6 · 0,7 · 0,8 · 1,0. CSV sous
`data/backtest_options/optimize_convergence_<stratégie>_<horodatage>.csv`.

Deux avertissements que le script émet lui-même :

- **Pas de walk-forward** à ce stade (contrairement à `11` et `11b`) : le
  classement est *in-sample*, la fraction retenue est choisie sur les données
  qui servent ensuite à la juger.
- Une fraction basse écarte beaucoup de lignes pour espérance négative et peut
  afficher un Sharpe flatteur sur une poignée de trades. Le script refuse de
  recommander une fraction sous `--min-trades` (20 par défaut).

### Optimisation stop-loss / take-profit (11_optimize_options_stops.py)

Rejoue le backtest pour **tous les binômes** (stop_loss_pct, take_profit_pct)
d'une grille, sur des données chargées une seule fois, et classe les
résultats par une métrique choisie (Sharpe par défaut) :

```bash
python 11_optimize_options_stops.py
python 11_optimize_options_stops.py --strategy valuation_gap_multiples_options
python 11_optimize_options_stops.py --start-date 2015-01-01 --objective calmar_ratio
python 11_optimize_options_stops.py \
    --stop-loss-grid -10 -15 -20 -25 -30 -35 -40 -50 \
    --take-profit-grid 20 30 40 60 80 100 150 200
python 11_optimize_options_stops.py --workers 4   # parallélise sur des process
```

**`--workers` fonctionne aussi sous Windows.** `ProcessPoolExecutor` n'y
utilise pas `fork()` (inexistant hors POSIX) : chaque worker y est un
interpréteur neuf, qui réimporte le module et ne voit donc jamais le `_DATA`
chargé par le process parent -- sans le correctif, **100 % des combinaisons**
échouaient avec `KeyError: 'price_panel'`, quel que soit `--workers`. Les
données sont désormais repassées explicitement une fois par worker
(`initializer=_pool_initializer, initargs=(_DATA,)`), ce qui fonctionne à
l'identique sur Linux (où `fork()` continue de faire le travail, gratuitement)
comme sur Windows. Même correctif dans `11b_optimize_rebalance_threshold.py`.
`rank_results` ne plante plus non plus quand *toutes* les combinaisons
échouent (ex: la même cause) : le message d'erreur prévu s'affiche, au lieu
d'un `KeyError: 'sharpe_ratio'` levé avant de l'atteindre.

Tous les autres réglages moteur (échéance, base des stops, roulement...)
restent ceux résolus par `10_backtest_options.resolve_engine_settings` --
donc ceux imposés par `engine_defaults` de la stratégie choisie, sauf
override explicite. Seuls stop_loss_pct/take_profit_pct varient d'une
combinaison à l'autre.

Sorties :
    - `data/backtest_options/optimize_<stratégie>_<horodatage>.csv` : une
      ligne par binôme, toutes les métriques (Sharpe, Sortino, Calmar, CAGR,
      max drawdown, profit factor, win rate, nombre de trades...).
    - Console : tableau des `--top-n` meilleures combinaisons, et une
      heatmap texte (stop_loss en lignes, take_profit en colonnes) pour
      repérer un plateau ou un optimum en bord de grille.

`--min-trades` (15 par défaut) écarte du CLASSEMENT (pas du CSV) les
combinaisons trop peu tradées pour être statistiquement significatives : un
stop si serré qu'il ne laisse que 3 trades peut afficher un Sharpe flatteur
par pur hasard d'échantillon. Si la meilleure combinaison retenue tombe en
bord de grille, un avertissement invite à élargir `--stop-loss-grid`/
`--take-profit-grid` -- un optimum au bord de la plage testée n'est pas prouvé
être un optimum réel.

**Séparation apprentissage / test** (`--train-fraction`, 0,60 par défaut). Le
classement se fait sur `train_<objectif>`, calculé sur les 60 % initiaux de
l'historique ; `test_<objectif>`, calculé sur le reste, est affiché à côté et
écrit au CSV — et c'est lui qui compte. Un grid-search classé sur l'historique
complet retient, par construction, la combinaison qui colle le mieux à cet
historique-là : avec 64 combinaisons sur une seule période, le meilleur Sharpe
est en grande partie du bruit sélectionné. `--min-trades` et l'avertissement de
bord de grille écartent les artefacts d'échantillon trop petit, pas le
sur-ajustement ; le seul test qui le détecte est de regarder ce que la
combinaison retenue fait sur des données qui n'ont pas servi à la choisir.

Le portefeuille n'est **pas** remis à zéro au changement de période : le run
est unique et on découpe sa courbe de NAV — les positions ouvertes à la fin de
l'apprentissage sont bien celles qu'on porterait en entrant dans le test. Un
avertissement explicite est émis si l'objectif s'effondre hors échantillon.
`--no-walk-forward` rétablit le classement in-sample.

`11b_optimize_rebalance_threshold.py` applique exactement la même séparation.

### Score d'écart symétrique (base `log`)

L'écart de `valuation_gap_multiples_options` se mesure en **points de log** :
`100 × ln(théorique / cours)`, seuil d'entrée à `100 × ln(1,20) ≈ 18,23`.

Les deux conventions en pourcentage conservées (`--strategy-param
gap_basis=theoretical` ou `close`, pour rejouer un run ancien) sont
**asymétriques en miroir l'une de l'autre** — aucune des deux n'est neutre :

| | base `theoretical` | base `close` | base `log` |
|---|---|---|---|
| côté CALL | borné à **+100%** | non borné | non borné |
| côté PUT | non borné | borné à **−100%** | non borné |
| symétrique ? | non | non | **oui** |

L'asymétrie ne déformait pas que le classement, elle déformait la
**sélection** : à seuil 20% en base `theoretical`, un CALL exigeait un cours à
≤80% de la théorique (|ln| = 0,223) là où un PUT se contentait de ≥120%
(|ln| = 0,182). Le PUT était donc structurellement plus facile à qualifier —
et recevait en prime une conviction plus élevée du côté non borné. Le livre
penchait vers le PUT par convention de calcul, pas par signal.

**Ce que le passage au log ne change pas** : la correction d'inflation reste
asymétrique, et c'est voulu. La valeur théorique est nominale, donc la
convergence se fait vers `V × (1+π)^T` — la dérive des prix aide un CALL et
durcit la thèse d'un PUT. En log cette correction devient simplement additive
(`+ T × ln(1+π)`, cf. `base.inflation_adjusted_log_gap`). Le log supprime
l'asymétrie **de convention** ; il laisse intacte celle qui a un contenu
**économique**.

### Refonte du rebalancement : deux mécanismes disjoints

Un seul point d'entrée (`_rebalance`) recalculait auparavant les poids de
TOUTES les positions détenues à CHAQUE occasion (dépôt SEC comme
réévaluation quotidienne), produisant un churn massif sans lien avec de
l'information nouvelle. Le moteur (`backtest/options_engine.py`) sépare
maintenant deux mécanismes à des périmètres volontairement différents :

1. **Mécanisme JOURNALIER** (`_rebalance_daily`, jours SANS nouveau dépôt,
   `daily_rebalance=True`) : uniquement des **ouvertures** de symboles
   nouvellement éligibles, à leur poids ISOLÉ (`conviction_X` / somme des
   convictions de tous les éligibles du jour — ce que la stratégie calcule
   déjà via `base.capped_weights`). **Aucun redimensionnement** sur une
   position déjà détenue : un mouvement de cours pur, sans nouvelle
   publication, ouvre une opportunité neuve mais ne justifie pas de retoucher
   une thèse déjà engagée.

2. **Mécanisme SUR DÉPÔT SEC** (`_rebalance_on_signals`, 10-K/10-Q/8-K) :
   scopé au(x) SEUL(S) symbole(s) dont le dépôt du jour vient de mettre à
   jour `known_signals` — jamais aux autres positions détenues, même si une
   renormalisation globale les aurait affectées. Nouveau candidat : ouvert
   sans condition. Position déjà détenue : redimensionnée seulement si
   l'écart en log a bougé de plus de `rebalance_log_gap_threshold` (ε,
   `OPTIONS_REBALANCE_LOG_GAP_THRESHOLD`, 0.15 par défaut = un rapport
   théorique/cours qui a bougé d'un facteur e^0.15 ≈ 1.16 depuis le dernier
   trade) **depuis le dernier TRADE RÉEL sur cette position**
   (`OptionPosition.last_rebalance_log_gap`), pas depuis le dernier signal
   connu. En dessous du seuil, `known_signals` est mis à jour (les calculs
   futurs utilisent la nouvelle valorisation théorique) mais aucun ordre
   n'est généré et la référence ne bouge pas : le seuil suivant continue de
   se mesurer depuis ce dernier trade, pour qu'une dérive lente qui ne
   franchit jamais ε d'un coup mais s'accumule sur plusieurs dépôts
   successifs finisse par se rattraper.

Dans les deux cas, `exit_when_signal_lost` (sortie sur perte de signal ou
retournement de direction) reste actif à son périmètre habituel : ce n'est
pas un redimensionnement, c'est une décision d'exposition orthogonale à ce
que ces deux mécanismes contraignent. `min_resize_relative_pct` reste un
**second filet** après ε : un changement de conviction qui passe ε peut
encore ne se traduire que par un micro-ajustement en nombre de contrats.

**Ce même filtre borne aussi `_deploy_idle_cash`** (le redéploiement du cash
oisif, config.OPTIONS_MIN_DEPLOYMENT_PCT — **désactivé par défaut depuis
l'audit**, voir plus bas). Cette méthode est appelée CHAQUE
jour de bourse, sans mémoire d'un renfort récent : sans le filtre, une
position à peine sous le plancher de déploiement se fait renforcer d'un ou
deux contrats CHAQUE JOUR, indéfiniment, en payant plein tarif de slippage
et de commission minimum à chaque fois — sur un backtest de plusieurs années,
cette friction pure peut consommer la quasi-totalité du capital initial,
sans qu'aucune thèse n'ait perdu quoi que ce soit. Sur un scénario de test à
2 positions / 120 jours, ce filtre à 15% fait passer les renforcements de
AAA de 18 micro-ajustements distincts à... 1 (l'ouverture initiale). `--min-
resize-relative-pct 0` rétablit l'ancien comportement (tout changement,
même infime, déclenche un ordre) pour les deux mécanismes à la fois.

**Provenance des positions** (`trades.parquet`, colonne `open_reason`) :
`"rebalance"` (dépôt SEC), `"rebalance_daily"` (mécanisme journalier) ou
`"roll"` (réouverture après roulement) — distinct d'`exit_reason`, qui décrit
toujours la SORTIE et reste inchangé par cette refonte.

```bash
python 10_backtest_options.py --strategy valuation_gap_multiples_options --rebalance-log-gap-threshold 0.20
```

`11b_optimize_rebalance_threshold.py` (calqué sur
`11_optimize_options_stops.py`) rejoue le backtest pour chaque valeur de ε
d'une grille et rapporte, par ε : `total_friction_dollar`/
`total_friction_pct_of_initial`, `num_trades` total et par motif,
`num_rebalance_trades` (le churn RÉSIDUEL du mécanisme sur dépôt SEC après
filtrage — le chiffre qui dit si ε a réellement coupé le churn), `cagr_pct`,
`sharpe_ratio`, `max_drawdown_pct` :

```bash
python 11b_optimize_rebalance_threshold.py
python 11b_optimize_rebalance_threshold.py --epsilon-grid 0 0.05 0.10 0.15 0.20 0.30
```

### Take-profit par fraction de convergence

Pour les stratégies qui visent une valeur théorique, le take-profit n'est plus
un seuil fixe mais une **fraction du chemin parcouru** vers cette valeur
(`OPTIONS_TAKE_PROFIT_CONVERGENCE_FRACTION`, 0,80 par défaut).

Un seuil fixe n'est pas atteignable de la même façon des deux côtés, par pure
géométrie. Au seuil d'entrée, la convergence **complète** vaut :

| | position entrée à | convergence complète | take-profit fixe +25% |
|---|---|---|---|
| CALL | cours = 83,3% de V | **+20,0%** de cours | atteignable |
| PUT | cours = 120% de V | **−16,7%** de cours | exige un **dépassement** |

Le take-profit fixe (+30%, ou même +25%) ne se déclenchait donc pratiquement
jamais côté PUT : la jambe ne savait pas prendre ses gains. À 0,80, un PUT
entré à P = 1,20 V sort à P = 1,04 V (−13,3% de cours) et un CALL entré à
P = 0,83 V sort à P = 0,96 V (+20,0%) — atteignable des deux côtés, et
proportionnel à l'écart réellement constaté à l'entrée.

Le **stop-loss**, lui, reste un seuil fixe en % du sous-jacent : il décrit une
perte, pas un degré d'avancement de la thèse. `valuation_gap_options` (ATM,
sans valeur théorique cible) garde `--take-profit-pct` inchangé.

Conséquence pour l'optimisation : sur une stratégie de convergence,
`take_profit_pct` est **inerte**. `11_optimize_options_stops.py` le détecte
(`OptionsStrategy.targets_convergence`) et balaie la fraction de convergence
à la place — `--take-profit-mode` force le choix, et le mode retenu est
reporté dans le CSV.

### Neutralité directionnelle de la file d'exécution

Les achats du jour s'exécutent en **alternance CALL / PUT**, chaque côté trié
par montant décroissant (les ventes passent toujours en premier : leur produit
finance les achats du même jour).

Auparavant l'ordre d'exécution suivait l'ordre d'insertion, c'est-à-dire le
classement par conviction de la stratégie. Or quand le cash s'épuise en cours
de file, `_affordable` tronque **les derniers servis** : un classement qui
place systématiquement une direction en tête la finance intégralement et
laisse l'autre absorber toute la troncature. Le classement par conviction
devenait ainsi un **filtre directionnel**, alors qu'il n'est censé exprimer
qu'une conviction.

L'alternance est **déterministe** et non aléatoire : un mélange par tirage
rendrait un run non reproductible d'une exécution à l'autre, ce qui
interdirait toute comparaison de paramètres. Une direction vide ne réserve
rien — l'autre prend tout le budget.

### Plafond par ordre (`OPTIONS_MAX_TRADE_PCT_OF_NAV`)

Aucun **ordre d'achat** ne décaisse plus de 10 % du NAV (frais inclus).
C'est un plafond **par ordre, pas par position** : une ligne peut le dépasser
en cumulant plusieurs renforcements sur des jours différents — c'est
`BACKTEST_MAX_WEIGHT_PER_POSITION_PCT` qui borne la taille d'une position.
Jamais appliqué aux **ventes** : plafonner une sortie interdirait de liquider
une position devenue grosse, exactement quand il faut pouvoir sortir. Jamais
appliqué non plus au **roulement** : celui-ci ne crée pas une position, il en
renouvelle une, et le plafonner revenait à plafonner une continuation.

Le plafond était auparavant un montant ABSOLU (15 000 $), calibré sur un
capital de 1 000 000 $ sans le dire et ne suivant ni la croissance ni la
baisse du portefeuille. Tant que le plancher de primes reconstruisait les
positions jour après jour, ce sous-dimensionnement se rattrapait tout seul ;
le plancher désactivé, il devenait la contrainte qui mord — l'ouverture visée
par le dimensionnement par delta était ramenée à 1/25e de sa taille.
`OPTIONS_MAX_TRADE_DOLLAR` existe toujours comme plafond absolu additionnel,
désactivé par défaut.

Les ordres ramenés à ce plafond sont comptés séparément des ordres tronqués
faute de cash (`capped_orders_count` vs `truncated_orders_count`) : les deux
causes n'ont rien à voir et les confondre rendrait le diagnostic illisible.
Les ordres **purement abandonnés** (frais excessifs, cash nul, plafond de
levier…) sont eux aussi comptés à part, par motif, dans
`dropped_orders_by_reason` — un ordre abandonné ne laisse aucune trace dans
`trades.parquet` ni dans l'equity_curve, et un run pouvait rester
intégralement en cash sans que rien ne le signale.

### Dimensionnement : une seule base, deux plafonds qui mordent vraiment

Voir `RAPPORT_AUDIT.md` pour les mesures. Trois réglages ont changé de valeur
par défaut, et il faut les connaître pour lire un run.

**Le plancher de primes est désactivé** (`OPTIONS_MIN_DEPLOYMENT_PCT = 0`).
Il exigeait que 25 % du NAV soit investi *en primes* — or une prime baisse
quand la thèse échoue, donc le plancher se trouvait violé précisément quand la
position perdait, et le moteur rachetait. Moins l'option valait cher, plus un
dollar achetait de contrats : le renforcement accélérait à mesure que la thèse
se dégradait, et une position gagnante n'était au contraire jamais renforcée.
Sur un CALL dont le sous-jacent perd 47 %, à signal et chemin de cours
identiques : **NAV finale 976 766 $ sans le plancher, 437 074 $ avec**. Le
mécanisme reste disponible (`--min-deployment-pct 25`) pour rejouer un run
ancien.

**Le plafond de levier borne désormais le portefeuille**, et plus seulement
le redéploiement du cash oisif. Il est vérifié à l'ouverture de chaque ordre
(`_open_or_resize`) *et* réévalué chaque jour : au-delà de
`OPTIONS_MAX_DELTA_NOTIONAL_PCT` majoré de `OPTIONS_DELEVER_TOLERANCE_PCT`,
toutes les positions sont réduites au même prorata. La bande de tolérance
évite de vendre trois contrats à chaque oscillation du marché ; la réduction
au prorata préserve la hiérarchie décidée par la stratégie, le plafond
n'exprimant aucune opinion sur la thèse à abandonner.

**Le roulement conserve l'exposition établie.** `target_dollar` enregistre
désormais l'exposition *réellement prise* — pas celle qui avait été demandée
avant que le cash, le plafond par ordre ou le plafond de levier ne rognent
l'ordre — et suit les renforcements. Sans quoi le roulement rejouait une cible
sans rapport avec la position détenue, dans un sens comme dans l'autre.

### Plancher de delta (`OPTIONS_MIN_DELTA_FOR_SIZING`)

Le moteur convertit une exposition $ visée en contrats par
`nb = target_dollar / (|delta| x spot x multiplicateur)`. Cette expression
**diverge** quand le delta tend vers zéro : à delta 0,01 elle attribue cent
fois plus de contrats qu'à delta 1,0, pour la même exposition notionnelle
affichée. Le seul garde-fou était `abs(delta) < 1e-6`, qui protège d'un
`OverflowError` mais pas de l'absurdité économique.

Sans stop-loss, les positions perdantes survivent et dérivent loin hors de la
monnaie ; leur delta et leur prime tendent vers 0, et chaque renforcement leur
attribue un nombre de contrats colossal. Mesuré sur deux runs identiques à un
paramètre près (`--stop-loss-pct -1000`) : **commissions ×26** (17 910 $ →
470 888 $) pendant que le **slippage baissait**. La commission suit le NOMBRE
de contrats, le slippage leur VALEUR — le volume avait explosé sans que la
valeur engagée bouge.

Ni le plafond par ordre ni le plafond de levier n'y suffisaient : le premier
est en dollars (sur une option à 0,02 $, 10 % d'un NAV de 1 M$ autorise
50 000 contrats), et le second a exactement la même forme que le
dimensionnement, donc il diverge avec lui.

**Le plancher ne s'applique qu'aux ACHATS.** Une position sous le plancher
doit rester vendable — par stop-loss, perte de signal, roulement, expiration
ou réduction. Le test le vérifie explicitement : le plancher est évalué
*après* le calcul du delta de contrats, uniquement dans la branche « achat ».

### Intérêts sur le cash oisif (`OPTIONS_CREDIT_IDLE_CASH`)

Le cash est capitalisé au taux sans risque de l'année, sur les jours
calendaires écoulés (base 365 : un week-end rapporte). Cette stratégie porte
en moyenne **74 % de cash** — le dimensionnement par delta n'engage qu'une
prime, soit une fraction de l'exposition — et le laisser stérile la pénalisait
pour une raison étrangère à la thèse.

C'est aussi ce biais qui rendait `OPTIONS_MIN_DEPLOYMENT_PCT` tentant : le
plancher de primes ne faisait que compenser un manque à gagner artificiel, en
payant frais et slippage pour le faire.

Les intérêts sont publiés à part (`total_cash_interest_dollar`, colonne
`total_cash_interest` de l'`equity_curve`) pour qu'une performance portée par
les taux ne se confonde pas avec une performance portée par la thèse — et
`put_call_analysis` les retire du NAV avant d'attribuer quoi que ce soit à une
jambe, sinon sa réconciliation ne boucle plus.

### Durée de détention minimale (`OPTIONS_MIN_HOLDING_DAYS`)

Les contrats sont achetés à 730 jours d'échéance, mais la durée de détention
médiane d'une sortie `signal_lost` était de **79 jours**, avec un minimum
mesuré à **un jour**. Tous motifs confondus, la moyenne est de 193 jours : la
stratégie consommait 26 % de l'optionalité qu'elle achète et jetait le reste.

Ne s'applique **jamais** aux motifs `stop_loss`, `take_profit`, `roll`,
`expiry` ni `data_gap` : un garde-fou de risque ou une échéance ne se négocie
pas contre un calendrier. À périmètre égal (`--exit-when-signal-lost` des deux
côtés), `--min-holding-days 180` fait passer la médiane `signal_lost` de 87 à
202 jours, et retire 24 % des trades comme de la friction.

### Diagnostics du dimensionnement

`metrics.json` porte désormais `total_contracts_traded`,
`max_contracts_single_order` (+ symbole et date), `min_delta_at_sizing`,
`days_above_delta_cap` / `pct_days_above_delta_cap` /
`median_excess_above_delta_cap_pct`.

Ce sont les chiffres qui auraient rendu les deux défauts ci-dessus visibles
immédiatement : le **volume de contrats** est la seule grandeur qui distingue
« j'engage plus de capital » de « j'achète des milliers d'options mortes »
— la friction totale, elle, ne le dit pas, puisque ses deux composantes
bougent alors en sens inverse.

### Hystérésis entre l'entrée et la sortie (`OPTIONS_EXIT_THRESHOLD_RATIO`)

Entrée et sortie ne partagent plus le même seuil. Une position s'ouvre à
`|écart| >= 18,23` points de log et n'est vendue qu'une fois l'écart repassé
sous `0,70 x 18,23 = 12,76` — un rapport théorique/cours de 1,136 au lieu de
1,20.

Avec la réévaluation quotidienne, un seuil unique faisait qu'un titre
oscillant autour de la barre déclenchait des allers-retours complets, chacun
payant deux fois le slippage, deux commissions minimum, et **abandonnant toute
la valeur temps déjà achetée** sur un contrat à deux ans. Le filtre ε et
`min_resize_relative_pct` protègent tous les deux le *redimensionnement* et lui
seul, jamais la décision d'ouvrir ou de fermer — qui est pourtant la plus
chère. Une convergence de 30 % n'est pas une raison de solder un pari à deux
ans : c'est le début de ce qu'on attendait. `--strategy-param
exit_threshold_ratio=1` rétablit l'ancien comportement.

### Coût de portage et exposition vega dans les métriques

`metrics.json` porte maintenant `total_theta_decay_dollar` /
`total_theta_decay_pct_of_initial` et `avg_vega_notional_pct`, et
l'`equity_curve` les colonnes `theta_per_day` / `vega_notional`.

Le theta est le **seuil que la thèse doit battre avant de gagner quoi que ce
soit** : mesuré sur un sous-jacent quasi plat, acheter des options à deux ans
coûte ~18 % de NAV par deux ans en pure valeur temps, et aucune métrique ne
l'exprimait. Le vega, lui, mesure la part du P&L que ce backtest **ne simule
pas du tout** : la stratégie achetant CALL *et* PUT, elle est longue de vega
des deux côtés, et le repricing ne suit jamais l'implicite.

### Journal des exécutions (`executions.parquet`)

`trades.parquet` n'enregistre **que les ventes**, et présente chacune comme un
aller-retour : `entry_date` y est la date de **première** ouverture de la
position et `entry_price` le prix de revient **moyen** de tous les achats qui
l'ont constituée. Or une position se construit en plusieurs fois
(renforcement au rebalancement, redéploiement du cash oisif) — un trade peut
donc légitimement solder **285 contrats** alors que **94 seulement** ont été
achetés à son `entry_date`. La comptabilité du moteur est juste (aucun contrat
n'est vendu sans avoir été acheté), mais elle était **invérifiable depuis les
sorties** : aucun achat intermédiaire n'apparaissait nulle part.

`executions.parquet` corrige ça : **une ligne par fill, achat comme vente**,
avec `contracts`, `price`, `cash_flow` (signé, frais inclus), `commission`,
`slippage` et `reason`. Trois invariants en découlent, tous testés :

| invariant | vérifie |
|---|---|
| Σ achats − Σ ventes = contrats encore détenus, par symbole | rien n'est vendu sans avoir été acheté |
| capital initial + Σ `cash_flow` = cash final | le journal explique tout le mouvement de cash |
| Σ `commission` / Σ `slippage` = totaux du moteur | aucun fill n'échappe à la friction |

**Bug corrigé au passage** : `_deploy_idle_cash` (le redéploiement du cash
oisif, appelé **chaque jour de bourse**) débitait le cash **sans jamais
comptabiliser sa friction**. `total_friction_dollar` sous-estimait donc le
coût réel sur le chemin de code le plus actif du moteur — mesuré à **44% de
la friction totale** sur un run de test. Les coûts et le journal passent
désormais par un point d'entrée unique (`_record_fill`), ce qui rend cet
oubli structurellement impossible.

### Diagnostic de friction (13_diagnostic_friction.py)

Le moteur accumule désormais la friction payée, décomposée, et la publie jour
par jour dans l'equity_curve (`total_commission`, `total_slippage`) comme en
cumul dans `metrics.json` (`total_friction_dollar`,
`total_friction_pct_of_initial`). Le slippage est compté **des deux côtés** :
la prime est payée majorée à l'achat et encaissée minorée à la vente.

`13_diagnostic_friction.py` rejoue la stratégie sur les 4 combinaisons du plan
`slippage × {réel, 0}` par `rebalancement quotidien × {activé, désactivé}` et
sépare les trois postes :

```bash
python 13_diagnostic_friction.py --strategy valuation_gap_multiples_options --start-date 2015-01-01
```

    thèse    = rendement du run slippage 0 + rebalancement désactivé
    friction = écart imputable au seul slippage, à rebalancement égal
    churn    = écart imputable au seul rebalancement quotidien, à slippage égal

Les deux ne se corrigent pas de la même façon — la friction en tradant moins
gros ou moins souvent, le churn en ne réagissant qu'aux publications — d'où
l'intérêt de ne pas les confondre dans un seul chiffre de perte. Les
commissions et frais tiers restent payés dans **les quatre cases** (les
annuler ne décrirait plus aucun courtier réel) : la colonne « pure thèse » est
un plafond, pas un contrefactuel atteignable.

### Décomposition CALL / PUT (12_analyse_put_call.py)

Le résumé de `10_backtest_options.py` agrège les deux paris en un seul NAV :
il dit combien la stratégie gagne, jamais **laquelle des deux jambes** le
gagne. Or un call parie sur une convergence à la hausse et un put sur la
baisse d'un titre survalorisé : sur un marché haussier de long terme, la
seconde peut saigner en silence pendant que la première la masque. Ce script
rouvre un run déjà sauvegardé et sépare tout ce qui peut l'être :

```bash
python 10_backtest_options.py --strategy valuation_gap_multiples_options --start-date 2015-01-01
python 12_analyse_put_call.py            # dernier run de cette stratégie
python 12_analyse_put_call.py --run-id 20260810_143000 --export
```

Sept tableaux :
    1. **Métriques habituelles en trois colonnes** ALL / CALL / PUT (mêmes
       définitions que `metrics.py`, donc lisibles ligne à ligne).
    2. **Répartition des valorisations** : prime immobilisée par chaque jambe
       (moyenne, pic, part du portefeuille, jours d'exposition).
    3. **Répartition des volumes de trade** : nombre de trades, contrats,
       montants décaissés/encaissés, P&L réalisé, et leurs parts en %.
    4. **Coûts d'exécution par jambe** : commission (frais tiers ORF/CAT/OCC/
       TAF/SEC inclus), slippage, friction totale et friction rapportée au
       volume échangé. Lus dans `executions.parquet`, donc **frais d'achat
       inclus** — invisibles dans `trades.parquet`, qui n'a que les ventes.
    5. **Coûts d'exécution par motif d'ordre** : dit quel mécanisme du moteur
       consomme la friction (`deploy_idle_cash`, `rebalance`,
       `rebalance_daily`, `roll`, `stop_loss`…).
    6. **Réconciliation des quantités** : contrats achetés / vendus / encore
       détenus par jambe, dont l'`ecart` doit valoir zéro.
    7. **P&L par (jambe, motif de sortie)** puis **P&L réalisé par année et
       par jambe** — les tableaux qui localisent réellement la perte : un
       `stop_loss` négatif dit que les seuils coupent au mauvais moment, une
       `expiry` négative que le pari ne se réalise pas dans le temps imparti.

Les tableaux 4 à 6 sont omis (avec un avertissement) pour un run antérieur à
`executions.parquet`.

`--export` écrit en plus ces tableaux en CSV dans le répertoire du run, avec
la contribution cumulée quotidienne de chaque jambe (pour voir *quand* une
jambe décroche, ce qu'aucun agrégat de fin de période ne montre).

**Ce que mesure une colonne de jambe.** La contribution d'une jambe est son
P&L réalisé cumulé + son P&L latent du jour. L'identité
`NAV = capital initial + réalisé + latent` est **exacte** (le moteur
incorpore commission et slippage à `entry_premium`), et le script la vérifie
à chaque exécution plutôt que de la supposer — il affiche l'écart de
réconciliation et avertit s'il dépasse 0,01% du capital. Pour rendre Sharpe
et drawdown calculables par jambe, chaque jambe est rejouée comme un
portefeuille partant du **capital initial complet** et ne recevant que son
propre P&L : ce n'est donc pas « ce qu'aurait donné une stratégie qui
n'achète que des calls » (le dimensionnement aurait été tout autre), mais la
**contribution** de la jambe, dans une unité comparable à l'agrégat. Seule la
ligne `leg_pnl_dollar` s'additionne entre CALL et PUT.

**Correctif** : `07_calcul_dcf.py::calculer_dcf` calculait
`equity_value = ev - dette_nette + cash`, alors que `net_debt` (produit par
`04_recuperation_10k.py`) est déjà net de cash (`dette brute - cash`) — le
cash était donc compté deux fois (valeur DCF surestimée pour les entreprises
avec beaucoup de trésorerie nette). Corrigé en `equity_value = ev - dette_nette`
(paramètre `cash` supprimé de `calculer_dcf`, devenu inutile) ; c'est la
formule que `06b_calcul_valorisation_combinee.py` utilisait déjà. Si tu as
des runs `07`/`09_backtest.py` antérieurs à ce correctif, relance
`07_calcul_dcf.py` pour régénérer des valeurs DCF exactes avant de
retro-comparer des résultats de backtest.

## Biais et limites connus

Ces points sont **documentés mais NON corrigés** : soit la donnée nécessaire
n'est pas disponible, soit la correction est un rattrapage long laissé à ta
décision. Ils vont tous dans le même sens — ils rendent les résultats de
backtest **plus optimistes** que la réalité. À garder en tête avant de
conclure quoi que ce soit d'un run.

### Médianes sectorielles calculées sur les survivants (06b)

Les multiples sont calculés par millésime de publication, et — depuis l'audit —
`compute_pit_sector_multiples` restreint en plus la médiane de chaque ligne aux
pairs **déjà déposés à sa propre `filed_date`**. Le regroupement par millésime
supprimait le mélange *entre* millésimes ; il laissait intact le décalage *à
l'intérieur* d'un millésime, où les 10-K s'étalent sur près de trois mois et où
le multiple d'un pair porte son cours à *sa* date de dépôt. La médiane servant
à valoriser un déposant de février intégrait donc les cours de ses pairs
jusqu'en avril. Conséquence assumée du correctif : les premiers déposants d'un
millésime voient moins de pairs, parfois moins que le minimum requis, et
retombent alors sur le repli DCF — c'est la réalité de l'information
disponible à cette date. Le nombre de pairs réellement utilisés est reporté
**par ligne** (`n_peers`, propagé jusqu'au signal).

Reste le biais de composition, lui **non corrigé** : les multiples viennent de
`multiples.parquet`, qui ne contient que l'univers **actuel** : les médianes
sectorielles de 2012 sont établies sur les seules entreprises encore
présentes dans l'indice aujourd'hui.

Les entreprises disparues (faillite, rachat, sortie d'indice) étaient en
moyenne moins bien valorisées que les survivantes : la médiane sectorielle
historique est donc probablement **surestimée**, et avec elle les
valorisations théoriques et les écarts calculés contre elles.

`06b` journalise le nombre de pairs par (secteur, millésime) et, si `01b` a
tourné, avertit des tickers radiés absents de `multiples.parquet`.

**Correction complète** (longue : plusieurs milliers de requêtes SEC et IBKR) :

```bash
python 01b_historique_univers_sp500.py                              # produit UNIVERSE_FULL_FILE
python 03b_recuperation_cours_quotidiens.py --tickers data/universe/sp500_universe_full.csv
python 04_recuperation_10k.py  --tickers data/universe/sp500_universe_full.csv --force-refresh
python 04b_recuperation_10q.py --tickers data/universe/sp500_universe_full.csv --force-refresh
python 05_calcul_multiples.py && python 07_calcul_dcf.py && python 06b_calcul_valorisation_combinee.py
```

### WACC indexé sur la courbe de taux (07)

*(Corrigé — le WACC était auparavant figé de 2010 à 2026.)*

`config.sector_dcf_params(secteur, année)` calcule désormais
`wacc = taux sans risque de l'année + prime de risque du secteur`, la prime
étant celle implicite dans `SECTOR_DCF_PARAMS` (WACC calibré moins
`WACC_CALIBRATION_RISK_FREE_RATE`, 4 %). L'année retenue est celle du **dépôt
SEC** — c'est au moment où l'information devient publique que le marché
actualise.

Un WACC figé était un pari de taux non voulu, et systématiquement à
contretemps : le dépôt connaît pourtant la courbe réelle
(`RISK_FREE_RATE_BY_YEAR`, de 0,05 % à 5,3 %) et s'en sert déjà pour pricer les
options et calculer le Sharpe, mais pas pour actualiser les flux — alors que
le taux est le premier déterminant d'un DCF.

| | WACC figé à 10 % | Effet sur la valeur théorique | Conséquence |
|---|---|---|---|
| 2020-2021 (taux ~0 %) | trop **haut** | sous-estimée | excès de **PUT** |
| 2023-2024 (taux ~5 %) | trop **bas** | surestimée | excès de **CALL** |

Un plancher (`DCF_MIN_WACC_MINUS_TERMINAL_GROWTH`, 3 points) garde le WACC
suffisamment au-dessus de la croissance terminale : en 2011-2015 le
sans-risque tombe à 0,05 %, et la valeur terminale
`FCF x (1+g) / (wacc - g)` cesse d'être une estimation dès que le dénominateur
s'approche de zéro. `DCF_WACC_FOLLOWS_RATE_CURVE = False` rétablit le WACC
figé.

**Régénération nécessaire** : `python 07_calcul_dcf.py` puis
`python 06b_calcul_valorisation_combinee.py`.

### Secteur GICS rétroactif (05, 07)

*(Corrigé — 06b d'abord, puis 07 lors de l'audit.)*

La colonne `sector` produite par 02 est la classification GICS
**d'aujourd'hui**. Elle pilote le WACC et les taux de croissance du DCF
(`config.SECTOR_DCF_PARAMS`), les multiples jugés pertinents
(`config.SECTOR_MULTIPLES`), le rendement du dividende du pricing d'options
(`config.SECTOR_DIVIDEND_YIELD`) et l'exclusion du DCF
(`config.SECTORS_SANS_DCF`).

`06b_calcul_valorisation_combinee.py` la ramène au secteur d'époque avant de
composer ses groupes de pairs (`sector_history.sector_asof`), et
`07_calcul_dcf.py` fait désormais de même à la `filed_date` de chaque ligne.
Le parquet de sortie porte les deux : `sector` (le secteur d'alors, celui qui
a servi au calcul, propagé jusqu'aux stratégies de backtest) et
`sector_current`.

L'effet le plus visible ne passe pas par le WACC mais par l'**exclusion** :
Visa et Mastercard sont aujourd'hui des financières, donc écartées du DCF ; en
2015 elles étaient en technologie, et un DCF y était parfaitement légitime.
Les priver de valorisation sur toute leur histoire au motif d'un reclassement
GICS de mars 2023 revient à décider avec l'avenir.

Limite résiduelle : `sector_history.GICS_RECLASSIFICATIONS` ne couvre que les
**trois remaniements structurels** de la nomenclature (immobilier 2016,
Communication Services 2018, paiements 2023) et une quarantaine de tickers
nommés. Les reclassements individuels au fil de l'eau ne le sont pas, faute de
source historique gratuite : la rigueur point-in-time est réelle mais
**partielle**.

### Taux sans risque moyen appliqué en cours d'année (07, 10)

*(Corrigé.)*

`RISK_FREE_RATE_BY_YEAR` porte des **moyennes annuelles** (3-Month T-Bill).
Actualiser un 10-K déposé en février 2020 au taux « 2020 » revient à utiliser
0,37 % — une moyenne écrasée par l'effondrement de mars, que personne ne
connaissait en février, où le T-Bill cotait encore ~1,55 %.

Le biais n'était pas centré : les années où la moyenne s'écarte le plus du
taux réel du moment sont les années de retournement, où elle s'effondre en
cours de route. Un WACC trop bas gonfle la valeur théorique, donc l'écart, donc
le nombre de signaux d'achat — **juste avant un krach**.

`config.risk_free_rate_known_at()` retient la moyenne de l'année
**précédente** (même discipline que `inflation_known_at`), et c'est elle
qu'utilisent désormais `sector_dcf_params` et le pricing d'options du
backtest. `sector_dcf_params(..., point_in_time=False)` rétablit le taux
contemporain pour reproduire un run antérieur.

Restent volontairement au taux **contemporain**, parce que ce sont des
grandeurs *ex-post* et non des décisions : le Sharpe/Sortino
(`metrics._risk_free_daily`) et les intérêts effectivement perçus sur le cash
oisif du backtest options.

### Hypothèses DCF choisies aujourd'hui (07)

**Non corrigé, et non corrigeable en l'état.** Les valeurs de
`SECTOR_DCF_PARAMS` (WACC, croissance FCF, croissance terminale par secteur)
ont été écrites à la main, aujourd'hui, en connaissant l'histoire boursière de
2010-2026. Aucune date n'est violée — mais le *choix* des paramètres, lui,
connaît la suite, et il n'est pas neutre entre secteurs : la techno reçoit 7 %
de croissance et 3 % de terminal, l'agro-alimentaire 3 % et 2 %.

Conséquence directe : à flux identique, certains secteurs ressortent
structurellement « sous-évalués », et une stratégie qui classe sur l'écart
brut classe pour partie cette table.

La parade n'est pas dans les données mais dans la stratégie :
`valuation_gap_sector_neutral` mesure l'écart en excès de la médiane du
secteur, ce qui annule tout décalage de niveau commun à un secteur — quelle
qu'en soit la cause.

### Financières traitées comme un bloc (02, 07)

*(Corrigé.)*

`GICS_TO_SECTEUR` mappait les 11 secteurs GICS un pour un, si bien que tout
« Financials » atterrissait dans « Services financiers » — un seul bucket pour
JPMorgan, Visa et Aon. Les clés `"Banques"` et `"Assurance"` de
`SECTOR_DCF_PARAMS` n'étaient donc **jamais produites** (8 des 19 clés de la
table étaient mortes), et `SECTORS_SANS_DCF` excluait les trois métiers d'un
bloc : **107 entreprises sur 503, soit 21 % de l'indice**, sans aucune
valorisation DCF.

Or la critique du FCFF ne vaut que pour les métiers de **bilan**. Elle ne
s'applique ni à Visa et Mastercard (péages à 65 % de marge, capex
négligeable), ni à S&P Global, Moody's, MSCI, FactSet, ni aux opérateurs de
marchés (CME, ICE, Nasdaq, Cboe), ni aux **courtiers** d'assurance (Aon,
Marsh, Gallagher, Brown & Brown), qui encaissent des commissions sans porter
le moindre risque au bilan.

`01_build_universe.py` récupère désormais la colonne **GICS Sub-Industry**
(elle était dans la même table Wikipedia, et simplement jetée), et
`config.GICS_SUB_INDUSTRY_TO_SECTEUR` découpe les financières selon un critère
économique — le FCFF décrit-il l'entreprise ? :

| Métier | Bucket | DCF |
|---|---|---|
| Banques, financement à la consommation, crédit hypothécaire, courtage/banque d'affaires, gestion d'actifs | `Banques` | non |
| Vie, dommages, réassurance, multiligne, holdings multi-secteurs | `Assurance` | non |
| Paiements, opérateurs de marchés et données, courtiers d'assurance | `Services financiers` | **oui** |

Deux points de méthode :

- La sous-industrie **prime sur le cache** de `02` : c'est une donnée
  officielle lue dans une table, le cache n'existe que pour éviter des appels
  LLM. Sans cette priorité, les « Services financiers » déjà écrits en bloc par
  les runs précédents auraient figé l'ancien découpage.
- Une financière dont la sous-industrie est absente ou inconnue est rabattue
  sur `Banques`, donc **exclue** du DCF. Les deux erreurs n'ont pas le même
  coût : exclure à tort un encaisseur de commissions fait perdre un signal ;
  inclure à tort un prêteur fabrique une valorisation qui ne veut rien dire, et
  sur laquelle la stratégie prendrait position. C'est le cas des entreprises
  radiées, absentes de la table Wikipedia des membres actuels — `02` les
  journalise.

### Composition point-in-time du groupe de pairs (06b)

Les médianes sectorielles de `06b` sont calculées sur les pairs tels qu'ils
étaient **à la `filed_date` de chaque ligne**, sur deux dimensions
(`sector_history.py`) :

- **Appartenance à l'indice.** Un pair n'est retenu que s'il était membre du
  S&P 500 à cette date (spans de `01b`). Cela écarte les entreprises entrées
  depuis — une entrée dans l'indice récompense en général un parcours
  boursier, donc les laisser peser sur une médiane de 2012 la pousse vers le
  haut — et, une fois les données backfillées, remet les radiées dans les
  millésimes où elles comptaient.
- **Secteur d'époque.** Les remaniements GICS documentés sont rejoués à
  l'envers : immobilier sorti des financières (2016), création de
  Communication Services (2018 — Alphabet et Meta étaient en technologie
  avant), paiements passés en financières (2023). La colonne `sector` du
  parquet de sortie porte le secteur d'alors, `sector_current` celui
  d'aujourd'hui.

`--no-point-in-time-peers` rétablit l'ancien comportement, pour chiffrer
l'écart entre les deux.

**Ce que le code ne peut pas faire seul.** Restreindre les pairs aux membres
d'alors ne crée pas les lignes manquantes : tant que `03b`/`04`/`04b` n'ont
pas été backfillés sur `sp500_universe_full.csv`, les radiées restent absentes
de `multiples.parquet`. La différence est que le trou est maintenant **mesuré**
— 06b journalise la couverture réelle de l'univers point-in-time par millésime
et avertit en dessous de 95 %.

```bash
python 01b_historique_univers_sp500.py
python 02_categoriser_secteurs.py --universe data/universe/sp500_universe_full.csv
python 03b_recuperation_cours_quotidiens.py --tickers data/universe/sp500_universe_full.csv
python 04_recuperation_10k.py  --tickers data/universe/sp500_universe_full.csv
python 04b_recuperation_10q.py --tickers data/universe/sp500_universe_full.csv
python 05_calcul_multiples.py && python 07_calcul_dcf.py
python 06b_calcul_valorisation_combinee.py
```

L'étape `02 --universe` n'est pas optionnelle : les entreprises radiées
n'ayant pas de secteur GICS (elles ne figurent plus dans la table Wikipédia
des membres actuels), sans elle le backfill coûte des milliers de requêtes SEC
pour des lignes que 06b écarte faute de secteur. `05` le signale.

Limite résiduelle : la table des changements de Wikipédia ne remonte qu'à
~1996-2000, et `sector_history.GICS_RECLASSIFICATIONS` ne couvre que les
remaniements structurels de la nomenclature — pas les reclassements
individuels au fil de l'eau, faute de source historique gratuite.

### Risque de volatilité non modélisé (10, backtest/options_pricing.py)

Le repricing quotidien des positions d'options se fait à volatilité **figée**
(`--vol-mode frozen`) ou suivant la volatilité **réalisée**
(`--vol-mode rolling`), jamais suivant la volatilité **implicite** : le
pipeline ne collecte aucune surface de volatilité historique.

Or une option longue est longue de vega. Un effondrement de l'implicite lui
fait perdre de l'argent même quand le sous-jacent va dans son sens, et ce P&L
n'apparaît nulle part dans les résultats. Les rendements du backtest options
sont donc une **borne optimiste** sur toute période de compression de
volatilité. `10_backtest_options.py` l'annonce au démarrage de chaque run.

Le pricing simulé corrige en revanche deux approximations qui allaient dans
un sens systématique : les dividendes (`config.SECTOR_DIVIDEND_YIELD`, faute
de donnée par titre) et le taux sans risque
(`config.RISK_FREE_RATE_BY_YEAR`).

### Cours IBKR sans dividendes (03, 03b)

`03_recuperation_cours.py` et `03b_recuperation_cours_quotidiens.py`
demandent `whatToShow="TRADES"` à IBKR : des cours de transaction, ajustés
des splits mais **pas des dividendes**. Le P&L du backtest actions ignore
donc les dividendes réinvestis, soit environ **2 %/an** de rendement manquant
sur le S&P 500 — davantage sur les secteurs à haut rendement, précisément
ceux qu'un signal « value » sélectionne.

Ce biais joue **contre** la stratégie (il la sous-estime), à l'inverse des
trois précédents. Il joue en revanche aussi contre l'indice de référence
reconstruit en équipondéré (`build_benchmark_series`), donc l'`alpha_pct`
reste à peu près comparable ; un `SPY` collecté par la même voie porterait la
même sous-estimation.

### Biais de survivance de l'univers (09, 10)

Si `01b_historique_univers_sp500.py` n'a jamais tourné, l'univers **actuel**
du S&P 500 est appliqué à toutes les dates passées. Les deux moteurs le
signalent par un avertissement au démarrage. Lance `01b` pour l'éliminer.

Même avec `01b`, la table des changements de Wikipédia ne remonte qu'à
~1996-2000 : une entreprise sortie de l'indice avant le début de ce suivi
n'apparaît pas.

### Biais de survivance RÉSIDUEL : cours des radiées sans leurs signaux (09)

**C'est le biais le plus coûteux du backtest actions, et le plus facile à
manquer** — parce que `01b` a tourné, que le run affiche bien « univers
point-in-time », et que rien ne semble donc clocher.

*(Corrigé pour les runs à venir — les quatre collecteurs partagent désormais
`config.default_universe_file()`. Ce qui suit décrit ce qui a produit les
données déjà en cache, et reste vrai tant que `04`/`04b` n'ont pas été
relancés.)*

Les deux moitiés de la donnée n'avaient pas le même univers par défaut :

| Script | Ancien univers par défaut | Ce qu'il alimente |
|---|---|---|
| `03b_recuperation_cours_quotidiens.py` | `UNIVERSE_FULL_FILE` **si elle existe** (actuels + radiés) | les cours — donc l'indice de référence équipondéré |
| `04_recuperation_10k.py` / `04b` / `04c` | `UNIVERSE_FILE` — l'univers **ACTUEL**, toujours | les fondamentaux — donc les signaux DCF |

Lancer le pipeline sans passer explicitement `--tickers
data/universe/sp500_universe_full.csv` à `04`/`04b` produisait donc un run où :

- l'**indice de référence** porte l'indice entier, radiées comprises ;
- la **stratégie** ne peut choisir que parmi les entreprises encore membres
  aujourd'hui, faute de signal pour les autres.

L'univers point-in-time n'y change rien : il ne fait que RESTREINDRE les
candidates, il ne crée pas les fondamentaux manquants. La stratégie se mesure
alors contre un repère qu'elle n'avait pas le droit de perdre — et un signal
« value » est précisément celui que ce biais flatte le plus, puisque les
entreprises les moins chères sont aussi celles qui sortent le plus souvent de
l'indice. **L'alpha affiché est surestimé, et il n'y a pas de moyen de savoir
de combien sans backfiller.**

Deux façons de le constater sur un run existant, sans rien relancer :

```bash
python 14_audit_backtest.py --run-id <run_id>   # section 1
```

et, dans `metrics.json`, `exits_by_reason` : **aucune sortie `data_gap`**
signifie qu'aucune position détenue n'a jamais cessé d'être cotée sur toute la
période — ce qui n'arrive pas dans un vrai S&P 500 sur quinze ans.

Depuis l'audit, `09_backtest.py` mesure la couverture lui-même et l'écrit dans
`metrics.json` (`signal_coverage_avg_ratio`, `signal_coverage_min_ratio`,
`signal_coverage_min_year`), avec un avertissement en dessous de 95 %.

**Correction** (longue au premier passage : plusieurs milliers de requêtes SEC
pour les radiées absentes du cache ; `should_skip` ignore ensuite tout ticker
déjà à jour). `--tickers` n'est plus nécessaire sur `03b`/`04`/`04b`/`04c` —
ils prennent l'univers point-in-time dès que `01b` a tourné :

```bash
python 01b_historique_univers_sp500.py
python 02_categoriser_secteurs.py --universe data/universe/sp500_universe_full.csv
python 03b_recuperation_cours_quotidiens.py
python 04_recuperation_10k.py
python 04b_recuperation_10q.py
python 05_calcul_multiples.py && python 07_calcul_dcf.py
python 09_backtest.py
```

L'étape `02 --universe` reste explicite : elle met à jour le fichier d'univers
**sur place**, ce n'est pas une simple lecture.

### Exposition delta : plafonnée en continu, avec un jour de retard (10)

*(Corrigé — cette section décrivait auparavant une limite bien plus large :
le plafond n'était vérifié que dans `_deploy_idle_cash`, et 281 % de delta
notionnel ont été mesurés pour un plafond déclaré à 100 %.)*

`config.OPTIONS_MAX_DELTA_NOTIONAL_PCT` borne désormais le levier **à l'ordre
et à la position** : vérifié dans `_open_or_resize`, puis réévalué chaque jour
de bourse, avec réduction au prorata au-delà de la bande de tolérance
(`OPTIONS_DELEVER_TOLERANCE_PCT`).

Il reste un dépassement résiduel, et il est structurel : comme tout le reste
du moteur, le dé-levier est **décidé à la clôture de J et exécuté à
l'ouverture de J+1**. Entre les deux, le sous-jacent bouge et le NAV avec lui —
or le ratio a le NAV au dénominateur, si bien qu'un portefeuille qui perd voit
son levier monter sans avoir rien acheté. Sur un scénario adverse (sous-jacent
en baisse de 47 %), le maximum observé passe de 281 % à ~112 % pour un plafond
à 100 % et une tolérance de 10 %. La colonne `delta_notional_pct` de
l'`equity_curve` et les champs `avg_delta_notional_pct` /
`max_delta_notional_pct_observed` / `delever_events_count` de `metrics.json`
permettent de le constater run par run.

## Installation

```bash
pip install -r report/requirements.txt
```

## Lancement

**Important : lance la commande depuis la racine du dépôt** (là où se
trouvent `config.py` et `01_build_universe.py`), pas depuis `report/` —
`config.py` résout ses chemins (`./data/...`) relativement au répertoire
d'exécution, exactement comme les scripts `01` à `08`.

```bash
streamlit run report/Home.py
```

## Pages

- **📊 Data** — tableau de couverture par entreprise : années de cours,
  exercices 10-K, contrats d'options collectés, dernière mise à jour de
  chaque source. Filtres par secteur / recherche / couverture complète.
- **📈 Analyse** — pour l'entreprise sélectionnée dans la barre latérale :
  nappe de volatilité implicite (lissée par processus gaussien,
  scikit-learn), structure par terme, skew, greeks par strike, indicateurs
  de liquidité (spread, volume, OI, put/call, détection d'anomalies par
  IsolationForest), et repères de valorisation (multiples sectoriels, DCF, et
  la valorisation combinée de `06b_calcul_valorisation_combinee.py` utilisée
  par la stratégie options). En bas de page : clustering KMeans des multiples
  de valorisation sur
  l'ensemble du portefeuille suivi, avec projection PCA 2D et recoupement
  avec le secteur GICS déclaré.

## Historique des nappes de volatilité

Depuis le correctif apporté à `08_recuperation_options.py`, chaque run
archive en plus un snapshot horodaté dans `data/options/history/`
(jamais écrasé), en parallèle du fichier courant `data/options/option_chains.parquet`
(toujours écrasé par le run suivant). La page Analyse lit l'historique
complet et propose un sélecteur de date dès qu'au moins deux snapshots sont
disponibles pour l'entreprise choisie — donc **relance `04` plusieurs fois
dans le temps** (ex: une fois par semaine) pour voir la nappe évoluer.

Avant le premier run depuis ce correctif, `data/options/history/` est vide :
le rapport retombe alors sur le seul snapshot courant.

`08_recuperation_options.py --av-backfill-dates AAAA-MM-JJ ...` peuple aussi
`data/options/history/` directement avec de vraies dates passées (source
Alpha Vantage, gratuite avec clé), sans attendre l'accumulation de runs
futurs : voir la docstring en tête de `08_recuperation_options.py`.
