"""Classification des thèses d'un run : qu'ont en commun les perdantes, et
qu'ont en commun les gagnantes ?

Relit les sorties d'un run (09_backtest.py, ou 10_backtest_options.py) SANS le
relancer, reconstruit une ligne par THÈSE, l'étiquette gagnante ou perdante,
la décrit par tout ce qui était connu le jour de la décision, puis cherche ce
qui sépare les deux groupes. Le script d'appel est 18_classification_trades.py.

QUATRE CHOIX DE MÉTHODE, ET POURQUOI.

1. L'UNITÉ EST LA THÈSE, pas l'exécution. `trades.parquet` logue une ligne par
   VENTE, allègements de rebalancement compris : une thèse perdante de
   -226 k$ y apparaissait comme 32 trades gagnants à 91 % (audit, défaut A3).
   Classer des exécutions apprendrait donc surtout à reconnaître les lignes
   qu'on allège, pas les paris qui perdent. La clé est (symbol, entry_date),
   la même que metrics._position_level_metrics.

2. L'ÉTIQUETTE PAR DÉFAUT EST L'ALPHA DE LA THÈSE : son rendement moins celui
   de l'indice de référence du run (SPY) sur les mêmes dates. Avec un P&L
   brut, les « perdantes » sont d'abord les thèses ouvertes juste avant un
   krach, et le modèle apprend des DATES, pas des titres. `--label pnl` et
   `--label extremes` restent disponibles.

3. LES VARIABLES DU MODÈLE SONT TOUTES CONNUES À LA CLÔTURE DE LA VEILLE DE
   L'ENTRÉE, c'est-à-dire au moment où le moteur décide (il exécute à
   l'ouverture suivante). Tout ce qui s'observe PENDANT la détention -- durée,
   motif de sortie, pire baisse, 8-K survenus -- est calculé à part
   (préfixe `pendant_`) et ne sert qu'à DÉCRIRE : le donner au modèle
   reviendrait à lui montrer la réponse, et aucune règle trouvée ainsi ne
   serait utilisable pour décider d'une entrée.

4. LA VALIDATION EST TEMPORELLE, comme partout dans le dépôt : apprentissage
   sur les thèses CLOSES avant la date de coupure (2022-01-01 par défaut,
   celle de 16_optimize_strategie_actions.py), test sur les thèses OUVERTES
   après. Celles qui sont à cheval sont écartées des deux -- leur issue est
   connue en partie grâce à la période de test. Les intervalles de confiance
   sont tirés par bootstrap PAR MOIS D'ENTRÉE et non par thèse : les thèses
   ouvertes le même mois partagent le même marché, et les rééchantillonner
   une à une ferait croire à beaucoup plus d'observations indépendantes
   qu'il n'y en a.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from backtest import data_loader

logger = logging.getLogger("backtest.classification_trades")

DATE_COUPURE_DEFAUT = "2022-01-01"

# Fenêtres de prix, en séances.
SEANCES_MOIS = 21
SEANCES_TRIMESTRE = 63
SEANCES_SEMESTRE = 126
SEANCES_AN = 252

# 8-K comptés AVANT la décision, par code d'item. Les codes matériels de
# config, plus ceux que config déclare ambigus (5.02 : départ d'un dirigeant
# ou élection routinière ?) : ici on ne périme rien, on mesure seulement si
# leur présence distingue les perdantes -- un code ambigu peut très bien le
# faire en moyenne.
ITEMS_8K_SUIVIS = ("1.01", "2.02", "2.03", "5.02", "7.01", "8.01")
FENETRE_8K_JOURS = 180

# Variables numériques dont la distribution est trop étirée pour qu'une
# médiane ou un quintile lisent autre chose que les extrêmes : on les passe
# en log signé. Les arbres y sont insensibles, pas les tableaux du rapport.
_ITEM_RE = re.compile(r"(\d+\.\d+)")


# --------------------------------------------------------------------------- #
# Chargement d'un run
# --------------------------------------------------------------------------- #
def resoudre_run(run_id: Optional[str] = None, run_dir: Optional[Path] = None) -> Path:
    """Dossier du run à analyser : explicite, sinon cherché par nom dans les
    sorties actions PUIS options, sinon le plus récent des runs actions."""
    if run_dir is not None:
        return Path(run_dir)
    racines = [config.DIR_BACKTEST, config.DIR_BACKTEST_OPTIONS]
    if run_id is not None:
        for racine in racines:
            if (racine / run_id / "trades.parquet").exists():
                return racine / run_id
        raise FileNotFoundError(f"Aucun run '{run_id}' avec trades.parquet dans {racines}.")
    runs = sorted(p for p in config.DIR_BACKTEST.glob("*") if (p / "trades.parquet").exists())
    if not runs:
        raise FileNotFoundError(
            f"Aucun run exploitable dans {config.DIR_BACKTEST}. Lance d'abord 09_backtest.py.")
    return runs[-1]


@dataclass
class Run:
    dossier: Path
    trades: pd.DataFrame
    positions_history: pd.DataFrame
    equity_curve: pd.DataFrame
    run_config: dict = field(default_factory=dict)


def charger_run(dossier: Path) -> Run:
    def parquet(nom: str) -> pd.DataFrame:
        chemin = dossier / f"{nom}.parquet"
        return pd.read_parquet(chemin) if chemin.exists() else pd.DataFrame()

    chemin_config = dossier / "run_config.json"
    run_config = json.loads(chemin_config.read_text(encoding="utf-8")) if chemin_config.exists() else {}
    trades = parquet("trades")
    if trades.empty:
        raise ValueError(f"{dossier / 'trades.parquet'} vide ou absent : rien à classer.")
    for col in ("entry_date", "exit_date"):
        trades[col] = pd.to_datetime(trades[col])
    positions = parquet("positions_history")
    for col in ("date", "entry_date"):
        if col in positions.columns:
            positions[col] = pd.to_datetime(positions[col])
    equity = parquet("equity_curve")
    if "date" in equity.columns:
        equity["date"] = pd.to_datetime(equity["date"])
    return Run(dossier, trades, positions, equity, run_config)


# --------------------------------------------------------------------------- #
# 1. Une ligne par thèse
# --------------------------------------------------------------------------- #
CLE_THESE = ["symbol", "entry_date"]

# Colonnes de trades.parquet propres aux runs OPTIONS, fixées à l'ouverture de
# la position et donc connues à la décision : elles deviennent des variables
# catégorielles quand elles existent.
COLONNES_OUVERTURE = ("option_type", "open_reason")


def construire_theses(trades: pd.DataFrame, positions_history: pd.DataFrame) -> pd.DataFrame:
    """Regroupe les ventes par thèse et écarte celles qui sont ENCORE OUVERTES
    à la fin du run.

    Une thèse allégée par le rebalancement puis toujours détenue au dernier
    jour a des lignes dans trades.parquet, mais son issue n'est pas connue :
    la compter sur ses seules ventes partielles reviendrait à juger un pari à
    mi-parcours, et toujours dans le même sens (on allège ce qui a monté).

    Rendement de la thèse = P&L total / coût de revient total des titres
    vendus. `entry_price` étant le prix de revient MOYEN au moment de chaque
    vente (renforts compris, frais inclus), la somme `shares x entry_price`
    reconstitue exactement le capital engagé dans les titres vendus."""
    t = trades.copy()
    t["cout"] = t["shares"] * t["entry_price"]
    t = t.sort_values(["symbol", "entry_date", "exit_date"], kind="stable")
    agregats = {
        "exit_date": ("exit_date", "max"),
        "pnl": ("pnl", "sum"),
        "cout": ("cout", "sum"),
        "pendant_nb_ventes": ("pnl", "size"),
        "pendant_motif_sortie": ("exit_reason", "last"),
    }
    for col in COLONNES_OUVERTURE:
        if col in t.columns:
            agregats[col] = (col, "first")
    theses = t.groupby(CLE_THESE, sort=False).agg(**agregats).reset_index()
    theses["rendement_pct"] = theses["pnl"] / theses["cout"] * 100
    theses["pendant_duree_jours"] = (theses["exit_date"] - theses["entry_date"]).dt.days

    if not positions_history.empty and {"date", "entry_date", "symbol"} <= set(positions_history.columns):
        derniere = positions_history["date"].max()
        ouvertes = positions_history.loc[positions_history["date"] == derniere, CLE_THESE]
        avant = len(theses)
        theses = theses.merge(ouvertes.assign(_ouverte=True), on=CLE_THESE, how="left")
        theses = theses[theses["_ouverte"].isna()].drop(columns="_ouverte")
        if avant - len(theses):
            logger.info("%d thèse(s) encore ouverte(s) en fin de run écartée(s) : issue inconnue.",
                        avant - len(theses))
    return theses.reset_index(drop=True)


def ajouter_rendement_indice(theses: pd.DataFrame, trades: pd.DataFrame, indice: pd.Series) -> pd.DataFrame:
    """Rendement de l'indice de référence sur la vie de chaque thèse, PONDÉRÉ
    comme le rendement de la thèse : chaque vente compte pour son coût de
    revient, sur ses propres dates. Une thèse allégée tôt puis soldée tard a
    eu la majeure partie de son capital exposée au marché sur la période
    courte : la comparer à l'indice sur toute sa vie fausserait son alpha."""
    indice = indice.dropna().sort_index()

    def niveau(dates: pd.Series) -> np.ndarray:
        pos = indice.index.searchsorted(dates.to_numpy(), side="right") - 1
        valeurs = np.full(len(dates), np.nan)
        ok = pos >= 0
        valeurs[ok] = indice.to_numpy()[pos[ok]]
        return valeurs

    t = trades[CLE_THESE + ["exit_date", "shares", "entry_price"]].copy()
    t["cout"] = t["shares"] * t["entry_price"]
    t["r_indice"] = niveau(t["exit_date"]) / niveau(t["entry_date"]) - 1
    t["pondere"] = t["r_indice"] * t["cout"]
    agg = t.groupby(CLE_THESE).agg(pondere=("pondere", "sum"), cout=("cout", "sum")).reset_index()
    agg["rendement_indice_pct"] = agg["pondere"] / agg["cout"] * 100
    out = theses.merge(agg[CLE_THESE + ["rendement_indice_pct"]], on=CLE_THESE, how="left")
    out["alpha_pct"] = out["rendement_pct"] - out["rendement_indice_pct"]
    return out


def etiqueter(theses: pd.DataFrame, label: str = "alpha", extremes_pct: float = 30.0) -> pd.DataFrame:
    """Colonne `gagnante` (1/0). `extremes` garde les `extremes_pct` % de
    meilleures et de pires thèses EN ALPHA et écarte le milieu, où gagnant et
    perdant ne se distinguent que par du bruit."""
    out = theses.copy()
    if label == "alpha":
        out["gagnante"] = (out["alpha_pct"] > 0).astype(float)
    elif label == "pnl":
        out["gagnante"] = (out["pnl"] > 0).astype(float)
    elif label == "extremes":
        bas, haut = np.nanpercentile(out["alpha_pct"], [extremes_pct, 100 - extremes_pct])
        out["gagnante"] = np.where(out["alpha_pct"] >= haut, 1.0,
                                   np.where(out["alpha_pct"] <= bas, 0.0, np.nan))
    else:
        raise ValueError(f"Étiquette inconnue : {label!r} (attendu alpha, pnl ou extremes).")
    out.loc[out["alpha_pct"].isna() & (label != "pnl"), "gagnante"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# 2. Variables connues à la décision
# --------------------------------------------------------------------------- #
@dataclass
class Donnees:
    """Tables du pipeline lues une fois. Chacune est facultative : une
    variable dont la source manque est simplement absente du rapport."""
    panel: data_loader.PricePanel
    indice: Optional[pd.Series]
    signaux: Optional[pd.DataFrame] = None
    fondamentaux: Optional[pd.DataFrame] = None
    evenements_8k: Optional[pd.DataFrame] = None
    univers: Optional[pd.DataFrame] = None
    verdicts_07b: Optional[pd.DataFrame] = None


def _lire(chemin: Path) -> Optional[pd.DataFrame]:
    if not chemin.exists():
        logger.warning("%s absent : les variables qui en dépendent sont ignorées.", chemin)
        return None
    try:
        return pd.read_parquet(chemin)
    except Exception as exc:  # pointeur Git LFS non rapatrié, fichier corrompu
        logger.warning("%s illisible (%s) -- pointeur LFS ? `git lfs pull`. Variables ignorées.",
                       chemin, exc)
        return None


def charger_donnees(benchmark_symbol: Optional[str] = None) -> Donnees:
    panel = data_loader.build_price_panel(data_loader.load_daily_prices())
    univers = data_loader.UniverseResolver(
        data_loader.load_universe_history(), data_loader.load_current_universe_symbols())
    indice, libelle = data_loader.build_benchmark_series(panel, univers, benchmark_symbol)
    logger.info("Indice de référence de l'alpha : %s", libelle)
    return Donnees(
        panel=panel,
        indice=indice,
        signaux=_lire(config.VALORISATION_COMBINEE_FILE),
        fondamentaux=_lire(config.MULTIPLES_FILE),
        evenements_8k=_lire(config.MATERIAL_EVENTS_8K_FILE),
        univers=data_loader.load_universe_history(),
        verdicts_07b=_lire(config.QUALITATIVE_VALIDATION_FILE),
    )


def dates_de_decision(entrees: pd.Series, calendrier: pd.DatetimeIndex) -> pd.Series:
    """Séance dont la CLÔTURE a décidé l'entrée : la dernière strictement
    avant le jour d'exécution (le moteur exécute à l'ouverture de J+1 ce qu'il
    a décidé à la clôture de J)."""
    pos = calendrier.searchsorted(entrees.to_numpy(), side="left") - 1
    dates = np.where(pos >= 0, calendrier.to_numpy()[np.clip(pos, 0, None)], np.datetime64("NaT"))
    return pd.Series(pd.to_datetime(dates), index=entrees.index)


def _merge_asof(gauche: pd.DataFrame, droite: pd.DataFrame, cle_droite: str) -> pd.DataFrame:
    """Dernière ligne de `droite` publiée au plus tard à la date de décision,
    par symbole. Les deux côtés doivent être triés sur leur clé de temps."""
    # Types alignés des deux côtés : merge_asof refuse un symbole `string`
    # face à un `object`, et deux résolutions de datetime différentes.
    g = gauche.rename_axis("_ligne").reset_index().dropna(subset=["date_decision"])
    g = g.assign(symbol=g["symbol"].astype(object),
                 date_decision=g["date_decision"].astype("datetime64[ns]"))
    d = droite.dropna(subset=[cle_droite, "symbol"])
    d = d.assign(symbol=d["symbol"].astype(object), **{cle_droite: d[cle_droite].astype("datetime64[ns]")})
    fusion = pd.merge_asof(g.sort_values("date_decision", kind="stable"),
                           d.sort_values(cle_droite, kind="stable"),
                           left_on="date_decision", right_on=cle_droite,
                           by="symbol", direction="backward")
    return fusion.set_index("_ligne").reindex(gauche.index)


def _ratio(num, den) -> pd.Series:
    num, den = pd.to_numeric(num, errors="coerce"), pd.to_numeric(den, errors="coerce")
    out = num / den
    return out.where(np.isfinite(out) & (den != 0))


def variables_signal(theses: pd.DataFrame, signaux: pd.DataFrame, cours_decision: pd.Series) -> pd.DataFrame:
    """Le signal sur lequel la thèse a été ouverte : le dernier publié à la
    date de décision, exactement ce que `engine.known_signals` détenait.

    `ecart_pct` est celui du moteur, figé au cours du jour de DÉPÔT ;
    `ecart_a_la_decision_pct` le recalcule au cours de la décision. Les deux
    diffèrent d'autant que le cours a bougé depuis le dépôt -- et c'est
    peut-être là qu'une thèse se perd : un écart qui s'est déjà refermé
    quand on achète."""
    s = signaux.copy()
    s["filed_date"] = pd.to_datetime(s["filed_date"])
    s = s.rename(columns={"sector": "secteur"})
    garder = ["symbol", "filed_date", "secteur", "period_type", "close", "gap_pct", "source",
              "n_multiples_used", "n_peers", "valuation_theoretical_per_share",
              "valuation_multiples_per_share", "valuation_dcf_per_share",
              "price_from_ev_ebitda", "price_from_ev_sales", "price_from_pe"]
    s = s[[c for c in garder if c in s.columns]]
    f = _merge_asof(theses[["symbol", "date_decision"]], s, "filed_date")

    def col(nom: str) -> pd.Series:
        # Colonne facultative : absente d'un cache ancien, elle vaut NaN et la
        # variable devient constante, donc ignorée par colonnes_modele.
        return f[nom] if nom in f.columns else pd.Series(np.nan, index=f.index)

    out = pd.DataFrame(index=theses.index)
    out["ecart_pct"] = f["gap_pct"]
    theorique = f["valuation_theoretical_per_share"]
    out["ecart_a_la_decision_pct"] = (_ratio(theorique, cours_decision) - 1) * 100
    out["cours_depuis_depot_pct"] = (_ratio(cours_decision, f["close"]) - 1) * 100
    out["age_signal_jours"] = (theses["date_decision"] - f["filed_date"]).dt.days
    out["nb_multiples"] = col("n_multiples_used")
    out["nb_pairs"] = col("n_peers")
    # Désaccord entre les deux méthodes de valorisation : quand le DCF et les
    # multiples racontent des histoires opposées, l'écart repose sur un seul.
    out["dcf_vs_multiples_pct"] = (_ratio(col("valuation_dcf_per_share"),
                                          col("valuation_multiples_per_share")) - 1) * 100
    # Dispersion des trois prix implicites (EV/EBITDA, EV/Sales, P/E) : un
    # écart que les trois multiples confirment n'est pas celui qu'un seul porte.
    implicites = f[[c for c in ("price_from_ev_ebitda", "price_from_ev_sales", "price_from_pe")
                    if c in f.columns]].where(lambda x: x > 0)
    if implicites.shape[1]:
        out["dispersion_multiples"] = _ratio(implicites.max(axis=1), implicites.min(axis=1))
    out["secteur"] = f["secteur"]
    out["source_valorisation"] = col("source")
    out["type_periode"] = col("period_type")
    return out


def variables_fondamentales(theses: pd.DataFrame, fondamentaux: pd.DataFrame,
                            cours_decision: pd.Series) -> pd.DataFrame:
    """Comptes du dernier 10-K/10-Q déposé à la décision (multiples.parquet :
    annuel et TTM, tels que 05 les a assemblés), rapportés au cours DU JOUR
    de la décision pour tout ce qui est un rendement ou un multiple."""
    f = fondamentaux.copy()
    f["filed_date"] = pd.to_datetime(f["filed_date"], errors="coerce")
    f = f.dropna(subset=["filed_date"])
    # Croissance sur un an : même type de période, même trimestre, exercice
    # précédent -- jamais un TTM comparé à un exercice annuel.
    trimestre = f["fiscal_quarter"].astype("object").where(f["fiscal_quarter"].notna(), "") \
        if "fiscal_quarter" in f.columns else ""
    exercice = f["fiscal_year"] if "fiscal_year" in f.columns else f["year"]
    f = f.assign(_trim=trimestre, _ex=pd.to_numeric(exercice, errors="coerce"))
    f = f.sort_values(["symbol", "period_type", "_trim", "_ex", "filed_date"], kind="stable")
    f = f.drop_duplicates(["symbol", "period_type", "_trim", "_ex"], keep="last")
    groupe = f.groupby(["symbol", "period_type", "_trim"], sort=False)
    consecutif = groupe["_ex"].diff() == 1
    for col in ("revenue", "ebitda", "net_income"):
        precedent = groupe[col].shift()
        f[f"_croiss_{col}"] = (_ratio(f[col] - precedent, precedent.abs()) * 100).where(consecutif)

    garder = ["symbol", "filed_date", "revenue", "ebit", "ebitda", "net_income", "fcf", "gross_profit",
              "total_assets", "total_liabilities", "cash", "net_debt", "interest_expense", "capex",
              "current_assets", "current_liabilities", "shares_outstanding",
              "_croiss_revenue", "_croiss_ebitda", "_croiss_net_income"]
    m = _merge_asof(theses[["symbol", "date_decision"]], f[[c for c in garder if c in f.columns]],
                    "filed_date")

    out = pd.DataFrame(index=theses.index)
    capitalisation = cours_decision * m["shares_outstanding"]
    out["log_capitalisation"] = np.log(capitalisation.where(capitalisation > 0))
    out["croissance_ca_pct"] = m["_croiss_revenue"]
    out["croissance_ebitda_pct"] = m["_croiss_ebitda"]
    out["croissance_resultat_pct"] = m["_croiss_net_income"]
    out["marge_brute_pct"] = _ratio(m["gross_profit"], m["revenue"]) * 100
    out["marge_ebitda_pct"] = _ratio(m["ebitda"], m["revenue"]) * 100
    out["marge_nette_pct"] = _ratio(m["net_income"], m["revenue"]) * 100
    out["marge_fcf_pct"] = _ratio(m["fcf"], m["revenue"]) * 100
    out["rendement_fcf_pct"] = _ratio(m["fcf"], capitalisation) * 100
    out["rendement_benefice_pct"] = _ratio(m["net_income"], capitalisation) * 100
    valeur_entreprise = capitalisation + m["net_debt"].fillna(0)
    out["ve_ebitda"] = _ratio(valeur_entreprise, m["ebitda"]).where(m["ebitda"] > 0)
    out["ve_ca"] = _ratio(valeur_entreprise, m["revenue"]).where(m["revenue"] > 0)
    out["dette_nette_ebitda"] = _ratio(m["net_debt"], m["ebitda"]).where(m["ebitda"] > 0)
    out["ebitda_negatif"] = (m["ebitda"] <= 0).astype(float).where(m["ebitda"].notna())
    out["passif_actif_pct"] = _ratio(m["total_liabilities"], m["total_assets"]) * 100
    out["liquidite_generale"] = _ratio(m["current_assets"], m["current_liabilities"])
    out["roa_pct"] = _ratio(m["net_income"], m["total_assets"]) * 100
    out["couverture_interets"] = _ratio(m["ebit"], m["interest_expense"].abs())
    out["capex_ca_pct"] = _ratio(m["capex"].abs(), m["revenue"]) * 100
    out["tresorerie_capitalisation_pct"] = _ratio(m["cash"], capitalisation) * 100
    out["age_comptes_jours"] = (theses["date_decision"] - m["filed_date"]).dt.days
    return out


class _Panneaux:
    """Indicateurs de prix calculés UNE fois sur tout le panel (date x
    symbole), puis lus en (ligne, colonne). Chaque indicateur à la ligne r ne
    lit que les clôtures jusqu'à r incluse : point-in-time par construction."""

    def __init__(self, panel: data_loader.PricePanel, indice: Optional[pd.Series]):
        close = panel.close
        self.lignes = {d: i for i, d in enumerate(close.index)}
        self.colonnes = {s: j for j, s in enumerate(close.columns)}
        c = close.to_numpy(dtype=float)

        def decale(k: int) -> np.ndarray:
            out = np.full_like(c, np.nan)
            out[k:] = c[:-k]
            return out

        with np.errstate(divide="ignore", invalid="ignore"):
            self.ret_1m = c / decale(SEANCES_MOIS) - 1
            self.ret_3m = c / decale(SEANCES_TRIMESTRE) - 1
            self.ret_6m = c / decale(SEANCES_SEMESTRE) - 1
            self.mom_12_1 = decale(SEANCES_MOIS) / decale(SEANCES_AN) - 1
            haut = close.rolling(SEANCES_AN, min_periods=SEANCES_TRIMESTRE).max().to_numpy()
            bas = close.rolling(SEANCES_AN, min_periods=SEANCES_TRIMESTRE).min().to_numpy()
            self.depuis_plus_haut = c / haut - 1
            self.depuis_plus_bas = c / bas - 1
        self.vol_60 = panel.realized_vol_panel(60)
        self.volume_dollar = panel._dollar_volume_values

        rendements = close.pct_change(fill_method=None)
        self.beta = None
        self.marche = None
        if indice is not None:
            m = indice.reindex(close.index).ffill()
            rm = m.pct_change(fill_method=None)
            fen = SEANCES_AN
            moy_rm = rm.rolling(fen, min_periods=SEANCES_SEMESTRE).mean()
            var_rm = rm.rolling(fen, min_periods=SEANCES_SEMESTRE).var()
            cov = rendements.mul(rm, axis=0).rolling(fen, min_periods=SEANCES_SEMESTRE).mean() \
                - rendements.rolling(fen, min_periods=SEANCES_SEMESTRE).mean().mul(moy_rm, axis=0)
            # cov de population sur var d'échantillon : écart de (n-1)/n, sans
            # effet sur un classement.
            self.beta = cov.div(var_rm, axis=0).to_numpy()
            self.marche = pd.DataFrame({
                "marche_3m_pct": (m / m.shift(SEANCES_TRIMESTRE) - 1) * 100,
                "marche_12m_pct": (m / m.shift(SEANCES_AN) - 1) * 100,
                "marche_vol_60j_pct": np.log(m).diff().rolling(60, min_periods=20).std() * np.sqrt(252) * 100,
                "marche_depuis_plus_haut_pct": (m / m.rolling(SEANCES_AN, min_periods=20).max() - 1) * 100,
            })
            self.ret_3m_vs_marche = (self.ret_3m - (m / m.shift(SEANCES_TRIMESTRE) - 1).to_numpy()[:, None])

    def lire(self, tableau: Optional[np.ndarray], r: np.ndarray, j: np.ndarray) -> np.ndarray:
        out = np.full(len(r), np.nan)
        if tableau is None:
            return out
        ok = (r >= 0) & (j >= 0)
        out[ok] = tableau[r[ok], j[ok]]
        return out


def variables_prix(theses: pd.DataFrame, panneaux: _Panneaux) -> pd.DataFrame:
    r = theses["date_decision"].map(panneaux.lignes).fillna(-1).astype(int).to_numpy()
    j = theses["symbol"].map(panneaux.colonnes).fillna(-1).astype(int).to_numpy()
    lire = panneaux.lire
    out = pd.DataFrame(index=theses.index)
    out["rendement_1m_pct"] = lire(panneaux.ret_1m, r, j) * 100
    out["rendement_3m_pct"] = lire(panneaux.ret_3m, r, j) * 100
    out["rendement_6m_pct"] = lire(panneaux.ret_6m, r, j) * 100
    out["momentum_12_1_pct"] = lire(panneaux.mom_12_1, r, j) * 100
    out["depuis_plus_haut_1an_pct"] = lire(panneaux.depuis_plus_haut, r, j) * 100
    out["depuis_plus_bas_1an_pct"] = lire(panneaux.depuis_plus_bas, r, j) * 100
    out["volatilite_60j_pct"] = lire(panneaux.vol_60, r, j) * 100
    volume = lire(panneaux.volume_dollar, r, j)
    out["log_volume_dollar"] = np.log(np.where(volume > 0, volume, np.nan))
    if panneaux.beta is not None:
        out["beta_1an"] = lire(panneaux.beta, r, j)
        out["rendement_3m_vs_marche_pct"] = lire(panneaux.ret_3m_vs_marche, r, j) * 100
        marche = panneaux.marche.reindex(theses["date_decision"]).to_numpy()
        for k, col in enumerate(panneaux.marche.columns):
            out[col] = marche[:, k]
    return out


def _codes_items(valeur) -> set:
    if valeur is None:
        return set()
    items = valeur if isinstance(valeur, (list, tuple, np.ndarray)) else [valeur]
    return {m.group(1) for item in items if (m := _ITEM_RE.search(str(item)))}


def _compteur_8k(evenements: pd.DataFrame):
    """Dates triées des 8-K par symbole (tous, matériels, et par item suivi),
    pour compter dans une fenêtre en O(log n)."""
    ev = evenements.dropna(subset=["symbol", "filed_date"]).copy()
    ev["filed_date"] = config.to_naive_day(ev["filed_date"])
    ev = ev.dropna(subset=["filed_date"])
    codes = ev["item_codes"].map(_codes_items) if "item_codes" in ev.columns else pd.Series(
        [set()] * len(ev), index=ev.index)
    materiels = set(getattr(config, "MATERIAL_8K_ITEM_CODES", ()))
    familles = {"tous": pd.Series(True, index=ev.index),
                "materiels": codes.map(lambda c: bool(c & materiels))}
    for item in ITEMS_8K_SUIVIS:
        familles[item] = codes.map(lambda c, i=item: i in c)
    index = {}
    for nom, masque in familles.items():
        sous = ev[masque.to_numpy(dtype=bool)]
        index[nom] = {s: np.sort(g["filed_date"].to_numpy(dtype="datetime64[ns]"))
                      for s, g in sous.groupby("symbol")}
    return index


def _compter(index: dict, symboles: pd.Series, debut: pd.Series, fin: pd.Series) -> np.ndarray:
    """Nombre de dépôts dans ]debut, fin], par ligne."""
    out = np.zeros(len(symboles))
    for k, (s, a, b) in enumerate(zip(symboles, debut, fin)):
        dates = index.get(s)
        if dates is None or pd.isna(a) or pd.isna(b):
            continue
        out[k] = np.searchsorted(dates, np.datetime64(b), "right") - np.searchsorted(dates, np.datetime64(a), "right")
    return out


def variables_8k(theses: pd.DataFrame, evenements: pd.DataFrame) -> pd.DataFrame:
    index = _compteur_8k(evenements)
    d = theses["date_decision"]
    fenetre = d - pd.Timedelta(days=FENETRE_8K_JOURS)
    an = d - pd.Timedelta(days=365)
    out = pd.DataFrame(index=theses.index)
    out["nb_8k_180j"] = _compter(index["tous"], theses["symbol"], fenetre, d)
    out["nb_8k_materiels_1an"] = _compter(index["materiels"], theses["symbol"], an, d)
    for item in ITEMS_8K_SUIVIS:
        out[f"nb_8k_item_{item}_180j"] = _compter(index[item], theses["symbol"], fenetre, d)
    # Descriptif seulement : survenus PENDANT la détention.
    out["pendant_nb_8k_materiels"] = _compter(index["materiels"], theses["symbol"],
                                              theses["entry_date"], theses["exit_date"])
    out["pendant_nb_8k"] = _compter(index["tous"], theses["symbol"],
                                    theses["entry_date"], theses["exit_date"])
    return out


def variables_univers(theses: pd.DataFrame, historique: pd.DataFrame) -> pd.DataFrame:
    """Ancienneté dans l'indice à la décision : une entrée récente dans le
    S&P 500 suit souvent un parcours boursier exceptionnel, une ancienneté
    longue un titre mûr."""
    h = historique.dropna(subset=["ric"]).copy()
    h["symbol"] = h["ric"].map(config.to_ib_symbol)
    h["start_date"] = pd.to_datetime(h["start_date"], errors="coerce")
    debut = h.groupby("symbol")["start_date"].min()
    anciennete = (theses["date_decision"] - theses["symbol"].map(debut)).dt.days / 365.25
    return pd.DataFrame({"anciennete_indice_ans": anciennete}, index=theses.index)


def variables_07b(theses: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    """Dernier verdict qualitatif (07b) connu à la décision. Le plus souvent
    `non_evalue` tant qu'aucune clé LLM n'a tourné : la variable est alors
    constante et disparaît d'elle-même de l'analyse."""
    v = verdicts.copy()
    v["filed_date"] = config.to_naive_day(v["filed_date"])
    m = _merge_asof(theses[["symbol", "date_decision"]], v[["symbol", "filed_date", "verdict"]], "filed_date")
    return pd.DataFrame({"verdict_07b": m["verdict"]}, index=theses.index)


def variables_portefeuille(theses: pd.DataFrame, run: Run) -> pd.DataFrame:
    """Contexte du portefeuille au jour de l'entrée : combien de lignes il
    tenait, quelle part en cash, quel poids la thèse a reçu, combien
    d'entrées le même jour. Décidés par la stratégie à la clôture de la
    veille, donc connus avant l'exécution."""
    out = pd.DataFrame(index=theses.index)
    eq = run.equity_curve
    if not eq.empty:
        e = eq.set_index("date")
        out["nb_positions_portefeuille"] = theses["entry_date"].map(e["num_positions"])
        out["cash_portefeuille_pct"] = theses["entry_date"].map(e["cash"] / e["nav"] * 100)
        ph = run.positions_history
        if not ph.empty and {"market_value", "entry_date"} <= set(ph.columns):
            premiere = ph[ph["date"] == ph["entry_date"]][CLE_THESE + ["market_value"]]
            poids = theses[CLE_THESE].merge(premiere, on=CLE_THESE, how="left")["market_value"].to_numpy()
            out["poids_initial_pct"] = poids / theses["entry_date"].map(e["nav"]).to_numpy() * 100
    out["nb_entrees_meme_jour"] = theses.groupby("entry_date")["symbol"].transform("size")
    return out


def variables_pendant(theses: pd.DataFrame, run: Run, indice: Optional[pd.Series]) -> pd.DataFrame:
    """Ce qui s'est passé PENDANT la détention : jamais dans le modèle (cf.
    en-tête), seulement dans la description des deux groupes."""
    out = pd.DataFrame(index=theses.index)
    ph = run.positions_history
    if not ph.empty and "unrealized_return_pct" in ph.columns:
        chemin = ph.groupby(CLE_THESE)["unrealized_return_pct"].agg(["min", "max"]).reset_index()
        m = theses[CLE_THESE].merge(chemin, on=CLE_THESE, how="left")
        out["pendant_pire_baisse_pct"] = m["min"].to_numpy()
        out["pendant_meilleure_hausse_pct"] = m["max"].to_numpy()
    if "rendement_indice_pct" in theses.columns:
        out["pendant_marche_pct"] = theses["rendement_indice_pct"]
    return out


def construire_variables(theses: pd.DataFrame, run: Run, donnees: Donnees) -> pd.DataFrame:
    """Toutes les variables, une ligne par thèse. Les colonnes `pendant_*`
    sont descriptives ; toutes les autres sont connues à la décision."""
    theses = theses.copy()
    theses["date_decision"] = dates_de_decision(theses["entry_date"], donnees.panel.close.index)
    r = theses["date_decision"].map({d: i for i, d in enumerate(donnees.panel.close.index)})
    j = theses["symbol"].map({s: k for k, s in enumerate(donnees.panel.close.columns)})
    cours = np.full(len(theses), np.nan)
    ok = r.notna() & j.notna()
    cours[ok.to_numpy()] = donnees.panel.close.to_numpy()[r[ok].astype(int), j[ok].astype(int)]
    cours_decision = pd.Series(cours, index=theses.index)

    blocs = [variables_prix(theses, _Panneaux(donnees.panel, donnees.indice)),
             variables_portefeuille(theses, run)]
    if donnees.signaux is not None:
        blocs.append(variables_signal(theses, donnees.signaux, cours_decision))
    if donnees.fondamentaux is not None:
        blocs.append(variables_fondamentales(theses, donnees.fondamentaux, cours_decision))
    if donnees.evenements_8k is not None:
        blocs.append(variables_8k(theses, donnees.evenements_8k))
    if donnees.univers is not None:
        blocs.append(variables_univers(theses, donnees.univers))
    if donnees.verdicts_07b is not None:
        blocs.append(variables_07b(theses, donnees.verdicts_07b))
    blocs.append(variables_pendant(theses, run, donnees.indice))
    return pd.concat([theses] + blocs, axis=1)


# Colonnes qui décrivent la thèse sans être des variables explicatives.
COLONNES_HORS_MODELE = {
    "symbol", "entry_date", "exit_date", "date_decision", "pnl", "cout", "rendement_pct",
    "rendement_indice_pct", "alpha_pct", "gagnante",
}


def colonnes_modele(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(numériques, catégorielles) utilisables par le modèle : ni étiquette,
    ni issue, ni rien de `pendant_`, ni variable constante."""
    numeriques, categorielles = [], []
    for col in df.columns:
        if col in COLONNES_HORS_MODELE or col.startswith("pendant_"):
            continue
        serie = df[col]
        if serie.nunique(dropna=True) < 2:
            continue
        if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie):
            numeriques.append(col)
        else:
            categorielles.append(col)
    return numeriques, categorielles


def matrice(df: pd.DataFrame, numeriques: list[str], categorielles: list[str],
            modalites: Optional[dict] = None) -> tuple[pd.DataFrame, dict]:
    """Matrice du modèle : numériques telles quelles (NaN conservés, les deux
    modèles d'arbres les gèrent), catégorielles en indicatrices. Les modalités
    sont fixées sur l'APPRENTISSAGE et réappliquées au test : une modalité
    vue seulement au test n'a rien à apprendre au modèle."""
    X = df[numeriques].astype(float).copy()
    if modalites is None:
        modalites = {c: sorted(df[c].dropna().astype(str).unique()) for c in categorielles}
    for c in categorielles:
        valeurs = df[c].astype("object").where(df[c].notna(), None)
        for m in modalites.get(c, []):
            X[f"{c}={m}"] = (valeurs.astype(str) == m).astype(float)
    return X, modalites


# --------------------------------------------------------------------------- #
# 3. Ce qui distingue les deux groupes
# --------------------------------------------------------------------------- #
def decouper(df: pd.DataFrame, date_coupure: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """Masques apprentissage / test. Une thèse ouverte avant la coupure et
    close après n'appartient à aucun des deux : son étiquette dépend de la
    période de test."""
    etiquetee = df["gagnante"].notna()
    apprentissage = etiquetee & (df["exit_date"] < date_coupure)
    test = etiquetee & (df["entry_date"] >= date_coupure)
    return apprentissage, test


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Aire sous la courbe ROC = probabilité qu'une gagnante tirée au hasard
    ait un score plus élevé qu'une perdante. 0,5 = aucun pouvoir ; les
    valeurs manquantes sont écartées."""
    from sklearn.metrics import roc_auc_score
    y, score = np.asarray(y, dtype=float), np.asarray(score, dtype=float)
    ok = ~np.isnan(score) & ~np.isnan(y)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], score[ok]))


def auc_bootstrap(y: np.ndarray, score: np.ndarray, grappes: np.ndarray,
                  n: int = 1000, graine: int = 0) -> tuple[float, float]:
    """Intervalle à 95 % de l'AUC, rééchantillonné PAR GRAPPE (mois
    d'entrée) : les thèses d'un même mois partagent le même marché et ne
    sont pas des tirages indépendants."""
    rng = np.random.default_rng(graine)
    y, score, grappes = np.asarray(y), np.asarray(score), np.asarray(grappes)
    uniques = np.unique(grappes)
    membres = [np.flatnonzero(grappes == g) for g in uniques]
    valeurs = []
    for _ in range(n):
        tirage = rng.integers(0, len(uniques), len(uniques))
        idx = np.concatenate([membres[k] for k in tirage])
        a = auc(y[idx], score[idx])
        if a == a:
            valeurs.append(a)
    if not valeurs:
        return np.nan, np.nan
    return float(np.percentile(valeurs, 2.5)), float(np.percentile(valeurs, 97.5))


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    """q-valeurs de Benjamini-Hochberg : avec quarante variables testées, deux
    seront « significatives à 5 % » par pur hasard. La q-valeur contrôle la
    part de fausses découvertes parmi celles qu'on retient."""
    p = p.astype(float)
    ok = p.notna()
    q = pd.Series(np.nan, index=p.index)
    if not ok.any():
        return q
    v = p[ok].sort_values()
    m = len(v)
    brut = v.to_numpy() * m / np.arange(1, m + 1)
    ajuste = np.minimum.accumulate(brut[::-1])[::-1].clip(max=1.0)
    q[v.index] = ajuste
    return q


def analyse_univariee(df: pd.DataFrame, numeriques: list[str], apprentissage: pd.Series,
                      test: pd.Series) -> pd.DataFrame:
    """Variable par variable : médiane chez les gagnantes et chez les
    perdantes, AUC de la variable SEULE sur l'apprentissage puis sur le test.

    La colonne qui compte est `stable` : significative après correction des
    tests multiples sur l'apprentissage, PUIS confirmée au test dans le même
    sens (test unilatéral orienté par l'apprentissage, à 5 %). « Du même côté
    de 0,5 » ne suffit pas : une AUC de 0,502 l'est aussi.
    Un effet qui s'inverse d'une période à l'autre n'est pas un point commun
    des perdantes, c'est une coïncidence de la première période."""
    from scipy.stats import mannwhitneyu
    rows = []
    y = df["gagnante"]
    for col in numeriques + [c for c in df.columns if c.startswith("pendant_")
                             and pd.api.types.is_numeric_dtype(df[c])]:
        x = pd.to_numeric(df[col], errors="coerce")
        a = apprentissage & x.notna()
        g, p_ = x[a & (y == 1)], x[a & (y == 0)]
        if len(g) < 10 or len(p_) < 10:
            continue
        pval = float(mannwhitneyu(g, p_, alternative="two-sided").pvalue)
        # Au test, l'hypothèse est ORIENTÉE par l'apprentissage : on ne
        # demande pas « y a-t-il un écart ? » mais « l'écart vu avant
        # tient-il, dans le même sens ? ».
        t_ = test & x.notna()
        g_t, p_t = x[t_ & (y == 1)], x[t_ & (y == 0)]
        sens = "greater" if auc(y[a], x[a]) >= 0.5 else "less"
        p_test = (float(mannwhitneyu(g_t, p_t, alternative=sens).pvalue)
                  if len(g_t) >= 10 and len(p_t) >= 10 else np.nan)
        rows.append({
            "variable": col,
            "descriptive": col.startswith("pendant_"),
            "n_apprentissage": int(a.sum()),
            "couverture_pct": float(x[apprentissage | test].notna().mean() * 100),
            "mediane_gagnantes": float(g.median()),
            "mediane_perdantes": float(p_.median()),
            "auc_apprentissage": auc(y[a], x[a]),
            "auc_test": auc(y[test], x[test]),
            "p_valeur": pval,
            "p_valeur_test": p_test,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Les descriptives n'entrent pas dans la famille corrigée : elles ne sont
    # pas candidates à une règle d'entrée.
    predictives = ~out["descriptive"]
    out["q_valeur"] = np.nan
    out.loc[predictives, "q_valeur"] = benjamini_hochberg(out.loc[predictives, "p_valeur"])
    out["stable"] = predictives & (out["q_valeur"] < 0.05) & (out["p_valeur_test"] < 0.05)
    out["force"] = (out["auc_apprentissage"] - 0.5).abs()
    return out.sort_values(["descriptive", "stable", "force"], ascending=[True, False, False]).reset_index(drop=True)


def quintiles(df: pd.DataFrame, variable: str, apprentissage: pd.Series, test: pd.Series) -> pd.DataFrame:
    """Taux de gagnantes et alpha moyen par quintile de la variable. Les
    bornes sont celles de l'APPRENTISSAGE, réappliquées au test : c'est la
    seule façon de lire une règle qu'on aurait pu fixer à l'avance."""
    x = pd.to_numeric(df[variable], errors="coerce")
    bornes = np.unique(np.nanpercentile(x[apprentissage], [0, 20, 40, 60, 80, 100]))
    if len(bornes) < 3:
        return pd.DataFrame()
    bornes[0], bornes[-1] = -np.inf, np.inf
    classe = pd.cut(x, bornes, labels=False, include_lowest=True)
    rows = []
    for k in range(len(bornes) - 1):
        ligne = {"variable": variable, "quintile": k + 1,
                 "borne_basse": bornes[k], "borne_haute": bornes[k + 1]}
        for nom, masque in (("apprentissage", apprentissage), ("test", test)):
            sel = masque & (classe == k)
            ligne[f"n_{nom}"] = int(sel.sum())
            ligne[f"taux_gagnantes_{nom}_pct"] = float(df.loc[sel, "gagnante"].mean() * 100) if sel.any() else np.nan
            ligne[f"alpha_moyen_{nom}_pct"] = float(df.loc[sel, "alpha_pct"].mean()) if sel.any() else np.nan
        rows.append(ligne)
    return pd.DataFrame(rows)


def taux_par_modalite(df: pd.DataFrame, colonne: str, apprentissage: pd.Series, test: pd.Series) -> pd.DataFrame:
    rows = []
    valeurs = df[colonne].astype("object").where(df[colonne].notna(), "(manquant)")
    for modalite in valeurs[apprentissage | test].unique():
        ligne = {"variable": colonne, "modalite": modalite}
        for nom, masque in (("apprentissage", apprentissage), ("test", test)):
            sel = masque & (valeurs == modalite)
            ligne[f"n_{nom}"] = int(sel.sum())
            ligne[f"taux_gagnantes_{nom}_pct"] = float(df.loc[sel, "gagnante"].mean() * 100) if sel.any() else np.nan
            ligne[f"alpha_moyen_{nom}_pct"] = float(df.loc[sel, "alpha_pct"].mean()) if sel.any() else np.nan
        rows.append(ligne)
    return pd.DataFrame(rows).sort_values("n_apprentissage", ascending=False)


def _boosting(graine: int):
    """Gradient boosting volontairement BRIDÉ : arbres de profondeur 3,
    feuilles d'au moins 40 thèses, apprentissage lent et arrêt précoce sur
    une validation interne. Avec deux mille thèses et cinquante variables, un
    modèle laissé libre apprend l'échantillon, pas la règle."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.03, max_iter=300, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.2,
        n_iter_no_change=20, random_state=graine)


@dataclass
class ResultatModele:
    auc_test: float
    ic_test: tuple[float, float]
    auc_reference_ecart: float
    ic_reference: tuple[float, float]
    auc_logistique: float
    ic_logistique: tuple[float, float]
    importance: pd.DataFrame
    proba_test: pd.Series
    n_apprentissage: int
    n_test: int


def modele_principal(df: pd.DataFrame, numeriques: list[str], categorielles: list[str],
                     apprentissage: pd.Series, test: pd.Series, graine: int = 0,
                     n_bootstrap: int = 1000) -> ResultatModele:
    """Apprend sur l'apprentissage, mesure sur le test, et se compare à deux
    références : l'ÉCART DE VALORISATION SEUL -- la variable que la stratégie
    utilise déjà pour choisir -- et une régression logistique. Un modèle qui
    ne bat pas l'écart seul n'a rien appris que la stratégie ne sache déjà."""
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_app, modalites = matrice(df[apprentissage], numeriques, categorielles)
    X_test, _ = matrice(df[test], numeriques, categorielles, modalites)
    y_app, y_test = df.loc[apprentissage, "gagnante"].to_numpy(), df.loc[test, "gagnante"].to_numpy()
    grappes = df.loc[test, "entry_date"].dt.to_period("M").astype(str).to_numpy()

    modele = _boosting(graine).fit(X_app, y_app)
    proba = modele.predict_proba(X_test)[:, 1]
    auc_test = auc(y_test, proba)

    # Référence : l'écart seul, orienté comme sur l'apprentissage.
    reference = np.full(len(y_test), np.nan)
    if "ecart_pct" in X_test.columns:
        sens = 1.0 if auc(y_app, X_app["ecart_pct"]) >= 0.5 else -1.0
        reference = sens * X_test["ecart_pct"].to_numpy()

    logistique = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(C=0.1, max_iter=2000))
    colonnes_pleines = [c for c in X_app.columns if X_app[c].notna().any()]
    logistique.fit(X_app[colonnes_pleines], y_app)
    proba_log = logistique.predict_proba(X_test[colonnes_pleines])[:, 1]
    auc_log = auc(y_test, proba_log)

    perm = permutation_importance(modele, X_test, y_test, scoring="roc_auc",
                                  n_repeats=20, random_state=graine)
    importance = pd.DataFrame({
        "variable": X_test.columns,
        "baisse_auc_moyenne": perm.importances_mean,
        "baisse_auc_ecart_type": perm.importances_std,
    }).sort_values("baisse_auc_moyenne", ascending=False).reset_index(drop=True)

    return ResultatModele(
        auc_test=auc_test,
        ic_test=auc_bootstrap(y_test, proba, grappes, n_bootstrap, graine),
        auc_reference_ecart=auc(y_test, reference),
        ic_reference=auc_bootstrap(y_test, reference, grappes, n_bootstrap, graine),
        auc_logistique=auc_log,
        ic_logistique=auc_bootstrap(y_test, proba_log, grappes, n_bootstrap, graine),
        importance=importance,
        proba_test=pd.Series(proba, index=df.index[test]),
        n_apprentissage=int(apprentissage.sum()),
        n_test=int(test.sum()),
    )


def walk_forward(df: pd.DataFrame, numeriques: list[str], categorielles: list[str],
                 premiere_annee: int, graine: int = 0, min_apprentissage: int = 300) -> tuple[pd.DataFrame, pd.Series]:
    """Réapprend chaque année sur les seules thèses CLOSES avant le 1er
    janvier, prédit celles ouvertes dans l'année. Donne une AUC par année --
    un effet qui ne tient qu'une année sur deux se voit ici, pas dans une AUC
    globale -- et une probabilité hors échantillon pour CHAQUE thèse testée,
    qui sert à la lecture économique (alpha par quintile de probabilité)."""
    etiquetee = df["gagnante"].notna()
    annees = sorted(df.loc[etiquetee, "entry_date"].dt.year.unique())
    rows, probas = [], []
    for annee in annees:
        if annee < premiere_annee:
            continue
        debut = pd.Timestamp(year=annee, month=1, day=1)
        app = etiquetee & (df["exit_date"] < debut)
        tst = etiquetee & (df["entry_date"].dt.year == annee)
        if app.sum() < min_apprentissage or tst.sum() < 20:
            continue
        X_app, modalites = matrice(df[app], numeriques, categorielles)
        X_tst, _ = matrice(df[tst], numeriques, categorielles, modalites)
        modele = _boosting(graine).fit(X_app, df.loc[app, "gagnante"].to_numpy())
        p = modele.predict_proba(X_tst)[:, 1]
        probas.append(pd.Series(p, index=df.index[tst]))
        rows.append({
            "annee": int(annee), "n_apprentissage": int(app.sum()), "n_test": int(tst.sum()),
            "taux_gagnantes_pct": float(df.loc[tst, "gagnante"].mean() * 100),
            "auc": auc(df.loc[tst, "gagnante"], p),
        })
    proba = pd.concat(probas) if probas else pd.Series(dtype=float)
    return pd.DataFrame(rows), proba


def lecture_economique(df: pd.DataFrame, proba: pd.Series, n_classes: int = 5) -> pd.DataFrame:
    """Alpha moyen par quintile de probabilité hors échantillon. Une AUC de
    0,55 peut valoir beaucoup ou rien : la question utile est de savoir si
    les thèses que le modèle juge perdantes RAPPORTENT effectivement moins,
    et de combien."""
    if proba.empty:
        return pd.DataFrame()
    d = df.loc[proba.index, ["alpha_pct", "rendement_pct", "pnl", "gagnante"]].assign(proba=proba)
    d["classe"] = pd.qcut(d["proba"].rank(method="first"), n_classes, labels=False) + 1
    return d.groupby("classe").agg(
        n=("proba", "size"), proba_min=("proba", "min"), proba_max=("proba", "max"),
        taux_gagnantes_pct=("gagnante", lambda s: s.mean() * 100),
        alpha_moyen_pct=("alpha_pct", "mean"), alpha_median_pct=("alpha_pct", "median"),
        rendement_moyen_pct=("rendement_pct", "mean"), pnl_total=("pnl", "sum"),
    ).reset_index()


def regles(df: pd.DataFrame, numeriques: list[str], categorielles: list[str],
           apprentissage: pd.Series, test: pd.Series, profondeur: int = 3,
           graine: int = 0) -> pd.DataFrame:
    """Profils lisibles : un arbre de décision PEU PROFOND, dont chaque
    feuille est une règle (« écart > 150 % et momentum < -20 % »). Chaque
    feuille rapporte son taux de gagnantes et son alpha sur l'apprentissage,
    puis sur le TEST : une règle dont le taux s'effondre au test est un
    souvenir de l'échantillon, pas un point commun."""
    from sklearn.tree import DecisionTreeClassifier

    X_app, modalites = matrice(df[apprentissage], numeriques, categorielles)
    X_test, _ = matrice(df[test], numeriques, categorielles, modalites)
    y_app = df.loc[apprentissage, "gagnante"].to_numpy()
    feuille_min = max(30, int(0.05 * len(y_app)))
    arbre = DecisionTreeClassifier(max_depth=profondeur, min_samples_leaf=feuille_min,
                                   random_state=graine).fit(X_app, y_app)
    t = arbre.tree_
    noms = list(X_app.columns)
    manquant_gauche = getattr(t, "missing_go_to_left", None)

    chemins: dict[int, list[str]] = {}

    def parcourir(noeud: int, conditions: list[str]) -> None:
        if t.children_left[noeud] == -1:
            chemins[noeud] = conditions
            return
        nom, seuil = noms[t.feature[noeud]], t.threshold[noeud]
        gauche_nan = bool(manquant_gauche[noeud]) if manquant_gauche is not None else False
        if "=" in nom:  # indicatrice : <= 0,5 veut dire « n'est pas »
            cond_g, cond_d = nom.replace("=", " ≠ "), nom.replace("=", " = ")
        else:
            cond_g, cond_d = f"{nom} <= {seuil:.4g}", f"{nom} > {seuil:.4g}"
        if gauche_nan:
            cond_g += " (ou manquant)"
        else:
            cond_d += " (ou manquant)"
        parcourir(t.children_left[noeud], conditions + [cond_g])
        parcourir(t.children_right[noeud], conditions + [cond_d])

    parcourir(0, [])
    feuilles_app = pd.Series(arbre.apply(X_app), index=df.index[apprentissage])
    feuilles_test = pd.Series(arbre.apply(X_test), index=df.index[test])
    rows = []
    for feuille, conditions in chemins.items():
        ligne = {"feuille": int(feuille), "regle": " ET ".join(conditions) or "(toutes)"}
        for nom, f in (("apprentissage", feuilles_app), ("test", feuilles_test)):
            idx = f.index[f == feuille]
            ligne[f"n_{nom}"] = len(idx)
            ligne[f"taux_gagnantes_{nom}_pct"] = float(df.loc[idx, "gagnante"].mean() * 100) if len(idx) else np.nan
            ligne[f"alpha_moyen_{nom}_pct"] = float(df.loc[idx, "alpha_pct"].mean()) if len(idx) else np.nan
        rows.append(ligne)
    return pd.DataFrame(rows).sort_values("taux_gagnantes_apprentissage_pct").reset_index(drop=True)
