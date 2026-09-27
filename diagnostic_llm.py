"""
Diagnostic du LLM en quelques secondes, sans lancer 04c : quelle clé est vue,
quel modèle est appelé, et ce que le fournisseur répond vraiment.

    python diagnostic_llm.py                  # 3 requêtes de test au modèle configuré
    python diagnostic_llm.py --essais 10      # mesurer la surcharge (503) d'un modèle
    python diagnostic_llm.py --modele NOM     # essayer un autre modèle, sans toucher à .env
    python diagnostic_llm.py --modeles        # les modèles Gemini ouverts à la clé

POURQUOI. 04c ne rencontre le modèle qu'après ses premiers téléchargements
SEC, et un refus s'y lit au milieu de centaines de lignes. Ici, la réponse
brute du fournisseur arrive en quelques secondes : un modèle retiré (404) se
corrige dans .env, un modèle surchargé (503) se constate avant de lancer un run
de plusieurs heures -- et se compare d'un modèle à l'autre.

La requête a la forme exacte de celles de 04c, 07b et 02
(sec_filings_text.analyser_document : un document, une consigne et un schéma de
réponse, avec le même modèle et le même niveau de réflexion), sans leurs
réessais ni leur disjoncteur. Une réponse qui ne respecte pas le schéma compte
comme un échec. Chaque essai consomme un appel du quota, pour quelques dizaines
de jetons. La clé n'est jamais affichée.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from typing import List, Optional

import requests

import config  # noqa: F401  -- charge .env, comme pour tous les scripts
import env_local
import sec_filings_text as sft

GEMINI_MODELES_URL = "https://generativelanguage.googleapis.com/v1beta/models"
# Une requête de test à trois zones, comme celles de 04c.
DOCUMENT_TEST = "Document de test envoyé par diagnostic_llm.py."
CONSIGNE_TEST = "Tu vérifies une connexion : indique si le document fourni t'est bien parvenu."
SCHEMA_TEST = {
    "type": "object",
    "properties": {"ok": {"type": "boolean", "description": "Vrai si le document est bien parvenu."}},
    "required": ["ok"],
}


def origine_de(nom: str) -> str:
    """D'où vient la variable -- jamais sa valeur, qui peut être une clé."""
    if not os.environ.get(nom, "").strip():
        return "absente"
    return "fichier .env" if nom in env_local.CHARGEES else "environnement du terminal"


def modeles_gemini(cle: str) -> List[dict]:
    """Les modèles ouverts à la clé qui acceptent generateContent (ListModels)."""
    modeles: List[dict] = []
    jeton: Optional[str] = None
    while True:
        params = {"pageSize": 1000, **({"pageToken": jeton} if jeton else {})}
        # Clé en en-tête, comme pour les appels : jamais dans l'URL.
        resp = requests.get(GEMINI_MODELES_URL, headers={"x-goog-api-key": cle}, params=params, timeout=30)
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code} : {sft._extrait_erreur(resp)}")
        corps = resp.json()
        modeles += [m for m in corps.get("models") or []
                    if "generateContent" in (m.get("supportedGenerationMethods") or [])]
        jeton = corps.get("nextPageToken")
        if not jeton:
            return modeles


def afficher_modeles(cle: str, courant: str) -> int:
    try:
        modeles = modeles_gemini(cle)
    except (requests.exceptions.RequestException, RuntimeError, ValueError) as e:
        print(f"Liste des modèles impossible : {e}")
        return 1
    print(f"{len(modeles)} modèle(s) Gemini ouverts à cette clé :")
    for modele in sorted(modeles, key=lambda m: m.get("name", "")):
        nom = modele.get("name", "")
        nom = nom[len("models/"):] if nom.startswith("models/") else nom
        repere = "   <-- utilisé" if nom == courant else ""
        print(f"  {nom:<42} {modele.get('displayName', '')}{repere}")
    print(f"\nEssai d'un modèle : python diagnostic_llm.py --modele NOM ; "
          f"pour l'adopter : {sft.GEMINI_MODEL_ENV}=NOM dans .env.")
    return 0


def essayer(essais: int, pause: float) -> List[sft.EssaiLLM]:
    resultats = []
    for i in range(1, essais + 1):
        essai = sft.essai_unique_llm(DOCUMENT_TEST, consigne=CONSIGNE_TEST, schema=SCHEMA_TEST)
        etat = "OK" if essai.ok else (f"HTTP {essai.statut}" if essai.statut else "pas de réponse")
        print(f"  essai {i}/{essais} : {etat} en {essai.duree_s:.1f} s -- {' '.join(essai.texte.split())[:200]}")
        resultats.append(essai)
        if i < essais:
            time.sleep(pause)
    return resultats


def conclusion(resultats: List[sft.EssaiLLM]) -> str:
    reussis = sum(essai.ok for essai in resultats)
    if reussis == len(resultats):
        return "Le modèle répond, dans le format demandé : 04c, 07b et 02 peuvent s'en servir."
    refus = next((e for e in resultats if e.statut and e.statut >= 400
                  and sft._refus_de_configuration(e.statut, e.texte)), None)
    if refus:
        return (f"Refus de configuration (HTTP {refus.statut}) : même réponse pour chaque document. "
                + sft._aide_configuration(refus.statut, refus.texte))
    statuts = Counter(e.statut for e in resultats if not e.ok)
    if statuts.get(429):
        return (f"{reussis}/{len(resultats)} réponse(s). Quota atteint (429) : attends la fin de sa "
                "fenêtre (la minute, ou le lendemain pour un quota quotidien), ou passe à une offre payante.")
    if statuts.get(None) or any(s and s >= 500 for s in statuts):
        return (f"{reussis}/{len(resultats)} réponse(s). Le fournisseur est surchargé ou injoignable : la "
                "panne est de son côté. 04c s'en accommode (réessais, puis pause du modèle) ; si ça dure, "
                "compare un autre modèle avec --modele NOM (--modeles les liste).")
    return f"{reussis}/{len(resultats)} réponse(s) exploitable(s) : voir le détail ci-dessus."


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Teste la clé et le modèle LLM en quelques secondes.")
    parser.add_argument("--essais", type=int, default=3, help="Requêtes de test (défaut: %(default)s).")
    parser.add_argument("--pause", type=float, default=2.0,
                        help="Secondes entre deux requêtes (défaut: %(default)s).")
    parser.add_argument("--modele", help="Modèle Gemini à essayer à la place de celui configuré "
                                         "(pour ce test seulement, .env n'est pas modifié).")
    parser.add_argument("--modeles", action="store_true",
                        help="Liste les modèles Gemini ouverts à la clé, sans rien essayer.")
    args = parser.parse_args(argv)

    origines = {nom: origine_de(nom) for nom in (
        sft.GEMINI_API_KEY_ENV, sft.GEMINI_MODEL_ENV, sft.GEMINI_THINKING_LEVEL_ENV)}
    if args.modele:
        os.environ[sft.GEMINI_MODEL_ENV] = args.modele
        origines[sft.GEMINI_MODEL_ENV] = "option --modele"
    elif origines[sft.GEMINI_MODEL_ENV] == "absente":
        origines[sft.GEMINI_MODEL_ENV] = f"absente (défaut du code : {sft.GEMINI_DEFAULT_MODEL})"
    if origines[sft.GEMINI_THINKING_LEVEL_ENV] == "absente":
        origines[sft.GEMINI_THINKING_LEVEL_ENV] = (
            f"absente (défaut du code : {sft.GEMINI_DEFAULT_THINKING_LEVEL})")

    print(f"Modèle : {sft.description_llm()}")
    for nom, origine in origines.items():
        print(f"  {nom:<22} {origine}")
    if not sft.llm_disponible():
        print(f"\nAucune clé vue par ce processus. {sft.aide_cle_absente()}")
        return 1

    if args.modeles:
        print()
        return afficher_modeles(os.environ[sft.GEMINI_API_KEY_ENV].strip(), sft._gemini_model())

    essais = max(args.essais, 1)
    print(f"\n{essais} requête(s) de test (document, consigne et schéma de réponse, "
          "comme 04c, 07b et 02) :")
    resultats = essayer(essais, args.pause)
    print("\n" + conclusion(resultats))
    return 0 if any(e.ok for e in resultats) else 1


if __name__ == "__main__":
    sys.exit(main())
