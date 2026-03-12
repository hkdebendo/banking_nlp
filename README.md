# banking_nlp 2.0.0

![Version](https://img.shields.io/badge/version-2.0.0-2563eb?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-3.7%2B-09A3D5?style=for-the-badge)
![French NLP](https://img.shields.io/badge/French-NLU-059669?style=for-the-badge)
![Banking](https://img.shields.io/badge/domain-banking-7c3aed?style=for-the-badge)
![Packaged Models](https://img.shields.io/badge/models-embedded-f59e0b?style=for-the-badge)

`banking_nlp` est un package Python prêt à l’emploi pour analyser des requêtes bancaires en français et renvoyer un JSON structuré avec :

- l’intention prédite
- les entités extraites
- le score du modèle
- l’indicateur de fallback

La version `2.0.0` embarque le **modèle actuellement entraîné** et le **moteur d’inférence aligné sur `infer_intent_entities.py`** du projet principal.

## Nouveautés de la v2.0.0

![Inference](https://img.shields.io/badge/inference-synced_with_root-success?style=flat-square)
![Entities](https://img.shields.io/badge/entities-more_robust-success?style=flat-square)
![Packaging](https://img.shields.io/badge/package-production_ready-success?style=flat-square)
![Models](https://img.shields.io/badge/models-latest_trained-blue?style=flat-square)

- Synchronisation du package avec la logique actuelle de `infer_intent_entities.py`
- Embarquement des modèles spaCy les plus récents du projet
- Meilleure normalisation des textes bruités et des accents
- Extraction plus robuste des montants : `2000`, `25 mille`, `un million`
- Canonicalisation plus fiable des bénéficiaires : `mon frère` → `frere`, `ma tante` → `tante`
- Meilleure gestion des formulations naturelles : `place 2000 sur le compte de Chantal`
- Overrides d’intentions plus solides pour `solde`, `historique`, `aide`, `salutation`, `annulation` et `virement`

## Installation

### Depuis le dossier local

```bash
pip install .
```

### Depuis GitHub

```bash
pip install git+https://github.com/hkdebendo/banking_nlp.git
```

## API publique

- `banking_nlp.predict(text: str, score_threshold: float = 0.60, margin_threshold: float = 0.15) -> dict`
- `banking_nlp.warmup() -> None`

## Exemple rapide

```python
import banking_nlp

result = banking_nlp.predict("Je veux envoyer 5000 francs à Chantal")
print(result)
```

Sortie possible :

```json
{
  "intention": "VIREMENT_INIT",
  "entités": {
    "beneficiaire": "chantal",
    "montant": "5000"
  },
  "score": 0.9987,
  "fallback": false
}
```

## Exemples d’utilisation

```python
import banking_nlp

print(banking_nlp.predict("Quel est mon solde actuel ?"))
print(banking_nlp.predict("Montre mes opérations du mois dernier"))
print(banking_nlp.predict("Place 2000 sur le compte de Fiacre"))
print(banking_nlp.predict("Envoie un million à ma tante"))
print(banking_nlp.predict("Comment puis-je utiliser le service ?"))
```

## Intentions supportées

- `CONSULTER_SOLDE`
- `HISTORIQUE_SIMPLE`
- `HISTORIQUE_FILTRE`
- `VIREMENT_INIT`
- `ANNULATION`
- `aide`
- `salutation`
- `INCOMPRIS`

## Entités extraites

- `VIREMENT_INIT` → `beneficiaire`, `montant`
- `HISTORIQUE_FILTRE` → `type`, `time`
- `CONSULTER_SOLDE` → `compte`

## Ce que la v2.0.0 gère mieux

### Virements

- `Envoie 10000 francs à mon frère`
- `Place 2000 sur le compte de Chantal`
- `Transfère 4000 à ma dada`
- `Fais un transfert d’un million à Fiacre`

### Historique

- `Montre les transactions du mois dernier`
- `Affiche les débits depuis lundi`
- `Je veux revoir ce qui est passé récemment`

### Solde

- `Quel est mon solde actuel ?`
- `Combien reste-t-il sur mon épargne ?`

### Aide et salutation

- `Bonjour`
- `Wesh`
- `Quelles commandes sont disponibles ?`

## Structure du package

```text
banking_nlp/
  pyproject.toml
  README.md
  MANIFEST.in
  banking_nlp/
    __init__.py
    cli.py
    predictor.py
    py.typed
    resources/
      known_beneficiaires.json
    models/
      intent_textcat_fr/
      slots/
```

## Build de la distribution

```bash
python -m pip install --upgrade build
python -m build
```

Les nouveaux artefacts de `2.0.0` sont générés dans `dist/` sans supprimer les anciens fichiers déjà présents.

## Commande CLI

```bash
banking-nlp-welcome
```

## Auteur

- Ricardo AMOUSSOU
- GitHub : https://github.com/hkdebendo
- Email : dgamoussouricardo@gmail.com

## Notes

- Python recommandé : `3.10+`
- Les modèles spaCy sont embarqués dans le package
- Aucun entraînement n’est requis côté utilisateur final
- La sortie est directement exploitable dans une API, un chatbot ou une application bancaire
