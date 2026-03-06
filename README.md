# banking_nlp

`banking_nlp` est un package Python prêt à installer via `pip` pour extraire:
- l'intention principale
- les entités associées

à partir d'un texte utilisateur en français (contexte bancaire).

## Auteur
- Auteur: Ricardo AMOUSSOU
- GitHub: https://github.com/hkdebendo
- Email: dgamoussouricardo@gmail.com

## Fonction principale
```python
import banking_nlp

result = banking_nlp.predict("Je veux envoyer 5000 à Jean")
print(result)
```

Exemple de sortie:
```json
{
  "intention": "VIREMENT_INIT",
  "entités": {
    "beneficiaire": "jean",
    "montant": "5000"
  },
  "score": 0.9987,
  "fallback": false
}
```

## Installation

### Depuis un dossier local (développement)
```bash
pip install .
```

### Depuis GitHub (quand le repo est publié)
```bash
pip install git+https://github.com/hkdebendo/banking_nlp.git
```

## Utilisation
```python
import banking_nlp

print(banking_nlp.predict("Quel est mon solde actuel ?"))
print(banking_nlp.predict("Montre mes opérations du lundi passé"))
print(banking_nlp.predict("Securise 2000 sur mes économies"))
```

## API publique
- `banking_nlp.predict(text: str, score_threshold: float = 0.60, margin_threshold: float = 0.15) -> dict`
- `banking_nlp.warmup() -> None` (précharge les modèles en mémoire)

## Intentions supportées
- `CONSULTER_SOLDE`
- `HISTORIQUE_SIMPLE`
- `HISTORIQUE_FILTRE`
- `VIREMENT_INIT`
- `ANNULATION`
- `aide`
- `salutation`
- `INCOMPRIS` (fallback)

## Entités extraites
- `VIREMENT_INIT`: `beneficiaire`, `montant`
- `HISTORIQUE_FILTRE`: `type`, `time`
- `CONSULTER_SOLDE`: `compte`

## Message de bienvenue
Le package affiche un message de bienvenue et de remerciement à la première importation:
- Bienvenue dans `banking_nlp`
- Auteur et contacts

Vous pouvez aussi l'afficher manuellement:
```bash
banking-nlp-welcome
```

## Structure
```text
banking_nlp/
  pyproject.toml
  README.md
  MANIFEST.in
  banking_nlp/
    __init__.py
    predictor.py
    cli.py
    models/
      intent_textcat_fr/
      slots/
```

## Build distribution
```bash
python -m pip install --upgrade build
python -m build
```

Les artefacts seront générés dans `dist/` (`.whl` et `.tar.gz`).

## Publication PyPI (optionnel)
```bash
python -m pip install --upgrade twine
twine upload dist/*
```

## Notes
- Python recommandé: 3.10+
- Le package embarque les modèles spaCy entraînés.
- Aucun entraînement n'est requis côté utilisateur final.
