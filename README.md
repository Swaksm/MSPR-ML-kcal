# HealthAI — Module IA / Analyse Nutritionnelle

Module d'intelligence artificielle du projet **Jarmy**, développé dans le cadre de la
formation Concepteur Développeur d'Applications (RNCP36581 — Bloc 2 / E6.1).

Ce module est **indépendant** et peut s'intégrer dans un backend FastAPI, Flask, ou
un microservice Python.

---

## Objectif

Entrer un repas en texte libre (anglais), extraire les aliments et quantités,
calculer les calories totales.

Exemples :

- `"2 eggs and a banana"` → 293 kcal
- `"200g of grilled chicken with brown rice and broccoli"` → 532 kcal
- `"pizza and french fries with a soda"` → 828 kcal
- `"had chiken and rise"` → 482 kcal (fautes corrigées via synonymes/fuzzy matching)

---

## Fonctionnement (vue d'ensemble)

```
texte libre
    │
    ▼
nlp/parser.parse()            ← modèle spaCy NER custom (nlp/food_ner_model)
    │  entités FOOD / QUANTITY
    ▼
_match_food()                 ← synonymes (typos connus) + fuzzy matching (difflib)
_qty_from_text() / _grams()   ← conversion unités (g, cups, slices...) et portions par défaut
    │  liste {food, grams}
    ▼
analyze.analyze()             ← lookup kcal/100g dans FOOD_DB, calcul du total
    │
    ▼
MealResult(items, total_kcal, message)
```

---

## Arborescence du projet

```
ia-kcal/
├── app.py                          Interface CLI (interactive + mode --test)
├── analyze.py                      Orchestration parsing → résolution → calcul kcal
├── README                          Ce fichier
│
├── data/
│   ├── nutrition_data.py           Chargement + normalisation du CSV → FOOD_DB (kcal/100g)
│   ├── kaggle_nutrition.csv        Dataset Kaggle "Daily Food & Nutrition Dataset" (965 lignes)
│   └── training_sentences.json     996 phrases annotées à la main (entités FOOD/QUANTITY)
│
└── nlp/
    ├── parser.py                   Résolution post-NER : synonymes, fuzzy matching, quantités
    ├── train_ner.py                Entraînement du modèle NER de production
    ├── evaluate_ner.py             Évaluation rigoureuse (split train/test, métriques, baseline)
    ├── food_ner_model/             Modèle spaCy entraîné, sauvegardé (généré par train_ner.py)
    └── eval_output/                Résultats d'évaluation (généré par evaluate_ner.py)
        ├── metrics_report.md       Rapport lisible (precision/recall/F1)
        ├── metrics_report.json     Mêmes métriques en JSON
        └── loss_curve.png          Courbe de convergence de la loss NER
```

---

## Détail de chaque fichier

### `app.py`
Interface en ligne de commande. Deux modes :
- `python app.py` → mode interactif, on tape un repas au clavier
- `python app.py --test` → lance une liste de phrases de test prédéfinies (`TESTS`)
  et affiche le détail par aliment + le total kcal.

### `analyze.py`
Point d'entrée principal du module. Fonction `analyze(text) -> MealResult` :
appelle `nlp.parser.parse()` pour extraire les aliments/quantités, puis calcule
le kcal de chaque item (`kcal_100g * grammes / 100`) et le total. `MealResult`
est un `dataclass` avec `items`, `total_kcal`, `message`.

### `data/nutrition_data.py`
Charge `kaggle_nutrition.csv` et construit `FOOD_DB`, un dictionnaire
`{nom_aliment: kcal_pour_100g}`. Le CSV donne des calories par portion
(ex: "1 apple = 95 kcal") ; ce fichier les convertit en kcal/100g grâce à
`PORTION_WEIGHT_G` (poids de référence par aliment), et nettoie les noms bruts
du CSV (`_normalize` : enlève parenthèses, quantités, adjectifs de cuisson).
817 aliments uniques chargés au final.

### `data/kaggle_nutrition.csv`
Dataset brut source, tel que téléchargé sur Kaggle. 965 lignes de données
(+ en-tête). Colonnes utilisées : `Food_Item`, `Calories (kcal)`.

### `data/training_sentences.json`
996 phrases en anglais annotées manuellement, chacune avec ses entités
`FOOD`/`QUANTITY` repérées par leur position dans le texte. Générées avec
l'aide d'une IA conversationnelle, fautes de frappe volontaires incluses pour
entraîner la robustesse du modèle. Sert de dataset d'entraînement/évaluation
pour le NER (à distinguer des données auto-générées par templates dans
`train_ner.py`, qui sont un jeu de données synthétique séparé).

### `nlp/parser.py`
Étape de résolution après la détection d'entités par le modèle NER. Charge le
modèle spaCy (`_load_model`), puis pour chaque phrase :
- `_match_food(raw)` : normalise le texte détecté comme FOOD, essaie une
  correspondance exacte dans `FOOD_DB`, sinon un dictionnaire de synonymes
  connus (`SYNONYMS`, ex: `"chiken"` → `"roast chicken"`), sinon une
  correspondance approximative (`difflib.get_close_matches`, seuil 0.72).
- `_qty_from_text(raw)` : parse une quantité détectée (`"200g"`, `"two"`,
  `"half"`...) en `(valeur, unité)`.
- `_grams(food, qty, unit)` : convertit en grammes via `UNITS` (g, cup, slice,
  tbsp...) ou, à défaut, via une portion par défaut (`PORTIONS`) propre à
  chaque aliment.
- `parse(text)` : fonction principale, pivote les entités NER en liste
  `[{"food": ..., "grams": ...}]`, avec dédoublonnage (`_dedup`).

### `nlp/train_ner.py`
Entraîne le modèle NER de production utilisé par l'application. Combine :
- des exemples **auto-générés** par templates (`TEMPLATES_SINGLE`,
  `TEMPLATES_MULTI`) appliqués aux 80 aliments les plus fréquents, + variantes
  de fautes de frappe (`TYPO_VARIANTS`) et de synonymes — 4589 exemples.
- les 996 phrases **annotées à la main** (`training_sentences.json`), répétées
  ×3 pour leur donner plus de poids face aux données synthétiques.

Entraînement : `spacy.blank("en")`, pipeline NER seul, labels `FOOD` et
`QUANTITY`, 40 epochs, dropout 0.25, batchs de taille croissante
(`compounding(8, 64, 1.001)`), optimiseur Adam (learning rate 0.001, valeur
par défaut spaCy). Sauvegarde le modèle dans `nlp/food_ner_model/`.

⚠️ Ce script n'effectue pas de split train/test : c'est un entraînement de
production sur 100% des données disponibles. L'évaluation rigoureuse avec
données jamais vues se fait séparément, dans `evaluate_ner.py`.

### `nlp/evaluate_ner.py`
Script d'évaluation, séparé de l'entraînement de production, pour mesurer
objectivement la performance du modèle :
1. Split 80/20 (seed=42, reproductible) sur les 996 phrases annotées à la
   main uniquement → 796 phrases train / 199 phrases test.
2. Réentraîne un modèle (mêmes hyperparamètres que `train_ner.py`) sur les
   données auto-générées + le split train uniquement — le split test n'est
   **jamais** vu pendant l'entraînement.
3. Évalue avec `spacy.scorer.Scorer` sur le split test : precision, recall,
   F1 globaux et détaillés par entité (FOOD vs QUANTITY).
4. Compare à une baseline sans apprentissage (liste de correspondance
   `FOOD_DB` + regex pour les quantités) sur le même split test, pour
   objectiver l'apport réel du modèle entraîné.
5. Sauvegarde le rapport (`eval_output/metrics_report.{json,md}`) et une
   courbe de convergence de la loss (`eval_output/loss_curve.png`).

Run : `python nlp/evaluate_ner.py` (~13 minutes).

### `nlp/food_ner_model/`
Modèle spaCy entraîné, sauvegardé sur disque (config, vocabulaire, poids du
réseau). Généré par `train_ner.py` — ne pas éditer à la main. Si absent,
`nlp/parser.py` lève une `FileNotFoundError` explicite au chargement.

### `nlp/eval_output/`
Sortie générée par `evaluate_ner.py` — rapport de métriques et courbe de loss,
utilisés comme preuves chiffrées de performance (voir section Résultats
ci-dessous).

---

## Résultats d'évaluation

Évalué sur 199 phrases annotées manuellement, jamais vues pendant
l'entraînement (split 80/20, seed=42).

| | Precision | Recall | F1 |
|---|---|---|---|
| **Modèle entraîné — Global** | 0.859 | 0.865 | **0.862** |
| Modèle entraîné — FOOD | 0.796 | 0.796 | 0.796 |
| Modèle entraîné — QUANTITY | 0.974 | 0.995 | 0.984 |
| **Baseline règles — Global** | 0.749 | 0.593 | 0.662 |
| Baseline règles — FOOD | 0.695 | 0.453 | 0.548 |
| Baseline règles — QUANTITY | 0.811 | 0.858 | 0.834 |

Le modèle entraîné dépasse nettement une approche par règles simples
(liste de mots-clés + regex), surtout sur la détection des aliments
(F1 0.796 contre 0.548), ce qui objective l'apport de l'entraînement NER
par rapport à une approche purement déterministe.

Détail complet, courbe de loss et méthodologie : voir
[`nlp/eval_output/metrics_report.md`](nlp/eval_output/metrics_report.md).

---

## Stack technique

- Python 3.11+
- spaCy (NER custom, pipeline `spacy.blank("en")`) — `nlp/food_ner_model`
- Pas de dépendance à `pandas` (CSV traité en Python pur)
- `matplotlib` (uniquement pour la courbe de loss, optionnel)

---

## Installation

```powershell
python -m pip install -U pip
pip install spacy matplotlib
```

---

## Entraînement du modèle (obligatoire avant utilisation)

```powershell
python nlp/train_ner.py
```

- génère `nlp/food_ner_model`
- si ce dossier est absent : `parser.parse` lève une erreur claire

Pour évaluer le modèle avec des métriques chiffrées (precision/recall/F1,
comparaison baseline) :

```powershell
python nlp/evaluate_ner.py
```

---

## Exécution

```powershell
python app.py
```

- mode interactif : tape un repas en anglais
- `test` : lance des exemples automatiques
- `quit` : sortie

---

## Limites connues

- La résolution post-NER (`_match_food`) est le maillon le plus fragile : le
  modèle NER détecte parfois correctement un aliment composé
  (ex: `"thanksgiving turkey"`) mais le fuzzy matching échoue à le relier à
  une entrée de `FOOD_DB` et l'entité est silencieusement ignorée.
- `FOOD_DB` ne couvre que les aliments présents dans le CSV Kaggle — les
  aliments hors dataset (marques, plats très spécifiques) ne sont pas
  reconnus.
- Une partie (39 sur 2729) des entités annotées dans
  `training_sentences.json` utilisent un libellé canonique plutôt que le mot
  exact du texte (ex: `"bowl of muesli"` annoté `"oats"`), ce qui les fait
  ignorer silencieusement par `load_manual_data()` lors du chargement des
  données d'entraînement.

---

## Intégration API

```python
from analyze import analyze
r = analyze("200g chicken and rice")
print(r.total_kcal, r.items, r.message)
```

---

## Sources de données

- `kaggle_nutrition.csv` : dataset Kaggle "Daily Food & Nutrition Dataset"
- enrichissements manuels (fast-food, plats du quotidien)
