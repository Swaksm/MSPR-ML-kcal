# Rapport d'evaluation NER — FOOD / QUANTITY

- Seed: 42 | Epochs: 40
- Corpus annote: 995 phrases (796 train / 199 test, split 80/20)
- Exemples auto-generes (templates): 4589
- Total exemples d'entrainement: 6977

## Modele entraine (spaCy NER) — evalue sur le test set jamais vu

| | Precision | Recall | F1 |
|---|---|---|---|
| Global | 0.859 | 0.865 | 0.862 |
| QUANTITY | 0.974 | 0.995 | 0.984 |
| FOOD | 0.796 | 0.796 | 0.796 |

## Baseline regles (liste FOOD_DB + regex QUANTITY, sans apprentissage)

| | Precision | Recall | F1 |
|---|---|---|---|
| Global | 0.749 | 0.593 | 0.662 |
| QUANTITY | 0.811 | 0.858 | 0.834 |
| FOOD | 0.695 | 0.453 | 0.548 |
