"""
Evaluation rigoureuse du modele NER (FOOD / QUANTITY).

Methodologie :
  1. Split train/test (80/20, seed=42) sur data/training_sentences.json UNIQUEMENT.
     Les 20% de test ne sont jamais vus pendant l'entrainement.
  2. Reentrainement d'un modele (memes hyperparametres que nlp/train_ner.py :
     40 epochs, dropout 0.25, batches compounding(8,64,1.001)) sur :
        - les donnees auto-generees (templates, jamais issues des phrases annotees)
        - le split TRAIN des phrases annotees (repete x3, comme en prod)
  3. Evaluation avec spacy.scorer.Scorer sur le split TEST (jamais vu) :
     precision / recall / F1 globaux + detail par entite (FOOD, QUANTITY).
  4. Baseline de comparaison : liste de correspondance (mots-cles FOOD_DB +
     synonymes) + regex pour QUANTITY, sans apprentissage. Evaluee sur le
     meme split TEST pour objectiver l'apport du modele entraine.

Sortie : nlp/eval_output/metrics_report.json + metrics_report.md
         + loss_curve.png (bonus, convergence de la loss NER par epoch)

Run: python nlp/evaluate_ner.py
"""

import sys, re, json, random, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import spacy
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.util import minibatch, compounding

from nlp.train_ner import generate_auto_data, load_manual_data, _clean_spans
from nlp.parser import FOOD_DB, SYNONYMS, WORDS

SEED = 42
EPOCHS = 40
OUT_DIR = Path(__file__).parent / "eval_output"


# ── 1. Split train/test sur les phrases annotees ────────────────────────────

def split_manual_data(test_ratio=0.2, seed=SEED):
    random.seed(seed)
    manual = load_manual_data()
    manual = manual[:]
    random.shuffle(manual)
    n_test = int(len(manual) * test_ratio)
    test_set = manual[:n_test]
    train_set = manual[n_test:]
    return train_set, test_set


# ── 2. Entrainement (memes hyperparametres que train_ner.py) ────────────────

def train_eval_model(train_manual, auto_data, epochs=EPOCHS, seed=SEED):
    random.seed(seed)
    all_data = auto_data + train_manual * 3
    random.shuffle(all_data)

    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    ner.add_label("FOOD")
    ner.add_label("QUANTITY")

    optimizer = nlp.begin_training()

    loss_history = []
    print(f"[eval] Training on {len(all_data)} examples for {epochs} epochs...")
    t0 = time.time()
    for epoch in range(epochs):
        random.shuffle(all_data)
        losses = {}
        for batch in minibatch(all_data, size=compounding(8.0, 64.0, 1.001)):
            examples = []
            for text, annotations in batch:
                try:
                    examples.append(Example.from_dict(nlp.make_doc(text), annotations))
                except Exception:
                    continue
            nlp.update(examples, drop=0.25, losses=losses, sgd=optimizer)
        loss = float(losses.get("ner", 0.0))
        loss_history.append(loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} — NER loss: {loss:.3f}")
    print(f"[eval] Training done in {time.time()-t0:.1f}s")

    return nlp, loss_history


# ── 3. Scoring avec spacy.scorer.Scorer ──────────────────────────────────────

def score_trained_model(nlp, test_manual):
    examples = []
    for text, annotations in test_manual:
        pred_doc = nlp(text)
        examples.append(Example.from_dict(pred_doc, annotations))
    scores = Scorer().score(examples)
    return scores


# ── 4. Baseline rule-based (liste de correspondance + regex) ────────────────

FOOD_TERMS = sorted(set(list(FOOD_DB.keys()) + list(SYNONYMS.keys())), key=len, reverse=True)
QTY_WORDS = sorted(set(WORDS.keys()) | {"bowl", "glass", "cup", "cups", "slice", "slices", "handful", "some"},
                    key=len, reverse=True)
QTY_NUM_RE = re.compile(
    r'\b\d+(\.\d+)?\s*(g|gr|grams?|kg|ml|cl|oz|lb|cups?|slices?|tbsp|tsp)?\b'
)


def baseline_predict(text: str) -> list[tuple[int, int, str]]:
    t = text.lower()
    spans = []
    for term in FOOD_TERMS:
        for m in re.finditer(r'\b' + re.escape(term) + r'\b', t):
            spans.append((m.start(), m.end(), "FOOD"))
    for m in QTY_NUM_RE.finditer(t):
        if any(c.isdigit() for c in m.group()):
            spans.append((m.start(), m.end(), "QUANTITY"))
    for term in QTY_WORDS:
        for m in re.finditer(r'\b' + re.escape(term) + r'\b', t):
            spans.append((m.start(), m.end(), "QUANTITY"))
    return _clean_spans(spans, t)


def score_baseline(test_manual):
    blank = spacy.blank("en")
    examples = []
    for text, annotations in test_manual:
        pred_doc = blank.make_doc(text)
        spans = []
        for start, end, label in baseline_predict(text):
            span = pred_doc.char_span(start, end, label=label, alignment_mode="expand")
            if span is not None:
                spans.append(span)
        pred_doc.ents = _dedupe_doc_spans(spans)
        examples.append(Example.from_dict(pred_doc, annotations))
    scores = Scorer().score(examples)
    return scores


def _dedupe_doc_spans(spans):
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    result, last_end = [], -1
    for s in spans:
        if s.start >= last_end:
            result.append(s)
            last_end = s.end
    return result


# ── Formatting ────────────────────────────────────────────────────────────

def fmt_scores(scores) -> dict:
    per_type = scores.get("ents_per_type") or {}
    return {
        "precision": round(scores.get("ents_p", 0.0), 4),
        "recall": round(scores.get("ents_r", 0.0), 4),
        "f1": round(scores.get("ents_f", 0.0), 4),
        "per_entity": {
            label: {
                "precision": round(v["p"], 4),
                "recall": round(v["r"], 4),
                "f1": round(v["f"], 4),
            }
            for label, v in per_type.items()
        },
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_manual, test_manual = split_manual_data()
    print(f"[eval] Manual dataset: {len(train_manual) + len(test_manual)} total "
          f"-> {len(train_manual)} train / {len(test_manual)} test (80/20, seed={SEED})")

    random.seed(SEED)
    auto_data = generate_auto_data()
    print(f"[eval] Auto-generated (template) examples: {len(auto_data)}")

    nlp, loss_history = train_eval_model(train_manual, auto_data)

    print("\n[eval] Scoring trained model on held-out test set...")
    trained_scores = fmt_scores(score_trained_model(nlp, test_manual))

    print("[eval] Scoring rule-based baseline on the SAME held-out test set...")
    baseline_scores = fmt_scores(score_baseline(test_manual))

    report = {
        "seed": SEED,
        "epochs": EPOCHS,
        "corpus": {
            "manual_total": len(train_manual) + len(test_manual),
            "manual_train": len(train_manual),
            "manual_test": len(test_manual),
            "auto_generated": len(auto_data),
            "total_training_examples": len(auto_data) + len(train_manual) * 3,
        },
        "trained_model": trained_scores,
        "baseline_rule_based": baseline_scores,
        "loss_history": [round(l, 4) for l in loss_history],
    }

    with open(OUT_DIR / "metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Markdown summary (pastable into the dossier)
    md = [
        "# Rapport d'evaluation NER — FOOD / QUANTITY",
        "",
        f"- Seed: {SEED} | Epochs: {EPOCHS}",
        f"- Corpus annote: {report['corpus']['manual_total']} phrases "
        f"({report['corpus']['manual_train']} train / {report['corpus']['manual_test']} test, split 80/20)",
        f"- Exemples auto-generes (templates): {report['corpus']['auto_generated']}",
        f"- Total exemples d'entrainement: {report['corpus']['total_training_examples']}",
        "",
        "## Modele entraine (spaCy NER) — evalue sur le test set jamais vu",
        "",
        f"| | Precision | Recall | F1 |",
        f"|---|---|---|---|",
        f"| Global | {trained_scores['precision']:.3f} | {trained_scores['recall']:.3f} | {trained_scores['f1']:.3f} |",
    ]
    for label, v in trained_scores["per_entity"].items():
        md.append(f"| {label} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |")
    md += [
        "",
        "## Baseline regles (liste FOOD_DB + regex QUANTITY, sans apprentissage)",
        "",
        f"| | Precision | Recall | F1 |",
        f"|---|---|---|---|",
        f"| Global | {baseline_scores['precision']:.3f} | {baseline_scores['recall']:.3f} | {baseline_scores['f1']:.3f} |",
    ]
    for label, v in baseline_scores["per_entity"].items():
        md.append(f"| {label} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |")

    with open(OUT_DIR / "metrics_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # Bonus: loss curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", markersize=3)
        plt.xlabel("Epoch")
        plt.ylabel("NER loss")
        plt.title("Convergence de la loss NER pendant l'entrainement")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "loss_curve.png", dpi=150)
        print(f"[eval] Loss curve saved -> {OUT_DIR / 'loss_curve.png'}")
    except ImportError:
        print("[eval] matplotlib not installed, skipping loss curve plot")

    print(f"\n[eval] Report saved -> {OUT_DIR / 'metrics_report.json'} / metrics_report.md")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
