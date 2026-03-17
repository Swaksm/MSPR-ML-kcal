"""
NLP parser — uses trained spaCy NER model if available,
falls back to regex otherwise.
"""

import re, sys
from pathlib import Path
from difflib import get_close_matches

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.nutrition_data import FOOD_DB

FOODS     = sorted(FOOD_DB.keys(), key=len, reverse=True)
MODEL_DIR = Path(__file__).parent / "food_ner_model"

PORTIONS = {
    "egg": 60, "grilled chicken": 150, "chicken": 150, "steak": 200,
    "salmon": 150, "tuna": 120, "beef": 150, "shrimp": 100, "turkey": 150,
    "white rice": 180, "brown rice": 180, "pasta": 200,
    "bread": 60, "whole wheat bread": 60, "toast": 40,
    "potato": 200, "sweet potato": 150, "quinoa": 185, "oats": 60,
    "milk": 200, "yogurt": 125, "cheese": 40, "butter": 15,
    "apple": 150, "banana": 120, "orange": 180, "strawberry": 150,
    "grape": 100, "pear": 160, "mango": 200, "kiwi": 80,
    "broccoli": 150, "carrot": 100, "tomato": 120, "lettuce": 80,
    "spinach": 100, "pepper": 100, "onion": 80,
    "almonds": 30, "walnuts": 30,
    "soda": 330, "orange juice": 200, "coffee": 200, "tea": 200, "water": 300,
    "pizza": 300, "hamburger": 200, "french fries": 150, "chips": 50,
    "dark chocolate": 30, "honey": 15, "olive oil": 10,
}

UNITS = {
    "g":1,"gr":1,"gram":1,"grams":1,"kg":1000,
    "ml":1,"cl":10,"dl":100,"l":1000,
    "oz":28.35,"lb":453.6,
    "cup":240,"cups":240,"bowl":300,"glass":200,
    "slice":35,"slices":35,"piece":100,"pieces":100,
    "tbsp":15,"tsp":5,"handful":30,"scoop":35,
}

WORDS = {
    "one":1,"two":2,"three":3,"four":4,"five":5,
    "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "a":1,"an":1,"half":0.5,
}

IGNORED = {
    "grilled","baked","steamed","fried","boiled","cooked","raw","roasted",
    "scrambled","fresh","whole","large","small","with","and","a","an",
    "the","of","some","bit","little","had","ate","have","eat","just",
}

# Explicit synonyms → canonical food name
SYNONYMS = {
    "eggs": "egg",
    "salad": "lettuce",
    "fries": "french fries",
    "coke": "soda", "cola": "soda", "pepsi": "soda",
    "juice": "orange juice",
    "rice": "white rice",
    "noodles": "pasta", "spaghetti": "pasta",
    "meat": "steak",
    "chiken": "chicken", "chikn": "chicken",
    "samon": "salmon", "salman": "salmon",
    "tomatoe": "tomato", "tomatoes": "tomato",
    "carrots": "carrot", "carots": "carrot",
    "potatoes": "potato",
    "burger": "hamburger", "burguer": "hamburger",
    "chocolate": "dark chocolate",
    "oil": "olive oil",
}


def _clean(raw: str) -> str:
    return " ".join(w for w in raw.strip().split() if w not in IGNORED and len(w) > 1)

def _match_food(raw: str) -> str | None:
    raw = raw.strip().lower()

    # Check synonyms first (before cleaning)
    if raw in SYNONYMS:
        return SYNONYMS[raw]
    # Multi-word synonym check
    for syn, canonical in SYNONYMS.items():
        if syn in raw:
            return canonical

    raw = _clean(raw)
    if not raw:
        return None

    # Exact match
    if raw in FOOD_DB:
        return raw

    # Synonym after clean
    if raw in SYNONYMS:
        return SYNONYMS[raw]

    # Partial match — longest food name first to avoid "egg" matching "egg white"
    for f in FOODS:
        if f == raw:
            return f
        if f in raw and len(f) >= 3:
            return f
        if raw in f and len(raw) >= 4:
            return f

    # Fuzzy match
    m = get_close_matches(raw, FOODS, n=1, cutoff=0.72)
    return m[0] if m else None

def _grams(food: str, qty: float, unit: str) -> float:
    unit = unit.lower().strip()
    if unit in ("g","gr","gram","grams"): return qty
    if unit in UNITS: return qty * UNITS[unit]
    return qty * PORTIONS.get(food, 100)

def _dedup(items: list) -> list:
    result, seen = [], []
    for item in sorted(items, key=lambda x: len(x["food"]), reverse=True):
        f = item["food"]
        if not any(f in s or s in f for s in seen):
            result.append(item)
            seen.append(f)
    return result

def _qty_from_text(text: str) -> tuple[float, str]:
    text = text.strip().lower()
    m = re.match(r'^(\d+(?:\.\d+)?)\s*(g|gr|grams?|kg|ml|cl|oz|lb)$', text)
    if m: return float(m.group(1)), m.group(2)
    m = re.match(r'^(\d+(?:\.\d+)?)\s*(\w+)?$', text)
    if m: return float(m.group(1)), m.group(2) or ""
    if text in WORDS: return WORDS[text], ""
    return 1.0, ""


# ── spaCy NER ─────────────────────────────────────────────────────────────────

_nlp = None

def _load_spacy():
    global _nlp
    if _nlp is None and MODEL_DIR.exists():
        import spacy
        _nlp = spacy.load(MODEL_DIR)
    return _nlp

def _parse_spacy(text: str) -> list[dict] | None:
    nlp = _load_spacy()
    if nlp is None:
        return None

    doc = nlp(text.lower())
    items = []
    last_qty, last_unit = 1.0, ""

    for ent in doc.ents:
        if ent.label_ == "QUANTITY":
            last_qty, last_unit = _qty_from_text(ent.text)
        elif ent.label_ == "FOOD":
            food = _match_food(ent.text)
            if food:
                grams = _grams(food, last_qty, last_unit)
                items.append({"food": food, "grams": grams})
                last_qty, last_unit = 1.0, ""

    return _dedup(items) if items else None


# ── Regex fallback ────────────────────────────────────────────────────────────

def _parse_regex(text: str) -> list[dict]:
    text = re.sub(r"[,;.!?()\[\]]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    found, pos = [], []

    for m in re.finditer(
        r'(\d+(?:[.,]\d+)?)\s*(g|gr|grams?|kg|ml|cl|oz|lb|cups?|bowls?|glasses?|slices?|pieces?|tbsp|tsp|handfuls?|scoops?)\s*(?:of\s+)?([a-z][a-z\s]{1,30})',
        text
    ):
        food = _match_food(m.group(3))
        if food:
            found.append({"food": food, "grams": _grams(food, float(m.group(1).replace(",",".")), m.group(2))})
            pos.append((m.start(), m.end()))

    for m in re.finditer(r'(\d+(?:[.,]\d+)?)\s+(?:of\s+)?([a-z][a-z\s]{1,25})', text):
        if any(s <= m.start() <= e for s,e in pos): continue
        food = _match_food(m.group(2))
        if food:
            found.append({"food": food, "grams": _grams(food, float(m.group(1).replace(",",".")), "")})
            pos.append((m.start(), m.end()))

    for m in re.finditer(
        r'\b(one|two|three|four|five|six|seven|eight|nine|ten|a|an|half)\b\s+(?:cups?\s+of\s+|slices?\s+of\s+|pieces?\s+of\s+)?([a-z][a-z\s]{1,25})',
        text
    ):
        if any(s <= m.start() <= e for s,e in pos): continue
        food = _match_food(m.group(2))
        if food:
            found.append({"food": food, "grams": _grams(food, WORDS.get(m.group(1), 1), "")})
            pos.append((m.start(), m.end()))

    for food in FOODS:
        if food in text and not any(i["food"] == food for i in found):
            found.append({"food": food, "grams": PORTIONS.get(food, 100)})

    return _dedup(found)


# ── Public interface ──────────────────────────────────────────────────────────

def parse(text: str) -> list[dict]:
    result = _parse_spacy(text)
    if result is not None:
        if not result:
            result = _parse_regex(text)
        return result
    return _parse_regex(text)

def parser_info() -> str:
    if MODEL_DIR.exists():
        return "spaCy NER (trained model)"
    return "regex fallback (run nlp/train_ner.py to train)"