import re, csv
from pathlib import Path

CSV_PATH = Path(__file__).parent / "kaggle_nutrition.csv"

PORTION_WEIGHT_G = {
    "egg": 60, "apple": 150, "banana": 120, "orange": 180,
    "strawberry": 150, "grape": 100, "pear": 160, "mango": 200,
    "pineapple": 150, "kiwi": 80, "peach": 150, "blueberry": 100,
    "broccoli": 150, "carrot": 100, "tomato": 120, "lettuce": 80,
    "spinach": 100, "zucchini": 150, "pepper": 100, "onion": 80,
    "chicken": 150, "salmon": 150, "tuna": 120, "beef": 150,
    "shrimp": 100, "turkey": 150, "pork": 150,
    "rice": 180, "pasta": 200, "bread": 60, "toast": 60,
    "quinoa": 185, "oats": 60, "potato": 200, "sweet potato": 150,
    "milk": 200, "yogurt": 125, "cheese": 40, "butter": 15,
    "lentils": 200, "chickpeas": 150, "beans": 150, "tofu": 150,
    "almonds": 30, "walnuts": 30,
    "coffee": 200, "tea": 200, "juice": 200, "soda": 330, "water": 300,
    "pizza": 300, "burger": 200, "fries": 150, "chips": 50,
    "chocolate": 30, "honey": 15, "oil": 10, "default": 100,
}

FALLBACK = [
    ("apple", 52), ("banana", 89), ("orange", 47), ("strawberry", 32),
    ("grilled chicken", 165), ("chicken", 165), ("salmon", 208),
    ("tuna", 132), ("egg", 155), ("steak", 271), ("beef", 250),
    ("white rice", 130), ("brown rice", 112), ("pasta", 131),
    ("bread", 265), ("whole wheat bread", 247), ("potato", 77),
    ("sweet potato", 86), ("quinoa", 120), ("oats", 389),
    ("milk", 61), ("yogurt", 59), ("cheese", 402), ("butter", 717),
    ("broccoli", 34), ("carrot", 41), ("tomato", 18), ("lettuce", 15),
    ("spinach", 23), ("onion", 40), ("pepper", 31),
    ("lentils", 116), ("chickpeas", 164), ("tofu", 76),
    ("almonds", 579), ("walnuts", 654),
    ("pizza", 266), ("hamburger", 295), ("french fries", 312), ("chips", 536),
    ("soda", 42), ("orange juice", 45), ("coffee", 2), ("tea", 1), ("water", 0),
    ("dark chocolate", 546), ("honey", 304), ("olive oil", 884),
]

def _normalize(raw):
    raw = re.sub(r"\(.*?\)", "", raw)
    raw = re.sub(r"\b\d+\s*(oz|g|ml|cup|cups|slice|slices|tbsp|tsp)\b", "", raw, flags=re.I)
    raw = re.sub(r"\b(grilled|steamed|baked|fried|boiled|cooked|raw|large|small|whole|black|white|brown|low-fat|non-fat|scrambled|hard-boiled|roasted|mixed)\b", "", raw, flags=re.I)
    return re.sub(r"\s+", " ", raw).strip().lower()

def _portion(name):
    for k, v in PORTION_WEIGHT_G.items():
        if k in name:
            return v
    return PORTION_WEIGHT_G["default"]

def load():
    if not CSV_PATH.exists():
        return {n: k for n, k in FALLBACK}
    try:
        db = {}
        with open(CSV_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = _normalize(row.get("Food_Item", ""))
                if not name:
                    continue
                raw_kcal = str(row.get("Calories (kcal)", 0) or 0).strip()
                m_kcal = re.search(r"[\d.]+", raw_kcal)
                kcal_portion = float(m_kcal.group()) if m_kcal else 0.0
                portion_g = _portion(name)
                kcal_100g = round(kcal_portion * 100 / portion_g, 1)
                if name not in db:
                    db[name] = kcal_100g
        # fill gaps with fallback
        for name, kcal in FALLBACK:
            if name not in db:
                db[name] = kcal
        return db
    except Exception as e:
        print(f"[nutrition] CSV error: {e}, using fallback")
        return {n: k for n, k in FALLBACK}

FOOD_DB = load()