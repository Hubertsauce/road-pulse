"""
Road Pulse - collect.py
Haalt nieuwe meldingen over wegen en stoepen op bij FixMyStreet (VK),
analyseert het sentiment met een voorgetraind Hugging Face-model
en voegt ze toe aan data/sentiment.csv.

Wordt elke 15 minuten gedraaid door GitHub Actions (.github/workflows/collect.yml).
"""

import csv
import json
import os
import re
from datetime import datetime, timedelta, timezone

import requests

# ---------------------------------------------------------------- instellingen
API_URL = "https://www.fixmystreet.com/open311/v2/requests.json"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "sentiment.csv")
STATE_PATH = os.path.join(DATA_DIR, "state.json")

# Categorieen die over het wegdek / de stoep gaan (hoofdletters maken niet uit).
ROAD_PATTERN = re.compile(
    r"pothole|road surface|carriageway|pavement|footway|footpath|kerb|"
    r"tarmac|road defect|surface defect|crack|sunken|uneven",
    re.IGNORECASE,
)

FIRST_RUN_LOOKBACK_HOURS = 1   # bij de allereerste run: zoveel uur terug
OVERLAP_HOURS = 2              # extra overlap, want de API negeert tijdzones
MAX_GAP_HOURS = 12             # nooit verder terug dan dit (na een lange storing)

FIELDS = [
    "id", "requested_at", "fetched_at", "category", "council",
    "text", "label", "score_neg", "score_neu", "score_pos", "sentiment",
]
# Bewust NIET opgeslagen: naam melder (zit niet in de API), coordinaten
# (lat/long kunnen naar iemands huis wijzen), foto's.


# ---------------------------------------------------------------- hulpfuncties
def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def known_ids():
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f)}


def fetch_window(start, end):
    """Haal alle meldingen op tussen start en end, in blokken van 1 uur."""
    items = []
    t = start
    while t < end:
        t2 = min(t + timedelta(hours=1), end)
        params = {
            "jurisdiction_id": "fixmystreet",
            "start_date": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_date": t2.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        r = requests.get(
            API_URL, params=params, timeout=60,
            headers={"User-Agent": "RoadPulse student project (hackathon)"},
        )
        r.raise_for_status()
        items.extend(r.json().get("service_requests", []))
        t = t2
    return items


def is_road_item(item):
    category = f"{item.get('service_name', '')} {item.get('service_code', '')}"
    return bool(ROAD_PATTERN.search(category))


def clean_text(item):
    text = item.get("description") or item.get("detail") or item.get("title") or ""
    return " ".join(text.split())  # enters/dubbele spaties weg


def council_of(item):
    agency = item.get("agency_responsible") or {}
    rec = agency.get("recipient") if isinstance(agency, dict) else agency
    if isinstance(rec, list):
        return "; ".join(map(str, rec))
    return str(rec or "")


def load_model():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=MODEL_NAME, top_k=None)


def analyse(clf, texts):
    """Geeft per tekst een dict met kansen voor negative/neutral/positive."""
    out = []
    results = clf(texts, truncation=True, max_length=512, batch_size=16)
    for res in results:
        if isinstance(res, dict):  # oudere transformers-versies
            res = [res]
        scores = {d["label"].lower(): d["score"] for d in res}
        out.append(scores)
    return out


# ---------------------------------------------------------------- hoofdprogramma
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    now = datetime.now(timezone.utc)
    state = load_state()

    if "last_run" in state:
        last = datetime.fromisoformat(state["last_run"])
        start = max(last - timedelta(hours=OVERLAP_HOURS),
                    now - timedelta(hours=MAX_GAP_HOURS))
    else:
        start = now - timedelta(hours=FIRST_RUN_LOOKBACK_HOURS + OVERLAP_HOURS)
    end = now + timedelta(hours=OVERLAP_HOURS)

    raw = fetch_window(start, end)
    seen = known_ids()
    new_items, batch_ids = [], set()
    for item in raw:
        sid = str(item.get("service_request_id", ""))
        if not sid or sid in seen or sid in batch_ids:
            continue
        if not is_road_item(item):
            continue
        text = clean_text(item)
        if len(text) < 3:
            continue
        batch_ids.add(sid)
        new_items.append((sid, item, text))

    print(f"Opgehaald: {len(raw)} meldingen, nieuw over wegen/stoepen: {len(new_items)}")

    if new_items:
        clf = load_model()
        scores = analyse(clf, [t for _, _, t in new_items])
        write_header = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            for (sid, item, text), s in zip(new_items, scores):
                neg, neu, pos = (s.get("negative", 0.0), s.get("neutral", 0.0),
                                 s.get("positive", 0.0))
                label = max(("negative", neg), ("neutral", neu),
                            ("positive", pos), key=lambda x: x[1])[0]
                w.writerow({
                    "id": sid,
                    "requested_at": item.get("requested_datetime", ""),
                    "fetched_at": now.isoformat(timespec="seconds"),
                    "category": item.get("service_name", ""),
                    "council": council_of(item),
                    "text": text,
                    "label": label,
                    "score_neg": round(neg, 4),
                    "score_neu": round(neu, 4),
                    "score_pos": round(pos, 4),
                    # -1 = heel negatief, 0 = neutraal, +1 = heel positief
                    "sentiment": round(pos - neg, 4),
                })

    state["last_run"] = now.isoformat(timespec="seconds")
    state["total_items"] = len(seen) + len(new_items)
    save_state(state)
    print(f"Totaal in dataset: {state['total_items']}")


if __name__ == "__main__":
    main()
