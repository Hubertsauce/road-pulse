"""
Road Pulse - test/compare.py
Compares our own labels (test/label_sheet.xlsx) with:
  - model 1: cardiffnlp/twitter-roberta-base-sentiment-latest
             (its labels are already stored by collect.py in data/sentiment.csv)
  - model 2: VADER, a simple word-list baseline (runs here)
and writes test/test_table.csv.

Run from the repository root:
    pip install pandas openpyxl vaderSentiment
    python test/compare.py
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

LABELS = "test/label_sheet.xlsx"
DATA = "data/sentiment.csv"
OUT = "test/test_table.csv"

# 1. Our own labels (one person, see README)
lab = pd.read_excel(LABELS, sheet_name="Label")
lab["my_label"] = lab["my_label"].str.strip().str.lower()

# 2. Model 1: RoBERTa output, stored by collect.py, joined on the report id
data = pd.read_csv(DATA)[["id", "label", "sentiment"]]
data = data.rename(columns={"label": "roberta", "sentiment": "roberta_score"})
t = lab.merge(data, on="id", how="left")

# 3. Model 2: VADER word list (compound score; standard thresholds)
vader = SentimentIntensityAnalyzer()
t["vader_score"] = t["report"].map(lambda x: vader.polarity_scores(x)["compound"])
t["vader"] = t["vader_score"].map(
    lambda c: "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral")

# 4. Agreement
t["roberta_agrees"] = (t["roberta"] == t["my_label"]).map({True: "yes", False: "no"})
t["vader_agrees"] = (t["vader"] == t["my_label"]).map({True: "yes", False: "no"})
t["models_disagree"] = (t["roberta"] != t["vader"]).map({True: "yes", False: "no"})

n = len(t)
for model in ["roberta", "vader"]:
    hits = (t[model] == t["my_label"]).sum()
    print(f"\n{model}: {hits}/{n} = {hits / n:.0%} agreement with our labels")
    print(pd.crosstab(t["my_label"], t[model], rownames=["our label"], colnames=[model]))
print(f"\nModels disagree with each other on {(t['models_disagree'] == 'yes').sum()} reports")

cols = ["nr", "id", "category", "report", "my_label", "note", "roberta", "roberta_score",
        "vader", "vader_score", "roberta_agrees", "vader_agrees", "models_disagree"]
t[cols].to_csv(OUT, index=False)
print(f"\nSaved {OUT}")
