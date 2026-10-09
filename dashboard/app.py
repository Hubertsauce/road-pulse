"""
Road Pulse - dashboard
Shows how frustrated people are about roads and pavements (FixMyStreet, UK),
per hour or per day, and which reports sit behind a peak or a dip.

Run locally:  streamlit run dashboard/app.py
"""

import os

import altair as alt
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------ settings
# Fill in your own GitHub username and repository name, so the dashboard always
# reads the latest data collected by GitHub Actions.
GITHUB_USER = "Hubertsauce"
GITHUB_REPO = "road-pulse"
DATA_URL = (f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}"
            f"/main/data/sentiment.csv")
LOCAL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sentiment.csv")

MIN_ITEMS = 5  # periods with fewer reports are flagged as unreliable
LINE_COLOR = "#2a6fdb"
BAR_COLOR = "#8a94a6"

st.set_page_config(page_title="Road Pulse", page_icon="🛣️", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data(ttl=300)  # reload every 5 minutes
def load_data():
    source = "GitHub (live)"
    try:
        if "YOUR-GITHUB-USERNAME" in DATA_URL:
            raise ValueError("no GitHub username filled in")
        df = pd.read_csv(DATA_URL)
    except Exception:
        df = pd.read_csv(LOCAL_PATH)
        source = "local file"
    df["time"] = (pd.to_datetime(df["requested_at"], utc=True, errors="coerce")
                    .dt.tz_convert("Europe/London"))
    df = df.dropna(subset=["time"])
    df["council"] = df["council"].fillna("unknown")
    df["category"] = df["category"].fillna("unknown")
    return df, source


df, source = load_data()

# ------------------------------------------------------------------ header
st.title("🛣️ Road Pulse")
st.caption("How frustrated are people about roads and pavements? "
           "Live reports from FixMyStreet (United Kingdom), "
           "read by a sentiment model.")

st.warning(
    "**Read this before drawing conclusions.** "
    "These are only people who report a problem online themselves. Those who don't "
    "(many older people, people without a smartphone, non-English speakers) are missing. "
    "The model was trained on English tweets and misses sarcasm and factual reports "
    "without emotion. Always look at the **number of reports** next to the score: "
    "a dip based on 3 reports means very little.",
    icon="⚠️",
)

# ------------------------------------------------------------------ filters
c1, c2, c3 = st.columns([1, 2, 2])
grain = c1.radio("Per", ["hour", "day"], horizontal=True)
councils = c2.multiselect("Council", sorted(df["council"].unique()),
                          placeholder="All councils")
cats = c3.multiselect("Category", sorted(df["category"].unique()),
                      placeholder="All categories")

f = df.copy()
if councils:
    f = f[f["council"].isin(councils)]
if cats:
    f = f[f["category"].isin(cats)]

if f.empty:
    st.info("No reports for this selection.")
    st.stop()

freq = "h" if grain == "hour" else "D"
f["period"] = f["time"].dt.floor(freq)

agg = (f.groupby("period")
         .agg(n=("id", "size"),
              sentiment=("sentiment", "mean"),
              pct_neg=("label", lambda s: (s == "negative").mean() * 100))
         .reset_index())
agg["period_naive"] = agg["period"].dt.tz_localize(None)
agg["reliable"] = agg["n"].ge(MIN_ITEMS).map({True: "yes", False: "too few reports"})

# ------------------------------------------------------------------ key figures
k1, k2, k3, k4 = st.columns(4)
k1.metric("Reports", f"{len(f):,}")
k2.metric("Negative", f"{(f['label'] == 'negative').mean() * 100:.0f}%")
k3.metric("Avg. sentiment", f"{f['sentiment'].mean():+.2f}",
          help="-1 = very negative, 0 = neutral, +1 = very positive")
k4.metric("Latest report", f["time"].max().strftime("%d %b %H:%M"))

# ------------------------------------------------------------------ charts
x = alt.X("period_naive:T", title=None,
          axis=alt.Axis(format="%d %b %H:%M" if grain == "hour" else "%a %d %b"))
tooltip = [
    alt.Tooltip("period_naive:T", title="Period",
                format="%d %b %H:%M" if grain == "hour" else "%d %b %Y"),
    alt.Tooltip("sentiment:Q", title="Avg. sentiment", format="+.2f"),
    alt.Tooltip("pct_neg:Q", title="% negative", format=".0f"),
    alt.Tooltip("n:Q", title="Number of reports"),
    alt.Tooltip("reliable:N", title="Enough data?"),
]

st.subheader(f"Average sentiment per {grain}")
st.caption("Lower = angrier. Open circles = fewer than "
           f"{MIN_ITEMS} reports, so unreliable.")
base = alt.Chart(agg).encode(x=x)
zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
    color="#c4c9d4", strokeDash=[4, 4]).encode(y="y:Q")
line = base.mark_line(color=LINE_COLOR, strokeWidth=2).encode(
    y=alt.Y("sentiment:Q", title="Sentiment (-1 to +1)",
            scale=alt.Scale(domain=[-1, 1])))
points = base.mark_point(size=70, strokeWidth=2, color=LINE_COLOR).encode(
    y="sentiment:Q",
    fill=alt.condition(alt.datum.n >= MIN_ITEMS, alt.value(LINE_COLOR),
                       alt.value("white")),
    tooltip=tooltip)
st.altair_chart((zero + line + points).properties(height=300),
                use_container_width=True)

st.subheader(f"Number of reports per {grain}")
bars = base.mark_bar(color=BAR_COLOR, cornerRadiusTopLeft=3,
                     cornerRadiusTopRight=3).encode(
    y=alt.Y("n:Q", title="Reports"), tooltip=tooltip)
st.altair_chart(bars.properties(height=160), use_container_width=True)

# ------------------------------------------------------------------ behind the peak/dip
st.subheader("🔎 What is behind a peak or a dip?")

reliable = agg[agg["n"] >= MIN_ITEMS]
pool = reliable if not reliable.empty else agg
dip = pool.loc[pool["sentiment"].idxmin(), "period"]
peak = pool.loc[pool["sentiment"].idxmax(), "period"]

fmt = "%a %d %b %H:00" if grain == "hour" else "%a %d %b %Y"
labels = {
    row.period: f"{row.period.strftime(fmt)}  ·  {row.n} reports  ·  "
                f"sentiment {row.sentiment:+.2f}"
    for row in agg.sort_values("period", ascending=False).itertuples()
}

b1, b2, _ = st.columns([1, 1, 3])
if "chosen" not in st.session_state or st.session_state["chosen"] not in labels:
    st.session_state["chosen"] = dip
if b1.button("⬇️ Lowest point (dip)"):
    st.session_state["chosen"] = dip
if b2.button("⬆️ Highest point (peak)"):
    st.session_state["chosen"] = peak

chosen = st.selectbox("Or pick a period yourself", list(labels),
                      format_func=labels.get, key="chosen")

items = (f[f["period"] == chosen]
         .sort_values("sentiment")
         [["time", "label", "sentiment", "category", "council", "text"]]
         .rename(columns={"sentiment": "score", "text": "report"}))
items["time"] = items["time"].dt.strftime("%d %b %H:%M")
st.dataframe(
    items, hide_index=True, use_container_width=True,
    column_config={
        "score": st.column_config.ProgressColumn(
            "score", min_value=-1, max_value=1, format="%.2f"),
        "report": st.column_config.TextColumn("report", width="large"),
    },
)

# ------------------------------------------------------------------ about
with st.expander("About this data"):
    st.markdown(f"""
- **Source:** FixMyStreet (mySociety), Open311 API. Only road surface and pavement categories.
- **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (English, trained on tweets).
- **Score:** probability of positive minus probability of negative (-1 to +1).
- **Privacy:** no names, coordinates or photos are stored.
- **Data loaded from:** {source} · {len(df)} reports in total.
- **Do not use for:** judging individual staff or reporters, or as the only reason to
  repair a road or not. This is a signal, not a measurement of road quality.
""")
