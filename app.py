
import pandas as pd
import streamlit as st

from src.charts import (
    render_machine_learning,
    render_part1,
    render_part2,
    render_part3,
    usd,
)
from src.data_loader import DISCOUNT_LABELS, load_data, margin_by

st.set_page_config(
    page_title="Global Superstore Analytics",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1500px !important;
        }
        h1, h2, h3, h4, p, span, label, div, small, [data-testid="stCaptionContainer"] {
            color: #000000 !important;
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        h1 {
            font-size: 2rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
            text-transform: uppercase;
            margin-bottom: 0 !important;
        }
        h3 {
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase;
            border-bottom: 2px solid #000000;
            padding-bottom: 6px !important;
            margin-bottom: 8px !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #ffffff !important;
            border: 1px solid #000000 !important;
            border-radius: 0px !important;
            padding: 18px !important;
            box-shadow: none !important;
        }
        div[data-testid="stMetric"] {
            background-color: #ffffff !important;
            border: 1px solid #000000 !important;
            border-radius: 0px !important;
            padding: 14px 18px !important;
        }
        div[data-testid="stMetricLabel"] > div {
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
        }
        div[data-testid="stMetricValue"] > div {
            font-size: 1.45rem !important;
            font-weight: 800 !important;
        }
        div[role="radiogroup"] {
            border-top: 2px solid #000000;
            border-bottom: 2px solid #000000;
            padding: 8px 0;
            gap: 1.5rem;
        }
        div[data-testid="stExpander"],
        details[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary {
            border: 1px solid #000000 !important;
            border-radius: 0px !important;
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="popover"] > div,
        ul[data-baseweb="menu"],
        ul[data-baseweb="menu"] li {
            background-color: #ffffff !important;
            color: #000000 !important;
            border-color: #000000 !important;
        }
        ul[data-baseweb="menu"] li:hover {
            background-color: #f0f0f0 !important;
        }
        span[data-baseweb="tag"],
        span[data-baseweb="tag"] span,
        span[data-baseweb="tag"] svg {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        .stButton > button {
            border: 1px solid #000000 !important;
            border-radius: 0px !important;
            background-color: #ffffff !important;
            color: #000000 !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            font-size: 0.75rem !important;
            letter-spacing: 0.05em;
        }
        .stButton > button:hover {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        hr {
            border-color: #000000 !important;
            opacity: 1 !important;
        }
        .chapter {
            border-left: 6px solid #000000;
            padding: 2px 0 2px 14px;
            margin: 1.5rem 0 1rem 0;
        }
        .chapter .kicker {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12em;
        }
        .chapter .title {
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            text-transform: uppercase;
        }
        .chapter .question {
            font-size: 0.95rem;
        }
        .takeaway {
            border-left: 3px solid #000000;
            background-color: #f4f4f4;
            padding: 8px 12px;
            margin-top: 10px;
            font-size: 0.88rem;
        }
        .insight-strip {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin: 10px 0 14px 0;
        }
        .insight-box {
            border: 1px solid #000000;
            border-left: 5px solid #000000;
            padding: 10px 14px;
            background: #fafafa;
        }
        .insight-box .lbl {
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #444444 !important;
        }
        .insight-box .val {
            font-size: 1.05rem;
            font-weight: 800;
            margin: 2px 0;
        }
        .insight-box .sub {
            font-size: 0.78rem;
            color: #333333 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

df, cleaning_report = load_data()

YEAR_MIN, YEAR_MAX = int(df["year"].min()), int(df["year"].max())
ALL_MARKETS = sorted(df["market"].unique())
ALL_CATS = sorted(df["category"].unique())


def reset_filters():
    st.session_state["f_years"] = (YEAR_MIN, YEAR_MAX)
    st.session_state["f_markets"] = list(ALL_MARKETS)
    st.session_state["f_cats"] = list(ALL_CATS)
    st.session_state["f_disc"] = list(DISCOUNT_LABELS)
    st.session_state["f_outliers"] = False


head_left, head_right = st.columns([3, 1])
with head_left:
    st.title("Global Superstore Analytics")
    st.caption("A STORY IN THREE PARTS AND A FINALE: WHERE DOES THE MONEY COME FROM, AND WHERE DOES IT LEAK?")
with head_right:
    with st.expander("DATA QUALITY REPORT"):
        st.dataframe(pd.Series(cleaning_report, name="value").astype(str), width="stretch")

with st.container(border=True):
    f1, f2, f3, f4, f5, f6 = st.columns([1.2, 1.4, 1.2, 1.2, 0.9, 0.7])
    with f1:
        year_range = st.slider("Year Range", YEAR_MIN, YEAR_MAX, (YEAR_MIN, YEAR_MAX), key="f_years")
    with f2:
        markets = st.multiselect("Market", ALL_MARKETS, default=ALL_MARKETS, key="f_markets")
    with f3:
        categories = st.multiselect("Category", ALL_CATS, default=ALL_CATS, key="f_cats")
    with f4:
        disc_groups = st.multiselect("Discount Group", DISCOUNT_LABELS, default=DISCOUNT_LABELS, key="f_disc")
    with f5:
        st.write("")
        hide_outliers = st.checkbox(
            "Exclude IQR Outliers", value=False, key="f_outliers",
            help="Hides orders whose profit is an extreme value (outside 1.5 x IQR).",
        )
    with f6:
        st.write("")
        st.button("Reset", on_click=reset_filters, use_container_width=True)

f = df[
    df["year"].between(*year_range)
    & df["market"].isin(markets)
    & df["category"].isin(categories)
    & df["discount_group"].isin(disc_groups)
]
if hide_outliers:
    f = f[~f["profit_outlier"]]

if f.empty:
    st.warning("No data for the selected filters. Please widen your selection or click Reset.")
    st.stop()

total_sales, total_profit = f["sales"].sum(), f["profit"].sum()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Sales", f"${total_sales:,.0f}")
k2.metric("Total Profit", f"${total_profit:,.0f}")
k3.metric("Profit Margin", f"{total_profit / total_sales * 100:.1f}%" if total_sales else "n/a")
k4.metric("Filtered Orders", f"{len(f):,}")
k5.metric("Loss-Making Orders", f"{f['is_loss'].mean() * 100:.1f}%")

sub_m = margin_by(f, "sub_category").dropna(subset=["margin"])
mkt_m = margin_by(f, "market").dropna(subset=["margin"])
disc_m = margin_by(f, "discount_group").dropna(subset=["margin"])

cards_html = []
if not sub_m.empty:
    best = sub_m.loc[sub_m["margin"].idxmax()]
    worst = sub_m.loc[sub_m["margin"].idxmin()]
    cards_html.append(
        f'<div class="insight-box" style="border-left-color:#2b83ba;">'
        f'<div class="lbl">Star Sub-Category</div>'
        f'<div class="val">{best["sub_category"]} ({best["margin"]:+.1f}%)</div>'
        f'<div class="sub">{usd(best["profit"])} net profit</div></div>'
    )
    cards_html.append(
        f'<div class="insight-box" style="border-left-color:#c62828;">'
        f'<div class="lbl">Biggest Margin Drag</div>'
        f'<div class="val">{worst["sub_category"]} ({worst["margin"]:+.1f}%)</div>'
        f'<div class="sub">{usd(worst["profit"])} net impact</div></div>'
    )
if not mkt_m.empty:
    top_m = mkt_m.loc[mkt_m["profit"].idxmax()]
    cards_html.append(
        f'<div class="insight-box">'
        f'<div class="lbl">Top Profit Market</div>'
        f'<div class="val">{top_m["market"]} ({usd(top_m["profit"])})</div>'
        f'<div class="sub">{top_m["margin"]:+.1f}% overall margin</div></div>'
    )
if not disc_m.empty:
    weak_d = disc_m.loc[disc_m["margin"].idxmin()]
    cards_html.append(
        f'<div class="insight-box" style="border-left-color:#e69f00;">'
        f'<div class="lbl">Weakest Discount Band</div>'
        f'<div class="val">{weak_d["discount_group"]} ({weak_d["margin"]:+.1f}%)</div>'
        f'<div class="sub">On {usd(weak_d["sales"])} sales volume</div></div>'
    )
if cards_html:
    st.markdown(f'<div class="insight-strip">{"".join(cards_html)}</div>', unsafe_allow_html=True)

PAGES = {
    "PART 1 / GROWTH & FOOTPRINT": lambda: render_part1(f),
    "PART 2 / THE DISCOUNT EFFECT": lambda: render_part2(f),
    "PART 3 / PINPOINTING THE LEAK": lambda: render_part3(f, total_profit),
    "FINALE / MACHINE LEARNING": lambda: render_machine_learning(f, df, ALL_CATS, ALL_MARKETS),
}

section = st.radio(
    "SECTION",
    [*PAGES.keys(), "FULL STORY"],
    horizontal=True,
    label_visibility="collapsed",
)

if section == "FULL STORY":
    for fn in PAGES.values():
        fn()
else:
    PAGES[section]()

st.caption(
    f"Source: Global Superstore (Kaggle), {len(df):,} rows after cleaning · "
    "CEN445 Introduction to Data Visualization"
)