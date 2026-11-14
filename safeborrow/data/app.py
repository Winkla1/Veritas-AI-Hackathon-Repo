"""SafeBorrow demo app.

Run locally with Streamlit, or deploy this file to Streamlit Community Cloud.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SafeBorrow", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#18312f; --muted:#637874; --teal:#087f70; --teal-soft:#e7f4f0; --line:#e2ebe7; }
.stApp { background:radial-gradient(ellipse at 92% 0%,#e6f2ed 0,transparent 34%),linear-gradient(180deg,#f8faf9 0%,#f4f7f5 100%); color:var(--ink); }
.block-container { max-width:1360px; padding-top:2.4rem; padding-bottom:3.5rem; }
html, body, [class*="css"] { font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
h1,h2,h3 { font-family:'Space Grotesk','Inter',sans-serif; color:var(--ink); letter-spacing:-.04em; }
h1 { font-size:2.55rem !important; font-weight:700 !important; }
h2 { font-size:1.5rem !important; font-weight:600 !important; }
[data-testid="stSidebar"] { background:#edf4f1; border-right:1px solid #dce8e3; }
[data-testid="stSidebar"] * { color:#18312f !important; }
[data-testid="stSidebar"] h2 { font-family:'Space Grotesk','Inter',sans-serif; font-size:1.08rem !important; letter-spacing:-.02em; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#37534d !important; font-weight:600; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea { background:#fff !important; color:#18312f !important; border-color:#d6e3dd !important; border-radius:10px !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background:#fff !important; border-color:#d6e3dd !important; border-radius:10px !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] { background:#087f70 !important; }
[data-testid="stMetric"] { background:rgba(255,255,255,.92); border:1px solid var(--line); border-radius:18px; padding:20px 22px; box-shadow:0 8px 28px rgba(26,61,51,.055); }
[data-testid="stMetricLabel"] p { color:var(--muted); font-weight:600; }
[data-testid="stMetricValue"] { color:var(--ink); font-family:'Space Grotesk','Inter',sans-serif; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:20px; border-color:var(--line); background:rgba(255,255,255,.88); box-shadow:0 8px 28px rgba(26,61,51,.045); }
.eyebrow { color:var(--teal); font-size:.74rem; text-transform:uppercase; letter-spacing:.16em; font-weight:700; margin-bottom:.45rem; }
.subcopy { color:var(--muted); font-size:1.04rem; margin-top:-.55rem; }
.risk-card { border-radius:20px; padding:22px 26px; margin:.75rem 0 1.15rem; box-shadow:0 8px 22px rgba(26,61,51,.045); }
.risk-label { font-size:.76rem; text-transform:uppercase; letter-spacing:.13em; font-weight:700; opacity:.8; }
.risk-number { font-family:'Space Grotesk','Inter',sans-serif; font-size:2.4rem; font-weight:700; line-height:1.2; margin:.3rem 0; letter-spacing:-.04em; }
.risk-note { font-size:.88rem; opacity:.82; }
div[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; }
button[kind="secondary"] { border-radius:11px; }
</style>
""", unsafe_allow_html=True)


# ---------- Load the trained model saved by the notebook ----------
@st.cache_resource
def load_bundle():
    model_path = Path(__file__).with_name("safeborrow_model.joblib")
    return joblib.load(model_path)


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("safeborrow_model.joblib not found. Run every cell in safeborrow.ipynb first (Step 13 saves it).")
    st.stop()

models = bundle["models"]
blocks = bundle["blocks"]
BLOCK = bundle["block_size"]
START = bundle["start"]
FORECAST_YEAR = bundle["forecast_year"]
SAFE_RISK = 0.10


# ---------- Same functions as the notebook (keep them in sync) ----------
def make_features(frame):
    X = pd.DataFrame(index=frame.index)
    X["dayofweek"] = frame["Date"].dt.dayofweek
    X["day"] = frame["Date"].dt.day
    X["month"] = frame["Date"].dt.month
    X["weekofyear"] = frame["Date"].dt.isocalendar().week.astype(int).values
    X["holiday"] = frame["holiday"].astype(int)
    X["workingday"] = frame["workingday"].astype(int)
    X["year"] = frame["Date"].dt.year - START.year
    return X


@st.cache_data
def simulate(start_month, term_weeks, n_sims=1000, seed=0):
    """Forecast the loan period and simulate n_sims possible futures.
    A fixed seed keeps the numbers steady while you move the sliders."""
    start = pd.Timestamp(year=FORECAST_YEAR, month=start_month, day=1)
    dates = pd.date_range(start, periods=term_weeks * 7, freq="D")
    fut = pd.DataFrame({"Date": dates, "holiday": 0,
                        "workingday": (dates.dayofweek < 5).astype(int)})
    median = np.expm1(models[0.5].predict(make_features(fut)))

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(median) / BLOCK))
    picks = rng.integers(0, len(blocks), size=(n_sims, n_blocks))
    r = blocks[picks].reshape(n_sims, -1)[:, :len(median)]
    return dates, median, median * r


def repayment_plan(amount, monthly_rate, term_weeks, schedule):
    n_days = term_weeks * 7
    total = amount * (1 + monthly_rate * n_days / 30)
    days = np.arange(1, n_days + 1)
    if schedule == "daily":
        due = days
    elif schedule == "weekly":
        due = days[days % 7 == 0]
    else:
        due = np.array([n_days])
    plan = np.zeros(n_days)
    plan[due - 1] = total / len(due)
    return plan, total


def missed_payment_risk(sales_sims, plan, margin, daily_costs, start_cash):
    profit = sales_sims * margin - daily_costs
    cash = start_cash + profit.cumsum(axis=1) - plan.cumsum()
    due = plan > 0
    missed = (cash[:, due] < 0).any(axis=1)
    return missed.mean(), cash


def safe_amount(sims, rate, term_weeks, schedule, margin, daily_costs, start_cash):
    best = 0
    for a in range(50, 5001, 50):
        plan, _ = repayment_plan(a, rate, term_weeks, schedule)
        risk, _ = missed_payment_risk(sims, plan, margin, daily_costs, start_cash)
        if risk > SAFE_RISK:
            break
        best = a
    return best


SCHEDULE_NAMES = {"daily": "Every day", "weekly": "Every week", "end": "All at the end"}

# ---------- Inputs ----------
with st.sidebar:
    st.header("Your business")
    margin = st.slider("Profit margin (%)", 5, 60, 25) / 100
    daily_costs = st.number_input("Daily costs: rent, transport, family (GHS)", 0, 2000, 50, step=10)
    start_cash = st.number_input("Cash you have today (GHS)", 0, 20000, 200, step=50)

    st.header("The loan offer")
    amount = st.slider("Loan amount (GHS)", 50, 3000, 800, step=50)
    rate = st.slider("Interest per month (%)", 0.0, 30.0, 10.0, step=0.5) / 100
    term_weeks = st.slider("Loan length (weeks)", 1, 12, 4)
    start_month = st.selectbox("Loan starts in", range(1, 13), index=4,
                               format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"))
    schedule = st.radio("Repayment schedule", list(SCHEDULE_NAMES), format_func=SCHEDULE_NAMES.get)

# ---------- Results ----------
dates, median, sims = simulate(start_month, term_weeks)
plan, total = repayment_plan(amount, rate, term_weeks, schedule)
risk, cash = missed_payment_risk(sims, plan, margin, daily_costs, start_cash)

st.markdown('<div class="eyebrow">Cash-flow confidence for small businesses</div>', unsafe_allow_html=True)
st.title("Know what you can safely borrow.")
st.markdown('<div class="subcopy">See how a loan could fit your business before you accept the offer.</div>', unsafe_allow_html=True)

if risk <= SAFE_RISK:
    risk_colour, verdict, risk_bg = "#087f70", "Comfortable buffer", "#e7f4f0"
elif risk <= 0.25:
    risk_colour, verdict, risk_bg = "#b97812", "Worth a closer look", "#fff4dc"
else:
    risk_colour, verdict, risk_bg = "#b94b45", "High chance of strain", "#fcebea"

safe = safe_amount(sims, rate, term_weeks, schedule, margin, daily_costs, start_cash)
st.markdown(f'''<div class="risk-card" style="background:{risk_bg};color:{risk_colour}">
<div class="risk-label">Estimated missed-payment chance</div>
<div class="risk-number">{risk:.0%} <span style="font-size:1.05rem;font-weight:700">· {verdict}</span></div>
<div class="risk-note">Based on 1,000 simulated cash-flow paths over the loan period.</div></div>''', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Loan amount", f"GHS {amount:,.0f}", f"{term_weeks} week term")
with col2:
    st.metric("Total repayment", f"GHS {total:,.0f}", f"GHS {total - amount:,.0f} interest", delta_color="off")
with col3:
    st.metric("Suggested safe limit", f"GHS {safe:,}", "10% risk threshold",
              help="The largest tested loan amount that keeps estimated missed-payment risk at 10% or less.")

st.write("")
chart_col, info_col = st.columns([1.65, 1], gap="large")
with chart_col:
    with st.container(border=True):
        st.subheader("Cash through the loan period")
        st.caption("Simulated business cash balance after operating costs and repayments")
        fig, ax = plt.subplots(figsize=(10, 4.1))
        low, high = np.percentile(cash, [10, 90], axis=0)
        typical = np.median(cash, axis=0)
        ax.fill_between(dates, low, high, color="#b8ded2", alpha=.55, label="middle 80% of outcomes")
        ax.plot(dates, typical, color="#087f70", linewidth=2.8, label="typical outcome")
        ax.axhline(0, color="#cc6d61", linewidth=1.5, linestyle=(0, (4, 4)), label="zero cash")
        ax.set_ylabel("Cash balance (GHS)", color="#647780", labelpad=10)
        ax.grid(axis="y", color="#e8eeeb", linewidth=.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#dce6e1")
        ax.tick_params(colors="#647780", length=0, pad=8)
        ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

rows = []
for sch in SCHEDULE_NAMES:
    p, _ = repayment_plan(amount, rate, term_weeks, sch)
    r, _ = missed_payment_risk(sims, p, margin, daily_costs, start_cash)
    rows.append({"Repayment timing": SCHEDULE_NAMES[sch],
                 "Missed-payment chance": f"{r:.0%}",
                 "Safe limit": f"GHS {safe_amount(sims, rate, term_weeks, sch, margin, daily_costs, start_cash):,}"})
with info_col:
    with st.container(border=True):
        st.subheader("Compare repayment timing")
        st.caption("The same loan can feel very different depending on when payments are due.")
        st.dataframe(pd.DataFrame(rows).set_index("Repayment timing"), use_container_width=True)
        st.caption("Safe limit keeps estimated risk at or below 10%.")

st.markdown("---")
st.caption("Illustrative estimate using historical bike-rental demand patterns scaled to GHS. Actual sales and outcomes may differ. Assumes the loan funds are spent immediately.")
