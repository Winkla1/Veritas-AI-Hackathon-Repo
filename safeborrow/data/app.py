"""SafeBorrow demo app.

Run the notebook first (it saves safeborrow_model.joblib), then in the terminal:
    streamlit run app.py
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SafeBorrow", layout="wide")


# ---------- Load the trained model saved by the notebook ----------
@st.cache_resource
def load_bundle():
    return joblib.load("safeborrow_model.joblib")


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

st.title("SafeBorrow")
st.caption("Can my business actually afford this loan?")

if risk <= SAFE_RISK:
    colour, verdict = "green", "Low risk"
elif risk <= 0.25:
    colour, verdict = "orange", "Risky"
else:
    colour, verdict = "red", "High risk"

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Chance of missing a repayment**")
    st.markdown(f"<h1 style='color:{colour}; margin:0'>{risk:.0%}</h1><b style='color:{colour}'>{verdict}</b>",
                unsafe_allow_html=True)
with col2:
    st.metric("Total you will repay", f"GHS {total:,.0f}", f"GHS {total - amount:,.0f} interest", delta_color="off")
with col3:
    safe = safe_amount(sims, rate, term_weeks, schedule, margin, daily_costs, start_cash)
    st.metric(f"Safe loan size ({SCHEDULE_NAMES[schedule].lower()})", f"GHS {safe:,}",
              help="The biggest loan that keeps the chance of missing a repayment at 10% or less.")

st.subheader("Your cash balance in 100 possible futures")
fig, ax = plt.subplots(figsize=(11, 4))
for i in range(100):
    ax.plot(dates, cash[i], color="tab:blue", alpha=0.15)
ax.plot(dates, np.median(cash, axis=0), color="tab:blue", linewidth=2, label="typical future")
ax.axhline(0, color="red", linewidth=2, label="cash runs out")
ax.set_ylabel("GHS")
ax.legend(loc="upper left")
st.pyplot(fig)

st.subheader("Which schedule is safest?")
rows = []
for sch in SCHEDULE_NAMES:
    p, _ = repayment_plan(amount, rate, term_weeks, sch)
    r, _ = missed_payment_risk(sims, p, margin, daily_costs, start_cash)
    rows.append({"Schedule": SCHEDULE_NAMES[sch],
                 "Risk for this loan": f"{r:.0%}",
                 "Safe loan size": f"GHS {safe_amount(sims, rate, term_weeks, sch, margin, daily_costs, start_cash):,}"})
st.table(pd.DataFrame(rows).set_index("Schedule"))

st.caption("Example business: real daily rental demand (Bike Sharing dataset, 2011–2012) scaled to GHS. "
           "Risk comes from 1,000 simulated futures built from the forecaster's real errors. "
           "Assumes the loan money is spent straight away.")
