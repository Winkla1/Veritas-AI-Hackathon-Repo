"""SafeBorrow demo app.

Run locally with Streamlit, or deploy this file to Streamlit Community Cloud.
"""
from datetime import date, timedelta
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from forecasting import forecast_sales, parse_sales_csv, train_sales_model, fetch_weather

st.set_page_config(page_title="SafeBorrow", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');
:root { --ink:#f6dfd5; --muted:#caa99e; --coral:#ff693e; --line:#63352d; --panel:#492621; --panel-2:#542c26; }
.stApp { background:radial-gradient(ellipse at 50% -20%,#5a2a22 0,transparent 48%),#351a17; color:var(--ink); }
.block-container { max-width:1480px; padding:1.4rem 3rem 3.5rem; animation:fadeUp .75s ease-out both; }
html, body, [class*="css"] { font-family:'DM Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
h1,h2,h3 { color:var(--ink); }
h1 { font-family:'DM Serif Display',Georgia,serif; font-size:3.3rem !important; line-height:1.04 !important; font-weight:400 !important; letter-spacing:-.035em; }
h2 { font-family:'DM Sans',sans-serif; font-size:1.42rem !important; font-weight:600 !important; letter-spacing:-.025em; }
p, label, [data-testid="stCaptionContainer"] { color:var(--ink); }
[data-testid="stSidebar"] { background:#291311; border-right:1px solid #63352d; }
[data-testid="stSidebar"] * { color:#f6dfd5 !important; }
[data-testid="stSidebar"] h2 { font-family:'DM Sans',sans-serif; font-size:1.08rem !important; letter-spacing:-.01em; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#e7c5b9 !important; font-weight:600; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea { background:#492621 !important; color:#fff0e9 !important; border-color:#75443a !important; border-radius:11px !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background:#492621 !important; border-color:#75443a !important; border-radius:11px !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] { background:var(--coral) !important; border-color:var(--coral) !important; }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); border-radius:17px; padding:19px 21px; box-shadow:0 10px 30px rgba(0,0,0,.14); animation:fadeUp .65s ease-out both; transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease; }
[data-testid="stMetric"]:hover { transform:translateY(-4px); border-color:#b95235; box-shadow:0 15px 34px rgba(0,0,0,.22); }
[data-testid="stMetricLabel"] p { color:#d4b6ab; font-weight:500; }
[data-testid="stMetricValue"] { color:var(--coral); font-family:'DM Sans',sans-serif; font-weight:600; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px; border-color:var(--line); background:var(--panel); box-shadow:0 10px 30px rgba(0,0,0,.12); animation:fadeUp .8s ease-out both; }
[data-testid="stAlert"] { background:#492621; border:1px solid #75443a; border-radius:14px; color:#f6dfd5; }
[data-testid="stAlert"] p { color:#f6dfd5 !important; }
[data-testid="stDownloadButton"] button { background:#ff693e; color:#351a17; border:0; border-radius:999px; font-weight:700; }
[data-testid="stDownloadButton"] button:hover { background:#ff835e; color:#351a17; border:0; }
.topbar { display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #63352d; padding:.2rem 0 1.1rem; margin-bottom:2.8rem; color:#e9c5b9; }
.brand { color:var(--coral); font-size:1.55rem; font-weight:600; letter-spacing:-.07em; }
.brand-mark { color:#f6dfd5; margin-left:.18rem; }
.top-note { color:#c59b8f; text-transform:uppercase; letter-spacing:.16em; font-size:.66rem; font-weight:700; }
.eyebrow { color:var(--coral); font-size:.72rem; text-transform:uppercase; letter-spacing:.17em; font-weight:700; margin-bottom:.7rem; }
.hero-title { color:var(--ink); font-family:'DM Serif Display',Georgia,serif; font-size:clamp(3.2rem,6vw,5.5rem); line-height:.98; letter-spacing:-.045em; text-align:center; margin:.4rem auto .75rem; max-width:1000px; animation:fadeUp .85s .08s ease-out both; }
.hero-title em { color:var(--coral); font-weight:400; }
.subcopy { color:#d0b2a7; font-size:1rem; text-align:center; margin:.75rem 0 1.5rem; animation:fadeUp .85s .18s ease-out both; }
.risk-card { border:1px solid #814234; border-radius:18px; padding:20px 24px; margin:.65rem 0 1.1rem; box-shadow:0 10px 30px rgba(0,0,0,.14); animation:fadeUp .7s .12s ease-out both; }
.risk-label { font-size:.75rem; text-transform:uppercase; letter-spacing:.13em; font-weight:700; opacity:.86; }
.risk-number { font-family:'DM Sans',sans-serif; font-size:2.25rem; font-weight:700; line-height:1.2; margin:.25rem 0; letter-spacing:-.04em; }
.risk-note { font-size:.87rem; opacity:.86; }
.schedule-table { width:100%; border-collapse:collapse; color:#f2dcd2; font-size:.9rem; margin-top:.7rem; }
.schedule-table th { color:#d19b89; font-size:.73rem; text-transform:uppercase; letter-spacing:.08em; text-align:left; padding:.7rem .45rem; border-bottom:1px solid #75443a; }
.schedule-table td { padding:.8rem .45rem; border-bottom:1px solid #60362f; }
.schedule-table tr:last-child td { border-bottom:0; }
div[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; }
button[kind="secondary"] { border-radius:11px; }
@keyframes fadeUp { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration:.01ms !important; transition-duration:.01ms !important; } }
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

SAFE_RISK = 0.10


@st.cache_resource(show_spinner=False)
def cached_upload_model(csv_bytes):
    return train_sales_model(csv_bytes)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_weather(latitude, longitude):
    return fetch_weather(latitude, longitude, date.today())


GHANA_LOCATIONS = {
    "Accra": (5.6037, -0.1870), "Kumasi": (6.6885, -1.6244),
    "Tamale": (9.4034, -0.8424), "Takoradi": (4.8845, -1.7554),
    "Cape Coast": (5.1053, -1.2466), "Koforidua": (6.0941, -0.2591),
    "Ho": (6.6008, 0.4713), "Sunyani": (7.3349, -2.3123),
    "Bolgatanga": (10.7856, -0.8514),
}

CSV_TEMPLATE = "Date,Sales,Market_Day,Temperature_C,Feels_Like_C,Humidity_Pct,Rain_mm,Wind_Speed_kmh,Weather\n2026-01-01,420,0,29,32,75,0,12,2\n"


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
    st.header("Sales forecast")
    data_source = st.radio("Sales history", ["Example business", "Upload a CSV"])
    uploaded_file = None
    if data_source == "Upload a CSV":
        uploaded_file = st.file_uploader("Daily sales history", type=["csv"],
                                         help="Required: Date and Sales. Add optional market and weather columns for a more tailored forecast.")
        st.download_button("Download CSV template", CSV_TEMPLATE, "safeborrow_sales_template.csv", "text/csv")

    st.header("Your business")
    margin = st.slider("Profit margin (%)", 5, 60, 25) / 100
    daily_costs = st.number_input("Daily costs: rent, transport, family (GHS)", 0, 2000, 50, step=10)
    start_cash = st.number_input("Cash you have today (GHS)", 0, 20000, 200, step=50)

    st.header("The loan offer")
    amount = st.slider("Loan amount (GHS)", 50, 3000, 800, step=50)
    rate = st.slider("Interest per month (%)", 0.0, 30.0, 10.0, step=0.5) / 100
    term_weeks = st.slider("Loan length (weeks)", 1, 12, 4)
    today = date.today()
    loan_start = st.date_input("Loan starts on", value=today,
                               min_value=today, max_value=today + timedelta(days=9))
    schedule = st.radio("Repayment schedule", list(SCHEDULE_NAMES), format_func=SCHEDULE_NAMES.get)
    sales_uplift = st.slider("Sales increase from new equipment (%)", 0, 30, 10, step=5) / 100

    st.header("Local market context")
    location = st.selectbox("Business location", list(GHANA_LOCATIONS))
    use_market_days = st.checkbox("Include market days", value=False)
    market_interval = None
    next_market_date = None
    if use_market_days:
        market_interval = st.selectbox("Market day repeats every", [2, 3, 4, 5, 6, 7, 14], index=4,
                                       format_func=lambda days: f"{days} days")
        next_market_date = st.date_input("Next market day", value=loan_start,
                                         min_value=today, max_value=loan_start + timedelta(days=14))

if data_source == "Upload a CSV":
    if uploaded_file is None:
        st.info("Upload at least 56 days of daily sales. The template shows the required and optional columns.")
        st.stop()
    csv_bytes = uploaded_file.getvalue()
    try:
        with st.spinner("Learning your sales pattern from the uploaded history…"):
            bundle = cached_upload_model(csv_bytes)
        _, weather_fields, has_market_history = parse_sales_csv(csv_bytes)
    except Exception as exc:
        st.error(f"Could not use this sales file: {exc}")
        st.stop()
    business_label = uploaded_file.name
else:
    business_label = "Example bike-rental business"
    weather_fields = [name for name in bundle.get("feature_cols", [])
                      if name in ("temperature_c", "feels_like_c", "humidity_pct", "rain_mm", "wind_speed_kmh", "weather")]
    has_market_history = bundle.get("has_market_history", False)

weather_error = None
try:
    weather_forecast = cached_weather(*GHANA_LOCATIONS[location])
except Exception as exc:
    weather_forecast = None
    weather_error = str(exc)

try:
    dates, median, sims, future_features, weather_used = forecast_sales(
        bundle, loan_start, term_weeks, market_interval, next_market_date,
        weather=weather_forecast)
except Exception as exc:
    st.error(f"Could not create this forecast: {exc}")
    st.stop()

median *= 1 + sales_uplift
sims *= 1 + sales_uplift
plan, total = repayment_plan(amount, rate, term_weeks, schedule)
risk, cash = missed_payment_risk(sims, plan, margin, daily_costs, start_cash)

# ---------- Results ----------

st.markdown('<div class="topbar"><div class="brand">safeborrow<span class="brand-mark">{</span></div><div class="top-note">Ghana · Small business finance</div></div>', unsafe_allow_html=True)
st.markdown('<div class="eyebrow" style="text-align:center">Cash-flow clarity before you borrow</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Know what you can<br/>borrow <em>with confidence.</em></div>', unsafe_allow_html=True)
st.markdown(f'<div class="subcopy">{business_label} · See how a loan could fit before you accept the offer.</div>', unsafe_allow_html=True)

feature_cols = bundle.get("feature_cols") or list(bundle["models"][0.5].feature_name_)
if not bundle.get("uploaded") and not any(
        name in feature_cols for name in ("temperature_c", "feels_like_c", "humidity_pct", "weather")):
    st.warning("The saved example model predates the Ghana holiday and weather features. Rerun the updated training notebook once to train those effects into the example forecast.")
if weather_error:
    st.warning("Live weather is unavailable right now. The forecast continues without current weather data.")
elif weather_forecast is not None:
    first_week = weather_forecast.loc[
        (weather_forecast.index >= pd.Timestamp(loan_start)) &
        (weather_forecast.index < pd.Timestamp(loan_start) + pd.Timedelta(days=7))]
    if not first_week.empty:
        avg_temp = first_week["temperature_c"].mean()
        rainy_days = int((first_week["rain_mm"] >= 1).sum())
        st.info(f"{location} weather for the first loan week: about {avg_temp:.0f}°C on average, with rain expected on {rainy_days} of 7 days. "
                + ("Weather is included in the sales forecast." if weather_used else "Weather is shown for context; this model has no matching historical weather data to learn its sales impact."))
        st.caption("Weather forecast: Open-Meteo. Future days beyond the forecast window use the model's typical conditions.")
        if not bundle.get("uploaded") and weather_used:
            st.caption("The example weather response was learned from historical bike-rental data; your own sales and weather history will provide a better business-specific estimate.")

if use_market_days and not has_market_history:
    st.caption("Market dates are marked in the forecast, but the uploaded history does not label past market days, so the model cannot learn their sales effect yet. Add a MarketDay column (1/0) to your CSV to train that relationship.")

if risk <= SAFE_RISK:
    risk_colour, verdict, risk_bg = "#ff9672", "Comfortable buffer", "#4a2923"
elif risk <= 0.25:
    risk_colour, verdict, risk_bg = "#ffc174", "Worth a closer look", "#493322"
else:
    risk_colour, verdict, risk_bg = "#ff8876", "High chance of strain", "#4c2521"

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
        fig.patch.set_facecolor("#492621")
        ax.set_facecolor("#492621")
        ax.fill_between(dates, low, high, color="#ff693e", alpha=.24, label="middle 80% of outcomes")
        ax.plot(dates, typical, color="#f6dfd5", linewidth=2.8, label="typical outcome")
        ax.axhline(0, color="#ff693e", linewidth=1.5, linestyle=(0, (4, 4)), label="zero cash")
        ax.set_ylabel("Cash balance (GHS)", color="#d4b6ab", labelpad=10)
        ax.grid(axis="y", color="#63352d", linewidth=.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#75443a")
        ax.tick_params(colors="#d4b6ab", length=0, pad=8)
        ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=8, labelcolor="#e9c5b9")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

rows = []
for sch in SCHEDULE_NAMES:
    p, _ = repayment_plan(amount, rate, term_weeks, sch)
    r, _ = missed_payment_risk(sims, p, margin, daily_costs, start_cash)
    rows.append((SCHEDULE_NAMES[sch], f"{r:.0%}",
                 f"GHS {safe_amount(sims, rate, term_weeks, sch, margin, daily_costs, start_cash):,}"))
table_html = "<table class='schedule-table'><thead><tr><th>Repayment timing</th><th>Miss risk</th><th>Safe limit</th></tr></thead><tbody>"
table_html += "".join(f"<tr><td>{timing}</td><td>{miss_risk}</td><td>{safe_limit}</td></tr>"
                      for timing, miss_risk, safe_limit in rows)
table_html += "</tbody></table>"
with info_col:
    with st.container(border=True):
        st.subheader("Compare repayment timing")
        st.caption("The same loan can feel very different depending on when payments are due.")
        st.markdown(table_html, unsafe_allow_html=True)
        st.caption("Safe limit keeps estimated risk at or below 10%.")

st.markdown("---")
st.caption("Ghana public holidays are flagged in the calendar. The selected equipment uplift is an assumption applied to forecast sales. Actual demand and loan outcomes may differ. Assumes the loan funds are spent immediately.")
