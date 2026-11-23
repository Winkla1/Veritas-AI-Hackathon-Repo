# SafeBorrow

### A cash-flow planning prototype for small businesses considering a loan

**Track:** AI + Finance

**Team:** Pierre Awoniyi, Joshua Areh, and Michelle Okyere

SafeBorrow helps business owners explore whether a proposed loan could fit their cash flow. It forecasts sales, models repayments against business costs and available cash, and lets users compare different loan scenarios. It is a planning prototype, not a credit-scoring or loan-approval system.

## The problem

Small businesses may borrow to fund equipment, inventory, or growth without knowing how repayments will line up with variable income and operating costs. SafeBorrow makes those trade-offs easier to see before a business accepts an offer.

## What the app does

- Forecasts daily sales at low, typical, and high levels using LightGBM quantile regression.
- Simulates 1,000 possible sales paths by resampling consecutive seven-day blocks of historical forecast errors.
- Estimates the chance that simulated cash falls below zero on a repayment date, based on the user’s assumptions.
- Shows total repayment, estimated interest and fees, cash-flow paths, and a suggested loan limit at a 10% simulated-risk threshold.
- Compares daily, weekly, and end-of-term repayment schedules, with an option to compare a second loan offer.
- Lets users upload a sales CSV and reports invalid rows, duplicate dates, missing dates, and unusually high or low daily sales.
- Includes Ghana public holidays, repeating or manually entered market days, and a local weather forecast for the first loan week.
- Lets users explore a sales downside scenario, a suggested cash buffer, and an optional sales uplift to represent possible new-equipment effects.

The model can only learn a market-day or weather effect when the uploaded sales history includes matching market-day labels or weather fields. The equipment uplift is a user-selected scenario assumption; it is not a measured causal effect.

## How it works

### Example training data

The notebook retrieves the [Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike-sharing-dataset) through scikit-learn’s [`fetch_openml`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html). The source contains hourly and daily Capital Bikeshare rental counts from Washington, DC, for 2011–2012, with weather and calendar information. SafeBorrow aggregates hourly rentals into 731 daily totals and treats those counts as an income proxy for the demonstration. They are **not actual Ghanaian business sales, GHS revenue, or loan records**. Dates are rebuilt from the row order and checked against the weekday information.

### Fictional upload example

[`Sample CSV file.csv`](Sample%20CSV%20file.csv) is a separate, fictional 56-week business dataset for demonstrating the upload flow. It contains 392 daily records from January 2025 to January 2026, with sales, market-day flags, weather fields, and a constant 45% profit-margin column. Its sales average about GHS 5,080 per day. The file is hypothetical demo data—not observed business activity. The app does not read its `Profit_Margin_Pct` column, so the user must enter the margin in the sidebar. Daily costs and cash on hand are also user-entered assumptions.

### Forecasts and affordability scenarios

The model forecasts sales; it does not learn whether a borrower repaid or defaulted. For affordability scenarios, the app applies the user-entered profit margin, daily costs, starting cash, loan amount, rate, fee, term, and repayment schedule to the simulated sales paths. A path is counted as having a missed payment when its cash balance is below zero on a scheduled payment date. Interest is estimated as simple interest on the original principal, and any one-time fee is added to total repayment and spread over the selected payment dates.

## Model evaluation and limitations

In the notebook’s recorded holdout evaluation, the example model had a mean absolute error (MAE) of **GHS 81**, compared with **GHS 206** for a weekday-average baseline. Its low-to-high forecast interval covered **51%** of held-out days, below the intended 80%. The model beat the baseline on average in that evaluation, but its uncertainty range was too narrow. These figures measure sales forecasts on the bike-rental example; they do **not** measure missed-payment prediction accuracy.

The example is based on an older US bike-rental dataset, not representative Ghanaian small-business data. The fictional CSV demonstrates the upload feature but does not validate the model. Weather and market-day patterns may not transfer to real businesses. The cash-flow and missed-payment results are scenario estimates driven by user inputs, not validated credit decisions or guarantees. Real Ghanaian business sales and repayment data would be needed to assess performance for the intended users.

## Run locally

Python 3.9 or later is recommended.

1. Clone or download this repository and open a terminal in its root folder.
2. Create and activate a virtual environment if desired.
3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Launch the app:

   ```bash
   streamlit run safeborrow/data/app.py
   ```

The app uses the included `safeborrow/data/safeborrow_model.joblib` for the example business. To retrain it, run [`safeborrow/data/safeborrow.ipynb`](safeborrow/data/safeborrow.ipynb) from top to bottom. The first notebook run downloads the OpenML example data. On macOS, LightGBM may also need OpenMP: `brew install libomp`.

## Demo scenario

For the fictional CSV, the sample business assumes a 45% profit margin. A demo can pair that with GHS 2,000 in daily costs and GHS 12,000 in starting cash, then explore a GHS 1,400 loan at 3.5% monthly interest over two weeks with daily repayments. The costs, cash, and loan details are illustrative inputs, not values contained in the sales file.

## Sources and acknowledgements

- Example training data: [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike-sharing-dataset), retrieved through scikit-learn [`fetch_openml`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html).
- Weather forecasts: [Open-Meteo Forecast API](https://open-meteo.com/en/docs).
- Ghana public holidays: [Python Holidays](https://github.com/vacanza/holidays).
- Forecasting library: [LightGBM](https://lightgbm.readthedocs.io/). App framework: [Streamlit](https://docs.streamlit.io/).
- AI assistance: Claude and OpenAI Codex supported brainstorming, debugging, feature development, interface improvements, and documentation.

## What we would do next

- Evaluate and calibrate the forecast on multiple time periods and real, consented Ghanaian small-business data.
- Build a repayment calendar with low-cash warnings and a shareable loan-readiness report.
- Add lender-offer matching using reliable, current offer data, while presenting matches as comparisons rather than guaranteed approvals.
