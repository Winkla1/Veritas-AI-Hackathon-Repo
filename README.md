# SafeBorrow — Finance Track

**A cash-flow planning prototype for small businesses considering a loan.** A loan can look affordable on paper and still leave a shop short of cash during a slow week. SafeBorrow helps an owner explore how expected sales, operating costs, available cash and repayment timing interact before accepting an offer.

## How it works

Use the example business or upload at least 56 daily observations (`Date`, `Sales`). Three LightGBM quantile models forecast low, typical and high sales. The app resamples consecutive seven-day forecast errors into 1,000 possible sales paths, then tracks cash after costs and loan repayments. It reports estimated missed-payment risk, total repayment, a suggested loan limit at a 10% simulated-risk threshold, and a cash-flow chart. Owners can compare daily, weekly and end-of-term payments.

The app adds Ghana public holidays, repeating or manually entered market dates, a local 16-day weather forecast and a user-set equipment sales uplift. Uploads receive quality checks for invalid values, repeated dates, missing dates and unusual sales days. The dashboard shows holdout accuracy, loan interest and fees, scheduled payment dates, a sales downside scenario, a suggested cash buffer, and an optional side-by-side offer comparison with plain-language risk drivers. Market-day impact requires historic labels; weather impact requires matching historical weather fields. The example is Washington, DC bike-rental demand from 2011–2012, scaled to GHS—not Ghanaian shop sales. The equipment uplift is a scenario assumption, not a measured causal effect.

## Evaluation and limitations

On the final 12 weeks held out from training, the updated example reported **GHS 81 MAE**, versus **GHS 206** for a weekday-average baseline. Its 10th–90th percentile interval covered **51%** of held-out days, below the 80% target. That under-calibration means the simulated risk is not validated for credit underwriting. The weather response learned from bike rentals may not transfer to Ghanaian businesses; real sales, local market patterns, loan fees and household cash flows can differ.

**SafeBorrow is a finance prototype, not financial advice, a credit decision or a repayment guarantee.** Its potential impact is to make repayment trade-offs easier to examine; no real-world impact has yet been measured.

## Run it

Install with `python -m pip install -r requirements.txt`, then run `streamlit run safeborrow/data/app.py` from the repository root. macOS also needs `brew install libomp`. To retrain the example, run `safeborrow/data/safeborrow.ipynb`; the app includes a CSV template. Demo with public or synthetic data, not personal or confidential records.

## Credits

- Example data: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike-sharing-dataset), retrieved as `Bike_Sharing_Demand` through scikit-learn’s [`fetch_openml`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html).
- Weather: [Open-Meteo Forecast API](https://open-meteo.com/en/docs). Ghana holidays: [Python Holidays](https://github.com/vacanza/holidays).
- Model and interface: [LightGBM](https://lightgbm.readthedocs.io/) and [Streamlit](https://docs.streamlit.io/).
- AI coding assistance: OpenAI Codex (code editing, debugging and documentation).

## Demo flow

Set business inputs and a loan offer; show risk, safe limit and cash-flow paths. Compare repayment timings, upload synthetic sales, then adjust the equipment uplift and discuss its effect on risk.
