# SafeBorrow — Finance Track

**A cash-flow planning prototype for small businesses considering a loan.** SafeBorrow compares an offer with expected sales, operating costs, available cash and repayment timing, helping owners see how a loan could strain cash during a slow week.

## How it works

Use the example or upload at least 56 daily observations (`Date`, `Sales`). Three LightGBM quantile models forecast log sales; the app resamples consecutive seven-day forecast errors into 1,000 sales paths, then tracks cash after costs and repayments. It shows missed-payment risk, total repayment, a suggested loan limit at a 10% simulated-risk threshold, and comparisons of daily, weekly and end-of-term payments.

The app adds Ghana public holidays, optional repeating market days, a local 16-day weather forecast and a user-set equipment sales uplift. Market-day impact requires historic labels; weather impact requires historic weather data. The example is Washington, DC bike-rental demand scaled to GHS—not Ghanaian shop sales—and the uplift is an assumption, not a measured causal effect.

## What the current evaluation says

On the final 12 weeks held out from training, the updated example reported **GHS 81 MAE**, versus **GHS 206** for a weekday-average baseline. Its 10th–90th percentile interval covered **51%** of held-out days, below the 80% target. That under-calibration means the simulated risk is not validated for credit underwriting.

**SafeBorrow is a finance prototype, not financial advice, a credit decision or a repayment guarantee.** Its sample data is old and from another country and business type; real sales, market days, weather effects, fees and household cash flows may differ.

## Run it

Install with `python -m pip install -r requirements.txt`, then run `streamlit run safeborrow/data/app.py` from the repository root. On this Mac, double-click `SafeBorrow.app` to open the demo without a terminal. macOS also needs `brew install libomp`. To retrain, run `safeborrow/data/safeborrow.ipynb`; the app includes a CSV template. Demo with public or synthetic data, not personal or confidential records.

## Credits

- Example demand data: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike-sharing-dataset), retrieved as `Bike_Sharing_Demand` with scikit-learn’s [`fetch_openml`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html).
- Weather: [Open-Meteo Forecast API](https://open-meteo.com/en/docs).
- Ghana holiday dates: [Python Holidays library](https://github.com/vacanza/holidays).
- Model and interface libraries: [LightGBM](https://lightgbm.readthedocs.io/) and [Streamlit](https://docs.streamlit.io/).
- AI coding assistance: OpenAI Codex (debugging, code editing, and documentation support). The team is responsible for understanding and explaining the submitted code.

## Demo

1. Set business inputs and a loan offer; show risk, safe limit and cash-flow paths.
2. Compare repayment timings, then upload synthetic sales and adjust the equipment uplift.

For the optional video, keep it under two minutes and make the link viewable without signing in. The submission form separately requires the Finance track, team details and credits.
