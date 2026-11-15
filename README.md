# SafeBorrow

SafeBorrow is a Streamlit demo that estimates whether a small business can afford a loan. It compares repayment schedules against simulated daily cash flows and shows a missed-payment risk estimate.

## Live demo without a terminal

Deploy the app to [Streamlit Community Cloud](https://share.streamlit.io/) and use the resulting web link during the presentation. The app and saved model must both be committed to the GitHub repository.

1. Push this repository to GitHub, including `safeborrow/data/app.py`, `safeborrow/data/safeborrow_model.joblib`, and `requirements.txt`.
2. Sign in to Streamlit Community Cloud with GitHub and choose **Create app**.
3. Select this repository and its demo branch, and set the app file path to `safeborrow/data/app.py`.
4. Deploy, open the resulting `*.streamlit.app` link, and keep that page ready for the demo.

The app loads its model file next to `app.py`, so it works whether Streamlit starts from the repository root or the app's directory. Community Cloud installs the packages listed in `requirements.txt`.

## Run locally during development

On this Mac, double-click `SafeBorrow.app` in the repository folder. It starts Streamlit on port 8501 and opens the app in your browser; no terminal window is needed. When you're done, double-click `Stop SafeBorrow.app` to stop the server.

From the repository root, run `streamlit run safeborrow/data/app.py` to start it manually. If the saved model is missing, run the training notebook first so it creates `safeborrow/data/safeborrow_model.joblib`.

For notebook development, install the packages in `requirements.txt` into `.venv`. On macOS, LightGBM also needs the OpenMP runtime (`brew install libomp`).

## Use your own business data

In the app sidebar, choose **Upload a CSV** and provide at least 56 days of daily history. `Date` and `Sales` are required. Optional columns are `Market_Day` (1/0), `Temperature_C`, `Feels_Like_C`, `Humidity_Pct`, `Rain_mm`, `Wind_Speed_kmh`, and numeric `Weather` categories (1 clear, 2 cloudy, 3 rain, 4 storm). Ghana public holidays are added automatically. For the model to learn a market-day sales effect, include past market days in `Market_Day`; otherwise the app only marks future market dates.

The app requests a live 16-day forecast from [Open-Meteo](https://open-meteo.com/en/docs) for the selected Ghana location. Its forecast influences the sales estimate only when the saved model or uploaded CSV has matching historical weather fields. For the updated example model, rerun all cells in `safeborrow/data/safeborrow.ipynb` once to regenerate `safeborrow_model.joblib` with Ghana holiday and weather features.
