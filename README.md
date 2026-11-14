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

From the repository root, run `streamlit run safeborrow/data/app.py`. If the saved model is missing, run the training notebook first so it creates `safeborrow/data/safeborrow_model.joblib`.
