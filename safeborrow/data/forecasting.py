"""Data preparation and forecasts used by the SafeBorrow demo."""
import io

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests


BASE_FEATURES = ["dayofweek", "day", "month", "weekofyear", "holiday",
                 "workingday", "market_day", "year"]
WEATHER_FEATURES = ["temperature_c", "feels_like_c", "humidity_pct",
                    "rain_mm", "wind_speed_kmh", "weather"]
WEATHER_LABELS = {
    "temperature_c": "temperature_c", "temperature_2m_mean": "temperature_c",
    "temp_c": "temperature_c", "feels_like_c": "feels_like_c",
    "apparent_temperature_mean": "feels_like_c", "humidity_pct": "humidity_pct",
    "relative_humidity_2m_mean": "humidity_pct", "rain_mm": "rain_mm",
    "precipitation_sum": "rain_mm", "wind_speed_kmh": "wind_speed_kmh",
    "wind_speed_10m_mean": "wind_speed_kmh", "weather": "weather",
}


def _truthy(values):
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0).ne(0)
    return values.astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "holiday", "market"})


def parse_sales_csv(content):
    frame = pd.read_csv(io.BytesIO(content))
    if frame.empty:
        raise ValueError("The CSV has no sales rows.")
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    aliases = {
        "date": ("date", "day", "sales_date", "transaction_date"),
        "sales": ("sales", "revenue", "income", "daily_sales", "amount"),
    }
    rename = {}
    for canonical, options in aliases.items():
        found = next((name for name in options if name in frame.columns), None)
        if found is None:
            raise ValueError("CSV needs a Date column and a Sales (or Revenue/Income) column.")
        rename[found] = canonical
    for name, canonical in WEATHER_LABELS.items():
        if name in frame.columns:
            rename[name] = canonical
    if "holiday" in frame.columns:
        rename["holiday"] = "holiday"
    market_flag = next((name for name in ("market_day", "marketday", "market_day_flag")
                        if name in frame.columns), None)
    if market_flag is not None:
        rename[market_flag] = "market_day"
    frame = frame.rename(columns=rename)
    frame["Date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["Sales"] = pd.to_numeric(frame["sales"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    invalid_date_rows = int(frame["Date"].isna().sum())
    invalid_sales_rows = int(frame["Sales"].isna().sum())
    negative_sales_rows = int((frame["Sales"] < 0).sum())
    frame = frame.dropna(subset=["Date", "Sales"])
    frame = frame[frame["Sales"] >= 0]
    if frame.empty:
        raise ValueError("No valid, non-negative sales values and dates were found.")

    duplicate_date_rows = int(frame.duplicated(subset=["Date"], keep=False).sum())
    aggregations = {"Sales": ("Sales", "sum")}
    for field in ("holiday", "market_day"):
        if field in frame:
            frame[field] = _truthy(frame[field]).astype(int)
            aggregations[field] = (field, "max")
    weather_fields = []
    for field in WEATHER_FEATURES:
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
            if frame[field].notna().sum() >= max(14, len(frame) * 0.5):
                aggregations[field] = (field, "mean")
                weather_fields.append(field)
    daily = frame.groupby("Date").agg(**aggregations).sort_index()
    if len(daily) < 56:
        raise ValueError(f"Found {len(daily)} distinct sales days. Upload at least 56 days; 6+ months gives a more useful forecast.")
    # A missing date in a daily sales export is treated as a no-sales day.
    date_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    missing_dates = int(len(date_range) - len(daily))
    daily = daily.reindex(date_range)
    daily.index.name = "Date"
    daily["Sales"] = daily["Sales"].fillna(0)
    for field in ("holiday", "market_day"):
        if field not in daily:
            daily[field] = 0
        daily[field] = daily[field].fillna(0).astype(int)
    for field in weather_fields:
        daily[field] = daily[field].interpolate(limit_direction="both")
    daily = daily.reset_index()
    q01, q99 = daily["Sales"].quantile([0.01, 0.99])
    quality = {
        "rows": len(frame), "invalid_date_rows": invalid_date_rows,
        "invalid_sales_rows": invalid_sales_rows,
        "negative_sales_rows": negative_sales_rows,
        "duplicate_date_rows": duplicate_date_rows,
        "missing_dates_filled_zero": missing_dates,
        "extreme_sales_days": int(((daily["Sales"] < q01) | (daily["Sales"] > q99)).sum()),
    }
    return daily, weather_fields, "market_day" in frame.columns, quality


def _holiday_set(dates):
    years = sorted(set(pd.DatetimeIndex(dates).year.tolist()))
    return holidays.country_holidays("GH", years=years)


def make_calendar_frame(dates, start, market_dates=None, source_holidays=None):
    dates = pd.DatetimeIndex(dates)
    gh_holidays = _holiday_set(dates)
    frame = pd.DataFrame({"Date": dates})
    frame["holiday"] = [int(d.date() in gh_holidays) for d in dates]
    if source_holidays is not None:
        frame["holiday"] = np.maximum(frame["holiday"], np.asarray(source_holidays, dtype=int))
    frame["workingday"] = ((dates.dayofweek < 5) & (frame["holiday"].to_numpy() == 0)).astype(int)
    frame["market_day"] = 0 if market_dates is None else dates.isin(pd.DatetimeIndex(market_dates)).astype(int)
    frame["year"] = dates.year - pd.Timestamp(start).year
    return frame


def make_features(frame, start, feature_cols):
    dates = pd.DatetimeIndex(frame["Date"])
    features = pd.DataFrame(index=frame.index)
    features["dayofweek"] = dates.dayofweek
    features["day"] = dates.day
    features["month"] = dates.month
    features["weekofyear"] = dates.isocalendar().week.astype(int).to_numpy()
    for name in ("holiday", "workingday", "market_day"):
        features[name] = frame[name].astype(int).to_numpy()
    features["year"] = dates.year - pd.Timestamp(start).year
    for name in WEATHER_FEATURES:
        if name in feature_cols:
            features[name] = pd.to_numeric(frame[name], errors="coerce").to_numpy()
    return features.reindex(columns=feature_cols)


def train_sales_model(content):
    daily, weather_fields, has_market_history, quality = parse_sales_csv(content)
    feature_cols = BASE_FEATURES + weather_fields
    start = daily["Date"].min()
    gh_holidays = holidays.country_holidays(
        "GH", years=range(daily["Date"].dt.year.min(), daily["Date"].dt.year.max() + 1))
    daily["holiday"] = np.maximum(
        daily["holiday"].to_numpy(),
        np.array([int(d.date() in gh_holidays) for d in daily["Date"]]))
    daily["workingday"] = ((daily["Date"].dt.dayofweek < 5) & (daily["holiday"] == 0)).astype(int)
    medians = {}
    for field in weather_fields:
        medians[field] = float(daily[field].median())
        daily[field] = daily[field].fillna(medians[field])
    holdout_days = max(14, min(56, int(len(daily) * 0.2)))
    split = len(daily) - holdout_days
    train, valid = daily.iloc[:split], daily.iloc[split:]
    X_train = make_features(train, start, feature_cols)
    X_valid = make_features(valid, start, feature_cols)
    models = {}
    for quantile in (0.1, 0.5, 0.9):
        model = lgb.LGBMRegressor(objective="quantile", alpha=quantile,
                                  n_estimators=220, learning_rate=0.04,
                                  num_leaves=9, min_child_samples=8, verbose=-1)
        model.fit(X_train, np.log1p(train["Sales"].to_numpy()))
        models[quantile] = model
    predictions = {q: np.maximum(np.expm1(model.predict(X_valid)), 0)
                   for q, model in models.items()}
    median_valid = np.maximum(predictions[0.5], 1.0)
    actual = valid["Sales"].to_numpy()
    recent = train[train["Date"] > valid["Date"].min() - pd.Timedelta(days=28)]
    dow_avg = recent.groupby(recent["Date"].dt.dayofweek)["Sales"].mean()
    baseline = valid["Date"].dt.dayofweek.map(dow_avg).fillna(train["Sales"].mean()).to_numpy()
    evaluation = {
        "mae": float(np.mean(np.abs(actual - median_valid))),
        "baseline_mae": float(np.mean(np.abs(actual - baseline))),
        "coverage": float(np.mean((actual >= predictions[0.1]) & (actual <= predictions[0.9]))),
        "holdout_days": len(valid),
    }
    ratios = np.clip(valid["Sales"].to_numpy() / median_valid, 0.05, 5.0)
    block_size = 7
    blocks = np.array([ratios[i:i + block_size]
                       for i in range(len(ratios) - block_size + 1)])
    if len(blocks) == 0:
        blocks = np.ones((1, block_size))
    # Score on held-out days first, then refit the serving models on all history.
    X_all = make_features(daily, start, feature_cols)
    for quantile in (0.1, 0.5, 0.9):
        model = lgb.LGBMRegressor(objective="quantile", alpha=quantile,
                                  n_estimators=220, learning_rate=0.04,
                                  num_leaves=9, min_child_samples=8, verbose=-1)
        model.fit(X_all, np.log1p(daily["Sales"].to_numpy()))
        models[quantile] = model
    return {
        "models": models, "blocks": blocks, "block_size": block_size,
        "start": start, "forecast_year": int(daily["Date"].max().year + 1),
        "feature_cols": feature_cols, "weather_medians": medians,
        "has_market_history": has_market_history, "uploaded": True,
        "history_days": len(daily), "evaluation": evaluation,
        "csv_quality": quality,
    }


def fetch_weather(latitude, longitude, start_date):
    """Get 16-day Open-Meteo data; the first loan week is forecast-only."""
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude, "longitude": longitude,
            "daily": "temperature_2m_mean,apparent_temperature_mean,relative_humidity_2m_mean,precipitation_sum,wind_speed_10m_mean,weather_code",
            "forecast_days": 16, "timezone": "Africa/Accra",
        }, timeout=12)
    response.raise_for_status()
    data = response.json()["daily"]
    weather = pd.DataFrame(data)
    weather["Date"] = pd.to_datetime(weather["time"])
    weather = weather.rename(columns={
        "temperature_2m_mean": "temperature_c",
        "apparent_temperature_mean": "feels_like_c",
        "relative_humidity_2m_mean": "humidity_pct",
        "precipitation_sum": "rain_mm",
        "wind_speed_10m_mean": "wind_speed_kmh",
    })
    weather["weather"] = weather["weather_code"].map(_weather_category)
    return weather.drop(columns=["time", "weather_code"]).set_index("Date")


def _weather_category(code):
    if pd.isna(code):
        return np.nan
    code = int(code)
    if code <= 1:
        return 1
    if code <= 3 or code in (45, 48):
        return 2
    if code < 95:
        return 3
    return 4


def forecast_sales(bundle, start_date, term_weeks, market_interval,
                   next_market_date, weather=None, n_sims=1000, seed=0,
                   explicit_market_dates=None):
    start_date = pd.Timestamp(start_date)
    dates = pd.date_range(start_date, periods=term_weeks * 7, freq="D")
    market_dates = None
    if explicit_market_dates:
        market_dates = pd.to_datetime(explicit_market_dates).tolist()
    elif market_interval and next_market_date:
        anchor = pd.Timestamp(next_market_date)
        market_dates = [d for d in dates if (d - anchor).days % market_interval == 0]
    fut = make_calendar_frame(dates, bundle["start"], market_dates)
    feature_cols = bundle.get("feature_cols") or list(bundle["models"][0.5].feature_name_)
    weather_features = [name for name in WEATHER_FEATURES if name in feature_cols]
    medians = bundle.get("weather_medians", {})
    for field in weather_features:
        fut[field] = medians.get(field, 0.0)
    if weather is not None and weather_features:
        for field in weather_features:
            if field in weather.columns:
                available = weather[field].dropna()
                if not available.empty:
                    fut[field] = fut["Date"].map(available).fillna(fut[field])
    X = make_features(fut, bundle["start"], feature_cols)
    median = np.maximum(np.expm1(bundle["models"][0.5].predict(X)), 0)
    blocks = np.asarray(bundle["blocks"])
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(median) / bundle["block_size"]))
    picks = rng.integers(0, len(blocks), size=(n_sims, n_blocks))
    ratios = blocks[picks].reshape(n_sims, -1)[:, :len(median)]
    return dates, median, median * ratios, fut, bool(weather_features and weather is not None)
