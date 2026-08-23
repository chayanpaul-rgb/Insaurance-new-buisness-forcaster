import logging
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").disabled = True


@st.cache_data
def load_data():
    df = pd.read_csv("master_sales_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={
        "Company": "Insurer",
        "Product_Category": "Category",
        "Premium_Current_Month": "Total_Premium",
        "Policies_Current_Month": "Total_Policies",
    })
    df = df[df["Category"] != "Total"]  # this row is just a rollup of the other categories
    df = df.dropna(subset=["Total_Premium", "Total_Policies"])
    return df


def compute_metrics(actual, pred):
    actual = np.array(actual)
    pred = np.array(pred)
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    denom = np.sum(np.abs(actual))
    wape = np.sum(np.abs(actual - pred)) / denom * 100 if denom else np.nan
    bias = np.sum(pred - actual) / denom * 100 if denom else np.nan
    return {"MAE": mae, "RMSE": rmse, "WAPE": wape, "Bias": bias}


def run_sarima(train, steps, seasonal):
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12) if seasonal else (0, 0, 0, 0)
    fit = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    pred = fit.forecast(steps)
    pred[pred < 0] = 0
    return pred


def run_prophet(train, steps):
    dfp = train.reset_index()
    dfp.columns = ["ds", "y"]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=steps, freq="MS")
    fc = m.predict(future)
    pred = fc.set_index("ds")["yhat"].iloc[-steps:]
    pred[pred < 0] = 0
    return pred


def make_lag_features(series):
    data = series.to_frame(name="y")
    data["month"] = data.index.month
    for lag in [1, 2, 3, 12]:
        data[f"lag_{lag}"] = data["y"].shift(lag)
    return data.dropna()


def run_xgboost(train, steps):
    feature_cols = ["month", "lag_1", "lag_2", "lag_3", "lag_12"]
    feat = make_lag_features(train)
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, verbosity=0)
    model.fit(feat[feature_cols], feat["y"])

    history = train.copy()
    preds = []
    for _ in range(steps):
        next_date = history.index[-1] + pd.DateOffset(months=1)
        row = pd.DataFrame([{
            "month": next_date.month,
            "lag_1": history.iloc[-1],
            "lag_2": history.iloc[-2],
            "lag_3": history.iloc[-3],
            "lag_12": history.iloc[-12],
        }])[feature_cols]
        pred_val = model.predict(row)[0]
        preds.append(pred_val)
        history.loc[next_date] = pred_val

    pred = pd.Series(preds, index=history.index[-steps:])
    pred[pred < 0] = 0
    return pred


def best_forecast(series, horizon):
    seasonal_ok = len(series) >= 24 and series.tail(24).sum() > 0

    if len(series) < 8:
        pred = run_sarima(series, horizon, False)
        return pred, "SARIMA", {}

    test_size = min(12, max(3, len(series) // 5))
    train, test = series.iloc[:-test_size], series.iloc[-test_size:]

    results = {}
    try:
        pred = run_sarima(train, test_size, seasonal_ok)
        results["SARIMA"] = compute_metrics(test, pred)
    except Exception:
        results["SARIMA"] = None
    try:
        pred = run_prophet(train, test_size)
        results["Prophet"] = compute_metrics(test, pred)
    except Exception:
        results["Prophet"] = None
    try:
        pred = run_xgboost(train, test_size)
        results["XGBoost"] = compute_metrics(test, pred)
    except Exception:
        results["XGBoost"] = None

    valid = {k: v for k, v in results.items() if v and not np.isnan(v["WAPE"])}
    best_name = min(valid, key=lambda k: valid[k]["WAPE"]) if valid else "SARIMA"

    if best_name == "Prophet":
        final = run_prophet(series, horizon)
    elif best_name == "XGBoost":
        final = run_xgboost(series, horizon)
    else:
        final = run_sarima(series, horizon, seasonal_ok)

    return final, best_name, results


@st.cache_data(show_spinner="Fitting SARIMA, Prophet and XGBoost...")
def get_forecast(df, insurer, category, horizon):
    data = df.copy()
    if insurer != "All":
        data = data[data["Insurer"] == insurer]
    if category != "All":
        data = data[data["Category"] == category]

    monthly = data.groupby("Date")[["Total_Premium", "Total_Policies"]].sum()
    monthly = monthly.sort_index().asfreq("MS").fillna(0)

    p_pred, p_best, p_metrics = best_forecast(monthly["Total_Premium"], horizon)
    q_pred, q_best, q_metrics = best_forecast(monthly["Total_Policies"], horizon)

    return monthly, p_pred, p_best, p_metrics, q_pred, q_best, q_metrics