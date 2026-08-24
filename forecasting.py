import logging
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

    fit = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    result = fit.get_forecast(steps=steps)
    pred = result.predicted_mean
    conf = result.conf_int(alpha=0.05)

    lower = conf.iloc[:, 0]
    upper = conf.iloc[:, 1]

    pred[pred < 0] = 0
    lower[lower < 0] = 0
    upper[upper < 0] = 0

    return pred, lower, upper


def run_prophet(train, steps):

    dfp = train.reset_index()
    dfp.columns = ["ds", "y"]

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=steps, freq="MS")
    fc = m.predict(future)
    fc = fc.set_index("ds").iloc[-steps:]

    pred = fc["yhat"]
    lower = fc["yhat_lower"]
    upper = fc["yhat_upper"]

    pred[pred < 0] = 0
    lower[lower < 0] = 0
    upper[upper < 0] = 0

    return pred, lower, upper


def best_forecast(series, horizon):
    seasonal_ok = len(series) >= 24 and series.tail(24).sum() > 0

    if len(series) < 8:
        pred = run_sarima(series, horizon, False)
        return pred, "SARIMA", {}

    test_size = min(12, max(3, len(series) // 5))
    train, test = series.iloc[:-test_size], series.iloc[-test_size:]

    results = {}
    try:
        pred, _, _ = run_sarima(train, test_size, seasonal_ok)
        results["SARIMA"] = compute_metrics(test, pred)
    except Exception:
        results["SARIMA"] = None
    try:
        pred, _, _ = run_prophet(train, test_size)
        results["Prophet"] = compute_metrics(test, pred)
    except Exception:
        results["Prophet"] = None

    valid = {k: v for k, v in results.items() if v and not np.isnan(v["WAPE"])}
    best_name = min(valid, key=lambda k: valid[k]["WAPE"]) if valid else "SARIMA"

    if best_name == "Prophet":
        final, lower, upper = run_prophet(series, horizon)
    else:
        final, lower, upper = run_sarima(series, horizon, seasonal_ok)

    return final, lower, upper, best_name, results


@st.cache_data(show_spinner="Fitting SARIMA and Prophet...")
def get_forecast(df, insurer, category, horizon):
    data = df.copy()
    if insurer != "All":
        data = data[data["Insurer"] == insurer]
    if category != "All":
        data = data[data["Category"] == category]

    monthly = data.groupby("Date")[["Total_Premium", "Total_Policies"]].sum()
    monthly = monthly.sort_index().asfreq("MS").fillna(0)

    p_pred, p_lower, p_upper, p_best, p_metrics = best_forecast(monthly["Total_Premium"], horizon)
    q_pred, q_lower, q_upper, q_best, q_metrics = best_forecast(monthly["Total_Policies"], horizon)

    return (monthly, p_pred, p_lower, p_upper, p_best, p_metrics, q_pred, q_lower, q_upper, q_best, q_metrics)
