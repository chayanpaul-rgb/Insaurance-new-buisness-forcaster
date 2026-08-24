import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forecasting as fc

st.set_page_config(page_title="Life Insurance Sales Forecast", layout="wide")

df = fc.load_data()

st.title("Life Insurance New Business - Sales Analysis and Forecasting")

insurers = ["All"] + sorted(df["Insurer"].unique().tolist())
categories = ["All"] + sorted(df["Category"].unique().tolist())

c1, c2, c3 = st.columns(3)
insurer_choice = c1.selectbox("Insurer", insurers)
category_choice = c2.selectbox("Premium Category", categories)
horizon = c3.slider("Forecast horizon (months)", 3, 36, 12)

monthly, p_pred, p_lower, p_upper, p_best, p_metrics, q_pred, q_lower, q_upper, q_best, q_metrics = fc.get_forecast(
    df, insurer_choice, category_choice, horizon
)

tab1, tab2 = st.tabs(["Premium", "Policies"])

with tab1:
    st.caption(f"Best model: {p_best}")
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Total_Premium"], mode="lines+markers", name="Actual", line=dict(color="#1f77b4")))

    fig.add_trace(go.Scatter(x=p_pred.index, y=p_upper.values, mode="lines", name="Upper Bound", line=dict(width=0), showlegend=False))

    fig.add_trace(go.Scatter(x=p_pred.index, y=p_lower.values, mode="lines", name="95% Prediction Interval", line=dict(width=0), fill="tonexty", fillcolor="rgba(255,127,14,0.15)"))

    fig.add_trace(go.Scatter(x=p_pred.index, y=p_pred.values, mode="lines+markers", name="Forecast", line=dict(color="#ff7f0e")))

    fig.update_layout(xaxis_title="Month", yaxis_title="Total Premium", height=450)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.caption(f"Best model: {q_best}")
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=monthly.index, y=monthly["Total_Policies"], mode="lines+markers", name="Actual", line=dict(color="#1f77b4")))

    fig2.add_trace(go.Scatter(x=q_pred.index, y=q_upper.values, mode="lines", name="Upper Bound", line=dict(width=0), showlegend=False))

    fig2.add_trace(go.Scatter(x=q_pred.index, y=q_lower.values, mode="lines", name="95% Prediction Interval", line=dict(width=0), fill="tonexty", fillcolor="rgba(255,127,14,0.15)"))

    fig2.add_trace(go.Scatter(x=q_pred.index, y=q_pred.values, mode="lines+markers", name="Forecast", line=dict(color="#ff7f0e")
    ))

    fig2.update_layout(xaxis_title="Month", yaxis_title="Total Policies", height=450)
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Forecast Table")

table = pd.DataFrame({
    "Month": p_pred.index.strftime("%b %Y"),
    "Forecast Premium": p_pred.values.round(0),
    "Premium Lower Bound": p_lower.values.round(0),
    "Premium Upper Bound": p_upper.values.round(0),
    "Forecast Policies": q_pred.values.round(0),
    "Policies Lower Bound": q_lower.values.round(0),
    "Policies Upper Bound": q_upper.values.round(0)
})

st.dataframe(table, use_container_width=True, hide_index=True)

st.subheader("Model Comparison")


def metrics_table(metrics_dict):
    rows = []
    for model_name, m in metrics_dict.items():
        if m is None:
            rows.append({"Model": model_name, "MAE": "failed", "RMSE": "-", "WAPE %": "-", "Bias %": "-"})
        else:
            rows.append({
                "Model": model_name,
                "MAE": round(m["MAE"], 2),
                "RMSE": round(m["RMSE"], 2),
                "WAPE %": round(m["WAPE"], 2),
                "Bias %": round(m["Bias"], 2)
            })
    return pd.DataFrame(rows)


mc1, mc2 = st.columns(2)
with mc1:
    st.write("Premium")
    if p_metrics:
        st.dataframe(metrics_table(p_metrics), use_container_width=True, hide_index=True)
    else:
        st.write("Not enough history to backtest.")
with mc2:
    st.write("Policies")
    if q_metrics:
        st.dataframe(metrics_table(q_metrics), use_container_width=True, hide_index=True)
    else:
        st.write("Not enough history to backtest.")
