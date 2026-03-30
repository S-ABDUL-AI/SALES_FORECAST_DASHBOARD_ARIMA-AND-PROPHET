import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import os

# Must be first Streamlit call.
st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")

# ----------------------------
# Reduce top margin
# ----------------------------
st.markdown(
    """
    <style>
    .css-18e3th9 {padding-top: 1rem;}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Demo CSV
# ----------------------------
DEMO_FILE = "demo_sales.csv"
if not os.path.exists(DEMO_FILE):
    demo_dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
    demo_sales = np.random.randint(80, 200, size=len(demo_dates))
    demo_df = pd.DataFrame({"date": demo_dates, "sales": demo_sales})
    demo_df.to_csv(DEMO_FILE, index=False)

# ----------------------------
# Streamlit Layout
# ----------------------------
st.title("📊 Sales Forecast Dashboard with ARIMA and Prophet")

# ----------------------------
# Sidebar Options + Author Info
# ----------------------------
st.sidebar.header("⚙️ Forecast Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["ARIMA", "Prophet"])
forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 30, 120, 60)

st.sidebar.markdown("---")
st.sidebar.subheader("👨‍💻 About the Developer")
st.sidebar.markdown(
    """
    **Sherriff Abdul-Hamid**  
    AI Engineer | Data Scientist/Analyst | Economist  

    **Contact:**  
    [GitHub](https://github.com/S-ABDUL-AI) |  
    [LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/) |  
    📧 Sherriffhamid001@gmail.com
    """
)

# ----------------------------
# Upload CSV or use demo
# ----------------------------
uploaded_file = st.file_uploader("Upload your sales CSV (date, sales)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using demo sales data for demonstration. Download below to try.")
    df = pd.read_csv(DEMO_FILE)

st.download_button(
    "⬇️ Download Demo CSV",
    data=open(DEMO_FILE, "rb").read(),
    file_name="demo_sales.csv",
    mime="text/csv"
)

# ----------------------------
# Normalize columns
# ----------------------------
df.columns = df.columns.str.lower()
if "date" not in df.columns or "sales" not in df.columns:
    st.error("CSV must have 'date' and 'sales' columns.")
    st.stop()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ----------------------------
# Train/Test Split
# ----------------------------
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# ----------------------------
# Forecasting Functions
# ----------------------------
def forecast_arima(train_df, horizon):
    model = ARIMA(train_df["sales"], order=(5, 1, 0))
    model_fit = model.fit()
    pred_test = model_fit.forecast(steps=len(test))
    pred_future = model_fit.forecast(steps=horizon)
    future_dates = pd.date_range(start=df["date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon)
    forecast_df = pd.DataFrame({"date": future_dates, "forecasted_sales": pred_future})
    return pred_test, forecast_df

def forecast_prophet(df_train, horizon):
    df_prophet = df_train.rename(columns={"date": "ds", "sales": "y"})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    pred_test = forecast.iloc[len(df_train):len(df_train)+len(test)]["yhat"].values
    pred_future = forecast.iloc[-horizon:][["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "forecasted_sales"})
    return pred_test, pred_future

# ----------------------------
# Forecast
# ----------------------------
if model_choice == "ARIMA":
    pred_test, future_forecast_df = forecast_arima(train, forecast_horizon)
else:
    pred_test, future_forecast_df = forecast_prophet(train, forecast_horizon)

# Merge with actuals
merged = pd.concat([df, future_forecast_df], ignore_index=True)

# ----------------------------
# KPI Calculations
# ----------------------------
common = pd.DataFrame({"actual": test["sales"], "forecast": pred_test})
mae = mean_absolute_error(common["actual"], common["forecast"])
rmse = math.sqrt(mean_squared_error(common["actual"], common["forecast"]))
mape = np.mean(np.abs((common["actual"] - common["forecast"]) / common["actual"])) * 100

def kpi_color(val, metric="mape"):
    if metric=="mape":
        if val < 5: return "green"
        elif val < 15: return "orange"
        else: return "red"
    else:
        if val < 50: return "green"
        elif val < 100: return "orange"
        else: return "red"

# ----------------------------
# Executive Summary
# ----------------------------
summary_text = f"""
The {model_choice} model predicts sales with **MAPE of {mape:.2f}%**.  

- Forecasts deviate by ~{mape:.2f}% on average from actual sales.  
- Lower MAPE values indicate higher accuracy.  
- This forecast can be used to plan for inventory, staffing, and promotions.  
- The plots below show historical trends, forecast, and residuals.  

💡 **Tip:** Green KPIs indicate excellent accuracy, orange moderate, red low – quick executive interpretation.
"""
st.subheader("Executive Summary")
st.info(summary_text)

# ----------------------------
# KPI Legend
# ----------------------------
st.markdown(
    """
    <div style="display:flex; gap:1rem; margin-bottom:0.5rem;">
        <div style="background-color:green;color:white;padding:0.3rem 0.6rem;border-radius:5px;">Excellent</div>
        <div style="background-color:orange;color:white;padding:0.3rem 0.6rem;border-radius:5px;">Moderate</div>
        <div style="background-color:red;color:white;padding:0.3rem 0.6rem;border-radius:5px;">Low</div>
    </div>
    """, unsafe_allow_html=True
)

# ----------------------------
# Horizontal KPI row
# ----------------------------
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.markdown(f"**MAPE (%)**: <span style='color:{kpi_color(mape, 'mape')}'>{mape:.2f}%</span>", unsafe_allow_html=True)
kpi1.caption("Mean Absolute Percentage Error (smaller is better)")

kpi2.markdown(f"**MAE**: <span style='color:{kpi_color(mae, 'other')}'>{mae:.2f}</span>", unsafe_allow_html=True)
kpi2.caption("Mean Absolute Error in sales units")

kpi3.markdown(f"**RMSE**: <span style='color:{kpi_color(rmse, 'other')}'>{rmse:.2f}</span>", unsafe_allow_html=True)
kpi3.caption("Root Mean Squared Error in sales units")

# ----------------------------
# Residuals toggle
# ----------------------------
residual_type = st.radio("Select Residuals Type", ["Absolute Residuals", "Percentage Residuals"])

if residual_type == "Absolute Residuals":
    residuals_plot = common["actual"] - common["forecast"]
    y_label = "Residuals (units)"
else:
    residuals_plot = (common["actual"] - common["forecast"]) / common["actual"] * 100
    y_label = "Residuals (%)"

# ----------------------------
# Plots row with interactive hover
# ----------------------------
plot1, plot2 = st.columns(2)

with plot1:
    st.subheader("📈 Forecast vs Actual (Interactive)")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df["date"], y=df["sales"], mode="lines+markers", name="Actual",
        hovertemplate='Date: %{x}<br>Actual Sales: %{y}<extra></extra>'
    ))
    fig1.add_trace(go.Scatter(
        x=merged["date"], y=merged["forecasted_sales"], mode="lines+markers", name="Forecast",
        hovertemplate='Date: %{x}<br>Forecasted Sales: %{y}<extra></extra>'
    ))
    st.plotly_chart(fig1, use_container_width=True)

with plot2:
    st.subheader("📊 Residuals(Interactive)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=test["date"], y=residuals_plot, mode="lines+markers", name=residual_type,
        hovertemplate='Date: %{x}<br>Residual: %{y}<extra></extra>'
    ))
    fig2.add_hline(y=0, line_dash="dash", line_color="red")
    fig2.update_yaxes(title=y_label)
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Download Future Forecast
# ----------------------------
future_csv = future_forecast_df.to_csv(index=False)
st.download_button("📥 Download Future Forecast CSV", data=future_csv, file_name="future_forecast.csv", mime="text/csv")
