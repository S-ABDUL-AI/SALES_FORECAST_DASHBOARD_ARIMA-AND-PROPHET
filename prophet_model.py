# model.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# ===========================
# Forecasting Functions
# ===========================

def forecast_arima(train_df: pd.DataFrame, test_df: pd.DataFrame, full_df: pd.DataFrame, horizon: int):
    """
    Train ARIMA on train_df and forecast for test + future horizon.
    Returns:
        pred_test (np.array): Predictions for test set
        forecast_df (pd.DataFrame): Future forecast dataframe with dates + forecasted_sales
    """
    model = ARIMA(train_df["sales"], order=(5, 1, 0))
    model_fit = model.fit()

    # Predict on test
    pred_test = model_fit.forecast(steps=len(test_df))

    # Forecast future
    pred_future = model_fit.forecast(steps=horizon)
    future_dates = pd.date_range(
        start=full_df["date"].iloc[-1] + pd.Timedelta(days=1),
        periods=horizon
    )
    forecast_df = pd.DataFrame({"date": future_dates, "forecasted_sales": pred_future})

    return pred_test, forecast_df


def forecast_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame, horizon: int):
    """
    Train Prophet on train_df and forecast for test + future horizon.
    Returns:
        pred_test (np.array): Predictions for test set
        forecast_df (pd.DataFrame): Future forecast dataframe with dates + forecasted_sales
    """
    df_prophet = train_df.rename(columns={"date": "ds", "sales": "y"})
    model = Prophet()
    model.fit(df_prophet)

    # Create future dataframe
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    # Predictions for test
    pred_test = forecast.iloc[len(train_df):len(train_df) + len(test_df)]["yhat"].values

    # Future forecast
    pred_future = forecast.iloc[-horizon:][["ds", "yhat"]].rename(
        columns={"ds": "date", "yhat": "forecasted_sales"}
    )

    return pred_test, pred_future
