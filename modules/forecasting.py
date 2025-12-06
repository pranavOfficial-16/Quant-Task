from typing import Optional

import matplotlib.pyplot as plt
import os
import warnings

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Suppress statsmodels warnings (convergence, index, etc.)
warnings.filterwarnings("ignore")


def forecast_portfolio_arima(
    series: pd.Series, steps: int = 90
) -> pd.DataFrame:
    """
    ARIMA-based forecast of portfolio value.
    Returns DataFrame with forecast and confidence intervals.
    """
    # Ensure a proper daily frequency
    series = series.asfreq("D")

    model = ARIMA(
        series,
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit()

    forecast_res = fit.get_forecast(steps=steps)
    mean_forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()

    df = pd.DataFrame(
        {
            "forecast": mean_forecast,
            "lower_ci": conf_int.iloc[:, 0],
            "upper_ci": conf_int.iloc[:, 1],
        }
    )

    # Build a proper datetime index for forecast
    forecast_index = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=len(df),
        freq="D",
    )
    df.index = forecast_index

    return df


def plot_forecast(
    history: pd.Series,
    forecast_df: pd.DataFrame,
    save_path: str,
    title: str = "Portfolio Value Forecast",
) -> None:
    """
    Plot historical portfolio series and ARIMA forecast with confidence intervals.
    """
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(history.index, history.values, label="Historical")
    plt.plot(forecast_df.index, forecast_df["forecast"], label="Forecast")

    plt.fill_between(
        forecast_df.index,
        forecast_df["lower_ci"],
        forecast_df["upper_ci"],
        alpha=0.2,
        label="Confidence Interval",
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
