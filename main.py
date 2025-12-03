import pandas as pd

from modules.data_loader import fetch_price_data, compute_daily_returns
from modules.db_handler import get_connection, save_dataframe
from modules.forecasting import forecast_portfolio_arima, plot_forecast
from modules.optimizer import (
    max_sharpe_weights,
)
from modules.risk_metrics import (
    asset_metrics_table,
    benchmark_metrics_table,
    portfolio_metrics_table,
)
from modules.visualizations import (
    plot_correlation_matrix,
    plot_drawdown_from_prices,
    plot_optimal_weights,
    plot_prices,
    plot_returns_hist,
    plot_rolling_volatility,
)


# -------- CONFIG --------

TICKERS = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "RELIANCE.NS", "ICICIBANK.NS"]
BENCHMARK_TICKER = "^NSEI"  # NIFTY 50 as benchmark
START_DATE = "2015-01-01"
END_DATE = None
RISK_FREE_RATE = 0.0  # as per task: assume rf = 0


def main():
    # 1) DATA LOADING
    prices, benchmark_prices = fetch_price_data(
        TICKERS, BENCHMARK_TICKER, start=START_DATE, end=END_DATE
    )
    asset_returns, benchmark_returns = compute_daily_returns(prices, benchmark_prices)

    # 2) OPTIMIZATION (MEAN-VARIANCE → MAX SHARPE)
    weights = max_sharpe_weights(asset_returns, rf=RISK_FREE_RATE)
    weights_series = pd.Series(weights, index=TICKERS, name="Weight")

    # Portfolio returns and value
    portfolio_returns = (asset_returns * weights).sum(axis=1)
    portfolio_prices = (1 + portfolio_returns).cumprod()
    portfolio_prices.name = "Portfolio"

    # 3) METRICS TABLES
    asset_metrics = asset_metrics_table(
        prices, asset_returns, benchmark_returns, rf=RISK_FREE_RATE
    )
    benchmark_metrics = benchmark_metrics_table(benchmark_prices, benchmark_returns)
    portfolio_metrics = portfolio_metrics_table(
        portfolio_prices,
        portfolio_returns,
        benchmark_prices,
        benchmark_returns,
        rf=RISK_FREE_RATE,
    )

    # 4) VISUALIZATIONS (ALL REQUIRED PLOTS)

    # Prices
    plot_prices(prices, "plots/prices/asset_prices.png")

    # Returns
    plot_returns_hist(asset_returns, "plots/returns/asset_returns_hist.png")

    # Volatility (rolling standard deviation)
    plot_rolling_volatility(
        asset_returns, window=63, save_path="plots/volatility/rolling_vol_63d.png"
    )

    # Drawdowns
    plot_drawdown_from_prices(
        portfolio_prices,
        "Portfolio Drawdown",
        "plots/drawdowns/portfolio_drawdown.png",
    )
    plot_drawdown_from_prices(
        benchmark_prices,
        "Benchmark Drawdown",
        "plots/drawdowns/benchmark_drawdown.png",
    )

    # Correlation matrix (diversification)
    plot_correlation_matrix(asset_returns, "plots/correlations/correlation_matrix.png")

    # Optimizer weights
    plot_optimal_weights(
        TICKERS,
        weights,
        "plots/optimizer/optimal_weights.png",
    )

    # 5) TIME SERIES PROJECTION (ARIMA FORECAST)
    forecast_df = forecast_portfolio_arima(portfolio_prices, steps=90)
    plot_forecast(
        portfolio_prices,
        forecast_df,
        "plots/forecast/portfolio_forecast.png",
        title="Portfolio Value Forecast (Next 90 Days)",
    )

    # 6) DATABASE STORAGE FOR ALL ANALYTICS
    conn = get_connection("db/analytics.db")

    # Raw & returns
    save_dataframe(conn, prices, "asset_prices", index_label="date")
    save_dataframe(
        conn,
        benchmark_prices.to_frame("Benchmark"),
        "benchmark_prices",
        index_label="date",
    )
    save_dataframe(conn, asset_returns, "asset_returns", index_label="date")
    save_dataframe(
        conn,
        benchmark_returns.to_frame("Benchmark"),
        "benchmark_returns",
        index_label="date",
    )

    # Metrics
    save_dataframe(conn, asset_metrics, "asset_metrics")
    save_dataframe(conn, benchmark_metrics, "benchmark_metrics")
    save_dataframe(conn, portfolio_metrics, "portfolio_metrics")

    # Optimizer results
    save_dataframe(conn, weights_series.to_frame(), "optimization_results", index_label="asset")

    # Correlation matrix (diversification)
    corr = asset_returns.corr()
    save_dataframe(conn, corr, "correlation_matrix")

    # Forecast / projections
    save_dataframe(conn, forecast_df, "portfolio_forecast", index_label="date")

    conn.close()

    # 7) TERMINAL SUMMARY
    print("\nOptimal Weights (Max Sharpe):")
    for asset, w in weights_series.items():
        print(f"{asset}: {w:.4f}")

    print("\nPortfolio Summary:")
    ann_ret = portfolio_returns.mean() * 252
    ann_vol = portfolio_returns.std() * (252**0.5)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    print(f"Annual Return: {ann_ret:.4f}")
    print(f"Annual Volatility: {ann_vol:.4f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")

    print("\nDone — all analytics saved!")


if __name__ == "__main__":
    main()
