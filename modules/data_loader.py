import os
from typing import List, Tuple, Optional

import pandas as pd
import yfinance as yf


TRADING_DAYS_YEAR = 252


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fetch_price_data(
    tickers: List[str],
    benchmark_ticker: str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fetch historical daily adjusted close prices for a list of tickers and benchmark.
    Saves raw prices into data/raw.
    """
    _ensure_dir("data/raw")

    all_tickers = tickers + [benchmark_ticker]
    raw = yf.download(all_tickers, start=start, end=end, auto_adjust=False)

    # Handle multi-index vs single-index columns
    if isinstance(raw.columns, pd.MultiIndex):
        adj_close = raw["Adj Close"]
    else:
        adj_close = raw[["Adj Close"]]
        adj_close.columns = all_tickers

    prices = adj_close[tickers].dropna(how="all")
    benchmark = adj_close[benchmark_ticker].dropna()

    # Align on common dates
    common_idx = prices.index.intersection(benchmark.index)
    prices = prices.loc[common_idx]
    benchmark = benchmark.loc[common_idx]

    prices.to_csv("data/raw/asset_prices_raw.csv")
    benchmark.to_frame("Benchmark").to_csv("data/raw/benchmark_prices_raw.csv")

    return prices, benchmark


def compute_daily_returns(
    prices: pd.DataFrame, benchmark: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute simple daily returns for assets and benchmark.
    Saves cleaned prices and returns into data/cleaned.
    """
    _ensure_dir("data/cleaned")

    asset_returns = prices.pct_change().dropna()
    benchmark_returns = benchmark.pct_change().dropna()

    common_idx = asset_returns.index.intersection(benchmark_returns.index)
    asset_returns = asset_returns.loc[common_idx]
    benchmark_returns = benchmark_returns.loc[common_idx]
    prices = prices.loc[common_idx]
    benchmark = benchmark.loc[common_idx]

    prices.to_csv("data/cleaned/asset_prices_clean.csv")
    benchmark.to_frame("Benchmark").to_csv("data/cleaned/benchmark_prices_clean.csv")
    asset_returns.to_csv("data/cleaned/asset_returns.csv")
    benchmark_returns.to_frame("Benchmark").to_csv(
        "data/cleaned/benchmark_returns.csv"
    )

    return asset_returns, benchmark_returns
