import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir_for_file(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def plot_prices(prices: pd.DataFrame, save_path: str) -> None:
    _ensure_dir_for_file(save_path)
    plt.figure(figsize=(10, 6))
    for col in prices.columns:
        plt.plot(prices.index, prices[col], label=col)
    plt.title("Asset Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    


def plot_returns_hist(returns: pd.DataFrame, save_path: str) -> None:
    _ensure_dir_for_file(save_path)
    plt.figure(figsize=(10, 6))
    for col in returns.columns:
        plt.hist(returns[col].dropna(), bins=50, alpha=0.5, label=col)
    plt.title("Daily Returns Distribution")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()



def plot_rolling_volatility(
    returns: pd.DataFrame, window: int, save_path: str
) -> None:
    _ensure_dir_for_file(save_path)
    plt.figure(figsize=(10, 6))
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    for col in rolling_vol.columns:
        plt.plot(rolling_vol.index, rolling_vol[col], label=col)
    plt.title(f"Rolling {window}-Day Annualized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    


def plot_drawdown_from_prices(
    price_series: pd.Series, title: str, save_path: str
) -> None:
    _ensure_dir_for_file(save_path)

    cumulative = price_series / price_series.iloc[0]
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0

    plt.figure(figsize=(10, 5))
    plt.plot(drawdown.index, drawdown.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()



def plot_correlation_matrix(returns: pd.DataFrame, save_path: str) -> None:
    _ensure_dir_for_file(save_path)
    corr = returns.corr()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr.values, interpolation="none")
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    

def plot_optimal_weights(
    tickers: List[str], weights: np.ndarray, save_path: str
) -> None:
    _ensure_dir_for_file(save_path)
    plt.figure(figsize=(8, 5))
    plt.bar(tickers, weights)
    plt.title("Optimal Portfolio Weights (Max Sharpe)")
    plt.xlabel("Asset")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
