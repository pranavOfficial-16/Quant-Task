from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .risk_metrics import TRADING_DAYS_YEAR


def portfolio_return(weights: np.ndarray, returns: pd.DataFrame) -> float:
    mean_daily = returns.mean()
    return float(np.dot(weights, mean_daily) * TRADING_DAYS_YEAR)


def portfolio_volatility(weights: np.ndarray, returns: pd.DataFrame) -> float:
    cov_daily = returns.cov()
    var = np.dot(weights.T, np.dot(cov_daily, weights))
    return float(np.sqrt(var * TRADING_DAYS_YEAR))


def portfolio_sharpe(weights: np.ndarray, returns: pd.DataFrame, rf: float = 0.0) -> float:
    pr = portfolio_return(weights, returns)
    pv = portfolio_volatility(weights, returns)
    if pv == 0:
        return -1e9
    excess = pr - rf
    return excess / pv


def max_sharpe_weights(
    returns: pd.DataFrame,
    rf: float = 0.0,
) -> np.ndarray:
    """
    Mean-variance optimizer to find weights that maximize Sharpe ratio.
    Constraints:
      - sum(weights) = 1
      - 0 <= weight <= 1
    """
    n = returns.shape[1]
    init = np.ones(n) / n
    bounds = tuple((0.0, 1.0) for _ in range(n))

    def objective(w):
        return -portfolio_sharpe(np.array(w), returns, rf)

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        objective,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=[cons],
    )

    if not res.success:
        return init

    return res.x
