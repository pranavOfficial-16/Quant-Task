from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


TRADING_DAYS_YEAR = 252


# ---------- BASIC HELPERS ----------

def annualize_return(daily_returns: pd.Series) -> float:
    mean_daily = daily_returns.mean()
    return (1.0 + mean_daily) ** TRADING_DAYS_YEAR - 1.0


def annualize_vol(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(TRADING_DAYS_YEAR)


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    excess = daily_returns - rf / TRADING_DAYS_YEAR
    return np.sqrt(TRADING_DAYS_YEAR) * excess.mean() / excess.std()


def sortino_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    excess = daily_returns - rf / TRADING_DAYS_YEAR
    downside = excess[excess < 0]
    if downside.std() == 0:
        return np.nan
    return np.sqrt(TRADING_DAYS_YEAR) * excess.mean() / downside.std()


def information_ratio(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    diff = portfolio_returns - benchmark_returns
    return np.sqrt(TRADING_DAYS_YEAR) * diff.mean() / diff.std()


def max_drawdown_from_prices(price_series: pd.Series) -> float:
    """
    Max drawdown using price series.
    """
    cumulative = price_series / price_series.iloc[0]
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1.0
    return drawdown.min()


def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    cov = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    var = np.var(benchmark_returns)
    return cov / var if var != 0 else np.nan


def tracking_error(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    diff = portfolio_returns - benchmark_returns
    return diff.std() * np.sqrt(TRADING_DAYS_YEAR)


def jensens_alpha_daily(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    beta_value: float,
    rf: float = 0.0,
) -> float:
    """
    Daily Jensen's alpha.
    """
    rf_daily = rf / TRADING_DAYS_YEAR
    excess_port = portfolio_returns - rf_daily
    excess_bench = benchmark_returns - rf_daily
    expected_port = rf_daily + beta_value * excess_bench
    alpha = excess_port - (expected_port - rf_daily)
    return alpha.mean()


def r_squared(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    cov = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    var_p = np.var(portfolio_returns)
    var_b = np.var(benchmark_returns)
    if var_p == 0 or var_b == 0:
        return np.nan
    corr = cov / np.sqrt(var_p * var_b)
    return corr**2


def value_at_risk_1d(returns: pd.Series, level: float = 0.95) -> float:
    """
    Returns VaR at given confidence (e.g. 95%) as the alpha percentile of returns.
    """
    alpha = 1.0 - level
    return float(np.percentile(returns, 100.0 * alpha))


def conditional_var_1d(returns: pd.Series, level: float = 0.95) -> float:
    var = value_at_risk_1d(returns, level)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return np.nan
    return float(tail.mean())


def annualize_var(var_1d: float) -> float:
    return var_1d * np.sqrt(TRADING_DAYS_YEAR)


def annualize_cvar(cvar_1d: float) -> float:
    return cvar_1d * np.sqrt(TRADING_DAYS_YEAR)


def horizon_return_from_prices(
    prices: pd.Series, days: int
) -> Optional[float]:
    """
    Simple horizon return over last 'days' (daily steps).
    """
    if len(prices) <= days:
        return None
    end = prices.iloc[-1]
    start = prices.iloc[-days - 1]
    return float(end / start - 1.0)


def cagr_from_prices(prices: pd.Series, years: float) -> Optional[float]:
    """
    CAGR computed over given number of years (approx using trading days).
    """
    if years <= 0:
        return None
    n_days = int(years * TRADING_DAYS_YEAR)
    if len(prices) <= n_days:
        return None
    end = prices.iloc[-1]
    start = prices.iloc[-n_days]
    return float((end / start) ** (1.0 / years) - 1.0)


# ---------- TABLE BUILDERS ----------

def asset_metrics_table(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    rf: float = 0.0,
) -> pd.DataFrame:
    """
    Compute required metrics for each individual asset.
    """
    rows = []

    for col in returns.columns:
        r = returns[col]
        p = prices[col]

        rf_daily = rf / TRADING_DAYS_YEAR

        # Basic distribution
        skewness = float(skew(r, bias=False))
        kurt = float(kurtosis(r, fisher=False, bias=False))

        # Vols & ratios
        sh = sharpe_ratio(r, rf)
        so = sortino_ratio(r, rf)
        std_ann = annualize_vol(r)

        # Beta vs benchmark
        b = beta(r, benchmark_returns)
        alpha_daily = jensens_alpha_daily(r, benchmark_returns, b, rf)
        rsq = r_squared(r, benchmark_returns)

        # Alpha distribution
        excess_bench = benchmark_returns - rf_daily
        exp_r = rf_daily + b * excess_bench
        alpha_series = r - exp_r
        alpha_skew = float(skew(alpha_series, bias=False))
        alpha_kurt = float(kurtosis(alpha_series, fisher=False, bias=False))

        # Stress days: bottom 10% benchmark returns
        stress_threshold = np.percentile(benchmark_returns, 10)
        stress_days = alpha_series[benchmark_returns <= stress_threshold]
        mean_alpha_stress = float(stress_days.mean()) if len(stress_days) > 0 else np.nan

        # VaR / CVaR
        var_1d = value_at_risk_1d(r, 0.95)
        cvar_1d = conditional_var_1d(r, 0.95)

        row = {
            "Asset": col,
            "1D Return": horizon_return_from_prices(p, 1),
            "5D Return": horizon_return_from_prices(p, 5),
            "1M Return": horizon_return_from_prices(p, 21),
            "3M Return": horizon_return_from_prices(p, 63),
            "6M Return": horizon_return_from_prices(p, 126),
            "1Y CAGR": cagr_from_prices(p, 1),
            "3Y CAGR": cagr_from_prices(p, 3),
            "5Y CAGR": cagr_from_prices(p, 5),
            "Standard Deviation (Ann.)": std_ann,
            "Sharpe Ratio": sh,
            "Sortino Ratio": so,
            "Beta (vs Benchmark)": b,
            "Jensen Alpha (Daily)": alpha_daily,
            "R-squared": rsq,
            "Alpha Skewness": alpha_skew,
            "Alpha Kurtosis": alpha_kurt,
            "Mean Alpha on Stress Days": mean_alpha_stress,
            "VaR 1D @95%": var_1d,
            "CVaR 1D @95%": cvar_1d,
            "VaR Annualized (Approx.)": annualize_var(var_1d),
            "CVaR Annualized (Approx.)": annualize_cvar(cvar_1d),
            "Skewness": skewness,
            "Kurtosis": kurt,
            "Max Drawdown": max_drawdown_from_prices(p),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def benchmark_metrics_table(
    benchmark_prices: pd.Series, benchmark_returns: pd.Series
) -> pd.DataFrame:
    """
    Compute required metrics for the benchmark.
    """
    p = benchmark_prices
    r = benchmark_returns

    row = {
        "Benchmark": p.name if p.name is not None else "Benchmark",
        "1D Return": horizon_return_from_prices(p, 1),
        "5D Return": horizon_return_from_prices(p, 5),
        "1M Return": horizon_return_from_prices(p, 21),
        "3M Return": horizon_return_from_prices(p, 63),
        "6M Return": horizon_return_from_prices(p, 126),
        "1Y CAGR": cagr_from_prices(p, 1),
        "3Y CAGR": cagr_from_prices(p, 3),
        "5Y CAGR": cagr_from_prices(p, 5),
        "Standard Deviation (Ann.)": annualize_vol(r),
        "Sharpe Ratio": sharpe_ratio(r),
        "Sortino Ratio": sortino_ratio(r),
        "Max Drawdown": max_drawdown_from_prices(p),
        "Skewness": float(skew(r, bias=False)),
        "Kurtosis": float(kurtosis(r, fisher=False, bias=False)),
        "VaR 1D @95%": value_at_risk_1d(r, 0.95),
        "CVaR 1D @95%": conditional_var_1d(r, 0.95),
    }

    return pd.DataFrame([row])


def portfolio_metrics_table(
    portfolio_prices: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_prices: pd.Series,
    benchmark_returns: pd.Series,
    rf: float = 0.0,
) -> pd.DataFrame:
    """
    Compute required portfolio-level metrics.
    """
    p = portfolio_prices
    r = portfolio_returns
    rf_daily = rf / TRADING_DAYS_YEAR

    skewness = float(skew(r, bias=False))
    kurt = float(kurtosis(r, fisher=False, bias=False))

    sh = sharpe_ratio(r, rf)
    so = sortino_ratio(r, rf)
    info = information_ratio(r, benchmark_returns)

    std_ann = annualize_vol(r)
    std_bench_ann = annualize_vol(benchmark_returns)
    te = tracking_error(r, benchmark_returns)

    b = beta(r, benchmark_returns)
    alpha_daily = jensens_alpha_daily(r, benchmark_returns, b, rf)
    rsq = r_squared(r, benchmark_returns)

    excess_bench = benchmark_returns - rf_daily
    exp_r = rf_daily + b * excess_bench
    alpha_series = r - exp_r
    alpha_skew = float(skew(alpha_series, bias=False))
    alpha_kurt = float(kurtosis(alpha_series, fisher=False, bias=False))
    stress_threshold = np.percentile(benchmark_returns, 10)
    stress_days = alpha_series[benchmark_returns <= stress_threshold]
    mean_alpha_stress = float(stress_days.mean()) if len(stress_days) > 0 else np.nan

    var_1d = value_at_risk_1d(r, 0.95)
    cvar_1d = conditional_var_1d(r, 0.95)

    row = {
        "Standard Deviation (Portfolio, Ann.)": std_ann,
        "Standard Deviation (Benchmark, Ann.)": std_bench_ann,
        "Tracking Error (Ann.)": te,
        "Sharpe Ratio": sh,
        "Sharpe Ratio (Benchmark)": sharpe_ratio(benchmark_returns, rf),
        "Sortino Ratio": so,
        "Sortino Ratio (Benchmark)": sortino_ratio(benchmark_returns, rf),
        "Information Ratio": info,
        "Max Drawdown (Portfolio)": max_drawdown_from_prices(p),
        "Max Drawdown (Benchmark)": max_drawdown_from_prices(benchmark_prices),
        "Beta (Portfolio)": b,
        "Beta (General)": b,
        "Jensen Alpha (Daily)": alpha_daily,
        "R-squared": rsq,
        "Alpha Skewness": alpha_skew,
        "Alpha Kurtosis": alpha_kurt,
        "Mean Alpha on Stress Days": mean_alpha_stress,
        "VaR 1D @95%": var_1d,
        "CVaR 1D @95%": cvar_1d,
        "VaR Annualized (Approx.)": annualize_var(var_1d),
        "CVaR Annualized (Approx.)": annualize_cvar(cvar_1d),
        "Skewness": skewness,
        "Kurtosis": kurt,
        "1D Return": horizon_return_from_prices(p, 1),
        "5D Return": horizon_return_from_prices(p, 5),
        "1M Return": horizon_return_from_prices(p, 21),
        "3M Return": horizon_return_from_prices(p, 63),
        "6M Return": horizon_return_from_prices(p, 126),
        "1Y CAGR": cagr_from_prices(p, 1),
        "3Y CAGR": cagr_from_prices(p, 3),
        "5Y CAGR": cagr_from_prices(p, 5),
    }

    return pd.DataFrame([row])
