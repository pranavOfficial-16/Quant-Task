# Quantitative Portfolio Analysis & Optimization

## 1. Project Brief

This project implements a complete quantitative research pipeline for equity analysis and portfolio optimization. It downloads market data, computes asset-level and portfolio-level risk metrics, generates visualizations, optimizes portfolio weights using Mean-Variance (Max Sharpe Ratio), and forecasts portfolio value using an ARIMA time-series model.

The output includes:

- Cleaned price and return data
- Asset metrics and benchmark metrics
- Portfolio risk and performance metrics
- Optimized portfolio weights
- Correlation and diversification analysis
- Time-series forecast
- All required plots
- A SQLite database containing all analytics

---

## 2. Tech Stack

- **Python**
- **yfinance** – Market data ingestion
- **pandas / numpy** – Data processing & analytics
- **matplotlib** – Visualization
- **statsmodels** – ARIMA forecasting
- **scipy.optimize** – Portfolio optimization
- **sqlite3** – Database storage

---

## 3. Project Structure

```sh
quant_project/
│
├── data/
│ ├── raw/ # Raw downloaded market data
│ └── cleaned/ # Cleaned prices & daily returns
│
├── plots/ # All generated visualizations
│ ├── prices/
│ ├── returns/
│ ├── volatility/
│ ├── drawdowns/
│ ├── correlations/
│ ├── optimizer/
│ └── forecast/
│
├── db/
│ └── analytics.db # SQLite database with all tables
│
├── modules/
│ ├── data_loader.py # Data download, cleaning, saving
│ ├── risk_metrics.py # Asset & portfolio risk metrics
│ ├── optimizer.py # Max Sharpe portfolio optimizer
│ ├── visualizations.py # All plotting utilities
│ ├── forecasting.py # ARIMA forecasting + plotting
│ └── db_handler.py # SQLite saving helper
│
└── main.py # Complete pipeline runner
```

---

## 4. Output

After running the project, the following outputs are produced:

### A. Terminal Output

```sh
- Optimal Max Sharpe weights
- Annual return
- Annual volatility
- Sharpe ratio
- Completion message
```

### B. Plots (PNG files)

```sh
Generated in `plots/`:

- `prices/asset_prices.png`
- `returns/asset_returns_hist.png`
- `volatility/rolling_vol_63d.png`
- `drawdowns/portfolio_drawdown.png`
- `drawdowns/benchmark_drawdown.png`
- `correlations/correlation_matrix.png`
- `optimizer/optimal_weights.png`
- `forecast/portfolio_forecast.png`
```

### C. Data Files

```sh
Stored under:

data/raw/
data/cleaned/
```

### D. Database

```sh
SQLite DB:  
db/analytics.db

Tables include:

- asset_prices
- benchmark_prices
- asset_returns
- benchmark_returns
- asset_metrics
- benchmark_metrics
- portfolio_metrics
- optimization_results
- correlation_matrix
- portfolio_forecast
```
---

## 5. How to Run

### Step 1 — Create virtual environment

```sh
python -m venv venv
venv\Scripts\activate
```

### Step 2 — Install Dependencies

```sh
pip install -r requirements.txt
```

### Step 3 — Run the project

```sh
python main.py
```

### Step 3 — View Outputs

```sh
- Plots → in the `plots/` folder
- Cleaned data → in `data/cleaned/`
- Raw data → in `data/raw/`
- Full analytics database → `db/analytics.db`
- Summary → shown in terminal
```