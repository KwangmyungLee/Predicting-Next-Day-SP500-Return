# Predicting Next-Day S&P 500 Returns Using Machine Learning and Deep Learning

This project explores whether machine-learning models can forecast next-day S&P 500 index returns using a combination of technical indicators and macroeconomic variables.  
Five models were implemented and compared:

- **ARIMA (baseline time-series model)**
- **VECM (cointegration-based multivariate time-series model)**
- **Random Forest (tree-based ML)**
- **XGBoost (gradient-boosted trees)**
- **LSTM (deep learning sequence model)**

The goal is not to “beat the market,” but to evaluate how different model families behave when applied to real financial data without look-ahead bias or data leakage.

---

## 1. Data Sources

| Source | Dataset | Frequency |
|--------|---------|-----------|
| Yahoo Finance | S&P500, NASDAQ, VIX, Dollar Index, Oil, Gold, Silver, Copper, NatGas | Daily |
| FRED (Federal Reserve) | Fed Funds Rate, CPI, Unemployment Rate, UMich Sentiment, St. Louis FSI | Monthly → forward-filled to daily |

All data is pulled dynamically in the notebook using `yfinance` and `pandas_datareader`, so no CSVs are required.

---

## 2. Feature Engineering

| Category | Features |
|----------|----------|
| Technical Indicators | SMA(5/20/60), EMA(12/26), RSI(14), MACD, Bollinger Bands |
| Market Features | VIX, US10Y, Dollar Index, Oil, Gold, Copper, NatGas |
| Macro Features | CPI, Unemployment, Fed Rate, UMich Sentiment, FSI |
| Target | Next-day % return of S&P 500 (`shift(-1)` applied) |

---

## 3. Model Comparison

| Model        | RMSE ↓     | MAE ↓     | Directional Accuracy ↑ |
|--------------|------------|-----------|-------------------------|
| **ARIMA**        | 0.010186   | 0.006320  | 58.04%                 |
| **VECM**         | 0.010196   | 0.006333  | 54.71%                 |
| **Random Forest**| 0.011865   | 0.008631  | 43.30%                 |
| **XGBoost**      | 0.010220   | 0.006349  | 56.70%                 |
| **LSTM**         | **0.008234** | **0.006136** | **51.55%**             |

In short:

- Classical models (ARIMA, VECM) struggled when trained on raw price-based series.
- Tree-based methods produced usable but not outstanding forecasts.
- The LSTM model captured more of the return dynamics, especially in turning points.

---

## 4. Repository Structure

```
project-root/
│── Predicting_S&P_Next_Day_Return.ipynb   # Full workflow: data, features, training, evaluation
│── Predicting_S&P_Next_Day_Return.pdf     # Full written report
│── README.md                              # You are here
└── requirements.txt                       # Python dependencies


The notebook downloads all data dynamically through API calls — no static datasets are required.

---
```

## 5. How to Run

bash
pip install -r requirements.txt
jupyter notebook "Predicting_S&P_Next_Day_Return.ipynb"

---

## 6. Key Takeaways

- Predicting returns is more realistic than predicting price levels in financial time series.
- Traditional econometric models showed weak performance when applied to short-horizon return prediction.
- LSTM delivered the strongest result, but even the best model did not produce reliable alpha on its own.
- A predictive model alone is not a trading strategy — risk, execution, and costs matter.
- This project serves as a research baseline for further portfolio or strategy development.

---

## 7. Possible Extensions

- Add walk-forward / expanding-window training instead of a fixed split
- Incorporate news sentiment, earnings revisions, or macro surprise indices
- Convert regression output into classification (up vs down) and test confusion matrix
- Build a long/flat or long/short trading strategy and backtest with costs
- Extend from single-asset prediction to portfolio allocation / factor modeling


