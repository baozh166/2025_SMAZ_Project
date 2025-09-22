# 📊 SMAZ 2025 Project: Financial Forecasting and Trading Simulation

## 🧠 Description

Stock market prediction is a notoriously complex and dynamic problem. Prices are influenced by a multitude of factors including macroeconomic indicators, company fundamentals, geopolitical events, investor sentiment, and technical patterns. Traditional models often struggle to capture the nonlinear and stochastic nature of financial markets.

This project tackles the challenge by combining:

- **Macroeconomic context** (e.g., GDP, CPI, interest rates)
- **Technical analysis** (e.g., momentum, volatility, volume indicators)
- **Machine learning models** (e.g., decision trees, logistic regression, LSTM neural networks)

The goal is to forecast short-term stock growth and simulate trading strategies that outperform benchmark indices like the S&P 500. By integrating diverse data sources and modeling techniques, the project aims to uncover actionable insights and build a robust decision-making framework for investors.
4 years' data for 16 US financial instruments with high liquidility ['TSLA', '^GSPC', '^NDX', 'BTC-USD', 'ETH-USD', 'AAPL', 'MSFT', 'GOOG', 'NVDA', 'AMZN', 'META', 'BRK-B', 'LLY', 'AVGO','V', 'JPM'] were selected to train and test the a few models. The best model (Ranfom Forest) achieved a **CAGR of 26.36%** over 4 years.

## 🚀 Project Overview

The notebook performs the following key tasks:

1. **Data Acquisition**
   - Pulls OHLCV data for major stocks and ETFs using `yfinance` and `pandas_datareader`.
   - Includes macroeconomic indicators (GDP, CPI, FED rates, Treasury yields, VIX, Gold, Oil).

2. **Feature Engineering**
   - Calculates historical and future growth rates.
   - Generates technical indicators using TA-Lib (momentum, volume, volatility, cycle, price transforms, and pattern recognition).
   - Adds categorical and dummy variables for modeling.

3. **Machine Learning Models**
   - Decision Tree and Random Forest classifiers with hyperparameter tuning.
   - Logistic Regression with feature scaling.
   - LSTM neural network with Keras for deep learning (To be completed).

4. **Trading Simulations**
   - Simulates daily trading based on model predictions.
   - Implements stop-loss and take-profit logic.
   - Calculates net returns, CAGR, and capital growth over time.

5. **Benchmark Comparison**
   - Compares model performance against S&P 500 benchmark CAGR.

## 📈 Results

- Best model achieved a **CAGR of 26.36%** over 4 years.
- Compared to S&P 500 benchmark CAGR of ~9.11%.
- Simulation includes dynamic capital allocation, realistic trading constraints, and probabilistic decision rules.

## 🛠️ Technologies Used

- Python (NumPy, Pandas, Matplotlib, Seaborn, Plotly)
- yFinance, pandas_datareader
- scikit-learn (DecisionTree, RandomForest, LogisticRegression)
- TA-Lib for technical indicators
- Keras/TensorFlow for LSTM modeling
- Jupyter Notebook (Google Colab compatible)

## 📂 Data Access

The dataset used in this project is available for download:

🔗 [Click here to access the data](https://drive.google.com/file/d/1T9o24Z9NevrZcl4VHCMb5ZaZEZ7CQU44/view?usp=drive_link)


