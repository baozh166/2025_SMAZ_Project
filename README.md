# 📊 SMAZ 2025 Project: Financial Forecasting and Trading Simulation

This project explores financial time series analysis, macroeconomic indicators, technical analysis, and machine learning models to predict stock growth and simulate trading strategies. It leverages Python, Jupyter Notebook, and various data science libraries to build a robust pipeline for investment decision-making. a set of stock ['TSLA', '^GSPC', '^NDX', 'BTC-USD', 'ETH-USD', 'AAPL', 'MSFT', 'GOOG', 'NVDA', 'AMZN', 'META', 'BRK-B', 'LLY', 'AVGO','V', 'JPM'] with the past 4 years' data was selected to train and test the a few models. The best model (Ranfom Forest) achieved a **CAGR of 26.36%** over 4 years.

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

## 📁 File Structure
├── SMAZ_2025_project_final.ipynb # Main notebook 
├── README.md # Project documentation 
├── stocks_df_combined_YYYY_MM_DD.parquet.brotli # Final dataset

## 📂 Data Access

The dataset used in this project is available for download:

🔗 [Click here to access the data](https://drive.google.com/file/d/1T9o24Z9NevrZcl4VHCMb5ZaZEZ7CQU44/view?usp=drive_link)


