# DS4420 Project
## Stock Price Prediction Using Machine Learning

## Model 1: Time Series Model (SARIMAX)

This model uses a price-based SARIMAX model to predict stock prices for 8 large technology stocks.

### Objective
The goal of this model is to predict future stock prices.  
For each stock, the model uses its own past price information and lagged information from the other technology stocks.

### Dataset
The data comes from Yahoo Finance.  
We use daily historical stock price data.

**Time period:**  
January 1, 2015 – December 31, 2025

### Stocks
The 8 stocks used in this project are:

- NVDA (NVIDIA)
- AAPL (Apple)
- MSFT (Microsoft)
- AVGO (Broadcom)
- MU (Micron)
- ORCL (Oracle)
- AMD (Advanced Micro Devices)
- TSM (Taiwan Semiconductor Manufacturing Company)

### Features
For each target stock, the model uses:

- Daily closing price
- Log-transformed stock price
- Lagged 20-day moving average of the target stock
- Lagged 50-day moving average of the target stock
- Lagged log prices of the other technology stocks

### Method
We build one SARIMAX model for each stock.  
The data is split into training data and testing data.  
We use 80% of the data for training and 20% for testing.  
The ARIMA order is selected automatically by `auto_arima`.

### Evaluation
We use these two metrics to evaluate the model:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

### Output
This model gives:

- Forecast plots for the 8 stocks
- A summary table with ARIMA order, RMSE, and MAE for each stock
