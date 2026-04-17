# DS4420 Project
## Stock Price Prediction Using Machine Learning

## Streamlit App

We also created a Streamlit app for this project.

**Deployed app link:**  
https://ds4420-project-et8kx8roxsd3n3byf8tiyj.streamlit.app/

The app includes two pages:

- An overview page with a short description of the project, data, methods, findings, and future work
- An interactive forecast viewer that lets users choose a stock, choose a model view, select a date range, and compare actual and forecasted stock prices

The app uses the saved result files from the two models:

- `sarimax_all.csv`
- `bayesian_all.csv`

## Model 1: Time Series Model (SARIMAX)

This model uses a log-price-based SARIMAX model to predict stock prices for 8 large technology stocks.

### Objective
The goal of this model is to predict future stock prices.  
For each stock, the model uses lagged moving averages of the target stock and lagged information from the other technology stocks.

### Dataset
The data comes from Yahoo Finance.  
We use daily historical stock price data.

**Time period:**  
January 1, 2015 – January 1, 2026

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
- Daily closing price from Yahoo Finance
- Log-transformed stock price as the modeling scale
- One-day lagged 20-day moving average of the target stock on the log scale
- One-day lagged 50-day moving average of the target stock on the log scale
- One-day lagged log prices of the other technology stocks

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
- A summary table with ADF results, selected ARIMA order, RMSE, and MAE for each stock

## Model 2: Bayesian Model

This model uses a Bayesian linear regression model to predict stock prices for 8 large technology stocks.

### Objective
The goal of this model is to predict stock prices.  
For each stock, the model uses its own lagged log price information, lagged moving average information, and lagged information from the other technology stocks.

### Dataset
The data comes from Yahoo Finance.  
We use daily historical closing price data.

**Time period:**  
January 1, 2015 – January 1, 2026

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

- Log-transformed stock price
- One-day lagged log price of the target stock
- One-day lagged 20-day moving average of the target stock on the log scale
- One-day lagged 50-day moving average of the target stock on the log scale
- One-day lagged log prices of the other technology stocks

### Method
We build one Bayesian linear regression model for each stock.  
The data is split into training data and testing data.  
We use 80% of the data for training and 20% for testing.  
The model is fitted on log-transformed prices using Gibbs sampling.  
A zero-mean normal prior with variance controlled by `tau2` is used for the regression coefficients, and an inverse-gamma prior is used for Sigma².  
Posterior predictive draws are generated on the log scale.  
After prediction, the forecast values are transformed back to the original price scale.

### Evaluation
We use these two metrics to evaluate the model:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

### Output
This model gives:

- Prediction plots for the 8 stocks
- A Bayesian model summary table with posterior mean of Sigma², RMSE, and MAE for each stock
