# Import libraries that we need for data download, data processing,
# time series modeling, and plotting results.

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error


# Download historical stock prices from Yahoo Finance
# for several large technology companies.

tickers = ["NVDA","AAPL","MSFT","AVGO","MU","ORCL","PLTR","AMD"]

data = yf.download(tickers, start="2015-01-01", end="2025-01-01")["Close"]

# Remove missing values from the dataset
data = data.dropna()


# Apply log transformation to stock prices.
# This helps make the time series more stable.

log_data = np.log(data)


# Create two technical indicators for NVDA.
# These are 20-day and 50-day moving averages.

log_data["NVDA_MA20"] = log_data["NVDA"].rolling(20).mean()
log_data["NVDA_MA50"] = log_data["NVDA"].rolling(50).mean()

log_data = log_data.dropna()


# Define the target variable and the explanatory variables.
# NVDA is the stock we want to predict.
# Other stocks and indicators are used as input variables.

target = log_data["NVDA"]

exog = log_data[[
    "AAPL","MSFT","AVGO","MU","ORCL","PLTR","AMD",
    "NVDA_MA20","NVDA_MA50"
]]


# Split the dataset into training and testing data.
# 80% of the data is used for training the model.

train_size = int(len(log_data)*0.8)

y_train = target[:train_size]
y_test = target[train_size:]

X_train = exog[:train_size]
X_test = exog[train_size:]


# Perform Augmented Dickey-Fuller test to check
# whether the time series is stationary.

result = adfuller(y_train)

print("ADF Statistic:", result[0])
print("p-value:", result[1])


# Use auto_arima to automatically select
# the best ARIMA parameters.

auto_model = auto_arima(
    y_train,
    exogenous=X_train,
    seasonal=False,
    trace=True,
    stepwise=True
)

print(auto_model.summary())

order = auto_model.order


# Train the SARIMAX model using the training data
# and the selected ARIMA parameters.

model = SARIMAX(
    y_train,
    exog=X_train,
    order=order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit()

print(results.summary())


# Use the trained model to make predictions
# for the testing period.

forecast = results.forecast(
    steps=len(y_test),
    exog=X_test
)


# Convert the predicted values from log scale
# back to the original price scale.

forecast_price = np.exp(forecast)
test_price = np.exp(y_test)
train_price = np.exp(y_train)


# Calculate RMSE to measure prediction error.

rmse = np.sqrt(mean_squared_error(test_price, forecast_price))

print("RMSE:", rmse)


# Plot the training data, testing data,
# and predicted values to compare results.

plt.figure(figsize=(12,6))

plt.plot(train_price.index, train_price, label="Train")
plt.plot(test_price.index, test_price, label="Test")
plt.plot(test_price.index, forecast_price, label="Forecast")

plt.title("SARIMAX Forecast for NVDA Stock Price")
plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()
plt.show()
