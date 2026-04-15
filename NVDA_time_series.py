# Import the libraries used in this code.
# They are used for downloading data, processing data,
# building the time series model, checking results, and plotting.

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Download stock price data for the selected tech companies.
# The end date is 2026-01-01, so all 2025 data is included.

tickers = ["NVDA", "AAPL", "MSFT", "AVGO", "MU", "ORCL", "AMD", "TSM"]

data = yf.download(tickers, start="2015-01-01", end="2026-01-01")["Close"]

# Remove rows with missing values.
data = data.dropna()


# Take log of the stock prices.
# This can help make the series more stable.

log_data = np.log(data)


# Create lagged moving average variables for NVDA.
# shift(1) means we only use past information.

log_data["NVDA_MA20_lag1"] = log_data["NVDA"].rolling(20).mean().shift(1)
log_data["NVDA_MA50_lag1"] = log_data["NVDA"].rolling(50).mean().shift(1)


# Create one-day lagged variables for the other stocks.
# This is used as explanatory variables in the model.

peer_stocks = ["AAPL", "MSFT", "AVGO", "MU", "ORCL", "AMD", "TSM"]

for stock in peer_stocks:
    log_data[f"{stock}_lag1"] = log_data[stock].shift(1)


# Drop missing rows after creating the lagged variables.
log_data = log_data.dropna()


# Set the target variable and the explanatory variables.
# The target is NVDA log price.
# The explanatory variables are lagged values from other stocks
# and lagged moving averages from NVDA.

target = log_data["NVDA"]

exog = log_data[
    [
        "AAPL_lag1",
        "MSFT_lag1",
        "AVGO_lag1",
        "MU_lag1",
        "ORCL_lag1",
        "AMD_lag1",
        "TSM_lag1",
        "NVDA_MA20_lag1",
        "NVDA_MA50_lag1"
    ]
]


# Split the data into training part and testing part.
# Here 80% is for training and 20% is for testing.

train_size = int(len(log_data) * 0.8)

y_train = target.iloc[:train_size]
y_test = target.iloc[train_size:]

X_train = exog.iloc[:train_size]
X_test = exog.iloc[train_size:]


# Run the ADF test on the training series.
# This is used to check stationarity.

result = adfuller(y_train)

print("ADF Statistic:", result[0])
print("p-value:", result[1])


# Use auto_arima to choose the ARIMA order automatically.
# We use a non-seasonal model here.

auto_model = auto_arima(
    y_train,
    X=X_train,
    seasonal=False,
    trace=True,
    stepwise=True,
    error_action="ignore",
    suppress_warnings=True
)

print(auto_model.summary())

order = auto_model.order


# Fit the SARIMAX model using the training data.

model = SARIMAX(
    y_train,
    exog=X_train,
    order=order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)

print(results.summary())


# Make forecast for the testing period.

forecast = results.forecast(
    steps=len(y_test),
    exog=X_test
)


# Change the forecast and actual values back to original price scale.

forecast_price = np.exp(forecast)
test_price = np.exp(y_test)
train_price = np.exp(y_train)


# Compute RMSE and MAE to evaluate the forecast result.

rmse = np.sqrt(mean_squared_error(test_price, forecast_price))
mae = mean_absolute_error(test_price, forecast_price)

print("RMSE:", rmse)
print("MAE:", mae)


# Plot the training data, testing data, and forecast in one figure.

plt.figure(figsize=(12, 6))

plt.plot(train_price.index, train_price, label="Train")
plt.plot(test_price.index, test_price, label="Test")
plt.plot(test_price.index, forecast_price, label="Forecast")

plt.title("SARIMAX Forecast for NVDA Stock Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()