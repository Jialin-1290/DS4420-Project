# Import the libraries used in this code.
# They are for downloading data, processing data,
# building the time series model, checking the model,
# and plotting the forecast result.

import warnings
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Hide warning messages that do not stop the code.
# This makes the output easier to read.

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Download daily closing prices for the selected tech stocks.
# The end date is 2026-01-01, so all 2025 data is included.

tickers = ["NVDA", "AAPL", "MSFT", "AVGO", "MU", "ORCL", "AMD", "TSM"]

data = yf.download(
    tickers,
    start="2015-01-01",
    end="2026-01-01",
    auto_adjust=False
)["Close"]

# Remove rows with missing values.
data = data.dropna()

# Make sure the date index is in datetime format
# and the rows are in the correct time order.
data.index = pd.to_datetime(data.index)
data = data.sort_index()


# Define a function to run one SARIMAX model for one stock.
def run_sarimax_for_stock(stock_name, full_data, ticker_list, train_ratio=0.8):
    """
    Run one SARIMAX model for one selected stock.

    Parameters:
    stock_name : str
        The stock ticker that we want to predict.
    full_data : pandas.DataFrame
        The full stock price data for all selected stocks.
    ticker_list : list
        The list of all stock tickers used in the model.
    train_ratio : float, optional
        The ratio of data used for training.
        The default value is 0.8.

    Returns:
    dict
        A dictionary with the main result for this stock,
        including ADF statistic, ADF p-value, selected order,
        RMSE, and MAE.
    """

    # Take log of the stock prices.
    # This can help make the series more stable.
    log_data = np.log(full_data.copy())

    # Create lagged moving average variables for the target stock.
    # shift(1) means only past information is used.
    log_data[f"{stock_name}_MA20_lag1"] = log_data[stock_name].rolling(20).mean().shift(1)
    log_data[f"{stock_name}_MA50_lag1"] = log_data[stock_name].rolling(50).mean().shift(1)

    # Create one-day lagged variables for the other stocks.
    # These are used as explanatory variables.
    peer_stocks = [stock for stock in ticker_list if stock != stock_name]

    for stock in peer_stocks:
        log_data[f"{stock}_lag1"] = log_data[stock].shift(1)

    # Drop missing rows after creating lagged variables.
    log_data = log_data.dropna()

    # Set the target variable.
    # This is the log price of the selected stock.
    target = log_data[stock_name]

    # Set the explanatory variables.
    # These include lagged peer stock prices
    # and lagged moving averages of the target stock.
    exog_columns = [f"{stock}_lag1" for stock in peer_stocks]
    exog_columns += [f"{stock_name}_MA20_lag1", f"{stock_name}_MA50_lag1"]
    exog = log_data[exog_columns]

    # Split the data into training and testing parts.
    # 80% is used for training and 20% is used for testing.
    train_size = int(len(log_data) * train_ratio)
    y_train = target.iloc[:train_size]
    y_test = target.iloc[train_size:]
    X_train = exog.iloc[:train_size]
    X_test = exog.iloc[train_size:]

    # Run the ADF test on the training series.
    # This is used to check stationarity.
    adf_result = adfuller(y_train)
    adf_stat = adf_result[0]
    adf_pvalue = adf_result[1]

    print("\n" + "=" * 70)
    print(f"Running SARIMAX for {stock_name}")
    print(f"ADF Statistic: {adf_stat}")
    print(f"ADF p-value: {adf_pvalue}")

    # Use auto_arima to choose the ARIMA order automatically.
    # A non-seasonal model is used here.
    auto_model = auto_arima(
        y_train,
        X=X_train,
        seasonal=False,
        trace=False,
        stepwise=True,
        error_action="ignore",
        suppress_warnings=True,
        max_p=3,
        max_q=3,
        max_d=2
    )

    order = auto_model.order
    print(f"Selected order for {stock_name}: {order}")

    # Fit the SARIMAX model using the training data.
    model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = model.fit(disp=False)

    # Make forecasts for the testing period.
    forecast_log = results.forecast(steps=len(y_test), exog=X_test)

    # Change the values back to the original price scale.
    train_price = np.exp(y_train)
    test_price = np.exp(y_test)
    forecast_price = np.exp(forecast_log)

    # Compute RMSE and MAE to evaluate forecast performance.
    rmse = np.sqrt(mean_squared_error(test_price, forecast_price))
    mae = mean_absolute_error(test_price, forecast_price)

    print(f"RMSE for {stock_name}: {rmse}")
    print(f"MAE for {stock_name}: {mae}")

    # Plot the training data, testing data, and forecast.
    plt.figure(figsize=(12, 6))
    plt.plot(train_price.index, train_price, label="Train")
    plt.plot(test_price.index, test_price, label="Test")
    plt.plot(test_price.index, forecast_price, label="Forecast")
    plt.title(f"SARIMAX Forecast for {stock_name}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Return the main results for this stock.
    return {
        "Stock": stock_name,
        "ADF Statistic": adf_stat,
        "ADF p-value": adf_pvalue,
        "Order": order,
        "RMSE": rmse,
        "MAE": mae
    }


# Run the model for all selected stocks
# and save the results in a list.
all_results = []

for stock in tickers:
    result = run_sarimax_for_stock(stock, data, tickers)
    all_results.append(result)

# Make the final summary table.
results_df = pd.DataFrame(all_results)

print("\nFinal Summary Table:")
print(results_df)