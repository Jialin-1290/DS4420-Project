import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Set the page title and layout
st.set_page_config(page_title="Stock Forecasting Project", layout="wide")

# Load the forecast result files
sarimax_df = pd.read_csv("sarimax_all.csv")
bayesian_df = pd.read_csv("bayesian_all.csv")

# Change Date to datetime format
sarimax_df["Date"] = pd.to_datetime(sarimax_df["Date"])
bayesian_df["Date"] = pd.to_datetime(bayesian_df["Date"])

# Get the stock names
tickers = sorted(sarimax_df["Stock"].unique().tolist())

# Show the title and project information
st.title("Stock Forecasting Project")
st.caption("DS4420 course project")
st.caption("Authors: Jialin Weng and Mengyang Wang")

# Let the user choose the page
page = st.sidebar.radio(
    "Go to",
    ["Project Overview", "Interactive Forecast Viewer"]
)

if page == "Project Overview":
    st.header("Project Overview")

    st.write(
        "This project studies stock price prediction for eight large technology companies. "
        "The stocks are NVDA, AAPL, MSFT, AVGO, MU, ORCL, AMD, and TSM."
    )

    st.write(
        "We used daily closing prices from Yahoo Finance. "
        "The data cover 2015 through 2025."
    )

    st.write(
        "We compared two models on the same dataset. "
        "One is a SARIMAX time series model. "
        "The other is a Bayesian linear regression model."
    )

    st.write(
        "Both models used past price-based information to make forecasts. "
        "We used RMSE and MAE to compare prediction performance."
    )

    st.write(
        "Both models gave useful forecasts, but they worked best for different stocks. "
        "In the SARIMAX model, NVDA had the lowest forecast error. "
        "In the Bayesian model, AAPL had the lowest forecast error."
    )

    st.write(
        "Use the second page to explore the forecast results with an interactive plot."
    )

elif page == "Interactive Forecast Viewer":
    st.header("Interactive Forecast Viewer")

    # Make two columns for the user choices
    col1, col2 = st.columns(2)

    with col1:
        selected_ticker = st.selectbox("Choose a stock", tickers)

    with col2:
        selected_model = st.selectbox(
            "Choose a model view",
            ["SARIMAX", "Bayesian", "Both"]
        )

    # Keep only the selected stock data
    sarimax_stock = sarimax_df[sarimax_df["Stock"] == selected_ticker].copy()
    bayesian_stock = bayesian_df[bayesian_df["Stock"] == selected_ticker].copy()

    # Build the full date range
    all_dates = pd.concat([sarimax_stock["Date"], bayesian_stock["Date"]])
    min_date = all_dates.min().to_pydatetime()
    max_date = all_dates.max().to_pydatetime()

    # Let the user choose the date range
    date_range = st.slider(
        "Choose date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    # Filter the data by the selected date range
    sarimax_stock = sarimax_stock[
        (sarimax_stock["Date"] >= start_date) & (sarimax_stock["Date"] <= end_date)
    ]
    bayesian_stock = bayesian_stock[
        (bayesian_stock["Date"] >= start_date) & (bayesian_stock["Date"] <= end_date)
    ]

    # Create the figure
    fig = go.Figure()

    if selected_model == "SARIMAX":
        fig.add_trace(go.Scatter(
            x=sarimax_stock["Date"],
            y=sarimax_stock["Actual"],
            mode="lines",
            name="Actual",
            line=dict(color="orange")
        ))
        fig.add_trace(go.Scatter(
            x=sarimax_stock["Date"],
            y=sarimax_stock["Forecast"],
            mode="lines",
            name="SARIMAX Forecast",
            line=dict(color="blue")
        ))

    elif selected_model == "Bayesian":
        fig.add_trace(go.Scatter(
            x=bayesian_stock["Date"],
            y=bayesian_stock["Actual"],
            mode="lines",
            name="Actual",
            line=dict(color="orange")
        ))
        fig.add_trace(go.Scatter(
            x=bayesian_stock["Date"],
            y=bayesian_stock["Forecast"],
            mode="lines",
            name="Bayesian Forecast",
            line=dict(color="green")
        ))

    else:
        fig.add_trace(go.Scatter(
            x=sarimax_stock["Date"],
            y=sarimax_stock["Actual"],
            mode="lines",
            name="Actual",
            line=dict(color="orange")
        ))
        fig.add_trace(go.Scatter(
            x=sarimax_stock["Date"],
            y=sarimax_stock["Forecast"],
            mode="lines",
            name="SARIMAX Forecast",
            line=dict(color="blue")
        ))
        fig.add_trace(go.Scatter(
            x=bayesian_stock["Date"],
            y=bayesian_stock["Forecast"],
            mode="lines",
            name="Bayesian Forecast",
            line=dict(color="green")
        ))

    # Set the plot title and labels
    fig.update_layout(
        title=f"{selected_ticker} Forecast Comparison",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        legend_title="Series"
    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Selected Data")

    if selected_model == "SARIMAX":
        st.dataframe(sarimax_stock, use_container_width=True)

    elif selected_model == "Bayesian":
        st.dataframe(bayesian_stock, use_container_width=True)

    else:
        merged_df = sarimax_stock.merge(
            bayesian_stock[["Date", "Forecast"]],
            on="Date",
            how="inner",
            suffixes=("_SARIMAX", "_Bayesian")
        )

        # Rename the columns for easier reading
        merged_df = merged_df.rename(columns={
            "Actual": "Actual",
            "Forecast_SARIMAX": "SARIMAX Forecast",
            "Forecast_Bayesian": "Bayesian Forecast"
        })

        st.dataframe(merged_df, use_container_width=True)