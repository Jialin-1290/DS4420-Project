import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Tech Stock Price Prediction", layout="wide")

# Load data
sarimax_df = pd.read_csv("sarimax_all.csv")
bayesian_df = pd.read_csv("bayesian_all.csv")

sarimax_df["Date"] = pd.to_datetime(sarimax_df["Date"])
bayesian_df["Date"] = pd.to_datetime(bayesian_df["Date"])

tickers = sorted(sarimax_df["Stock"].unique().tolist())

# Title
st.title("Tech Stock Price Prediction")

# Sidebar page selection
page = st.sidebar.radio(
    "Go to",
    ["Project Overview", "Interactive Forecast Viewer"]
)

if page == "Project Overview":
    st.header("Project Overview")

    st.subheader("Purpose")
    st.write(
        "This project studies stock price prediction for eight large technology companies. "
        "We compare a SARIMAX time series model and a Bayesian linear regression model "
        "on the same dataset."
    )

    st.subheader("Data")
    st.write(
        "The data were downloaded from Yahoo Finance. "
        "We used daily closing prices for NVDA, AAPL, MSFT, AVGO, MU, ORCL, AMD, and TSM "
        "from 2015 to 2025."
    )

    st.subheader("Methods")
    st.write(
        "The SARIMAX model uses lagged peer stock prices and lagged moving averages. "
        "The Bayesian linear regression model uses lagged own-stock information, "
        "lagged peer stock prices, and lagged moving averages. "
        "Both models are evaluated using RMSE and MAE on the original price scale."
    )

    st.subheader("Main Finding")
    st.write(
        "Both models produced useful forecasts, but the Bayesian model had lower forecast error overall on this dataset."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("SARIMAX Sample")
        st.dataframe(sarimax_df.head(10), use_container_width=True)

    with col2:
        st.subheader("Bayesian Sample")
        st.dataframe(bayesian_df.head(10), use_container_width=True)

elif page == "Interactive Forecast Viewer":
    st.header("Interactive Forecast Viewer")

    col1, col2 = st.columns(2)

    with col1:
        selected_ticker = st.selectbox("Choose a stock", tickers)

    with col2:
        selected_model = st.selectbox(
            "Choose a model view",
            ["SARIMAX", "Bayesian", "Both"]
        )

    sarimax_stock = sarimax_df[sarimax_df["Stock"] == selected_ticker].copy()
    bayesian_stock = bayesian_df[bayesian_df["Stock"] == selected_ticker].copy()

    # Build overall date range
    all_dates = pd.concat([sarimax_stock["Date"], bayesian_stock["Date"]])
    min_date = all_dates.min().to_pydatetime()
    max_date = all_dates.max().to_pydatetime()

    date_range = st.slider(
        "Choose date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    sarimax_stock = sarimax_stock[
        (sarimax_stock["Date"] >= start_date) & (sarimax_stock["Date"] <= end_date)
    ]
    bayesian_stock = bayesian_stock[
        (bayesian_stock["Date"] >= start_date) & (bayesian_stock["Date"] <= end_date)
    ]

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
        # Use SARIMAX actual only once for display
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

    fig.update_layout(
        title=f"{selected_ticker} Forecast Comparison",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        legend_title="Series"
    )

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
        merged_df = merged_df.rename(columns={
            "Actual": "Actual",
            "Forecast_SARIMAX": "SARIMAX Forecast",
            "Forecast_Bayesian": "Bayesian Forecast"
        })
        st.dataframe(merged_df, use_container_width=True)