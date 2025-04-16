import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Stock Price Forecast", layout="wide")

# Title
st.title("📈 Stock Price Forecast with Weighted Expected Return")

# Sidebar Inputs
st.sidebar.header("📌 Forecast Settings")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL, RELIANCE.NS)", "RELIANCE.NS").strip().upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
user_sector_growth = st.sidebar.number_input(
    "Optional: Enter sector growth rate (e.g., 0.1 for 10%, leave 0 to calculate)", 
    min_value=-1.0, max_value=1.0, value=0.0, step=0.01
)

# Function to estimate sector growth (proxy using historical sector ETF)
def get_sector_growth(ticker, start_date, end_date, price_data=None, user_sector_growth=None):
    if user_sector_growth != 0.0:
        st.write(f"Debug: Using user-provided sector growth for {ticker}: {user_sector_growth:.2%}")
        return user_sector_growth

    stock = yf.Ticker(ticker)
    info = stock.info
    sector = info.get("sector", "Unknown")
    st.write(f"Debug: Stock {ticker} Sector: {sector}")
    
    # Map sectors to Indian ETFs
    sector_etf_map = {
        "Technology": "ITBEES.NS",
        "Financial Services": "BFSI.NS",
        "Energy": "ENERGYBEES.NS",
        "Consumer Cyclical": "CONSUMBEES.NS",
        "Healthcare": "HEALTHYBEES.NS",
        "Industrials": "INFRABEES.NS",
        "Utilities": "ENERGYBEES.NS",
        "Basic Materials": "NIFTYMIDCAP150.NS",
        "Communication Services": "NIFTY500BEES.NS",
        "Consumer Defensive": "FMCGBEES.NS",
        "Real Estate": "REALTBEES.NS"
    }
    etf_ticker = sector_etf_map.get(sector, "^NSEI")  # Default to Nifty 50
    st.write(f"Debug: Using ETF/Index for {sector}: {etf_ticker}")
    
    try:
        etf_data = yf.download(etf_ticker, start=start_date, end=end_date)["Close"]
        st.write(f"Debug: ETF Data Length for {etf_ticker}: {len(etf_data)}")
        if etf_data.empty or etf_data.isna().all():
            st.warning(f"No valid data for {etf_ticker}. Using stock fallback.")
            if price_data is not None:
                stock_returns = price_data.pct_change().dropna()
                if not stock_returns.empty:
                    annualized_growth = ((1 + stock_returns.mean()) ** 252 - 1)
                    st.write(f"Debug: Fallback to stock return: {annualized_growth:.2%}")
                    return annualized_growth if np.isfinite(annualized_growth) else 0.05
            st.warning(f"Using default 5% growth for {ticker}.")
            return 0.05
        returns = etf_data.pct_change().dropna()
        st.write(f"Debug: ETF Returns Length: {len(returns)}")
        if returns.empty:
            st.warning(f"No valid returns for {etf_ticker}. Using default 5%.")
            return 0.05
        annualized_growth = ((1 + returns.mean()) ** 252 - 1)
        st.write(f"Debug: Calculated Sector Growth for {etf_ticker}: {annualized_growth:.2%}")
        return annualized_growth if np.isfinite(annualized_growth) else 0.05
    except Exception as e:
        st.warning(f"Error fetching data for {etf_ticker}: {e}. Using stock fallback.")
        if price_data is not None:
            stock_returns = price_data.pct_change().dropna()
            if not stock_returns.empty:
                annualized_growth = ((1 + stock_returns.mean()) ** 252 - 1)
                st.write(f"Debug: Fallback to stock return: {annualized_growth:.2%}")
                return annualized_growth if np.isfinite(annualized_growth) else 0.05
        st.warning(f"Using default 5% growth for {ticker}.")
        return 0.05

# Function to estimate fundamental valuation (simplified DCF proxy)
def get_fundamental_valuation(ticker, current_price):
    stock = yf.Ticker(ticker)
    info = stock.info
    target_price = info.get('targetMeanPrice')
    if target_price and np.isfinite(target_price):
        return (target_price - current_price) / current_price
    else:
        try:
            earnings = stock.earnings
            if earnings is not None and 'Earnings' in earnings:
                growth_rate = earnings['Earnings'].pct_change().mean()
                return growth_rate * 252 if np.isfinite(growth_rate) else 0.0
        except:
            return 0.0

# Button to trigger forecast
if st.sidebar.button("Generate Forecast"):
    try:
        # Validate date range
        if start_date >= end_date:
            st.error("End date must be after start date.")
            st.stop()
        if start_date > datetime.now().date() or end_date > datetime.now().date():
            st.error("Date range is in the future. Please select historical dates.")
            st.stop()

        # Fetch historical price data
        price_data = yf.download(ticker, start=start_date, end=end_date)["Close"]
        if price_data.empty:
            st.error(f"No data available for {ticker} in the specified date range.")
            st.stop()

        # Ensure price_data is a Series and numeric
        if isinstance(price_data, pd.DataFrame):
            price_data = price_data.squeeze()
        price_data = pd.to_numeric(price_data, errors='coerce').dropna()

        if price_data.empty or price_data.isna().all():
            st.error(f"Price data for {ticker} is empty or contains only invalid values.")
            st.write("Debug: Price Data Sample:", price_data.head())
            st.stop()

        # Monte Carlo Simulation
        last_price = price_data.iloc[-1]
        returns = price_data.pct_change().dropna()
        if returns.empty:
            st.error(f"Insufficient data to calculate returns for {ticker}.")
            st.stop()

        # Dynamic drift
        fundamental_return = get_fundamental_valuation(ticker, last_price)
        sector_growth = get_sector_growth(ticker, start_date, end_date, price_data, user_sector_growth)
        mu = 0.5 * returns.mean() + 0.25 * fundamental_return + 0.25 * sector_growth
        sigma = returns.std()
        n_days = 252
        n_sims = 10000
        sims = np.zeros((n_days, n_sims))
        for i in range(n_sims):
            sims[:, i] = last_price * np.exp(np.cumsum((mu - 0.5 * sigma**2) * 1/252 + sigma * np.sqrt(1/252) * np.random.normal(size=n_days)))
        avg_forecast_price = np.mean(sims[-1, :])
        monte_carlo_return = (avg_forecast_price - last_price) / last_price if np.isfinite(avg_forecast_price) else 0.0

        # Weighted expected return
        weighted_expected_return = (
            0.35 * monte_carlo_return +
            0.50 * fundamental_return +
            0.15 * sector_growth
        )

        # Risk metrics
        sorted_prices = np.sort(sims[-1, :])
        var_95 = sorted_prices[int(0.05 * n_sims)]
        cvar_95 = sorted_prices[sorted_prices <= var_95].mean()
        cvar_return = (cvar_95 - last_price) / last_price if np.isfinite(cvar_95) else 0.0

        # Display results
        st.subheader(f"📡 {ticker} Price Forecast (Monte Carlo)")
        fig, ax = plt.subplots()
        ax.plot(sims[:, :100])
        ax.set_title(f"{ticker} Monte Carlo Price Paths (1 Year)")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        st.pyplot(fig)

        st.subheader("📊 Forecast Results")
        st.markdown(f"**Current Price**: {last_price:.2f}")
        st.markdown(f"**Average Forecasted Price (1 Year)**: {avg_forecast_price:.2f}")
        st.markdown(f"**Monte Carlo Return **: {monte_carlo_return:.2%}")
        st.markdown(f"**Fundamental Return **: {fundamental_return:.2%}")
        st.markdown(f"**Sector Growth Return **: {sector_growth:.2%}")
        st.markdown(f"**Weighted Expected Return**: {weighted_expected_return:.2%}")
        st.markdown(f"**95% Value-at-Risk (VaR)**: {var_95:.2f}")
        st.markdown(f"**95% Conditional Value-at-Risk (CVaR)**: {cvar_95:.2f} ({cvar_return:.2%})")

    except Exception as e:
        st.error(f"⚠️ Error fetching data or generating forecast: {e}")