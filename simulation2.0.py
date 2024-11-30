import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import datetime

# Title of the Streamlit app
st.markdown("<h3 style='font-size: 40px;'>Stock Price Simulation with GBM and Prediction Accuracy</h3>", unsafe_allow_html=True)

# Sidebar title
st.sidebar.header("Control Panel")

# Expanded stock selection
stock_options = {
    # Technology
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'NVDA': 'NVIDIA Corporation',
    'META': 'Meta Platforms Inc.',
    'AVGO': 'Broadcom Inc.',
    'CSCO': 'Cisco Systems Inc.',
    'ORCL': 'Oracle Corporation',
    'CRM': 'Salesforce Inc.',
    
    # Finance
    'JPM': 'JPMorgan Chase & Co.',
    'BAC': 'Bank of America Corp.',
    'V': 'Visa Inc.',
    'MA': 'Mastercard Inc.',
    'WFC': 'Wells Fargo & Co.',
    'GS': 'Goldman Sachs Group Inc.',
    'MS': 'Morgan Stanley',
    'BLK': 'BlackRock Inc.',
    
    # Healthcare
    'JNJ': 'Johnson & Johnson',
    'UNH': 'UnitedHealth Group Inc.',
    'PFE': 'Pfizer Inc.',
    'ABT': 'Abbott Laboratories',
    'TMO': 'Thermo Fisher Scientific',
    'MRK': 'Merck & Co.',
    'LLY': 'Eli Lilly and Company',
    
    # Consumer
    'PG': 'Procter & Gamble Co.',
    'KO': 'Coca-Cola Company',
    'PEP': 'PepsiCo Inc.',
    'WMT': 'Walmart Inc.',
    'COST': 'Costco Wholesale Corp.',
    'MCD': 'McDonald\'s Corp.',
    'NKE': 'Nike Inc.',
    'DIS': 'Walt Disney Co.',
    
    # Industrial
    'CAT': 'Caterpillar Inc.',
    'BA': 'Boeing Company',
    'HON': 'Honeywell International',
    'UPS': 'United Parcel Service',
    'MMM': '3M Company',
    'GE': 'General Electric Co.',
    
    # Energy
    'XOM': 'Exxon Mobil Corp.',
    'CVX': 'Chevron Corporation',
    'COP': 'ConocoPhillips',
    
    # Telecommunications
    'VZ': 'Verizon Communications',
    'T': 'AT&T Inc.',
    'TMUS': 'T-Mobile US Inc.',
    
    # Real Estate
    'AMT': 'American Tower Corp.',
    'PLD': 'Prologis Inc.',
    
    # Materials
    'LIN': 'Linde plc',
    'APD': 'Air Products & Chemicals',
    
    # Utilities
    'NEE': 'NextEra Energy Inc.',
    'DUK': 'Duke Energy Corp.',
    'SO': 'Southern Company'
}

# Stock selection with company names (in sidebar)
stock_symbol = st.sidebar.selectbox(
    "Select a stock symbol",
    options=list(stock_options.keys()),
    format_func=lambda x: f"{x} - {stock_options[x]}"
)

# Simulation parameters (in sidebar)
st.sidebar.subheader("Simulation Parameters")
num_paths = st.sidebar.slider("Number of simulation paths", min_value=1, max_value=100, value=5)
timespan_days = st.sidebar.slider("Comparison timespan (days)", min_value=5, max_value=30, value=15)

# Date range selection (in sidebar)
st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime.date(2023, 1, 1),
    min_value=datetime.date(2010, 1, 1),
    max_value=datetime.date.today()
)
end_date = st.sidebar.date_input(
    "End Date",
    value=datetime.date(2024, 11, 1),
    min_value=start_date,
    max_value=datetime.date.today()
)

# Fetch and clean historical data
def get_clean_financial_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.columns = data.columns.get_level_values(0)
    data = data.ffill()
    data.index = data.index.tz_localize(None)
    return data

# Get stock data
googl_hist = get_clean_financial_data(stock_symbol, start_date=start_date, end_date=end_date)

# Calculate daily returns
googl_hist['Return'] = googl_hist['Close'].pct_change().dropna()

# Drift and volatility
returns = googl_hist['Return'].dropna()
mu = returns.mean() * 252  # Annualized drift
sigma = returns.std() * (252 ** 0.5)  # Annualized volatility

# Display key statistics
st.sidebar.subheader("Stock Statistics")
st.sidebar.text(f"Annual Drift: {mu:.2%}")
st.sidebar.text(f"Annual Volatility: {sigma:.2%}")

# Initial stock price
S0 = googl_hist['Close'].iloc[-1]

# Simulation parameters
T = (datetime.datetime(2025, 12, 31) - googl_hist.index[-1]).days / 365  # Total simulation time (years)
N = int(T * 252)  # Number of time steps

# Function to simulate GBM paths
def simulate_gbm_multiple_paths(S0, mu, sigma, T, N, num_paths):
    dt = T / N
    t = np.linspace(0, T, N)
    paths = np.zeros((N, num_paths))
    for i in range(num_paths):
        W = np.random.standard_normal(size=N)
        W = np.cumsum(W) * np.sqrt(dt)
        paths[:, i] = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return paths

# Simulate GBM paths
simulated_paths = simulate_gbm_multiple_paths(S0, mu, sigma, T, N, num_paths)

# Actual final price and dates
last_trading_date = googl_hist.index[-1]
future_dates = pd.date_range(last_trading_date, periods=N, freq='B')  # Business days
comparison_date = last_trading_date + pd.Timedelta(days=timespan_days)

# Fetch actual price at comparison date
actual_comparison_price = None
try:
    actual_comparison_price = get_clean_financial_data(
        stock_symbol, start_date=last_trading_date, end_date=comparison_date
    )['Close'].iloc[-1]
except Exception as e:
    st.error(f"Unable to fetch actual price for comparison date: {e}")

# Predicted prices for comparison
comparison_step = int((timespan_days / 252) * N)  # Corresponding time step
predicted_prices = simulated_paths[comparison_step] if comparison_step < N else None

# Display predictions and metrics in main area
st.subheader("Prediction Results")
col1, col2 = st.columns(2)

# Mean predicted price at the comparison step (if within range)
if comparison_step < N:
    with col1:
        mean_predicted_price_comparison = predicted_prices.mean()
        st.metric(
            label=f"Mean Predicted Price ({comparison_date.date()})", 
            value=f"${mean_predicted_price_comparison:.2f}"
        )

# Calculate prediction accuracy
if actual_comparison_price and predicted_prices is not None:
    with col2:
        st.metric(
            label=f"Actual Price ({comparison_date.date()})",
            value=f"${actual_comparison_price:.2f}"
        )
    
    errors = (predicted_prices - actual_comparison_price) / actual_comparison_price * 100
    mean_error = errors.mean()
    min_error = errors.min()
    max_error = errors.max()
    
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Mean Error", f"{mean_error:.2f}%")
    with col4:
        st.metric("Min Error", f"{min_error:.2f}%")
    with col5:
        st.metric("Max Error", f"{max_error:.2f}%")

    # Plot comparison
    st.subheader("Error Distribution")
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=10, alpha=0.7, label="Prediction Errors (%)")
    plt.axvline(0, color='red', linestyle='--', label="Zero Error")
    plt.title("Prediction Errors Distribution")
    plt.xlabel("Error (%)")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(plt)

# Plot the actual price and simulated paths
st.subheader("Price Simulation")
plt.figure(figsize=(10, 5))
plt.plot(googl_hist.index, googl_hist['Close'], color='blue', label='Actual Closing Prices')
for i in range(num_paths):
    plt.plot(future_dates, simulated_paths[:, i], alpha=0.7, label='Simulated Price' if i == 0 else "")
plt.axhline(y=S0, color='red', linestyle='--', label='Starting Price')
plt.title(f"Simulated Stock Prices for {stock_symbol} ({stock_options[stock_symbol]})")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
st.pyplot(plt)