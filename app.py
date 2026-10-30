
import streamlit as st
import yfinance as yf
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="📈",
    layout="wide"
)

# Dashboard title
st.title("📈 AI Trading Bot")
st.caption("Nifty 50 Market Analysis Dashboard")

# Download market data
@st.cache_data(ttl=300)
def get_data():

    data = yf.download(
        "^NSEI",
        period="6mo",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    close = data["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    df = pd.DataFrame({"Close": close}).dropna()

    # Moving averages
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    # RSI calculation
    change = df["Close"].diff()

    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    return df


# Load data
df = get_data()

if df.empty or df["RSI"].dropna().empty:

    st.error("Market data unavailable. Please refresh later.")

else:

    # Latest market values
    latest = df.iloc[-1]

    price = latest["Close"]
    sma20 = latest["SMA20"]
    sma50 = latest["SMA50"]
    rsi = latest["RSI"]

    # Market overview
    st.subheader("Market Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Nifty 50 Price", f"{price:,.2f}")
    col2.metric("SMA 20", f"{sma20:,.2f}")
    col3.metric("SMA 50", f"{sma50:,.2f}")

    # RSI section
    st.subheader("RSI Indicator")
    st.metric("Current RSI", f"{rsi:.2f}")

    # Improved Market Analysis
    st.subheader("Market Analysis")

    # Moving Average Analysis
    if sma20 > sma50:

        trend = "Upward Trend Observation"
        trend_detail = "SMA 20 is above SMA 50."

    elif sma20 < sma50:

        trend = "Downward Trend Observation"
        trend_detail = "SMA 20 is below SMA 50."

    else:

        trend = "Neutral Trend Observation"
        trend_detail = "SMA 20 and SMA 50 are equal."

    st.markdown("### Trend")
    st.write(trend)
    st.info(trend_detail)

    # RSI Interpretation
    st.markdown("### RSI Interpretation")

    if rsi >= 70:

        rsi_detail = "RSI is in the traditionally overbought zone."

    elif rsi <= 30:

        rsi_detail = "RSI is in the traditionally oversold zone."

    else:

        rsi_detail = "RSI is in the middle range (30–70)."

    st.write(f"Current RSI: {rsi:.2f}")
    st.info(rsi_detail)

    # Historical chart
    st.subheader("Historical Market Chart")

    st.line_chart(df[["Close", "SMA20", "SMA50"]])

    # Recent market data
    st.subheader("Recent Market Data")

    st.dataframe(df.tail(10))

    # Disclaimer
    st.caption(
        "Educational market analysis only. "
        "Indicators are not guaranteed predictions "
        "or financial advice. No real trades are executed."
)
    
