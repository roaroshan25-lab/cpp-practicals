
import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="📈",
    layout="wide"
)

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

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

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

df = get_data()

if df.empty or df["RSI"].dropna().empty:
    st.error("Market data unavailable. Please refresh later.")

else:
    latest = df.iloc[-1]

    price = latest["Close"]
    sma20 = latest["SMA20"]
    sma50 = latest["SMA50"]
    rsi = latest["RSI"]

    st.subheader("Market Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Nifty 50 Price", f"{price:,.2f}")
    col2.metric("SMA 20", f"{sma20:,.2f}")
    col3.metric("SMA 50", f"{sma50:,.2f}")

    st.subheader("RSI Indicator")
    st.metric("Current RSI", f"{rsi:.2f}")

    if sma20 > sma50 and 30 < rsi < 70:
        signal = "BUY CANDIDATE"
    elif sma20 < sma50 and 30 < rsi < 70:
        signal = "SELL CANDIDATE"
    else:
        signal = "HOLD"

    st.subheader("Market Analysis")

    if signal == "BUY CANDIDATE":
        st.success(signal)
    elif signal == "SELL CANDIDATE":
        st.warning(signal)
    else:
        st.info(signal)

    st.subheader("Historical Market Chart")

    st.line_chart(
        df[["Close", "SMA20", "SMA50"]]
    )

    st.subheader("Recent Market Data")
    st.dataframe(df.tail(10))

    st.caption(
        "Educational analysis only. "
        "Signals are not financial advice or guaranteed predictions. "
        "No real trades are placed."
    )
