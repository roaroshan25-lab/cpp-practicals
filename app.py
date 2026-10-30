
import streamlit as st
import yfinance as yf
import pandas as pd

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Trading Bot")
st.caption("Nifty 50 Market Analysis Dashboard")

# --------------------------------
# REFRESH BUTTON
# --------------------------------

if st.button("🔄 Refresh Market Data"):
    st.cache_data.clear()
    st.rerun()

# --------------------------------
# DOWNLOAD REAL MARKET DATA
# --------------------------------

@st.cache_data(ttl=300)
def get_data():

    data = yf.download(
        "^NSEI",
        period="2y",
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

    # --------------------------------
    # SMA CALCULATION
    # --------------------------------

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    # --------------------------------
    # RSI CALCULATION
    # --------------------------------

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


# --------------------------------
# LOAD DATA
# --------------------------------

df = get_data()

if df.empty or df["RSI"].dropna().empty:

    st.error(
        "Market data unavailable. "
        "Please try refreshing later."
    )

else:

    latest = df.iloc[-1]

    price = latest["Close"]
    sma20 = latest["SMA20"]
    sma50 = latest["SMA50"]
    rsi = latest["RSI"]

    # Latest available data date
    st.caption(
        "Latest available data date: "
        + df.index[-1].strftime("%d %b %Y")
    )

    # --------------------------------
    # MARKET OVERVIEW
    # --------------------------------

    st.subheader("Market Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Nifty 50 Price",
        f"{price:,.2f}"
    )

    col2.metric(
        "SMA 20",
        f"{sma20:,.2f}"
    )

    col3.metric(
        "SMA 50",
        f"{sma50:,.2f}"
    )

    # --------------------------------
    # RSI
    # --------------------------------

    st.subheader("RSI Indicator")

    st.metric(
        "Current RSI",
        f"{rsi:.2f}"
    )

    st.progress(
        min(max(float(rsi) / 100, 0.0), 1.0)
    )

    # --------------------------------
    # MARKET ANALYSIS
    # --------------------------------

    st.subheader("Market Analysis")

    if sma20 > sma50:

        trend = "Upward Trend Observation"
        trend_detail = (
            "SMA 20 is above SMA 50."
        )

    elif sma20 < sma50:

        trend = "Downward Trend Observation"
        trend_detail = (
            "SMA 20 is below SMA 50."
        )

    else:

        trend = "Neutral Trend Observation"
        trend_detail = (
            "SMA 20 and SMA 50 are equal."
        )

    st.markdown("### Trend")

    st.write(trend)
    st.info(trend_detail)

    # RSI interpretation

    st.markdown("### RSI Interpretation")

    if rsi >= 70:

        rsi_detail = (
            "RSI is in the traditionally "
            "overbought zone."
        )

    elif rsi <= 30:

        rsi_detail = (
            "RSI is in the traditionally "
            "oversold zone."
        )

    else:

        rsi_detail = (
            "RSI is in the middle range (30–70)."
        )

    st.write(f"Current RSI: {rsi:.2f}")
    st.info(rsi_detail)

    # --------------------------------
    # HISTORICAL MARKET CHART
    # --------------------------------

    st.subheader("Historical Market Chart")

    st.line_chart(
        df[["Close", "SMA20", "SMA50"]]
    )

    # --------------------------------
    # RECENT MARKET DATA
    # --------------------------------

    st.subheader("Recent Market Data")

    st.dataframe(
        df.tail(10),
        use_container_width=True
    )

    # --------------------------------
    # HISTORICAL BACKTESTING
    # --------------------------------

    st.subheader(
        "Historical Strategy Backtesting"
    )

    st.write(
        "Educational simulation using historical "
        "Nifty 50 prices and moving averages."
    )

    backtest = df.dropna(
        subset=["SMA20", "SMA50"]
    ).copy()

    # Simulated position
    backtest["Position"] = (
        backtest["SMA20"] > backtest["SMA50"]
    ).astype(int)

    # Use previous day's signal
    backtest["Position"] = (
        backtest["Position"].shift(1).fillna(0)
    )

    # Daily market return
    backtest["Market Return"] = (
        backtest["Close"].pct_change().fillna(0)
    )

    # Strategy return
    backtest["Strategy Return"] = (
        backtest["Position"]
        * backtest["Market Return"]
    )

    # Cumulative growth
    backtest["Strategy Growth"] = (
        1 + backtest["Strategy Return"]
    ).cumprod()

    backtest["Buy and Hold Growth"] = (
        1 + backtest["Market Return"]
    ).cumprod()

    strategy_return = (
        backtest["Strategy Growth"].iloc[-1] - 1
    ) * 100

    market_return = (
        backtest["Buy and Hold Growth"].iloc[-1] - 1
    ) * 100

    col1, col2 = st.columns(2)

    col1.metric(
        "Simulated Strategy Return",
        f"{strategy_return:.2f}%"
    )

    col2.metric(
        "Buy & Hold Return",
        f"{market_return:.2f}%"
    )

    st.subheader("Strategy vs Buy & Hold")

    st.line_chart(
        backtest[
            ["Strategy Growth", "Buy and Hold Growth"]
        ]
    )

    # --------------------------------
    # DISCLAIMER
    # --------------------------------

    st.caption(
        "Data provided by Yahoo Finance. "
        "Daily market data may be delayed and is not "
        "guaranteed to be real-time. "
        "Historical backtesting is simulated and excludes "
        "transaction costs, taxes and slippage. "
        "Educational analysis only. No real trades are executed."
    )

