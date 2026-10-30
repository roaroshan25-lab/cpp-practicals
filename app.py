
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Trading Bot Dashboard")

st.markdown(
    "### Nifty 50 Market Analysis & Historical Backtesting"
)

st.caption(
    "An educational dashboard using historical market data "
    "and technical indicators."
)

# ==========================================
# 2. REFRESH MARKET DATA
# ==========================================

if st.button("🔄 Refresh Market Data"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 3. DOWNLOAD MARKET DATA
# ==========================================

@st.cache_data(ttl=300)
def load_data():

    df = yf.download(
        "^NSEI",
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()

    close = df["Close"]

    # Handle different Yahoo Finance column formats
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    result = pd.DataFrame(index=df.index)
    result["Close"] = close

    return result.dropna()


df = load_data()

if df.empty:
    st.error(
        "Unable to download market data. "
        "Please try refreshing later."
    )
    st.stop()

# ==========================================
# 4. CALCULATE TECHNICAL INDICATORS
# ==========================================

# Simple Moving Averages

df["SMA 20"] = df["Close"].rolling(window=20).mean()

df["SMA 50"] = df["Close"].rolling(window=50).mean()

# RSI Calculation

delta = df["Close"].diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

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

# ==========================================
# 5. LATEST MARKET INFORMATION
# ==========================================

latest = df.iloc[-1]

price = float(latest["Close"])

sma20 = float(latest["SMA 20"])

sma50 = float(latest["SMA 50"])

rsi = float(latest["RSI"])

st.caption(
    "Latest available data date: "
    + df.index[-1].strftime("%d %b %Y")
)

# ==========================================
# 6. MARKET OVERVIEW
# ==========================================

st.header("Market Overview")

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

# ==========================================
# 7. RSI ANALYSIS
# ==========================================

st.header("RSI Indicator")

st.metric(
    "Current RSI",
    f"{rsi:.2f}"
)

st.progress(
    min(max(rsi / 100, 0.0), 1.0)
)

if rsi >= 70:

    st.warning(
        "RSI is in the traditionally overbought zone."
    )

elif rsi <= 30:

    st.info(
        "RSI is in the traditionally oversold zone."
    )

else:

    st.success(
        "RSI is in the middle range (30–70)."
    )

# ==========================================
# 8. MARKET TREND ANALYSIS
# ==========================================

st.header("Market Trend Analysis")

if sma20 > sma50:

    st.write("**Trend observation: Upward**")

    st.info(
        "SMA 20 is above SMA 50."
    )

elif sma20 < sma50:

    st.write("**Trend observation: Downward**")

    st.info(
        "SMA 20 is below SMA 50."
    )

else:

    st.write("**Trend observation: Neutral**")

    st.info(
        "SMA 20 and SMA 50 are equal."
    )

# ==========================================
# 9. HISTORICAL PRICE CHART
# ==========================================

st.header("Nifty 50 Historical Chart")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df.index,
    df["Close"],
    label="Nifty 50 Price"
)

ax.plot(
    df.index,
    df["SMA 20"],
    label="SMA 20"
)

ax.plot(
    df.index,
    df["SMA 50"],
    label="SMA 50"
)

ax.set_xlabel("Date")

ax.set_ylabel("Price")

ax.legend()

ax.grid(True)

st.pyplot(fig)

plt.close(fig)

# ==========================================
# 10. RECENT MARKET DATA
# ==========================================

st.header("Recent Market Data")

st.dataframe(
    df.tail(10).sort_index(ascending=False),
    use_container_width=True
)

# ==========================================
# 11. HISTORICAL BACKTESTING
# ==========================================

st.header("Historical Strategy Backtesting")

backtest = df.dropna(
    subset=["SMA 20", "SMA 50"]
).copy()

# Daily market returns

backtest["Daily Returns"] = (
    backtest["Close"].pct_change()
)

# Educational strategy:
# Hold when SMA 20 is above SMA 50.
# Use the previous day's signal to avoid
# using today's closing signal for today's return.

backtest["Position"] = (
    (backtest["SMA 20"] > backtest["SMA 50"])
    .shift(1)
    .fillna(0)
)

# Strategy returns

backtest["Strategy Returns"] = (
    backtest["Position"] * backtest["Daily Returns"]
)

# Remove first row with missing daily return

backtest = backtest.dropna(
    subset=["Daily Returns"]
).copy()

if backtest.empty:
    st.error("Not enough data available for backtesting.")
    st.stop()

# ==========================================
# 12. CUMULATIVE PERFORMANCE
# ==========================================

backtest["Strategy Growth"] = (
    1 + backtest["Strategy Returns"]
).cumprod()

backtest["Buy & Hold Growth"] = (
    1 + backtest["Daily Returns"]
).cumprod()

# ==========================================
# 13. STRATEGY PERFORMANCE SUMMARY
# ==========================================

st.header("Strategy Performance Summary")

positive_days = int(
    (backtest["Strategy Returns"] > 0).sum()
)

negative_days = int(
    (backtest["Strategy Returns"] < 0).sum()
)

total_days = positive_days + negative_days

if total_days > 0:

    win_rate = (
        positive_days / total_days
    ) * 100

else:

    win_rate = 0

col1, col2, col3 = st.columns(3)

col1.metric(
    "Positive Days",
    positive_days
)

col2.metric(
    "Negative Days",
    negative_days
)

col3.metric(
    "Positive-Day Rate",
    f"{win_rate:.2f}%"
)

# ==========================================
# 14. STRATEGY VS BUY & HOLD
# ==========================================

strategy_return = (
    backtest["Strategy Growth"].iloc[-1] - 1
) * 100

buy_hold_return = (
    backtest["Buy & Hold Growth"].iloc[-1] - 1
) * 100

st.header("Performance Comparison")

col1, col2 = st.columns(2)

col1.metric(
    "SMA Strategy Return",
    f"{strategy_return:.2f}%"
)

col2.metric(
    "Buy & Hold Return",
    f"{buy_hold_return:.2f}%"
)

# ==========================================
# 15. BACKTESTING CHART
# ==========================================

st.subheader("Strategy Growth Comparison")

fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(
    backtest.index,
    backtest["Strategy Growth"],
    label="SMA Strategy"
)

ax2.plot(
    backtest.index,
    backtest["Buy & Hold Growth"],
    label="Buy & Hold"
)

ax2.set_title("Strategy vs Buy & Hold")

ax2.set_xlabel("Date")

ax2.set_ylabel("Growth of 1 Unit")

ax2.legend()

ax2.grid(True)

st.pyplot(fig2)

plt.close(fig2)

# ==========================================
# 16. MONTHLY PERFORMANCE
# ==========================================

st.header("Monthly Strategy Performance")

monthly_returns = (
    (1 + backtest["Strategy Returns"])
    .resample("ME")
    .prod()
    - 1
) * 100

monthly_returns = monthly_returns.dropna()

if not monthly_returns.empty:

    fig3, ax3 = plt.subplots(figsize=(12, 5))

    ax3.bar(
        monthly_returns.index.strftime("%b %Y"),
        monthly_returns.values
    )

    ax3.set_title("Monthly Strategy Returns")

    ax3.set_xlabel("Month")

    ax3.set_ylabel("Return (%)")

    ax3.tick_params(
        axis="x",
        rotation=45
    )

    ax3.axhline(
        0,
        linewidth=1
    )

    ax3.grid(
        axis="y",
        alpha=0.3
    )

    st.pyplot(fig3)

    plt.close(fig3)

    st.dataframe(
        monthly_returns.rename(
            "Monthly Return (%)"
        ).round(2),
        use_container_width=True
    )

else:

    st.info("Monthly performance data is unavailable.")

# ==========================================
# 17. MAXIMUM DRAWDOWN
# ==========================================

st.header("Risk Analysis")

running_peak = (
    backtest["Strategy Growth"].cummax()
)

backtest["Drawdown"] = (
    backtest["Strategy Growth"] / running_peak
) - 1

max_drawdown = (
    backtest["Drawdown"].min()
) * 100

st.metric(
    "Maximum Strategy Drawdown",
    f"{max_drawdown:.2f}%"
)

fig4, ax4 = plt.subplots(figsize=(12, 4))

ax4.plot(
    backtest.index,
    backtest["Drawdown"] * 100,
    label="Strategy Drawdown"
)

ax4.set_title("Historical Strategy Drawdown")

ax4.set_xlabel("Date")

ax4.set_ylabel("Drawdown (%)")

ax4.legend()

ax4.grid(True)

st.pyplot(fig4)

plt.close(fig4)

# ==========================================
# 18. DOWNLOAD DATA AS CSV
# ==========================================

st.header("Download Historical Analysis")

csv_data = backtest.to_csv().encode("utf-8")

st.download_button(
    label="⬇️ Download Backtesting Data (CSV)",
    data=csv_data,
    file_name="nifty50_backtesting_analysis.csv",
    mime="text/csv"
)

# ==========================================
# 19. DISCLAIMER
# ==========================================

st.divider()

st.subheader("Important Disclaimer")

st.caption(
    "This is an educational market-analysis project. "
    "Yahoo Finance data may be delayed and is not guaranteed "
    "to be real-time. Historical backtesting is simulated "
    "and excludes transaction costs, taxes and slippage. "
    "Past performance does not predict future results. "
    "This dashboard does not provide financial advice "
    "or execute real trades."
)
