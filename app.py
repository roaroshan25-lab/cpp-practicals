
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="MarketMind AI",
    page_icon="📈",
    layout="wide"
)

# ==================================================
# 2. CUSTOM WEBSITE DESIGN
# ==================================================

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
}

h1 {
    color: #12345a;
}

h2, h3 {
    color: #234b70;
}

.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e1e8f0;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. WEBSITE HEADER
# ==================================================

st.title("📈 MarketMind AI")

st.subheader("Stock Research & Market Analysis")

st.write(
    "Explore historical stock prices, technical indicators, "
    "market momentum and risk analysis in one place."
)

st.info(
    "MarketMind AI is an educational research platform. "
    "It does not provide guaranteed predictions or "
    "personalised investment advice."
)

st.divider()

# ==================================================
# 4. PROFESSIONAL HOMEPAGE
# ==================================================

st.header("Welcome to MarketMind AI")

st.write(
    "Understand market behaviour using historical stock data "
    "and technical research tools."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Stock Scanner")
    st.write("Analyse multiple NSE stocks.")

with col2:
    st.markdown("### 📈 Technical Analysis")
    st.write("Explore SMA, RSI and momentum.")

with col3:
    st.markdown("### 🛡️ Risk Analysis")
    st.write("Study volatility and historical losses.")

st.divider()

st.subheader("Explore Our Features")

st.markdown("""
- **Stock Scanner:** Analyse selected NSE stocks.
- **Technical Indicators:** Study SMA 20, SMA 50 and RSI.
- **Momentum Analysis:** Explore historical price movement.
- **Risk Analysis:** Examine volatility and drawdowns.
- **Research Reports:** Download analysis as CSV files.
""")

st.divider()

# ==================================================
# 5. SIDEBAR SETTINGS
# ==================================================

st.sidebar.title("📈 MarketMind AI")

st.sidebar.subheader("Stock Scanner Settings")

default_stocks = (
    "RELIANCE.NS,TCS.NS,INFY.NS,"
    "HDFCBANK.NS,ICICIBANK.NS,SBIN.NS"
)

stock_input = st.sidebar.text_area(
    "Enter NSE stock symbols separated by commas",
    default_stocks
)

period = st.sidebar.selectbox(
    "Historical data period",
    ["6mo", "1y", "2y", "5y"],
    index=2
)

st.sidebar.caption(
    "Data is historical and may be delayed."
)

run_scan = st.sidebar.button(
    "🔍 Run Stock Analysis",
    use_container_width=True
)

# ==================================================
# 6. DOWNLOAD STOCK DATA
# ==================================================

@st.cache_data(ttl=900)
def get_stock_data(symbol, selected_period):

    df = yf.download(
        symbol,
        period=selected_period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Close", "High", "Low", "Volume"]

    if not all(col in df.columns for col in required):
        return None

    df = df[required].copy()

    df = df.dropna(subset=["Close"])

    if len(df) < 50:
        return None

    return df

# ==================================================
# 7. RSI CALCULATION
# ==================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi

# ==================================================
# 8. STOCK ANALYSIS
# ==================================================

def analyse_stock(symbol, df):

    df = df.copy()

    close = df["Close"]

    volume = df["Volume"]

    df["SMA20"] = close.rolling(20).mean()

    df["SMA50"] = close.rolling(50).mean()

    df["RSI"] = calculate_rsi(close)

    df["Daily_Return"] = close.pct_change()

    df["Momentum20"] = (
        close / close.shift(20) - 1
    ) * 100

    df["Volume_Average"] = volume.rolling(20).mean()

    latest = df.iloc[-1]

    price = float(latest["Close"])

    sma20 = float(latest["SMA20"])

    sma50 = float(latest["SMA50"])

    rsi = float(latest["RSI"])

    momentum = float(latest["Momentum20"])

    volume_average = float(latest["Volume_Average"])

    current_volume = float(latest["Volume"])

    volume_ratio = (
        current_volume / volume_average
        if volume_average > 0 else np.nan
    )

    volatility = (
        df["Daily_Return"].tail(30).std()
        * np.sqrt(252) * 100
    )

    rolling_peak = close.cummax()

    drawdown = (
        close / rolling_peak - 1
    ) * 100

    max_drawdown = float(drawdown.min())

    # Research score out of 100

    score = 0

    if price > sma20:
        score += 20

    if sma20 > sma50:
        score += 25

    if 50 <= rsi <= 70:
        score += 20

    if momentum > 0:
        score += 20

    if pd.notna(volume_ratio) and volume_ratio >= 1:
        score += 15

    if score >= 80:
        category = "Strong research conditions"

    elif score >= 60:
        category = "Positive research conditions"

    elif score >= 40:
        category = "Mixed conditions"

    else:
        category = "Weak research conditions"

    result = {
        "Stock": symbol,
        "Price": round(price, 2),
        "SMA20": round(sma20, 2),
        "SMA50": round(sma50, 2),
        "RSI": round(rsi, 2),
        "20D Momentum %": round(momentum, 2),
        "Volume Ratio": round(volume_ratio, 2)
            if pd.notna(volume_ratio) else np.nan,
        "Volatility %": round(volatility, 2),
        "Max Drawdown %": round(max_drawdown, 2),
        "Research Score": score,
        "Research Category": category,
        "Latest Data Date": df.index[-1].strftime("%d %b %Y")
    }

    return result, df

# ==================================================
# 9. RUN STOCK SCANNER
# ==================================================

if run_scan:

    symbols = list(dict.fromkeys(
        symbol.strip().upper()
        for symbol in stock_input.split(",")
        if symbol.strip()
    ))

    if not symbols:

        st.warning("Please enter stock symbols.")

    else:

        results = []

        chart_data = {}

        progress = st.progress(0)

        for index, symbol in enumerate(symbols):

            try:

                with st.spinner(f"Analysing {symbol}..."):

                    df = get_stock_data(symbol, period)

                    if df is not None:

                        analysis, processed_df = analyse_stock(
                            symbol, df
                        )

                        results.append(analysis)

                        chart_data[symbol] = processed_df

                    else:

                        st.warning(
                            f"Insufficient data for {symbol}."
                        )

            except Exception as error:

                st.warning(
                    f"Could not analyse {symbol}: {error}"
                )

            progress.progress(
                (index + 1) / len(symbols)
            )

        if results:

            result_df = pd.DataFrame(results)

            result_df = result_df.sort_values(
                "Research Score",
                ascending=False
            )

            st.session_state["results"] = result_df

            st.session_state["chart_data"] = chart_data

        else:

            st.error("No usable stock data was returned.")

# ==================================================
# 10. DISPLAY STOCK RESULTS
# ==================================================

if "results" in st.session_state:

    results = st.session_state["results"]

    chart_data = st.session_state["chart_data"]

    st.header("📊 Stock Research Results")

    st.write(
        "Stocks are ranked by the technical research score."
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Scores represent historical technical conditions. "
        "They are not forecasts or buy recommendations."
    )

    # ==================================================
    # 11. DOWNLOAD CSV REPORT
    # ==================================================

    csv = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Research Report",
        data=csv,
        file_name="marketmind_research_report.csv",
        mime="text/csv"
    )

    # ==================================================
    # 12. RESEARCH CATEGORY CHART
    # ==================================================

    st.subheader("Research Category Overview")

    counts = results["Research Category"].value_counts()

    st.bar_chart(counts)

    # ==================================================
    # 13. INDIVIDUAL STOCK ANALYSIS
    # ==================================================

    st.subheader("🔎 Individual Stock Analysis")

    selected_stock = st.selectbox(
        "Select a stock",
        results["Stock"].tolist()
    )

    selected_df = chart_data[selected_stock]

    latest = selected_df.iloc[-1]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Latest Price",
            f"₹{latest['Close']:,.2f}"
        )

    with col2:

        st.metric(
            "RSI",
            f"{latest['RSI']:.2f}"
        )

    with col3:

        st.metric(
            "20-Day Momentum",
            f"{latest['Momentum20']:.2f}%"
        )

    # ==================================================
    # 14. HISTORICAL PRICE CHART
    # ==================================================

    st.subheader("Historical Price Chart")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        selected_df.index,
        selected_df["Close"],
        label="Closing Price"
    )

    ax.plot(
        selected_df.index,
        selected_df["SMA20"],
        label="SMA 20"
    )

    ax.plot(
        selected_df.index,
        selected_df["SMA50"],
        label="SMA 50"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel("Adjusted Price (₹)")

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)

    plt.close(fig)

    # ==================================================
    # 15. HISTORICAL PERFORMANCE
    # ==================================================

    st.subheader("Historical Performance")

    daily_returns = (
        selected_df["Close"].pct_change().dropna()
    )

    cumulative_returns = (
        (1 + daily_returns).cumprod() - 1
    ) * 100

    st.line_chart(cumulative_returns)

    st.caption(
        "This chart shows historical cumulative returns, "
        "not expected future returns."
    )

    # ==================================================
    # 16. RISK ANALYSIS
    # ==================================================

    st.subheader("🛡️ Historical Risk Analysis")

    rolling_peak = selected_df["Close"].cummax()

    drawdown = (
        selected_df["Close"] / rolling_peak - 1
    ) * 100

    st.line_chart(drawdown)

    st.caption(
        "Drawdown shows the historical decline from "
        "a previous peak."
    )

# ==================================================
# 17. WEBSITE FOOTER
# ==================================================

st.divider()

st.markdown("""
### About MarketMind AI

MarketMind AI is an educational stock research
and market analysis platform.

It helps users explore historical prices,
technical indicators, momentum and risk.

**Disclaimer:** This platform does not provide
personalised financial advice, guaranteed predictions
or instructions to purchase securities.

Historical performance does not guarantee future results.
Market data may be delayed or incomplete.
""")

st.caption(
    "MarketMind AI | Educational Market Research | "
    + datetime.now().strftime("%Y")
)
