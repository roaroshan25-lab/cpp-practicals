
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="MarketMind AI",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# 2. DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
h1 {
    color: #12345a;
}
h2, h3 {
    color: #234b70;
}
.stMetric {
    background: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e1e8f0;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("📈 MarketMind AI")
st.sidebar.caption("Stock Research Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Stock Scanner",
        "Market Analysis",
        "Risk Analysis",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Educational market research only."
)

# ==========================================
# 4. DATA DOWNLOAD
# ==========================================

@st.cache_data(ttl=900)
def get_stock_data(symbol, period):

    df = yf.download(
        symbol,
        period=period,
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

# ==========================================
# 5. RSI
# ==========================================

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

    return 100 - (100 / (1 + rs))

# ==========================================
# 6. STOCK ANALYSIS
# ==========================================

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
        "Momentum %": round(momentum, 2),
        "Volume Ratio": round(volume_ratio, 2)
            if pd.notna(volume_ratio) else np.nan,
        "Volatility %": round(volatility, 2),
        "Max Drawdown %": round(max_drawdown, 2),
        "Research Score": score,
        "Category": category,
        "Latest Data Date": df.index[-1].strftime("%d %b %Y")
    }

    return result, df

# ==========================================
# 7. COMMON HEADER
# ==========================================

st.title("📈 MarketMind AI")
st.caption("Stock Research & Market Analysis")

# ==========================================
# 8. HOME PAGE
# ==========================================

if page == "Home":

    st.header("Welcome to MarketMind AI")

    st.write(
        "Explore historical stock prices, technical indicators, "
        "market momentum and risk analysis in one place."
    )

    st.info(
        "An educational platform for understanding "
        "stock market behaviour."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Stock Scanner")
        st.write("Analyse multiple NSE stocks.")

    with col2:
        st.subheader("📈 Technical Analysis")
        st.write("Explore SMA, RSI and momentum.")

    with col3:
        st.subheader("🛡️ Risk Analysis")
        st.write("Study volatility and drawdowns.")

    st.divider()

    st.subheader("How to use MarketMind AI")

    st.markdown("""
    1. Open Stock Scanner.
    2. Enter NSE stock symbols.
    3. Choose a historical data period.
    4. Run the analysis.
    5. Explore the results and risk charts.
    """)

    st.caption(
        "Historical indicators do not guarantee future performance."
    )

# ==========================================
# 9. STOCK SCANNER PAGE
# ==========================================

elif page == "Stock Scanner":

    st.header("🔍 NSE Stock Scanner")

    default_stocks = (
        "RELIANCE.NS,TCS.NS,INFY.NS,"
        "HDFCBANK.NS,ICICIBANK.NS,SBIN.NS"
    )

    stock_input = st.text_area(
        "Enter stock symbols separated by commas",
        default_stocks
    )

    period = st.selectbox(
        "Historical data period",
        ["6mo", "1y", "2y", "5y"],
        index=2
    )

    if st.button("Run Stock Analysis"):

        symbols = list(dict.fromkeys(
            x.strip().upper()
            for x in stock_input.split(",")
            if x.strip()
        ))

        results = []
        chart_data = {}

        progress = st.progress(0)

        for i, symbol in enumerate(symbols):

            try:
                df = get_stock_data(symbol, period)

                if df is not None:

                    result, processed = analyse_stock(
                        symbol, df
                    )

                    results.append(result)
                    chart_data[symbol] = processed

                else:
                    st.warning(f"No sufficient data for {symbol}")

            except Exception:
                st.warning(f"Could not analyse {symbol}")

            progress.progress((i + 1) / len(symbols))

        if results:

            result_df = pd.DataFrame(results)

            result_df = result_df.sort_values(
                "Research Score",
                ascending=False
            )

            st.session_state["results"] = result_df
            st.session_state["chart_data"] = chart_data

            st.success("Analysis completed!")

        else:
            st.error("No stock data could be analysed.")

    if "results" in st.session_state:

        results = st.session_state["results"]

        st.subheader("Stock Research Results")

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV Report",
            csv,
            "marketmind_report.csv",
            "text/csv"
        )

        st.caption(
            "Research scores are rule-based indicators, "
            "not predictions or buy recommendations."
        )

# ==========================================
# 10. MARKET ANALYSIS PAGE
# ==========================================

elif page == "Market Analysis":

    st.header("📊 Market Analysis")

    if "chart_data" not in st.session_state:

        st.info(
            "First run Stock Analysis from the Stock Scanner page."
        )

    else:

        chart_data = st.session_state["chart_data"]

        selected = st.selectbox(
            "Select stock",
            list(chart_data.keys())
        )

        df = chart_data[selected]

        latest = df.iloc[-1]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Latest Price",
            f"₹{latest['Close']:,.2f}"
        )

        col2.metric(
            "RSI",
            f"{latest['RSI']:.2f}"
        )

        col3.metric(
            "20-Day Momentum",
            f"{latest['Momentum20']:.2f}%"
        )

        st.subheader("Historical Price and Moving Averages")

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(df.index, df["Close"], label="Close")
        ax.plot(df.index, df["SMA20"], label="SMA 20")
        ax.plot(df.index, df["SMA50"], label="SMA 50")

        ax.set_xlabel("Date")
        ax.set_ylabel("Adjusted Price (₹)")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("RSI History")

        st.line_chart(df["RSI"])

# ==========================================
# 11. RISK ANALYSIS PAGE
# ==========================================

elif page == "Risk Analysis":

    st.header("🛡️ Historical Risk Analysis")

    if "chart_data" not in st.session_state:

        st.info(
            "First run Stock Analysis from the Stock Scanner page."
        )

    else:

        chart_data = st.session_state["chart_data"]

        selected = st.selectbox(
            "Select stock for risk analysis",
            list(chart_data.keys())
        )

        df = chart_data[selected]

        daily_returns = df["Close"].pct_change().dropna()

        volatility = (
            daily_returns.tail(30).std()
            * np.sqrt(252) * 100
        )

        rolling_peak = df["Close"].cummax()

        drawdown = (
            df["Close"] / rolling_peak - 1
        ) * 100

        max_drawdown = drawdown.min()

        col1, col2 = st.columns(2)

        col1.metric(
            "Annualised Historical Volatility",
            f"{volatility:.2f}%"
        )

        col2.metric(
            "Maximum Historical Drawdown",
            f"{max_drawdown:.2f}%"
        )

        st.subheader("Historical Drawdown")

        st.line_chart(drawdown)

        st.caption(
            "Historical risk measurements are not guarantees "
            "of future risk."
        )

# ==========================================
# 12. ABOUT PAGE
# ==========================================

elif page == "About":

    st.header("About MarketMind AI")

    st.write("""
    MarketMind AI is an educational stock research
    and market analysis platform.

    It uses historical market data to demonstrate:

    - Moving averages
    - Relative Strength Index (RSI)
    - Price momentum
    - Volume analysis
    - Historical volatility
    - Maximum drawdown
    """)

    st.subheader("Data Source")

    st.write("Yahoo Finance historical market data.")

    st.subheader("Disclaimer")

    st.warning("""
    This platform is for educational purposes only.

    It does not provide personalised financial advice,
    guaranteed stock predictions or instructions to
    purchase securities.

    Historical performance does not guarantee future results.
    Market data may be delayed or incomplete.
    """)

# ==========================================
# 13. FOOTER
# ==========================================

st.divider()

st.caption(
    "© MarketMind AI | Educational Market Research"
)
