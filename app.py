import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="MarketMind AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. RESPONSIVE WEBSITE DESIGN
# =========================================================
st.markdown("""
<style>
.stApp { background-color: #f3f6fc; }
[data-testid="stSidebar"] { background-color: #10243e; }
[data-testid="stSidebar"] * { color: #ffffff; }
.block-container { padding-top: 1.5rem; max-width: 1250px; }
h1 { color: #12345a; font-weight: 800; }
h2, h3 { color: #234b70; }
.stMetric {
    background: #ffffff;
    padding: 13px;
    border-radius: 12px;
    border: 1px solid #dce5f0;
}
.stButton button, .stDownloadButton button {
    border-radius: 8px;
    min-height: 42px;
}
div[data-testid="stDataFrame"] { width: 100%; }
@media (max-width: 700px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    h1 { font-size: 1.8rem; }
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. CONSTANTS AND SESSION STATE
# =========================================================
DEFAULT_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"
]
PERIOD_OPTIONS = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = DEFAULT_STOCKS.copy()

if "scanner_results" not in st.session_state:
    st.session_state["scanner_results"] = None

if "chart_data" not in st.session_state:
    st.session_state["chart_data"] = {}

# =========================================================
# 4. SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("📈 MarketMind AI")
st.sidebar.caption("Stock Research Platform")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Stock Scanner",
        "Market Analysis",
        "Risk Analysis",
        "Stock Comparison",
        "Watchlist",
        "About"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Educational historical market research only.")

# =========================================================
# 5. DATA DOWNLOAD AND VALIDATION
# =========================================================
@st.cache_data(ttl=900, show_spinner=False)
def get_stock_data(symbol, period):
    """Download and validate daily historical data from Yahoo Finance."""
    try:
        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )
        if df is None or df.empty:
            return None

        # yfinance may return MultiIndex columns for one ticker too.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Close", "High", "Low", "Volume"]
        if not all(column in df.columns for column in required):
            return None

        df = df[required].copy()
        for column in required:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=["Close"]).sort_index()
        if len(df) < 2:
            return None

        return df
    except Exception:
        return None

# =========================================================
# 6. TECHNICAL INDICATORS
# =========================================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # A series with no losses has RSI 100 after the warm-up period.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    # A completely unchanged series is assigned neutral RSI.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)
    return rsi

def add_indicators(df):
    """Return a copy with historical indicators and risk measures."""
    data = df.copy()
    close = data["Close"]

    data["SMA20"] = close.rolling(20, min_periods=20).mean()
    data["SMA50"] = close.rolling(50, min_periods=50).mean()
    data["RSI"] = calculate_rsi(close)
    data["Daily_Return"] = close.pct_change()
    data["Momentum20"] = (close / close.shift(20) - 1) * 100
    data["Volume_Average"] = data["Volume"].rolling(
        20, min_periods=5
    ).mean()
    data["Drawdown"] = (close / close.cummax() - 1) * 100
    return data

def analyse_stock(symbol, raw_df):
    data = add_indicators(raw_df)
    latest = data.iloc[-1]

    price = float(latest["Close"])
    sma20 = latest["SMA20"]
    sma50 = latest["SMA50"]
    rsi = latest["RSI"]
    momentum = latest["Momentum20"]
    volume_average = latest["Volume_Average"]
    current_volume = latest["Volume"]

    volume_ratio = np.nan
    if pd.notna(volume_average) and volume_average > 0:
        volume_ratio = float(current_volume / volume_average)

    daily_returns = data["Daily_Return"].dropna()
    volatility = (
        daily_returns.tail(30).std() * np.sqrt(252) * 100
        if len(daily_returns) >= 2 else np.nan
    )
    max_drawdown = float(data["Drawdown"].min())

    # Score only indicators that are available; missing values do not
    # accidentally become positive signals.
    score = 0
    if pd.notna(sma20) and price > sma20:
        score += 20
    if pd.notna(sma20) and pd.notna(sma50) and sma20 > sma50:
        score += 25
    if pd.notna(rsi) and 50 <= rsi <= 70:
        score += 20
    if pd.notna(momentum) and momentum > 0:
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

    def safe_round(value, digits=2):
        return round(float(value), digits) if pd.notna(value) else np.nan

    result = {
        "Stock": symbol,
        "Price (₹)": safe_round(price),
        "SMA20": safe_round(sma20),
        "SMA50": safe_round(sma50),
        "RSI": safe_round(rsi),
        "Momentum %": safe_round(momentum),
        "Volume Ratio": safe_round(volume_ratio),
        "Volatility %": safe_round(volatility),
        "Max Drawdown %": safe_round(max_drawdown),
        "Research Score": score,
        "Category": category,
        "Latest Data Date": data.index[-1].strftime("%d %b %Y")
    }
    return result, data

def clean_symbols(text):
    return list(dict.fromkeys(
        item.strip().upper()
        for item in text.split(",")
        if item.strip()
    ))

def format_symbol_help():
    st.caption(
        "For NSE-listed shares, use Yahoo Finance symbols such as "
        "`RELIANCE.NS`, `TCS.NS` and `INFY.NS`."
    )

# =========================================================
# 7. WEBSITE HEADER
# =========================================================
st.title("📈 MarketMind AI")
st.caption("Historical Stock Research • Technical Indicators • Risk Analysis")
st.info(
    "MarketMind AI is an educational research tool. Data may be delayed, "
    "incomplete or unavailable. Indicators describe historical data and "
    "are not predictions or investment recommendations."
)

# =========================================================
# 8. HOME PAGE
# =========================================================
if page == "Home":
    st.header("Welcome to MarketMind AI")
    st.write(
        "Explore historical stock prices, technical indicators, market "
        "momentum and risk measurements in one place."
    )

    st.subheader("📊 Nifty 50 Historical Overview")
    try:
        nifty = get_stock_data("^NSEI", "1y")
        if nifty is not None:
            latest_price = float(nifty["Close"].iloc[-1])
            previous_price = float(nifty["Close"].iloc[-2])
            daily_change = (
                (latest_price - previous_price) / previous_price * 100
                if previous_price else np.nan
            )
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Nifty 50 latest close", f"{latest_price:,.2f}",
                        f"{daily_change:.2f}%")
            col2.metric("52-week high", f"{nifty['High'].max():,.2f}")
            col3.metric("52-week low", f"{nifty['Low'].min():,.2f}")
            col4.metric("Trading sessions", f"{len(nifty):,}")
            st.caption(
                "Latest available daily data: "
                + nifty.index[-1].strftime("%d %b %Y")
            )
            st.line_chart(nifty["Close"], use_container_width=True)
        else:
            st.warning("Nifty 50 data is currently unavailable.")
    except Exception:
        st.warning("Could not load the Nifty 50 historical overview.")

    st.divider()
    st.subheader("Explore the platform")
    cards = st.columns(3)
    card_content = [
        ("🔍 Stock Scanner", "Review historical indicators for several symbols."),
        ("📈 Market Analysis", "Study closing prices, moving averages and RSI."),
        ("🛡️ Risk Analysis", "Explore historical volatility and drawdowns."),
        ("⚖️ Stock Comparison", "Compare two stocks on a normalised chart."),
        ("⭐ Watchlist", "Keep a session-based list of symbols to research."),
        ("📚 About", "Read about the calculations and data limitations.")
    ]
    for index, (heading, description) in enumerate(card_content):
        with cards[index % 3]:
            with st.container(border=True):
                st.markdown(f"**{heading}**")
                st.write(description)

    st.subheader("How to use")
    st.markdown(
        "1. Add symbols to your Watchlist or open Stock Scanner.\n"
        "2. Choose a historical period.\n"
        "3. Run the analysis and review the results.\n"
        "4. Use Stock Comparison to compare historical percentage changes.\n"
        "5. Explore Market Analysis and Risk Analysis."
    )

# =========================================================
# 9. STOCK SCANNER
# =========================================================
elif page == "Stock Scanner":
    st.header("🔍 NSE Stock Scanner")
    default_text = ",".join(DEFAULT_STOCKS)
    stock_input = st.text_area(
        "Enter stock symbols separated by commas",
        value=default_text,
        height=90
    )
    format_symbol_help()
    period = st.selectbox(
        "Historical data period", ["6mo", "1y", "2y", "5y"], index=2
    )

    if st.button("Run Stock Analysis", type="primary", use_container_width=True):
        symbols = clean_symbols(stock_input)
        if not symbols:
            st.error("Enter at least one valid stock symbol.")
        elif len(symbols) > 30:
            st.error("Please scan no more than 30 symbols at a time.")
        else:
            results = []
            chart_data = {}
            progress = st.progress(0)
            status = st.empty()

            for i, symbol in enumerate(symbols):
                status.write(f"Checking {symbol} ({i + 1}/{len(symbols)})...")
                raw = get_stock_data(symbol, period)
                if raw is not None and len(raw) >= 50:
                    try:
                        result, processed = analyse_stock(symbol, raw)
                        results.append(result)
                        chart_data[symbol] = processed
                    except Exception:
                        st.warning(f"Could not calculate indicators for {symbol}.")
                else:
                    st.warning(
                        f"No usable data or fewer than 50 trading sessions for {symbol}."
                    )
                progress.progress((i + 1) / len(symbols))

            status.empty()
            if results:
                result_df = pd.DataFrame(results).sort_values(
                    "Research Score", ascending=False
                )
                st.session_state["scanner_results"] = result_df
                st.session_state["chart_data"] = chart_data
                st.success(f"Completed analysis for {len(results)} stock(s).")
                else:
                st.error(
                    "No stock could be analysed. Check the symbols and try again."
                )

    if st.session_state["scanner_results"] is not None:
        results = st.session_state["scanner_results"]
        st.subheader("Historical Research Results")
        st.dataframe(
            results, use_container_width=True, hide_index=True
        )
        csv_data = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV Report",
            data=csv_data,
            file_name="marketmind_historical_report.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(
            "Research scores are simple rule-based summaries of historical "
            "indicators. They are not AI predictions or buy/sell signals."
        )

# =========================================================
# 10. MARKET ANALYSIS
# =========================================================
elif page == "Market Analysis":
    st.header("📊 Market Analysis")
    chart_data = st.session_state.get("chart_data", {})

    if not chart_data:
        st.info("Run Stock Analysis on the Stock Scanner page first.")
    else:
        selected = st.selectbox("Select a scanned stock", list(chart_data.keys()))
        data = chart_data[selected]
        latest = data.iloc[-1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Latest close", f"₹{latest['Close']:,.2f}")
        c2.metric(
            "RSI",
            f"{latest['RSI']:.2f}" if pd.notna(latest["RSI"]) else "Not available"
        )
        c3.metric(
            "20-day momentum",
            f"{latest['Momentum20']:.2f}%"
            if pd.notna(latest["Momentum20"]) else "Not available"
        )

        st.subheader("Historical Close and Moving Averages")
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(data.index, data["Close"], label="Close", linewidth=1.8)
        ax.plot(data.index, data["SMA20"], label="SMA 20")
        ax.plot(data.index, data["SMA50"], label="SMA 50")
        ax.set_xlabel("Date")
        ax.set_ylabel("Adjusted price (₹)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Relative Strength Index (RSI)")
        st.line_chart(data["RSI"], use_container_width=True)
        st.caption(
            "RSI is a historical momentum indicator. Common reference levels "
            "are 30 and 70; they are not automatic decision rules."
        )

        st.subheader("Historical price data")
        st.dataframe(
            data[["Close", "SMA20", "SMA50", "RSI", "Momentum20"]].tail(20),
            use_container_width=True
        )

# =========================================================
# 11. RISK ANALYSIS
# =========================================================
elif page == "Risk Analysis":
    st.header("🛡️ Historical Risk Analysis")
    chart_data = st.session_state.get("chart_data", {})

    if not chart_data:
        st.info("Run Stock Analysis on the Stock Scanner page first.")
    else:
        selected = st.selectbox(
            "Select stock for risk analysis", list(chart_data.keys())
        )
        data = chart_data[selected]
        daily_returns = data["Close"].pct_change().dropna()
        volatility = (
            daily_returns.tail(30).std() * np.sqrt(252) * 100
            if len(daily_returns) >= 2 else np.nan
        )
        drawdown = (data["Close"] / data["Close"].cummax() - 1) * 100
        max_drawdown = float(drawdown.min())

        col1, col2 = st.columns(2)
        col1.metric(
            "Annualised historical volatility",
            f"{volatility:.2f}%" if pd.notna(volatility) else "Not available"
        )
        col2.metric("Maximum historical drawdown", f"{max_drawdown:.2f}%")

        st.subheader("Historical Drawdown")
        st.line_chart(drawdown, use_container_width=True)
        st.caption(
            "Volatility estimates the variability of historical daily returns. "
            "Drawdown measures decline from a previous peak. Neither guarantees "
            "future risk or performance."
        )

# =========================================================
# 12. STOCK COMPARISON
# =========================================================
elif page == "Stock Comparison":
    st.header("⚖️ Stock Comparison")
    st.write(
        "Compare two symbols using historical closing prices, normalised "
        "performance, RSI, volatility and maximum drawdown."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        symbol_a = st.text_input("First stock symbol", "RELIANCE.NS").strip().upper()
    with col_b:
        symbol_b = st.text_input("Second stock symbol", "TCS.NS").strip().upper()

    format_symbol_help()
    compare_period = st.selectbox(
        "Comparison period", ["3mo", "6mo", "1y", "2y", "5y"], index=2
    )

    if st.button("Compare Stocks", type="primary", use_container_width=True):
        if not symbol_a or not symbol_b:
            st.error("Enter both stock symbols.")
        elif symbol_a == symbol_b:
            st.warning("Enter two different stock symbols.")
        else:
            with st.spinner("Downloading historical data..."):
                raw_a = get_stock_data(symbol_a, compare_period)
                raw_b = get_stock_data(symbol_b, compare_period)

            if raw_a is None:
                st.error(f"Could not retrieve data for {symbol_a}.")
            elif raw_b is None:
                st.error(f"Could not retrieve data for {symbol_b}.")
            else:
                # Align dates so the performance comparison uses shared sessions.
                closes = pd.concat(
                    [
                        raw_a["Close"].rename(symbol_a),
                        raw_b["Close"].rename(symbol_b)
                    ],
                    axis=1
                ).dropna()

                if len(closes) < 2 or (closes.iloc[0] == 0).any():
                    st.error("There is not enough overlapping data to compare.")
                else:
                    normalised = closes.divide(closes.iloc[0]).multiply(100)
                    st.subheader("Normalised Historical Performance")
                    st.caption(
                        "Both series start at 100 on the first shared trading date. "
                        "This compares percentage movement, not share prices."
                    )
                    st.line_chart(normalised, use_container_width=True)

                    returns = (closes.iloc[-1] / closes.iloc[0] - 1) * 100
                    metrics = []
                    for symbol, raw in [(symbol_a, raw_a), (symbol_b, raw_b)]:
                        processed = add_indicators(raw)
                        daily = processed["Close"].pct_change().dropna()
                        vol = (
                            daily.tail(30).std() * np.sqrt(252) * 100
                            if len(daily) >= 2 else np.nan
                        )
                        dd = (
                            processed["Close"] / processed["Close"].cummax() - 1
                        ) * 100
                        metrics.append({
                            "Stock": symbol,
                            "Latest Close (₹)": round(float(raw["Close"].iloc[-1]), 2),
                            "Period Return %": round(float(returns[symbol]), 2),
                            "Latest RSI": (
                                round(float(processed["RSI"].iloc[-1]), 2)
                                if pd.notna(processed["RSI"].iloc[-1]) else np.nan
                            ),
                            "Annualised Volatility %": (
                                round(float(vol), 2) if pd.notna(vol) else np.nan
                            ),
                            "Maximum Drawdown %": round(float(dd.min()), 2)
                        })

                    st.subheader("Comparison Summary")
                    st.dataframe(
                        pd.DataFrame(metrics),
                        use_container_width=True,
                        hide_index=True
                    )
                    st.caption(
                        "The period return uses the first and last shared dates. "
                        "Historical comparisons do not indicate which stock will "
                        "perform better in the future."
                    )

# =========================================================
# 13. WATCHLIST
# =========================================================
elif page == "Watchlist":
    st.header("⭐ My Research Watchlist")
    st.write(
        "Keep a list of symbols to research during this browser session. "
        "The list is session-based and may reset when the session ends."
    )

    new_symbol = st.text_input(
        "Add a stock symbol", placeholder="Example: ITC.NS"
    ).strip().upper()

    if st.button("Add to Watchlist", use_container_width=True):
        if not new_symbol:
            st.warning("Enter a stock symbol first.")
        elif new_symbol in st.session_state["watchlist"]:
            st.info("That symbol is already in your watchlist.")
        else:
            st.session_state["watchlist"].append(new_symbol)
            st.success(f"Added {new_symbol} to your watchlist.")

    watchlist = st.session_state["watchlist"]
    if watchlist:
        st.subheader("Saved symbols")
        st.write(", ".join(watchlist))

        remove_symbol = st.selectbox(
            "Choose a symbol to remove", ["Select a symbol"] + watchlist
        )
        if st.button("Remove Selected Symbol", use_container_width=True):
            if remove_symbol == "Select a symbol":
                st.warning("Choose a symbol to remove.")
            else:
                st.session_state["watchlist"].remove(remove_symbol)
                st.success(f"Removed {remove_symbol}.")
                st.rerun()

        watch_period = st.selectbox(
            "Watchlist data period", ["1mo", "3mo", "6mo", "1y"], index=2
        )
        if st.button("Load Watchlist Overview", type="primary",
                     use_container_width=True):
            rows = []
            progress = st.progress(0)
            for i, symbol in enumerate(watchlist):
                raw = get_stock_data(symbol, watch_period)
                if raw is not None:
                    last = float(raw["Close"].iloc[-1])
                    previous = float(raw["Close"].iloc[-2])
                    change = (
                        (last / previous - 1) * 100 if previous else np.nan
                    )
                    rows.append({
                        "Stock": symbol,
                        "Latest Close (₹)": round(last, 2),
                        "Latest Daily Change %": (
                            round(change, 2) if pd.notna(change) else np.nan
                        ),
                        "Latest Data Date": raw.index[-1].strftime("%d %b %Y")
                    })
                else:
                    st.warning(f"Data unavailable for {symbol}.")
                progress.progress((i + 1) / len(watchlist))

            if rows:
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No watchlist data could be loaded.")
    else:
        st.info("Your watchlist is empty. Add a symbol above.")

# =========================================================
# 14. ABOUT, DOCUMENTATION AND DISCLAIMER
# =========================================================
elif page == "About":
    st.header("About MarketMind AI")
    st.write(
        "MarketMind AI is an educational stock research and market analysis "
        "platform. It uses historical market data to demonstrate common "
        "financial data analysis techniques."
    )

    st.subheader("Platform features")
    st.markdown(
        "- **Home:** Nifty 50 historical overview and platform guide.\n"
        "- **Stock Scanner:** Rule-based summaries for multiple symbols.\n"
        "- **Market Analysis:** Closing price, SMA20, SMA50 and RSI charts.\n"
        "- **Risk Analysis:** Historical volatility and drawdown.\n"
        "- **Stock Comparison:** Normalised historical performance comparison.\n"
        "- **Watchlist:** Session-based saved stock symbols."
                )
                    st.subheader("How the indicators are calculated")
    st.markdown(
        "- **SMA 20 / SMA 50:** Average closing price over 20 / 50 sessions.\n"
        "- **RSI:** Momentum indicator calculated from average gains and losses.\n"
        "- **20-day momentum:** Percentage price change over 20 sessions.\n"
        "- **Volatility:** Standard deviation of recent daily returns, annualised "
        "using √252.\n"
        "- **Maximum drawdown:** Largest percentage fall from a previous "
        "historical peak in the selected data."
    )

    st.subheader("Data source and limitations")
    st.write(
        "Historical data is requested from Yahoo Finance through yfinance. "
        "Availability, accuracy and timing depend on the data provider. "
        "This application is not an exchange data feed."
    )

    st.subheader("Deployment checklist")
    st.markdown(
        "1. Save this file in your GitHub repository as `app.py`.\n"
        "2. Ensure `requirements.txt` contains the libraries listed below.\n"
        "3. Commit the changes to GitHub.\n"
        "4. Open your existing Streamlit Community Cloud app and wait for "
        "the deployment to finish.\n"
        "5. Test each page on both desktop and mobile."
    )

    st.code(
        "streamlit\\nyfinance\\npandas\\nnumpy\\nmatplotlib",
        language="text"
    )

    st.warning(
        "Educational use only. This application does not provide personalised "
        "financial advice, guaranteed predictions, or instructions to buy or "
        "sell securities. It does not place trades or connect to a brokerage. "
        "Historical performance does not guarantee future results."
    )

# =========================================================
# 15. FOOTER
# =========================================================
st.divider()
st.caption(
    "© MarketMind AI | Educational Historical Market Research | "
    "Data availability may vary."
        )
    
