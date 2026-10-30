
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import smtplib
from email.message import EmailMessage

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Stock Research Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Research Dashboard")
st.caption(
    "Educational stock analysis using historical market data. "
    "Scores are research indicators, not predictions or buy recommendations."
)

# ---------------- SIDEBAR ----------------

st.sidebar.header("Scanner Settings")

default_stocks = (
    "RELIANCE.NS,TCS.NS,INFY.NS,"
    "HDFCBANK.NS,ICICIBANK.NS,SBIN.NS"
)

stock_input = st.sidebar.text_area(
    "Enter NSE symbols separated by commas",
    default_stocks
)

period = st.sidebar.selectbox(
    "Historical data period",
    ["6mo", "1y", "2y", "5y"],
    index=2
)

run_scan = st.sidebar.button("🔍 Run Stock Analysis")

# ---------------- DATA FUNCTIONS ----------------

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

    # Handle Yahoo Finance MultiIndex columns
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


def analyse_stock(symbol, df):

    close = df["Close"]
    volume = df["Volume"]

    df = df.copy()

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

    # Historical volatility, annualised
    volatility = (
        df["Daily_Return"].tail(30).std() * np.sqrt(252) * 100
    )

    # Historical maximum drawdown
    rolling_peak = close.cummax()
    drawdown = (close / rolling_peak - 1) * 100
    max_drawdown = float(drawdown.min())

    # Transparent research score: maximum 100 points
    score = 0

    if price > sma20:
        score += 20

    if sma20 > sma50:
        score += 25

    if 50 <= rsi <= 70:
        score += 20

    if momentum > 0:
        score += 20

    if volume_ratio >= 1:
        score += 15

    # Research category, not a trading instruction
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
        "Volume Ratio": round(volume_ratio, 2),
        "Volatility %": round(volatility, 2),
        "Max Drawdown %": round(max_drawdown, 2),
        "Research Score": score,
        "Research Category": category,
        "Latest Data Date": df.index[-1].strftime("%d %b %Y")
    }

    return result, df


# ---------------- EMAIL FUNCTION ----------------

def send_research_email(results):

    try:
        email_settings = st.secrets["email"]

        sender = email_settings["sender"]
        password = email_settings["app_password"]
        recipient = email_settings["recipient"]

        matching = results[
            results["Research Score"] >= 60
        ]

        if matching.empty:
            st.info("No stocks currently meet the alert threshold.")
            return

        message = EmailMessage()

        message["Subject"] = "Educational Stock Research Alert"
        message["From"] = sender
        message["To"] = recipient

        message.set_content(
            "Stocks meeting your educational research conditions:\n\n"
            + matching.to_string(index=False)
            + "\n\nThese are historical technical indicators, "
            "not buy recommendations or predictions."
        )

        with smtplib.SMTP_SSL(
            "smtp.gmail.com", 465
        ) as server:

            server.login(sender, password)
            server.send_message(message)

        st.success("Research notification email sent.")

    except Exception as error:
        st.error(
            "Email could not be sent. Check your email configuration "
            "in Streamlit Secrets."
        )
        st.caption(str(error))


# ---------------- MAIN DASHBOARD ----------------

if run_scan:

    symbols = list(dict.fromkeys(
        s.strip().upper()
        for s in stock_input.split(",")
        if s.strip()
    ))

    if not symbols:
        st.warning("Enter at least one stock symbol.")

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

            except Exception as error:
                st.warning(f"Could not analyse {symbol}: {error}")

            progress.progress((index + 1) / len(symbols))

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

# ---------------- DISPLAY RESULTS ----------------

if "results" in st.session_state:

    results = st.session_state["results"]
    chart_data = st.session_state["chart_data"]

    st.header("📊 Stock Research Results")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The research score is a simple, rule-based indicator. "
        "It has not been proven to predict future returns."
    )

    # Download report
    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Research Report",
        data=csv,
        file_name="stock_research_report.csv",
        mime="text/csv"
    )

    # Research category counts
    st.subheader("Research Category Overview")

    counts = results["Research Category"].value_counts()

    st.bar_chart(counts)

    # Select stock for detailed analysis
    st.subheader("🔎 Individual Stock Analysis")

    selected_stock = st.selectbox(
        "Select a stock",
        results["Stock"].tolist()
    )

    selected_df = chart_data[selected_stock]

    st.write(f"### {selected_stock} historical price")

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
    ax.set_ylabel("Adjusted Price")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

    # Historical returns
    st.subheader("Historical Performance")

    daily_returns = selected_df["Close"].pct_change().dropna()

    cumulative_returns = (
        (1 + daily_returns).cumprod() - 1
    ) * 100

    st.line_chart(cumulative_returns)

    # Email notification
    st.subheader("📧 Research Notifications")

    st.write(
        "You can manually send an email when stocks meet "
        "the research score threshold of 60."
    )

    if st.button("Send Research Alert Email"):

        send_research_email(results)

    # Last update
    st.caption(
        "Analysis generated: "
        + datetime.now().strftime("%d %b %Y, %I:%M %p")
    )

else:

    st.info(
        "Enter your stock symbols in the sidebar and click "
        "'Run Stock Analysis' to begin."
    )

# ---------------- DISCLAIMER ----------------

st.divider()

st.warning(
    "Educational tool only. A high research score does not mean "
    "a stock should be purchased. Historical performance does not "
    "guarantee future results. Yahoo Finance data may be delayed "
    "or incomplete. Verify information independently."
    )
        
