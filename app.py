import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import requests
from streamlit_autorefresh import st_autorefresh

from src.ocr import extract_text
from src.preprocess import clean_text
from src.sentiment import get_sentiment
from src.features import compute_fear_index, create_feature_vector
from src.predict import predict

# ==============================
# ⚙️ CONFIG
# ==============================
st.set_page_config(page_title="Trading Dashboard", layout="wide")

# 🔄 AUTO REFRESH (5 sec)
st_autorefresh(interval=5000, limit=None, key="refresh")

# 🎨 CLEAN TITLE
st.markdown("<h1 style='text-align:center;'>📊 AI Trading Dashboard</h1>", unsafe_allow_html=True)

# ==============================
# 📰 NEWS FUNCTION
# ==============================
def get_live_news(ticker):
    API_KEY = "537d10b9f1d640c79152dd147c2d420b"

    query_map = {
        "AAPL": "Apple stock",
        "TSLA": "Tesla stock",
        "GOOGL": "Google stock",
        "MSFT": "Microsoft stock"
    }

    query = query_map.get(ticker, f"{ticker} stock")
    url = f"https://newsapi.org/v2/everything?q={requests.utils.requote_uri(query)}&apiKey={API_KEY}"

    res = requests.get(url)
    data = res.json()

    articles = []
    for a in data.get("articles", [])[:5]:
        articles.append(a["title"])

    return articles

# ==============================
# � WATCHLIST
# ==============================
st.sidebar.title("📊 Watchlist")

watchlist = ["AAPL", "TSLA", "GOOGL", "MSFT"]

ticker_to_company = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "GOOGL": "Google",
    "MSFT": "Microsoft"
}

selected_stock = st.sidebar.selectbox("Select Stock", watchlist)

custom_stock = st.text_input("Or Enter Custom Stock")

ticker = custom_stock.upper() if custom_stock else selected_stock
company_name = ticker_to_company.get(ticker, ticker)

# ==============================
# 📈 INPUTS
# ==============================
timeframe = st.selectbox("Timeframe", ["1d","5d","1mo","3mo","6mo","1y"])

# ==============================
# 📊 FETCH DATA
# ==============================
stock = yf.download(ticker, period=timeframe, interval="1d", progress=False)

if stock is not None and not stock.empty:

    if hasattr(stock.columns, "levels"):
        stock.columns = stock.columns.get_level_values(0)

    # ==============================
    # 📉 CANDLESTICK CHART
    # ==============================
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=stock.index,
        open=stock['Open'],
        high=stock['High'],
        low=stock['Low'],
        close=stock['Close'],
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4d4d',
        name="Price"
    ))

    fig.add_trace(go.Bar(
        x=stock.index,
        y=stock['Volume'],
        name="Volume",
        marker_color='rgba(0, 102, 255, 0.3)',
        yaxis='y2'
    ))

    fig.update_layout(
        template="plotly_dark",
        height=600,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis=dict(title="Price"),
        yaxis2=dict(
            overlaying='y',
            side='right',
            showgrid=False,
            title="Volume"
        ),
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig, width='stretch', key="main_chart")

    # ==============================
    # � MINI TREND GRAPH
    # ==============================
    st.subheader("📉 Price Trend")

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=stock.index,
        y=stock['Close'],
        mode='lines',
        line=dict(color='#00ff88', width=2),
        name="Trend"
    ))

    fig2.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=20, b=10)
    )

    st.plotly_chart(fig2, width='stretch', key="trend_chart")

    # ==============================    # �📊 MARKET OVERVIEW
    # ==============================
    st.subheader("📌 Market Overview")

    if len(stock) >= 2:
        latest = float(stock['Close'].iloc[-1])
        prev = float(stock['Close'].iloc[-2])

        change = latest - prev
        percent = (change / prev) * 100 if prev != 0 else 0.0

        color = "green" if change > 0 else "red"

        st.markdown(
            f"<h2 style='color:{color}'>💰 {latest:.2f} ({change:.2f}, {percent:.2f}%)</h2>",
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("📈 High", f"{float(stock['High'].iloc[-1]):.2f}")
        col2.metric("📉 Low", f"{float(stock['Low'].iloc[-1]):.2f}")
        col3.metric("📊 Volume", f"{float(stock['Volume'].iloc[-1]):,.0f}")
    else:
        st.warning("Not enough data to display market overview.")

    # ==============================
    # � WATCHLIST SNAPSHOT
    # ==============================
    st.subheader("📊 Watchlist Snapshot")

    cols = st.columns(len(watchlist))

    for i, stock_name in enumerate(watchlist):
        data = yf.download(stock_name, period="1d", interval="1m", progress=False)

        if data is not None and not data.empty:
            if hasattr(data.columns, "levels"):
                data.columns = data.columns.get_level_values(0)

            close_data = data['Close']
            if getattr(close_data, 'ndim', 1) > 1:
                close_data = close_data.iloc[:, 0]

            if len(close_data) >= 2:
                latest_snap = float(close_data.iloc[-1])
                prev_snap = float(close_data.iloc[0])

                change_snap = latest_snap - prev_snap
                percent_snap = (change_snap / prev_snap) * 100 if prev_snap != 0 else 0.0

                cols[i].metric(stock_name, f"{latest_snap:.2f}", f"{percent_snap:.2f}%")
                continue

        cols[i].metric(stock_name, "No data", "-")

    # ==============================
    # �📰 LIVE NEWS
    # ==============================
    st.subheader(f"📰 Live Market News for {company_name}")

    news_list = get_live_news(ticker)

    for news in news_list:
        st.write("•", news)

else:
    st.error("Invalid ticker")

# ==============================
# 📰 IMAGE INPUT
# ==============================
st.subheader(f"📰 Upload News for {company_name} ({ticker})")

uploaded_file = st.file_uploader("Upload News Image", type=["jpg","png","jpeg"])

if uploaded_file:
    uploaded_file.seek(0)
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded image", width=700)

    text = extract_text("temp.jpg")

    if not text or not text.strip():
        st.error("❌ No text detected from image. Please try a higher-quality image or a more complete screenshot.")
    else:
        st.write("### 📝 Extracted Text")
        st.write(text)

        clean = clean_text(text)

        sentiment, confidence_sent = get_sentiment(clean)
        fear = compute_fear_index(clean)

        features = create_feature_vector(sentiment, fear)

        movement, magnitude, confidence = predict(features)

        # ==============================
        # 🤖 AI SIGNAL PANEL
        # ==============================
        st.subheader(f"🤖 AI Decision for {company_name}")
        st.markdown(f"_This signal is based on OCR text extracted from the uploaded news image for **{company_name}**._")

        col1, col2 = st.columns([1,3])

        with col1:
            if movement == "UP" and sentiment == "POSITIVE":
                st.success("🟢 STRONG BUY")
            elif movement == "DOWN" and sentiment == "NEGATIVE":
                st.error("🔴 STRONG SELL")
            else:
                st.warning("⚠️ MIXED SIGNALS")

        with col2:
            st.write(f"Sentiment: **{sentiment}**")
            st.write(f"Confidence: **{confidence:.2f}**")
            st.progress(int(confidence * 100))

        # ==============================
        # 📊 METRICS
        # ==============================
        col1, col2, col3 = st.columns(3)

        col1.metric("Direction", movement)
        col2.metric("Confidence", f"{confidence*100:.2f}%")
        col3.metric("Expected Move", f"{magnitude*100:.2f}%")

        # ==============================
        # 🚀 FINAL TRADING SIGNAL
        # ==============================
        st.subheader("🚀 Final AI Decision")

        price_trend = "UP" if change > 0 else "DOWN"

        if movement == price_trend and confidence > 0.6:
            if movement == "UP":
                st.success("🟢 STRONG BUY SIGNAL")
            else:
                st.error("🔴 STRONG SELL SIGNAL")
        else:
            st.warning("⚠️ MIXED SIGNALS - BE CAUTIOUS")
