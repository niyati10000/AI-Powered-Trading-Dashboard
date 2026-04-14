# 📊 AI-Powered Trading Dashboard (Computer Vision + NLP)

An intelligent trading dashboard that combines **Computer Vision, NLP, and real-time stock data** to generate actionable trading signals.

---

## 🚀 Features

- 📈 Real-time stock visualization using candlestick charts
- 📰 Live financial news integration (stock-specific)
- 🖼️ Image-based news analysis using OCR (Computer Vision)
- 🤖 Sentiment analysis using FinBERT (financial NLP model)
- 🧠 AI-based trading signal (BUY / SELL / HOLD)
- 📊 Multi-stock watchlist dashboard
- ⚡ Auto-refreshing real-time data

---

## 🧠 Tech Stack

- **Frontend**: Streamlit
- **Data Source**: yFinance API
- **News API**: NewsAPI
- **Computer Vision**: EasyOCR
- **NLP Model**: FinBERT (Transformers)
- **ML**: Scikit-learn
- **Visualization**: Plotly

---

## ⚙️ How It Works

1. User selects a stock (e.g., AAPL)
2. System fetches real-time stock data using yFinance
3. Live news related to the stock is retrieved using NewsAPI
4. User can upload a news image (screenshot/article)
5. OCR extracts text from the image (Computer Vision)
6. Extracted text is cleaned and processed
7. Sentiment analysis is performed using FinBERT
8. Sentiment + market data → feature vector
9. ML model predicts stock movement
10. Final AI decision is displayed (BUY / SELL / HOLD)

---

## 🖼️ Computer Vision Pipeline

- Input: News Image
- Preprocessing: Image normalization
- OCR Engine: EasyOCR
- Output: Extracted text from image
- Integration: Text used for sentiment + prediction

---

## 📊 AI Decision Logic

- 📈 Stock trend (price movement)
- 🧠 Sentiment score (positive/negative)
- 📊 Model confidence

Final Signal:
- 🟢 STRONG BUY → Positive sentiment + Uptrend
- 🔴 STRONG SELL → Negative sentiment + Downtrend
- ⚠️ HOLD → Mixed signals

---

**▶️ Run the App**

streamlit run app.py

## 🛠️ Installation

```bash
pip install -r requirements.txt
