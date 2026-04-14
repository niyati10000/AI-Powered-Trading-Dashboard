import pandas as pd
import yfinance as yf
from src.sentiment import get_sentiment
from src.features import compute_fear_index

def build_dataset():
    df = pd.read_csv("data/raw/news_dataset_cleaned.csv")

    # Convert date properly
    df['date'] = pd.to_datetime(df['date']).dt.date

    print("Applying sentiment...")
    df['sentiment'] = df['text'].apply(get_sentiment)

    print("Applying fear index...")
    df['fear'] = df['text'].apply(compute_fear_index)

    print("Downloading stock data...")

    stock = yf.download("AAPL", start="2008-01-01")

    # 🔥 FIX MULTI-LEVEL COLUMNS
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)
    
    stock.reset_index(inplace=True)
    
    # Fix stock date
    stock['Date'] = pd.to_datetime(stock['Date']).dt.date

    print("Merging datasets...")
    merged = pd.merge(df, stock, left_on='date', right_on='Date', how='inner')

    print("Merged shape:", merged.shape)

    if merged.empty:
        print("❌ ERROR: Merge resulted in empty dataset")
        return

    # Create targets
    merged['target'] = (merged['Close'].shift(-1) > merged['Close']).astype(int)
    merged['pct_change'] = merged['Close'].pct_change().shift(-1)

    merged.dropna(inplace=True)

    merged.to_csv("data/processed/final_dataset.csv", index=False)

    print("✅ Dataset created successfully!")

if __name__ == "__main__":
    build_dataset()