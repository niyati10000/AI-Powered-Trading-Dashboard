import pandas as pd

# Load original dataset
df = pd.read_csv("data/raw/news_dataset.csv")

# Combine Top1 to Top25 columns into one text column
text_columns = [col for col in df.columns if "Top" in col]

df['text'] = df[text_columns].astype(str).apply(lambda x: " ".join(x), axis=1)

# Keep only required columns
df = df[['Date', 'text']]

# Rename columns
df.columns = ['date', 'text']

# Save cleaned dataset
df.to_csv("data/raw/news_dataset_cleaned.csv", index=False)

print("✅ Dataset prepared successfully!")