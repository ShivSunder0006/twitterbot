import pandas as pd
import regex as re
import nltk
from nltk.corpus import stopwords
import os

# --- File Paths ---
DATA_DIR = os.path.join("1_training", "data")
INPUT_PATH = os.path.join(DATA_DIR, "raw_tweets.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "processed_tweets.csv")

# --- Preprocessing Logic ---
def preprocess_text(text):
    """Cleans and prepares a single string of text."""
    stop_words = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    text = " ".join(tokens)
    
    return text.strip()

def main():
    """Main function to load, process, and save the data."""
    print("Starting data preprocessing...")
    
    try:
        df = pd.read_csv(
            INPUT_PATH,
            encoding="latin-1",
            header=None,
            names=["sentiment", "id", "date", "query", "user", "text"]
        )
    except FileNotFoundError:
        print(f"Error: The file {INPUT_PATH} was not found. Please place your data there.")
        return

    df = df[['sentiment', 'text']]
    df['sentiment'] = df['sentiment'].replace(4, 1)

    print("Cleaning tweet text (this may take a few minutes)...")
    df['text'] = df['text'].apply(preprocess_text)

    df.dropna(subset=['text'], inplace=True)
    df = df[df['text'] != '']

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Preprocessing complete. Cleaned data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
