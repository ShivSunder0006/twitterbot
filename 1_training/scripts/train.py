import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# --- File Paths ---
DATA_PATH = os.path.join("1_training", "data", "processed_tweets.csv")
MODEL_DIR = os.path.join("1_training", "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_pipeline.joblib")

def main():
    """Trains and saves the machine learning pipeline."""
    print("Starting model training...")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Processed data not found at {DATA_PATH}. Please run preprocess.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=['text'], inplace=True)

    X = df['text']
    y = df['sentiment']

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    print("Training the pipeline (this may take a few minutes)...")
    pipeline.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    
    print(f"Training complete. Model pipeline saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
