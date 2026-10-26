"""Standalone example: score review text with VADER sentiment analysis."""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

INPUT_FILE = "reviews.csv"
OUTPUT_FILE = "scored_reviews.csv"

df = pd.read_csv(INPUT_FILE)
if "Review" not in df.columns:
    raise ValueError("The input CSV must contain a 'Review' column.")

analyzer = SentimentIntensityAnalyzer()
df["Review"] = df["Review"].fillna("").astype(str)
df["Sentiment Score"] = df["Review"].apply(
    lambda review: analyzer.polarity_scores(review)["compound"]
)

def classify(score):
    if score > 0.01:
        return "Positive"
    if score < -0.01:
        return "Negative"
    return "Neutral"

df["Sentiment"] = df["Sentiment Score"].apply(classify)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(df):,} scored reviews to {OUTPUT_FILE}")
print(df["Sentiment"].value_counts())
