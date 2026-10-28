
import pandas as pd
import re

# Load the original dataset
df = pd.read_csv("tweet_emotions.csv")

print("Original dataset:", df.shape)

# Keep only the columns needed for prediction
df = df[["content", "sentiment"]].copy()

# Remove records with the "empty" label
df = df[df["sentiment"] != "empty"].copy()

# Clean the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["content"] = df["content"].apply(clean_text)

# Remove blank text records
df = df[df["content"] != ""]

# Save the prepared dataset
df.to_csv("clean_tweets.csv", index=False)

print("\nDataset prepared successfully!")
print("New dataset shape:", df.shape)

print("\nEmotion distribution:")
print(df["sentiment"].value_counts())

print("\nCleaned examples:")
print(df.head())

print("\nSaved as clean_tweets.csv")
