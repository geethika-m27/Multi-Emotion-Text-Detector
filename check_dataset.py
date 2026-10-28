
import pandas as pd

df = pd.read_csv("tweet_emotions.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nEmotion counts:")
print(df["sentiment"].value_counts())

print("\nSample tweets:")
print(df[["content", "sentiment"]].head())
