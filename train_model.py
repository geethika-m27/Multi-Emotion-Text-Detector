

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the cleaned dataset
df = pd.read_csv("clean_tweets.csv")

# Step 2: Remove missing and empty text or labels
df = df.dropna(subset=["content", "sentiment"])
df["content"] = df["content"].astype(str).str.strip()
df["sentiment"] = df["sentiment"].astype(str).str.strip()
df = df[
    (df["content"] != "") &
    (df["sentiment"] != "")
]

X = df["content"]
y = df["sentiment"]

print("Total records:", len(df))
print("Number of emotion classes:", y.nunique())

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Step 4: Create the TF-IDF and Logistic Regression pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words=None,
        sublinear_tf=True,
        min_df=2
    )),
    ("classifier", LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=42
    ))
])

# Step 5: Train the model
print("\nTraining model...")
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# Step 7: Save the trained model
joblib.dump(model, "emotion_model.pkl")

print("\nModel saved successfully as emotion_model.pkl")
