
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
import joblib

# Step 1: Load the prepared dataset
data = pd.read_csv("clean_tweets.csv")
data = data.dropna(subset=["content", "sentiment"])
data["content"] = data["content"].astype(str)
data["sentiment"] = data["sentiment"].astype(str)

X = data["content"]
y = data["sentiment"]

# Step 2: Use the same split for every algorithm
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Define the algorithms
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1500, class_weight="balanced"
    ),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(class_weight="balanced")
}

results = []
best_score = -1
best_name = None
best_pipeline = None

# Step 4: Train and evaluate each model
for name, classifier in models.items():
    print(f"\nTraining {name}...")
    start = time.time()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Time: {(time.time() - start):.1f} seconds")

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1
    })

    # Select using macro F1 to account for imbalanced classes
    if macro_f1 > best_score:
        best_score = macro_f1
        best_name = name
        best_pipeline = pipeline

# Step 5: Display comparison and save results
comparison = pd.DataFrame(results).sort_values(
    by="Macro F1", ascending=False
)
print("\n=== ALGORITHM COMPARISON ===")
print(comparison.to_string(index=False))

print(f"\nBest model by Macro F1: {best_name}")
print("\nClassification report for selected model:")
print(classification_report(
    y_test, best_pipeline.predict(X_test), zero_division=0
))

# Save separately to protect the currently working app model
joblib.dump(best_pipeline, "best_emotion_model.joblib")
comparison.to_csv("algorithm_comparison.csv", index=False)

print("\nSaved best_emotion_model.joblib")
print("Saved algorithm_comparison.csv")
