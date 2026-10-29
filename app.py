
import streamlit as st
import pandas as pd
import joblib
import os

# Step 1: Configure the page
st.set_page_config(
    page_title="Multi-Emotion Text Detector",
    page_icon="💬",
    layout="wide"
)

# Step 2: Application title
st.title("💬 Multi-Emotion Text Detector")
st.write(
    "Analyze a sentence or tweet using machine learning "
    "to predict its most likely emotion."
)

st.caption(
    "Model: TF-IDF + Logistic Regression | "
    "Dataset: labelled tweet emotion dataset"
)

# Step 3: Load the selected model
@st.cache_resource
def load_model():
    model_path = "best_emotion_model.joblib"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "best_emotion_model.joblib was not found. "
            "Run python compare_models.py first."
        )

    return joblib.load(model_path)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

# Step 4: Text input
st.subheader("Enter your text")

user_text = st.text_area(
    "Type a sentence or tweet:",
    placeholder="Example: I am so happy that I passed my exam!",
    height=150
)

# Step 5: Predict the emotion
if st.button("🔍 Detect Emotion", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text before predicting.")
    else:
        with st.spinner("Analyzing your text..."):
            try:
                prediction = model.predict([user_text])[0]

                st.success(
                    f"Predicted Emotion: {prediction.title()}"
                )

                # Display model scores
                classifier = model.named_steps["classifier"]

                if hasattr(classifier, "predict_proba"):
                    probabilities = model.predict_proba([user_text])[0]
                    classes = model.classes_

                    results = pd.DataFrame({
                        "Emotion": classes,
                        "Score (%)": probabilities * 100
                    }).sort_values(
                        "Score (%)", ascending=False
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Top emotion",
                            prediction.title(),
                            f"{probabilities[list(classes).index(prediction)] * 100:.1f}% model score"
                        )

                    with col2:
                        st.write("Top 5 emotion scores")
                        st.dataframe(
                            results.head(5).round(2),
                            hide_index=True,
                            use_container_width=True
                        )

                    st.subheader("Emotion Score Analysis")
                    st.bar_chart(
                        results.head(8).set_index("Emotion")
                    )

                    st.caption(
                        "Scores are model-estimated probabilities, "
                        "not guaranteed measures of emotional intensity."
                    )
                else:
                    st.info(
                        "This classifier does not provide probability "
                        "scores. The predicted label is shown above."
                    )

                st.info(
                    "This is a machine learning prediction, not a "
                    "reliable assessment of a person's mental health. "
                    "Sarcasm, context, and mixed emotions may be missed."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")

# Step 6: Project information
with st.expander("About this project"):
    st.write(
        "TF-IDF converts text into numerical features. "
        "Logistic Regression classifies the text into one of "
        "the dataset's emotion labels."
    )

    st.write(
        "Three classifiers were compared: Logistic Regression, "
        "Multinomial Naive Bayes, and Linear SVM."
    )

    st.write(
        "On the current test split, Logistic Regression achieved "
        "29.07% accuracy and a macro F1-score of 22.18%."
    )

    st.write(
        "The model is an experimental prototype and may make "
        "incorrect predictions, especially for underrepresented emotions."
    )

st.divider()
st.caption("Machine Learning Project | Multi-Emotion Text Detector")
