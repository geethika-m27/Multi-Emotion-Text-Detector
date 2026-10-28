
import streamlit as st
import pandas as pd
import joblib
import os

# Step 1: Configure the Streamlit page
st.set_page_config(
    page_title="Multi-Emotion Text Detector",
    page_icon="💬",
    layout="wide"
)

# Step 2: Add the application title
st.title("💬 Multi-Emotion Text Detector")
st.write(
    "Analyze a sentence or tweet using machine learning "
    "to predict its most likely emotion."
)

st.caption(
    "Model: TF-IDF + Logistic Regression | "
    "Dataset: 12 emotion categories"
)

# Step 3: Load the saved model
@st.cache_resource
def load_model():
    model_path = "emotion_model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "emotion_model.pkl was not found. "
            "Run train_model.py first."
        )

    return joblib.load(model_path)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

# Step 4: Create the text input
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
                # Predict the most likely emotion
                prediction = model.predict([user_text])[0]

                st.success(f"Predicted Emotion: {prediction.title()}")

                # Display scores when the classifier supports them
                classifier = model.named_steps.get("classifier")

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([user_text])[0]
                    classes = model.classes_

                    results = pd.DataFrame({
                        "Emotion": classes,
                        "Score (%)": probabilities * 100
                    }).sort_values(
                        by="Score (%)",
                        ascending=False
                    )

                    st.subheader("Emotion Score Analysis")

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.metric(
                            "Top predicted emotion",
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

                    st.bar_chart(
                        results.head(8).set_index("Emotion")
                    )

                    st.caption(
                        "Scores are model-estimated probabilities, "
                        "not guaranteed measures of emotional intensity."
                    )

                st.info(
                    "This tool predicts a label based on patterns in its "
                    "training data. It may misunderstand sarcasm, context, "
                    "or mixed emotions."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")

# Step 6: Add project information
with st.expander("About this project"):
    st.write(
        "This application uses TF-IDF to convert text into numerical "
        "features and Logistic Regression to classify text into one "
        "of 12 emotion labels."
    )
    st.write(
        "The model was trained on a labelled tweet dataset. "
        "Its measured test accuracy was approximately 29.07%."
    )

st.divider()
st.caption("Machine Learning Project | Multi-Emotion Text Detector")
