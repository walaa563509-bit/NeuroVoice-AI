
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="NeuroVoice AI",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("NeuroVoice_AI_Model_Final.pkl")

st.title("🧠 NeuroVoice AI")
st.subheader("Parkinson's Pattern Classification")

st.write(
    "An AI-powered research prototype that analyzes "
    "extracted voice features."
)

st.info(
    "Research prototype only. This application does not "
    "provide a medical diagnosis. It accepts extracted "
    "voice features, not raw audio recordings."
)

try:
    model = load_model()
    feature_names = list(model.feature_names_in_)
    st.success("AI Model loaded successfully!")

    uploaded_file = st.file_uploader(
        "Upload Voice Features CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        input_data = data.drop(
            columns=["id", "class"],
            errors="ignore"
        )

        missing = [
            col for col in feature_names
            if col not in input_data.columns
        ]

        if missing:
            st.error(
                f"Missing {len(missing)} required features."
            )
        else:
            input_data = input_data[feature_names]
            predictions = model.predict(input_data)

            results = data.copy()
            results["Prediction"] = [
                "Parkinson's" if p == 1 else "Healthy"
                for p in predictions
            ]

            st.success("Analysis completed successfully!")
            st.subheader("Prediction Results")
            st.dataframe(results, use_container_width=True)

            output = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Results",
                output,
                "NeuroVoice_AI_Results.csv",
                "text/csv"
            )

except Exception as error:
    st.error(f"Application error: {error}")
  
