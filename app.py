
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path

# ---------------------------------------
# STEP 1: Project configuration
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")

st.write(
    "An explainable fraud detection application "
    "using XGBoost, SHAP, and MLflow."
)

st.caption(
    "Educational demonstration using the ULB "
    "credit card fraud dataset."
)

# ---------------------------------------
# STEP 2: Load model and preprocessing
# ---------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(
        BASE_DIR / "models" / "xgboost_model.pkl"
    )

    scaler = joblib.load(
        BASE_DIR / "models" / "scaler.pkl"
    )

    threshold = float(joblib.load(
        BASE_DIR / "models" / "best_threshold.pkl"
    ))

    explainer = shap.TreeExplainer(model)

    return model, scaler, threshold, explainer


model, scaler, threshold, explainer = load_model()

feature_columns = list(model.feature_names_in_)

st.success("XGBoost model loaded successfully!")
st.info(f"Selected decision threshold: {threshold:.2f}")

st.caption(
    "Model scores are not calibrated real-world "
    "fraud probabilities."
)

# ---------------------------------------
# STEP 3: Load existing test data
# ---------------------------------------

@st.cache_data
def load_test_data():
    X_test = pd.read_csv(
        BASE_DIR / "data" / "processed" / "X_test.csv"
    )

    Y_test = pd.read_csv(
        BASE_DIR / "data" / "processed" / "y_test.csv"
    ).squeeze("columns")

    return X_test, Y_test


X_test, Y_test = load_test_data()

# ---------------------------------------
# STEP 4: Helper functions
# ---------------------------------------

def predict_transactions(features):
    scores = model.predict_proba(features)[:, 1]

    predictions = (scores >= threshold).astype(int)

    results = pd.DataFrame({
        "Fraud Score": scores,
        "Prediction": np.where(
            predictions == 1,
            "Fraud",
            "Legitimate"
        )
    }, index=features.index)

    return results


def show_shap_explanation(transaction):
    st.subheader("SHAP Explanation")

    shap_values = explainer(transaction)

    plt.close("all")

    shap.plots.waterfall(
        shap_values[0],
        max_display=10,
        show=False
    )

    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Red features increase the model's fraud "
        "score, while blue features decrease it. "
        "SHAP contributions are in log-odds units "
        "for this XGBoost model."
    )


def prepare_uploaded_data(data):
    expected = set(feature_columns)
    provided = set(data.columns)

    missing = expected - provided
    extra = provided - expected

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    if extra:
        raise ValueError(
            "Unexpected columns: "
            + ", ".join(sorted(extra))
        )

    # Ensure correct feature order
    features = data[feature_columns].copy()

    # Require numeric input
    for column in feature_columns:
        features[column] = pd.to_numeric(
            features[column],
            errors="raise"
        )

    if not np.isfinite(features.to_numpy()).all():
        raise ValueError(
            "CSV contains missing or infinite values."
        )

    # Uploaded CSVs must contain raw Time and Amount
    features[["Time", "Amount"]] = scaler.transform(
        features[["Time", "Amount"]]
    )

    return features


# ---------------------------------------
# STEP 5: Choose application mode
# ---------------------------------------

mode = st.radio(
    "Choose prediction mode",
    [
        "Single Transaction",
        "CSV Batch Prediction"
    ],
    horizontal=True
)

# ---------------------------------------
# STEP 6: Single transaction prediction
# ---------------------------------------

if mode == "Single Transaction":

    st.subheader("Transaction Fraud Prediction")

    transaction_index = st.number_input(
        "Select a transaction index",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1
    )

    transaction = X_test.iloc[[transaction_index]]

    st.write("Selected transaction:")
    st.dataframe(
        transaction,
        use_container_width=True
    )

    if st.button("Predict Fraud", type="primary"):

        fraud_score = float(
            model.predict_proba(transaction)[0][1]
        )

        prediction = int(fraud_score >= threshold)

        actual_label = int(
            Y_test.iloc[transaction_index]
        )

        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Model Fraud Score",
                f"{fraud_score:.4f}"
            )

        with col2:
            st.metric(
                "Decision Threshold",
                f"{threshold:.2f}"
            )

        with col3:
            st.metric(
                "Predicted Class",
                "Fraud" if prediction else "Legitimate"
            )

        if prediction == 1:
            st.error(
                "Fraudulent Transaction Detected!"
            )
        else:
            st.success(
                "Transaction Classified as Legitimate"
            )

        st.write(
            "Actual label:",
            "Fraud" if actual_label else "Legitimate"
        )

        if prediction == actual_label:
            st.success(
                "Prediction matches the actual label!"
            )
        else:
            st.warning(
                "Prediction does not match the actual label."
            )

        show_shap_explanation(transaction)

# ---------------------------------------
# STEP 7: CSV batch prediction
# ---------------------------------------

else:

    st.subheader("CSV Batch Fraud Detection")

    st.write(
        "Upload a CSV containing raw transaction "
        "features: Time, V1 through V28, and Amount."
    )

    st.warning(
        "Upload raw, unscaled features only. "
        "Do not upload X_test.csv from data/processed "
        "because its Time and Amount columns "
        "have already been scaled."
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            uploaded_data = pd.read_csv(uploaded_file)

            if uploaded_data.empty:
                st.error("The uploaded CSV is empty.")
                st.stop()

            st.write("Uploaded transactions:")
            st.dataframe(
                uploaded_data.head(10),
                use_container_width=True
            )

            prepared_data = prepare_uploaded_data(
                uploaded_data
            )

            if st.button(
                "Run Batch Prediction",
                type="primary"
            ):

                results = predict_transactions(
                    prepared_data
                )

                output = uploaded_data.copy()
                output["Fraud Score"] = (
                    results["Fraud Score"].to_numpy()
                )
                output["Prediction"] = (
                    results["Prediction"].to_numpy()
                )

                fraud_count = int(
                    (output["Prediction"] == "Fraud").sum()
                )

                legitimate_count = (
                    len(output) - fraud_count
                )

                st.subheader("Batch Prediction Results")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Transactions",
                    len(output)
                )

                col2.metric(
                    "Flagged Fraud",
                    fraud_count
                )

                col3.metric(
                    "Classified Legitimate",
                    legitimate_count
                )

                st.dataframe(
                    output,
                    use_container_width=True
                )

                csv_output = output.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_output,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

                # Explain one uploaded transaction
                st.subheader(
                    "Explain an Uploaded Transaction"
                )

                explanation_index = st.number_input(
                    "Transaction row to explain",
                    min_value=0,
                    max_value=len(prepared_data) - 1,
                    value=0,
                    step=1
                )

                selected = prepared_data.iloc[
                    [explanation_index]
                ]

                show_shap_explanation(selected)

        except (ValueError, TypeError) as error:
            st.error(f"Invalid CSV: {error}")

# ---------------------------------------
# STEP 8: Footer
# ---------------------------------------

st.divider()

st.caption(
    "Credit Card Fraud Detection | "
    "XGBoost + SHAP + MLflow | "
    "Educational use only"
)
