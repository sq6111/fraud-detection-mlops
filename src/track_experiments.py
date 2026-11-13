
from pathlib import Path
import pandas as pd
import joblib
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Load the processed test dataset
X_test = pd.read_csv(
    BASE_DIR / "data" / "processed" / "X_test.csv"
)

Y_test = pd.read_csv(
    BASE_DIR / "data" / "processed" / "y_test.csv"
).squeeze("columns")

# Load both trained models
logistic_model = joblib.load(
    BASE_DIR / "models" / "logistic_regression.pkl"
)

xgboost_model = joblib.load(
    BASE_DIR / "models" / "xgboost_model.pkl"
)

print("Test data loaded:", X_test.shape)
print("Logistic Regression and XGBoost models loaded successfully!")


# Store MLflow experiments in a local SQLite database
mlflow.set_tracking_uri(
    f"sqlite:///{BASE_DIR / 'mlflow.db'}"
)

# Create or select the experiment
mlflow.set_experiment("Fraud_Detection_Experiment")


# Start a run for Logistic Regression
with mlflow.start_run(run_name="Logistic_Regression"):

    # Make predictions
    Y_pred = logistic_model.predict(X_test)
    Y_prob = logistic_model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        "accuracy": accuracy_score(Y_test, Y_pred),
        "precision": precision_score(Y_test, Y_pred),
        "recall": recall_score(Y_test, Y_pred),
        "f1_score": f1_score(Y_test, Y_pred),
        "average_precision": average_precision_score(Y_test, Y_prob)
    }

    # Save model parameters
    mlflow.log_params(logistic_model.get_params())

    # Save evaluation metrics
    mlflow.log_metrics(metrics)

    # Save the existing trained model file
    mlflow.log_artifact(
        str(BASE_DIR / "models" / "logistic_regression.pkl"),
        artifact_path="model"
    )

    print("\nLogistic Regression experiment recorded!")
    print(metrics)

# Start a separate MLflow run for XGBoost
with mlflow.start_run(run_name="XGBoost"):

    # Make predictions using the saved XGBoost model
    Y_pred = xgboost_model.predict(X_test)
    Y_prob = xgboost_model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        "accuracy": accuracy_score(Y_test, Y_pred),
        "precision": precision_score(Y_test, Y_pred),
        "recall": recall_score(Y_test, Y_pred),
        "f1_score": f1_score(Y_test, Y_pred),
        "average_precision": average_precision_score(Y_test, Y_prob)
    }

    # Record model parameters
    mlflow.log_params(xgboost_model.get_params())

    # Record evaluation metrics
    mlflow.log_metrics(metrics)

    # Save a copy of the trained model
    mlflow.log_artifact(
        str(BASE_DIR / "models" / "xgboost_model.pkl"),
        artifact_path="model"
    )

    print("\nXGBoost experiment recorded!")
    print(metrics)
