
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

# Load saved model and threshold
model = joblib.load(
    BASE_DIR / "models" / "xgboost_model.pkl"
)

threshold = float(joblib.load(
    BASE_DIR / "models" / "best_threshold.pkl"
))

# Load sample test transactions
X_test = pd.read_csv(
    BASE_DIR / "data" / "processed" / "X_test.csv"
)


def test_model_loaded():
    """Check that the trained model loads successfully."""
    assert model is not None


def test_threshold_range():
    """Check that the threshold is between 0 and 1."""
    assert 0 <= threshold <= 1


def test_prediction_score_range():
    """Check that model scores are valid."""
    transaction = X_test.iloc[[0]]

    score = model.predict_proba(transaction)[0][1]

    assert np.isfinite(score)
    assert 0 <= score <= 1


def test_legitimate_transaction():
    """Check a known legitimate example."""
    transaction = X_test.iloc[[0]]

    score = model.predict_proba(transaction)[0][1]
    prediction = int(score >= threshold)

    assert prediction == 0


def test_fraud_transaction():
    """Check a known fraudulent example."""
    transaction = X_test.iloc[[840]]

    score = model.predict_proba(transaction)[0][1]
    prediction = int(score >= threshold)

    assert prediction == 1
