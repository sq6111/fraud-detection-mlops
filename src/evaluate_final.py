
from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Load existing final model and selected threshold
model = joblib.load(
    BASE_DIR / "models" / "xgboost_model.pkl"
)

best_threshold = joblib.load(
    BASE_DIR / "models" / "best_threshold.pkl"
)

# Load held-out test data
X_test = pd.read_csv(
    BASE_DIR / "data" / "processed" / "X_test.csv"
)

Y_test = pd.read_csv(
    BASE_DIR / "data" / "processed" / "y_test.csv"
).squeeze("columns")

# Predict probabilities
Y_prob = model.predict_proba(X_test)[:, 1]

print("Final test dataset:", X_test.shape)
print("Selected validation threshold:", best_threshold)

# Compare both thresholds on the same final model
for threshold in [0.5, best_threshold]:

    Y_pred = (Y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        Y_test, Y_pred, labels=[0, 1]
    ).ravel()

    print(f"\n--- Threshold: {threshold:.2f} ---")
    print("Precision:", precision_score(
        Y_test, Y_pred, zero_division=0
    ))
    print("Recall:", recall_score(
        Y_test, Y_pred, zero_division=0
    ))
    print("F1-score:", f1_score(
        Y_test, Y_pred, zero_division=0
    ))
    print("False positives:", fp)
    print("Missed fraud:", fn)
    print("Correctly detected fraud:", tp)

print("\nAverage Precision:",
      average_precision_score(Y_test, Y_prob))
