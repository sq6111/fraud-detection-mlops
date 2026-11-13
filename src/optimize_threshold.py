
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent

print("Threshold optimization started!")

# STEP 1: Load original dataset
df = pd.read_csv(
    BASE_DIR / "data" / "raw" / "creditcard.csv"
)

X = df.drop("Class", axis=1)
Y = df["Class"]

# STEP 2: Reproduce original train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# STEP 3: Create training and validation datasets
X_train_new, X_val, Y_train_new, Y_val = train_test_split(
    X_train,
    Y_train,
    test_size=0.2,
    random_state=42,
    stratify=Y_train
)

print("\nTraining shape:", X_train_new.shape)
print("Validation shape:", X_val.shape)

print("\nTraining class distribution:")
print(Y_train_new.value_counts())

print("\nValidation class distribution:")
print(Y_val.value_counts())

# STEP 4: Apply leakage-free feature scaling
scaler = StandardScaler()
columns_to_scale = ["Time", "Amount"]

X_train_new = X_train_new.copy()
X_val = X_val.copy()

X_train_new[columns_to_scale] = scaler.fit_transform(
    X_train_new[columns_to_scale]
)

X_val[columns_to_scale] = scaler.transform(
    X_val[columns_to_scale]
)

print("\nLeakage-free scaling completed!")

# STEP 5: Calculate class imbalance weight
negative_count = (Y_train_new == 0).sum()
positive_count = (Y_train_new == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nScale positive weight:", scale_pos_weight)

# STEP 6: Train a separate XGBoost model
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_new, Y_train_new)

print("\nValidation XGBoost model trained successfully!")

# STEP 7: Predict fraud probabilities
Y_val_prob = model.predict_proba(X_val)[:, 1]

# STEP 8: Compare different thresholds
thresholds = np.arange(0.05, 0.96, 0.05)

results = []

for threshold in thresholds:

    Y_val_pred = (Y_val_prob >= threshold).astype(int)

    precision = precision_score(
        Y_val, Y_val_pred, zero_division=0
    )

    recall = recall_score(
        Y_val, Y_val_pred, zero_division=0
    )

    f1 = f1_score(
        Y_val, Y_val_pred, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        Y_val, Y_val_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    })

# STEP 9: Display threshold comparison
results_df = pd.DataFrame(results)

print("\nThreshold comparison:")
print(results_df.to_string(index=False))

# STEP 10: Select best threshold using F1-score
best_row = results_df.loc[
    results_df["f1_score"].idxmax()
]

best_threshold = float(best_row["threshold"])

print("\nBest threshold:", best_threshold)
print("Best precision:", best_row["precision"])
print("Best recall:", best_row["recall"])
print("Best F1-score:", best_row["f1_score"])

# STEP 11: Save validation results
reports_dir = BASE_DIR / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    reports_dir / "threshold_comparison.csv",
    index=False
)

print("\nThreshold comparison saved successfully!")

# STEP 12: Save selected threshold
models_dir = BASE_DIR / "models"
models_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    best_threshold,
    models_dir / "best_threshold.pkl"
)

print("Best threshold saved successfully!")

import mlflow

# Connect to the same MLflow database
mlflow.set_tracking_uri(
    f"sqlite:///{BASE_DIR / 'mlflow.db'}"
)

mlflow.set_experiment("Fraud_Detection_Experiment")

with mlflow.start_run(run_name="XGBoost_Threshold_Optimization"):

    mlflow.log_param("selection_metric", "f1_score")
    mlflow.log_param("validation_fraction", 0.2)
    mlflow.log_param("best_threshold", best_threshold)

    mlflow.log_params(model.get_params())

    mlflow.log_metric(
        "validation_precision",
        float(best_row["precision"])
    )

    mlflow.log_metric(
        "validation_recall",
        float(best_row["recall"])
    )

    mlflow.log_metric(
        "validation_f1",
        float(best_row["f1_score"])
    )

    mlflow.log_artifact(
        str(reports_dir / "threshold_comparison.csv")
    )

print("\nThreshold optimization recorded in MLflow!")
