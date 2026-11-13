from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
BASE_DIR = Path(__file__).resolve().parent.parent
processed_dir = BASE_DIR / "data" / "processed"
X_train = pd.read_csv(processed_dir /  "X_train.csv")
Y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze("columns")
X_test = pd.read_csv(processed_dir / "X_test.csv")
Y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze("columns")
print("Training features:", X_train.shape)
print("Training labels:", Y_train.shape)
print("Testing features:", X_test.shape)
print("Testing labels:", Y_test.shape)
print("\nTraining class distribution:")
print(Y_train.value_counts())
# calculate class imbalance ratio
negative_counts = (Y_train == 0).sum()
positive_counts = (Y_train == 1).sum()
scale_pos_weight = negative_counts / positive_counts
print("Scale positive weight :", scale_pos_weight)
#create XGBoost model 
model = XGBClassifier(
    n_estimators = 100,
    max_depth = 4,
    learning_rate = 0.1 ,
    scale_pos_weight = scale_pos_weight,
    objective = "binary:logistic",
    eval_metric = "logloss",
    random_state =42,
    n_jobs = -1
)
model.fit(X_train, Y_train)
print("\nXGBoost model trained successfully!")
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
Y_pred = model.predict(X_test)
Y_prob = model.predict_proba(X_test)[:,1]
cm = confusion_matrix(Y_test, Y_pred)
TN, FP, FN, TP = cm.ravel()
print("\nConfusion Matrix:")
print(cm)


print("\nTrue Negatives:", TN)
print("False Positives:", FP)
print("False Negatives:", FN)
print("True Positives:", TP)
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
average_precision = average_precision_score(Y_test, Y_prob)
#Display results
print("\nXGBoost Evaluation Metrics")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")
print("Average Precision Score:", round(average_precision, 4))
import joblib 
models_dir = BASE_DIR / "models"
models_dir.mkdir(parents = True , exist_ok = True)
joblib.dump(model, models_dir / "xgboost_model.pkl")
print("\nXGBoost model saved successfully")



