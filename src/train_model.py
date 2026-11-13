
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load processed training data
processed_dir = BASE_DIR / "data" / "processed"

X_train = pd.read_csv(processed_dir / "X_train.csv")
Y_train = pd.read_csv(
    processed_dir / "y_train.csv"
).squeeze("columns")

# Verify data
print("Training features:", X_train.shape)
print("Training labels:", Y_train.shape)

# Create Logistic Regression model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# Train the model
model.fit(X_train, Y_train)

print("\nLogistic Regression model trained successfully!")
X_test = pd.read_csv(processed_dir / "X_test.csv")
Y_test = pd.read_csv(
    processed_dir / "y_test.csv"
).squeeze("columns")
Y_pred = model.predict(X_test)
print("\nFirst 10 predictions:")
print(Y_pred[:10])
print("\nFirst 10 actual labels:")
print(Y_test[:10].to_numpy())
from sklearn.metrics import average_precision_score, confusion_matrix
cm = confusion_matrix(Y_test ,Y_pred)
print(cm)
TN,FP,FN,TP = cm.ravel()
print("\nTrue Negatives:",TN)
print("\nFalse Positives:",FP)
print("\nFalse Negatives:",FN)
print("\nTrue Positives:",TP)
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
print("\n Model Evaluation Metrics:")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")
from sklearn.metrics import classification_report
Y_prob = model.predict_proba(X_test)[:, 1]
pr_auc = average_precision_score(Y_test, Y_prob)
print("\nPrecision-Recall AUC:", round(pr_auc, 4))
import joblib 
models_dir = BASE_DIR / "models"
models_dir .mkdir(parents = True , exist_ok = True)
joblib.dump(model, models_dir / " logistic regression.pkl")
print("\nLogistic Regression model saved successfully")
