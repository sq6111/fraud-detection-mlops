
from pathlib import Path
import pandas as pd
import joblib
import shap

# 1. Find project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Load trained XGBoost model
model_path = BASE_DIR / "models" / "xgboost_model.pkl"
model = joblib.load(model_path)

print("XGBoost model loaded successfully!")

# 3. Load processed test data
test_path = BASE_DIR / "data" / "processed" / "X_test.csv"
X_test = pd.read_csv(test_path)

print("Test data shape:", X_test.shape)

# 4. Select 100 transactions for explanation
X_sample = X_test.sample(n=100, random_state=42)

# 5. Create SHAP explainer
explainer = shap.TreeExplainer(model)

# 6. Calculate SHAP values
shap_values = explainer(X_sample)

print("SHAP values calculated successfully!")
print("SHAP values shape:", shap_values.values.shape)
print("\nFirst transaction SHAP values:")
print(shap_values.values[0])
print("\nFeature names:")
print(X_sample.columns.tolist())

import matplotlib.pyplot as plt
#create reports folder
reports_dir = BASE_DIR / "reports"
reports_dir.mkdir(parents = True, exist_ok = True)
#generate shap summary plot 
shap.plots.waterfall(shap_values[0], show = False)
#save the plots 
plt.savefig(
reports_dir / "shap_summary_plot.png",
dpi = 300,
bbox_inches = "tight"

)
plt.close()
print("Shap waterfall plot saved successfully")

# Generate SHAP beeswarm summary plot
shap.plots.beeswarm(
    shap_values,
    max_display=15,
    show=False
)

# Save the plot
plt.savefig(
    reports_dir / "shap_beeswarm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP beeswarm plot saved successfully!")
#load actual test labels 
Y_test_path = BASE_DIR / "data" / "processed" / "y_test.csv"
Y_test = pd.read_csv(Y_test_path).squeeze("columns")
print("\nTest class distribution:")
print(Y_test.value_counts())
#select 20 actual fraud transactions 
fraud_indicies = Y_test[Y_test == 1].index[:20]
X_fraud = X_test.loc[fraud_indicies]
#select 20 legitimate transactions 
legitimate_indices = Y_test[Y_test == 0 ].index[:20]
X_legitimate = X_test.loc[legitimate_indices]
print("\nFraud transaction:",X_fraud.shape)
print("Legitimate transaction:" ,X_legitimate.shape)
#calculate shap values for both groups
fraud_shap = explainer(X_fraud)
legitimate_shap = explainer(X_legitimate)
print("\nFraud SHAP shape:", fraud_shap.values.shape)
print("Legitimate SHAP shape:",legitimate_shap.values.shape)

# SHAP beeswarm plot for fraud transactions
shap.plots.beeswarm(
    fraud_shap,
    max_display=15,
    show=False
)

plt.savefig(
    reports_dir / "shap_fraud_beeswarm.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Fraud SHAP beeswarm plot saved!")

# SHAP beeswarm plot for legitimate transactions
shap.plots.beeswarm(
    legitimate_shap,
    max_display=15,
    show=False
)

plt.savefig(
    reports_dir / "shap_legitimate_beeswarm.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Legitimate SHAP beeswarm plot saved!")


# Predict all test transactions
Y_pred = model.predict(X_test)

# Find correctly detected fraud
correct_fraud_indices = Y_test[
    (Y_test == 1) & (Y_pred == 1)
].index

# Find missed fraud
missed_fraud_indices = Y_test[
    (Y_test == 1) & (Y_pred == 0)
].index

print("\nCorrectly detected fraud:", len(correct_fraud_indices))
print("Missed fraud:", len(missed_fraud_indices))

# Select one correctly detected fraud transaction
correct_index = correct_fraud_indices[0]
X_correct = X_test.loc[[correct_index]]

# Select one missed fraud transaction
missed_index = missed_fraud_indices[0]
X_missed = X_test.loc[[missed_index]]

# Calculate SHAP explanations
correct_shap = explainer(X_correct)
missed_shap = explainer(X_missed)

print("\nCorrect fraud transaction index:", correct_index)
print("Missed fraud transaction index:", missed_index)

# Plot correctly detected fraud
shap.plots.waterfall(correct_shap[0], show=False)

plt.savefig(
    reports_dir / "shap_correct_fraud.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Correct fraud waterfall plot saved!")

# Plot missed fraud
shap.plots.waterfall(missed_shap[0], show=False)

plt.savefig(
    reports_dir / "shap_missed_fraud.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Missed fraud waterfall plot saved!")
