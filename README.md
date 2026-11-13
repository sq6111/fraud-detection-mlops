# Credit Card Fraud Detection with Explainable AI

**An end-to-end machine learning project using XGBoost, SHAP, MLflow, and Streamlit.**

This project detects fraudulent credit card transactions using machine learning, explains individual predictions with SHAP, tracks experiments using MLflow, and provides an interactive Streamlit application for individual and batch predictions.

## Project Highlights

- **80.81% F1-score** on the held-out test set using a validation-selected threshold of 0.95
- **80.00% precision** and **81.63% recall**
- Reduced false positives from **161 to 20** compared with the default threshold of 0.50
- Explainable predictions using SHAP waterfall and beeswarm plots
- MLflow experiment tracking and threshold optimization
- Reproducible model-training pipeline
- Streamlit application with CSV batch prediction
- Five passing automated tests using pytest

## 1. Problem Statement

Credit card fraud detection is challenging because fraudulent transactions are extremely rare compared with legitimate transactions.

A machine learning model must identify fraudulent activity while minimizing unnecessary alerts and missed fraud.

This project investigates Logistic Regression and XGBoost, addresses class imbalance, and uses threshold optimization to balance precision and recall.

## 2. Dataset

**Dataset:** Credit Card Fraud Detection

**Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Fraud percentage | 0.173% |
| Input features | 30 |
| Target variable | Class |

The input features include `Time`, `Amount`, and anonymized variables `V1` through `V28`.

The target column `Class` uses 0 for legitimate transactions and 1 for fraudulent transactions.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| Web application | Streamlit |
| Visualization | Matplotlib |
| Automated testing | Pytest |
| Version control | Git, GitHub |

## 4. Project Workflow

The machine learning pipeline follows these stages:

1. **Exploratory Data Analysis:** Investigate fraud distribution, transaction amounts, and outliers.
2. **Preprocessing:** Perform stratified train/test splitting and standardize `Time` and `Amount`.
3. **Baseline Model:** Train Logistic Regression with class weighting.
4. **Advanced Model:** Train XGBoost with `scale_pos_weight` to address class imbalance.
5. **Evaluation:** Compare precision, recall, F1-score, and average precision.
6. **Explainability:** Generate SHAP feature importance and waterfall explanations.
7. **Experiment Tracking:** Record configurations and metrics in MLflow.
8. **Threshold Optimization:** Select a decision threshold using a separate validation set.
9. **Final Evaluation:** Evaluate the selected threshold on the held-out test dataset.
10. **Deployment Interface:** Provide single-transaction and batch predictions using Streamlit.

## 5. Model Performance

### Logistic Regression vs. XGBoost

Results on the held-out test set using the default threshold of 0.50:

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| Accuracy | 97.55% | 99.69% |
| Precision | 6.10% | 34.55% |
| Recall | 91.84% | 86.73% |
| F1-score | 11.44% | 49.42% |
| Average Precision | 71.59% | 83.27% |

XGBoost achieved substantially better precision, F1-score, and average precision than the Logistic Regression baseline.

### Threshold Optimization

The decision threshold was optimized by maximizing F1-score on a separate validation set.

| Metric | Default Threshold 0.50 | Selected Threshold 0.95 |
|---|---:|---:|
| Precision | 34.55% | 80.00% |
| Recall | 86.73% | 81.63% |
| F1-score | 49.42% | 80.81% |
| False positives | 161 | 20 |
| False negatives | 13 | 18 |
| True positives | 85 | 80 |

The selected threshold reduced false positives by approximately **87.6%**, while increasing missed fraudulent transactions from 13 to 18.

These results illustrate the trade-off between precision and recall. In a real fraud detection system, threshold selection would also consider financial losses, investigation costs, and risk tolerance.

## 6. Explainable AI with SHAP

SHAP is used to explain how individual features influence XGBoost predictions.

The project supports:

- Global feature importance
- SHAP beeswarm visualizations
- Individual waterfall explanations
- Analysis of correctly detected and missed fraud cases
- Interactive SHAP explanations in Streamlit

SHAP contributions for this XGBoost classifier are expressed in model-output units, typically log-odds.

## 7. MLflow Experiment Tracking

MLflow tracks:

- Model hyperparameters
- Precision, recall, F1-score, and average precision
- Logistic Regression and XGBoost experiments
- Validation threshold optimization
- Saved model artifacts and threshold comparison results

To start the local MLflow interface:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open `http://localhost:5000` to view the experiment dashboard after generating the local tracking database.

## 8. Project Structure

```text
fraud-detection-mlops/
├── app.py
├── setup_project.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── explore_data.py
│   ├── visualize_data.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   ├── train_xgboost.py
│   ├── explain_model.py
│   ├── track_experiments.py
│   ├── optimize_threshold.py
│   └── evaluate_final.py
├── tests/
│   └── test_model.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── reports/
```

The dataset, generated model files, MLflow database, and reports are stored locally and excluded from Git tracking.

## 9. Installation and Setup

### Clone the repository

```bash
git clone https://github.com/sq6111/fraud-detection-mlops.git
cd fraud-detection-mlops
```

### Create a Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows, activate the environment using `.venv\Scripts\activate`.

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Download the dataset

Download `creditcard.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Create the dataset folder if necessary:

```bash
mkdir -p data/raw
```

Place the downloaded file at:

```text
data/raw/creditcard.csv
```

### Run the training pipeline

```bash
python setup_project.py
```

This script runs preprocessing, Logistic Regression training, XGBoost training, and threshold optimization.

It generates the local files needed for model inference, including the scaler, trained models, and selected threshold.

### Run final model evaluation

```bash
python src/evaluate_final.py
```

### Launch the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## 10. Streamlit Application

The application provides two prediction modes.

**Single Transaction Prediction**

Select a transaction from the test dataset, view the model score and classification, compare the prediction against its known label, and inspect the SHAP waterfall explanation.

**CSV Batch Prediction**

Upload a CSV containing raw `Time`, `V1`–`V28`, and `Amount` features. The application validates the input, applies the saved preprocessing scaler, predicts fraud scores, and provides a downloadable results CSV.

Uploaded features must be unscaled. The application is designed for this dataset's anonymized feature format.

## 11. Automated Testing

Run the model tests:

```bash
python -m pytest tests/test_model.py -v
```

The five initial tests verify model loading, threshold validity, prediction score range, and predictions for selected legitimate and fraudulent transactions.

**Result: 5 tests passed.**

## 12. Limitations

- Fraud represents only approximately 0.173% of the dataset.
- The anonymized features limit direct interpretation of transaction characteristics.
- Model scores are not calibrated real-world fraud probabilities.
- The selected threshold may not be optimal for other datasets or business requirements.
- The application is an educational demonstration and is not designed for production financial decision-making.

## 13. Future Improvements

- Probability calibration and cost-sensitive threshold selection
- More comprehensive preprocessing and integration tests
- CI/CD using GitHub Actions
- Automated model monitoring and data drift detection
- FastAPI inference service
- Cloud deployment

## Author

**Jahnavi Saladhagu**

GitHub: [sq6111](https://github.com/sq6111)

---

*Built as an end-to-end machine learning portfolio project demonstrating fraud detection, model explainability, experiment tracking, and reproducible ML workflows.*
