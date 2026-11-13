import pandas as pd
from pathlib import Path

# ==========================================
# 1. LOAD DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "data" / "raw" / "creditcard.csv")

print("\nFirst 5 Rows:")
print(df.head())


# ==========================================
# 2. DATASET SHAPE
# ==========================================

print("\nDataset Shape:")
print(df.shape)


# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 4. CHECK DATA TYPES
# ==========================================

print("\nData Types:")
print(df.dtypes)


# ==========================================
# 5. CLASS DISTRIBUTION
# ==========================================

print("\nClass Distribution:")
print(df["Class"].value_counts())


# ==========================================
# 6. CLASS DISTRIBUTION PERCENTAGE
# ==========================================

print("\nClass Distribution Percentage:")
print(df["Class"].value_counts(normalize=True) * 100)


# ==========================================
# 7. DESCRIPTIVE STATISTICS
# ==========================================

print("\nDataset Statistics:")
print(df.describe())


# ==========================================
# 8. TRANSACTION AMOUNT STATISTICS
# ==========================================

print("\nTransaction Amount Statistics:")

print("Mean:", df["Amount"].mean())

print("Median:", df["Amount"].median())

print("Minimum:", df["Amount"].min())

print("Maximum:", df["Amount"].max())

print("Standard Deviation:", df["Amount"].std())


# ==========================================
# 9. AVERAGE TRANSACTION AMOUNT BY CLASS
# ==========================================

print("\nAverage Transaction Amount by Class:")
print(df.groupby("Class")["Amount"].mean())


# ==========================================
# 10. MEDIAN TRANSACTION AMOUNT BY CLASS
# ==========================================

print("\nMedian Transaction Amount by Class:")
print(df.groupby("Class")["Amount"].median())
#calculate quartiles
Q1 = df["Amount"].quantile(0.25)
Q3 = df["Amount"].quantile(0.75)
#calculate idr
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("upper limit:",upper_limit)
outliers = df[df["Amount"] > upper_limit]
print("Total outliers:",len(outliers))
print("Outlier percentage:",len(outliers) / len(df) * 100)
print("\nOutlier Class Distribution:")
print(outliers["Class"].value_counts())
fraud_outliers = outliers[outliers["Class"] == 1]
fraud_percentage = len(fraud_outliers) / len(outliers) * 100
print("Fraudulent outliers:", len(fraud_outliers))
print("Fraud Percentage Among Outliers:", fraud_percentage)
correlation = df.corr(numeric_only = True)
fraud_correlation = correlation["Class"].sort_values(ascending =False)
print("\nCorrelation with fraud:")
print(fraud_correlation)