from pathlib import Path
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR /"data" / "raw" / "creditcard.csv")
X = df.drop("Class", axis = 1)
Y = df["Class"]
print("Features shape:",X.shape)
print("Target shape:",Y.shape)
print("\nTarget distribution:")
print(Y.value_counts())
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(
    X,
    Y,
    test_size = 0.2,
    random_state = 42,
    stratify = Y
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
columns_to_scale = ["Time" , "Amount"]
X_Train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_Train_scaled[columns_to_scale] = scaler.fit_transform(
    X_train[columns_to_scale]
)
X_test_scaled[columns_to_scale] = scaler.transform(
    X_test[columns_to_scale]
)
print("\n Scaled training data")
print(X_Train_scaled[columns_to_scale].head())
print("\n Scaled testing data")
print(X_test_scaled[columns_to_scale].head())
#check fraud distribution in training and testing data
print("\nTraining class distribution:")
print(Y_train.value_counts())
print("\nTraining class percentage:")
print(Y_train.value_counts(normalize=True) * 100)
print("\nTesting class distribution:")
print(Y_test.value_counts())
print("\nTesting class percentage:")
print(Y_test.value_counts(normalize = True) * 100)



# Create processed data folder
processed_dir = BASE_DIR / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# Save scaled features
X_Train_scaled.to_csv(processed_dir / "X_train.csv", index=False)
X_test_scaled.to_csv(processed_dir / "X_test.csv", index=False)

# Save target labels
Y_train.to_csv(processed_dir / "y_train.csv", index=False)
Y_test.to_csv(processed_dir / "y_test.csv", index=False)

print("\nProcessed datasets saved successfully!")
import joblib
models_dir = BASE_DIR / "models"
models_dir .mkdir(parents = True , exist_ok = True)
joblib.dump(scaler , models_dir /"scaler.pkl")
print("\nPreprocessing scaler saved successfully")
loaded_scaler = joblib. load(models_dir / "scaler.pkl")
print("scaler means:", loaded_scaler.mean_)
print("scaler scales:", loaded_scaler.scale_)