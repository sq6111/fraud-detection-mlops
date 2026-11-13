
from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent

scripts = [
    "src/preprocess_data.py",
    "src/train_model.py",
    "src/train_xgboost.py",
    "src/optimize_threshold.py",
]

print("Starting fraud detection project setup...")

# Verify that the dataset exists
dataset_path = BASE_DIR / "data" / "raw" / "creditcard.csv"

if not dataset_path.exists():
    print("Dataset not found!")
    print("Please download creditcard.csv and place it in:")
    print(dataset_path)
    sys.exit(1)

# Run the training pipeline
for script in scripts:
    print(f"\nRunning {script}...", flush=True)

    subprocess.run(
        [sys.executable, str(BASE_DIR / script)],
        cwd=BASE_DIR,
        check=True
    )

print("\nProject setup completed successfully!")
print("Trained models and threshold are ready.")
print("Launch the app with:")
print("python -m streamlit run app.py")
