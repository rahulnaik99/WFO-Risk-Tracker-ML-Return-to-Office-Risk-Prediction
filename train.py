"""
Run this first before starting the API or UI.
Generates synthetic data + trains both models + saves best.

Usage: python train.py
"""
import os
import sys

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("Step 1/3 — Generating synthetic employee dataset...")
from app.ml.data_generator import generate_dataset
df = generate_dataset(n_samples=1000)
df.to_csv("data/employee_wfo_data.csv", index=False)
print(f"  ✓ {len(df)} employees generated → data/employee_wfo_data.csv")
print(f"  Risk distribution: {df['risk_category'].value_counts().to_dict()}")

print("\nStep 2/3 — Training Logistic Regression + Random Forest...")
from app.ml.trainer import train_and_compare
results = train_and_compare("data/employee_wfo_data.csv")

print("\nStep 3/3 — Summary")
lr = results["logistic_regression"]
rf = results["random_forest"]
best = results["best_model"]
print(f"  Logistic Regression → Accuracy: {lr['accuracy']} | ROC-AUC: {lr['roc_auc']}")
print(f"  Random Forest       → Accuracy: {rf['accuracy']} | ROC-AUC: {rf['roc_auc']}")
print(f"  ✓ Best model: {best.upper()} saved to models/best_model.pkl")
print("\n✅ Training complete. Now run:")
print("   python run.py          ← starts FastAPI on port 8000")
print("   streamlit run ui.py    ← starts dashboard on port 8501")
