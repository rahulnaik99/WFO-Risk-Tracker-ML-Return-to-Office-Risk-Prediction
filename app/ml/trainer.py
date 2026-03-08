"""
Model training pipeline.
Trains Logistic Regression and Random Forest, compares both,
saves the best model + scaler to disk using joblib.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)

FEATURE_COLS = [
    "age",
    "commute_distance_km",
    "has_children_under_5",
    "vaccination_status",
    "prior_wfo_days_per_week",
    "home_internet_quality",
    "team_size",
    "manager_wfo",
    "anxiety_score",
    "productivity_wfh_score",
]
TARGET_COL = "wfo_risk_label"
MODELS_DIR = "models"


def load_data(csv_path: str) -> tuple:
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y, df


def train_and_compare(csv_path: str = "data/employee_wfo_data.csv") -> dict:
    """
    Train both models, compare, save best.
    Returns comparison metrics dict.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    X, y, df = load_data(csv_path)

    # Train/test split — stratified to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features — required for Logistic Regression, harmless for RF
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # ── Logistic Regression ──────────────────────────────────────────────────
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
    lr_cv = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring="roc_auc")

    results["logistic_regression"] = {
        "accuracy": round(accuracy_score(y_test, lr_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, lr_prob), 4),
        "cv_roc_auc_mean": round(lr_cv.mean(), 4),
        "cv_roc_auc_std": round(lr_cv.std(), 4),
        "classification_report": classification_report(y_test, lr_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, lr_pred).tolist(),
    }

    # ── Random Forest ────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_prob = rf.predict_proba(X_test_scaled)[:, 1]
    rf_cv = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring="roc_auc")

    # Feature importances
    feature_importances = dict(zip(FEATURE_COLS, rf.feature_importances_.round(4)))

    results["random_forest"] = {
        "accuracy": round(accuracy_score(y_test, rf_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, rf_prob), 4),
        "cv_roc_auc_mean": round(rf_cv.mean(), 4),
        "cv_roc_auc_std": round(rf_cv.std(), 4),
        "classification_report": classification_report(y_test, rf_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, rf_pred).tolist(),
        "feature_importances": feature_importances,
    }

    # ── Pick best model by CV ROC-AUC ────────────────────────────────────────
    best_name = max(
        ["logistic_regression", "random_forest"],
        key=lambda k: results[k]["cv_roc_auc_mean"]
    )
    best_model = lr if best_name == "logistic_regression" else rf
    results["best_model"] = best_name

    print(f"\n{'='*50}")
    print(f"LOGISTIC REGRESSION — Accuracy: {results['logistic_regression']['accuracy']} | ROC-AUC: {results['logistic_regression']['roc_auc']}")
    print(f"RANDOM FOREST       — Accuracy: {results['random_forest']['accuracy']} | ROC-AUC: {results['random_forest']['roc_auc']}")
    print(f"BEST MODEL: {best_name.upper()}")
    print(f"{'='*50}\n")

    # ── Save best model + scaler ─────────────────────────────────────────────
    joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    joblib.dump(lr, f"{MODELS_DIR}/logistic_regression.pkl")
    joblib.dump(rf, f"{MODELS_DIR}/random_forest.pkl")

    # Save metrics for API serving
    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Models saved to {MODELS_DIR}/")
    return results


if __name__ == "__main__":
    from app.ml.data_generator import generate_dataset
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/employee_wfo_data.csv", index=False)
    train_and_compare()
