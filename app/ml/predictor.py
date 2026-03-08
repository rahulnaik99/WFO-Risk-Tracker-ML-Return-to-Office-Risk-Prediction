"""
Inference engine — loads trained model + scaler, runs predictions.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

MODELS_DIR = Path("models")
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

# Lazy-loaded singletons
_model = None
_scaler = None


def _load_artifacts():
    global _model, _scaler
    if _model is None:
        model_path = MODELS_DIR / "best_model.pkl"
        scaler_path = MODELS_DIR / "scaler.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                "Model not found. Run: python train.py"
            )
        _model = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)


def predict_single(features: dict) -> dict:
    """
    Predict WFO risk for a single employee.
    Returns risk_score (0-1), risk_category, label.
    """
    _load_artifacts()

    # Build feature row in correct column order
    row = pd.DataFrame([{col: features[col] for col in FEATURE_COLS}])
    row_scaled = _scaler.transform(row)

    prob = _model.predict_proba(row_scaled)[0][1]  # probability of high risk
    label = int(_model.predict(row_scaled)[0])

    if prob < 0.33:
        category = "Low"
    elif prob < 0.66:
        category = "Medium"
    else:
        category = "High"

    return {
        "risk_score": round(float(prob), 4),
        "risk_score_pct": round(float(prob) * 100, 1),
        "risk_category": category,
        "wfo_risk_label": label,
        "recommendation": _get_recommendation(category, features),
    }


def predict_batch(employees: list[dict]) -> list[dict]:
    """Predict for a list of employees."""
    _load_artifacts()

    rows = pd.DataFrame([
        {col: emp[col] for col in FEATURE_COLS}
        for emp in employees
    ])
    rows_scaled = _scaler.transform(rows)
    probs = _model.predict_proba(rows_scaled)[:, 1]
    labels = _model.predict(rows_scaled)

    results = []
    for i, emp in enumerate(employees):
        prob = float(probs[i])
        category = "Low" if prob < 0.33 else ("Medium" if prob < 0.66 else "High")
        results.append({
            "employee_id": emp.get("employee_id", f"EMP{i+1}"),
            "risk_score": round(prob, 4),
            "risk_score_pct": round(prob * 100, 1),
            "risk_category": category,
            "wfo_risk_label": int(labels[i]),
        })
    return results


def _get_recommendation(category: str, features: dict) -> str:
    """Generate human-readable recommendation based on risk factors."""
    if category == "Low":
        return "Employee is likely to return to office. Standard onboarding support recommended."
    elif category == "Medium":
        return (
            "Moderate WFO risk. Consider flexible hybrid arrangement. "
            + ("Long commute may be a factor — consider transport support. " if features.get("commute_distance_km", 0) > 40 else "")
            + ("Childcare support may help. " if features.get("has_children_under_5") else "")
        )
    else:
        high_factors = []
        if features.get("commute_distance_km", 0) > 50:
            high_factors.append("long commute")
        if features.get("has_children_under_5"):
            high_factors.append("childcare responsibilities")
        if features.get("anxiety_score", 0) > 7:
            high_factors.append("high anxiety score")
        if features.get("productivity_wfh_score", 0) > 7:
            high_factors.append("high WFH productivity")
        factors_str = ", ".join(high_factors) if high_factors else "multiple factors"
        return f"High WFO risk due to {factors_str}. Recommend HR conversation and flexible policy discussion."
