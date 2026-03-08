"""
WFO Risk Prediction API
FastAPI service exposing ML model predictions.
"""

import json
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from app.ml.predictor import predict_single, predict_batch

app = FastAPI(
    title="WFO Risk Prediction API",
    description="ML-powered employee return-to-office risk scoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ───────────────────────────────────────────────────────────

class EmployeeFeatures(BaseModel):
    employee_id: Optional[str] = "EMP0000"
    age: int = Field(..., ge=18, le=70, description="Employee age")
    commute_distance_km: float = Field(..., ge=0, le=200)
    has_children_under_5: int = Field(..., ge=0, le=1, description="0 or 1")
    vaccination_status: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    prior_wfo_days_per_week: int = Field(..., ge=0, le=5)
    home_internet_quality: int = Field(..., ge=1, le=10)
    team_size: int = Field(..., ge=1, le=200)
    manager_wfo: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    anxiety_score: int = Field(..., ge=1, le=10)
    productivity_wfh_score: int = Field(..., ge=1, le=10)

    class Config:
        json_schema_extra = {
            "example": {
                "employee_id": "EMP0042",
                "age": 34,
                "commute_distance_km": 45.0,
                "has_children_under_5": 1,
                "vaccination_status": 1,
                "prior_wfo_days_per_week": 3,
                "home_internet_quality": 8,
                "team_size": 12,
                "manager_wfo": 1,
                "anxiety_score": 6,
                "productivity_wfh_score": 7,
            }
        }


class PredictionResponse(BaseModel):
    employee_id: str
    risk_score: float
    risk_score_pct: float
    risk_category: str
    wfo_risk_label: int
    recommendation: str
    latency_ms: int


class BatchRequest(BaseModel):
    employees: List[EmployeeFeatures]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    model_ready = Path("models/best_model.pkl").exists()
    return {
        "status": "ok" if model_ready else "model_not_trained",
        "model_ready": model_ready,
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(employee: EmployeeFeatures):
    """Predict WFO risk for a single employee."""
    start = time.time()
    try:
        result = predict_single(employee.dict())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.time() - start) * 1000)
    return PredictionResponse(
        employee_id=employee.employee_id,
        latency_ms=latency,
        **result,
    )


@app.post("/predict/batch")
def predict_batch_endpoint(request: BatchRequest):
    """Predict WFO risk for multiple employees at once."""
    start = time.time()
    try:
        employees_dicts = [e.dict() for e in request.employees]
        results = predict_batch(employees_dicts)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.time() - start) * 1000)
    return {
        "total": len(results),
        "latency_ms": latency,
        "predictions": results,
        "summary": {
            "high_risk": sum(1 for r in results if r["risk_category"] == "High"),
            "medium_risk": sum(1 for r in results if r["risk_category"] == "Medium"),
            "low_risk": sum(1 for r in results if r["risk_category"] == "Low"),
        }
    }


@app.get("/model/metrics")
def model_metrics():
    """Return training metrics for both models."""
    metrics_path = Path("models/metrics.json")
    if not metrics_path.exists():
        raise HTTPException(status_code=503, detail="Model not trained yet. Run: python train.py")
    with open(metrics_path) as f:
        return json.load(f)


@app.get("/model/features")
def feature_info():
    """Return feature descriptions."""
    return {
        "features": [
            {"name": "age", "type": "int", "range": "18-70"},
            {"name": "commute_distance_km", "type": "float", "range": "0-200"},
            {"name": "has_children_under_5", "type": "binary", "values": "0 or 1"},
            {"name": "vaccination_status", "type": "binary", "values": "0=No, 1=Yes"},
            {"name": "prior_wfo_days_per_week", "type": "int", "range": "0-5"},
            {"name": "home_internet_quality", "type": "int", "range": "1-10"},
            {"name": "team_size", "type": "int", "range": "1-200"},
            {"name": "manager_wfo", "type": "binary", "values": "0=No, 1=Yes"},
            {"name": "anxiety_score", "type": "int", "range": "1-10"},
            {"name": "productivity_wfh_score", "type": "int", "range": "1-10"},
        ]
    }
