"""
api/main.py
===========
FastAPI REST endpoint for real-time churn prediction.
Loads trained XGBoost model from models/xgb_churn_model.joblib.

Endpoints:
  GET  /            → health check
  GET  /model/info  → feature names + model metadata
  POST /predict     → single employee churn prediction
  POST /predict/batch → batch predictions from JSON array
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import uvicorn

# ── Load model ─────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "xgb_churn_model.joblib"

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"[!] Model not found at {MODEL_PATH}\n"
        "    Run 02_mlflow_training.py first."
    )

bundle        = joblib.load(MODEL_PATH)
MODEL         = bundle["model"]
FEATURE_NAMES = bundle["feature_names"]
print(f"[✓] Model loaded — {len(FEATURE_NAMES)} features")

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="Real-time employee attrition risk scoring using XGBoost. "
                "Trained on IBM HR Analytics dataset (1,470 records, 35 features).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────
class EmployeeFeatures(BaseModel):
    """Raw employee features (pre-encoding). Most common fields exposed."""
    Age:                      int   = Field(..., ge=18, le=70, example=34)
    MonthlyIncome:            float = Field(..., ge=1000, example=5000)
    YearsAtCompany:           int   = Field(..., ge=0, example=5)
    YearsInCurrentRole:       int   = Field(..., ge=0, example=3)
    YearsSinceLastPromotion:  int   = Field(..., ge=0, example=1)
    YearsWithCurrManager:     int   = Field(..., ge=0, example=2)
    TotalWorkingYears:        int   = Field(..., ge=0, example=10)
    JobLevel:                 int   = Field(..., ge=1, le=5, example=2)
    JobSatisfaction:          int   = Field(..., ge=1, le=4, example=3)
    EnvironmentSatisfaction:  int   = Field(..., ge=1, le=4, example=3)
    RelationshipSatisfaction: int   = Field(..., ge=1, le=4, example=3)
    WorkLifeBalance:          int   = Field(..., ge=1, le=4, example=3)
    OverTime:                 str   = Field(..., example="No")   # "Yes" | "No"
    Department:               str   = Field(..., example="Sales")
    JobRole:                  str   = Field(..., example="Sales Executive")
    MaritalStatus:            str   = Field(..., example="Single")
    BusinessTravel:           str   = Field(..., example="Travel_Rarely")
    EducationField:           str   = Field(..., example="Life Sciences")
    Gender:                   str   = Field(..., example="Male")
    DistanceFromHome:         int   = Field(..., ge=1, example=5)
    Education:                int   = Field(..., ge=1, le=5, example=3)
    NumCompaniesWorked:       int   = Field(..., ge=0, example=2)
    PercentSalaryHike:        int   = Field(..., ge=0, example=14)
    StockOptionLevel:         int   = Field(..., ge=0, le=3, example=1)
    TrainingTimesLastYear:    int   = Field(..., ge=0, example=2)
    PerformanceRating:        int   = Field(..., ge=1, le=4, example=3)
    HourlyRate:               float = Field(..., ge=0, example=65)
    DailyRate:                float = Field(..., ge=0, example=800)
    MonthlyRate:              float = Field(..., ge=0, example=20000)
    JobInvolvement:           int   = Field(..., ge=1, le=4, example=3)


class PredictionResponse(BaseModel):
    ChurnProbability:  float
    ChurnRisk:         str        # "Low" | "Medium" | "High"
    PredictedAttrition: bool
    TopRiskFactors:    list[str]


def build_feature_vector(emp: EmployeeFeatures) -> pd.DataFrame:
    """
    Recreate the same feature engineering applied in 01_eda_preprocessing.py,
    then align to the exact columns the model was trained on.
    """
    d = emp.dict()

    # ── Engineered features ────────────────────────────────────────────────
    d["SalaryAgeRatio"]      = d["MonthlyIncome"] / max(d["Age"], 1)
    d["TenurePerRole"]       = (d["YearsInCurrentRole"] / d["YearsAtCompany"]
                                 if d["YearsAtCompany"] > 0 else 0)
    d["SatisfactionComposite"] = (
        d["JobSatisfaction"] + d["EnvironmentSatisfaction"] +
        d["RelationshipSatisfaction"] + d["WorkLifeBalance"]
    ) / 4.0
    d["PromotionLag"]        = d["YearsAtCompany"] - d["YearsSinceLastPromotion"]
    d["IncomeVsRoleAvg"]     = 1.0   # neutral; can't compute role avg at inference time
    d["OvertimeFlag"]        = 1 if d["OverTime"] == "Yes" else 0
    d["CareerVelocity"]      = (d["JobLevel"] / d["TotalWorkingYears"]
                                 if d["TotalWorkingYears"] > 0 else 0)
    d["ManagerTenureRatio"]  = (d["YearsWithCurrManager"] / d["YearsAtCompany"]
                                 if d["YearsAtCompany"] > 0 else 0)

    # ── One-hot encoding (match training schema) ───────────────────────────
    df = pd.DataFrame([d])
    df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0}).fillna(0)

    cat_cols = ["Department", "JobRole", "MaritalStatus",
                "BusinessTravel", "EducationField", "Gender"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Align to training feature set
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_NAMES]

    return df


def risk_label(prob: float) -> str:
    if prob >= 0.65:
        return "High"
    elif prob >= 0.35:
        return "Medium"
    return "Low"


def top_risk_factors(fv: pd.DataFrame, n: int = 5) -> list[str]:
    """Return top N features by contribution (feature value × importance)."""
    importances = MODEL.feature_importances_
    contribs    = np.abs(fv.values[0]) * importances
    top_idx     = np.argsort(contribs)[::-1][:n]
    return [FEATURE_NAMES[i] for i in top_idx]


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health():
    return {
        "status":    "ok",
        "model":     "XGBoost Churn Predictor",
        "version":   "1.0.0",
        "features":  len(FEATURE_NAMES),
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    return {
        "algorithm":     "XGBoost",
        "n_features":    len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "training_data": "IBM HR Analytics (1,470 records)",
        "imbalance_handling": "SMOTE",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(employee: EmployeeFeatures):
    try:
        fv   = build_feature_vector(employee)
        prob = float(MODEL.predict_proba(fv)[0][1])
        pred = bool(prob >= 0.5)
        return PredictionResponse(
            ChurnProbability   = round(prob, 4),
            ChurnRisk          = risk_label(prob),
            PredictedAttrition = pred,
            TopRiskFactors     = top_risk_factors(fv),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/batch", tags=["Inference"])
def predict_batch(employees: list[EmployeeFeatures]):
    if len(employees) > 500:
        raise HTTPException(status_code=400, detail="Batch size limit: 500")
    results = []
    for emp in employees:
        fv   = build_feature_vector(emp)
        prob = float(MODEL.predict_proba(fv)[0][1])
        results.append({
            "ChurnProbability":   round(prob, 4),
            "ChurnRisk":          risk_label(prob),
            "PredictedAttrition": bool(prob >= 0.5),
        })
    return {"count": len(results), "predictions": results}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
