"""
scripts/04_test_endpoint.py
===========================
Tests the FastAPI churn prediction endpoint with sample employee data.
Run after starting the API: uvicorn api.main:app --port 8000
Or via Docker: docker-compose up churn-api
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

# ── Sample employees ────────────────────────────────────────────────────────
SAMPLES = [
    {
        "label": "High-risk profile (overtime, low satisfaction, single, long commute)",
        "data": {
            "Age": 29,
            "MonthlyIncome": 2500,
            "YearsAtCompany": 2,
            "YearsInCurrentRole": 1,
            "YearsSinceLastPromotion": 2,
            "YearsWithCurrManager": 1,
            "TotalWorkingYears": 5,
            "JobLevel": 1,
            "JobSatisfaction": 1,
            "EnvironmentSatisfaction": 1,
            "RelationshipSatisfaction": 2,
            "WorkLifeBalance": 1,
            "OverTime": "Yes",
            "Department": "Sales",
            "JobRole": "Sales Representative",
            "MaritalStatus": "Single",
            "BusinessTravel": "Travel_Frequently",
            "EducationField": "Marketing",
            "Gender": "Male",
            "DistanceFromHome": 25,
            "Education": 2,
            "NumCompaniesWorked": 4,
            "PercentSalaryHike": 11,
            "StockOptionLevel": 0,
            "TrainingTimesLastYear": 1,
            "PerformanceRating": 3,
            "HourlyRate": 45,
            "DailyRate": 400,
            "MonthlyRate": 12000,
            "JobInvolvement": 2,
        }
    },
    {
        "label": "Low-risk profile (senior, high income, high satisfaction, no overtime)",
        "data": {
            "Age": 45,
            "MonthlyIncome": 12000,
            "YearsAtCompany": 15,
            "YearsInCurrentRole": 8,
            "YearsSinceLastPromotion": 2,
            "YearsWithCurrManager": 7,
            "TotalWorkingYears": 20,
            "JobLevel": 4,
            "JobSatisfaction": 4,
            "EnvironmentSatisfaction": 4,
            "RelationshipSatisfaction": 4,
            "WorkLifeBalance": 3,
            "OverTime": "No",
            "Department": "Research & Development",
            "JobRole": "Research Director",
            "MaritalStatus": "Married",
            "BusinessTravel": "Travel_Rarely",
            "EducationField": "Life Sciences",
            "Gender": "Female",
            "DistanceFromHome": 5,
            "Education": 5,
            "NumCompaniesWorked": 2,
            "PercentSalaryHike": 19,
            "StockOptionLevel": 3,
            "TrainingTimesLastYear": 3,
            "PerformanceRating": 4,
            "HourlyRate": 90,
            "DailyRate": 1200,
            "MonthlyRate": 25000,
            "JobInvolvement": 4,
        }
    },
]


def test_health():
    print("\n── Health Check ─────────────────────────────────────────────")
    r = requests.get(f"{BASE_URL}/")
    print(f"  Status: {r.status_code}")
    print(f"  Response: {json.dumps(r.json(), indent=2)}")
    assert r.status_code == 200, "Health check failed"


def test_model_info():
    print("\n── Model Info ───────────────────────────────────────────────")
    r = requests.get(f"{BASE_URL}/model/info")
    info = r.json()
    print(f"  Algorithm:  {info['algorithm']}")
    print(f"  Features:   {info['n_features']}")
    print(f"  Training:   {info['training_data']}")
    print(f"  Imbalance:  {info['imbalance_handling']}")


def test_single_predictions():
    print("\n── Single Predictions ───────────────────────────────────────")
    for sample in SAMPLES:
        r = requests.post(f"{BASE_URL}/predict", json=sample["data"])
        assert r.status_code == 200, f"Prediction failed: {r.text}"
        result = r.json()
        print(f"\n  Profile:    {sample['label']}")
        print(f"  Churn Prob: {result['ChurnProbability']:.2%}")
        print(f"  Risk Level: {result['ChurnRisk']}")
        print(f"  Will Leave: {'Yes' if result['PredictedAttrition'] else 'No'}")
        print(f"  Top Factors: {', '.join(result['TopRiskFactors'][:3])}")


def test_batch_prediction():
    print("\n── Batch Prediction ─────────────────────────────────────────")
    batch = [s["data"] for s in SAMPLES]
    r     = requests.post(f"{BASE_URL}/predict/batch", json=batch)
    assert r.status_code == 200, f"Batch failed: {r.text}"
    result = r.json()
    print(f"  Batch size: {result['count']}")
    for i, pred in enumerate(result["predictions"]):
        print(f"  Employee {i+1}: {pred['ChurnProbability']:.2%} ({pred['ChurnRisk']})")


def main():
    print("=" * 55)
    print("  Churn Prediction API — Endpoint Tests")
    print(f"  Target: {BASE_URL}")
    print("=" * 55)

    try:
        test_health()
        test_model_info()
        test_single_predictions()
        test_batch_prediction()
        print("\n[✓] All tests passed!\n")
    except requests.exceptions.ConnectionError:
        print(f"\n[!] Cannot connect to {BASE_URL}")
        print("    Start the API first:")
        print("    Option A: uvicorn api.main:app --host 0.0.0.0 --port 8000")
        print("    Option B: cd docker && docker-compose up churn-api\n")
    except AssertionError as e:
        print(f"\n[✗] Test failed: {e}\n")


if __name__ == "__main__":
    main()
