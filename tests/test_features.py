"""
tests/test_features.py
======================
Unit tests for feature engineering, data validation, and model pipeline.
Run: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib.util, sys
_spec = importlib.util.spec_from_file_location(
    "eda_preprocessing", ROOT / "scripts" / "01_eda_preprocessing.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
engineer_features = _mod.engineer_features
encode_and_clean  = _mod.encode_and_clean


# ── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_employee():
    """Minimal valid employee record matching IBM dataset schema."""
    return pd.DataFrame([{
        "Age": 35,
        "Attrition": "No",
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 800,
        "Department": "Sales",
        "DistanceFromHome": 5,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EmployeeCount": 1,
        "EmployeeNumber": 1001,
        "EnvironmentSatisfaction": 3,
        "Gender": "Male",
        "HourlyRate": 65,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Sales Executive",
        "JobSatisfaction": 3,
        "MaritalStatus": "Married",
        "MonthlyIncome": 5000,
        "MonthlyRate": 20000,
        "NumCompaniesWorked": 2,
        "Over18": "Y",
        "OverTime": "No",
        "PercentSalaryHike": 14,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StandardHours": 80,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 10,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 2,
    }])


@pytest.fixture
def high_risk_employee():
    """High-risk employee profile."""
    return pd.DataFrame([{
        "Age": 25, "Attrition": "Yes", "BusinessTravel": "Travel_Frequently",
        "DailyRate": 400, "Department": "Sales", "DistanceFromHome": 25,
        "Education": 2, "EducationField": "Marketing", "EmployeeCount": 1,
        "EmployeeNumber": 2001, "EnvironmentSatisfaction": 1, "Gender": "Male",
        "HourlyRate": 40, "JobInvolvement": 2, "JobLevel": 1,
        "JobRole": "Sales Representative", "JobSatisfaction": 1,
        "MaritalStatus": "Single", "MonthlyIncome": 2000, "MonthlyRate": 10000,
        "NumCompaniesWorked": 5, "Over18": "Y", "OverTime": "Yes",
        "PercentSalaryHike": 10, "PerformanceRating": 3, "RelationshipSatisfaction": 2,
        "StandardHours": 80, "StockOptionLevel": 0, "TotalWorkingYears": 3,
        "TrainingTimesLastYear": 1, "WorkLifeBalance": 1, "YearsAtCompany": 1,
        "YearsInCurrentRole": 1, "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 0,
    }])


# ── Feature Engineering Tests ───────────────────────────────────────────────
class TestFeatureEngineering:

    def test_salary_age_ratio_computed(self, sample_employee):
        result = engineer_features(sample_employee)
        assert "SalaryAgeRatio" in result.columns
        expected = 5000 / 35
        assert abs(result["SalaryAgeRatio"].iloc[0] - expected) < 0.01

    def test_tenure_per_role_bounded(self, sample_employee):
        result = engineer_features(sample_employee)
        assert "TenurePerRole" in result.columns
        val = result["TenurePerRole"].iloc[0]
        assert 0.0 <= val <= 1.0, f"TenurePerRole should be 0-1, got {val}"

    def test_satisfaction_composite_range(self, sample_employee):
        result = engineer_features(sample_employee)
        assert "SatisfactionComposite" in result.columns
        val = result["SatisfactionComposite"].iloc[0]
        assert 1.0 <= val <= 4.0, f"Composite score should be 1-4, got {val}"

    def test_satisfaction_composite_correctness(self, sample_employee):
        result = engineer_features(sample_employee)
        # (3+3+3+3)/4 = 3.0
        assert result["SatisfactionComposite"].iloc[0] == 3.0

    def test_overtime_flag_encoding(self, sample_employee, high_risk_employee):
        low  = engineer_features(sample_employee)
        high = engineer_features(high_risk_employee)
        assert low["OvertimeFlag"].iloc[0]  == 0
        assert high["OvertimeFlag"].iloc[0] == 1

    def test_promotion_lag_non_negative(self, sample_employee):
        result = engineer_features(sample_employee)
        assert result["PromotionLag"].iloc[0] >= 0

    def test_career_velocity_non_negative(self, sample_employee):
        result = engineer_features(sample_employee)
        assert result["CareerVelocity"].iloc[0] >= 0

    def test_manager_tenure_ratio_bounded(self, sample_employee):
        result = engineer_features(sample_employee)
        val = result["ManagerTenureRatio"].iloc[0]
        assert 0.0 <= val <= 1.0

    def test_all_8_features_present(self, sample_employee):
        result = engineer_features(sample_employee)
        expected = ["SalaryAgeRatio", "TenurePerRole", "SatisfactionComposite",
                    "PromotionLag", "IncomeVsRoleAvg", "OvertimeFlag",
                    "CareerVelocity", "ManagerTenureRatio"]
        for feat in expected:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_attrition_binary_encoded(self, sample_employee, high_risk_employee):
        low  = engineer_features(sample_employee)
        high = engineer_features(high_risk_employee)
        assert low["AttritionBinary"].iloc[0]  == 0
        assert high["AttritionBinary"].iloc[0] == 1

    def test_zero_years_at_company_no_division_error(self):
        """Edge case: employee with 0 YearsAtCompany."""
        emp = pd.DataFrame([{
            "Age": 22, "Attrition": "No", "BusinessTravel": "Travel_Rarely",
            "DailyRate": 500, "Department": "HR", "DistanceFromHome": 3,
            "Education": 2, "EducationField": "Human Resources",
            "EmployeeCount": 1, "EmployeeNumber": 3001,
            "EnvironmentSatisfaction": 2, "Gender": "Female", "HourlyRate": 50,
            "JobInvolvement": 2, "JobLevel": 1, "JobRole": "Human Resources",
            "JobSatisfaction": 2, "MaritalStatus": "Single",
            "MonthlyIncome": 2500, "MonthlyRate": 8000, "NumCompaniesWorked": 0,
            "Over18": "Y", "OverTime": "No", "PercentSalaryHike": 11,
            "PerformanceRating": 3, "RelationshipSatisfaction": 3,
            "StandardHours": 80, "StockOptionLevel": 0, "TotalWorkingYears": 0,
            "TrainingTimesLastYear": 1, "WorkLifeBalance": 3,
            "YearsAtCompany": 0, "YearsInCurrentRole": 0,
            "YearsSinceLastPromotion": 0, "YearsWithCurrManager": 0,
        }])
        result = engineer_features(emp)
        assert result["TenurePerRole"].iloc[0]      == 0.0
        assert result["CareerVelocity"].iloc[0]     == 0.0
        assert result["ManagerTenureRatio"].iloc[0] == 0.0


# ── Encode & Clean Tests ────────────────────────────────────────────────────
class TestEncodeAndClean:

    def test_drop_leakage_columns(self, sample_employee):
        df     = engineer_features(sample_employee)
        result = encode_and_clean(df)
        for col in ["Attrition", "EmployeeCount", "EmployeeNumber",
                    "Over18", "StandardHours"]:
            assert col not in result.columns, f"Leakage column not dropped: {col}"

    def test_no_object_columns_remain(self, sample_employee):
        df     = engineer_features(sample_employee)
        result = encode_and_clean(df)
        obj_cols = result.select_dtypes(include="object").columns.tolist()
        assert len(obj_cols) == 0, f"Object columns remain: {obj_cols}"

    def test_target_column_present(self, sample_employee):
        df     = engineer_features(sample_employee)
        result = encode_and_clean(df)
        assert "AttritionBinary" in result.columns

    def test_no_missing_values(self, sample_employee):
        df     = engineer_features(sample_employee)
        result = encode_and_clean(df)
        assert result.isnull().sum().sum() == 0


# ── Data Validation Tests ───────────────────────────────────────────────────
class TestDataValidation:

    def test_high_risk_has_lower_satisfaction(self, sample_employee, high_risk_employee):
        low  = engineer_features(sample_employee)
        high = engineer_features(high_risk_employee)
        assert high["SatisfactionComposite"].iloc[0] < low["SatisfactionComposite"].iloc[0]

    def test_high_risk_has_lower_income(self, sample_employee, high_risk_employee):
        assert high_risk_employee["MonthlyIncome"].iloc[0] < sample_employee["MonthlyIncome"].iloc[0]

    def test_income_vs_role_avg_positive(self, sample_employee):
        result = engineer_features(sample_employee)
        assert result["IncomeVsRoleAvg"].iloc[0] > 0
