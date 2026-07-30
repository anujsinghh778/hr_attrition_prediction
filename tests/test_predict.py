"""Unit tests for the prediction inference script."""

import pandas as pd
import pytest

from src.predict import predict_attrition, predict_batch


def test_predict_single_employee() -> None:
    """Verifies that predict_attrition works for a single employee record."""
    test_emp = {
        "Age": 30,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 500,
        "Department": "Research & Development",
        "DistanceFromHome": 5,
        "Education": 3,
        "EducationField": "Medical",
        "EnvironmentSatisfaction": 3,
        "Gender": "Male",
        "HourlyRate": 70,
        "JobInvolvement": 3,
        "JobLevel": 1,
        "JobRole": "Research Scientist",
        "JobSatisfaction": 3,
        "MaritalStatus": "Married",
        "MonthlyIncome": 2500,
        "MonthlyRate": 10000,
        "NumCompaniesWorked": 1,
        "OverTime": "No",
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 5,
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 3,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 2
    }
    
    pred, prob = predict_attrition(test_emp)
    
    assert pred in [0, 1]
    assert 0.0 <= prob <= 1.0


def test_predict_batch() -> None:
    """Verifies that predict_batch processes multiple records and appends outputs."""
    # Create batch data (2 records)
    batch_df = pd.DataFrame([
        {
            "Age": 45, "BusinessTravel": "Travel_Rarely", "DailyRate": 1200,
            "Department": "Sales", "DistanceFromHome": 2, "Education": 4,
            "EducationField": "Marketing", "EnvironmentSatisfaction": 4, "Gender": "Female",
            "HourlyRate": 90, "JobInvolvement": 4, "JobLevel": 3, "JobRole": "Sales Executive",
            "JobSatisfaction": 4, "MaritalStatus": "Married", "MonthlyIncome": 9500,
            "MonthlyRate": 18000, "NumCompaniesWorked": 3, "OverTime": "No",
            "PercentSalaryHike": 18, "PerformanceRating": 3, "RelationshipSatisfaction": 4,
            "StockOptionLevel": 1, "TotalWorkingYears": 18, "TrainingTimesLastYear": 2,
            "WorkLifeBalance": 3, "YearsAtCompany": 10, "YearsInCurrentRole": 7,
            "YearsSinceLastPromotion": 4, "YearsWithCurrManager": 7
        },
        {
            "Age": 22, "BusinessTravel": "Travel_Frequently", "DailyRate": 300,
            "Department": "Research & Development", "DistanceFromHome": 25, "Education": 1,
            "EducationField": "Life Sciences", "EnvironmentSatisfaction": 1, "Gender": "Male",
            "HourlyRate": 40, "JobInvolvement": 1, "JobLevel": 1, "JobRole": "Laboratory Technician",
            "JobSatisfaction": 1, "MaritalStatus": "Single", "MonthlyIncome": 2100,
            "MonthlyRate": 6000, "NumCompaniesWorked": 0, "OverTime": "Yes",
            "PercentSalaryHike": 11, "PerformanceRating": 3, "RelationshipSatisfaction": 1,
            "StockOptionLevel": 0, "TotalWorkingYears": 1, "TrainingTimesLastYear": 1,
            "WorkLifeBalance": 1, "YearsAtCompany": 1, "YearsInCurrentRole": 0,
            "YearsSinceLastPromotion": 0, "YearsWithCurrManager": 0
        }
    ])
    
    res = predict_batch(batch_df)
    
    assert res.shape[0] == 2
    assert "AttritionPrediction" in res.columns
    assert "AttritionRisk" in res.columns
    
    # Assert high risk row gets higher probability than low risk row
    # Row 0: high income, married, high satisfaction, no overtime -> low risk
    # Row 1: low age, low income, single, low satisfaction, overtime, far commute -> high risk
    assert res.loc[1, "AttritionRisk"] > res.loc[0, "AttritionRisk"]
