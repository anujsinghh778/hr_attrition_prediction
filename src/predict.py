"""Inference script for the HR Analytics attrition prediction model.

Provides a prediction interface for single-employee profiles and batch datasets.
Uses a pure-Python feature pipeline to ensure maximum compatibility.
"""

import os
import pickle
import json
from typing import Dict, Any, Tuple, List, Union
import pandas as pd
import numpy as np

# Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PREPROCESSOR_META_PATH = os.path.join(MODELS_DIR, "preprocessor_meta.json")
MODEL_PATH = os.path.join(MODELS_DIR, "attrition_model.pkl")

# Global cached artifacts
_PREPROCESSOR_META = None
_MODEL = None


def load_model_artifacts() -> Tuple[Dict[str, Any], Any]:
    """Loads and caches the model and preprocessor metadata.

    Returns:
        A tuple of (preprocessor_metadata_dict, xgboost_model_object).
    """
    global _PREPROCESSOR_META, _MODEL
    
    if _PREPROCESSOR_META is None or _MODEL is None:
        if not os.path.exists(PREPROCESSOR_META_PATH) or not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Trained model or preprocessor metadata not found. Run train_model.py first.")
            
        with open(PREPROCESSOR_META_PATH, "r") as f:
            _PREPROCESSOR_META = json.load(f)
        with open(MODEL_PATH, "rb") as f:
            _MODEL = pickle.load(f)
            
    return _PREPROCESSOR_META, _MODEL


def predict_attrition(employee_data: Dict[str, Any]) -> Tuple[int, float]:
    """Predicts attrition risk and probability for a single employee.

    Args:
        employee_data: Dictionary containing employee features.

    Returns:
        A tuple of (binary prediction 0/1, probability of attrition 0.0-1.0).
    """
    meta, model = load_model_artifacts()
    training_cols = meta["training_cols"]
    scaling_params = meta["scaling_params"]
    feature_names = meta["feature_names"]
    categorical_cols = meta["categorical_cols"]
    
    # Convert input dict to DataFrame
    df = pd.DataFrame([employee_data])
    
    # Recalculate engineered features
    joblevel_averages = {1: 2800.0, 2: 5400.0, 3: 9800.0, 4: 15500.0, 5: 19100.0}
    if "income_ratio_to_joblevel_avg" not in df.columns:
        if "MonthlyIncome" in df.columns and "JobLevel" in df.columns:
            lvl = int(df.loc[0, "JobLevel"])
            avg_inc = joblevel_averages.get(lvl, 5000.0)
            df["income_ratio_to_joblevel_avg"] = df["MonthlyIncome"] / avg_inc
        else:
            df["income_ratio_to_joblevel_avg"] = 1.0
            
    if "role_tenure_ratio" not in df.columns:
        if "YearsInCurrentRole" in df.columns and "YearsAtCompany" in df.columns:
            df["role_tenure_ratio"] = df["YearsInCurrentRole"] / (df["YearsAtCompany"] + 1)
        else:
            df["role_tenure_ratio"] = 0.5
            
    # Ensure all expected columns are present
    for col in training_cols:
        if col not in df.columns:
            if col in meta["numerical_cols"]:
                df[col] = 0.0
            else:
                df[col] = "Unknown"
                
    # Sort columns to match training_cols
    df_sorted = df[training_cols]
    
    # Construct preprocessed features dictionary in pure Python
    record = df_sorted.iloc[0].to_dict()
    prep_dict = {feat: 0.0 for feat in feature_names}
    
    # 1. Scale numerical columns
    for col, params in scaling_params.items():
        val = float(record.get(col, 0.0))
        prep_dict[col] = (val - params["mean"]) / (params["scale"] + 1e-10)
        
    # 2. One-hot encode categorical columns
    for col in categorical_cols:
        val = str(record.get(col, ""))
        dummy_name = f"{col}_{val}"
        if dummy_name in prep_dict:
            prep_dict[dummy_name] = 1.0
            
    # Convert prep_dict to numpy array in correct feature order
    X_preprocessed = np.array([prep_dict[feat] for feat in feature_names]).reshape(1, -1)
    
    # Predict probability
    prob = float(model.predict_proba(X_preprocessed)[0][1])
    tuned_threshold = meta.get("tuned_threshold", 0.5)
    pred = 1 if prob >= tuned_threshold else 0
    
    return pred, prob


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Predicts attrition risks for a batch DataFrame of employees.

    Args:
        df: Input DataFrame containing employee profiles.

    Returns:
        DataFrame with two additional columns: AttritionPrediction (0/1) and AttritionRisk (0.0-1.0).
    """
    meta, model = load_model_artifacts()
    training_cols = meta["training_cols"]
    scaling_params = meta["scaling_params"]
    feature_names = meta["feature_names"]
    categorical_cols = meta["categorical_cols"]
    
    df_copy = df.copy()
    
    # Recalculate engineered features
    if len(df_copy) > 5 and "MonthlyIncome" in df_copy.columns and "JobLevel" in df_copy.columns:
        joblevel_avg = df_copy.groupby("JobLevel")["MonthlyIncome"].transform("mean")
        df_copy["income_ratio_to_joblevel_avg"] = df_copy["MonthlyIncome"] / (joblevel_avg + 1e-5)
    else:
        joblevel_averages = {1: 2800.0, 2: 5400.0, 3: 9800.0, 4: 15500.0, 5: 19100.0}
        def get_ratio(row: pd.Series) -> float:
            lvl = int(row.get("JobLevel", 1))
            avg = joblevel_averages.get(lvl, 5000.0)
            return row.get("MonthlyIncome", avg) / avg
        df_copy["income_ratio_to_joblevel_avg"] = df_copy.apply(get_ratio, axis=1)
        
    if "YearsInCurrentRole" in df_copy.columns and "YearsAtCompany" in df_copy.columns:
        df_copy["role_tenure_ratio"] = df_copy["YearsInCurrentRole"] / (df_copy["YearsAtCompany"] + 1)
    else:
        df_copy["role_tenure_ratio"] = 0.5
        
    # Standardize column existence
    for col in training_cols:
        if col not in df_copy.columns:
            if col in meta["numerical_cols"]:
                df_copy[col] = 0.0
            else:
                df_copy[col] = "Unknown"
                
    X_sorted = df_copy[training_cols]
    
    # 1. Scale numerical columns
    X_prep = pd.DataFrame(index=X_sorted.index)
    for col, params in scaling_params.items():
        X_prep[col] = (X_sorted[col] - params["mean"]) / (params["scale"] + 1e-10)
        
    # 2. Add dummy categorical columns and initialize to 0.0
    for feat in feature_names:
        if feat not in X_prep.columns:
            X_prep[feat] = 0.0
            
    for col in categorical_cols:
        for feat in feature_names:
            if feat.startswith(f"{col}_"):
                cat_val = feat[len(col)+1:]
                X_prep.loc[X_sorted[col] == cat_val, feat] = 1.0
                
    # Sort columns to match feature_names
    X_prep_arr = X_prep[feature_names].values
    
    # Predict
    probs = model.predict_proba(X_prep_arr)[:, 1]
    tuned_threshold = meta.get("tuned_threshold", 0.5)
    preds = (probs >= tuned_threshold).astype(int)
    
    df_copy["AttritionPrediction"] = preds
    df_copy["AttritionRisk"] = probs
    
    return df_copy


if __name__ == "__main__":
    # Small test prediction
    test_emp = {
        "Age": 35,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 800,
        "Department": "Research & Development",
        "DistanceFromHome": 10,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 1,
        "Gender": "Male",
        "HourlyRate": 60,
        "JobInvolvement": 2,
        "JobLevel": 2,
        "JobRole": "Laboratory Technician",
        "JobSatisfaction": 1,
        "MaritalStatus": "Single",
        "MonthlyIncome": 3500,
        "MonthlyRate": 12000,
        "NumCompaniesWorked": 1,
        "OverTime": "Yes",
        "PercentSalaryHike": 12,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 2,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 8,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 1,
        "YearsAtCompany": 4,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 2
    }
    try:
        pred, prob = predict_attrition(test_emp)
        print(f"Test Prediction for At-Risk Profile:")
        print(f"  Flagged: {pred} (Expected: 1)")
        print(f"  Risk Probability: {prob:.2%}")
    except Exception as e:
        print(f"CLI Prediction error: {e}")
