"""Fairness and bias auditing script for the HR attrition prediction model.

Evaluates the model for disparate impact and predictive parity across protected attributes:
- Gender (Female vs Male)
- Age (Protected Age >= 40 vs Under 40)
Generates evaluation tables and exports them to JSON.
"""

import json
import os
import pickle
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

# Paths
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "hr_clean.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "attrition_model.pkl")
OUTPUT_REPORT_PATH = os.path.join(MODELS_DIR, "fairness_report.json")


def load_artifacts() -> Tuple[Any, Any, pd.DataFrame]:
    """Loads the model, preprocessor, and the clean dataset.

    Returns:
        A tuple of (model, preprocessor_meta, dataframe).
    """
    if not os.path.exists(PREPROCESSOR_PATH) or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model or Preprocessor files not found. Run train_model.py first.")
        
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor_meta = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return model, preprocessor_meta, df


def calculate_group_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Computes selection rate, True Positive Rate, and False Positive Rate.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Dictionary of selection rate, TPR, and FPR.
    """
    total = len(y_true)
    if total == 0:
        return {"selection_rate": 0.0, "tpr": 0.0, "fpr": 0.0}

    flagged = int(np.sum(y_pred == 1))
    selection_rate = flagged / total

    # True positives, False positives
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "selection_rate": float(selection_rate),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "count": total,
        "flagged": flagged
    }


def audit_fairness() -> Dict[str, Any]:
    """Audits the model's predictions for fairness across Gender and Age.

    Returns:
        A dictionary containing the audit findings.
    """
    model, prep_meta, df = load_artifacts()
    
    preprocessor = prep_meta["preprocessor"]
    training_cols = prep_meta["training_cols"]
    
    # Feature engineering for ratio features
    df_engineered = df.copy()
    joblevel_avg = df_engineered.groupby("JobLevel")["MonthlyIncome"].transform("mean")
    df_engineered["income_ratio_to_joblevel_avg"] = df_engineered["MonthlyIncome"] / (joblevel_avg + 1e-5)
    df_engineered["role_tenure_ratio"] = df_engineered["YearsInCurrentRole"] / (df_engineered["YearsAtCompany"] + 1)

    X = df_engineered[training_cols]
    y = df_engineered["Attrition"]
    
    # Preprocess and predict
    X_preprocessed = preprocessor.transform(X)
    y_prob = model.predict_proba(X_preprocessed)[:, 1]
    tuned_threshold = prep_meta.get("tuned_threshold", 0.5)
    y_pred = (y_prob >= tuned_threshold).astype(int)
    
    # Add predictions back for analysis
    audit_df = X.copy()
    audit_df["y_true"] = y.values
    audit_df["y_pred"] = y_pred
    audit_df["y_prob"] = y_prob
    
    report = {}
    
    # Protected Attribute 1: Gender (Female vs Male)
    female_mask = audit_df["Gender"] == "Female"
    male_mask = audit_df["Gender"] == "Male"
    
    female_metrics = calculate_group_metrics(audit_df.loc[female_mask, "y_true"].values, audit_df.loc[female_mask, "y_pred"].values)
    male_metrics = calculate_group_metrics(audit_df.loc[male_mask, "y_true"].values, audit_df.loc[male_mask, "y_pred"].values)
    
    # Disparate Impact Ratio (EEOC 80% Rule)
    # Target should ideally be between 0.8 and 1.25
    gender_di = (
        female_metrics["selection_rate"] / male_metrics["selection_rate"]
        if male_metrics["selection_rate"] > 0
        else 0.0
    )
    
    # Protected Attribute 2: Age (Under 40 vs 40 and Over)
    older_mask = audit_df["Age"] >= 40
    younger_mask = audit_df["Age"] < 40
    
    older_metrics = calculate_group_metrics(audit_df.loc[older_mask, "y_true"].values, audit_df.loc[older_mask, "y_pred"].values)
    younger_metrics = calculate_group_metrics(audit_df.loc[younger_mask, "y_true"].values, audit_df.loc[younger_mask, "y_pred"].values)
    
    age_di = (
        older_metrics["selection_rate"] / younger_metrics["selection_rate"]
        if younger_metrics["selection_rate"] > 0
        else 0.0
    )
    
    report["gender"] = {
        "Female": female_metrics,
        "Male": male_metrics,
        "disparate_impact_ratio": float(gender_di),
        "compliant_four_fifths": bool(0.80 <= gender_di <= 1.25),
        "fpr_difference": float(abs(female_metrics["fpr"] - male_metrics["fpr"]))
    }
    
    report["age"] = {
        "Age_40_Plus": older_metrics,
        "Age_Under_40": younger_metrics,
        "disparate_impact_ratio": float(age_di),
        "compliant_four_fifths": bool(0.80 <= age_di <= 1.25),
        "fpr_difference": float(abs(older_metrics["fpr"] - younger_metrics["fpr"]))
    }
    
    return report


def main() -> None:
    """Main execution function to run the fairness audit and save the report."""
    print("Executing Model Fairness and Bias Audit...")
    try:
        report = audit_fairness()
        
        # Display report in terminal
        print("\n=============================================")
        print("FAIRNESS AUDIT REPORT: GENDER")
        print("=============================================")
        g = report["gender"]
        print(f"Female Headcount: {g['Female']['count']} | Flagged: {g['Female']['flagged']} ({g['Female']['selection_rate']:.1%})")
        print(f"Male Headcount:   {g['Male']['count']}   | Flagged: {g['Male']['flagged']} ({g['Male']['selection_rate']:.1%})")
        print(f"Disparate Impact Ratio (Female / Male): {g['disparate_impact_ratio']:.3f}")
        print(f"EEOC 80% Rule Compliant: {g['compliant_four_fifths']}")
        print(f"False Positive Rate Difference: {g['fpr_difference']:.3f}")
        
        print("\n=============================================")
        print("FAIRNESS AUDIT REPORT: AGE (40+ vs <40)")
        print("=============================================")
        a = report["age"]
        print(f"Age 40+ Headcount:  {a['Age_40_Plus']['count']} | Flagged: {a['Age_40_Plus']['flagged']} ({a['Age_40_Plus']['selection_rate']:.1%})")
        print(f"Under 40 Headcount: {a['Age_Under_40']['count']} | Flagged: {a['Age_Under_40']['flagged']} ({a['Age_Under_40']['selection_rate']:.1%})")
        print(f"Disparate Impact Ratio (Older / Younger): {a['disparate_impact_ratio']:.3f}")
        print(f"EEOC 80% Rule Compliant: {a['compliant_four_fifths']}")
        print(f"False Positive Rate Difference: {a['fpr_difference']:.3f}")
        print("=============================================\n")
        
        # Save JSON
        with open(OUTPUT_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=4)
        print(f"Fairness report saved to {OUTPUT_REPORT_PATH}")
        
    except Exception as e:
        print(f"Fairness audit failed: {e}")


if __name__ == "__main__":
    main()
