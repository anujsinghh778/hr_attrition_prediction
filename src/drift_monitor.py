"""Data drift monitoring script for the HR Analytics project.

Uses statistical tests (Kolmogorov-Smirnov for numerical variables,
Chi-Square contingency for categorical variables) to detect shifts in feature
distributions between training data and incoming operational data.
"""

import json
import os
import pickle
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

# Paths
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "hr_clean.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")


def load_reference_columns() -> Tuple[List[str], List[str], List[str]]:
    """Loads feature categories from preprocessor metadata.

    Returns:
        A tuple of (numerical_columns, categorical_columns, all_training_columns).
    """
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Preprocessor file not found. Run train_model.py first.")
        
    with open(PREPROCESSOR_PATH, "rb") as f:
        meta = pickle.load(f)
        
    return meta["numerical_cols"], meta["categorical_cols"], meta["training_cols"]


def analyze_drift(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, alpha: float = 0.05
) -> Dict[str, Any]:
    """Tests all features for distribution drift.

    Uses KS test for numerical and Chi-Square Contingency for categorical features.

    Args:
        reference_df: Historical training baseline DataFrame.
        current_df: Current incoming employee DataFrame.
        alpha: Statistical significance threshold.

    Returns:
        A dictionary with detailed drift metrics per feature.
    """
    num_cols, cat_cols, _ = load_reference_columns()
    
    drift_report = {}
    drifting_features_count = 0
    
    # Check numerical columns
    for col in num_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
            
        ref_data = reference_df[col].dropna()
        cur_data = current_df[col].dropna()
        
        # KS Test
        stat, p_val = ks_2samp(ref_data, cur_data)
        has_drift = bool(p_val < alpha)
        if has_drift:
            drifting_features_count += 1
            
        drift_report[col] = {
            "type": "numerical",
            "test_name": "Kolmogorov-Smirnov",
            "statistic": float(stat),
            "p_value": float(p_val),
            "drift_detected": has_drift
        }
        
    # Check categorical columns
    for col in cat_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
            
        ref_counts = reference_df[col].value_counts()
        cur_counts = current_df[col].value_counts()
        
        # Build contingency table
        combined_index = ref_counts.index.union(cur_counts.index)
        contingency = pd.DataFrame(index=combined_index)
        contingency["ref"] = contingency.index.map(ref_counts).fillna(0) + 1  # Add constant to avoid zero
        contingency["cur"] = contingency.index.map(cur_counts).fillna(0) + 1
        
        # Chi-Square Test
        try:
            stat, p_val, _, _ = chi2_contingency(contingency.values)
            has_drift = bool(p_val < alpha)
        except Exception:
            stat, p_val = 0.0, 1.0
            has_drift = False
            
        if has_drift:
            drifting_features_count += 1
            
        drift_report[col] = {
            "type": "categorical",
            "test_name": "Chi-Square Contingency",
            "statistic": float(stat),
            "p_value": float(p_val),
            "drift_detected": has_drift
        }
        
    total_checked = len(drift_report)
    drift_fraction = drifting_features_count / total_checked if total_checked > 0 else 0.0
    
    summary = {
        "drift_detected": bool(drift_fraction >= 0.25),  # Warn if > 25% of features drift
        "drifting_features_percentage": float(drift_fraction * 100),
        "drifting_features_count": drifting_features_count,
        "total_features_checked": total_checked,
        "details": drift_report
    }
    
    return summary


def simulate_drifted_data(df: pd.DataFrame, severity: float = 1.0) -> pd.DataFrame:
    """Modifies features in a copy of the dataframe to simulate real-world drift.

    Args:
        df: Input clean DataFrame.
        severity: Drift magnitude (0 to 1).

    Returns:
        A drifted copy of the DataFrame.
    """
    df_drifted = df.copy()
    np.random.seed(99)
    
    if severity <= 0:
        return df_drifted
        
    # Simulate drift 1: DistanceFromHome increases (commute burnout)
    shift_dist = int(5 * severity)
    df_drifted["DistanceFromHome"] = df_drifted["DistanceFromHome"] + shift_dist
    
    # Simulate drift 2: MonthlyIncome drops relative to average
    income_cut = 1 - (0.15 * severity)
    df_drifted["MonthlyIncome"] = (df_drifted["MonthlyIncome"] * income_cut).astype(int)
    
    # Simulate drift 3: JobSatisfaction shifts lower
    # Shift probability toward satisfaction = 1
    satisfaction_probs = {
        4: 0.15,
        3: 0.25,
        2: 0.30,
        1: 0.30
    }
    indices = df_drifted.sample(frac=0.5 * severity).index
    df_drifted.loc[indices, "JobSatisfaction"] = np.random.choice(
        [1, 2, 3, 4], size=len(indices), p=[0.35, 0.35, 0.20, 0.10]
    )
    
    # Simulate drift 4: OverTime hours rise
    ot_indices = df_drifted[df_drifted["OverTime"] == "No"].sample(frac=0.4 * severity).index
    df_drifted.loc[ot_indices, "OverTime"] = "Yes"
    
    return df_drifted


def main() -> None:
    """Main execution function to test drift detection on a simulated shifted set."""
    print("Testing Data Drift Monitor Pipeline...")
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Processed dataset not found at {PROCESSED_DATA_PATH}. Run data_prep.py first.")
        return
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Split into mock baseline and current
    baseline_df = df.iloc[:1000].copy()
    normal_current_df = df.iloc[1000:].copy()
    
    # 1. Test clean data (should show zero/low drift)
    clean_report = analyze_drift(baseline_df, normal_current_df)
    print("\n--- [Test 1] Comparing Baseline to Normal Current Set ---")
    print(f"Drifting features: {clean_report['drifting_features_count']}/{clean_report['total_features_checked']}")
    print(f"Drift Percentage: {clean_report['drifting_features_percentage']:.2f}%")
    print(f"Alert Status (Drift Detected): {clean_report['drift_detected']}")
    
    # 2. Test drifted data
    drifted_current_df = simulate_drifted_data(normal_current_df, severity=1.0)
    drift_report = analyze_drift(baseline_df, drifted_current_df)
    print("\n--- [Test 2] Comparing Baseline to Simulated Drifted Set ---")
    print(f"Drifting features: {drift_report['drifting_features_count']}/{drift_report['total_features_checked']}")
    print(f"Drift Percentage: {drift_report['drifting_features_percentage']:.2f}%")
    print(f"Alert Status (Drift Detected): {drift_report['drift_detected']}")
    
    # List drifting columns
    drifting_cols = [col for col, d in drift_report["details"].items() if d["drift_detected"]]
    print(f"Drifting columns: {drifting_cols}")
    
    # Save test report
    with open(os.path.join(MODELS_DIR, "drift_report_sample.json"), "w") as f:
        json.dump(drift_report, f, indent=4)
        
    print("\nDrift Monitoring Pipeline completed.")


if __name__ == "__main__":
    main()
