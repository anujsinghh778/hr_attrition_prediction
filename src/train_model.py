"""Model training and evaluation script for the HR Attrition project.

Implements feature engineering, a pipeline preprocessing column transformer,
class weight balancing (scale_pos_weight & class_weight='balanced'),
hyperparameter tuning via Stratified 5-Fold Grid Search, threshold tuning
for high recall, SHAP explainability, and metrics exports (ROC-AUC, PR-AUC, Confusion Matrix).
"""

import json
import os
import pickle
from typing import Dict, Tuple, Any, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Constants
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "hr_clean.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard")

# Categorical and numerical column lists
CATEGORICAL_COLS = ["BusinessTravel", "Department", "EducationField", "Gender", "MaritalStatus", "OverTime", "JobRole"]
NUMERICAL_COLS = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager", "income_ratio_to_joblevel_avg",
    "role_tenure_ratio"
]


def ensure_models_directory() -> None:
    """Creates output directories if they do not exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DASHBOARD_DIR, exist_ok=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies custom feature engineering formulas to the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with engineered features.
    """
    df_copy = df.copy()

    # Feature 1: Monthly income relative to job level average
    joblevel_avg = df_copy.groupby("JobLevel")["MonthlyIncome"].transform("mean")
    df_copy["income_ratio_to_joblevel_avg"] = df_copy["MonthlyIncome"] / (joblevel_avg + 1e-5)

    # Feature 2: Tenure ratio (role tenure relative to company tenure)
    df_copy["role_tenure_ratio"] = df_copy["YearsInCurrentRole"] / (df_copy["YearsAtCompany"] + 1)

    return df_copy


def build_preprocessor() -> ColumnTransformer:
    """Constructs the ColumnTransformer for scaling and encoding.

    Returns:
        A ColumnTransformer.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ]
    )


def evaluate_model_at_threshold(
    model: Any, X_test: np.ndarray, y_test: pd.Series, threshold: float = 0.5
) -> Dict[str, Any]:
    """Computes standard classification metrics for a model at a specific threshold.

    Args:
        model: Trained classifier.
        X_test: Preprocessed test features.
        y_test: Test labels.
        threshold: Classification probability threshold.

    Returns:
        A dictionary containing performance metrics and confusion matrix elements.
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1_score": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "pr_auc": float(average_precision_score(y_test, probs)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
    }
    return metrics


def find_optimal_threshold(probs: np.ndarray, y_true: pd.Series) -> float:
    """Finds the decision threshold that maximizes the F1-score.

    Uses out-of-fold or validation probabilities to prevent overfitting the threshold.

    Args:
        probs: Probability predictions for the positive class.
        y_true: True labels.

    Returns:
        The probability threshold that maximizes F1-score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    # Check bounds
    threshold_idx = min(best_idx, len(thresholds) - 1)
    return float(thresholds[threshold_idx])


def main() -> None:
    """Main execution function to train baseline and advanced models with hyperparameter search."""
    ensure_models_directory()

    # Step 1: Load data
    print(f"Loading data from {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Step 2: Feature engineering
    df_engineered = engineer_features(df)

    # Separate target & features
    X = df_engineered.drop(columns=["Attrition", "EmployeeNumber"], errors="ignore")
    y = df_engineered["Attrition"]

    # Step 3: Train-Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Step 4: Fit preprocessor
    preprocessor = build_preprocessor()
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()
    feature_names_clean = [name.split("__")[1] for name in feature_names]

    # Precalculate imbalance scale factor
    scale_factor = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Dataset imbalance scale factor (Negative/Positive Ratio): {scale_factor:.2f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Step 5: Grid Search Baseline Model (Logistic Regression)
    print("\n[1/3] Hyperparameter tuning for Logistic Regression baseline...")
    lr_param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "class_weight": ["balanced", None],
        "solver": ["liblinear", "lbfgs"]
    }
    lr_grid = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        param_grid=lr_param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )
    lr_grid.fit(X_train_preprocessed, y_train)
    best_lr = lr_grid.best_estimator_
    print(f"Best Logistic Regression parameters: {lr_grid.best_params_}")

    # Tune threshold for Logistic Regression using out-of-fold cross-validated probabilities to maximize F1-score
    lr_oof_probs = cross_val_predict(best_lr, X_train_preprocessed, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    lr_threshold = find_optimal_threshold(lr_oof_probs, y_train)
    print(f"Tuned Logistic Regression Threshold (OOF): {lr_threshold:.4f}")
    
    # Evaluate Baseline
    baseline_metrics = evaluate_model_at_threshold(best_lr, X_test_preprocessed, y_test, threshold=lr_threshold)
    baseline_metrics["selected_threshold"] = lr_threshold
    print("Logistic Regression Metrics:")
    print(f"  ROC-AUC: {baseline_metrics['roc_auc']:.2%}")
    print(f"  PR-AUC: {baseline_metrics['pr_auc']:.2%}")
    print(f"  Recall: {baseline_metrics['recall']:.2%}")
    print(f"  Precision: {baseline_metrics['precision']:.2%}")

    # Save Baseline
    with open(os.path.join(MODELS_DIR, "baseline_model.pkl"), "wb") as f:
        pickle.dump(best_lr, f)

    # Step 6: Grid Search Advanced Model (XGBoost)
    print("\n[2/3] Hyperparameter tuning for XGBoost Classifier...")
    xgb_param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "scale_pos_weight": [1.0, scale_factor, scale_factor * 0.75],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    
    xgb_grid = GridSearchCV(
        estimator=XGBClassifier(eval_metric="logloss", random_state=42),
        param_grid=xgb_param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )
    xgb_grid.fit(X_train_preprocessed, y_train)
    best_xgb = xgb_grid.best_estimator_
    print(f"Best XGBoost parameters: {xgb_grid.best_params_}")

    # Tune threshold for XGBoost using out-of-fold cross-validated probabilities to maximize F1-score
    xgb_oof_probs = cross_val_predict(best_xgb, X_train_preprocessed, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    xgb_threshold = find_optimal_threshold(xgb_oof_probs, y_train)
    print(f"Tuned XGBoost Threshold (OOF): {xgb_threshold:.4f}")

    # Evaluate XGBoost
    xgb_metrics = evaluate_model_at_threshold(best_xgb, X_test_preprocessed, y_test, threshold=xgb_threshold)
    xgb_metrics["selected_threshold"] = xgb_threshold
    print("XGBoost Metrics:")
    print(f"  ROC-AUC: {xgb_metrics['roc_auc']:.2%}")
    print(f"  PR-AUC: {xgb_metrics['pr_auc']:.2%}")
    print(f"  Recall: {xgb_metrics['recall']:.2%}")
    print(f"  Precision: {xgb_metrics['precision']:.2%}")

    # Save Advanced
    with open(os.path.join(MODELS_DIR, "attrition_model.pkl"), "wb") as f:
        pickle.dump(best_xgb, f)

    # Save preprocessor and features alongside the tuned threshold
    preprocessor_meta = {
        "preprocessor": preprocessor,
        "feature_names": feature_names_clean,
        "numerical_cols": NUMERICAL_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "training_cols": X.columns.tolist(),
        "tuned_threshold": xgb_threshold,
        "baseline_threshold": lr_threshold
    }
    with open(os.path.join(MODELS_DIR, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor_meta, f)
    print("\nPreprocessor metadata and tuned threshold saved.")

    # Save a small sample of training data for SHAP background in the app
    background_df = pd.DataFrame(X_train_preprocessed, columns=feature_names_clean).sample(
        n=min(100, len(X_train_preprocessed)), random_state=42
    )
    with open(os.path.join(MODELS_DIR, "shap_background.pkl"), "wb") as f:
        pickle.dump(background_df, f)

    # Save metrics report
    metrics_report = {
        "LogisticRegression": baseline_metrics,
        "XGBoost": xgb_metrics,
    }
    with open(os.path.join(MODELS_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics_report, f, indent=4)
    print("Metrics report saved.")

    # Step 7: Compute SHAP Explainer
    print("\n[3/3] Computing SHAP values for XGBoost model...")
    explainer = shap.TreeExplainer(best_xgb)
    with open(os.path.join(MODELS_DIR, "shap_explainer.pkl"), "wb") as f:
        pickle.dump(explainer, f)

    # Plot and save
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        explainer.shap_values(X_test_preprocessed),
        X_test_preprocessed,
        feature_names=feature_names_clean,
        show=False,
    )
    plt.tight_layout()
    shap_plot_path = os.path.join(DASHBOARD_DIR, "shap_summary.png")
    plt.savefig(shap_plot_path, dpi=300)
    plt.close()
    print(f"SHAP summary plot saved to {shap_plot_path}")

    print("\nModel Training and Threshold Optimization completed.")


if __name__ == "__main__":
    main()
