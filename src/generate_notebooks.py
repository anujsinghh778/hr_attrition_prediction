"""Jupyter notebook generator for the HR Attrition project.

Programmatically writes the three project notebooks:
- notebooks/01_eda.ipynb
- notebooks/02_feature_engineering.ipynb
- notebooks/03_modeling.ipynb
with detailed code blocks and analysis commentary.
"""

import json
import os
from typing import Dict, Any, List

# Paths
NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")


def create_cell(cell_type: str, source: List[str]) -> Dict[str, Any]:
    """Helper to create a single notebook cell.

    Args:
        cell_type: 'markdown' or 'code'.
        source: List of string lines for cell content.

    Returns:
        Cell dictionary object.
    """
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def write_notebook(dest_path: str, title: str, cells: List[Dict[str, Any]]) -> None:
    """Writes cells list to a standard .ipynb file.

    Args:
        dest_path: Target path for the notebook.
        title: Title of the notebook.
        cells: List of cell dictionaries.
    """
    notebook = {
        "cells": [
            create_cell("markdown", [f"# {title}", "This notebook was programmatically generated to demonstrate key project steps."]),
            *cells
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Notebook generated successfully at: {dest_path}")


def generate_eda_notebook() -> None:
    """Generates the Exploratory Data Analysis notebook."""
    dest = os.path.join(NOTEBOOKS_DIR, "01_eda.ipynb")
    cells = [
        create_cell("markdown", [
            "## 1. Load and Inspect Cleaned Data",
            "First, we load the processed dataset and view its basic shape and features."
        ]),
        create_cell("code", [
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "",
            "df = pd.read_csv('../data/processed/hr_clean.csv')",
            "print(f'Dataset Shape: {df.shape}')",
            "df.head()"
        ]),
        create_cell("markdown", [
            "## 2. Turnover (Attrition Rate) Overview",
            "Let's see what percentage of employees have left during the audit window."
        ]),
        create_cell("code", [
            "attrition_rate = df['Attrition'].mean()",
            "print(f'Overall attrition rate: {attrition_rate:.2%}')"
        ]),
        create_cell("markdown", [
            "## 3. Department-wise Attrition",
            "Let's check turnover rate across departments."
        ]),
        create_cell("code", [
            "dept_attrition = df.groupby('Department')['Attrition'].mean().sort_values(ascending=False)",
            "plt.figure(figsize=(8, 4))",
            "sns.barplot(x=dept_attrition.index, y=dept_attrition.values, palette='viridis')",
            "plt.ylabel('Attrition Rate')",
            "plt.title('Attrition Rate by Department')",
            "plt.show()"
        ]),
        create_cell("markdown", [
            "## 4. Salary Trends vs Attrition",
            "Checking correlation between MonthlyIncome and flight risk."
        ]),
        create_cell("code", [
            "plt.figure(figsize=(8, 5))",
            "sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, palette='Set2')",
            "plt.title('Monthly Income Distribution by Attrition')",
            "plt.show()"
        ]),
        create_cell("markdown", [
            "## 5. Work-Life Balance and Overtime Analysis",
            "Checking two major predictors of employee burnout: WorkLifeBalance score and OverTime."
        ]),
        create_cell("code", [
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
            "sns.barplot(x='OverTime', y='Attrition', data=df, ax=axes[0], palette='pastel')",
            "axes[0].set_title('Attrition Rate by Overtime status')",
            "",
            "sns.barplot(x='WorkLifeBalance', y='Attrition', data=df, ax=axes[1], palette='rocket')",
            "axes[1].set_title('Attrition Rate by Work-Life Balance Rating')",
            "plt.show()"
        ])
    ]
    write_notebook(dest, "01 Exploratory Data Analysis (EDA)", cells)


def generate_fe_notebook() -> None:
    """Generates the Feature Engineering notebook."""
    dest = os.path.join(NOTEBOOKS_DIR, "02_feature_engineering.ipynb")
    cells = [
        create_cell("markdown", [
            "## 1. Load Data",
            "We load the clean data and add our business-driven custom feature formulas."
        ]),
        create_cell("code", [
            "import pandas as pd",
            "import numpy as np",
            "",
            "df = pd.read_csv('../data/processed/hr_clean.csv')",
            "df.head()"
        ]),
        create_cell("markdown", [
            "## 2. Feature Engineering",
            "We calculate two custom features:",
            "1. **income_ratio_to_joblevel_avg**: Income relative to peers of the same JobLevel.",
            "2. **role_tenure_ratio**: Ratio of tenure in current role relative to total company tenure."
        ]),
        create_cell("code", [
            "# 1. Income ratio relative to job level",
            "joblevel_avg = df.groupby('JobLevel')['MonthlyIncome'].transform('mean')",
            "df['income_ratio_to_joblevel_avg'] = df['MonthlyIncome'] / (joblevel_avg + 1e-5)",
            "",
            "# 2. Tenure ratio",
            "df['role_tenure_ratio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)",
            "",
            "df[['EmployeeNumber', 'MonthlyIncome', 'JobLevel', 'income_ratio_to_joblevel_avg', 'role_tenure_ratio']].head()"
        ]),
        create_cell("markdown", [
            "## 3. Distribution Check",
            "Let's check distributions of our engineered features."
        ]),
        create_cell("code", [
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
            "sns.histplot(data=df, x='income_ratio_to_joblevel_avg', hue='Attrition', kde=True, ax=axes[0], multiple='stack')",
            "axes[0].set_title('Distribution of Income Ratio by Attrition')",
            "",
            "sns.histplot(data=df, x='role_tenure_ratio', hue='Attrition', kde=True, ax=axes[1], multiple='stack')",
            "axes[1].set_title('Distribution of Role Tenure Ratio by Attrition')",
            "plt.show()"
        ])
    ]
    write_notebook(dest, "02 Feature Engineering", cells)


def generate_modeling_notebook() -> None:
    """Generates the Machine Learning Modeling notebook."""
    dest = os.path.join(NOTEBOOKS_DIR, "03_modeling.ipynb")
    cells = [
        create_cell("markdown", [
            "## 1. Import Packages & Load Preprocessed Data",
            "We prepare scikit-learn preprocessing pipelines, run SMOTE to address class imbalance, and fit models."
        ]),
        create_cell("code", [
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.model_selection import train_test_split",
            "from imblearn.over_sampling import SMOTE",
            "from sklearn.compose import ColumnTransformer",
            "from sklearn.preprocessing import OneHotEncoder, StandardScaler",
            "from xgboost import XGBClassifier",
            "from sklearn.linear_model import LogisticRegression",
            "from sklearn.metrics import classification_report, roc_auc_score",
            "import shap",
            "import matplotlib.pyplot as plt"
        ]),
        create_cell("markdown", [
            "## 2. Load Processed Dataset & Add Features"
        ]),
        create_cell("code", [
            "df = pd.read_csv('../data/processed/hr_clean.csv')",
            "joblevel_avg = df.groupby('JobLevel')['MonthlyIncome'].transform('mean')",
            "df['income_ratio_to_joblevel_avg'] = df['MonthlyIncome'] / (joblevel_avg + 1e-5)",
            "df['role_tenure_ratio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)",
            "",
            "X = df.drop(columns=['Attrition', 'EmployeeNumber'])",
            "y = df['Attrition']",
            "",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)"
        ]),
        create_cell("markdown", [
            "## 3. Apply Column Preprocessing Pipeline"
        ]),
        create_cell("code", [
            "cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'MaritalStatus', 'OverTime']",
            "num_cols = [c for c in X.columns if c not in cat_cols]",
            "",
            "preprocessor = ColumnTransformer(transformers=[",
            "    ('num', StandardScaler(), num_cols),",
            "    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)",
            "])",
            "",
            "X_train_prep = preprocessor.fit_transform(X_train)",
            "X_test_prep = preprocessor.transform(X_test)"
        ]),
        create_cell("markdown", [
            "## 4. Run SMOTE for Class Imbalance"
        ]),
        create_cell("code", [
            "sm = SMOTE(random_state=42)",
            "X_train_res, y_train_res = sm.fit_resample(X_train_prep, y_train)",
            "print(f'Original Class Ratio: {np.bincount(y_train)}')",
            "print(f'Resampled Class Ratio: {np.bincount(y_train_res)}')"
        ]),
        create_cell("markdown", [
            "## 5. Fit & Evaluate Baseline vs. XGBoost"
        ]),
        create_cell("code", [
            "lr = LogisticRegression(max_iter=1000, random_state=42)",
            "lr.fit(X_train_res, y_train_res)",
            "lr_preds = lr.predict(X_test_prep)",
            "lr_probs = lr.predict_proba(X_test_prep)[:, 1]",
            "print('--- Baseline Logistic Regression ---')",
            "print(classification_report(y_test, lr_preds))",
            "print(f'ROC-AUC: {roc_auc_score(y_test, lr_probs):.4f}')",
            "",
            "xgb = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, eval_metric='logloss', random_state=42)",
            "xgb.fit(X_train_res, y_train_res)",
            "xgb_preds = xgb.predict(X_test_prep)",
            "xgb_probs = xgb.predict_proba(X_test_prep)[:, 1]",
            "print('\\n--- XGBoost Classifier ---')",
            "print(classification_report(y_test, xgb_preds))",
            "print(f'ROC-AUC: {roc_auc_score(y_test, xgb_probs):.4f}')"
        ]),
        create_cell("markdown", [
            "## 6. Model Explainability with SHAP",
            "Let's see what drivers are contributing to model predictions."
        ]),
        create_cell("code", [
            "explainer = shap.TreeExplainer(xgb)",
            "shap_values = explainer.shap_values(X_test_prep)",
            "feature_names = [name.split('__')[1] for name in preprocessor.get_feature_names_out()]",
            "",
            "plt.figure(figsize=(10, 6))",
            "shap.summary_plot(shap_values, X_test_prep, feature_names=feature_names, show=True)"
        ])
    ]
    write_notebook(dest, "03 Machine Learning Modeling", cells)


def main() -> None:
    """Main execution function."""
    print("Generating project Jupyter notebooks...")
    generate_eda_notebook()
    generate_fe_notebook()
    generate_modeling_notebook()
    print("All notebooks created.")


if __name__ == "__main__":
    main()
