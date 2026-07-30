
# Workforce Stability & Attrition Analytics
## Predictive Flight-Risk Modeling, Bias Auditing, and Financial ROI Projections
[![Streamlit App](https://static.streamlit.io/badge_base.svg)](https://hrattritionprediction-kgybdm3r8wdvy4ozxhm9cl.streamlit.app/)
---
### Executive Summary & Business Impact
This repository contains an end-to-end predictive analytics system designed to identify employee flight risks and evaluate the financial return of targeted retention strategies. By moving from reactive exit interviews to proactive intervention, the system helps HR departments minimize the costs of unplanned turnover.
*   **Model Performance:** Our F1-optimized XGBoost classifier achieves a **50.0% Precision** and a **48.9% Recall** (ROC-AUC of **80.5%**). This represents a **3.1x lift** in risk detection accuracy compared to a random baseline selection in a standard workforce (where base attrition is 16.1%).
*   **Financial Return on Investment (ROI):** For a typical 1,500-person enterprise, the model identifies roughly 230 high-risk employees annually. Assuming a standard retention intervention (e.g., career pathing, salary adjustments) succeeds 30% of the time, the organization retains 34 employees who would have otherwise left. At an average replacement cost of $15,000 per employee, this system generates **$510,000 in gross savings** and **$165,000 in net annual savings** (accounting for a $1,500 budget per flagged employee).
---
### Core System Features
1.  **Relational SQL Pipeline:** Automates data consolidation by merging raw workforce tables (demographics, job details, and exit histories) into an analytical SQLite database to prepare clean modeling files.
2.  **Custom Feature Engineering:** Builds key predictive indicators such as `income_ratio_to_joblevel_avg` (identifies salary compression relative to job level peers) and `role_tenure_ratio` (tracks career stagnation).
3.  **Explainable AI (SHAP):** Integrates Shapley Additive Explanations to display the exact feature attributions shifting risk scores for individual employee records, ensuring transparent decision-making.
4.  **Bias & Demographic Parity Auditing:** Checks selection rates across protected characteristics (Gender and Age) to verify compliance with the EEOC's Four-Fifths rule.
5.  **Data Drift Monitoring:** Implements Kolmogorov-Smirnov (KS) and Chi-Square contingency tests to detect statistical shifts in incoming feature distributions, alerting operations when the model requires retraining.
6.  **Interactive Web Portal:** A multi-tab Streamlit dashboard containing a real-time ROI calculator, individual risk calculator, batch CSV uploader, and data drift simulator.
---
### Project Architecture
hr-attrition-prediction/ ├── client-deliverables/ │ └── executive_summary.pdf # Programmatically generated C-suite briefing ├── data/ │ ├── raw/ │ │ ├── HR_Employee_Attrition.csv # Primary workforce dataset │ │ └── Exit_Interview.csv # Relational table detailing reasons and rehire status │ └── processed/ │ └── hr_clean.csv # Final merged dataset ├── notebooks/ │ ├── 01_eda.ipynb # Exploratory Data Analysis │ ├── 02_feature_engineering.ipynb # Feature formulas development │ └── 03_modeling.ipynb # Grid searches, thresholding, and SHAP ├── sql/ │ └── attrition_queries.sql # SQLite analytical schema and joins ├── src/ │ ├── data_prep.py # Database ingestion and processing │ ├── train_model.py # Model training and threshold selection │ ├── predict.py # Pure-Python inference pipeline (no sklearn pickle dependency) │ ├── fairness_check.py # EEOC Four-Fifths compliance audit │ ├── drift_monitor.py # Statistical distribution drift engine │ └── generate_pdf.py # ReportLab executive PDF builder ├── dashboard/ │ └── streamlit_app.py # Dashboard front-end (ROI sliders, evaluation, drift) ├── tests/ │ ├── test_data_prep.py # Database integrity tests │ ├── test_feature_engineering.py # Custom feature formula validations │ └── test_predict.py # Inference checks ├── models/ │ ├── attrition_model.pkl # Trained XGBoost model │ ├── baseline_model.pkl # Baseline Logistic Regression model │ ├── preprocessor_meta.json # JSON metadata and scaler parameters (for robust cloud deploy) │ └── model_metrics.json # Saved evaluation scores ├── Dockerfile # Containerized deployment file ├── Makefile # Build automation commands ├── run.ps1 # Setup script for Windows (PowerShell) └── run.sh # Setup script for Unix (Bash)



---
### Local Installation & Getting Started
#### Windows (PowerShell)
Execute the setup script to establish a virtual environment, install packages, compile datasets, fit models, run unit tests, and launch the Streamlit dashboard locally:
```powershell
./run.ps1
macOS / Linux
Make the setup script executable and run it:

bash


chmod +x run.sh
./run.sh
Docker Container
To deploy the dashboard in a containerized environment:

bash


docker build -t hr-attrition-app .
docker run -p 8501:8501 hr-attrition-app
Model Performance & Decision Tradeoffs
Evaluated on a 20% holdout test set (294 employees, including 47 actual attrition cases), the F1-optimized metrics are as follows:

Model	Classification Threshold	ROC-AUC	PR-AUC (Avg. Precision)	Recall (Flight Catch)	Precision	Accuracy
XGBoost Classifier	0.268	80.49%	51.46%	48.94%	50.00%	84.01%
Logistic Regression	0.447	82.29%	57.24%	44.68%	67.74%	87.76%
The Precision-Recall Balance (Business Alignment)
Instead of forcing a high-recall boundary that flags a massive number of false positives (which strains HR team bandwidth with unnecessary check-ins), we optimized the threshold to maximize the F1-Score out-of-fold.

At a 0.268 threshold, the XGBoost model achieves an exact 50.0% Precision while catching 48.9% of exits (23 out of 47 departures). This reduces false alarms to just 23 out of 247 stays (a 9.3% False Positive Rate). This ensures that every second person flagged by the model is an actual flight risk, allowing HR to allocate retention budgets where they have the highest probability of impact.

Confusion Matrices (Holdout Test Set)
XGBoost Classifier (Tuned threshold: 0.268)



                      Predicted Stay    Predicted Leave
Actual Stay (247)          224                23         (False Positives)
Actual Leave (47)           24                23         (True Positives / Recall: 48.9%)
                            ^
                     (False Negatives)
Logistic Regression (Tuned threshold: 0.447)



                      Predicted Stay    Predicted Leave
Actual Stay (247)          237                10         (False Positives)
Actual Leave (47)           26                21         (True Positives / Recall: 44.7%)
                            ^
                     (False Negatives)
Legal compliance & Fairness Auditing
To protect the organization against legal exposure, the system audits selection rate ratios across protected categories (demographics) to verify compliance with the EEOC's Four-Fifths selection rule:

Gender Parity (Female vs. Male): The Disparate Impact Ratio is 0.977 (EEOC Compliant).
Age Parity (40+ vs. <40): The Disparate Impact Ratio is 0.396 (EEOC Non-Compliant).
Audit Warning: Younger employees are flagged as flight risks at a significantly higher rate (21.3%) than older employees (8.4%). This mirrors historical workforce trends where early career professionals change roles more frequently. However, if this model is used to distribute training opportunities or retention bonuses, it could create age discrimination liability under the ADEA. The system flags this selection bias so HR policy designers can adjust policy criteria manually.
