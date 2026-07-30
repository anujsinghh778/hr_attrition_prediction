# Workforce Stability & Flight Risk Audit
## Predictive Retention Intelligence & Financial ROI Dashboard

[![Streamlit App](https://static.streamlit.io/badge_base.svg)](https://share.streamlit.io/anujsinghh778/nirmal-masala-bhandar-webapp-)
[![Loom Walkthrough](https://img.shields.io/badge/Loom-Video%20Walkthrough-red?logo=loom)](https://www.loom.com/share/placeholder-video-id)

---

### 📊 Business Result & ROI Impact
By applying predictive attrition scoring to the workforce, this audit system enables HR leadership to identify and preempt employee flight risks before resignation occurs. Featuring an F1-optimized classification threshold, the XGBoost model successfully catches **~50% of departures (Recall: 48.9%)** while ensuring that **50% of flagged employees are true flight risks (Precision: 50.0%)**. This represents a massive **3.1x precision lift** over the baseline random employee attrition rate (16.1%), making interventions highly targeted and efficient.

**Financial Impact:** Across our workforce, the model flags approximately 230 employees as high-risk annually (consisting of 115 true flight risks and 115 false alarms). If HR prioritizes retention interventions on this group and standard strategies (salary reviews, workload rebalancing) succeed **30% of the time**, Global Tech Corp will successfully retain 34 employees. At an average replacement cost of **$15,000 per employee**, this targeted audit yields gross savings of **$510,000** and **$165,000 in net annual savings** (after accounting for a $1,500 intervention budget per flagged employee).

---

### 💼 Productized Consulting Service Pitch
**Repeatable Client Retention Engine**
This end-to-end analytical framework is packaged as a repeatable, production-grade consulting asset. The entire pipeline—incorporating automated data consolidation, relational SQL analytics, custom feature engineering, baseline/advanced modeling (Logistic Regression vs. XGBoost), SHAP explainability audits, legal disparate impact compliance checks, drift monitors, and the interactive web application—**can be customized and deployed to any employee attrition, customer subscription churn, or client risk-prediction problem in 1–2 weeks.**

---

### 📂 Client Deliverables & Key Assets
1. **Executive PDF Briefing (`client-deliverables/executive_summary.pdf`)**: A print-ready, consulting-grade PDF document containing cost assessments, key findings, and recommended interventions tailored for CHROs and C-suite executives.
2. **Demographic Fairness Audit (`models/fairness_report.json`)**: An automated audit evaluating selection rate ratios (Disparate Impact) and False Positive Rates across protected characteristics (Gender and Age >= 40) to guarantee legal compliance under the EEOC's Four-Fifths rule.
3. **Operational Drift Diagnostics (`models/drift_report_sample.json`)**: Month-over-month distribution stability checks running Kolmogorov-Smirnov (KS) and Chi-Square tests to detect workforce changes and alert when model retraining is needed.
4. **Interactive Retention Dashboard (`dashboard/streamlit_app.py`)**: A multi-tab web portal enabling interactive risk scoring, real-time ROI modeling, bias auditing, and drift simulations.

---

### 🛠️ Folder Structure
```
hr-attrition-prediction/
├── client-deliverables/
│   └── executive_summary.pdf           # Programmatically generated C-suite PDF briefing
├── data/
│   ├── raw/
│   │   ├── HR_Employee_Attrition.csv   # IBM Employee dataset
│   │   └── Exit_Interview.csv          # Relational table detailing reasons and rehire status
│   └── processed/
│       └── hr_clean.csv                # Final merged and cleaned dataset
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis notebook
│   ├── 02_feature_engineering.ipynb    # Feature Engineering notebook
│   └── 03_modeling.ipynb               # Model Training & SHAP notebook
├── sql/
│   └── attrition_queries.sql           # SQLite analytics queries
├── src/
│   ├── data_prep.py                    # Preprocessing, relational joins, and SQLite loading
│   ├── train_model.py                  # Pipeline engineering, SMOTE, modeling, and SHAP plots
│   ├── predict.py                      # CLI & API inference interface (with type hints)
│   ├── fairness_check.py               # Disparate impact and EEOC compliance checking
│   ├── drift_monitor.py                # Kolmogorov-Smirnov & Chi-Square drift monitor
│   └── generate_pdf.py                 # ReportLab executive PDF builder
├── dashboard/
│   └── streamlit_app.py                # Dashboard (ROI sliders, Single/Batch, Fairness, Drift)
├── tests/
│   ├── test_data_prep.py               # Database and clean dataset tests
│   ├── test_feature_engineering.py     # Custom feature formula tests
│   └── test_predict.py                 # Inference and model sanity tests
├── models/
│   ├── attrition_model.pkl             # Trained XGBoost model
│   ├── baseline_model.pkl              # Baseline Logistic Regression model
│   └── preprocessor.pkl                # Preprocessor pipeline and training metadata
├── requirements.txt                    # Python library requirements
├── Dockerfile                          # Deployment Docker configuration
├── Makefile                            # Automation command runner
├── run.ps1                             # PowerShell setup and run script (Windows)
├── run.sh                              # Bash setup and run script (Linux/macOS)
└── README.md                           # Business-first documentation (this file)
```

---

### 🚀 Setup & Execution (One Command)

Select the command corresponding to your operating system or deployment target:

#### Windows (PowerShell)
Execute the helper script to create a virtual environment, install requirements, prepare data, train models, run the test suite, and launch the dashboard:
```powershell
./run.ps1
```

#### Linux / macOS (Bash)
Make the helper script executable and run:
```bash
chmod +x run.sh
./run.sh
```

#### Makefile (Unix environments)
To build, train, run tests, and open the dashboard:
```bash
make all
```

#### Docker
To build and deploy the containerized web application:
```bash
docker build -t hr-attrition-app .
docker run -p 8501:8501 hr-attrition-app
```

### 🏆 Model Performance & Tradeoff Analysis

The trained models are evaluated on a 20% holdout test set (294 employees, including 47 actual attrition cases). The performance metrics at the cross-validated tuned thresholds are as follows:

| Model | Classification Threshold | ROC-AUC | PR-AUC (Avg. Precision) | Recall (Flight Catch) | Precision | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost Classifier** | `0.268` | **80.49%** | **51.46%** | **48.94%** | **50.00%** | **84.01%** |
| **Logistic Regression** | `0.447` | **82.29%** | **57.24%** | **44.68%** | **67.74%** | **87.76%** |

#### ⚖️ The Precision-Recall Tradeoff (Business Alignment)
We tuned the classification threshold by maximizing the F1-score out-of-fold. This strikes a balanced tradeoff: it keeps precision high so that HR resources are not wasted on false alarms, while still catching a significant portion of departures.

At a `0.268` threshold, the XGBoost model catches **~49%** of departures (23 out of 47 departures), while ensuring that **50%** of flagged employees are true flight risks (23 out of 46 flags) — reducing false positives to just 23 out of 247 stays (a low 9.3% False Positive Rate). This ensures that HR resources are focused on high-probability cases where retention programs are most cost-effective.

#### 🔲 Confusion Matrices (Holdout Test Set)

**XGBoost Classifier (Tuned threshold: 0.268)**
```
                      Predicted Stay    Predicted Leave
Actual Stay (247)          224                23         (False Positives)
Actual Leave (47)           24                23         (True Positives / Recall: 48.9%)
                            ^
                     (False Negatives)
```

**Logistic Regression (Tuned threshold: 0.447)**
```
                      Predicted Stay    Predicted Leave
Actual Stay (247)          237                10         (False Positives)
Actual Leave (47)           26                21         (True Positives / Recall: 44.7%)
                            ^
                     (False Negatives)
```

---

### 🔬 Technical Implementation Details
*   **Custom Ratios:** We engineered `income_ratio_to_joblevel_avg` (wage compression locator) and `role_tenure_ratio` (burnout tracker) to boost predictive signals.
*   **Imbalance Handling:** The target is naturally imbalanced (~16% attrition). Instead of basic oversampling, we used class weighting (`scale_pos_weight` for XGBoost and `class_weight='balanced'` for Logistic Regression) coupled with out-of-fold (`OOF`) probability threshold tuning to maximize F1-score.
*   **Explainable AI:** Rather than relying on black-box predictions, we integrated `SHAP` (SHapley Additive exPlanations) values to output local feature attributions for every single scored employee.
*   **Fairness Safeguards:** The pipeline checks selection rate ratios (DIR) to ensure compliance with the EEOC's Four-Fifths rule. Gender is compliant (DIR: `0.977`), while Age (40+ vs <40) reveals adverse impact (DIR: `0.396`) due to heavy historical turnover among younger employees. The system flags this so HR can adjust policy allocations and avoid legal exposure under the ADEA.
