# Workforce Stability & Flight Risk Audit
## Predictive Retention Intelligence & Financial ROI Dashboard

[![Streamlit App](https://static.streamlit.io/badge_base.svg)](https://hrattritionprediction-kgybdm3r8wdvy4ozxhm9cl.streamlit.app/)

HR Attrition Prediction

A model that flags employees who are likely to leave, built on IBM's HR Analytics dataset, with the full pipeline packaged so it can be reused on a real client's data.

Why this exists

Losing an employee costs money, roughly the equivalent of several months' salary once you account for hiring, onboarding, and lost productivity. Most of that cost is avoidable if HR has a heads-up. This project builds a model that scores every employee on their likelihood of leaving, so retention efforts can be targeted instead of blanket.

Results

Two models were trained and compared on a holdout set of 294 employees (47 of whom actually left).

Model	Threshold	ROC-AUC	PR-AUC	Recall	Precision	Accuracy
XGBoost	0.268	80.5%	51.5%	48.9%	50.0%	84.0%
Logistic Regression	0.447	82.3%	57.2%	44.7%	67.7%	87.8%

Neither model is dramatically better than the other overall. Logistic Regression separates the classes slightly better and gives fewer false alarms, XGBoost catches marginally more leavers. Given how close they are, I'd lean toward Logistic Regression for a real deployment since it's easier to explain to HR and legal, and it's not giving up much in recall to get there.

Confusion matrix, XGBoost at threshold 0.268:

                  Predicted stay   Predicted leave
Actual stay (247)      224              23
Actual leave (47)       24              23

Confusion matrix, Logistic Regression at threshold 0.447:

                  Predicted stay   Predicted leave
Actual stay (247)      237              10
Actual leave (47)       26              21

Read plainly: at these thresholds, roughly half of the people who actually left were flagged in advance, and for every person correctly flagged, there's roughly one false alarm. That's the tradeoff we chose, and it's tunable depending on how expensive a false alarm actually is in a given company.

Estimated financial impact

Using a workforce roughly the size of this dataset, and a 30% success rate on retention interventions (raise, workload change, role move) at flagged employees, with a replacement cost of $15,000 per departure:

~230 employees flagged per year (about half true risks, half false alarms)
~34 additional employees retained who otherwise would have left
gross savings around $510,000
net savings around $165,000 after intervention costs

These numbers depend heavily on the intervention success rate and replacement cost assumptions, which should be replaced with a client's actual figures before quoting anything.

Fairness check

Selection rate ratios were checked against the EEOC's four-fifths rule.

Gender: 0.977, no adverse impact
Age (40+ vs under 40): 0.396, this fails the four-fifths threshold

The age result is worth taking seriously before this model goes anywhere near real HR decisions. It likely reflects that younger employees genuinely leave more often in this dataset, but a model that reproduces that pattern could create legal exposure under the ADEA if it's used to make employment decisions. This is flagged rather than fixed, since the right fix depends on legal guidance, not just a modeling choice.

What's in here
hr-attrition-prediction/
  client-deliverables/
    executive_summary.pdf
  data/
    raw/
      HR_Employee_Attrition.csv
      Exit_Interview.csv
    processed/
      hr_clean.csv
  notebooks/
    01_eda.ipynb
    02_feature_engineering.ipynb
    03_modeling.ipynb
  sql/
    attrition_queries.sql
  src/
    data_prep.py
    train_model.py
    predict.py
    fairness_check.py
    drift_monitor.py
    generate_pdf.py
  dashboard/
    streamlit_app.py
  tests/
    test_data_prep.py
    test_feature_engineering.py
    test_predict.py
  models/
    attrition_model.pkl
    baseline_model.pkl
    preprocessor.pkl
  requirements.txt
  Dockerfile
  Makefile
  run.sh
  run.ps1
How the model works

The target class is imbalanced, about 16% of employees left. Rather than oversampling, both models use class weighting (scale_pos_weight for XGBoost, class_weight='balanced' for Logistic Regression), and the classification threshold was tuned out-of-fold to maximize F1 rather than using the default 0.5 cutoff.

Two engineered features made a real difference: income relative to the average for that job level (catches pay compression, where someone's pay has fallen behind peers at the same level), and time in current role relative to total tenure at the company (a rough proxy for stagnation).

Every prediction comes with a SHAP explanation, so a specific employee's risk score can be traced back to the two or three factors driving it, rather than treated as a black box output.

Running it

Linux or macOS:

chmod +x run.sh
./run.sh

Windows:

./run.ps1

Or with Make:

make all



Any of these will set up the environment, prepare the data, train both models, run the test suite, and launch the dashboard at localhost:8501.

Dashboard

The Streamlit app has four tabs: a single-employee risk calculator, a batch CSV uploader for scoring a whole team at once, an ROI calculator where you can plug in your own replacement cost and intervention success rate assumptions, and a fairness/drift panel.

Monitoring after deployment

A model trained on this year's employee data will drift as the workforce changes. drift_monitor.py runs Kolmogorov-Smirnov and chi-square tests month over month on the input features and flags when the incoming data has shifted enough that retraining is worth considering. This isn't wired up to run automatically here, but it's the piece that should get scheduled if this were handed off to run on its own.

Reusing this for other problems

The pipeline itself, data cleaning, SQL analysis, model comparison, SHAP explanations, fairness check, drift monitoring, and dashboard, isn't specific to attrition. The same structure works for customer churn, loan default risk, or any other binary outcome prediction problem where a client wants both a working model and a plain-language explanation of it. Swapping in a new dataset and adjusting the feature engineering step is a days-not-weeks change, not a rebuild.
