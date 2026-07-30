"""Data preparation script for the HR Analytics & Attrition project.

Downloads/generates the IBM Employee Attrition dataset, creates a secondary exit interview
table, merges them, loads them into a local SQLite database, and executes analytical queries.
"""

import os
import sqlite3
import urllib.request
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SQL_FILE = os.path.join(os.path.dirname(__file__), "..", "sql", "attrition_queries.sql")
DB_FILE = os.path.join(DATA_DIR, "hr_analytics.db")

PRIMARY_URL = (
    "https://raw.githubusercontent.com/pplonski/datasets-for-start/"
    "master/employee_attrition/HR-Employee-Attrition-All.csv"
)


def ensure_directories() -> None:
    """Creates the data directories if they do not exist."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def download_dataset(url: str, dest_path: str) -> bool:
    """Attempts to download the dataset from a public raw URL.

    Args:
        url: The URL to download from.
        dest_path: The local file path to write to.

    Returns:
        True if download succeeded, False otherwise.
    """
    try:
        print(f"Attempting to download dataset from: {url}")
        # Configure a simple User-Agent header to avoid blocking
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            with open(dest_path, "wb") as f:
                f.write(response.read())
        print(f"Successfully downloaded dataset to {dest_path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def generate_synthetic_dataset(dest_path: str) -> None:
    """Generates a high-fidelity synthetic employee attrition dataset matching the IBM schema.

    This ensures that the project remains fully runnable offline or if download endpoints fail.

    Args:
        dest_path: The local file path to write to.
    """
    print("Generating high-fidelity synthetic employee attrition dataset...")
    np.random.seed(42)
    n_records = 1470

    # Age: 18 to 60
    age = np.random.randint(18, 61, size=n_records)
    gender = np.random.choice(["Male", "Female"], size=n_records, p=[0.6, 0.4])
    marital = np.random.choice(["Single", "Married", "Divorced"], size=n_records, p=[0.3, 0.5, 0.2])
    travel = np.random.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], size=n_records, p=[0.7, 0.2, 0.1])
    dept = np.random.choice(["Research & Development", "Sales", "Human Resources"], size=n_records, p=[0.65, 0.30, 0.05])
    dist = np.random.randint(1, 30, size=n_records)
    educ = np.random.randint(1, 6, size=n_records)
    
    edu_fields = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"]
    edu_field = np.random.choice(edu_fields, size=n_records, p=[0.4, 0.3, 0.1, 0.1, 0.07, 0.03])
    
    env_sat = np.random.randint(1, 5, size=n_records)
    job_inv = np.random.randint(1, 5, size=n_records)
    job_sat = np.random.randint(1, 5, size=n_records)
    rel_sat = np.random.randint(1, 5, size=n_records)
    work_life = np.random.choice([1, 2, 3, 4], size=n_records, p=[0.1, 0.25, 0.5, 0.15])
    num_comp = np.random.randint(0, 10, size=n_records)
    
    # Overtime: 'Yes' or 'No'
    overtime = np.random.choice(["Yes", "No"], size=n_records, p=[0.3, 0.7])
    
    # JobLevel and JobRole logic
    job_level = np.random.choice([1, 2, 3, 4, 5], size=n_records, p=[0.35, 0.35, 0.15, 0.10, 0.05])
    
    roles = [
        "Sales Executive", "Research Scientist", "Laboratory Technician", 
        "Manufacturing Director", "Healthcare Representative", "Manager", 
        "Sales Representative", "Research Director", "Human Resources"
    ]
    
    job_role = []
    for d, lvl in zip(dept, job_level):
        if d == "Human Resources":
            job_role.append("Human Resources")
        elif d == "Sales":
            if lvl >= 4:
                job_role.append("Manager")
            elif lvl == 3:
                job_role.append("Sales Executive")
            else:
                job_role.append("Sales Representative")
        else:  # Research & Development
            if lvl >= 4:
                job_role.append("Manager" if np.random.rand() > 0.5 else "Research Director")
            elif lvl == 3:
                job_role.append("Healthcare Representative" if np.random.rand() > 0.5 else "Manufacturing Director")
            elif lvl == 2:
                job_role.append("Manufacturing Director" if np.random.rand() > 0.5 else "Laboratory Technician")
            else:
                job_role.append("Research Scientist" if np.random.rand() > 0.5 else "Laboratory Technician")
                
    # Income based on JobLevel
    income_base = {1: (2000, 3500), 2: (3500, 6000), 3: (6000, 10000), 4: (10000, 15000), 5: (15000, 20000)}
    monthly_income = []
    for lvl in job_level:
        low, high = income_base[lvl]
        monthly_income.append(np.random.randint(low, high))
    monthly_income = np.array(monthly_income)
    
    percent_hike = np.random.randint(11, 26, size=n_records)
    stock_level = np.random.choice([0, 1, 2, 3], size=n_records, p=[0.4, 0.4, 0.15, 0.05])
    
    # Tenure variables
    total_working_years = []
    for a in age:
        max_work = a - 18
        total_working_years.append(np.random.randint(0, max(1, max_work + 1)))
    total_working_years = np.array(total_working_years)
    
    years_at_company = []
    for wy in total_working_years:
        years_at_company.append(np.random.randint(0, wy + 1))
    years_at_company = np.array(years_at_company)
    
    years_in_role = []
    years_since_promo = []
    years_with_mgr = []
    for yc in years_at_company:
        years_in_role.append(np.random.randint(0, yc + 1))
        years_since_promo.append(np.random.randint(0, yc + 1))
        years_with_mgr.append(np.random.randint(0, yc + 1))
        
    years_in_role = np.array(years_in_role)
    years_since_promo = np.array(years_since_promo)
    years_with_mgr = np.array(years_with_mgr)
    
    training = np.random.randint(0, 7, size=n_records)
    perf_rating = np.random.choice([3, 4], size=n_records, p=[0.85, 0.15])
    
    daily_rate = np.random.randint(100, 1500, size=n_records)
    hourly_rate = np.random.randint(30, 101, size=n_records)
    monthly_rate = np.random.randint(2000, 26000, size=n_records)
    
    # Establish realistic attrition probabilities based on features
    # Positive attrition predictors: Low satisfaction, Overtime, low income relative to level, high promotion gap
    attrition_probs = []
    for i in range(n_records):
        prob = 0.05
        if overtime[i] == "Yes":
            prob += 0.25
        if job_sat[i] == 1:
            prob += 0.20
        if work_life[i] == 1:
            prob += 0.15
        if env_sat[i] == 1:
            prob += 0.10
        if monthly_income[i] < income_base[job_level[i]][0] + 500:
            prob += 0.15
        if years_since_promo[i] > 5:
            prob += 0.10
        if stock_level[i] == 0:
            prob += 0.08
        if age[i] < 30:
            prob += 0.08
        attrition_probs.append(min(0.95, prob))
        
    attrition_raw = np.random.rand(n_records) < np.array(attrition_probs)
    attrition = np.where(attrition_raw, "Yes", "No")

    # Construct DataFrame
    df = pd.DataFrame({
        "Age": age,
        "Attrition": attrition,
        "BusinessTravel": travel,
        "DailyRate": daily_rate,
        "Department": dept,
        "DistanceFromHome": dist,
        "Education": educ,
        "EducationField": edu_field,
        "EmployeeCount": 1,
        "EmployeeNumber": np.arange(1, n_records + 1),
        "EnvironmentSatisfaction": env_sat,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_inv,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_sat,
        "MaritalStatus": marital,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_comp,
        "Over18": "Y",
        "OverTime": overtime,
        "PercentSalaryHike": percent_hike,
        "PerformanceRating": perf_rating,
        "RelationshipSatisfaction": rel_sat,
        "StandardHours": 80,
        "StockOptionLevel": stock_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training,
        "WorkLifeBalance": work_life,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_role,
        "YearsSinceLastPromotion": years_since_promo,
        "YearsWithCurrManager": years_with_mgr
    })
    
    df.to_csv(dest_path, index=False)
    print(f"Synthetic dataset saved to {dest_path}")


def generate_exit_interviews(raw_emp_path: str, dest_path: str) -> None:
    """Generates a synthetic exit interview table for employees with Attrition = Yes.

    Shows standard relational data manipulation and join capabilities.

    Args:
        raw_emp_path: Path to the raw employee dataset.
        dest_path: Path to write the exit interview dataset.
    """
    print("Generating relational exit interview data...")
    df = pd.read_csv(raw_emp_path)
    
    # Filter for employees who left
    exits = df[df["Attrition"] == "Yes"].copy()
    num_exits = len(exits)
    
    np.random.seed(42)
    
    # Potential reasons
    reasons = [
        "Career Change", "Compensation & Benefits", "Workplace Culture", 
        "Lack of Career Growth", "Work-Life Balance", "Relocation", 
        "Retirement", "Personal Reasons"
    ]
    
    exit_reasons = []
    for idx, row in exits.iterrows():
        # Match reason to features to look highly realistic
        probs = [0.125] * 8
        if row["OverTime"] == "Yes" or row["WorkLifeBalance"] == 1:
            probs[reasons.index("Work-Life Balance")] += 0.4
        if row["MonthlyIncome"] < 4000:
            probs[reasons.index("Compensation & Benefits")] += 0.4
        if row["YearsSinceLastPromotion"] > 4:
            probs[reasons.index("Lack of Career Growth")] += 0.4
        if row["EnvironmentSatisfaction"] == 1:
            probs[reasons.index("Workplace Culture")] += 0.4
            
        # Normalize probabilities
        probs = np.array(probs)
        probs = probs / probs.sum()
        exit_reasons.append(np.random.choice(reasons, p=probs))
        
    rehire_eligibility = np.random.choice(["Yes", "No", "Conditional"], size=num_exits, p=[0.6, 0.2, 0.2])
    final_perf = np.random.choice([1, 2, 3, 4], size=num_exits, p=[0.05, 0.15, 0.65, 0.15])
    
    # Exit interview dates (simulated dates in 2025/2026)
    exit_dates = []
    for _ in range(num_exits):
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)
        exit_dates.append(f"2025-{month:02d}-{day:02d}")
        
    exit_df = pd.DataFrame({
        "EmployeeNumber": exits["EmployeeNumber"].values,
        "ExitReason": exit_reasons,
        "RehireEligibility": rehire_eligibility,
        "FinalPerformanceScore": final_perf,
        "ExitInterviewDate": exit_dates
    })
    
    exit_df.to_csv(dest_path, index=False)
    print(f"Exit interviews saved to {dest_path}")


def load_to_sqlite(emp_path: str, exit_path: str, db_path: str) -> None:
    """Loads employee and exit interview datasets into SQLite and runs analytics.

    Args:
        emp_path: Path to the clean employee dataset.
        exit_path: Path to the exit interview dataset.
        db_path: Path to the SQLite database file to create/populate.
    """
    print(f"Connecting to SQLite database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    # Load DataFrames
    emp_df = pd.read_csv(emp_path)
    exit_df = pd.read_csv(exit_path)
    
    # Save to SQL
    emp_df.to_sql("employees", conn, if_exists="replace", index=False)
    exit_df.to_sql("exit_interviews", conn, if_exists="replace", index=False)
    
    print("Executing analytical queries from SQL file...")
    if os.path.exists(SQL_FILE):
        with open(SQL_FILE, "r") as f:
            sql_script = f.read()
            
        queries = sql_script.split(";")
        for i, q in enumerate(queries):
            q_clean = q.strip()
            if not q_clean:
                continue
            
            # Print query description or first comment
            comment = "Query"
            for line in q_clean.split("\n"):
                if line.startswith("--"):
                    comment = line.replace("--", "").strip()
                    break
                    
            print(f"\n--- [SQL Query {i+1}] {comment} ---")
            try:
                res_df = pd.read_sql_query(q_clean, conn)
                print(res_df.to_string(index=False))
            except Exception as e:
                print(f"Error running query: {e}")
    else:
        print(f"Warning: SQL queries file not found at {SQL_FILE}")
        
    conn.close()
    print("\nDatabase operations completed and connection closed.")


def main() -> None:
    """Main execution block for data prep."""
    ensure_directories()
    
    raw_emp_path = os.path.join(RAW_DIR, "HR_Employee_Attrition.csv")
    exit_path = os.path.join(RAW_DIR, "Exit_Interview.csv")
    processed_emp_path = os.path.join(PROCESSED_DIR, "hr_clean.csv")
    
    # Step 1: Download or Generate primary employee attrition dataset
    if not os.path.exists(raw_emp_path):
        success = download_dataset(PRIMARY_URL, raw_emp_path)
        if not success:
            generate_synthetic_dataset(raw_emp_path)
    else:
        print("Employee attrition raw data already exists locally.")
        
    # Step 2: Generate Exit Interviews relational dataset
    if not os.path.exists(exit_path):
        generate_exit_interviews(raw_emp_path, exit_path)
    else:
        print("Exit interviews raw data already exists locally.")
        
    # Step 3: Process and clean
    print("Preprocessing and cleaning data...")
    df = pd.read_csv(raw_emp_path)
    
    # Drop constant/irrelevant columns
    useless_cols = ["EmployeeCount", "StandardHours", "Over18"]
    useless_cols_found = [c for c in useless_cols if c in df.columns]
    df = df.drop(columns=useless_cols_found)
    
    # Check duplicates
    df = df.drop_duplicates()
    
    # Map target column to numeric
    if "Attrition" in df.columns and df["Attrition"].dtype == "object":
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
        
    # Save cleaned employee table
    df.to_csv(processed_emp_path, index=False)
    print(f"Cleaned employee dataset saved to {processed_emp_path}")
    
    # Step 4: Relational SQLite DB load & Query execution
    load_to_sqlite(processed_emp_path, exit_path, DB_FILE)
    
    print("\nData Prep Pipeline Executed Successfully.")


if __name__ == "__main__":
    main()
