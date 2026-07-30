"""Unit tests for the data preparation pipeline."""

import os
import sqlite3
import pandas as pd
import pytest

from src.data_prep import ensure_directories, DATA_DIR, RAW_DIR, PROCESSED_DIR, DB_FILE


def test_directories_creation() -> None:
    """Verifies that the required directories are created by the prep script."""
    ensure_directories()
    assert os.path.exists(DATA_DIR)
    assert os.path.exists(RAW_DIR)
    assert os.path.exists(PROCESSED_DIR)


def test_processed_data_exists() -> None:
    """Verifies that processed files exist and contain appropriate columns."""
    clean_csv = os.path.join(PROCESSED_DIR, "hr_clean.csv")
    assert os.path.exists(clean_csv), "Run data_prep.py before running tests."
    
    df = pd.read_csv(clean_csv)
    assert "Attrition" in df.columns
    assert "EmployeeNumber" in df.columns
    # Ensure useless columns were successfully dropped
    assert "EmployeeCount" not in df.columns
    assert "StandardHours" not in df.columns
    assert "Over18" not in df.columns


def test_sqlite_database_tables() -> None:
    """Verifies that the SQLite database has the correct tables loaded."""
    assert os.path.exists(DB_FILE)
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    assert "employees" in tables
    assert "exit_interviews" in tables
    
    conn.close()
