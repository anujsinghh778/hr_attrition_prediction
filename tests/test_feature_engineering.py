"""Unit tests for the custom feature engineering formulas."""

import pandas as pd
import numpy as np
import pytest

from src.train_model import engineer_features


def test_engineer_features() -> None:
    """Validates feature engineering calculations on mock data."""
    # Create mock dataset
    mock_data = pd.DataFrame({
        "JobLevel": [1, 1, 2, 2],
        "MonthlyIncome": [2000, 4000, 6000, 8000],
        "YearsInCurrentRole": [2, 0, 5, 1],
        "YearsAtCompany": [5, 0, 10, 1]
    })
    
    res = engineer_features(mock_data)
    
    # Check shape
    assert res.shape[0] == 4
    assert "income_ratio_to_joblevel_avg" in res.columns
    assert "role_tenure_ratio" in res.columns
    
    # JobLevel 1 average income: 3000
    # Expected ratio: 2000/3000 = 0.6666, 4000/3000 = 1.3333
    assert np.isclose(res.loc[0, "income_ratio_to_joblevel_avg"], 2/3)
    assert np.isclose(res.loc[1, "income_ratio_to_joblevel_avg"], 4/3)
    
    # JobLevel 2 average income: 7000
    # Expected ratio: 6000/7000 = 0.8571, 8000/7000 = 1.1428
    assert np.isclose(res.loc[2, "income_ratio_to_joblevel_avg"], 6/7)
    
    # Check tenure ratio: YearsInCurrentRole / (YearsAtCompany + 1)
    # Row 0: 2 / (5 + 1) = 2/6 = 0.3333
    # Row 1: 0 / (0 + 1) = 0.0
    # Row 2: 5 / (10 + 1) = 5/11 = 0.4545
    # Row 3: 1 / (1 + 1) = 0.5
    assert np.isclose(res.loc[0, "role_tenure_ratio"], 2/6)
    assert np.isclose(res.loc[1, "role_tenure_ratio"], 0.0)
    assert np.isclose(res.loc[2, "role_tenure_ratio"], 5/11)
    assert np.isclose(res.loc[3, "role_tenure_ratio"], 0.5)
