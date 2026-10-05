"""
Safe automatic inference of problem type (classification vs regression).
"""
from __future__ import annotations

import pandas as pd

from .utils import is_numeric_series, is_bool_like

CLASSIFICATION = "classification"
REGRESSION = "regression"

# If a numeric target has more unique values than this (and isn't integer-coded
# with low cardinality), we lean towards regression.
MAX_UNIQUE_FOR_NUMERIC_CLASSIFICATION = 20
MIN_UNIQUE_RATIO_FOR_REGRESSION = 0.05


def infer_problem_type(target: pd.Series) -> str:
    """Infer whether the target represents a classification or regression problem.

    Rules (deliberately conservative / explainable):
      - Non-numeric (object/category) target -> classification
      - Boolean-like target -> classification
      - Numeric target with few unique values relative to row count -> classification
      - Numeric target with many unique / continuous-looking values -> regression
    """
    target = target.dropna()
    if len(target) == 0:
        raise ValueError("Target column has no non-missing values.")

    if is_bool_like(target):
        return CLASSIFICATION

    if not is_numeric_series(target):
        return CLASSIFICATION

    n_unique = target.nunique()
    n_rows = len(target)
    unique_ratio = n_unique / n_rows if n_rows else 0

    # Small number of distinct numeric values strongly suggests categorical labels
    if n_unique <= MAX_UNIQUE_FOR_NUMERIC_CLASSIFICATION and unique_ratio < 0.5:
        # Extra safety: if values are all integers and few, almost certainly classification
        return CLASSIFICATION

    if unique_ratio >= MIN_UNIQUE_RATIO_FOR_REGRESSION:
        return REGRESSION

    # Fallback: default to regression for high-unique numeric targets
    return REGRESSION if n_unique > MAX_UNIQUE_FOR_NUMERIC_CLASSIFICATION else CLASSIFICATION


def is_binary(target: pd.Series) -> bool:
    return target.dropna().nunique() == 2
