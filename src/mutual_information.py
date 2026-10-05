"""
Mutual information analysis between features and target.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from config import thresholds as T
from .utils import is_numeric_series, safe_label_encode


def _prepare_feature_matrix(df: pd.DataFrame, feature_cols: list[str]):
    """Build a numeric matrix suitable for sklearn's mutual_info_* functions,
    encoding categoricals and imputing missing values with a simple, explicit
    strategy (median for numeric, a distinct 'missing' code for categorical).
    Returns (X, discrete_mask)."""
    X_cols = []
    discrete_mask = []
    for col in feature_cols:
        s = df[col]
        if is_numeric_series(s):
            filled = s.fillna(s.median() if s.notna().any() else 0.0)
            X_cols.append(filled.values.astype(float))
            discrete_mask.append(False)
        else:
            codes = safe_label_encode(s.astype(str).where(s.notna(), other="__MISSING__"))
            X_cols.append(codes.astype(float))
            discrete_mask.append(True)
    X = np.column_stack(X_cols) if X_cols else np.empty((len(df), 0))
    return X, discrete_mask


def compute_mutual_information(df: pd.DataFrame, target_col: str,
                                target_encoded: np.ndarray,
                                feature_cols: list[str],
                                problem_type: str) -> dict[str, float]:
    """Compute (normalized-ish) mutual information for each feature against the
    target. Returns raw MI values; percentile-based flagging is handled by the
    risk engine using MUTUAL_INFO_HIGH_PERCENTILE / MUTUAL_INFO_MIN_ABS."""
    if not feature_cols:
        return {}

    X, discrete_mask = _prepare_feature_matrix(df, feature_cols)
    y = target_encoded

    valid_mask = ~pd.isna(y)
    X = X[valid_mask]
    y = np.asarray(y)[valid_mask]

    if X.shape[0] < 10 or X.shape[1] == 0:
        return {col: np.nan for col in feature_cols}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if problem_type == "classification":
                mi = mutual_info_classif(
                    X, y, discrete_features=discrete_mask,
                    random_state=T.RANDOM_STATE
                )
            else:
                mi = mutual_info_regression(
                    X, y, discrete_features=discrete_mask,
                    random_state=T.RANDOM_STATE
                )
        except Exception:
            return {col: np.nan for col in feature_cols}

    # Normalize MI to a rough 0-1 scale relative to the max observed, so the
    # "high percentile" heuristic threshold is meaningful across datasets.
    max_mi = mi.max() if len(mi) and mi.max() > 0 else 1.0
    normalized = mi / max_mi

    return {col: float(val) for col, val in zip(feature_cols, normalized)}
