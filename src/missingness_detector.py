"""
Missingness leakage: checks whether the pattern of missing values in a
feature (feature_is_missing) itself strongly predicts the target.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import thresholds as T


def _binary_auc_for_indicator(indicator: np.ndarray, target_encoded: np.ndarray,
                               n_splits: int = 3) -> float | None:
    """Cross-validated ROC-AUC of a single binary missingness indicator predicting
    a binary target. Returns None if not computable (e.g. degenerate classes)."""
    if len(np.unique(indicator)) < 2 or len(np.unique(target_encoded)) < 2:
        return None
    try:
        n_splits = min(n_splits, np.bincount(target_encoded.astype(int)).min())
        if n_splits < 2:
            return None
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=T.RANDOM_STATE)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200)),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                pipe, indicator.reshape(-1, 1), target_encoded,
                cv=skf, scoring="roc_auc"
            )
        return float(np.mean(scores))
    except Exception:
        return None


def detect_missingness_leakage(df: pd.DataFrame, target_col: str,
                                target_encoded: np.ndarray,
                                problem_type: str,
                                candidate_features: list[str]) -> dict[str, dict]:
    """For each candidate feature with a meaningful missing rate, test whether
    its missingness indicator predicts the (binarized, for classification)
    target unusually well.

    For regression targets we bucket into above/below median to get a binary
    signal purely for this check (documented as a heuristic).
    """
    results = {}

    if problem_type == "classification":
        # Use as-is if binary; otherwise skip (multiclass missingness AUC is
        # not well-defined without one-vs-rest, which we keep out for simplicity).
        uniq = np.unique(target_encoded)
        if len(uniq) != 2:
            return results
        y = target_encoded
    else:
        median = np.nanmedian(target_encoded)
        y = (target_encoded > median).astype(int)

    for col in candidate_features:
        if col == target_col:
            continue
        missing_rate = df[col].isna().mean()
        if missing_rate < T.MISSINGNESS_MIN_MISSING_RATE or missing_rate > 0.98:
            continue
        indicator = df[col].isna().astype(int).values
        auc = _binary_auc_for_indicator(indicator, y)
        if auc is None:
            continue
        # AUC below 0.5 means inverted signal; use distance from 0.5 for symmetry
        strength = max(auc, 1 - auc)
        if strength >= T.MISSINGNESS_AUC_HIGH:
            risk = "High"
        elif strength >= T.MISSINGNESS_AUC_MEDIUM:
            risk = "Medium"
        else:
            continue

        results[col] = {
            "missing_rate": float(missing_rate),
            "auc": float(strength),
            "risk": risk,
            "reason": (
                f"The pattern of missing values in '{col}' predicts the target "
                f"with an AUC of {strength:.3f} (missingness-indicator test). "
                "This may indicate that whether a value was recorded is itself "
                "influenced by the outcome (potential missingness leakage)."
            ),
        }

    return results
