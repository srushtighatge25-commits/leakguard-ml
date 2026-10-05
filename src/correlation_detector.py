"""
Numerical feature-target correlation analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import thresholds as T


def correlation_risk(abs_corr: float) -> str | None:
    if abs_corr >= T.CORRELATION_CRITICAL:
        return "Critical"
    if abs_corr >= T.CORRELATION_HIGH:
        return "High"
    if abs_corr >= T.CORRELATION_MEDIUM:
        return "Medium"
    return None


def detect_high_correlation(df: pd.DataFrame, target_encoded: np.ndarray,
                             numeric_feature_cols: list[str]) -> dict[str, dict]:
    """Compute Pearson correlation between each numeric feature and an
    encoded (numeric-compatible) target, and flag unusually high absolute
    correlations using configurable thresholds.

    Note: this is applicable when the target is numeric or a binary target
    that has been encoded to 0/1. High correlation is reported strictly as
    a "strong association" warning — never proof of leakage.
    """
    results = {}
    target_series = pd.Series(target_encoded, index=df.index)

    for col in numeric_feature_cols:
        feat = df[col]
        mask = feat.notna() & target_series.notna()
        if mask.sum() < 10:
            continue
        if feat[mask].std() == 0:
            continue
        corr = feat[mask].corr(target_series[mask])
        if pd.isna(corr):
            continue
        abs_corr = abs(corr)
        risk = correlation_risk(abs_corr)
        if risk is None:
            continue
        results[col] = {
            "correlation": float(corr),
            "risk": risk,
            "reason": (
                f"Strong target association (|r| = {abs_corr:.3f}) — investigate "
                "whether this feature would be available at prediction time. High "
                "correlation does not automatically mean leakage."
            ),
        }
    return results
