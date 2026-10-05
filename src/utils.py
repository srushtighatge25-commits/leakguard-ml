"""
Shared utility functions used across LeakGuard ML detectors.
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import thresholds as T


@dataclass
class Evidence:
    """A single piece of evidence attached to a feature's leakage analysis."""
    detector: str
    message: str
    value: Optional[float] = None


@dataclass
class FeatureFinding:
    """Aggregated leakage-analysis result for a single feature."""
    feature: str
    risk: str = "Low"
    reasons: list = field(default_factory=list)
    evidence: list = field(default_factory=list)   # list[Evidence]
    correlation: Optional[float] = None
    mutual_info: Optional[float] = None
    single_feature_score: Optional[float] = None
    single_feature_metric: Optional[str] = None
    is_id_like: bool = False
    tags: list = field(default_factory=list)   # e.g. ["exact_duplicate", "id_like"]
    recommendation: str = ""

    def add_evidence(self, detector: str, message: str, value: Optional[float] = None):
        self.evidence.append(Evidence(detector=detector, message=message, value=value))
        if message not in self.reasons:
            self.reasons.append(message)


RISK_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Review": 3, "Low": 4}


def risk_rank(risk: str) -> int:
    return RISK_ORDER.get(risk, 5)


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def is_bool_like(s: pd.Series) -> bool:
    non_null = s.dropna()
    if non_null.empty:
        return False
    if pd.api.types.is_bool_dtype(s):
        return True
    uniques = set(non_null.unique().tolist())
    bool_sets = [
        {0, 1}, {0.0, 1.0}, {"0", "1"}, {"True", "False"}, {"true", "false"},
        {"Yes", "No"}, {"yes", "no"}, {"Y", "N"}, {"T", "F"},
    ]
    return any(uniques <= bset for bset in bool_sets)


def is_datetime_like(s: pd.Series, name: str = "") -> bool:
    """Best-effort check for whether a column represents a date/time."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    name_lower = name.lower()
    from config.suspicious_patterns import DATETIME_NAME_HINTS
    if any(hint in name_lower for hint in DATETIME_NAME_HINTS):
        sample = s.dropna().astype(str).head(50)
        if sample.empty:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return parsed.notna().mean() > 0.8
    return False


def uniqueness_ratio(s: pd.Series) -> float:
    n = len(s)
    if n == 0:
        return 0.0
    return s.nunique(dropna=True) / n


def safe_label_encode(s: pd.Series) -> np.ndarray:
    """Encode a categorical/object series into integer codes, NaN -> -1."""
    codes, _ = pd.factorize(s, sort=True)
    return codes


def normalize_for_comparison(s: pd.Series) -> pd.Series:
    """Normalize a series so that trivially-equivalent encodings
    (0/1 vs Yes/No vs True/False) can be compared to the target."""
    if is_bool_like(s):
        mapping_true = {"yes", "true", "y", "t", "1", 1, 1.0, True}
        return s.apply(
            lambda v: 1 if str(v).strip().lower() in
            {"yes", "true", "y", "t", "1"} else (0 if pd.notna(v) else np.nan)
        )
    return s


def sample_rows_if_needed(df: pd.DataFrame, max_rows: int = T.MAX_ROWS_FOR_EXPENSIVE_CV,
                           random_state: int = T.RANDOM_STATE) -> pd.DataFrame:
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=random_state)
    return df


def format_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x * 100:.1f}%"


def format_float(x: Optional[float], digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:.{digits}f}"
