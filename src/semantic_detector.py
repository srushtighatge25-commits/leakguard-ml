"""
Semantic (name-based) warnings: suspicious column names and post-event
timestamp columns. These NEVER produce a Critical verdict on their own —
they are heuristic "Review" flags that ask a human to check availability
at prediction time.
"""
from __future__ import annotations

import pandas as pd

from config.suspicious_patterns import SUSPICIOUS_NAME_SUBSTRINGS, POST_EVENT_DATE_SUBSTRINGS
from .utils import is_datetime_like


def detect_suspicious_names(df: pd.DataFrame, target_col: str) -> dict[str, str]:
    """Return {column: explanation} for columns whose NAME matches a
    leakage-related vocabulary pattern."""
    findings = {}
    for col in df.columns:
        if col == target_col:
            continue
        name = col.lower()
        for pattern in SUSPICIOUS_NAME_SUBSTRINGS:
            if pattern in name:
                findings[col] = (
                    f"Column name '{col}' contains the pattern '{pattern}', which "
                    "often describes information available only after the "
                    "prediction event. Review manually to confirm this feature "
                    "would be known at prediction time."
                )
                break
    return findings


def detect_post_event_timestamps(df: pd.DataFrame, target_col: str) -> dict[str, str]:
    """Return {column: explanation} for datetime-like columns whose name
    suggests they represent information generated after the event being
    predicted (e.g. cancellation_date, closure_date)."""
    findings = {}
    for col in df.columns:
        if col == target_col:
            continue
        name = col.lower()
        looks_datetime = is_datetime_like(df[col], name)
        name_match = any(p in name for p in POST_EVENT_DATE_SUBSTRINGS)
        if name_match:
            findings[col] = (
                f"'{col}' appears to be a timestamp representing information "
                "that occurs after the prediction point (e.g., an event closure "
                "or resolution date). This may not be available at prediction "
                "time — review manually."
            )
        elif looks_datetime:
            findings[col] = (
                f"'{col}' looks like a date/time column. Confirm it reflects "
                "information available at prediction time rather than a "
                "post-outcome timestamp."
            )
    return findings
