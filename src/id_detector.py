"""
Detects columns likely to be identifiers (customer_id, UUIDs, row indices, etc).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import thresholds as T
from config.suspicious_patterns import ID_NAME_SUBSTRINGS
from .utils import uniqueness_ratio, is_numeric_series


def _name_suggests_id(col_name: str) -> bool:
    name = col_name.lower()
    return any(pattern in name for pattern in ID_NAME_SUBSTRINGS)


def _looks_like_continuous_measurement(s: pd.Series) -> bool:
    """Continuous numeric measurements (income, price, sensor readings) are
    naturally near-unique per row and should NOT be treated as identifiers
    just because of high cardinality. We treat a numeric column as
    'continuous' (i.e. NOT an ID candidate on cardinality alone) unless it is
    integer-valued, since IDs are virtually always integer or string codes."""
    non_null = s.dropna()
    if non_null.empty:
        return False
    if not is_numeric_series(s):
        return False
    # Float dtype with non-integer values -> continuous measurement
    try:
        is_all_integral = np.all(np.mod(non_null.astype(float), 1) == 0)
    except (TypeError, ValueError):
        return False
    return not is_all_integral


def detect_id_like(df: pd.DataFrame, exclude: list[str] | None = None) -> dict[str, dict]:
    """Return {column: {"is_id_like": bool, "uniqueness_ratio": float, "name_hint": bool}}
    for every column, with a combined judgment.

    Logic:
      - High uniqueness ratio (near 1 unique value per row) is the primary signal,
        but ONLY for columns that plausibly represent identifiers: text/object
        columns, or integer-valued numeric columns (e.g. row numbers, numeric
        customer IDs). Continuous numeric measurements (income, price, sensor
        readings) are naturally near-unique and are excluded from this
        cardinality-only rule to avoid false positives.
      - A name that looks like an identifier lowers the uniqueness bar slightly,
        since some IDs repeat legitimately (e.g., foreign keys) but are still
        not generalizable.
      - We never flag every high-cardinality column purely on cardinality alone
        (free-text columns can be high-cardinality without being IDs, though we
        accept this as a reasonable heuristic trade-off for structured tabular data).
    """
    exclude = exclude or []
    results = {}
    n_rows = len(df)
    for col in df.columns:
        if col in exclude or n_rows == 0:
            continue
        s = df[col]
        ratio = uniqueness_ratio(s)
        name_hint = _name_suggests_id(col)
        continuous_measurement = _looks_like_continuous_measurement(s)

        is_id = False
        reason = None
        if continuous_measurement and not name_hint:
            # Skip cardinality-only flagging for continuous numeric measurements
            pass
        elif ratio >= T.ID_UNIQUENESS_RATIO:
            is_id = True
            reason = (
                f"Column has a near-unique value for almost every row "
                f"({ratio * 100:.1f}% unique), consistent with an identifier."
            )
        elif name_hint and ratio >= 0.5:
            is_id = True
            reason = (
                f"Column name suggests an identifier and has high cardinality "
                f"({ratio * 100:.1f}% unique)."
            )

        results[col] = {
            "is_id_like": is_id,
            "uniqueness_ratio": ratio,
            "name_hint": name_hint,
            "reason": reason,
        }
    return results
