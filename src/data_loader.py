"""
CSV loading and dataset validation.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional, Union

import pandas as pd

from config import thresholds as T


class DatasetValidationError(Exception):
    """Raised when the uploaded CSV cannot be safely analyzed."""


@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str):
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)


def load_csv(file: Union[str, io.BytesIO, io.StringIO]) -> pd.DataFrame:
    """Load a CSV from a path or file-like object. Raises DatasetValidationError
    on malformed or empty files rather than letting pandas throw a raw exception."""
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        raise DatasetValidationError("The uploaded file is empty or contains no columns.")
    except pd.errors.ParserError as e:
        raise DatasetValidationError(f"The CSV file could not be parsed: {e}")
    except UnicodeDecodeError:
        raise DatasetValidationError(
            "The file could not be decoded as text. Please upload a valid UTF-8 CSV file."
        )
    except Exception as e:  # pragma: no cover - defensive catch-all
        raise DatasetValidationError(f"Could not read the uploaded file: {e}")

    if df.shape[1] == 0:
        raise DatasetValidationError("The uploaded file contains no columns.")
    if df.shape[0] == 0:
        raise DatasetValidationError("The uploaded file contains no rows.")

    return df


def validate_dataset(df: pd.DataFrame, target_col: Optional[str] = None) -> ValidationReport:
    """Run structural validation checks on a loaded dataframe."""
    report = ValidationReport()

    if df is None or df.shape[0] == 0:
        report.add_error("Dataset has no rows.")
        return report
    if df.shape[1] == 0:
        report.add_error("Dataset has no columns.")
        return report

    # Duplicate column names
    dup_cols = df.columns[df.columns.duplicated()].unique().tolist()
    if dup_cols:
        report.add_error(
            f"Duplicate column names found: {', '.join(map(str, dup_cols))}. "
            "Please rename or remove duplicate columns before analysis."
        )

    # Target selection checks
    if target_col is not None:
        if target_col not in df.columns:
            report.add_error(f"Target column '{target_col}' was not found in the dataset.")
        elif target_col in dup_cols:
            # Column name is duplicated, so df[target_col] is ambiguous (returns a
            # DataFrame, not a Series). Skip target-specific checks until the
            # duplicate-column error above is resolved.
            pass
        else:
            target_missing_rate = df[target_col].isna().mean()
            if target_missing_rate > 0:
                report.add_warning(
                    f"Target column '{target_col}' has {target_missing_rate * 100:.1f}% "
                    "missing values. Rows with a missing target will be excluded from analysis."
                )
            if df[target_col].nunique(dropna=True) < 2:
                report.add_error(
                    f"Target column '{target_col}' has fewer than 2 unique values "
                    "and cannot be used for supervised analysis."
                )

    # Excessive missingness per column
    missing_rates = df.isna().mean()
    excessive = missing_rates[missing_rates > T.EXCESSIVE_MISSING_COLUMN_RATE]
    if len(excessive) > 0:
        cols_list = ", ".join(f"{c} ({r * 100:.0f}%)" for c, r in excessive.items())
        report.add_warning(
            f"The following columns have excessive missing values and may reduce "
            f"analysis quality: {cols_list}."
        )

    # Wide dataset warning
    if df.shape[1] > T.MAX_COLUMNS_FOR_FULL_ANALYSIS:
        report.add_warning(
            f"Dataset has {df.shape[1]} columns, which exceeds the recommended limit of "
            f"{T.MAX_COLUMNS_FOR_FULL_ANALYSIS}. Expensive checks will be limited to a "
            "subset of candidate features to keep the app responsive."
        )

    # Fully-duplicated rows
    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        report.add_warning(f"Dataset contains {dup_rows} fully duplicated rows.")

    return report


def dataset_overview(df: pd.DataFrame, target_col: Optional[str] = None) -> dict:
    """Compute summary statistics shown in the Overview tab."""
    from .utils import is_numeric_series

    numeric_cols = [c for c in df.columns if is_numeric_series(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    overview = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "total_missing": int(df.isna().sum().sum()),
        "missing_by_column": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    if target_col and target_col in df.columns:
        target = df[target_col]
        overview["target_missing"] = int(target.isna().sum())
        overview["target_unique"] = int(target.nunique(dropna=True))
        if target_col in numeric_cols and overview["target_unique"] > 20:
            overview["target_summary"] = target.describe().to_dict()
            overview["target_kind_hint"] = "regression"
        else:
            overview["target_distribution"] = target.value_counts(dropna=True).to_dict()
            overview["target_kind_hint"] = "classification"

    return overview
