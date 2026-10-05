import io

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_csv, validate_dataset, DatasetValidationError
from src.problem_type import infer_problem_type
from src.leakage_detector import run_leakage_scan
from src.correlation_detector import detect_high_correlation
from src.structural_detector import detect_duplicate_features


def test_empty_csv_raises_validation_error():
    with pytest.raises(DatasetValidationError):
        load_csv(io.StringIO(""))


def test_csv_with_no_rows_raises_validation_error():
    with pytest.raises(DatasetValidationError):
        load_csv(io.StringIO("col_a,col_b\n"))


def test_missing_target_column_reported_as_error():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    report = validate_dataset(df, target_col="does_not_exist")
    assert not report.is_valid
    assert any("not found" in e for e in report.errors)


def test_duplicate_column_names_reported_as_error():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    report = validate_dataset(df, target_col="a")
    assert not report.is_valid
    assert any("Duplicate column" in e for e in report.errors)


def test_constant_target_reported_as_error():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "target": [1, 1, 1, 1]})
    report = validate_dataset(df, target_col="target")
    assert not report.is_valid


def test_classification_target_inferred_for_categorical_labels():
    target = pd.Series(["Yes", "No", "Yes", "No", "Yes"] * 20)
    assert infer_problem_type(target) == "classification"


def test_regression_target_inferred_for_continuous_values():
    rng = np.random.default_rng(0)
    target = pd.Series(rng.normal(100, 20, 200))
    assert infer_problem_type(target) == "regression"


def test_binary_numeric_target_inferred_as_classification():
    target = pd.Series([0, 1] * 100)
    assert infer_problem_type(target) == "classification"


def test_high_correlation_flagged():
    n = 400
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, n)
    y = x * 5 + rng.normal(0, 0.001, n)
    df = pd.DataFrame({"y": y, "x": x})
    results = detect_high_correlation(df, y, ["x"])
    assert "x" in results
    assert results["x"]["risk"] in ("Critical", "High", "Medium")


def test_low_correlation_not_flagged():
    n = 400
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)  # unrelated
    df = pd.DataFrame({"y": y, "x": x})
    results = detect_high_correlation(df, y, ["x"])
    assert "x" not in results


def test_duplicate_feature_columns_detected():
    n = 200
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, n)
    df = pd.DataFrame({"col_a": a, "col_b": a.copy(), "col_c": rng.normal(0, 1, n)})
    findings = detect_duplicate_features(df, ["col_a", "col_b", "col_c"])
    pairs = [tuple(sorted(f["columns"])) for f in findings]
    assert ("col_a", "col_b") in pairs


def test_end_to_end_regression_pipeline_runs():
    n = 300
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x": x, "noise": rng.normal(0, 1, n)})
    result = run_leakage_scan(df, target_col="y")
    assert result.problem_type == "regression"
    assert len(result.findings) == 2


def test_end_to_end_classification_pipeline_runs():
    n = 300
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "target": target,
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
    })
    result = run_leakage_scan(df, target_col="target")
    assert result.problem_type == "classification"
    assert len(result.findings) == 2


def test_rows_with_missing_target_are_excluded():
    n = 300
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, n).astype(float)
    target[:10] = np.nan
    df = pd.DataFrame({"target": target, "f1": rng.normal(0, 1, n)})
    result = run_leakage_scan(df, target_col="target")
    # Should not raise, and should complete successfully despite missing targets
    assert result.problem_type == "classification"
