import numpy as np
import pandas as pd

from src.missingness_detector import detect_missingness_leakage


def test_missingness_leakage_detected_when_missing_predicts_target():
    n = 800
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, n)

    # Feature is missing almost exactly when target == 1 (simulating a field
    # that's only recorded for non-churned/negative outcomes).
    feature = rng.normal(0, 1, n)
    missing_mask = (target == 1) & (rng.random(n) < 0.95)
    feature = feature.astype(float)
    feature[missing_mask] = np.nan

    df = pd.DataFrame({"target": target, "suspicious_feature": feature})
    results = detect_missingness_leakage(
        df, "target", target.astype(float), "classification", ["suspicious_feature"]
    )
    assert "suspicious_feature" in results
    assert results["suspicious_feature"]["risk"] in ("High", "Medium")


def test_no_missingness_leakage_for_random_missing_pattern():
    n = 800
    rng = np.random.default_rng(1)
    target = rng.integers(0, 2, n)
    feature = rng.normal(0, 1, n).astype(float)
    random_missing = rng.random(n) < 0.1
    feature[random_missing] = np.nan

    df = pd.DataFrame({"target": target, "clean_feature": feature})
    results = detect_missingness_leakage(
        df, "target", target.astype(float), "classification", ["clean_feature"]
    )
    assert "clean_feature" not in results


def test_missingness_skipped_below_minimum_missing_rate():
    n = 300
    rng = np.random.default_rng(2)
    target = rng.integers(0, 2, n)
    feature = rng.normal(0, 1, n).astype(float)
    # Only 1 missing value total -> below MISSINGNESS_MIN_MISSING_RATE
    feature[0] = np.nan

    df = pd.DataFrame({"target": target, "mostly_complete": feature})
    results = detect_missingness_leakage(
        df, "target", target.astype(float), "classification", ["mostly_complete"]
    )
    assert "mostly_complete" not in results
