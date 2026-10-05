import numpy as np
import pandas as pd
import pytest

from src.leakage_detector import run_leakage_scan
from src.structural_detector import detect_exact_and_near_duplicates


def make_exact_duplicate_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, n)
    return pd.DataFrame({
        "target": target,
        "exact_copy": target,
        "noise": rng.normal(0, 1, n),
    })


def test_exact_target_duplicate_flagged_critical():
    df = make_exact_duplicate_df()
    result = run_leakage_scan(df, target_col="target")
    finding = next(f for f in result.findings if f.feature == "exact_copy")
    assert finding.risk == "Critical"
    assert "exact_duplicate" in finding.tags


def test_exact_duplicate_reason_mentions_identical():
    df = make_exact_duplicate_df()
    findings = detect_exact_and_near_duplicates(df, "target", ["exact_copy", "noise"])
    assert "exact_copy" in findings
    assert findings["exact_copy"]["risk"] == "Critical"
    assert "identical" in findings["exact_copy"]["reason"].lower()


def test_near_duplicate_flagged_high_not_critical():
    rng = np.random.default_rng(1)
    n = 500
    target = rng.integers(0, 2, n)
    near = target.copy()
    flip_idx = rng.choice(n, size=int(n * 0.01), replace=False)
    near[flip_idx] = 1 - near[flip_idx]
    df = pd.DataFrame({"target": target, "near_copy": near, "noise": rng.normal(0, 1, n)})
    findings = detect_exact_and_near_duplicates(df, "target", ["near_copy", "noise"])
    assert "near_copy" in findings
    assert findings["near_copy"]["risk"] in ("Critical", "High")


def test_unrelated_feature_not_flagged_as_duplicate():
    df = make_exact_duplicate_df()
    findings = detect_exact_and_near_duplicates(df, "target", ["noise"])
    assert "noise" not in findings
