import numpy as np
import pandas as pd
import pytest

from src.leakage_detector import run_leakage_scan
from src.risk_engine import risk_distribution, sort_findings
from src.utils import risk_rank


def make_mixed_risk_df(n=600, seed=0):
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, n)
    return pd.DataFrame({
        "target": target,
        "exact_copy": target,                          # Critical
        "mild_corr": target * 0.3 + rng.normal(0, 1, n),  # Low/Medium
        "pure_noise": rng.normal(0, 1, n),              # Low
        "customer_id": [f"C{i}" for i in range(n)],     # Review/ID-like
    })


def test_findings_sorted_most_suspicious_first():
    df = make_mixed_risk_df()
    result = run_leakage_scan(df, target_col="target")
    ranks = [risk_rank(f.risk) for f in result.findings]
    assert ranks == sorted(ranks)


def test_risk_distribution_counts_all_features():
    df = make_mixed_risk_df()
    result = run_leakage_scan(df, target_col="target")
    dist = risk_distribution(result.findings)
    assert sum(dist.values()) == len(result.findings)
    assert dist["Critical"] >= 1  # exact_copy should be critical


def test_low_risk_feature_has_no_tags():
    df = make_mixed_risk_df()
    result = run_leakage_scan(df, target_col="target")
    noise_finding = next(f for f in result.findings if f.feature == "pure_noise")
    assert noise_finding.risk == "Low"


def test_risk_score_never_labeled_as_probability():
    """Ensure no recommendation or reason text falsely claims a proven
    probability of leakage."""
    df = make_mixed_risk_df()
    result = run_leakage_scan(df, target_col="target")
    for f in result.findings:
        for reason in f.reasons:
            assert "probability of leakage" not in reason.lower()
