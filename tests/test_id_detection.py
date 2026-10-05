import numpy as np
import pandas as pd

from src.id_detector import detect_id_like


def test_id_like_column_detected():
    n = 300
    df = pd.DataFrame({
        "customer_id": [f"CUST-{i}" for i in range(n)],
        "target": np.random.default_rng(0).integers(0, 2, n),
    })
    results = detect_id_like(df, exclude=["target"])
    assert results["customer_id"]["is_id_like"] is True


def test_continuous_numeric_not_flagged_as_id():
    """Continuous numeric measurements (e.g. income) are naturally near-unique
    and should NOT be flagged as identifiers purely on cardinality."""
    n = 500
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "income": rng.normal(50000, 15000, n),
        "target": rng.integers(0, 2, n),
    })
    results = detect_id_like(df, exclude=["target"])
    assert results["income"]["is_id_like"] is False


def test_low_cardinality_column_not_flagged_as_id():
    n = 300
    df = pd.DataFrame({
        "category": np.random.default_rng(3).choice(["A", "B", "C"], size=n),
        "target": np.random.default_rng(3).integers(0, 2, n),
    })
    results = detect_id_like(df, exclude=["target"])
    assert results["category"]["is_id_like"] is False


def test_integer_row_index_flagged_as_id():
    n = 300
    df = pd.DataFrame({
        "record_id": np.arange(n),
        "target": np.random.default_rng(4).integers(0, 2, n),
    })
    results = detect_id_like(df, exclude=["target"])
    assert results["record_id"]["is_id_like"] is True
