"""
Structural leakage checks: exact target duplicates, near-duplicate targets,
target-derived categorical features (near-deterministic mappings), and
duplicate/redundant feature columns.

These are the strongest, most explainable signals LeakGuard produces, so
"Critical" verdicts are reserved almost exclusively for this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from config import thresholds as T
from .utils import is_bool_like, normalize_for_comparison


def _row_agreement(a: pd.Series, b: pd.Series) -> float:
    """Fraction of non-missing, jointly-defined rows where a == b."""
    mask = a.notna() & b.notna()
    if mask.sum() == 0:
        return 0.0
    a_aligned = a[mask].astype(str).str.strip().str.lower()
    b_aligned = b[mask].astype(str).str.strip().str.lower()
    return float((a_aligned == b_aligned).mean())


def detect_exact_and_near_duplicates(df: pd.DataFrame, target_col: str,
                                      feature_cols: list[str]) -> dict[str, dict]:
    """Detect features that are exact or near-exact copies of the target,
    including trivial re-encodings (0/1 vs Yes/No vs True/False)."""
    target = df[target_col]
    target_norm = normalize_for_comparison(target) if is_bool_like(target) else target

    results = {}
    for col in feature_cols:
        feat = df[col]
        feat_norm = normalize_for_comparison(feat) if is_bool_like(feat) else feat

        agreement = _row_agreement(feat_norm, target_norm)

        if agreement >= T.NEAR_DUPLICATE_CRITICAL:
            results[col] = {
                "risk": "Critical",
                "agreement": agreement,
                "reason": "Feature is identical to the target.",
                "tag": "exact_duplicate",
            }
        elif agreement >= T.NEAR_DUPLICATE_HIGH:
            results[col] = {
                "risk": "High",
                "agreement": agreement,
                "reason": (
                    f"Feature matches the target in {agreement * 100:.1f}% of rows. "
                    "This near-perfect correspondence should be investigated — it may "
                    "be a re-encoded or partially-derived version of the target."
                ),
                "tag": "near_duplicate",
            }
    return results


def detect_categorical_determinism(df: pd.DataFrame, target_col: str,
                                    categorical_cols: list[str]) -> dict[str, dict]:
    """Detect categorical features whose categories almost perfectly determine
    the target class (e.g. Account_Status=Closed -> Churn=Yes deterministically)."""
    target = df[target_col]
    results = {}

    for col in categorical_cols:
        feat = df[col]
        mask = feat.notna() & target.notna()
        if mask.sum() < 10:
            continue
        f = feat[mask].astype(str)
        t = target[mask].astype(str)
        if f.nunique() < 2 or f.nunique() > 50:
            continue  # skip constant or extremely high-cardinality categoricals

        # Normalized mutual information between the categorical feature and target
        try:
            nmi = normalized_mutual_info_score(f, t)
        except Exception:
            continue

        # Conditional purity: for each category, how dominant is the majority target class?
        ctab = pd.crosstab(f, t)
        row_totals = ctab.sum(axis=1)
        majority_share = (ctab.max(axis=1) / row_totals)
        weighted_purity = float((majority_share * row_totals).sum() / row_totals.sum())

        if nmi >= T.CATEGORICAL_DETERMINISM_HIGH or weighted_purity >= 0.97:
            risk = "High"
        elif nmi >= T.CATEGORICAL_DETERMINISM_MEDIUM or weighted_purity >= 0.90:
            risk = "Medium"
        else:
            continue

        top_examples = []
        for cat_val, row in ctab.iterrows():
            dominant_class = row.idxmax()
            share = row.max() / row.sum()
            if share >= 0.9:
                top_examples.append(f"'{cat_val}' -> '{dominant_class}' ({share * 100:.0f}%)")
        example_str = "; ".join(top_examples[:3])

        results[col] = {
            "risk": risk,
            "normalized_mi": float(nmi),
            "weighted_purity": weighted_purity,
            "reason": (
                f"Category values in '{col}' almost deterministically map to specific "
                f"target values (e.g. {example_str}). Normalized mutual information = "
                f"{nmi:.3f}. Verify this relationship isn't a proxy for the outcome."
            ),
        }
    return results


def detect_duplicate_features(df: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    """Detect pairs of exactly duplicated or near-identical numerical feature
    columns (redundant features), separate from target leakage."""
    from .utils import is_numeric_series

    findings = []
    seen_pairs = set()

    # Exact duplicate columns (any dtype)
    for i, col_a in enumerate(feature_cols):
        for col_b in feature_cols[i + 1:]:
            pair = tuple(sorted((col_a, col_b)))
            if pair in seen_pairs:
                continue
            try:
                if df[col_a].equals(df[col_b]):
                    findings.append({
                        "columns": pair,
                        "type": "exact_duplicate_feature",
                        "reason": f"'{col_a}' and '{col_b}' are exact duplicate columns.",
                    })
                    seen_pairs.add(pair)
                    continue
            except Exception:
                pass

            if is_numeric_series(df[col_a]) and is_numeric_series(df[col_b]):
                a, b = df[col_a], df[col_b]
                mask = a.notna() & b.notna()
                if mask.sum() < 10:
                    continue
                if a[mask].std() == 0 or b[mask].std() == 0:
                    continue
                corr = a[mask].corr(b[mask])
                if pd.notna(corr) and abs(corr) >= T.REDUNDANT_FEATURE_CORRELATION:
                    findings.append({
                        "columns": pair,
                        "type": "near_identical_numeric",
                        "correlation": float(corr),
                        "reason": (
                            f"'{col_a}' and '{col_b}' are near-identical "
                            f"(correlation = {corr:.4f})."
                        ),
                    })
                    seen_pairs.add(pair)

    return findings
