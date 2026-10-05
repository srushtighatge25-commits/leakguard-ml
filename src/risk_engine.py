"""
Risk engine: aggregates evidence from all detectors into a single,
explainable Heuristic Leakage Risk verdict per feature.

Design principle: this is NOT a statistically validated probability of
leakage. It is a rule-based aggregation of independent heuristic signals,
always labeled "Heuristic Leakage Risk", never "Probability of Leakage".
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .utils import FeatureFinding, risk_rank


def build_feature_findings(
    feature_cols: list[str],
    exact_near_duplicates: dict,
    categorical_determinism: dict,
    correlations: dict,
    mutual_info: dict,
    single_feature: dict,
    id_like: dict,
    suspicious_names: dict,
    timestamp_warnings: dict,
    missingness: dict,
    mi_high_percentile_threshold: Optional[float],
    mi_min_abs: float,
) -> list[FeatureFinding]:
    """Combine all detector outputs into a list of FeatureFinding objects,
    one per feature, each carrying its aggregated risk level and the
    supporting evidence trail.
    """
    findings = {col: FeatureFinding(feature=col) for col in feature_cols}

    def upgrade(col: str, new_risk: str):
        f = findings[col]
        if risk_rank(new_risk) < risk_rank(f.risk):
            f.risk = new_risk

    # --- Structural: exact / near duplicate target ---
    for col, info in exact_near_duplicates.items():
        f = findings[col]
        f.add_evidence("exact_near_duplicate", info["reason"], info.get("agreement"))
        f.tags.append(info["tag"])
        upgrade(col, info["risk"])

    # --- Structural: categorical determinism ---
    for col, info in categorical_determinism.items():
        f = findings[col]
        f.add_evidence("categorical_determinism", info["reason"], info.get("normalized_mi"))
        f.tags.append("categorical_determinism")
        upgrade(col, info["risk"])

    # --- Correlation ---
    for col, info in correlations.items():
        f = findings[col]
        f.correlation = info["correlation"]
        f.add_evidence("correlation", info["reason"], info["correlation"])
        f.tags.append("high_correlation")
        upgrade(col, info["risk"])

    # --- Mutual information ---
    if mutual_info:
        mi_values = pd.Series(mutual_info).dropna()
        percentile_cutoff = (
            mi_values.quantile(mi_high_percentile_threshold) if len(mi_values) > 0 else None
        )
        for col, mi_val in mutual_info.items():
            f = findings[col]
            f.mutual_info = mi_val
            if pd.isna(mi_val):
                continue
            flagged = mi_val >= mi_min_abs and (
                percentile_cutoff is not None and mi_val >= percentile_cutoff
            )
            if flagged:
                f.add_evidence(
                    "mutual_information",
                    f"Unusually high mutual information with the target "
                    f"(normalized MI = {mi_val:.3f}, top "
                    f"{(1 - mi_high_percentile_threshold) * 100:.0f}% of features). "
                    "High mutual information does not by itself prove leakage.",
                    mi_val,
                )
                f.tags.append("high_mutual_information")
                upgrade(col, "Medium")

    # --- Single-feature predictive power ---
    for col, info in single_feature.items():
        f = findings[col]
        f.single_feature_score = info["score"]
        f.single_feature_metric = info["metric"]
        if info["risk"] != "Low":
            f.add_evidence("single_feature_power", info["reason"], info["score"])
            f.tags.append("strong_single_feature_predictor")
            upgrade(col, info["risk"])

    # --- ID-like ---
    for col, info in id_like.items():
        if info["is_id_like"]:
            f = findings[col]
            f.is_id_like = True
            f.add_evidence(
                "id_like",
                "Identifiers often provide little generalizable signal and may "
                "allow memorization or encode unintended information. " + info["reason"],
                info["uniqueness_ratio"],
            )
            f.tags.append("id_like")
            upgrade(col, "Review")

    # --- Suspicious names (semantic) ---
    for col, reason in suspicious_names.items():
        f = findings[col]
        f.add_evidence("semantic_name", reason)
        f.tags.append("suspicious_name")
        upgrade(col, "Review")

    # --- Timestamp / post-event warnings ---
    for col, reason in timestamp_warnings.items():
        f = findings[col]
        f.add_evidence("timestamp_warning", reason)
        f.tags.append("timestamp_warning")
        upgrade(col, "Review")

    # --- Missingness leakage ---
    for col, info in missingness.items():
        f = findings[col]
        f.add_evidence("missingness", info["reason"], info["auc"])
        f.tags.append("missingness_leakage")
        upgrade(col, info["risk"])

    return list(findings.values())


def sort_findings(findings: list[FeatureFinding]) -> list[FeatureFinding]:
    """Sort so the most suspicious features appear first."""
    return sorted(findings, key=lambda f: (risk_rank(f.risk), f.feature))


def risk_distribution(findings: list[FeatureFinding]) -> dict[str, int]:
    dist = {"Critical": 0, "High": 0, "Medium": 0, "Review": 0, "Low": 0}
    for f in findings:
        dist[f.risk] = dist.get(f.risk, 0) + 1
    return dist
