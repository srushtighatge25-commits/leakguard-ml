"""
Deterministic, rule-based recommendations for each flagged feature.
No external LLM is used — recommendations are generated from detection tags.
"""
from __future__ import annotations

from .utils import FeatureFinding

# Priority order: earlier tags produce the primary recommendation shown first.
_TAG_PRIORITY = [
    "exact_duplicate",
    "near_duplicate",
    "categorical_determinism",
    "strong_single_feature_predictor",
    "missingness_leakage",
    "id_like",
    "high_correlation",
    "high_mutual_information",
    "timestamp_warning",
    "suspicious_name",
]

_TAG_RECOMMENDATIONS = {
    "exact_duplicate": "Remove this feature before model training.",
    "near_duplicate": (
        "Investigate why this feature nearly matches the target. If it is a "
        "re-encoded or partially derived version of the target, remove it."
    ),
    "categorical_determinism": (
        "Verify whether this category is set as a consequence of the outcome "
        "(a proxy for the target) rather than a cause known beforehand. If so, remove it."
    ),
    "strong_single_feature_predictor": (
        "Investigate how and when this feature is generated. Near-perfect predictive "
        "power from one variable may indicate target leakage."
    ),
    "missingness_leakage": (
        "Check whether the reason this value is missing is itself related to the "
        "outcome (e.g., only recorded after the event occurs). Consider whether the "
        "missingness pattern should be excluded or handled differently."
    ),
    "id_like": (
        "Consider removing this identifier unless it has a valid generalizable "
        "relationship with the prediction task."
    ),
    "high_correlation": (
        "Strong correlation alone does not prove leakage. Confirm the feature's "
        "real-world availability at prediction time."
    ),
    "high_mutual_information": (
        "High mutual information does not by itself prove leakage. Confirm this "
        "feature would realistically be known before the prediction is made."
    ),
    "timestamp_warning": (
        "Verify whether this information exists at the moment the prediction is made. "
        "If it becomes available afterward, exclude it."
    ),
    "suspicious_name": (
        "Review this column's real-world source and timing. Column names alone are "
        "not proof of leakage, but this name pattern is commonly associated with "
        "post-outcome information."
    ),
}


def recommend_for_finding(finding: FeatureFinding) -> str:
    """Pick the highest-priority recommendation applicable to this feature,
    based on the detection tags that were triggered."""
    for tag in _TAG_PRIORITY:
        if tag in finding.tags:
            return _TAG_RECOMMENDATIONS[tag]
    if finding.tags:
        return "Review this feature manually to confirm it does not encode outcome information."
    return "No action needed based on current heuristics."


def attach_recommendations(findings: list[FeatureFinding]) -> None:
    for f in findings:
        f.recommendation = recommend_for_finding(f)
