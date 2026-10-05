"""
Main orchestrator: runs every detector against a dataset + target column and
returns a complete LeakageScanResult ready for display in the Streamlit UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import thresholds as T
from . import (
    correlation_detector,
    id_detector,
    missingness_detector,
    mutual_information,
    predictive_power,
    problem_type as problem_type_module,
    recommendations,
    risk_engine,
    semantic_detector,
    structural_detector,
)
from .utils import FeatureFinding, is_numeric_series, is_bool_like


@dataclass
class LeakageScanResult:
    target_col: str
    problem_type: str
    is_binary_target: bool
    feature_cols: list[str]
    findings: list = field(default_factory=list)          # list[FeatureFinding], sorted
    duplicate_features: list = field(default_factory=list)
    dataset_model_result: Optional[dict] = None
    dataset_level_warning: Optional[str] = None
    limits_applied: list = field(default_factory=list)


def _encode_target(target: pd.Series, problem_type: str) -> tuple[np.ndarray, bool]:
    """Encode the target into a numeric array suitable for correlation / MI /
    CV-based checks. Returns (encoded_array, is_binary)."""
    if problem_type == "regression":
        encoded = pd.to_numeric(target, errors="coerce").values
        return encoded, False

    # classification
    is_binary = target.dropna().nunique() == 2
    codes, _ = pd.factorize(target, sort=True)
    codes = codes.astype(float)
    codes[codes == -1] = np.nan
    return codes, is_binary


def run_leakage_scan(
    df: pd.DataFrame,
    target_col: str,
    problem_type: Optional[str] = None,
) -> LeakageScanResult:
    """Run the full leakage-detection pipeline and return a structured result."""
    limits_applied = []

    df = df.copy()
    target = df[target_col]

    # Drop rows with missing target for supervised analysis
    valid_target_mask = target.notna()
    if valid_target_mask.sum() < len(df):
        df = df[valid_target_mask].reset_index(drop=True)
        target = df[target_col]

    if problem_type is None:
        problem_type = problem_type_module.infer_problem_type(target)

    target_encoded, is_binary_target = _encode_target(target, problem_type)

    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = [c for c in feature_cols if is_numeric_series(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    if len(feature_cols) > T.MAX_COLUMNS_FOR_FULL_ANALYSIS:
        limits_applied.append(
            f"Dataset has {len(feature_cols)} feature columns; lightweight structural "
            f"checks ran on all of them, but expensive single-feature CV was capped."
        )

    # ---------------- Structural checks ----------------
    exact_near = structural_detector.detect_exact_and_near_duplicates(
        df, target_col, feature_cols
    )
    cat_determinism = structural_detector.detect_categorical_determinism(
        df, target_col, categorical_cols
    ) if problem_type == "classification" else {}
    duplicate_features = structural_detector.detect_duplicate_features(df, feature_cols)

    # ---------------- Correlation (numeric-compatible targets) ----------------
    correlations = {}
    if problem_type == "regression" or is_binary_target:
        correlations = correlation_detector.detect_high_correlation(
            df, target_encoded, [c for c in numeric_cols]
        )

    # ---------------- Mutual information ----------------
    mi_results = mutual_information.compute_mutual_information(
        df, target_col, target_encoded, feature_cols, problem_type
    )

    # ---------------- ID-like detection ----------------
    id_like = id_detector.detect_id_like(df, exclude=[target_col])

    # ---------------- Semantic warnings ----------------
    suspicious_names = semantic_detector.detect_suspicious_names(df, target_col)
    timestamp_warnings = semantic_detector.detect_post_event_timestamps(df, target_col)

    # ---------------- Missingness leakage ----------------
    missingness = missingness_detector.detect_missingness_leakage(
        df, target_col, target_encoded, problem_type, feature_cols
    )

    # ---------------- Single-feature predictive power ----------------
    # Prioritize candidates: anything already flagged, then remaining by MI, capped.
    already_flagged = set(exact_near) | set(cat_determinism) | set(correlations)
    mi_sorted = sorted(
        [c for c in feature_cols if c not in already_flagged],
        key=lambda c: mi_results.get(c, 0) if pd.notna(mi_results.get(c, np.nan)) else 0,
        reverse=True,
    )
    candidate_order = list(already_flagged) + mi_sorted
    if len(candidate_order) > T.MAX_CANDIDATE_FEATURES_FOR_CV:
        limits_applied.append(
            f"Single-feature cross-validated predictive power was computed for the "
            f"{T.MAX_CANDIDATE_FEATURES_FOR_CV} most relevant candidate features "
            f"(out of {len(candidate_order)}), prioritizing features already flagged "
            "by other detectors."
        )
    single_feature = predictive_power.run_single_feature_analysis(
        df, target_col, target_encoded, problem_type, is_binary_target, candidate_order
    )

    # ---------------- Aggregate into per-feature findings ----------------
    findings = risk_engine.build_feature_findings(
        feature_cols=feature_cols,
        exact_near_duplicates=exact_near,
        categorical_determinism=cat_determinism,
        correlations=correlations,
        mutual_info=mi_results,
        single_feature=single_feature,
        id_like=id_like,
        suspicious_names=suspicious_names,
        timestamp_warnings=timestamp_warnings,
        missingness=missingness,
        mi_high_percentile_threshold=T.MUTUAL_INFO_HIGH_PERCENTILE,
        mi_min_abs=T.MUTUAL_INFO_MIN_ABS,
    )
    recommendations.attach_recommendations(findings)
    findings = risk_engine.sort_findings(findings)

    # ---------------- Dataset-level "suspiciously perfect" check ----------------
    usable_features = [c for c in feature_cols if c not in [
        col for col, info in id_like.items() if info["is_id_like"]
    ]]
    dataset_model_result = predictive_power.evaluate_full_model(
        df, target_col, target_encoded, usable_features, problem_type, is_binary_target
    )
    dataset_level_warning = None
    if dataset_model_result and predictive_power.is_dataset_suspiciously_perfect(
        dataset_model_result["metric"], dataset_model_result["score"]
    ):
        dataset_level_warning = (
            f"Unusually high cross-validated performance was detected "
            f"({dataset_model_result['metric']} = {dataset_model_result['score']:.4f} "
            f"using {dataset_model_result['n_features']} features). This does not prove "
            "leakage, but the dataset should be reviewed for leaked or post-outcome features."
        )

    return LeakageScanResult(
        target_col=target_col,
        problem_type=problem_type,
        is_binary_target=is_binary_target,
        feature_cols=feature_cols,
        findings=findings,
        duplicate_features=duplicate_features,
        dataset_model_result=dataset_model_result,
        dataset_level_warning=dataset_level_warning,
        limits_applied=limits_applied,
    )
