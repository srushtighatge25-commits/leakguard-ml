"""
Optional educational feature: compare a lightweight baseline model's
cross-validated performance with vs. without user-selected suspicious
features.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .predictive_power import evaluate_full_model


def compare_with_without_features(df: pd.DataFrame, target_col: str,
                                   target_encoded: np.ndarray,
                                   all_feature_cols: list[str],
                                   features_to_remove: list[str],
                                   problem_type: str,
                                   is_binary_target: bool) -> Optional[dict]:
    """Train Model A (all usable features) and Model B (features remaining
    after removing user-selected suspicious columns), both via cross-validation,
    and return their scores for side-by-side comparison."""
    remaining_cols = [c for c in all_feature_cols if c not in features_to_remove]

    model_a = evaluate_full_model(
        df, target_col, target_encoded, all_feature_cols, problem_type, is_binary_target
    )
    model_b = evaluate_full_model(
        df, target_col, target_encoded, remaining_cols, problem_type, is_binary_target
    ) if remaining_cols else None

    if model_a is None:
        return None

    result = {
        "metric": model_a["metric"],
        "all_features": model_a,
        "selected_removed": model_b,
        "removed_features": features_to_remove,
        "remaining_feature_count": len(remaining_cols),
    }

    if model_b is not None:
        drop = model_a["score"] - model_b["score"]
        result["score_drop"] = float(drop)
        if drop > 0.05:
            result["interpretation"] = (
                f"Removing the selected feature(s) decreased {model_a['metric']} by "
                f"{drop:.3f}. A large decrease can be supporting evidence that the "
                "removed features carried substantial predictive information, but "
                "human review is still necessary to determine whether that "
                "information was legitimately available at prediction time."
            )
        else:
            result["interpretation"] = (
                f"Removing the selected feature(s) changed {model_a['metric']} by only "
                f"{drop:.3f}. This suggests the removed features were not the dominant "
                "source of predictive power in this baseline model, though they may "
                "still warrant review."
            )
    else:
        result["interpretation"] = (
            "No usable features remained after removal, so a comparison model "
            "could not be trained."
        )

    return result
