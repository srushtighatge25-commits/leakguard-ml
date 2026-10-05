"""
Single-feature predictive power (via lightweight cross-validation) and
dataset-level baseline performance checks.

All model evaluation strictly uses cross-validation, never bare training
accuracy, and all preprocessing (imputation/encoding/scaling) is fit inside
scikit-learn Pipelines within each CV fold to avoid preprocessing leakage.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import thresholds as T
from .utils import is_numeric_series, sample_rows_if_needed


def _make_single_feature_pipeline(is_numeric: bool, problem_type: str) -> Pipeline:
    if is_numeric:
        pre = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        pre = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

    if problem_type == "classification":
        model = LogisticRegression(max_iter=300)
    else:
        model = Ridge()

    return Pipeline([("pre", pre), ("model", model)])


def _safe_n_splits(y: np.ndarray, desired: int = T.CV_FOLDS) -> Optional[int]:
    counts = pd.Series(y).value_counts()
    min_count = counts.min()
    n_splits = min(desired, int(min_count))
    return n_splits if n_splits >= 2 else None


def evaluate_single_feature(df: pd.DataFrame, col: str, target_encoded: np.ndarray,
                             problem_type: str, is_binary_target: bool) -> Optional[dict]:
    """Cross-validate a model using ONLY this one feature to predict the target.
    Returns {"metric": name, "score": float} or None if not computable."""
    feat = df[[col]]
    y = target_encoded
    mask = feat[col].notna() if feat[col].isna().all() else pd.Series(True, index=feat.index)
    mask = mask & ~pd.isna(y)

    X = feat[mask]
    y = np.asarray(y)[mask.values] if hasattr(mask, "values") else np.asarray(y)[mask]

    if len(X) < 20:
        return None

    is_num = is_numeric_series(df[col])
    pipe = _make_single_feature_pipeline(is_num, problem_type)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if problem_type == "classification":
                n_splits = _safe_n_splits(y)
                if n_splits is None:
                    return None
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=T.RANDOM_STATE)
                if is_binary_target:
                    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
                    metric = "ROC-AUC"
                else:
                    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
                    metric = "Macro-F1"
            else:
                cv = KFold(n_splits=T.CV_FOLDS, shuffle=True, random_state=T.RANDOM_STATE)
                scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
                metric = "R2"
        except Exception:
            return None

    return {"metric": metric, "score": float(np.mean(scores))}


def single_feature_risk(metric: str, score: float) -> Optional[str]:
    if metric == "ROC-AUC":
        if score >= T.SINGLE_FEATURE_AUC_CRITICAL:
            return "Critical"
        if score >= T.SINGLE_FEATURE_AUC_HIGH:
            return "High"
        if score >= T.SINGLE_FEATURE_AUC_MEDIUM:
            return "Medium"
    elif metric == "Macro-F1":
        if score >= T.SINGLE_FEATURE_F1_CRITICAL:
            return "Critical"
        if score >= T.SINGLE_FEATURE_F1_HIGH:
            return "High"
        if score >= T.SINGLE_FEATURE_F1_MEDIUM:
            return "Medium"
    elif metric == "R2":
        if score >= T.SINGLE_FEATURE_R2_CRITICAL:
            return "Critical"
        if score >= T.SINGLE_FEATURE_R2_HIGH:
            return "High"
        if score >= T.SINGLE_FEATURE_R2_MEDIUM:
            return "Medium"
    return None


def run_single_feature_analysis(df: pd.DataFrame, target_col: str,
                                 target_encoded: np.ndarray, problem_type: str,
                                 is_binary_target: bool,
                                 candidate_features: list[str]) -> dict[str, dict]:
    """Run single-feature CV evaluation over a (possibly capped) list of
    candidate features, for performance safety on wide datasets."""
    results = {}
    capped = candidate_features[:T.MAX_CANDIDATE_FEATURES_FOR_CV]

    df_sampled = sample_rows_if_needed(df)
    if len(df_sampled) != len(df):
        target_encoded_sampled = pd.Series(target_encoded, index=df.index).loc[df_sampled.index].values
    else:
        target_encoded_sampled = target_encoded

    for col in capped:
        outcome = evaluate_single_feature(
            df_sampled, col, target_encoded_sampled, problem_type, is_binary_target
        )
        if outcome is None:
            continue
        risk = single_feature_risk(outcome["metric"], outcome["score"])
        if risk is None:
            results[col] = {**outcome, "risk": "Low"}
            continue
        results[col] = {
            **outcome,
            "risk": risk,
            "reason": (
                f"This single feature alone predicts the target with "
                f"{outcome['metric']} = {outcome['score']:.3f} using cross-validation. "
                "Near-perfect single-feature predictive power may indicate target "
                "leakage — verify when and how this feature is generated."
            ),
        }
    return results


def evaluate_full_model(df: pd.DataFrame, target_col: str, target_encoded: np.ndarray,
                         feature_cols: list[str], problem_type: str,
                         is_binary_target: bool) -> Optional[dict]:
    """Train/evaluate a lightweight baseline model using ALL usable features,
    via cross-validation, to check for suspiciously perfect dataset-level
    performance (Section 17)."""
    if not feature_cols:
        return None

    df_sampled = sample_rows_if_needed(df)
    if len(df_sampled) != len(df):
        y = pd.Series(target_encoded, index=df.index).loc[df_sampled.index].values
    else:
        y = target_encoded

    X = df_sampled[feature_cols]
    valid = ~pd.isna(y)
    X, y = X[valid], np.asarray(y)[valid]
    if len(X) < 20:
        return None

    numeric_cols = [c for c in feature_cols if is_numeric_series(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ))
    pre = ColumnTransformer(transformers)

    model = LogisticRegression(max_iter=300) if problem_type == "classification" else Ridge()
    pipe = Pipeline([("pre", pre), ("model", model)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if problem_type == "classification":
                n_splits = _safe_n_splits(y)
                if n_splits is None:
                    return None
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=T.RANDOM_STATE)
                if is_binary_target:
                    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
                    metric = "ROC-AUC"
                else:
                    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
                    metric = "Macro-F1"
            else:
                cv = KFold(n_splits=T.CV_FOLDS, shuffle=True, random_state=T.RANDOM_STATE)
                scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
                metric = "R2"
        except Exception:
            return None

    return {"metric": metric, "score": float(np.mean(scores)), "n_features": len(feature_cols)}


def is_dataset_suspiciously_perfect(metric: str, score: float) -> bool:
    if metric == "ROC-AUC":
        return score >= T.DATASET_AUC_SUSPICIOUS
    if metric == "Macro-F1":
        return score >= T.DATASET_F1_SUSPICIOUS
    if metric == "R2":
        return score >= T.DATASET_R2_SUSPICIOUS
    return False
