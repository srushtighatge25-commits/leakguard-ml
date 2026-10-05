"""
Central configuration for all heuristic thresholds used by LeakGuard ML.

Every threshold here is a heuristic, not a statistically validated cutoff.
They are intentionally exposed in one place so a user can tune sensitivity
without touching detector logic.
"""

# ---------------------------------------------------------------------------
# Correlation thresholds (absolute Pearson / point-biserial correlation)
# ---------------------------------------------------------------------------
CORRELATION_CRITICAL = 0.98
CORRELATION_HIGH = 0.90
CORRELATION_MEDIUM = 0.75

# ---------------------------------------------------------------------------
# Near-duplicate target agreement (fraction of rows where feature == target,
# after safe normalization/encoding)
# ---------------------------------------------------------------------------
NEAR_DUPLICATE_CRITICAL = 1.0    # exact duplicate
NEAR_DUPLICATE_HIGH = 0.98       # near duplicate, 98%+ row agreement

# ---------------------------------------------------------------------------
# Categorical -> target near-determinism
# (normalized mutual information, 0-1 scale)
# ---------------------------------------------------------------------------
CATEGORICAL_DETERMINISM_HIGH = 0.90
CATEGORICAL_DETERMINISM_MEDIUM = 0.70

# ---------------------------------------------------------------------------
# Mutual information: flag features whose MI is in the top percentile
# or exceeds an absolute normalized threshold.
# ---------------------------------------------------------------------------
MUTUAL_INFO_HIGH_PERCENTILE = 0.95   # top 5% of features by MI
MUTUAL_INFO_MIN_ABS = 0.30           # normalized MI floor to even consider flagging

# ---------------------------------------------------------------------------
# Single-feature predictive power (cross-validated)
# Classification uses ROC-AUC (binary) / macro-F1 (multiclass).
# Regression uses R^2.
# ---------------------------------------------------------------------------
SINGLE_FEATURE_AUC_CRITICAL = 0.98
SINGLE_FEATURE_AUC_HIGH = 0.93
SINGLE_FEATURE_AUC_MEDIUM = 0.85

SINGLE_FEATURE_R2_CRITICAL = 0.98
SINGLE_FEATURE_R2_HIGH = 0.90
SINGLE_FEATURE_R2_MEDIUM = 0.75

SINGLE_FEATURE_F1_CRITICAL = 0.97
SINGLE_FEATURE_F1_HIGH = 0.90
SINGLE_FEATURE_F1_MEDIUM = 0.80

# ---------------------------------------------------------------------------
# Dataset-level ("suspiciously perfect") baseline model performance
# ---------------------------------------------------------------------------
DATASET_AUC_SUSPICIOUS = 0.995
DATASET_F1_SUSPICIOUS = 0.98
DATASET_R2_SUSPICIOUS = 0.995

# ---------------------------------------------------------------------------
# ID-like column detection
# ---------------------------------------------------------------------------
ID_UNIQUENESS_RATIO = 0.98   # fraction of unique values relative to row count

# ---------------------------------------------------------------------------
# Missingness leakage
# (predictive power of the "is_missing" indicator against target)
# ---------------------------------------------------------------------------
MISSINGNESS_MIN_MISSING_RATE = 0.02   # ignore columns with almost no missing values
MISSINGNESS_AUC_HIGH = 0.90
MISSINGNESS_AUC_MEDIUM = 0.75

# ---------------------------------------------------------------------------
# Near-identical numerical column detection (redundant features)
# ---------------------------------------------------------------------------
REDUNDANT_FEATURE_CORRELATION = 0.999

# ---------------------------------------------------------------------------
# Performance / safety limits
# ---------------------------------------------------------------------------
MAX_COLUMNS_FOR_FULL_ANALYSIS = 300     # warn above this
MAX_CANDIDATE_FEATURES_FOR_CV = 40      # cap expensive single-feature CV
MAX_ROWS_FOR_EXPENSIVE_CV = 5000        # sample rows above this for CV-based checks
CV_FOLDS = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Missing value handling in dataset validation
# ---------------------------------------------------------------------------
EXCESSIVE_MISSING_COLUMN_RATE = 0.60    # warn if a column is >60% missing
