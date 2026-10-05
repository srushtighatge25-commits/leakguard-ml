"""
Configurable name-pattern lists used for semantic (column-name) warnings.

Column names alone are NEVER sufficient to declare "Critical" leakage.
They only produce a "Review" / semantic warning that asks a human to check
whether the column would realistically be available at prediction time.
"""

# Generic outcome / post-event vocabulary
SUSPICIOUS_NAME_SUBSTRINGS = [
    "target", "label", "outcome", "result", "final", "status_after",
    "approved", "rejected", "closed", "completed", "diagnosis_result",
    "actual", "future", "next_", "post_", "after_", "final_score",
    "decision", "verdict", "resolution", "settlement", "churned",
    "is_fraud_confirmed", "confirmed", "outcome_flag",
]

# Identifier-style name patterns (used together with uniqueness ratio)
ID_NAME_SUBSTRINGS = [
    "_id", "id_", "uuid", "guid", "customer_id", "transaction_id",
    "order_id", "user_id", "record_id", "account_number", "ssn",
    "email", "phone", "index", "row_number", "rownum",
]

# Timestamp / date column name patterns suggesting post-event information
POST_EVENT_DATE_SUBSTRINGS = [
    "cancellation_date", "closure_date", "completion_date",
    "discharge_date", "resolution_date", "delivery_date", "churn_date",
    "closed_date", "end_date", "termination_date", "settled_date",
    "resolved_at", "closed_at", "completed_at",
]

# General datetime-ish name patterns (used to detect candidate date columns
# even when dtype isn't already datetime)
DATETIME_NAME_HINTS = [
    "date", "_at", "_dt", "timestamp", "time",
]
