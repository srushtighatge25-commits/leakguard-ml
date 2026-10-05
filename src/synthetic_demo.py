"""
Generates the synthetic leakage demonstration dataset used only for the
optional educational "Learn" / demo feature. This is NEVER presented as a
real dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_churn_dataset(n_rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 75, size=n_rows)
    income = rng.normal(55000, 15000, size=n_rows).clip(15000, None).round(2)
    tenure_months = rng.integers(1, 120, size=n_rows)

    # Legitimate weak-to-moderate signal for churn based on real, pre-outcome features
    churn_logit = (
        -0.02 * tenure_months
        + -0.00002 * income
        + 0.01 * (age < 25).astype(int) * 10
        + rng.normal(0, 1.5, size=n_rows)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = (rng.random(n_rows) < churn_prob).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    # Deliberately leaked feature: account_closed is basically a restatement
    # of the outcome (post-outcome field), with a tiny bit of noise so it
    # isn't a literal 100% exact duplicate.
    noise_mask = rng.random(n_rows) < 0.02
    account_status_leak = np.where(churn == 1, "Closed", "Active")
    flipped = np.where(noise_mask, np.where(account_status_leak == "Closed", "Active", "Closed"),
                        account_status_leak)

    customer_id = [f"CUST-{100000 + i}" for i in range(n_rows)]

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "income": income,
        "tenure_months": tenure_months,
        "account_status": flipped,   # leaked proxy for churn
        "churn": churn_label,
    })
    return df


if __name__ == "__main__":
    df = generate_synthetic_churn_dataset()
    df.to_csv("data/sample_leakage_demo.csv", index=False)
    print(f"Wrote {len(df)} rows to data/sample_leakage_demo.csv")
