# LeakGuard ML — Dataset Leakage Detector

**An interactive ML validation tool that detects suspicious features and potential target leakage before model training.**

## Why This Project?

Target leakage is one of the most common — and most dangerous — mistakes in
applied machine learning. It happens when information that wouldn't actually
be available at prediction time, or that is directly/indirectly derived from
the outcome itself, ends up among a model's training features. The result is
almost always the same story: validation metrics look outstanding
(sometimes near-perfect), and then the model fails badly once it's deployed
against real, un-leaked data.

Leakage is also easy to miss. It rarely announces itself — it hides in an
innocently-named column, a field that happens to be populated only after an
event occurs, or a preprocessing step that was accidentally fit on the whole
dataset before splitting. LeakGuard ML runs a battery of structural,
statistical, and cross-validated checks against an uploaded dataset and
produces an explainable risk report, so leakage can be caught **before**
a model is trained on it — not after it fails in production.

## Features

- Upload any CSV dataset and get instant dataset validation and an overview
- Automatic (or manual) classification / regression problem-type inference
- A modular leakage-detection engine covering 10+ independent heuristics
- An explainable, tag-based **Heuristic Leakage Risk** score per feature —
  never presented as a validated "probability of leakage"
- A Feature Inspector for drilling into any flagged column's evidence
- An optional before/after model comparison to see how much predictive power
  a suspicious feature actually contributes
- A synthetic, clearly-labeled leakage demo dataset for learning the tool
- A clean, interactive Streamlit dashboard with Plotly visualizations

## Types of Leakage Checked

- Exact target duplicates
- Near-target duplicates (near-perfect row-level agreement with the target)
- Target-derived / near-deterministic categorical features
- Strong feature-target correlations (configurable thresholds)
- Mutual information (classification and regression)
- Single-feature predictive power via cross-validation
- ID-like columns (identifiers)
- Suspicious semantic column names
- Timestamp / post-outcome date warnings
- Missingness leakage (whether a feature's *missing pattern* predicts the target)
- Duplicate / redundant feature columns
- Suspiciously perfect dataset-level cross-validated performance

## How It Works

```
CSV Dataset
     ↓
Select Target
     ↓
Dataset Validation
     ↓
Structural Checks        (exact/near duplicates, categorical determinism, redundant features)
     ↓
Statistical Checks       (correlation, mutual information)
     ↓
Single-Feature Validation (cross-validated predictive power)
     ↓
Semantic Warnings         (names, timestamps, ID-like columns, missingness)
     ↓
Risk Engine               (aggregates all evidence into one verdict per feature)
     ↓
Leakage Report
     ↓
Model Comparison (optional)
```

Every check is implemented as an independent, modular detector. The risk
engine combines their outputs — it never invents a single "AI-scored"
probability; every verdict traces back to specific, inspectable evidence.

## Tech Stack

- **Python** — pandas, numpy, scipy
- **scikit-learn** — pipelines, cross-validation, mutual information, baseline models
- **Streamlit** — interactive dashboard
- **Plotly** — interactive charts
- **pytest** — test suite

No paid APIs, no external LLMs, no cloud dependencies. Everything runs
locally.

## Project Structure

```
leakguard-ml/
├── app.py                      # Streamlit dashboard
├── src/
│   ├── data_loader.py          # CSV loading & dataset validation
│   ├── problem_type.py         # Classification/regression inference
│   ├── structural_detector.py  # Exact/near duplicates, categorical determinism, redundant features
│   ├── correlation_detector.py # Numerical feature-target correlation
│   ├── mutual_information.py   # Mutual information analysis
│   ├── predictive_power.py     # Single-feature & dataset-level CV performance
│   ├── id_detector.py          # ID-like column detection
│   ├── semantic_detector.py    # Column-name & timestamp warnings
│   ├── missingness_detector.py # Missingness-pattern leakage
│   ├── risk_engine.py          # Aggregates evidence into per-feature risk
│   ├── recommendations.py      # Deterministic, rule-based recommendations
│   ├── model_comparison.py     # Before/after suspicious-feature removal
│   ├── synthetic_demo.py       # Synthetic leakage demo dataset generator
│   └── utils.py                # Shared helpers & data structures
├── config/
│   ├── thresholds.py           # All heuristic thresholds, in one place
│   └── suspicious_patterns.py  # Configurable name-pattern lists
├── data/
│   └── sample_leakage_demo.csv # Synthetic demo dataset
├── tests/                      # pytest suite
├── docs/screenshots/           # Screenshot placeholders
├── requirements.txt
└── README.md
```

## Leakage Risk Levels

| Level | Meaning |
|---|---|
| **Critical** | Strong structural evidence — e.g. an exact target duplicate or a deterministic target encoding |
| **High** | Multiple strong leakage signals, or one very strong signal (near-duplicate, near-perfect single-feature predictor) |
| **Medium** | A suspicious statistical relationship (correlation, mutual information) that requires investigation |
| **Review** | A semantic, timestamp, ID-like, or name-based warning requiring human judgment — never a statistically-derived verdict |
| **Low** | No detector flagged this feature |

**Many results are warnings requiring domain knowledge, not proof of
leakage.** Statistical association — even a very strong one — never proves
that a feature is unavailable at prediction time. Only a human who
understands the real-world data-generating process can make that call.

## Installation

```bash
git clone <repository-url>
cd leakguard-ml
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Run the tests:

```bash
pytest
```

## Example

Click **"Use synthetic leakage demo dataset"** in the sidebar to load a
generated churn dataset. It contains legitimate features (`age`, `income`,
`tenure_months`) alongside a deliberately-leaked feature, `account_status`,
which was constructed to almost perfectly restate the `churn` outcome
(the kind of field that, in a real dataset, is often only populated *after*
a customer has already churned). Run a scan and you'll see `account_status`
flagged as **High** risk, driven by both a near-deterministic categorical
mapping to the target and a near-perfect single-feature cross-validated
ROC-AUC.

This demo dataset is **synthetic** and is clearly labeled as such throughout
the app — it is never presented as real data.

## Screenshots

*(Placeholders — capture real screenshots after running the application.)*

- `docs/screenshots/overview.png`
- `docs/screenshots/leakage_scan.png`
- `docs/screenshots/feature_inspector.png`
- `docs/screenshots/feature_inspector(1).png`
- `docs/screenshots/model_comparison.png`

## Limitations

- Statistical relationships (correlation, mutual information, single-feature
  predictive power) cannot prove leakage on their own — they are evidence
  that something warrants investigation, not proof.
- Domain knowledge about how and when each feature is generated is always
  necessary to reach a final verdict.
- The tool cannot always determine whether a feature would genuinely be
  available at the moment a real-world prediction is made — it can only flag
  patterns that are commonly associated with leakage.
- Semantic column-name checks are heuristic pattern matches, not a
  guaranteed classification of a column's real-world meaning.
- Temporal leakage detection is limited to name-based heuristics; true
  temporal-split analysis requires event timestamps that aren't always present.
- High predictive power from a single feature or the full dataset can be
  entirely legitimate for some problems — it is not automatically leakage.
- Preprocessing leakage that occurs *outside* the uploaded dataset (e.g. in
  an upstream pipeline the user doesn't control) cannot be detected here.

## Future Improvements

- Train/test contamination detection
- Entity overlap detection
- Temporal split analysis using explicit timestamps
- Group leakage detection
- Feature provenance tracking
- SHAP-based investigation of flagged features
- Pipeline leakage auditing
- Automated ML validation reports
