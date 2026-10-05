"""
LeakGuard ML — Dataset Leakage Detector
Streamlit dashboard entry point.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import thresholds as T
from src.data_loader import load_csv, validate_dataset, dataset_overview, DatasetValidationError
from src.leakage_detector import run_leakage_scan, _encode_target
from src.model_comparison import compare_with_without_features
from src.problem_type import infer_problem_type
from src.synthetic_demo import generate_synthetic_churn_dataset
from src.utils import format_pct, format_float, risk_rank

st.set_page_config(
    page_title="LeakGuard ML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_COLORS = {
    "Critical": "#B00020",
    "High": "#E4572E",
    "Medium": "#E9A400",
    "Review": "#4C6EF5",
    "Low": "#2F9E44",
}

CUSTOM_CSS = """
<style>
.risk-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    color: white;
    font-weight: 600;
    font-size: 0.82rem;
}
.evidence-box {
    background-color: rgba(127,127,127,0.08);
    border-left: 4px solid #4C6EF5;
    padding: 10px 14px;
    border-radius: 4px;
    margin-bottom: 8px;
}
.small-note {
    color: #888;
    font-size: 0.85rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def risk_badge(risk: str) -> str:
    color = RISK_COLORS.get(risk, "#888")
    return f'<span class="risk-badge" style="background-color:{color}">{risk}</span>'


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "result" not in st.session_state:
    st.session_state.result = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None
if "validation_report" not in st.session_state:
    st.session_state.validation_report = None
if "using_demo" not in st.session_state:
    st.session_state.using_demo = False


# ---------------------------------------------------------------------------
# Sidebar: data loading & configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ LeakGuard ML")
    st.caption("Detect potential target leakage before training your machine-learning model.")

    st.markdown("### 1. Load a dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    use_demo = st.button("Use synthetic leakage demo dataset", use_container_width=True)

    if use_demo:
        st.session_state.df = generate_synthetic_churn_dataset()
        st.session_state.using_demo = True
        st.session_state.result = None
        st.session_state.target_col = None

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            df_loaded = load_csv(io.BytesIO(file_bytes))
            st.session_state.df = df_loaded
            st.session_state.using_demo = False
            st.session_state.result = None
            st.session_state.target_col = None
        except DatasetValidationError as e:
            st.error(f"Could not load file: {e}")

    df = st.session_state.df

    if df is not None:
        st.markdown("### 2. Select target")
        target_col = st.selectbox(
            "Target column",
            options=list(df.columns),
            index=(list(df.columns).index(st.session_state.target_col)
                   if st.session_state.target_col in df.columns else len(df.columns) - 1),
        )
        st.session_state.target_col = target_col

        st.markdown("### 3. Problem type")
        auto_infer = st.checkbox("Auto-infer problem type", value=True)
        problem_type_choice = None
        if not auto_infer:
            problem_type_choice = st.radio(
                "Problem type", options=["classification", "regression"], horizontal=True
            )

        st.markdown("### 4. Thresholds (advanced)")
        with st.expander("Adjust heuristic thresholds"):
            st.caption("All thresholds are heuristics, not statistically validated cutoffs.")
            T.CORRELATION_CRITICAL = st.slider(
                "Correlation — Critical", 0.5, 1.0, T.CORRELATION_CRITICAL, 0.01
            )
            T.CORRELATION_HIGH = st.slider(
                "Correlation — High", 0.5, 1.0, T.CORRELATION_HIGH, 0.01
            )
            T.CORRELATION_MEDIUM = st.slider(
                "Correlation — Medium", 0.3, 1.0, T.CORRELATION_MEDIUM, 0.01
            )
            T.ID_UNIQUENESS_RATIO = st.slider(
                "ID uniqueness ratio", 0.5, 1.0, T.ID_UNIQUENESS_RATIO, 0.01
            )

        run_clicked = st.button("🔍 Run Leakage Scan", type="primary", use_container_width=True)

        if run_clicked:
            validation_report = validate_dataset(df, target_col)
            st.session_state.validation_report = validation_report
            if validation_report.is_valid:
                with st.spinner("Running leakage analysis..."):
                    resolved_type = (
                        problem_type_choice if not auto_infer else None
                    )
                    result = run_leakage_scan(df, target_col, problem_type=resolved_type)
                st.session_state.result = result
            else:
                st.session_state.result = None

    st.markdown("---")
    st.caption(
        "LeakGuard ML never proves leakage from statistics alone. "
        "All findings require human review."
    )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("LeakGuard ML")
st.markdown("**Detect potential target leakage before training your machine-learning model.**")

df = st.session_state.df

if df is None:
    st.info(
        "👈 Upload a CSV file from the sidebar, or click **Use synthetic leakage demo dataset** "
        "to try LeakGuard ML immediately."
    )
    st.markdown("""
### What is target leakage?
Target leakage occurs when information that would not actually be available at
prediction time — or that is directly/indirectly derived from the outcome itself —
ends up among the features used to train a model. It's one of the most common (and
most dangerous) mistakes in applied machine learning: models look great in
validation and then fail in the real world.

LeakGuard ML runs a battery of structural, statistical, and cross-validated checks
against your dataset and produces an explainable risk report — without ever
claiming that a statistical association alone *proves* leakage.
""")
    st.stop()

validation_report = st.session_state.validation_report
if validation_report is not None and not validation_report.is_valid:
    for err in validation_report.errors:
        st.error(err)

if validation_report is not None:
    for w in validation_report.warnings:
        st.warning(w)

target_col = st.session_state.target_col
result = st.session_state.result

tab_overview, tab_scan, tab_inspector, tab_compare, tab_learn = st.tabs(
    ["📊 Overview", "🔎 Leakage Scan", "🧪 Feature Inspector", "⚖️ Model Comparison", "📚 Learn"]
)

# ---------------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------------
with tab_overview:
    if st.session_state.using_demo:
        st.info(
            "**Synthetic Leakage Demo** — this dataset is artificially generated to "
            "demonstrate LeakGuard ML. `account_status` was deliberately constructed "
            "as a near-restatement of the `churn` outcome. It is not a real dataset."
        )

    overview = dataset_overview(df, target_col)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{overview['n_rows']:,}")
    c2.metric("Columns", overview["n_columns"])
    c3.metric("Numeric features", overview["n_numeric"])
    c4.metric("Categorical features", overview["n_categorical"])

    c5, c6 = st.columns(2)
    c5.metric("Total missing values", f"{overview['total_missing']:,}")
    c6.metric("Duplicate rows", overview["duplicate_rows"])

    st.markdown("#### Data preview")
    st.dataframe(df.head(20), use_container_width=True)

    if target_col:
        st.markdown(f"#### Target: `{target_col}`")
        if overview.get("target_kind_hint") == "classification":
            dist = overview.get("target_distribution", {})
            fig = px.bar(
                x=list(map(str, dist.keys())), y=list(dist.values()),
                labels={"x": target_col, "y": "Count"},
                title="Target class distribution",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            summary = overview.get("target_summary", {})
            st.write(pd.DataFrame([summary]).T.rename(columns={0: "value"}))
            fig = px.histogram(df, x=target_col, title="Target distribution")
            st.plotly_chart(fig, use_container_width=True)

    if overview["total_missing"] > 0:
        st.markdown("#### Missing values by column")
        missing_series = pd.Series(overview["missing_by_column"])
        missing_series = missing_series[missing_series > 0].sort_values(ascending=False)
        fig = px.bar(
            x=missing_series.index, y=missing_series.values,
            labels={"x": "Column", "y": "Missing count"},
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Leakage Scan tab
# ---------------------------------------------------------------------------
with tab_scan:
    if result is None:
        st.info("Configure your target column and click **Run Leakage Scan** in the sidebar.")
    else:
        st.markdown(
            f"Inferred/selected problem type: **{result.problem_type}** "
            f"({'binary' if result.is_binary_target else 'multiclass/continuous'})"
        )

        if result.dataset_level_warning:
            st.error(f"⚠️ **Dataset-Level Leakage Warning**\n\n{result.dataset_level_warning}")

        if result.limits_applied:
            for msg in result.limits_applied:
                st.caption(f"ℹ️ {msg}")

        # Risk distribution chart
        from src.risk_engine import risk_distribution
        dist = risk_distribution(result.findings)
        order = ["Critical", "High", "Medium", "Review", "Low"]
        fig = go.Figure(go.Bar(
            x=order, y=[dist.get(r, 0) for r in order],
            marker_color=[RISK_COLORS[r] for r in order],
        ))
        fig.update_layout(title="Leakage Risk Distribution", xaxis_title="Risk level",
                           yaxis_title="Number of features")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Suspicious Features Report")
        rows = []
        for f in result.findings:
            rows.append({
                "Feature": f.feature,
                "Risk": f.risk,
                "Reason": f.reasons[0] if f.reasons else "—",
                "Correlation": format_float(f.correlation) if f.correlation is not None else "—",
                "Mutual Info": format_float(f.mutual_info) if f.mutual_info is not None else "—",
                "Single-Feature Score": (
                    f"{f.single_feature_metric}={format_float(f.single_feature_score)}"
                    if f.single_feature_score is not None else "—"
                ),
                "ID-Like": "Yes" if f.is_id_like else "—",
            })
        report_df = pd.DataFrame(rows)

        risk_filter = st.multiselect(
            "Filter by risk level", options=order, default=[r for r in order if r != "Low"]
        )
        filtered = report_df[report_df["Risk"].isin(risk_filter)] if risk_filter else report_df

        st.dataframe(
            filtered.style.map(
                lambda v: f"background-color: {RISK_COLORS.get(v, '')}33" if v in RISK_COLORS else "",
                subset=["Risk"],
            ),
            use_container_width=True,
            hide_index=True,
        )

        if result.duplicate_features:
            st.markdown("#### Duplicate / Redundant Features")
            st.caption("These are flagged as redundancy issues, separate from target leakage.")
            for d in result.duplicate_features:
                st.write(f"- {d['reason']}")

# ---------------------------------------------------------------------------
# Feature Inspector tab
# ---------------------------------------------------------------------------
with tab_inspector:
    if result is None:
        st.info("Run a leakage scan first.")
    else:
        feature_names = [f.feature for f in result.findings]
        selected = st.selectbox("Select a feature to inspect", options=feature_names)
        finding = next(f for f in result.findings if f.feature == selected)

        st.markdown(risk_badge(finding.risk), unsafe_allow_html=True)
        st.markdown(f"### `{finding.feature}`")

        col1, col2, col3 = st.columns(3)
        s = df[finding.feature]
        col1.metric("Data type", str(s.dtype))
        col2.metric("Unique values", s.nunique())
        col3.metric("Missing %", format_pct(s.isna().mean()))

        col4, col5, col6 = st.columns(3)
        col4.metric("Correlation", format_float(finding.correlation)
                    if finding.correlation is not None else "N/A")
        col5.metric("Mutual info (norm.)", format_float(finding.mutual_info)
                    if finding.mutual_info is not None else "N/A")
        col6.metric(
            "Single-feature score",
            f"{finding.single_feature_metric} = {format_float(finding.single_feature_score)}"
            if finding.single_feature_score is not None else "N/A",
        )

        st.markdown("#### Detection rules triggered")
        if finding.tags:
            st.write(", ".join(sorted(set(finding.tags))))
        else:
            st.write("None — no detector flagged this feature.")

        st.markdown("#### Evidence")
        if finding.evidence:
            for ev in finding.evidence:
                st.markdown(f'<div class="evidence-box"><b>{ev.detector}</b>: {ev.message}</div>',
                            unsafe_allow_html=True)
        else:
            st.write("No supporting evidence recorded.")

        st.markdown("#### Recommendation")
        st.success(finding.recommendation)

        st.markdown("#### Target relationship")
        try:
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
                if result.problem_type == "classification":
                    plot_df = pd.DataFrame({finding.feature: s, target_col: df[target_col]})
                    fig = px.box(plot_df, x=target_col, y=finding.feature,
                                 title=f"{finding.feature} vs {target_col}")
                else:
                    plot_df = pd.DataFrame({finding.feature: s, target_col: df[target_col]})
                    fig = px.scatter(plot_df, x=finding.feature, y=target_col,
                                      title=f"{finding.feature} vs {target_col}", opacity=0.5)
                st.plotly_chart(fig, use_container_width=True)
            else:
                plot_df = pd.DataFrame({finding.feature: s.astype(str), target_col: df[target_col].astype(str)})
                ctab = pd.crosstab(plot_df[finding.feature], plot_df[target_col], normalize="index")
                fig = px.imshow(ctab, text_auto=".2f", aspect="auto",
                                 title=f"{finding.feature} category vs {target_col} (row-normalized)")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.caption(f"Could not render relationship plot: {e}")

# ---------------------------------------------------------------------------
# Model Comparison tab
# ---------------------------------------------------------------------------
with tab_compare:
    if result is None:
        st.info("Run a leakage scan first.")
    else:
        st.markdown(
            "Compare a lightweight baseline model's cross-validated performance "
            "**with** vs **without** selected suspicious features."
        )
        suspicious_defaults = [
            f.feature for f in result.findings if risk_rank(f.risk) <= risk_rank("Medium")
        ]
        to_remove = st.multiselect(
            "Features to remove for Model B",
            options=result.feature_cols,
            default=suspicious_defaults,
        )

        if st.button("Run comparison"):
            target_encoded, is_binary = _encode_target(df[target_col], result.problem_type)
            with st.spinner("Training and cross-validating both models..."):
                comparison = compare_with_without_features(
                    df, target_col, target_encoded, result.feature_cols,
                    to_remove, result.problem_type, is_binary,
                )
            if comparison is None:
                st.error("Could not train a comparison model with the current selections.")
            else:
                metric = comparison["metric"]
                model_a = comparison["all_features"]
                model_b = comparison["selected_removed"]

                table_rows = [{
                    "Version": "All Features",
                    metric: format_float(model_a["score"]),
                }]
                if model_b:
                    table_rows.append({
                        "Version": "Selected Features Removed",
                        metric: format_float(model_b["score"]),
                    })
                st.table(pd.DataFrame(table_rows))

                fig = go.Figure(go.Bar(
                    x=[r["Version"] for r in table_rows],
                    y=[model_a["score"]] + ([model_b["score"]] if model_b else []),
                ))
                fig.update_layout(title=f"{metric} comparison", yaxis_title=metric)
                st.plotly_chart(fig, use_container_width=True)

                st.info(comparison["interpretation"])

# ---------------------------------------------------------------------------
# Learn tab
# ---------------------------------------------------------------------------
with tab_learn:
    st.markdown("""
### Target Leakage
Occurs when a feature contains information that would not actually be
available at the moment a prediction is made, or that is directly/indirectly
derived from the outcome itself. Models trained with leaked features often
show unrealistically strong validation performance that collapses in
production.

### Data Leakage (general)
A broader term that includes target leakage, but also covers cases like
train/test contamination (the same records or entities appearing in both
training and validation sets) and group leakage (related records split across
sets in a way that lets information "leak" between them).

### Temporal Leakage
Happens when data from the future — relative to the point a prediction would
actually be made — is used as a feature. Common in time-series and event
prediction problems, where a feature is only recorded after the event of
interest occurs.

### Preprocessing Leakage
Occurs when data transformations (scaling, imputation, feature selection,
encoding) are fit on the *entire* dataset before splitting into train/test or
cross-validation folds, letting information about validation data leak into
training. LeakGuard ML avoids this internally by fitting all preprocessing
steps inside scikit-learn Pipelines within cross-validation folds.

---

**Every finding LeakGuard ML produces is a heuristic warning, not proof.**
Confirming real leakage always requires understanding how and when each
feature is actually generated in the real-world process being modeled.
""")
