"""
SkyPulse - Airline Customer Satisfaction Machine Learning Web Application
Interactive Dashboard, Dataset Explorer, Model Training, and Prediction Engine.
"""

import os
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_preprocessing import (
    find_dataset_path,
    load_dataset,
    inspect_dataset,
    detect_target_column,
    detect_task_type,
    identify_columns
)
from src.train_model import (
    run_full_training,
    load_model_artifacts,
    get_default_hyperparameters
)
from src.evaluation import evaluate_model
from src.predict import (
    predict_sample,
    get_default_feature_values,
    get_preset_samples
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM MODERN STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SkyPulse | Airline Satisfaction ML",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
/* Modern typography and root variables */
:root {
    --primary: #4F46E5;
    --primary-light: #EEF2FF;
    --secondary: #06B6D4;
    --accent: #7C3AED;
    --success: #10B981;
    --warning: #F59E0B;
    --danger: #EF4444;
    --bg-main: #F8FAFC;
    --card-bg: #FFFFFF;
    --text-main: #0F172A;
    --text-muted: #64748B;
    --border-color: #E2E8F0;
}

/* Base app styling */
.main {
    background-color: var(--bg-main);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: var(--text-main);
}

/* Card container */
.metric-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 22px 24px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 16px;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.07), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
}

.metric-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-main);
    line-height: 1.1;
}

.metric-sub {
    font-size: 0.82rem;
    color: var(--primary);
    margin-top: 6px;
    font-weight: 500;
}

/* Section Header */
.section-header {
    margin-top: 10px;
    margin-bottom: 22px;
}

.section-title {
    font-size: 1.65rem;
    font-weight: 800;
    color: #1E293B;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-desc {
    color: #64748B;
    font-size: 0.95rem;
    margin-top: 4px;
}

/* Prediction results */
.pred-card-satisfied {
    background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
    border: 1.5px solid #10B981;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.2);
}

.pred-card-dissatisfied {
    background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
    border: 1.5px solid #EF4444;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.2);
}

.badge-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 6px;
}

.badge-indigo {
    background-color: #EEF2FF;
    color: #4F46E5;
}

.badge-emerald {
    background-color: #ECFDF5;
    color: #059669;
}

/* Custom button tweaks */
div.stButton > button:first-child {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s;
}

/* Sidebar style */
section[data-testid="stSidebar"] {
    background-color: #F8FAFC;
    border-right: 1px solid #E2E8F0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATA AND MODEL CACHING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dataset():
    """Loads and caches the dataset."""
    df, path = load_dataset()
    info = inspect_dataset(df)
    return df, path, info


def get_model():
    """Loads saved model artifacts from models/."""
    artifacts = load_model_artifacts("models")
    return artifacts


# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & INFO
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: linear-gradient(135deg, #4F46E5, #7C3AED); width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px; color: white;">
                ✈️
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #1E293B;">SkyPulse ML</h3>
                <p style="margin: 0; font-size: 0.8rem; color: #64748B;">Satisfaction Intelligence</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation Menu",
        options=[
            "📊 Dashboard",
            "🔍 Dataset Explorer",
            "⚙️ Model Training",
            "🎯 Prediction Studio"
        ],
        index=0
    )

    st.markdown("---")

    # Dynamic model status badge
    saved_artifacts = get_model()
    if saved_artifacts:
        pipeline, meta = saved_artifacts
        acc = meta.get("evaluation", {}).get("accuracy")
        acc_text = f"{acc * 100:.1f}%" if acc else "Active"
        st.markdown(f"""
        <div style="background: #F1F5F9; border-radius: 10px; padding: 12px 14px; border: 1px solid #E2E8F0; margin-bottom: 12px;">
            <div style="font-size: 0.72rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Engine Status</div>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 4px;">
                <span style="font-size: 0.88rem; font-weight: 700; color: #059669;">● Model Ready</span>
                <span style="font-size: 0.85rem; font-weight: 700; color: #4F46E5;">{acc_text} Acc</span>
            </div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">Algorithm: Random Forest</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #FFFBEB; border-radius: 10px; padding: 12px 14px; border: 1px solid #FDE68A; margin-bottom: 12px;">
            <div style="font-size: 0.75rem; color: #B45309; font-weight: 600;">⚠️ Model Not Trained Yet</div>
            <div style="font-size: 0.75rem; color: #78350F; margin-top: 4px;">Navigate to Model Training to train.</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption("Antigravity ML Suite • Scikit-Learn • Streamlit")


# -----------------------------------------------------------------------------
# LOAD DATASET
# -----------------------------------------------------------------------------
try:
    df, dataset_path, info = get_dataset()
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# PAGE A: DASHBOARD
# -----------------------------------------------------------------------------
if page == "📊 Dashboard":
    st.markdown("""
    <div class="section-header">
        <div class="section-title">✈️ Airline Customer Satisfaction Dashboard</div>
        <div class="section-desc">End-to-End Random Forest predictive machine learning platform analyzing passenger sentiments, flight metrics, and inflight services.</div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Records</div>
            <div class="metric-value">{info['total_rows']:,}</div>
            <div class="metric-sub">✓ Validated Data</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Features</div>
            <div class="metric-value">{len(info['numeric_features']) + len(info['categorical_features'])}</div>
            <div class="metric-sub">{len(info['numeric_features'])} Numeric • {len(info['categorical_features'])} Categorical</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Target Variable</div>
            <div class="metric-value" style="font-size: 1.55rem; text-transform: capitalize;">{info['target_column']}</div>
            <div class="metric-sub">Task: {info['task_type'].capitalize()}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        saved = get_model()
        model_acc = "N/A"
        if saved and "evaluation" in saved[1]:
            acc_val = saved[1]["evaluation"].get("accuracy")
            if acc_val:
                model_acc = f"{acc_val * 100:.2f}%"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Model Accuracy</div>
            <div class="metric-value" style="color: #4F46E5;">{model_acc}</div>
            <div class="metric-sub">Random Forest Classifier</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Middle Visualizations Row
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.markdown("#### 🎯 Target Class Distribution")
        if info["task_type"] == "classification" and info["class_distribution"]:
            labels = list(info["class_distribution"].keys())
            values = [v["count"] for v in info["class_distribution"].values()]
            
            fig_donut = go.Figure(data=[go.Pie(
                labels=[l.capitalize() for l in labels],
                values=values,
                hole=0.55,
                marker=dict(colors=["#4F46E5", "#06B6D4"]),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                insidetextorientation="radial"
            )])
            fig_donut.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
            # Quick summary text
            c1, c2 = st.columns(2)
            for i, (k, v) in enumerate(info["class_distribution"].items()):
                with (c1 if i == 0 else c2):
                    st.caption(f"**{k.capitalize()}**: {v['count']:,} passengers ({v['percentage']}%)")

    with col_right:
        st.markdown("#### 🏆 Key Satisfaction Drivers (Top Features)")
        saved = get_model()
        if saved and "evaluation" in saved[1] and saved[1]["evaluation"].get("top_features"):
            top_f = pd.DataFrame(saved[1]["evaluation"]["top_features"][:6])
            fig_bar = px.bar(
                top_f,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=["#C7D2FE", "#4F46E5"],
                text=[f"{imp*100:.1f}%" for imp in top_f["importance"]]
            )
            fig_bar.update_layout(
                yaxis=dict(autorange="reversed", title=""),
                xaxis=dict(title="Importance Weight", showgrid=True),
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                coloraxis_showscale=False
            )
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Train the model in the 'Model Training' tab to reveal feature importance rankings.")

    st.markdown("---")

    # Workflow & Architecture Summary
    st.markdown("#### 🛠️ Pipeline Architecture")
    c_arch1, c_arch2, c_arch3 = st.columns(3)

    with c_arch1:
        st.markdown("""
        <div class="metric-card" style="min-height: 190px;">
            <div style="font-size: 1.3rem; margin-bottom: 6px;">📥 1. Automated Ingestion</div>
            <p style="color: #64748B; font-size: 0.88rem;">
                Auto-locates <code>Airline_customer_satisfaction.csv</code>, inspects schema, infers binary target (<code>satisfaction</code>), and isolates 18 numerical and 3 categorical predictors.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c_arch2:
        st.markdown("""
        <div class="metric-card" style="min-height: 190px;">
            <div style="font-size: 1.3rem; margin-bottom: 6px;">⚙️ 2. Sklearn Preprocessing</div>
            <p style="color: #64748B; font-size: 0.88rem;">
                Built-in <code>ColumnTransformer</code> handles missing delay values via <b>Median Imputation</b> and encodes categorical variables (Customer Type, Class, Travel Type) via <b>One-Hot Encoding</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c_arch3:
        st.markdown("""
        <div class="metric-card" style="min-height: 190px;">
            <div style="font-size: 1.3rem; margin-bottom: 6px;">🌲 3. Random Forest Engine</div>
            <p style="color: #64748B; font-size: 0.88rem;">
                Ensemble of decision trees trained on an 80/20 stratified split. Artifacts persisted using <code>joblib</code> with consistent preprocessing at prediction time.
            </p>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE B: DATASET EXPLORER
# -----------------------------------------------------------------------------
elif page == "🔍 Dataset Explorer":
    st.markdown("""
    <div class="section-header">
        <div class="section-title">🔍 Dataset Explorer & Statistical Profiling</div>
        <div class="section-desc">Interactive data table, missing value analysis, feature correlations, and passenger experience breakdowns.</div>
    </div>
    """, unsafe_allow_html=True)

    # Sub-metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{info['total_rows']:,}")
    with col2:
        st.metric("Total Columns", f"{info['total_cols']}")
    with col3:
        st.metric("Duplicate Rows", f"{info['duplicate_rows']}")
    with col4:
        st.metric("Missing Cells", f"{info['total_missing_cells']}")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    tab_data, tab_stats, tab_charts = st.tabs(["📋 Data Preview & Filters", "📊 Statistical Summary", "📈 Interactive Visualizations"])

    # 1. Preview Tab
    with tab_data:
        st.markdown("##### Filtered Records View")
        f_col1, f_col2, f_col3 = st.columns([1, 1, 1])

        with f_col1:
            sat_filter = st.selectbox("Filter Satisfaction", options=["All"] + list(df["satisfaction"].unique()))
        with f_col2:
            class_filter = st.selectbox("Filter Cabin Class", options=["All"] + list(df["Class"].unique()))
        with f_col3:
            num_rows = st.slider("Display Limit", min_value=10, max_value=200, value=25, step=5)

        filtered_df = df
        if sat_filter != "All":
            filtered_df = filtered_df[filtered_df["satisfaction"] == sat_filter]
        if class_filter != "All":
            filtered_df = filtered_df[filtered_df["Class"] == class_filter]

        st.dataframe(filtered_df.head(num_rows), use_container_width=True, height=380)
        st.caption(f"Showing first {num_rows} of {len(filtered_df):,} matching rows.")

        st.markdown("##### Column Schema & Integrity")
        schema_data = []
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            null_pct = round((null_count / len(df)) * 100, 2)
            n_unique = int(df[col].nunique())
            dtype_str = str(df[col].dtype)
            sample_val = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "None"
            
            schema_data.append({
                "Column": col,
                "Type": dtype_str,
                "Role": "Target" if col == info["target_column"] else ("Numeric Feature" if col in info["numeric_features"] else "Categorical Feature"),
                "Missing Count": null_count,
                "Missing %": f"{null_pct}%",
                "Unique Values": n_unique,
                "Sample": sample_val
            })
        st.dataframe(pd.DataFrame(schema_data), use_container_width=True)

    # 2. Statistical Summary
    with tab_stats:
        st.markdown("##### Numerical Features Summary")
        num_summary = df[info["numeric_features"]].describe().T
        st.dataframe(num_summary.style.format("{:.2f}"), use_container_width=True)

        st.markdown("##### Categorical Features Summary")
        cat_summary = []
        for c in info["categorical_features"]:
            val_counts = df[c].value_counts().to_dict()
            cat_summary.append({
                "Feature": c,
                "Unique Values": df[c].nunique(),
                "Most Common": df[c].mode()[0],
                "Distribution": ", ".join([f"{k}: {v:,}" for k, v in val_counts.items()])
            })
        st.dataframe(pd.DataFrame(cat_summary), use_container_width=True)

    # 3. Interactive Visualizations
    with tab_charts:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### Satisfaction by Travel Class")
            class_sat = pd.crosstab(df["Class"], df["satisfaction"], normalize="index") * 100
            fig_class = px.bar(
                class_sat.reset_index(),
                x="Class",
                y=list(class_sat.columns),
                title="Class-wise Satisfaction Rate (%)",
                barmode="group",
                color_discrete_sequence=["#EF4444", "#10B981"]
            )
            fig_class.update_layout(height=350, yaxis_title="Percentage (%)", template="plotly_white")
            st.plotly_chart(fig_class, use_container_width=True)

        with chart_col2:
            st.markdown("##### Satisfaction by Customer Type")
            cust_sat = pd.crosstab(df["Customer Type"], df["satisfaction"], normalize="index") * 100
            fig_cust = px.bar(
                cust_sat.reset_index(),
                x="Customer Type",
                y=list(cust_sat.columns),
                title="Loyalty-wise Satisfaction Rate (%)",
                barmode="group",
                color_discrete_sequence=["#EF4444", "#10B981"]
            )
            fig_cust.update_layout(height=350, yaxis_title="Percentage (%)", template="plotly_white")
            st.plotly_chart(fig_cust, use_container_width=True)

        st.markdown("##### Mean Ratings Profile (Scale: 0-5)")
        rating_cols = [c for c in info["numeric_features"] if "Delay" not in c and "Age" not in c and "Distance" not in c]
        if rating_cols:
            mean_ratings = df.groupby("satisfaction")[rating_cols].mean().T.reset_index()
            mean_ratings.columns = ["Service Feature", "Dissatisfied", "Satisfied"]
            
            fig_ratings = px.bar(
                mean_ratings,
                x="Service Feature",
                y=["Dissatisfied", "Satisfied"],
                barmode="group",
                title="Service Ratings: Satisfied vs Dissatisfied Passengers",
                color_discrete_sequence=["#F87171", "#4F46E5"]
            )
            fig_ratings.update_layout(
                xaxis_tickangle=-45,
                height=400,
                yaxis=dict(range=[0, 5], title="Average Rating (0 to 5)"),
                template="plotly_white"
            )
            st.plotly_chart(fig_ratings, use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE C: MODEL TRAINING & EVALUATION
# -----------------------------------------------------------------------------
elif page == "⚙️ Model Training":
    st.markdown("""
    <div class="section-header">
        <div class="section-title">⚙️ Random Forest Model Training & Evaluation</div>
        <div class="section-desc">Configure hyperparameters, fit the Random Forest pipeline with 80/20 split, and inspect detailed evaluation metrics.</div>
    </div>
    """, unsafe_allow_html=True)

    # Hyperparameter Form & Training Trigger
    with st.expander("🛠️ Hyperparameter & Training Configuration", expanded=True):
        col_hp1, col_hp2, col_hp3 = st.columns(3)

        with col_hp1:
            n_estimators = st.slider("Number of Trees (n_estimators)", min_value=20, max_value=200, value=100, step=10)
            max_depth = st.slider("Max Tree Depth", min_value=5, max_value=30, value=18)

        with col_hp2:
            min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=5)
            min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=5, value=2)

        with col_hp3:
            test_split = st.slider("Test Split Ratio", min_value=0.10, max_value=0.30, value=0.20, step=0.05)
            sample_options = {
                "Quick Training (30,000 samples ~1 sec)": 30000,
                "Standard Training (60,000 samples ~2 sec)": 60000,
                "Full Dataset (129,880 samples ~4 sec)": None
            }
            sample_choice = st.selectbox("Training Data Size", options=list(sample_options.keys()), index=0)
            sample_size = sample_options[sample_choice]

        train_btn = st.button("🚀 Train Random Forest Model", type="primary", use_container_width=True)

    # Handle training execution
    if train_btn:
        progress_bar = st.progress(0, text="Initializing preprocessing and data split...")
        status_box = st.empty()

        try:
            time.sleep(0.2)
            progress_bar.progress(25, text="Building scikit-learn ColumnTransformer pipeline...")

            hyperparams = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "random_state": 42
            }

            progress_bar.progress(50, text=f"Fitting Random Forest ({n_estimators} trees)...")
            res = run_full_training(
                df=df,
                target_col=info["target_column"],
                test_size=test_split,
                sample_size=sample_size,
                hyperparameters=hyperparams
            )

            progress_bar.progress(90, text="Calculating evaluation metrics and feature importances...")
            time.sleep(0.3)
            progress_bar.progress(100, text="Model trained and saved successfully!")
            
            st.success(f"✅ Random Forest model successfully trained in **{res['metadata']['training_duration_seconds']}s** and serialized to `models/random_forest_model.pkl`!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Training error: {e}")

    # Display Current Model Evaluation Metrics
    saved_artifacts = get_model()
    if saved_artifacts:
        pipeline, metadata = saved_artifacts
        eval_data = metadata.get("evaluation", {})

        st.markdown("### 📊 Model Evaluation Performance")

        # Top Metric Tiles
        m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
        with m_c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Accuracy</div>
                <div class="metric-value" style="color: #4F46E5;">{eval_data.get('accuracy', 0)*100:.2f}%</div>
                <div class="metric-sub">Test Set</div>
            </div>
            """, unsafe_allow_html=True)

        with m_c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Precision (Weighted)</div>
                <div class="metric-value" style="color: #06B6D4;">{eval_data.get('precision', 0)*100:.2f}%</div>
                <div class="metric-sub">Weighted Avg</div>
            </div>
            """, unsafe_allow_html=True)

        with m_c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Recall (Weighted)</div>
                <div class="metric-value" style="color: #10B981;">{eval_data.get('recall', 0)*100:.2f}%</div>
                <div class="metric-sub">True Positive Rate</div>
            </div>
            """, unsafe_allow_html=True)

        with m_c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">F1-Score</div>
                <div class="metric-value" style="color: #7C3AED;">{eval_data.get('f1_score', 0)*100:.2f}%</div>
                <div class="metric-sub">Harmonic Mean</div>
            </div>
            """, unsafe_allow_html=True)

        with m_c5:
            roc_val = eval_data.get("roc_auc")
            roc_str = f"{roc_val*100:.2f}%" if roc_val else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ROC-AUC</div>
                <div class="metric-value" style="color: #F59E0B;">{roc_str}</div>
                <div class="metric-sub">Class Separability</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Confusion Matrix and Classification Report
        col_cm, col_cr = st.columns([1, 1.2])

        with col_cm:
            st.markdown("#### 🟦 Confusion Matrix")
            cm = eval_data.get("confusion_matrix")
            classes = eval_data.get("classes", ["dissatisfied", "satisfied"])

            if cm and len(cm) == 2:
                labels_cap = [c.capitalize() for c in classes]
                # Annotations with counts and percentages
                total_samples = np.sum(cm)
                z_text = [[f"{val:,}<br>({(val/total_samples)*100:.1f}%)" for val in row] for row in cm]

                fig_cm = ff_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=[f"Predicted {c}" for c in labels_cap],
                    y=[f"Actual {c}" for c in labels_cap],
                    text=z_text,
                    texttemplate="%{text}",
                    textfont={"size": 15, "color": "white"},
                    colorscale="Purples",
                    showscale=False
                ))
                fig_cm.update_layout(
                    height=340,
                    margin=dict(t=20, b=20, l=40, r=20),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        with col_cr:
            st.markdown("#### 📋 Classification Report")
            cr_dict = eval_data.get("classification_report_dict", {})
            if cr_dict:
                report_rows = []
                for k, v in cr_dict.items():
                    if isinstance(v, dict):
                        report_rows.append({
                            "Class / Metric": k.capitalize(),
                            "Precision": f"{v.get('precision', 0):.4f}",
                            "Recall": f"{v.get('recall', 0):.4f}",
                            "F1-Score": f"{v.get('f1-score', 0):.4f}",
                            "Support": f"{int(v.get('support', 0)):,}"
                        })
                st.dataframe(pd.DataFrame(report_rows), use_container_width=True, height=230)
                st.caption(f"Evaluated on {metadata.get('test_rows', 0):,} holdout test samples with stratified 80/20 split.")

        st.markdown("---")

        # Feature Importance Chart
        st.markdown("#### 🌲 Random Forest Feature Importance")
        st.markdown("Gini impurity reduction across all ensemble decision trees:")
        
        top_features = eval_data.get("top_features", [])
        if top_features:
            df_feat = pd.DataFrame(top_features[:15])
            fig_imp = px.bar(
                df_feat,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=["#C7D2FE", "#4F46E5", "#312E81"],
                text=[f"{imp*100:.2f}%" for imp in df_feat["importance"]],
                labels={"importance": "Importance Weight", "feature": "Feature"}
            )
            fig_imp.update_layout(
                yaxis=dict(autorange="reversed"),
                height=440,
                coloraxis_showscale=False,
                template="plotly_white",
                margin=dict(t=20, b=20, l=10, r=10)
            )
            fig_imp.update_traces(textposition="outside")
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("No saved model found. Click the 'Train Random Forest Model' button above to train and save the model.")


# -----------------------------------------------------------------------------
# PAGE D: PREDICTION STUDIO
# -----------------------------------------------------------------------------
elif page == "🎯 Prediction Studio":
    st.markdown("""
    <div class="section-header">
        <div class="section-title">🎯 Real-time Passenger Satisfaction Predictor</div>
        <div class="section-desc">Input passenger flight characteristics to instantly predict satisfaction with confidence scores and probability breakdown.</div>
    </div>
    """, unsafe_allow_html=True)

    saved_artifacts = get_model()
    if not saved_artifacts:
        st.warning("⚠️ No trained model found in `models/`. Please visit the 'Model Training' page first to train the Random Forest model.")
        st.stop()

    pipeline, metadata = saved_artifacts

    # Quick Scenario Presets
    st.markdown("##### ⚡ Quick Test Presets")
    col_p1, col_p2, col_p3 = st.columns(3)

    preset_values = None

    with col_p1:
        if st.button("🌟 Happy Business Traveler", use_container_width=True):
            preset_values = {
                "Customer Type": "Loyal Customer",
                "Age": 42,
                "Type of Travel": "Business travel",
                "Class": "Business",
                "Flight Distance": 2100,
                "Seat comfort": 5,
                "Departure/Arrival time convenient": 4,
                "Food and drink": 4,
                "Gate location": 4,
                "Inflight wifi service": 5,
                "Inflight entertainment": 5,
                "Online support": 5,
                "Ease of Online booking": 5,
                "On-board service": 5,
                "Leg room service": 5,
                "Baggage handling": 5,
                "Checkin service": 5,
                "Cleanliness": 5,
                "Online boarding": 5,
                "Departure Delay in Minutes": 0,
                "Arrival Delay in Minutes": 0
            }

    with col_p2:
        if st.button("⚠️ Frustrated Economy Traveler", use_container_width=True):
            preset_values = {
                "Customer Type": "disloyal Customer",
                "Age": 28,
                "Type of Travel": "Personal Travel",
                "Class": "Eco",
                "Flight Distance": 650,
                "Seat comfort": 1,
                "Departure/Arrival time convenient": 1,
                "Food and drink": 1,
                "Gate location": 2,
                "Inflight wifi service": 1,
                "Inflight entertainment": 1,
                "Online support": 1,
                "Ease of Online booking": 1,
                "On-board service": 1,
                "Leg room service": 1,
                "Baggage handling": 2,
                "Checkin service": 1,
                "Cleanliness": 1,
                "Online boarding": 1,
                "Departure Delay in Minutes": 75,
                "Arrival Delay in Minutes": 85
            }

    with col_p3:
        if st.button("🎲 Random Sample from Dataset", use_container_width=True):
            sample_row = df.sample(1, random_state=int(time.time()) % 1000).drop(columns=[info["target_column"]], errors="ignore").iloc[0]
            preset_values = sample_row.to_dict()

    # Store preset values in session_state if triggered
    if preset_values:
        for k, v in preset_values.items():
            st.session_state[f"input_{k}"] = v

    # Organize Input Form into logical cards
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        # Section 1: Passenger Profile
        st.markdown("#### 👤 1. Passenger Profile")
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            customer_type_default = st.session_state.get("input_Customer Type", "Loyal Customer")
            cust_type = st.selectbox(
                "Customer Type",
                options=["Loyal Customer", "disloyal Customer"],
                index=0 if customer_type_default == "Loyal Customer" else 1
            )
            age_default = int(st.session_state.get("input_Age", 40))
            age = st.slider("Passenger Age", min_value=7, max_value=85, value=age_default)

        with col_f2:
            travel_type_default = st.session_state.get("input_Type of Travel", "Business travel")
            travel_type = st.selectbox(
                "Type of Travel",
                options=["Business travel", "Personal Travel"],
                index=0 if travel_type_default == "Business travel" else 1
            )
            flight_dist_default = int(st.session_state.get("input_Flight Distance", 1200))
            flight_dist = st.number_input("Flight Distance (miles)", min_value=50, max_value=5000, value=flight_dist_default, step=50)

        with col_f3:
            class_default = st.session_state.get("input_Class", "Business")
            class_options = ["Business", "Eco", "Eco Plus"]
            cabin_class = st.selectbox(
                "Cabin Class",
                options=class_options,
                index=class_options.index(class_default) if class_default in class_options else 0
            )

        st.markdown("---")

        # Section 2: Inflight Comfort & Entertainment (0 to 5)
        st.markdown("#### 💺 2. Inflight Experience Ratings (0: Poor → 5: Excellent)")
        col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)

        with col_r1:
            seat_comfort = st.selectbox("Seat Comfort", options=list(range(6)), index=int(st.session_state.get("input_Seat comfort", 4)))
        with col_r2:
            inflight_ent = st.selectbox("Inflight Entertainment", options=list(range(6)), index=int(st.session_state.get("input_Inflight entertainment", 4)))
        with col_r3:
            food_drink = st.selectbox("Food and Drink", options=list(range(6)), index=int(st.session_state.get("input_Food and drink", 3)))
        with col_r4:
            cleanliness = st.selectbox("Cleanliness", options=list(range(6)), index=int(st.session_state.get("input_Cleanliness", 4)))
        with col_r5:
            leg_room = st.selectbox("Leg Room Service", options=list(range(6)), index=int(st.session_state.get("input_Leg room service", 4)))

        # Section 3: Digital & Online Services (0 to 5)
        st.markdown("#### 💻 3. Digital & Online Experience (0: Poor → 5: Excellent)")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)

        with col_d1:
            wifi = st.selectbox("Inflight WiFi Service", options=list(range(6)), index=int(st.session_state.get("input_Inflight wifi service", 3)))
        with col_d2:
            online_book = st.selectbox("Ease of Online Booking", options=list(range(6)), index=int(st.session_state.get("input_Ease of Online booking", 3)))
        with col_d3:
            online_board = st.selectbox("Online Boarding", options=list(range(6)), index=int(st.session_state.get("input_Online boarding", 4)))
        with col_d4:
            support = st.selectbox("Online Support", options=list(range(6)), index=int(st.session_state.get("input_Online support", 4)))

        # Section 4: Airport & Ground Services (0 to 5)
        st.markdown("#### 🛫 4. Airport & Ground Services (0: Poor → 5: Excellent)")
        col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)

        with col_g1:
            time_conv = st.selectbox("Time Convenient", options=list(range(6)), index=int(st.session_state.get("input_Departure/Arrival time convenient", 3)))
        with col_g2:
            gate_loc = st.selectbox("Gate Location", options=list(range(6)), index=int(st.session_state.get("input_Gate location", 3)))
        with col_g3:
            onboard = st.selectbox("On-board Service", options=list(range(6)), index=int(st.session_state.get("input_On-board service", 4)))
        with col_g4:
            baggage = st.selectbox("Baggage Handling", options=list(range(6)), index=int(st.session_state.get("input_Baggage handling", 4)))
        with col_g5:
            checkin = st.selectbox("Check-in Service", options=list(range(6)), index=int(st.session_state.get("input_Checkin service", 4)))

        # Section 5: Delays
        st.markdown("#### ⏱️ 5. Flight Schedule & Delays (Minutes)")
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            dep_delay = st.number_input("Departure Delay in Minutes", min_value=0, max_value=1500, value=int(st.session_state.get("input_Departure Delay in Minutes", 0)), step=5)
        with col_dl2:
            arr_delay = st.number_input("Arrival Delay in Minutes", min_value=0, max_value=1500, value=int(st.session_state.get("input_Arrival Delay in Minutes", 0)), step=5)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        predict_submit = st.form_submit_button("🔮 Predict Customer Satisfaction", type="primary", use_container_width=True)

    # Prediction Outcome Render
    if predict_submit:
        # Build payload matching the dataset's features
        input_payload = {
            "Customer Type": cust_type,
            "Age": age,
            "Type of Travel": travel_type,
            "Class": cabin_class,
            "Flight Distance": flight_dist,
            "Seat comfort": seat_comfort,
            "Departure/Arrival time convenient": time_conv,
            "Food and drink": food_drink,
            "Gate location": gate_loc,
            "Inflight wifi service": wifi,
            "Inflight entertainment": inflight_ent,
            "Online support": support,
            "Ease of Online booking": online_book,
            "On-board service": onboard,
            "Leg room service": leg_room,
            "Baggage handling": baggage,
            "Checkin service": checkin,
            "Cleanliness": cleanliness,
            "Online boarding": online_board,
            "Departure Delay in Minutes": dep_delay,
            "Arrival Delay in Minutes": float(arr_delay)
        }

        try:
            pred_output = predict_sample(pipeline, input_payload, task_type=metadata.get("task_type", "classification"))
            predicted_class = str(pred_output["prediction"]).lower()
            confidence = pred_output.get("confidence_score", 0)
            probas = pred_output.get("probabilities", {})

            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

            is_satisfied = "satisfied" in predicted_class and "dis" not in predicted_class

            # Prediction Card
            if is_satisfied:
                st.markdown(f"""
                <div class="pred-card-satisfied">
                    <div style="font-size: 2.8rem; margin-bottom: 4px;">🎉</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #065F46;">
                        PREDICTED STATUS: SATISFIED
                    </div>
                    <div style="font-size: 1.15rem; color: #047857; margin-top: 6px; font-weight: 600;">
                        Model Confidence: {confidence:.1f}%
                    </div>
                    <div style="color: #065F46; font-size: 0.9rem; margin-top: 8px;">
                        Passenger profile indicates high loyalty and appreciation of onboard entertainment & comfort.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-card-dissatisfied">
                    <div style="font-size: 2.8rem; margin-bottom: 4px;">⚠️</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #991B1B;">
                        PREDICTED STATUS: DISSATISFIED
                    </div>
                    <div style="font-size: 1.15rem; color: #B91C1C; margin-top: 6px; font-weight: 600;">
                        Model Confidence: {confidence:.1f}%
                    </div>
                    <div style="color: #991B1B; font-size: 0.9rem; margin-top: 8px;">
                        Attention needed: Experience ratings or flight delays negatively impact customer satisfaction.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Probabilities Bar
            if probas:
                st.markdown("##### 📊 Class Probabilities Breakdown")
                df_prob = pd.DataFrame([
                    {"Class": k.capitalize(), "Probability": round(v * 100, 2)}
                    for k, v in probas.items()
                ])
                fig_prob = px.bar(
                    df_prob,
                    x="Probability",
                    y="Class",
                    orientation="h",
                    color="Class",
                    color_discrete_map={"Satisfied": "#10B981", "Dissatisfied": "#EF4444"},
                    text=[f"{p}%" for p in df_prob["Probability"]]
                )
                fig_prob.update_layout(
                    height=200,
                    xaxis=dict(range=[0, 100], title="Probability (%)"),
                    yaxis=dict(title=""),
                    template="plotly_white",
                    showlegend=False,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                fig_prob.update_traces(textposition="outside")
                st.plotly_chart(fig_prob, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
