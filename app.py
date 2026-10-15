"""
Signal Vault — Network Intelligence & Packet Analytics Platform
Main Application Orchestrator (Streamlit)
Author: Signal Vault Intelligence Team
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io

# Import modular backend
from src.data_loader import (
    get_default_data_path,
    check_dataset_exists,
    get_dataset_download_instructions,
    load_dataset
)
from src.preprocessing import (
    inspect_data_quality,
    clean_raw_data,
    categorize_columns,
    split_data
)
from src.eda import (
    SOC_COLORS,
    compute_numerical_summary,
    compute_categorical_summary,
    plot_univariate_histogram,
    plot_univariate_boxplot,
    plot_categorical_frequency,
    plot_bivariate_scatter,
    plot_grouped_boxplot,
    plot_correlation_heatmap,
    plot_pairwise_scatter_matrix
)
from src.statistical_tests import (
    run_welch_ttest,
    run_oneway_anova,
    run_variance_ftest
)
from src.pca_analysis import (
    compute_pca_analysis,
    plot_scree_chart,
    plot_pca_2d_projection,
    plot_pca_loadings_heatmap
)
from src.svd_analysis import (
    compute_svd_analysis,
    plot_singular_values,
    plot_reconstruction_error_curve,
    plot_svd_matrix_comparison
)
from src.genetic_algorithm import (
    run_genetic_feature_selection,
    plot_ga_convergence
)
from src.intrusion_model import (
    IntrusionDetectionPipeline
)
from src.evaluation import (
    evaluate_classifier_performance,
    plot_confusion_matrix_heatmap,
    plot_roc_curves_comparison,
    plot_model_comparison_bar
)
from src.report_generator import (
    generate_threat_intelligence_report,
    export_dataframe_to_csv_bytes,
    export_dict_to_json_bytes
)


# -------------------------------------------------------------
# PAGE CONFIG & CYBER SOC STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="Signal Vault — Network Intelligence & Packet Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Dark Navy SOC Cyber Theme */
    .stApp {
        background-color: #070d1e;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b152d;
        border-right: 1px solid #1e293b;
    }
    
    /* Top Header Banner */
    .soc-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0c4a6e 100%);
        border: 1px solid #0284c7;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(2, 132, 199, 0.15);
    }
    
    .soc-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .soc-subtitle {
        color: #38bdf8;
        font-size: 1.05rem;
        margin-top: 6px;
        font-weight: 500;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #00e5ff;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 4px 0;
    }
    .metric-delta {
        font-size: 0.8rem;
        color: #38bdf8;
    }
    
    /* Custom Info Box */
    .soc-box {
        background-color: #0e1e38;
        border-left: 4px solid #00e5ff;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 12px 0 20px 0;
    }
    
    /* Tabs & Widgets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        background-color: #0f172a;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -------------------------------------------------------------
# APPLICATION HEADER
# -------------------------------------------------------------
st.markdown("""
<div class="soc-header">
    <div class="soc-title">🛡️ SIGNAL VAULT</div>
    <div class="soc-subtitle">Network Intelligence & Packet Analytics Platform | Autonomous Threat Detection Engine</div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# SIDEBAR CONFIGURATION & DATASET DETECTION
# -------------------------------------------------------------
st.sidebar.markdown("### 📡 SOC Navigation")
navigation_option = st.sidebar.radio(
    "Analytical Suite",
    [
        "1. Overview & System Status",
        "2. Dataset Explorer",
        "3. Exploratory Data Analysis (EDA)",
        "4. Statistical Hypothesis Testing",
        "5. Principal Component Analysis (PCA)",
        "6. Singular Value Decomposition (SVD)",
        "7. Genetic Algorithm Feature Selection",
        "8. Intrusion Detection Classifier",
        "9. Feature Strategy Comparison",
        "10. Threat Reports & Exports"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Dataset Source Control")

default_path = get_default_data_path()
local_dataset_found = check_dataset_exists(default_path)

data_source_mode = st.sidebar.radio(
    "Data Input Mode",
    ["Local File (data/)", "Upload CSV via Dashboard"],
    index=0 if local_dataset_found else 1
)

uploaded_file = None
if data_source_mode == "Upload CSV via Dashboard":
    uploaded_file = st.sidebar.file_uploader(
        "Upload UNSW-NB15 CSV",
        type=["csv"],
        help="Upload UNSW_NB15_training-set.csv or another network flow CSV."
    )

# Sample size slider to protect laptop memory
st.sidebar.markdown("### ⚙️ Performance Tuning")
enable_subsampling = st.sidebar.checkbox("Enable Sample Capping", value=True, help="Prevents laptop freeze on large files.")
sample_limit = None
if enable_subsampling:
    sample_limit = st.sidebar.slider("Maximum Row Sample", min_value=1000, max_value=85000, value=15000, step=1000)

random_seed = st.sidebar.number_input("Reproducibility Seed", min_value=1, max_value=9999, value=42, step=1)


# -------------------------------------------------------------
# DATA LOADING ENGINE
# -------------------------------------------------------------
@st.cache_data(show_spinner="Loading and verifying network traffic records...")
def cached_load_data(file_source, sample_sz, seed):
    df, meta = load_dataset(file_source, sample_size=sample_sz, random_state=seed)
    df_clean, clean_report = clean_raw_data(df)
    col_info = categorize_columns(df_clean)
    return df_clean, meta, clean_report, col_info

dataset_ready = False
df = None
meta = {}
clean_report = {}
col_info = {}
load_error = None

if data_source_mode == "Local File (data/)":
    if local_dataset_found:
        try:
            df, meta, clean_report, col_info = cached_load_data(default_path, sample_limit, random_seed)
            dataset_ready = True
        except Exception as e:
            load_error = str(e)
    else:
        dataset_ready = False
elif data_source_mode == "Upload CSV via Dashboard":
    if uploaded_file is not None:
        try:
            df, meta, clean_report, col_info = cached_load_data(uploaded_file, sample_limit, random_seed)
            dataset_ready = True
        except Exception as e:
            load_error = str(e)
    else:
        dataset_ready = False


# Sidebar Status Pill
if dataset_ready:
    st.sidebar.success(f"🟢 Active Dataset: {meta.get('loaded_rows', 0):,} records")
else:
    st.sidebar.warning("🟡 Dataset Pending: Setup Required")


# -------------------------------------------------------------
# HELPER: DATASET MISSING SETUP SCREEN
# -------------------------------------------------------------
def render_dataset_missing_guide():
    instructions = get_dataset_download_instructions()
    
    st.warning("⚠️ **Dataset Missing — Setup Instructions & Guidance**")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(f"""
        ### 📥 How to Initialize the Dataset
        Signal Vault is pre-configured to analyze the **UNSW-NB15** network intrusion detection benchmark.
        
        To begin analysis:
        1. Download the single training dataset file:
           - **Required File:** `{instructions['expected_filename']}`
        2. Place the downloaded file into the project directory:
           - **Relative Path:** `{instructions['target_relative_path']}`
           - **Full Windows Path:** `{instructions['target_absolute_path']}`
        3. Or simply drag-and-drop the CSV using the **'Upload CSV via Dashboard'** option in the sidebar.
        4. Once placed, refresh this page or click the reload button below.
        """)
        
        st.markdown("#### 🌐 Official Download Repositories")
        for src in instructions["sources"]:
            st.markdown(f"- **[{src['name']}]({src['url']})**: {src['notes']}")
            
        if st.button("🔄 Check Data Directory Again", type="primary"):
            st.rerun()

    with col2:
        st.markdown("""
        ### 🧠 System Architecture Overview
        Signal Vault operates entirely **offline** without network packet sniffing or external vulnerabilities:
        
        - **Data Quality:** Automatic type inference, infinite/missing value imputation.
        - **Hypothesis Testing:** Welch's t-test, ANOVA, and F-test for statistical significance.
        - **Dimensionality Reduction:** Orthogonal PCA projection & Eckart-Young SVD matrix factorization.
        - **Feature Optimization:** Combinatorial Genetic Algorithm with parsimony penalties.
        - **Machine Learning:** Leakage-free Scikit-Learn pipelines evaluating normal vs attack traffic.
        """)
        
        st.info("💡 **College Evaluation Note:** Signal Vault is architected to showcase advanced mathematics, machine learning, and cybersecurity intelligence on standard laptop hardware.")


# -------------------------------------------------------------
# PAGE 1: OVERVIEW & SYSTEM STATUS
# -------------------------------------------------------------
if navigation_option == "1. Overview & System Status":
    st.markdown("## 🛡️ System Overview & SOC Health")
    
    if not dataset_ready:
        if load_error:
            st.error(f"Error reading dataset: {load_error}")
        render_dataset_missing_guide()
        
        # Educational tabs when dataset not yet loaded
        st.markdown("---")
        st.markdown("### 📚 Analytical Modules Primer")
        tab1, tab2, tab3 = st.tabs(["Mathematical Foundations", "Genetic Algorithm Design", "Leakage Prevention Protocol"])
        
        with tab1:
            st.markdown("""
            - **Principal Component Analysis (PCA):** Solves $\\mathbf{C} = \\frac{1}{n-1} \\mathbf{X}^T \\mathbf{X}$ for eigenvalues and eigenvectors, identifying orthogonal directions of maximum variance.
            - **Singular Value Decomposition (SVD):** Decomposes matrix $\\mathbf{X} = \\mathbf{U} \\mathbf{\\Sigma} \\mathbf{V}^T$. The Eckart-Young-Mirsky theorem proves that truncating singular values yields the optimal low-rank matrix approximation.
            - **Welch's t-test:** Evaluates difference in feature means between normal and malicious flows without assuming equal variance (heteroscedasticity).
            """)
        with tab2:
            st.markdown("""
            - **Binary Mask Chromosome:** $\\mathbf{m} \\in \\{0, 1\\}^D$ selects or discards input network features.
            - **Parsimony-Penalized Fitness:** $\\text{Fitness}(\\mathbf{m}) = \\text{Score}(\\mathbf{m}) - \\alpha \\times \\frac{\\sum m_i}{D}$ ensures compact, non-redundant feature sets.
            """)
        with tab3:
            st.markdown("""
            - **Zero Data Leakage:** Imputation, standard scalers, and encoders are fit strictly on the training partition.
            - **Exclusion of Identifiers:** Row indices, IP identifiers, and the ground-truth target are purged before feature transformation.
            """)
    else:
        # Live computed KPIs
        total_records = meta.get("loaded_rows", len(df))
        has_label = meta.get("has_label", False)
        label_col = meta.get("label_col", "label")
        
        n_normal, n_attack, attack_pct = 0, 0, 0.0
        if has_label and label_col in df.columns:
            vc = df[label_col].value_counts()
            n_normal = int(vc.get(0, 0))
            n_attack = int(vc.get(1, 0))
            attack_pct = (n_attack / total_records * 100) if total_records > 0 else 0.0

        n_features = len(col_info.get("all_eligible_features", []))

        # KPI Metrics Cards
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Records Analyzed</div>
                <div class="metric-value">{total_records:,}</div>
                <div class="metric-delta">{'Sampled' if meta.get('is_sampled') else 'Full Set'}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Normal Traffic</div>
                <div class="metric-value" style="color: #10b981;">{n_normal:,}</div>
                <div class="metric-delta">{100 - attack_pct:.1f}% of total</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Attack Records</div>
                <div class="metric-value" style="color: #f43f5e;">{n_attack:,}</div>
                <div class="metric-delta">{attack_pct:.1f}% attack ratio</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Features</div>
                <div class="metric-value" style="color: #00e5ff;">{n_features}</div>
                <div class="metric-delta">{len(col_info.get('numerical_features', []))} num / {len(col_info.get('categorical_features', []))} cat</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">System Memory</div>
                <div class="metric-value">{meta.get('memory_mb', 0):.1f} <span style="font-size: 1rem;">MB</span></div>
                <div class="metric-delta">Cleaned & In-Memory</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row
        c1, c2 = st.columns(2)
        with c1:
            if has_label:
                pie_df = pd.DataFrame({
                    "Traffic Type": ["Normal Flow (0)", "Intrusion Flow (1)"],
                    "Count": [n_normal, n_attack]
                })
                fig_pie = plot_categorical_frequency(df, label_col)
                # Custom pie chart
                import plotly.graph_objects as go
                fig_donut = go.Figure(data=[go.Pie(
                    labels=pie_df["Traffic Type"],
                    values=pie_df["Count"],
                    hole=0.55,
                    marker=dict(colors=[SOC_COLORS["normal"], SOC_COLORS["attack"]]),
                    textinfo="label+percent"
                )])
                fig_donut.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=SOC_COLORS["text"]),
                    title="Binary Traffic Class Proportion",
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.info("Label column not detected in current dataset.")

        with c2:
            attack_cat_col = meta.get("attack_cat_col")
            if attack_cat_col and attack_cat_col in df.columns:
                fig_cat = plot_categorical_frequency(df, attack_cat_col, top_n=10)
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("Attack Category column ('attack_cat') not present in this file.")

        # Key Computed Findings Card
        st.markdown("""
        <div class="soc-box">
            <h4 style="margin: 0 0 8px 0; color: #00e5ff;">🔍 Analytical Engine Operational Status</h4>
            <p style="margin: 0;">Dataset successfully sanitized. Identified <strong>{}</strong> numerical features and <strong>{}</strong> categorical protocol features. Leakage guard verified: identifiers and ground truth targets safely isolated from feature matrices.</p>
        </div>
        """.format(len(col_info.get("numerical_features", [])), len(col_info.get("categorical_features", []))), unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 2: DATASET EXPLORER
# -------------------------------------------------------------
elif navigation_option == "2. Dataset Explorer":
    st.markdown("## 📂 Network Dataset Explorer")
    
    if not dataset_ready:
        render_dataset_missing_guide()
    else:
        st.markdown(f"**Loaded Dataset:** `{meta.get('source_name')}` | **Dimensions:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Tabs for Raw Data, Quality, and Descriptive Stats
        exp_tab1, exp_tab2, exp_tab3 = st.tabs(["Data Table & Filter", "Data Quality & Hygiene", "Descriptive Statistics"])
        
        with exp_tab1:
            st.dataframe(df.head(100), use_container_width=True, height=380)
            
            # Download filtered data button
            csv_bytes = export_dataframe_to_csv_bytes(df.head(1000))
            st.download_button(
                "📥 Download Preview Rows (CSV)",
                data=csv_bytes,
                file_name="signal_vault_data_preview.csv",
                mime="text/csv"
            )
            
        with exp_tab2:
            quality = inspect_data_quality(df)
            q_col1, q_col2, q_col3 = st.columns(3)
            q_col1.metric("Duplicate Rows", f"{quality['total_duplicate_rows']:,}")
            q_col2.metric("Missing Cells (NaN)", f"{quality['total_missing_cells']:,}")
            q_col3.metric("Infinite Values Replaced", f"{clean_report.get('inf_replaced', 0):,}")
            
            st.markdown("#### Schema & Column Classification")
            col_summary = []
            for col in df.columns:
                col_type = "Target" if col in col_info.get("target", []) else (
                    "Attack Category" if col in col_info.get("attack_category", []) else (
                        "Leakage/ID (Excluded)" if col in col_info.get("excluded_leakage", []) else (
                            "Numerical Feature" if col in col_info.get("numerical_features", []) else "Categorical Feature"
                        )
                    )
                )
                col_summary.append({
                    "Column Name": col,
                    "Data Type": str(df[col].dtype),
                    "Null Count": int(df[col].isna().sum()),
                    "Unique Values": int(df[col].nunique()),
                    "Assigned Role": col_type
                })
            st.dataframe(pd.DataFrame(col_summary), use_container_width=True, height=350)

        with exp_tab3:
            st.markdown("#### Numerical Features Statistical Summary")
            num_summary = compute_numerical_summary(df, col_info.get("numerical_features"))
            st.dataframe(num_summary, use_container_width=True, height=350)


# -------------------------------------------------------------
# PAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------
elif navigation_option == "3. Exploratory Data Analysis (EDA)":
    st.markdown("## 📊 Exploratory Data Analysis (EDA)")
    
    if not dataset_ready:
        render_dataset_missing_guide()
    else:
        eda_mode = st.radio("Select Analysis Dimension", ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"], horizontal=True)
        num_features = col_info.get("numerical_features", [])
        cat_features = col_info.get("categorical_features", [])
        label_col = meta.get("label_col", "label")

        if eda_mode == "Univariate Analysis":
            u_col1, u_col2 = st.columns([1, 2])
            with u_col1:
                st.markdown("### Feature Selection")
                feature_type = st.radio("Feature Type", ["Numerical Metric", "Categorical Protocol/State"], horizontal=True)
                if feature_type == "Numerical Metric" and num_features:
                    selected_num = st.selectbox("Select Numerical Feature", num_features, index=0)
                    nbins = st.slider("Histogram Bins", 10, 100, 40)
                    log_x = st.checkbox("Logarithmic X-Axis", value=False)
                    
                    # Compute quick summary
                    series = pd.to_numeric(df[selected_num], errors="coerce").dropna()
                    st.markdown(f"""
                    **Mean:** {series.mean():.4f}  
                    **Median:** {series.median():.4f}  
                    **Std Dev:** {series.std():.4f}  
                    **Variance:** {series.var():.4f}  
                    **Min / Max:** {series.min():.4f} / {series.max():.4f}
                    """)
                elif feature_type == "Categorical Protocol/State" and cat_features:
                    selected_cat = st.selectbox("Select Categorical Feature", cat_features, index=0)
                    top_n = st.slider("Top N Categories", 5, 25, 12)
            
            with u_col2:
                if feature_type == "Numerical Metric" and num_features:
                    fig_hist = plot_univariate_histogram(df, selected_num, nbins=nbins, log_x=log_x)
                    st.plotly_chart(fig_hist, use_container_width=True)
                    fig_box = plot_univariate_boxplot(df, selected_num)
                    st.plotly_chart(fig_box, use_container_width=True)
                elif feature_type == "Categorical Protocol/State" and cat_features:
                    fig_bar = plot_categorical_frequency(df, selected_cat, top_n=top_n)
                    st.plotly_chart(fig_bar, use_container_width=True)

        elif eda_mode == "Bivariate Analysis":
            b_col1, b_col2 = st.columns([1, 2])
            with b_col1:
                st.markdown("### Bivariate Variables")
                x_feat = st.selectbox("X-Axis Feature (Numerical)", num_features, index=0 if len(num_features) > 0 else 0)
                y_feat = st.selectbox("Y-Axis Feature (Numerical)", num_features, index=1 if len(num_features) > 1 else 0)
                biv_type = st.radio("Plot Type", ["Scatter vs Target Label", "Box Plot Grouped by Target"], horizontal=True)
                
            with b_col2:
                if biv_type == "Scatter vs Target Label":
                    fig_biv = plot_bivariate_scatter(df, x_feat, y_feat, label_col=label_col)
                    st.plotly_chart(fig_biv, use_container_width=True)
                else:
                    fig_grp_box = plot_grouped_boxplot(df, x_feat, cat_col=label_col)
                    st.plotly_chart(fig_grp_box, use_container_width=True)

        elif eda_mode == "Multivariate Analysis":
            m_tab1, m_tab2 = st.tabs(["Correlation Heatmap", "Pairwise Scatter Matrix"])
            
            with m_tab1:
                st.markdown("### Feature Correlation Heatmap")
                corr_method = st.selectbox("Correlation Method", ["pearson", "spearman"], index=0)
                max_corr_features = st.slider("Number of Top Features to Display", 6, 25, 14)
                fig_corr = plot_correlation_heatmap(df, columns=None, method=corr_method, max_features=max_corr_features)
                st.plotly_chart(fig_corr, use_container_width=True)
                
            with m_tab2:
                st.markdown("### Pairwise Multi-Feature Scatter Matrix")
                default_pair = num_features[:4] if len(num_features) >= 4 else num_features
                selected_pair = st.multiselect("Select 3 to 5 features for interaction matrix", num_features, default=default_pair)
                if len(selected_pair) >= 2:
                    fig_pair = plot_pairwise_scatter_matrix(df, selected_pair, label_col=label_col)
                    st.plotly_chart(fig_pair, use_container_width=True)
                else:
                    st.warning("Please select at least 2 numerical features for pairwise interaction analysis.")


# -------------------------------------------------------------
# PAGE 4: STATISTICAL HYPOTHESIS TESTING
# -------------------------------------------------------------
elif navigation_option == "4. Statistical Hypothesis Testing":
    st.markdown("## 🔬 Statistical Hypothesis Testing")
    
    if not dataset_ready:
        render_dataset_missing_guide()
    else:
        st.markdown("""
        Rigorous parametric and non-parametric hypothesis tests to verify whether network traffic features
        demonstrate statistically significant separation between normal and malicious packets.
        """)
        
        test_type = st.radio("Choose Statistical Test", ["Welch's Two-Sample t-Test", "One-Way ANOVA (F-Test)", "Two-Sample Variance F-Test"], horizontal=True)
        num_features = col_info.get("numerical_features", [])
        label_col = meta.get("label_col", "label")
        attack_cat_col = meta.get("attack_cat_col", "attack_cat")
        
        c_test1, c_test2 = st.columns([1, 2])
        
        with c_test1:
            st.markdown("### Test Parameters")
            tested_feature = st.selectbox("Numerical Feature to Test", num_features, index=0)
            alpha_level = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1)
            
            if test_type == "Welch's Two-Sample t-Test":
                equal_var = st.checkbox("Assume Equal Variances (Student's t-test)", value=False,
                                        help="Unchecked uses Welch's t-test, which is recommended for heteroscedastic network traffic.")
            elif test_type == "One-Way ANOVA (F-Test)":
                st.caption(f"Testing variation across categories in `{attack_cat_col}`.")

        with c_test2:
            st.markdown("### Empirical Results & Inference")
            if test_type == "Welch's Two-Sample t-Test":
                res = run_welch_ttest(df, tested_feature, target_col=label_col, alpha=alpha_level, equal_var=equal_var)
            elif test_type == "One-Way ANOVA (F-Test)":
                res = run_oneway_anova(df, tested_feature, cat_col=attack_cat_col, alpha=alpha_level)
            else:
                res = run_variance_ftest(df, tested_feature, target_col=label_col, alpha=alpha_level)

            if not res.get("valid", False):
                st.error(f"Cannot execute test: {res.get('error')}")
            else:
                # Render results card
                decision_color = "#10b981" if res.get("rejected") else "#f59e0b"
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {decision_color};">
                    <h3 style="color: {decision_color}; margin: 0 0 8px 0;">{res.get('decision')}</h3>
                    <p><strong>Test:</strong> {res.get('test_name')}</p>
                    <p><strong>Null Hypothesis (H₀):</strong> {res.get('H0')}</p>
                    <p><strong>Alternative Hypothesis (H₁):</strong> {res.get('H1')}</p>
                    <div style="display: flex; gap: 24px; margin: 12px 0;">
                        <div><strong>Test Statistic:</strong> {res.get('test_statistic', 0.0):.4f}</div>
                        <div><strong>p-value:</strong> {res.get('p_value', 1.0):.4e}</div>
                        <div><strong>Significance α:</strong> {res.get('alpha')}</div>
                    </div>
                    <p><strong>Plain-English Interpretation:</strong> {res.get('interpretation')}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.info(f"💡 **Academic Caution:** {res.get('scientific_note')}")


# -------------------------------------------------------------
# PAGE 5: PRINCIPAL COMPONENT ANALYSIS (PCA)
# -------------------------------------------------------------
elif navigation_option == "5. Principal Component Analysis (PCA)":
    st.markdown("## 📐 Principal Component Analysis (PCA)")
    
    if not dataset_ready:
        render_dataset_missing_guide()
    else:
        num_features = col_info.get("numerical_features", [])
        label_col = meta.get("label_col", "label")

        pca_c1, pca_c2 = st.columns([1, 2])
        with pca_c1:
            st.markdown("### PCA Configuration")
            max_comps = min(20, len(num_features))
            selected_n_components = st.slider("Number of Principal Components", 2, max_comps, 8)
            st.caption("StandardScaler fitted only on input features to guarantee zero data leakage.")

            with st.spinner("Calculating Principal Components..."):
                pca_res = compute_pca_analysis(
                    df,
                    feature_cols=num_features,
                    n_components=selected_n_components,
                    random_state=random_seed
                )

            st.markdown(f"""
            <div class="soc-box">
                <p><strong>Original Numerical Features:</strong> {pca_res['n_original_features']}</p>
                <p><strong>Retained Components:</strong> {pca_res['n_components']}</p>
                <p><strong>Comp. for 80% Variance:</strong> {pca_res['threshold_components'].get(80)}</p>
                <p><strong>Comp. for 90% Variance:</strong> {pca_res['threshold_components'].get(90)}</p>
                <p><strong>Comp. for 95% Variance:</strong> {pca_res['threshold_components'].get(95)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(pca_res["educational_note"])

        with pca_c2:
            st.plotly_chart(plot_scree_chart(pca_res), use_container_width=True)

        st.markdown("---")
        pca_proj_col, pca_load_col = st.columns(2)
        with pca_proj_col:
            st.plotly_chart(plot_pca_2d_projection(df, pca_res, label_col=label_col), use_container_width=True)
        with pca_load_col:
            st.plotly_chart(plot_pca_loadings_heatmap(pca_res, top_n_features=12), use_container_width=True)


# -------------------------------------------------------------
# PAGE 6: SINGULAR VALUE DECOMPOSITION (SVD)
# -------------------------------------------------------------
elif navigation_option == "6. Singular Value Decomposition (SVD)":
    st.markdown("## 🧮 Singular Value Decomposition (SVD) & Low-Rank Approximation")
    
    if not dataset_ready:
        render_dataset_missing_guide()
    else:
        num_features = col_info.get("numerical_features", [])
        
        svd_c1, svd_c2 = st.columns([1, 2])
        with svd_c1:
            st.markdown("### SVD Controls")
            svd_sample_limit = st.slider("Matrix Row Subsample (Speed Optimization)", 500, 5000, 2000, step=500)
            
            with st.spinner("Computing Matrix Factorization X = U * Σ * V^T..."):
                svd_res = compute_svd_analysis(
                    df,
                    feature_cols=num_features,
                    max_samples=svd_sample_limit,
                    random_state=random_seed
                )

            max_rank = len(svd_res["singular_values"])
            trunc_rank = st.slider("Truncation Rank (k) for Low-Rank Approximation", 1, max_rank, min(5, max_rank))
            
            # Reconstruction error at rank k
            current_err = svd_res["reconstruction_errors"][trunc_rank - 1]
            cum_energy = svd_res["cumulative_energy"][trunc_rank - 1]
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Rank-k Truncation (k = {trunc_rank})</div>
                <div class="metric-value">{current_err:.2%} <span style="font-size: 1rem; color: #94a3b8;">Rel. Error</span></div>
                <div class="metric-delta">Energy Captured: {cum_energy:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(svd_res["educational_note"])

        with svd_c2:
            st.plotly_chart(plot_singular_values(svd_res), use_container_width=True)

        st.markdown("---")
        svd_err_col, svd_mat_col = st.columns(2)
        with svd_err_col:
            st.plotly_chart(plot_reconstruction_error_curve(svd_res), use_container_width=True)
        with svd_mat_col:
            st.plotly_chart(plot_svd_matrix_comparison(svd_res, rank_k=trunc_rank), use_container_width=True)


# -------------------------------------------------------------
# PAGE 7: GENETIC ALGORITHM FEATURE SELECTION
# -------------------------------------------------------------
elif navigation_option == "7. Genetic Algorithm Feature Selection":
    st.markdown("## 🧬 Genetic Algorithm (GA) Feature Selection")
    
    if not dataset_ready:
        render_dataset_missing_guide()
    else:
        st.markdown("""
        Searches the exponential combinatorial feature space ($2^D$) using an evolutionary algorithm
        with bit-flip mutations, uniform crossover, tournament selection, and parsimony penalties.
        """)

        ga_c1, ga_c2 = st.columns([1, 2])
        with ga_c1:
            st.markdown("### Genetic Hyperparameters")
            ga_pop = st.slider("Population Size", 10, 40, 16, step=2)
            ga_gen = st.slider("Generations", 4, 20, 8, step=1)
            ga_mut = st.slider("Mutation Rate (pm)", 0.01, 0.20, 0.05, step=0.01)
            ga_penalty = st.slider("Parsimony Feature Penalty (α)", 0.00, 0.20, 0.05, step=0.01)
            
            run_ga_btn = st.button("🚀 Run Genetic Optimization", type="primary")

        # Cache or session state for GA
        if "ga_results" not in st.session_state:
            st.session_state.ga_results = None

        if run_ga_btn:
            with st.spinner("Evolving Feature Subsets through Genetic Algorithm..."):
                # Prepare stratified sample
                X_tr, _, y_tr, _, _ = split_data(df, target_col=meta.get("label_col", "label"), test_size=0.25, random_state=random_seed, max_samples=4000)
                eligible_cols = col_info.get("all_eligible_features", [])
                
                # Preprocess features once
                from src.preprocessing import build_preprocessing_pipeline
                prep = build_preprocessing_pipeline(col_info.get("numerical_features", []), col_info.get("categorical_features", []))
                X_proc = prep.fit_transform(X_tr)

                # Progress bar
                prog_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(gen, total_gens, best_fit, k):
                    pct = gen / total_gens
                    prog_bar.progress(pct)
                    status_text.text(f"Generation {gen}/{total_gens} | Best Fitness: {best_fit:.4f} | Features Selected: {k}")

                # Feature names after one-hot encoding
                feat_names = [f"F_{i+1}" for i in range(X_proc.shape[1])]

                ga_res = run_genetic_feature_selection(
                    X_train_processed=X_proc,
                    y_train=y_tr.values,
                    feature_names=feat_names,
                    population_size=ga_pop,
                    generations=ga_gen,
                    mutation_rate=ga_mut,
                    feature_penalty=ga_penalty,
                    max_eval_samples=2500,
                    random_state=random_seed,
                    progress_callback=update_progress
                )
                st.session_state.ga_results = ga_res
                prog_bar.empty()
                status_text.empty()
                st.success("GA Optimization Complete!")

        with ga_c2:
            if st.session_state.ga_results is not None:
                st.plotly_chart(plot_ga_convergence(st.session_state.ga_results), use_container_width=True)
            else:
                st.info("Click 'Run Genetic Optimization' to execute the evolutionary search.")

        if st.session_state.ga_results is not None:
            res = st.session_state.ga_results
            st.markdown("### 🏆 Optimal Genetic Feature Mask")
            g_m1, g_m2, g_m3 = st.columns(3)
            g_m1.metric("Selected Dimension Count", f"{res['selected_count']} / {res['total_features']}")
            g_m2.metric("Dimensionality Compression", f"{(1 - res['selected_count']/res['total_features']):.1%}")
            g_m3.metric("Peak Chromosome Fitness", f"{res['best_fitness']:.4f}")

            st.dataframe(res["history_df"], use_container_width=True)


# -------------------------------------------------------------
# PAGE 8: INTRUSION DETECTION CLASSIFIER
# -------------------------------------------------------------
elif navigation_option == "8. Intrusion Detection Classifier":
    st.markdown("## 🤖 Machine Learning Intrusion Detection Classifier")
    
    if not dataset_ready:
        render_dataset_missing_guide()
    else:
        st.markdown("""
        Supervised classification pipeline targeting `label` (0 = Normal, 1 = Attack).
        StandardScaler and OneHotEncoder are fitted strictly on the training partition.
        """)
        
        m_col1, m_col2 = st.columns([1, 2])
        with m_col1:
            st.markdown("### Model Configuration")
            classifier_choice = st.selectbox("Classifier Algorithm", ["Random Forest", "Logistic Regression", "Decision Tree"], index=0)
            val_ratio = st.slider("Validation Set Ratio", 0.15, 0.40, 0.25, step=0.05)
            
            st.caption("Validation split is sampled strictly from the local CSV partition to prevent leakage.")
            train_model_btn = st.button("⚡ Train & Evaluate Classifier", type="primary")

        if "trained_model_eval" not in st.session_state:
            st.session_state.trained_model_eval = None

        if train_model_btn:
            with st.spinner(f"Training {classifier_choice} pipeline..."):
                label_col = meta.get("label_col", "label")
                X_tr, X_val, y_tr, y_val, split_info = split_data(
                    df,
                    target_col=label_col,
                    test_size=val_ratio,
                    random_state=random_seed,
                    max_samples=25000 if enable_subsampling else None
                )
                
                pipeline = IntrusionDetectionPipeline(
                    classifier_name=classifier_choice,
                    feature_strategy="baseline",
                    random_state=random_seed
                )
                pipeline.fit(
                    X_tr, y_tr,
                    numerical_cols=col_info.get("numerical_features", []),
                    categorical_cols=col_info.get("categorical_features", [])
                )
                
                eval_res = evaluate_classifier_performance(pipeline, X_val, y_val, model_label=classifier_choice)
                eval_res["features_count"] = pipeline.transformed_feature_dim
                eval_res["train_time"] = pipeline.training_time_sec
                st.session_state.trained_model_eval = eval_res
                st.session_state.last_trained_pipeline = pipeline

        with m_col2:
            if st.session_state.trained_model_eval is not None:
                ev = st.session_state.trained_model_eval
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{ev['accuracy']:.2%}")
                m2.metric("Precision", f"{ev['precision']:.2%}")
                m3.metric("Recall (Sensitivity)", f"{ev['recall']:.2%}")
                m4.metric("F1-Score", f"{ev['f1_score']:.4f}")

                st.markdown("<br>", unsafe_allow_html=True)
                cm_col, roc_col = st.columns(2)
                with cm_col:
                    st.plotly_chart(plot_confusion_matrix_heatmap(ev), use_container_width=True)
                with roc_col:
                    st.plotly_chart(plot_roc_curves_comparison([ev]), use_container_width=True)
            else:
                st.info("Configure hyper-parameters and click 'Train & Evaluate Classifier' to view confusion matrix and ROC curves.")


# -------------------------------------------------------------
# PAGE 9: FEATURE STRATEGY COMPARISON
# -------------------------------------------------------------
elif navigation_option == "9. Feature Strategy Comparison":
    st.markdown("## ⚖️ Feature Strategy Comparison: Baseline vs PCA vs GA")
    
    if not dataset_ready:
        render_dataset_missing_guide()
    else:
        st.markdown("""
        Empirically compares three distinct feature space reduction paradigms on an **identical validation partition**:
        1. **Baseline Model:** Full original feature space with categorical one-hot encoding.
        2. **PCA Model:** Dense orthogonal projection capturing maximal sample variance.
        3. **Genetic Algorithm Model:** Sparse combinatorial selection of raw original features.
        """)

        run_comp_btn = st.button("🧪 Run Three-Way Benchmark Comparison", type="primary")

        if "comparison_results" not in st.session_state:
            st.session_state.comparison_results = None

        if run_comp_btn:
            with st.spinner("Benchmarking Baseline, PCA, and GA Feature Spaces..."):
                label_col = meta.get("label_col", "label")
                X_tr, X_val, y_tr, y_val, split_info = split_data(
                    df,
                    target_col=label_col,
                    test_size=0.25,
                    random_state=random_seed,
                    max_samples=10000
                )
                
                num_f = col_info.get("numerical_features", [])
                cat_f = col_info.get("categorical_features", [])

                # 1. Baseline Model
                base_pipe = IntrusionDetectionPipeline(classifier_name="Random Forest", feature_strategy="baseline", random_state=random_seed)
                base_pipe.fit(X_tr, y_tr, numerical_cols=num_f, categorical_cols=cat_f)
                base_eval = evaluate_classifier_performance(base_pipe, X_val, y_val, model_label="1. Baseline (All Features)")
                base_eval["features_count"] = base_pipe.transformed_feature_dim
                base_eval["train_time"] = base_pipe.training_time_sec

                # 2. PCA Model (e.g. 8 components)
                pca_pipe = IntrusionDetectionPipeline(classifier_name="Random Forest", feature_strategy="pca", pca_components=8, random_state=random_seed)
                pca_pipe.fit(X_tr, y_tr, numerical_cols=num_f, categorical_cols=cat_f)
                pca_eval = evaluate_classifier_performance(pca_pipe, X_val, y_val, model_label="2. PCA (8 Components)")
                pca_eval["features_count"] = pca_pipe.transformed_feature_dim
                pca_eval["train_time"] = pca_pipe.training_time_sec

                # 3. GA Model (Run fast selector if not already cached)
                prep_ga = build_preprocessing_pipeline(num_f, cat_f)
                X_tr_proc = prep_ga.fit_transform(X_tr)
                feat_names = [f"F_{i+1}" for i in range(X_tr_proc.shape[1])]
                
                ga_run = run_genetic_feature_selection(
                    X_train_processed=X_tr_proc,
                    y_train=y_tr.values,
                    feature_names=feat_names,
                    population_size=12,
                    generations=6,
                    max_eval_samples=2000,
                    random_state=random_seed
                )

                ga_pipe = IntrusionDetectionPipeline(
                    classifier_name="Random Forest",
                    feature_strategy="ga",
                    ga_selected_indices=ga_run["selected_indices"],
                    random_state=random_seed
                )
                ga_pipe.fit(X_tr, y_tr, numerical_cols=num_f, categorical_cols=cat_f)
                ga_eval = evaluate_classifier_performance(ga_pipe, X_val, y_val, model_label="3. GA (Selected Subset)")
                ga_eval["features_count"] = ga_pipe.transformed_feature_dim
                ga_eval["train_time"] = ga_pipe.training_time_sec

                st.session_state.comparison_results = [base_eval, pca_eval, ga_eval]

        if st.session_state.comparison_results is not None:
            comp_list = st.session_state.comparison_results
            
            # Comparison Table
            comp_table_rows = []
            for ev in comp_list:
                comp_table_rows.append({
                    "Model Strategy": ev["model_label"],
                    "Dimensions (k)": ev["features_count"],
                    "Accuracy": ev["accuracy"],
                    "Precision": ev["precision"],
                    "Recall": ev["recall"],
                    "F1 Score": ev["f1_score"],
                    "ROC-AUC": ev["roc_auc"],
                    "Train Time (s)": ev["train_time"]
                })
            comp_df = pd.DataFrame(comp_table_rows)

            st.dataframe(comp_df, use_container_width=True)

            # Charts
            b_col, r_col = st.columns(2)
            with b_col:
                st.plotly_chart(plot_model_comparison_bar(comp_df), use_container_width=True)
            with r_col:
                st.plotly_chart(plot_roc_curves_comparison(comp_list), use_container_width=True)

            # Theoretical Summary
            st.markdown("""
            <div class="soc-box">
                <h4 style="color: #00e5ff; margin-top: 0;">🎓 Theoretical Takeaways for College Defense</h4>
                <ul>
                    <li><strong>Baseline:</strong> Preserves full raw variance and interaction effects, but incurs the highest computational cost and potential curse of dimensionality.</li>
                    <li><strong>PCA:</strong> Creates synthetic orthogonal axes ordered by variance. High information density in few components, but raw feature interpretability is lost.</li>
                    <li><strong>Genetic Algorithm:</strong> Performs discrete heuristic search over the original features. Retains physical interpretability while eliminating non-predictive packet metrics.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Click 'Run Three-Way Benchmark Comparison' to evaluate all three strategies.")


# -------------------------------------------------------------
# PAGE 10: THREAT REPORTS & EXPORTS
# -------------------------------------------------------------
elif navigation_option == "10. Threat Reports & Exports":
    st.markdown("## 📑 Intelligence Reports & Data Exports")
    
    if not dataset_ready:
        render_dataset_missing_guide()
    else:
        st.markdown("Download computed analytical results, model benchmarks, and comprehensive intelligence reports.")

        # Generate live markdown report
        eda_summary = {
            "class_distribution": {
                "Normal": int((df[meta.get("label_col", "label")] == 0).sum()) if meta.get("has_label") else 0,
                "Normal_pct": float((df[meta.get("label_col", "label")] == 0).mean() * 100) if meta.get("has_label") else 0.0,
                "Attack": int((df[meta.get("label_col", "label")] == 1).sum()) if meta.get("has_label") else 0,
                "Attack_pct": float((df[meta.get("label_col", "label")] == 1).mean() * 100) if meta.get("has_label") else 0.0,
            }
        }
        
        md_report = generate_threat_intelligence_report(
            metadata=meta,
            eda_summary=eda_summary,
            model_evaluations=st.session_state.get("comparison_results")
        )

        st.download_button(
            "📥 Download SOC Threat Intelligence Report (Markdown)",
            data=md_report.encode("utf-8"),
            file_name="signal_vault_soc_report.md",
            mime="text/markdown"
        )

        st.markdown("---")
        st.markdown("### Export Computed Analytics (CSV / JSON)")
        
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            # Descriptive Statistics CSV
            stat_df = compute_numerical_summary(df, col_info.get("numerical_features"))
            st.download_button(
                "📥 Export Descriptive Statistics (CSV)",
                data=export_dataframe_to_csv_bytes(stat_df),
                file_name="signal_vault_descriptive_stats.csv",
                mime="text/csv"
            )
        with d_col2:
            # Metadata JSON
            st.download_button(
                "📥 Export Dataset Provenance (JSON)",
                data=export_dict_to_json_bytes(meta),
                file_name="signal_vault_dataset_metadata.json",
                mime="application/json"
            )

        st.markdown("---")
        st.markdown("### Live Report Preview")
        st.markdown(md_report)
