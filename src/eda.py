"""
Signal Vault — Network Intelligence & Packet Analytics Platform
Module: eda.py
Description: Exploratory Data Analysis (Univariate, Bivariate, Multivariate)
with professional SOC dark-mode Plotly visualizations.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# SOC Cyber Dark Palette
SOC_COLORS = {
    "bg": "#0a192f",
    "card_bg": "#112240",
    "text": "#e2e8f0",
    "grid": "#233554",
    "normal": "#10b981",    # Emerald Green
    "attack": "#f43f5e",    # Crimson Alert
    "accent": "#00e5ff",    # Neon Cyan
    "secondary": "#a855f7", # Violet
    "warning": "#f59e0b"    # Amber
}

SOC_PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10, 25, 47, 0.7)",
    font=dict(color=SOC_COLORS["text"], family="Inter, system-ui, sans-serif"),
    xaxis=dict(gridcolor=SOC_COLORS["grid"], zerolinecolor=SOC_COLORS["grid"]),
    yaxis=dict(gridcolor=SOC_COLORS["grid"], zerolinecolor=SOC_COLORS["grid"]),
    margin=dict(l=40, r=40, t=50, b=40)
)


def compute_numerical_summary(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Computes comprehensive descriptive statistics for numerical columns:
    Count, Mean, Std, Variance, Median, Min, 25%, 50%, 75%, Max, Skew, Kurtosis.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        return pd.DataFrame()

    stats_list = []
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        stats_list.append({
            "Feature": col,
            "Count": len(series),
            "Mean": series.mean(),
            "Std Dev": series.std(),
            "Variance": series.var(),
            "Median": series.median(),
            "Min": series.min(),
            "Q25 (25%)": series.quantile(0.25),
            "Q75 (75%)": series.quantile(0.75),
            "Max": series.max(),
            "Skewness": series.skew(),
            "Kurtosis": series.kurtosis()
        })

    summary_df = pd.DataFrame(stats_list)
    return summary_df


def compute_categorical_summary(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Computes frequency and cardinality metrics for categorical columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    if not columns:
        return pd.DataFrame()

    stats_list = []
    for col in columns:
        series = df[col].astype(str)
        vc = series.value_counts()
        stats_list.append({
            "Feature": col,
            "Total Entries": len(series),
            "Unique Values": series.nunique(),
            "Most Frequent": vc.index[0] if not vc.empty else "N/A",
            "Top Count": int(vc.iloc[0]) if not vc.empty else 0,
            "Top Ratio (%)": round(float(vc.iloc[0] / len(series) * 100), 2) if not vc.empty else 0.0
        })

    return pd.DataFrame(stats_list)


# -------------------------------------------------------------
# UNIVARIATE VISUALIZATIONS
# -------------------------------------------------------------

def plot_univariate_histogram(
    df: pd.DataFrame,
    column: str,
    nbins: int = 40,
    log_x: bool = False
) -> go.Figure:
    """Creates an interactive histogram for a numerical feature."""
    clean_series = pd.to_numeric(df[column], errors="coerce").dropna()
    
    fig = px.histogram(
        clean_series,
        x=column,
        nbins=nbins,
        log_x=log_x,
        title=f"Distribution of '{column}'",
        color_discrete_sequence=[SOC_COLORS["accent"]]
    )
    fig.update_layout(**SOC_PLOT_LAYOUT)
    fig.update_traces(marker_line_color=SOC_COLORS["bg"], marker_line_width=1)
    return fig


def plot_univariate_boxplot(df: pd.DataFrame, column: str) -> go.Figure:
    """Creates a standalone box plot for detecting outliers and dispersion."""
    clean_series = pd.to_numeric(df[column], errors="coerce").dropna()
    
    fig = px.box(
        clean_series,
        y=column,
        title=f"Box Plot & Outlier Spread: '{column}'",
        color_discrete_sequence=[SOC_COLORS["secondary"]]
    )
    fig.update_layout(**SOC_PLOT_LAYOUT)
    return fig


def plot_categorical_frequency(df: pd.DataFrame, column: str, top_n: int = 15) -> go.Figure:
    """Plots top N categories in a bar chart."""
    vc = df[column].astype(str).value_counts().head(top_n).reset_index()
    vc.columns = [column, "Count"]

    fig = px.bar(
        vc,
        x=column,
        y="Count",
        title=f"Category Frequency: '{column}' (Top {top_n})",
        color="Count",
        color_continuous_scale=[[0, "#1e3a8a"], [1, SOC_COLORS["accent"]]]
    )
    fig.update_layout(**SOC_PLOT_LAYOUT)
    return fig


# -------------------------------------------------------------
# BIVARIATE VISUALIZATIONS
# -------------------------------------------------------------

def plot_bivariate_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: Optional[str] = "label",
    sample_size: int = 3000
) -> go.Figure:
    """
    Scatter plot between two numerical features, stratified or colored by class label.
    """
    plot_df = df[[x_col, y_col] + ([label_col] if label_col and label_col in df.columns else [])].dropna()
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(n=sample_size, random_state=42)

    color_discrete_map = None
    if label_col and label_col in plot_df.columns:
        plot_df["Traffic Class"] = plot_df[label_col].map({0: "Normal (0)", 1: "Attack (1)"}).fillna(plot_df[label_col].astype(str))
        color_arg = "Traffic Class"
        color_discrete_map = {"Normal (0)": SOC_COLORS["normal"], "Attack (1)": SOC_COLORS["attack"]}
    else:
        color_arg = None

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_arg,
        color_discrete_map=color_discrete_map,
        title=f"Bivariate Scatter: {x_col} vs {y_col}",
        opacity=0.7,
        render_mode="webgl"
    )
    fig.update_layout(**SOC_PLOT_LAYOUT)
    return fig


def plot_grouped_boxplot(
    df: pd.DataFrame,
    num_col: str,
    cat_col: str = "label",
    log_y: bool = False
) -> go.Figure:
    """
    Box plot of a numerical metric grouped by categorical or target label (Normal vs Attack).
    """
    plot_df = df[[num_col, cat_col]].dropna().copy()
    if cat_col == "label":
        plot_df["Traffic Class"] = plot_df[cat_col].map({0: "Normal", 1: "Attack"}).fillna(plot_df[cat_col].astype(str))
        group_col = "Traffic Class"
        color_map = {"Normal": SOC_COLORS["normal"], "Attack": SOC_COLORS["attack"]}
    else:
        group_col = cat_col
        color_map = None

    fig = px.box(
        plot_df,
        x=group_col,
        y=num_col,
        color=group_col,
        color_discrete_map=color_map,
        log_y=log_y,
        title=f"Feature Distribution Grouped by {cat_col}: {num_col}"
    )
    fig.update_layout(**SOC_PLOT_LAYOUT)
    return fig


# -------------------------------------------------------------
# MULTIVARIATE VISUALIZATIONS
# -------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
    max_features: int = 16
) -> go.Figure:
    """
    Computes and plots correlation matrix heatmap for numerical features.
    """
    if columns is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Drop ID / labels if present to focus on network features
        safe_cols = [c for c in num_cols if c.lower() not in ("id", "unnamed: 0", "label")]
        # Pick top variance features if too many
        if len(safe_cols) > max_features:
            variances = df[safe_cols].var().sort_values(ascending=False)
            columns = variances.head(max_features).index.tolist()
        else:
            columns = safe_cols

    corr = df[columns].corr(method=method)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"Multivariate Correlation Heatmap ({method.capitalize()})"
    )
    fig.update_layout(**SOC_PLOT_LAYOUT)
    return fig


def plot_pairwise_scatter_matrix(
    df: pd.DataFrame,
    features: List[str],
    label_col: Optional[str] = "label",
    sample_size: int = 1500
) -> go.Figure:
    """
    Plots a pairwise scatter matrix for 3-5 selected features.
    """
    sub_df = df[features + ([label_col] if label_col and label_col in df.columns else [])].dropna()
    if len(sub_df) > sample_size:
        sub_df = sub_df.sample(n=sample_size, random_state=42)

    color_arg = None
    color_map = None
    if label_col and label_col in sub_df.columns:
        sub_df["Class"] = sub_df[label_col].map({0: "Normal", 1: "Attack"}).fillna(sub_df[label_col].astype(str))
        color_arg = "Class"
        color_map = {"Normal": SOC_COLORS["normal"], "Attack": SOC_COLORS["attack"]}

    fig = px.scatter_matrix(
        sub_df,
        dimensions=features,
        color=color_arg,
        color_discrete_map=color_map,
        title="Pairwise Feature Interaction Matrix",
        opacity=0.6
    )
    fig.update_layout(**SOC_PLOT_LAYOUT)
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    return fig
