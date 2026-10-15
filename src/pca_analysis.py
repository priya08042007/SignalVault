"""
Signal Vault — Network Intelligence & Packet Analytics Platform
Module: pca_analysis.py
Description: Principal Component Analysis (PCA) workflow with leakage-free scaling,
scree plots, cumulative explained variance, 2D projections, and component loadings.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from src.eda import SOC_COLORS, SOC_PLOT_LAYOUT


def compute_pca_analysis(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_components: Optional[int] = None,
    variance_threshold: Optional[float] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes PCA on numerical features:
    1. Imputes missing values with median.
    2. Centers and standardizes features (mean=0, variance=1).
    3. Calculates principal components, eigenvalues, and explained variance ratios.
    
    Returns structured results including loadings and variance metrics.
    """
    if feature_cols is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in num_cols if c.lower() not in ("id", "unnamed: 0", "label")]

    if not feature_cols:
        raise ValueError("No numerical features available for PCA.")

    X_raw = df[feature_cols].copy()

    # Step 1: Imputation
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    # Step 2: Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Step 3: Determine number of components
    max_components = min(X_scaled.shape[0], X_scaled.shape[1])
    target_components = n_components if n_components else max_components
    target_components = min(target_components, max_components)

    pca_full = PCA(n_components=target_components, random_state=random_state)
    transformed_data = pca_full.fit_transform(X_scaled)

    explained_var = pca_full.explained_variance_
    explained_var_ratio = pca_full.explained_variance_ratio_
    cum_var_ratio = np.cumsum(explained_var_ratio)

    # Calculate components required for threshold
    threshold_components = {}
    for th in [0.80, 0.90, 0.95]:
        idx = np.argmax(cum_var_ratio >= th)
        # If threshold not reached, set to max components
        threshold_components[int(th * 100)] = int(idx + 1) if cum_var_ratio[idx] >= th else target_components

    # Build Loadings DataFrame
    pc_names = [f"PC{i+1}" for i in range(target_components)]
    loadings_df = pd.DataFrame(
        pca_full.components_.T,
        index=feature_cols,
        columns=pc_names
    )

    # DataFrame with coordinates
    transformed_df = pd.DataFrame(transformed_data[:, :min(5, target_components)], columns=pc_names[:min(5, target_components)])

    return {
        "n_original_features": len(feature_cols),
        "feature_names": feature_cols,
        "n_components": target_components,
        "explained_variance": explained_var.tolist(),
        "explained_variance_ratio": explained_var_ratio.tolist(),
        "cumulative_variance_ratio": cum_var_ratio.tolist(),
        "threshold_components": threshold_components,
        "loadings": loadings_df,
        "transformed_df": transformed_df,
        "scaler": scaler,
        "pca_model": pca_full,
        "educational_note": (
            "Principal Component Analysis (PCA) performs an orthogonal linear transformation to project "
            "the original features onto an uncorrelated coordinate system ordered by variance. "
            "Crucially, PCA creates *new synthetic components* and does not select or discard specific raw features."
        )
    }


def plot_scree_chart(pca_results: Dict[str, Any]) -> go.Figure:
    """Plots individual and cumulative explained variance ratios."""
    ratios = pca_results["explained_variance_ratio"]
    cum_ratios = pca_results["cumulative_variance_ratio"]
    n_comp = len(ratios)
    x_axis = [f"PC{i+1}" for i in range(n_comp)]

    fig = go.Figure()

    # Individual Variance Bar
    fig.add_trace(go.Bar(
        x=x_axis,
        y=ratios,
        name="Individual Variance Ratio",
        marker_color=SOC_COLORS["accent"],
        opacity=0.85
    ))

    # Cumulative Variance Line
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=cum_ratios,
        name="Cumulative Variance Ratio",
        mode="lines+markers",
        line=dict(color=SOC_COLORS["normal"], width=3),
        marker=dict(size=7, color=SOC_COLORS["normal"])
    ))

    # 90% benchmark line
    fig.add_hline(y=0.90, line_dash="dash", line_color=SOC_COLORS["warning"],
                  annotation_text="90% Variance Target", annotation_position="bottom right")

    fig.update_layout(
        **SOC_PLOT_LAYOUT,
        title="PCA Scree Plot & Cumulative Explained Variance",
        xaxis_title="Principal Components",
        yaxis_title="Explained Variance Ratio",
        yaxis=dict(range=[0, 1.05], gridcolor=SOC_COLORS["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_pca_2d_projection(
    df: pd.DataFrame,
    pca_results: Dict[str, Any],
    label_col: Optional[str] = "label",
    sample_size: int = 3000
) -> go.Figure:
    """Projects data onto PC1 and PC2, colored by target class."""
    coords = pca_results["transformed_df"].copy()
    if label_col and label_col in df.columns:
        coords["Target"] = df[label_col].values[:len(coords)]
        coords["Traffic Class"] = coords["Target"].map({0: "Normal", 1: "Attack"}).fillna(coords["Target"].astype(str))
        color_arg = "Traffic Class"
        color_map = {"Normal": SOC_COLORS["normal"], "Attack": SOC_COLORS["attack"]}
    else:
        color_arg = None
        color_map = None

    if len(coords) > sample_size:
        coords = coords.sample(n=sample_size, random_state=42)

    fig = px.scatter(
        coords,
        x="PC1",
        y="PC2",
        color=color_arg,
        color_discrete_map=color_map,
        title="2D Principal Component Space (PC1 vs PC2)",
        opacity=0.7,
        render_mode="webgl"
    )
    fig.update_layout(**SOC_PLOT_LAYOUT)
    return fig


def plot_pca_loadings_heatmap(pca_results: Dict[str, Any], top_n_features: int = 15) -> go.Figure:
    """Displays heatmap of feature contributions (loadings) for top principal components."""
    loadings = pca_results["loadings"].iloc[:, :min(4, pca_results["n_components"])]
    # Filter to top features by absolute magnitude in PC1 + PC2
    abs_importance = loadings.abs().sum(axis=1).sort_values(ascending=False)
    filtered_loadings = loadings.loc[abs_importance.head(top_n_features).index]

    fig = px.imshow(
        filtered_loadings,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Viridis",
        title=f"PCA Factor Loadings Matrix (Top {top_n_features} Features)"
    )
    fig.update_layout(**SOC_PLOT_LAYOUT)
    return fig
