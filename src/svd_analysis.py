"""
Signal Vault — Network Intelligence & Packet Analytics Platform
Module: svd_analysis.py
Description: Singular Value Decomposition (SVD) and Eckart-Young-Mirsky low-rank
approximation for network traffic matrices with energy spectrum and reconstruction error.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from src.eda import SOC_COLORS, SOC_PLOT_LAYOUT


def compute_svd_analysis(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    max_samples: int = 2500,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Computes Singular Value Decomposition (X = U * Sigma * V^T) on a standardized
    traffic feature matrix.
    
    Subsamples rows to max_samples for interactive laptop performance.
    """
    if feature_cols is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in num_cols if c.lower() not in ("id", "unnamed: 0", "label")]

    if not feature_cols:
        raise ValueError("No numerical features available for SVD.")

    sub_df = df[feature_cols].copy()
    if len(sub_df) > max_samples:
        sub_df = sub_df.sample(n=max_samples, random_state=random_state)

    # Impute and standardize
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_mat = scaler.fit_transform(imputer.fit_transform(sub_df))

    n_samples, n_features = X_mat.shape

    # Economy / Reduced SVD: U is (n, k), s is (k,), Vt is (k, m) where k = min(n, m)
    U, s, Vt = linalg.svd(X_mat, full_matrices=False)

    # Total squared Frobenius energy of singular values
    total_energy = np.sum(s ** 2)
    energy_ratios = (s ** 2) / total_energy
    cumulative_energy = np.cumsum(energy_ratios)

    # Compute reconstruction errors for ranks k = 1..len(s)
    ranks = list(range(1, len(s) + 1))
    rel_errors = [float(np.sqrt(max(0.0, 1.0 - cumulative_energy[k - 1]))) for k in ranks]

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "feature_names": feature_cols,
        "singular_values": s.tolist(),
        "energy_ratios": energy_ratios.tolist(),
        "cumulative_energy": cumulative_energy.tolist(),
        "reconstruction_errors": rel_errors,
        "ranks": ranks,
        "U": U,
        "s": s,
        "Vt": Vt,
        "X_original": X_mat,
        "educational_note": (
            "Singular Value Decomposition (SVD) factors an (m x n) network matrix into X = U * Σ * Vᵀ, "
            "where U and V contain orthonormal singular vectors and Σ contains singular values in descending order. "
            "By the Eckart-Young theorem, truncating Σ at rank k produces the optimal low-rank rank-k matrix approximation "
            "minimizing the Frobenius reconstruction error."
        )
    }


def reconstruct_low_rank_matrix(
    svd_results: Dict[str, Any],
    rank_k: int
) -> Tuple[np.ndarray, float]:
    """
    Reconstructs the low-rank approximation X_k = U_k * Sigma_k * V_k^T
    and calculates empirical Frobenius reconstruction error.
    """
    U = svd_results["U"]
    s = svd_results["s"]
    Vt = svd_results["Vt"]
    X_orig = svd_results["X_original"]

    k = max(1, min(rank_k, len(s)))

    # Reconstruct rank-k matrix
    X_k = np.dot(U[:, :k] * s[:k], Vt[:k, :])

    # Empirical Relative Frobenius Error: ||X - X_k||_F / ||X||_F
    norm_diff = linalg.norm(X_orig - X_k, "fro")
    norm_orig = linalg.norm(X_orig, "fro")
    rel_error = float(norm_diff / norm_orig) if norm_orig > 0 else 0.0

    return X_k, rel_error


def plot_singular_values(svd_results: Dict[str, Any]) -> go.Figure:
    """Visualizes singular values and their cumulative energy capture."""
    s = svd_results["singular_values"]
    cum_energy = svd_results["cumulative_energy"]
    ranks = list(range(1, len(s) + 1))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=ranks,
        y=s,
        name="Singular Value (σ)",
        marker_color=SOC_COLORS["accent"],
        opacity=0.85
    ))

    fig.add_trace(go.Scatter(
        x=ranks,
        y=cum_energy,
        name="Cumulative Energy Ratio",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color=SOC_COLORS["warning"], width=3),
        marker=dict(size=6)
    ))

    fig.update_layout(
        **SOC_PLOT_LAYOUT,
        title="SVD Spectrum: Singular Values (σᵢ) & Cumulative Energy",
        xaxis_title="Index / Rank (i)",
        yaxis_title="Singular Value Magnitude",
        yaxis2=dict(
            title="Cumulative Energy (Σ σ² / Total)",
            overlaying="y",
            side="right",
            range=[0, 1.05],
            gridcolor=SOC_COLORS["grid"]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_reconstruction_error_curve(svd_results: Dict[str, Any]) -> go.Figure:
    """Plots the theoretical and empirical relative Frobenius reconstruction error as rank k increases."""
    ranks = svd_results["ranks"]
    errors = svd_results["reconstruction_errors"]

    fig = px.line(
        x=ranks,
        y=errors,
        markers=True,
        title="Low-Rank Approximation: Frobenius Reconstruction Error vs Rank (k)",
        labels={"x": "Truncated Rank (k)", "y": "Relative Frobenius Error (||X - X_k|| / ||X||)"}
    )
    fig.update_traces(line_color=SOC_COLORS["attack"], line_width=3, marker_color=SOC_COLORS["attack"], marker_size=7)
    fig.update_layout(**SOC_PLOT_LAYOUT)
    return fig


def plot_svd_matrix_comparison(
    svd_results: Dict[str, Any],
    rank_k: int,
    n_display_rows: int = 15,
    n_display_cols: int = 12
) -> go.Figure:
    """
    Creates a visual side-by-side comparison between the original standardized traffic matrix slice,
    its rank-k SVD reconstruction, and the residual error matrix.
    """
    X_orig = svd_results["X_original"][:n_display_rows, :n_display_cols]
    X_k, err = reconstruct_low_rank_matrix(svd_results, rank_k)
    X_k_slice = X_k[:n_display_rows, :n_display_cols]
    residual = np.abs(X_orig - X_k_slice)

    features = svd_results["feature_names"][:n_display_cols]

    fig = go.Figure()

    # Residual Heatmap
    fig = px.imshow(
        residual,
        x=features,
        labels=dict(x="Feature", y="Connection Sample", color="Absolute Residual"),
        title=f"SVD Reconstruction Residual (|X - X_k|) at Rank k={rank_k} (Rel. Error: {err:.2%})",
        color_continuous_scale="Reds"
    )
    fig.update_layout(**SOC_PLOT_LAYOUT)
    return fig
