"""
Signal Vault — Network Intelligence & Packet Analytics Platform
Module: evaluation.py
Description: Comprehensive cybersecurity evaluation metrics, confusion matrices,
ROC-AUC analysis, and three-way comparative benchmarking (Baseline vs PCA vs GA).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
import plotly.express as px
import plotly.graph_objects as go
from src.eda import SOC_COLORS, SOC_PLOT_LAYOUT


def evaluate_classifier_performance(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_label: str = "Classifier"
) -> Dict[str, Any]:
    """
    Computes rigorous intrusion detection performance metrics on the validation set.
    """
    y_true = np.asarray(y_val).astype(int)
    y_pred = model.predict(X_val)

    # Probabilities for ROC-AUC
    y_proba = None
    roc_auc = None
    fpr, tpr = None, None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)
        if probs is not None and probs.shape[1] > 1:
            y_proba = probs[:, 1]
            try:
                roc_auc = float(roc_auc_score(y_true, y_proba))
                fpr_arr, tpr_arr, _ = roc_curve(y_true, y_proba)
                fpr = fpr_arr.tolist()
                tpr = tpr_arr.tolist()
            except Exception:
                roc_auc = None

    # Core Metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec_binary = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_binary = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_binary = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])) if cm.shape == (2, 2) else (0, 0, 0, 0)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] if cm.size > 0 else np.zeros((2, 2))

    # Detailed report dict
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, zero_division=0)

    return {
        "model_label": model_label,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec_binary, 4),
        "recall": round(rec_binary, 4),
        "f1_score": round(f1_binary, 4),
        "f1_macro": round(f1_macro, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "fpr": fpr,
        "tpr": tpr,
        "confusion_matrix": cm.tolist(),
        "cm_normalized": np.round(cm_norm, 4).tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text
    }


def plot_confusion_matrix_heatmap(eval_results: Dict[str, Any]) -> go.Figure:
    """Renders an interactive confusion matrix heatmap with counts and percentages."""
    cm = np.array(eval_results["confusion_matrix"])
    cm_norm = np.array(eval_results["cm_normalized"])

    # Labels
    labels = ["Normal (0)", "Attack (1)"]
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            annotations.append(f"{val:,}<br>({pct:.1f}%)")

    annotations_grid = np.array(annotations).reshape(len(labels), len(labels))

    fig = px.imshow(
        cm_norm,
        x=labels,
        y=labels,
        labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Proportion"),
        color_continuous_scale=[[0, "#0a192f"], [0.5, "#1e3a8a"], [1, SOC_COLORS["accent"]]],
        zmin=0,
        zmax=1,
        title=f"Confusion Matrix: {eval_results['model_label']}"
    )

    fig.update_traces(text=annotations_grid, texttemplate="%{text}")
    fig.update_layout(**SOC_PLOT_LAYOUT)
    return fig


def plot_roc_curves_comparison(eval_list: List[Dict[str, Any]]) -> go.Figure:
    """Overlays ROC curves for multiple models."""
    fig = go.Figure()

    # Diagonal reference line (Random chance)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color=SOC_COLORS["grid"], width=2),
        name="Chance (AUC = 0.50)"
    ))

    colors = [SOC_COLORS["accent"], SOC_COLORS["normal"], SOC_COLORS["warning"], SOC_COLORS["secondary"]]

    for i, res in enumerate(eval_list):
        if res.get("fpr") and res.get("tpr") and res.get("roc_auc") != "N/A":
            auc_val = res["roc_auc"]
            fig.add_trace(go.Scatter(
                x=res["fpr"],
                y=res["tpr"],
                mode="lines",
                name=f"{res['model_label']} (AUC = {auc_val:.4f})",
                line=dict(color=colors[i % len(colors)], width=3)
            ))

    fig.update_layout(
        **SOC_PLOT_LAYOUT,
        title="Receiver Operating Characteristic (ROC) Comparison",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Recall / Sensitivity)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_model_comparison_bar(comparison_df: pd.DataFrame) -> go.Figure:
    """Generates grouped bar chart comparing performance metrics across Baseline, PCA, and GA."""
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score"]
    
    # Filter available metrics
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]

    melted = comparison_df.melt(
        id_vars=["Model Strategy"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="Model Strategy",
        barmode="group",
        color_discrete_sequence=[SOC_COLORS["secondary"], SOC_COLORS["accent"], SOC_COLORS["normal"]],
        title="Intrusion Detection Performance: Baseline vs PCA vs Genetic Algorithm"
    )
    fig.update_layout(
        **SOC_PLOT_LAYOUT,
        yaxis=dict(range=[0, 1.05], gridcolor=SOC_COLORS["grid"])
    )
    return fig
