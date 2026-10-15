"""
Signal Vault — Network Intelligence & Packet Analytics Platform
Module: report_generator.py
Description: Generates structured Markdown and plain-text SOC intelligence reports,
and prepares downloadable CSV/JSON data bundles for all analytical pipelines.
"""

from typing import Dict, Any, List, Optional
import datetime
import pandas as pd
import json


def generate_threat_intelligence_report(
    metadata: Dict[str, Any],
    eda_summary: Optional[Dict[str, Any]] = None,
    stat_results: Optional[List[Dict[str, Any]]] = None,
    pca_results: Optional[Dict[str, Any]] = None,
    svd_results: Optional[Dict[str, Any]] = None,
    ga_results: Optional[Dict[str, Any]] = None,
    model_evaluations: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Assembles a comprehensive, academic and SOC-grade Markdown report detailing
    all computed metrics, statistical hypothesis tests, dimensionality reduction results,
    and intrusion detection benchmarks.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append("# 🛡️ SIGNAL VAULT — NETWORK INTELLIGENCE & PACKET ANALYTICS REPORT")
    md.append(f"**Generated:** {now} | **Environment:** Offline Analytical Engine\n")
    md.append("---")

    # Section 1: Dataset & System Provenance
    md.append("## 1. Executive Summary & Dataset Provenance")
    md.append(f"- **Dataset Source:** {metadata.get('source_name', 'UNSW-NB15')}")
    md.append(f"- **Total Records Analyzed:** {metadata.get('loaded_rows', 'N/A'):,}")
    md.append(f"- **Original Dataset Size:** {metadata.get('total_rows', 'N/A'):,} records")
    md.append(f"- **Feature Dimension Count:** {metadata.get('num_columns', 'N/A')} columns")
    md.append(f"- **Memory Footprint:** {metadata.get('memory_mb', 'N/A')} MB")
    md.append(f"- **Subsampling Applied:** {'Yes' if metadata.get('is_sampled') else 'No'}\n")

    md.append("> **Academic Methodology Note & Dataset Limitation:**")
    md.append("> This report was generated using the single `UNSW_NB15_training-set.csv` file. ")
    md.append("> All validation scores reflect a strictly stratified hold-out partition (e.g. 75/25 train-val split) ")
    md.append("> derived from this training set to prevent data leakage. This internal validation set is distinct from ")
    md.append("> the official UNSW-NB15 testing partition and provides offline proof-of-concept benchmarks.\n")

    # Section 2: Traffic Class Distribution
    if eda_summary and "class_distribution" in eda_summary:
        cd = eda_summary["class_distribution"]
        md.append("## 2. Traffic Composition & Target Distribution")
        md.append(f"- **Normal Traffic (0):** {cd.get('Normal', 0):,} flows ({cd.get('Normal_pct', 0):.2f}%)")
        md.append(f"- **Malicious Attack Traffic (1):** {cd.get('Attack', 0):,} flows ({cd.get('Attack_pct', 0):.2f}%)")
        md.append(f"- **Attack Class Ratio:** {cd.get('Attack_pct', 0):.2f}%\n")

    # Section 3: Statistical Hypothesis Testing
    if stat_results:
        md.append("## 3. Statistical Hypothesis Testing Findings")
        for res in stat_results:
            if not res.get("valid", False):
                continue
            md.append(f"### {res['test_name']} — Feature: `{res.get('feature')}`")
            md.append(f"- **H₀ (Null Hypothesis):** {res.get('H0')}")
            md.append(f"- **H₁ (Alternative Hypothesis):** {res.get('H1')}")
            md.append(f"- **Test Statistic:** {res.get('test_statistic', 0.0):.4f}")
            md.append(f"- **p-value:** {res.get('p_value', 1.0):.4e} (Significance Level α = {res.get('alpha', 0.05)})")
            md.append(f"- **Empirical Decision:** **{res.get('decision')}**")
            md.append(f"- **Interpretation:** {res.get('interpretation')}")
            md.append(f"- *Methodological Note:* {res.get('scientific_note')}\n")

    # Section 4: Dimensionality Reduction (PCA & SVD)
    if pca_results or svd_results:
        md.append("## 4. Dimensionality Reduction & Matrix Factorization")
        if pca_results:
            md.append("### A. Principal Component Analysis (PCA)")
            md.append(f"- **Original Numerical Input Features:** {pca_results.get('n_original_features')}")
            md.append(f"- **Components Extracted:** {pca_results.get('n_components')}")
            th = pca_results.get("threshold_components", {})
            md.append(f"- **Components for 80% Variance:** {th.get(80, 'N/A')}")
            md.append(f"- **Components for 90% Variance:** {th.get(90, 'N/A')}")
            md.append(f"- **Components for 95% Variance:** {th.get(95, 'N/A')}")
            top_var = pca_results.get("explained_variance_ratio", [])[:3]
            var_str = ", ".join([f"PC{i+1}: {v:.2%}" for i, v in enumerate(top_var)])
            md.append(f"- **Top Component Variance Capture:** {var_str}\n")

        if svd_results:
            md.append("### B. Singular Value Decomposition (SVD)")
            md.append(f"- **Standardized Traffic Matrix Rank:** {len(svd_results.get('singular_values', []))}")
            s_top = svd_results.get("singular_values", [])[:3]
            s_str = ", ".join([f"σ{i+1} = {val:.2f}" for i, val in enumerate(s_top)])
            md.append(f"- **Leading Singular Values:** {s_str}")
            if svd_results.get("reconstruction_errors"):
                md.append(f"- **Rank-5 Low-Rank Reconstruction Error:** {svd_results['reconstruction_errors'][min(4, len(svd_results['reconstruction_errors'])-1)]:.2%}")
                md.append(f"- **Rank-10 Low-Rank Reconstruction Error:** {svd_results['reconstruction_errors'][min(9, len(svd_results['reconstruction_errors'])-1)]:.2%}\n")

    # Section 5: Genetic Algorithm Feature Selection
    if ga_results:
        md.append("## 5. Genetic Algorithm (GA) Combinatorial Feature Selection")
        md.append(f"- **Initial Candidate Features:** {ga_results.get('total_features')}")
        md.append(f"- **Optimally Selected Features:** {ga_results.get('selected_count')}")
        md.append(f"- **Compression Ratio:** {(1 - ga_results.get('selected_count', 0)/max(1, ga_results.get('total_features', 1))):.1%} feature reduction")
        md.append(f"- **Peak Fitness Achieved:** {ga_results.get('best_fitness', 0.0):.4f}")
        md.append(f"- **Selected Feature Set:** `{', '.join(ga_results.get('selected_features', []))}`\n")

    # Section 6: Intrusion Detection Classifier Benchmarks
    if model_evaluations:
        md.append("## 6. Intrusion Detection Model Benchmarks & Comparison")
        md.append("| Strategy / Model | Features | Accuracy | Precision | Recall | F1-Score | ROC-AUC |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for ev in model_evaluations:
            auc = f"{ev['roc_auc']:.4f}" if isinstance(ev.get("roc_auc"), (int, float)) else str(ev.get("roc_auc"))
            feat_count = ev.get("features_count", "N/A")
            md.append(
                f"| **{ev.get('model_label')}** | {feat_count} | {ev.get('accuracy', 0):.4f} | "
                f"{ev.get('precision', 0):.4f} | {ev.get('recall', 0):.4f} | {ev.get('f1_score', 0):.4f} | {auc} |"
            )
        md.append("\n")

    md.append("---")
    md.append("*Report generated autonomously by Signal Vault Intelligence Platform.*")

    return "\n".join(md)


def export_dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Converts a pandas DataFrame to UTF-8 CSV bytes for Streamlit download."""
    return df.to_csv(index=False).encode("utf-8")


def export_dict_to_json_bytes(data: Dict[str, Any]) -> bytes:
    """Serializes a dictionary to formatted JSON bytes."""
    return json.dumps(data, indent=2, default=str).encode("utf-8")
