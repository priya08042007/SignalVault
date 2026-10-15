"""
Signal Vault — Network Intelligence & Packet Analytics Platform
Module: statistical_tests.py
Description: Rigorous hypothesis testing module implementing Welch's t-test,
Student's t-test, One-way ANOVA, and Variance F-test with clear scientific interpretations.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


def run_welch_ttest(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str = "label",
    alpha: float = 0.05,
    equal_var: bool = False
) -> Dict[str, Any]:
    """
    Performs an independent two-sample t-test comparing a network feature between
    Normal traffic (label=0) and Attack traffic (label=1).
    
    Welch's t-test (equal_var=False) is used by default because network packet metrics
    rarely exhibit homoscedasticity (equal variance) between normal and malicious bursts.
    """
    test_name = "Student's Two-Sample t-Test" if equal_var else "Welch's Two-Sample t-Test"
    h0 = f"The mean of '{feature_col}' is identical between Normal and Attack traffic (μ_normal = μ_attack)."
    h1 = f"The mean of '{feature_col}' differs significantly between Normal and Attack traffic (μ_normal ≠ μ_attack)."

    if feature_col not in df.columns or target_col not in df.columns:
        return {
            "valid": False,
            "test_name": test_name,
            "error": f"Required columns '{feature_col}' or '{target_col}' not found."
        }

    # Extract non-null numeric values
    sub_df = df[[feature_col, target_col]].dropna()
    sub_df[feature_col] = pd.to_numeric(sub_df[feature_col], errors="coerce")
    sub_df = sub_df.dropna()

    group_0 = sub_df[sub_df[target_col] == 0][feature_col].to_numpy()
    group_1 = sub_df[sub_df[target_col] == 1][feature_col].to_numpy()

    n0, n1 = len(group_0), len(group_1)
    if n0 < 2 or n1 < 2:
        return {
            "valid": False,
            "test_name": test_name,
            "error": f"Insufficient sample size: Normal (n={n0}), Attack (n={n1}). Minimum 2 required per group."
        }

    var0, var1 = np.var(group_0, ddof=1), np.var(group_1, ddof=1)
    if var0 == 0 and var1 == 0:
        return {
            "valid": False,
            "test_name": test_name,
            "error": "Both groups have zero variance (constant values). Statistical t-test cannot be calculated."
        }

    # Calculate t-statistic and p-value
    t_stat, p_val = stats.ttest_ind(group_0, group_1, equal_var=equal_var, nan_policy="omit")

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2)) if (n0 + n1 - 2) > 0 else 1.0
    cohens_d = (np.mean(group_1) - np.mean(group_0)) / pooled_std if pooled_std > 0 else 0.0

    rejected = bool(p_val < alpha)

    if rejected:
        decision = f"Reject Null Hypothesis H₀ at α={alpha}"
        interpretation = (
            f"Statistically significant difference detected (p = {p_val:.4e} < {alpha}). "
            f"The feature '{feature_col}' has a mean of {np.mean(group_0):.4f} for Normal traffic "
            f"and {np.mean(group_1):.4f} for Attack traffic (Cohen's d = {cohens_d:.3f}). "
            "This suggests the feature offers strong discriminative potential for anomaly detection."
        )
    else:
        decision = f"Fail to Reject Null Hypothesis H₀ at α={alpha}"
        interpretation = (
            f"No statistically significant difference detected (p = {p_val:.4e} >= {alpha}). "
            f"Normal mean ({np.mean(group_0):.4f}) and Attack mean ({np.mean(group_1):.4f}) "
            "do not show sufficient evidence of separation in this sample."
        )

    return {
        "valid": True,
        "test_name": test_name,
        "feature": feature_col,
        "H0": h0,
        "H1": h1,
        "alpha": alpha,
        "test_statistic": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d),
        "group_normal": {
            "n": n0,
            "mean": float(np.mean(group_0)),
            "std": float(np.std(group_0, ddof=1)),
            "median": float(np.median(group_0))
        },
        "group_attack": {
            "n": n1,
            "mean": float(np.mean(group_1)),
            "std": float(np.std(group_1, ddof=1)),
            "median": float(np.median(group_1))
        },
        "decision": decision,
        "rejected": rejected,
        "interpretation": interpretation,
        "scientific_note": (
            "Cautionary Note: Statistical significance (low p-value) indicates evidence against the null "
            "hypothesis under the sampled data, but does NOT imply physical causation or that the feature alone "
            "is sufficient for accurate intrusion detection without multivariate modeling."
        )
    }


def run_oneway_anova(
    df: pd.DataFrame,
    feature_col: str,
    cat_col: str = "attack_cat",
    alpha: float = 0.05,
    min_group_size: int = 5
) -> Dict[str, Any]:
    """
    Performs a One-Way Analysis of Variance (ANOVA) to test whether the mean of a numerical
    feature differs across distinct attack categories.
    """
    test_name = "One-Way ANOVA (F-Test across Categories)"
    h0 = f"The mean of '{feature_col}' is equal across all attack categories (μ_1 = μ_2 = ... = μ_k)."
    h1 = f"At least one attack category has a significantly different mean for '{feature_col}'."

    if feature_col not in df.columns or cat_col not in df.columns:
        return {
            "valid": False,
            "test_name": test_name,
            "error": f"Required columns '{feature_col}' or '{cat_col}' not found."
        }

    sub_df = df[[feature_col, cat_col]].dropna()
    sub_df[feature_col] = pd.to_numeric(sub_df[feature_col], errors="coerce")
    sub_df = sub_df.dropna()

    groups = []
    group_labels = []
    group_stats = {}

    for cat_name, group_data in sub_df.groupby(cat_col):
        vals = group_data[feature_col].to_numpy()
        if len(vals) >= min_group_size and np.var(vals) > 0:
            groups.append(vals)
            group_labels.append(str(cat_name))
            group_stats[str(cat_name)] = {
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1))
            }

    if len(groups) < 2:
        return {
            "valid": False,
            "test_name": test_name,
            "error": f"Fewer than 2 valid categories with at least {min_group_size} samples and non-zero variance."
        }

    f_stat, p_val = stats.f_oneway(*groups)
    rejected = bool(p_val < alpha)

    if rejected:
        decision = f"Reject Null Hypothesis H₀ at α={alpha}"
        interpretation = (
            f"Statistically significant variation across attack categories (F = {f_stat:.4f}, p = {p_val:.4e}). "
            f"At least one category exhibits a distinctly different mean for '{feature_col}'."
        )
    else:
        decision = f"Fail to Reject Null Hypothesis H₀ at α={alpha}"
        interpretation = (
            f"No statistically significant difference among attack categories (F = {f_stat:.4f}, p = {p_val:.4e}). "
            f"The feature '{feature_col}' appears relatively homogeneous across categories."
        )

    return {
        "valid": True,
        "test_name": test_name,
        "feature": feature_col,
        "category_column": cat_col,
        "H0": h0,
        "H1": h1,
        "alpha": alpha,
        "test_statistic": float(f_stat),
        "p_value": float(p_val),
        "categories_evaluated": group_labels,
        "group_stats": group_stats,
        "decision": decision,
        "rejected": rejected,
        "interpretation": interpretation,
        "scientific_note": (
            "ANOVA identifies overall mean heterogeneity across groups but does not identify which specific "
            "pairs differ. A significant ANOVA should typically be followed by post-hoc analysis (e.g., Tukey HSD)."
        )
    }


def run_variance_ftest(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str = "label",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Performs an F-test for the equality of two variances:
    H0: sigma^2_normal = sigma^2_attack
    H1: sigma^2_normal != sigma^2_attack
    """
    test_name = "Two-Sample F-Test for Equality of Variances"
    h0 = f"The variance of '{feature_col}' is equal between Normal and Attack traffic (σ²_normal = σ²_attack)."
    h1 = f"The variance of '{feature_col}' differs between Normal and Attack traffic (σ²_normal ≠ σ²_attack)."

    if feature_col not in df.columns or target_col not in df.columns:
        return {
            "valid": False,
            "test_name": test_name,
            "error": f"Required columns '{feature_col}' or '{target_col}' not found."
        }

    sub_df = df[[feature_col, target_col]].dropna()
    sub_df[feature_col] = pd.to_numeric(sub_df[feature_col], errors="coerce")
    sub_df = sub_df.dropna()

    group_0 = sub_df[sub_df[target_col] == 0][feature_col].to_numpy()
    group_1 = sub_df[sub_df[target_col] == 1][feature_col].to_numpy()

    n0, n1 = len(group_0), len(group_1)
    if n0 < 2 or n1 < 2:
        return {
            "valid": False,
            "test_name": test_name,
            "error": "Insufficient sample size (minimum 2 samples required per class)."
        }

    var0 = np.var(group_0, ddof=1)
    var1 = np.var(group_1, ddof=1)

    if var0 == 0 or var1 == 0:
        return {
            "valid": False,
            "test_name": test_name,
            "error": "One or both groups have zero variance."
        }

    # F statistic defined as var0 / var1
    f_stat = var0 / var1
    df0 = n0 - 1
    df1 = n1 - 1

    # Two-tailed p-value from F-distribution
    p_lower = stats.f.cdf(f_stat, df0, df1)
    p_val = 2 * min(p_lower, 1.0 - p_lower)

    rejected = bool(p_val < alpha)

    decision = f"Reject Null Hypothesis H₀ at α={alpha}" if rejected else f"Fail to Reject H₀ at α={alpha}"
    interpretation = (
        f"Variance ratio F = {f_stat:.4f} (p = {p_val:.4e}). "
        + ("Normal and Attack traffic exhibit significantly unequal variances (heteroscedasticity)."
           if rejected else "Variances between Normal and Attack traffic are not statistically distinguishable.")
    )

    return {
        "valid": True,
        "test_name": test_name,
        "feature": feature_col,
        "H0": h0,
        "H1": h1,
        "alpha": alpha,
        "test_statistic": float(f_stat),
        "p_value": float(p_val),
        "variance_normal": float(var0),
        "variance_attack": float(var1),
        "df_normal": int(df0),
        "df_attack": int(df1),
        "decision": decision,
        "rejected": rejected,
        "interpretation": interpretation,
        "scientific_note": (
            "Network packet metrics often violate strict normality assumptions. Rejection of equal variance "
            "strongly justifies using Welch's t-test over standard Student's t-test."
        )
    }
