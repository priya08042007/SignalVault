"""
Signal Vault — Network Intelligence & Packet Analytics Platform
Module: preprocessing.py
Description: Robust, leakage-safe data preprocessing, feature classification,
missing/infinite value handling, and Scikit-Learn transformers.
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Common column names that introduce target leakage or are non-predictive identifiers
LEAKAGE_AND_IDENTIFIER_CANDIDATES = {
    "id", "id_seq", "index", "unnamed: 0", "label", "attack_cat", "attack_category", "attack"
}


def inspect_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes dataset quality, including missing values, infinite values,
    and duplicate rows without modifying the underlying DataFrame.
    """
    missing_series = df.isna().sum()
    missing_dict = missing_series[missing_series > 0].to_dict()

    # Infinite values check on numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    inf_dict = {}
    for col in numeric_cols:
        inf_count = int(np.isinf(df[col]).sum())
        if inf_count > 0:
            inf_dict[col] = inf_count

    duplicate_rows = int(df.duplicated().sum())

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns_with_missing": missing_dict,
        "total_missing_cells": int(missing_series.sum()),
        "columns_with_inf": inf_dict,
        "total_duplicate_rows": duplicate_rows,
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


def clean_raw_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs initial data sanitation:
    1. Replaces infinite values with NaN for safe imputation.
    2. Strips whitespace in object/string columns.
    3. Standardizes common null tokens (e.g. '-', ' ', 'null').
    4. Reports sanitation actions performed.
    
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned copy of DataFrame and report.
    """
    df_clean = df.copy()
    report = {
        "inf_replaced": 0,
        "null_tokens_standardized": 0,
        "duplicates_flagged": int(df_clean.duplicated().sum())
    }

    # Replace inf/-inf with NaN
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        inf_mask = np.isinf(df_clean[col])
        if inf_mask.any():
            count = int(inf_mask.sum())
            report["inf_replaced"] += count
            df_clean.loc[inf_mask, col] = np.nan

    # Clean object/string columns
    obj_cols = df_clean.select_dtypes(include=["object", "string", "category"]).columns
    for col in obj_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        # In UNSW-NB15, '-' often indicates missing/empty service
        placeholder_mask = df_clean[col].isin(["-", "", "None", "nan", "null"])
        if placeholder_mask.any():
            report["null_tokens_standardized"] += int(placeholder_mask.sum())
            df_clean.loc[placeholder_mask, col] = "missing"

    return df_clean, report


def categorize_columns(
    df: pd.DataFrame,
    target_col: Optional[str] = "label",
    attack_cat_col: Optional[str] = "attack_cat"
) -> Dict[str, List[str]]:
    """
    Partitions columns into:
    - target: The binary classification target
    - attack_category: Multi-class attack description column
    - excluded_leakage: IDs and columns that directly disclose the target
    - numerical_features: Numeric features safe for modeling
    - categorical_features: Categorical features safe for modeling
    """
    cols = list(df.columns)
    excluded = []
    
    # Identify target column
    resolved_target = None
    if target_col and target_col in cols:
        resolved_target = target_col
    else:
        for c in cols:
            if c.lower() == "label":
                resolved_target = c
                break

    # Identify attack_cat column
    resolved_attack_cat = None
    if attack_cat_col and attack_cat_col in cols:
        resolved_attack_cat = attack_cat_col
    else:
        for c in cols:
            if c.lower() in ("attack_cat", "attack_category", "attack"):
                resolved_attack_cat = c
                break

    # Build leakage / identifier exclusion set
    for c in cols:
        low = c.lower()
        if low in LEAKAGE_AND_IDENTIFIER_CANDIDATES:
            excluded.append(c)
        elif low.startswith("unnamed") or low == "id":
            excluded.append(c)

    # Eligible features are everything except target and excluded
    eligible_cols = [c for c in cols if c not in excluded]

    numerical_features = []
    categorical_features = []

    for c in eligible_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numerical_features.append(c)
        else:
            categorical_features.append(c)

    return {
        "target": [resolved_target] if resolved_target else [],
        "attack_category": [resolved_attack_cat] if resolved_attack_cat else [],
        "excluded_leakage": list(set(excluded)),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "all_eligible_features": numerical_features + categorical_features
    }


def split_data(
    df: pd.DataFrame,
    target_col: str = "label",
    test_size: float = 0.25,
    random_state: int = 42,
    max_samples: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Performs a leakage-safe, stratified train/validation split on the single dataset.

    Note: As specified in project requirements, this validation split is derived
    from the single training-set CSV file for offline validation.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    # Drop rows where target is NaN
    valid_mask = df[target_col].notna()
    df_valid = df[valid_mask].copy()

    # Subsample if max_samples is set
    if max_samples and max_samples < len(df_valid):
        df_valid = df_valid.sample(n=max_samples, random_state=random_state)

    column_info = categorize_columns(df_valid, target_col=target_col)
    feature_cols = column_info["all_eligible_features"]

    X = df_valid[feature_cols]
    y = df_valid[target_col].astype(int)

    # Check stratification feasibility
    stratify = y if y.nunique() > 1 and y.value_counts().min() > 1 else None

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    split_info = {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_ratio": test_size,
        "stratified": stratify is not None,
        "target_column": target_col,
        "num_features": len(feature_cols),
        "feature_columns": feature_cols,
        "excluded_columns": column_info["excluded_leakage"]
    }

    return X_train, X_val, y_train, y_val, split_info


def build_preprocessing_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Constructs a Scikit-Learn ColumnTransformer that handles:
    - Numerical: Median imputation followed by standard scaling.
    - Categorical: Most frequent imputation followed by one-hot encoding (ignoring unseen classes).
    """
    transformers = []

    if numerical_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, numerical_cols))

    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor
