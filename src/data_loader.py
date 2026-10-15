"""
Signal Vault — Network Intelligence & Packet Analytics Platform
Module: data_loader.py
Description: Safe, cached loading of UNSW-NB15 network traffic dataset with
robust schema validation, sampling controls, and fallback setup messaging.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import io
import pandas as pd
import numpy as np


DEFAULT_DATA_DIR = Path("data")
DEFAULT_DATA_FILENAME = "UNSW_NB15_training-set.csv"
DEFAULT_DATA_PATH = DEFAULT_DATA_DIR / DEFAULT_DATA_FILENAME


def get_default_data_path() -> Path:
    """Returns the default expected path for the UNSW-NB15 training dataset."""
    return DEFAULT_DATA_PATH


def check_dataset_exists(filepath: Optional[Union[str, Path]] = None) -> bool:
    """
    Checks if the local CSV dataset exists and is a readable file.
    
    Args:
        filepath: Optional path to dataset. Defaults to 'data/UNSW_NB15_training-set.csv'.
        
    Returns:
        bool: True if file exists and has size > 0, False otherwise.
    """
    target = Path(filepath) if filepath else DEFAULT_DATA_PATH
    try:
        return target.is_file() and target.stat().st_size > 0
    except (OSError, PermissionError):
        return False


def get_dataset_download_instructions() -> Dict[str, Any]:
    """
    Provides clear, beginner-friendly instructions on where and how to obtain
    the UNSW-NB15 training set CSV file.
    """
    return {
        "title": "Dataset Setup Instructions",
        "expected_filename": DEFAULT_DATA_FILENAME,
        "target_relative_path": str(DEFAULT_DATA_PATH),
        "target_absolute_path": str(DEFAULT_DATA_PATH.resolve()),
        "dataset_name": "UNSW-NB15 Network Intrusion Detection Dataset",
        "sources": [
            {
                "name": "Kaggle (UNSW-NB15 Dataset by mrwellsdavid)",
                "url": "https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15",
                "notes": "Look for the single file: 'UNSW_NB15_training-set.csv' (~82,332 records)."
            },
            {
                "name": "UNSW Canberra Cyber Research Repository",
                "url": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
                "notes": "Official academic source created by the Cyber Range Lab of UNSW Canberra."
            }
        ],
        "instructions": (
            f"1. Download '{DEFAULT_DATA_FILENAME}'.\n"
            f"2. Place the file inside the project 'data/' directory:\n"
            f"   --> {DEFAULT_DATA_PATH.resolve()}\n"
            "3. Alternatively, use the 'Upload Dataset' button in the dashboard sidebar.\n"
            "4. Refresh the application to explore live analytics."
        )
    }


def load_dataset(
    file_or_path: Union[str, Path, io.BytesIO],
    nrows: Optional[int] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Safely loads the network traffic CSV dataset from a file path or file-like object.
    Performs column whitespace normalization and schema verification.

    Args:
        file_or_path: File path (str/Path) or file-like stream (BytesIO/UploadedFile).
        nrows: Optional maximum number of rows to read from CSV directly.
        sample_size: Optional stratified subsample size to keep dashboard fast.
        random_state: Seed for reproducible sampling.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (Loaded DataFrame, metadata dictionary)

    Raises:
        FileNotFoundError: If a file path is provided but does not exist.
        ValueError: If the file is empty or corrupted.
    """
    source_name = "Unknown Source"
    is_stream = hasattr(file_or_path, "read")

    if not is_stream:
        path = Path(file_or_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset not found at '{path.resolve()}'. "
                f"Please place '{DEFAULT_DATA_FILENAME}' inside the 'data/' folder."
            )
        source_name = str(path.name)
        df = pd.read_csv(path, nrows=nrows)
    else:
        source_name = getattr(file_or_path, "name", "Uploaded File")
        # Ensure pointer at start if seekable
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        df = pd.read_csv(file_or_path, nrows=nrows)

    if df.empty:
        raise ValueError(f"The loaded dataset from '{source_name}' is empty (0 rows).")

    # Normalize column names: strip whitespace and convert to lowercase for internal consistency
    # while keeping track of original column names
    original_cols = list(df.columns)
    cleaned_cols = [c.strip() for c in original_cols]
    df.columns = cleaned_cols

    # Detect primary target 'label' (case-insensitive)
    label_col = None
    for col in df.columns:
        if col.lower() == "label":
            label_col = col
            break

    # Detect attack category column 'attack_cat' (case-insensitive)
    attack_cat_col = None
    for col in df.columns:
        if col.lower() in ("attack_cat", "attack_category", "attack"):
            attack_cat_col = col
            break

    total_rows = len(df)

    # Apply stratified or random sampling if requested
    if sample_size and sample_size < total_rows:
        if label_col and label_col in df.columns and df[label_col].nunique() > 1:
            # Stratified sample on label
            try:
                frac = sample_size / total_rows
                # Pandas modern groupby.sample directly performs stratified sampling without deprecation warnings
                df = df.groupby(label_col, group_keys=False, observed=False).sample(
                    frac=frac, random_state=random_state
                ).reset_index(drop=True)
            except Exception:
                df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        else:
            df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    loaded_rows = len(df)
    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))

    metadata = {
        "source_name": source_name,
        "is_stream": is_stream,
        "total_rows": total_rows,
        "loaded_rows": loaded_rows,
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "has_label": label_col is not None,
        "label_col": label_col,
        "has_attack_cat": attack_cat_col is not None,
        "attack_cat_col": attack_cat_col,
        "memory_mb": round(memory_mb, 2),
        "is_sampled": loaded_rows < total_rows,
    }

    return df, metadata
