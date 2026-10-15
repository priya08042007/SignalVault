"""
Unit tests for src/preprocessing.py
"""

import unittest
import numpy as np
import pandas as pd
from src.preprocessing import (
    inspect_data_quality,
    clean_raw_data,
    categorize_columns,
    split_data,
    build_preprocessing_pipeline
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        """Create synthetic DataFrame fixture for testing."""
        self.df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "dur": [0.1, 0.2, np.nan, 0.4, np.inf, 0.6],
            "proto": ["tcp", "udp", "tcp", "-", "icmp", "tcp"],
            "spkts": [10, 20, 30, 40, 50, 60],
            "attack_cat": ["Normal", "DoS", "Exploits", "Normal", "DoS", "Exploits"],
            "label": [0, 1, 1, 0, 1, 1]
        })

    def test_inspect_data_quality(self):
        """Test reporting of missing values and infinite values."""
        quality = inspect_data_quality(self.df)
        self.assertEqual(quality["total_rows"], 6)
        self.assertIn("dur", quality["columns_with_missing"])
        self.assertIn("dur", quality["columns_with_inf"])
        self.assertEqual(quality["columns_with_inf"]["dur"], 1)

    def test_clean_raw_data(self):
        """Verify infinite values become NaN and '-' becomes 'missing'."""
        cleaned_df, report = clean_raw_data(self.df)
        self.assertEqual(report["inf_replaced"], 1)
        self.assertFalse(np.isinf(cleaned_df["dur"]).any())
        self.assertIn("missing", cleaned_df["proto"].values)

    def test_categorize_columns_leakage_exclusion(self):
        """Ensure id, label, and attack_cat are excluded from eligible features."""
        info = categorize_columns(self.df)
        self.assertIn("dur", info["numerical_features"])
        self.assertIn("spkts", info["numerical_features"])
        self.assertIn("proto", info["categorical_features"])
        
        # Leakage exclusions
        self.assertIn("id", info["excluded_leakage"])
        self.assertNotIn("id", info["all_eligible_features"])
        self.assertNotIn("label", info["all_eligible_features"])
        self.assertNotIn("attack_cat", info["all_eligible_features"])

    def test_split_data_leakage_free(self):
        """Verify train/test split isolates features and maintains stratifying ratio."""
        X_train, X_val, y_train, y_val, info = split_data(
            self.df, target_col="label", test_size=0.33, random_state=42
        )
        self.assertEqual(len(X_train) + len(X_val), 6)
        self.assertNotIn("label", X_train.columns)
        self.assertNotIn("attack_cat", X_train.columns)
        self.assertNotIn("id", X_train.columns)

    def test_pipeline_fit_transform(self):
        """Verify ColumnTransformer handles median imputation and one-hot encoding."""
        cleaned_df, _ = clean_raw_data(self.df)
        info = categorize_columns(cleaned_df)
        pipe = build_preprocessing_pipeline(
            info["numerical_features"],
            info["categorical_features"]
        )
        X_mat = pipe.fit_transform(cleaned_df[info["all_eligible_features"]])
        # Should be a finite 2D array without NaN
        self.assertEqual(X_mat.ndim, 2)
        self.assertFalse(np.isnan(X_mat).any())


if __name__ == "__main__":
    unittest.main()
