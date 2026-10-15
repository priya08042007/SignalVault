"""
Unit tests for src/statistical_tests.py
"""

import unittest
import numpy as np
import pandas as pd
from src.statistical_tests import (
    run_welch_ttest,
    run_oneway_anova,
    run_variance_ftest
)


class TestStatisticalTests(unittest.TestCase):

    def setUp(self):
        """Create synthetic data with known statistical differences."""
        np.random.seed(42)
        n = 100
        # Normal traffic has mean ~10, Attack traffic has mean ~50
        normal_vals = np.random.normal(loc=10.0, scale=2.0, size=n)
        attack_vals = np.random.normal(loc=50.0, scale=8.0, size=n)

        labels = [0] * n + [1] * n
        durations = np.concatenate([normal_vals, attack_vals])
        categories = ["Normal"] * n + ["Exploits"] * (n // 2) + ["DoS"] * (n // 2)

        self.df = pd.DataFrame({
            "dur": durations,
            "label": labels,
            "attack_cat": categories,
            "constant_col": [5.0] * (2 * n)
        })

    def test_welch_ttest_valid_rejection(self):
        """Test Welch's t-test rejects H0 when means are clearly separated."""
        res = run_welch_ttest(self.df, feature_col="dur", target_col="label", alpha=0.05)
        self.assertTrue(res["valid"])
        self.assertTrue(res["rejected"])
        self.assertLess(res["p_value"], 0.001)
        self.assertIn("Reject", res["decision"])

    def test_welch_ttest_constant_values(self):
        """Test that zero-variance columns are safely intercepted without crashing."""
        res = run_welch_ttest(self.df, feature_col="constant_col", target_col="label")
        self.assertFalse(res["valid"])
        self.assertIn("zero variance", res["error"].lower())

    def test_welch_ttest_insufficient_samples(self):
        """Test graceful handling when a group has fewer than 2 records."""
        tiny_df = pd.DataFrame({"feat": [10.0, 20.0], "label": [0, 0]})
        res = run_welch_ttest(tiny_df, feature_col="feat", target_col="label")
        self.assertFalse(res["valid"])
        self.assertIn("insufficient sample size", res["error"].lower())

    def test_oneway_anova_valid(self):
        """Test ANOVA detects variance between Normal, Exploits, and DoS."""
        res = run_oneway_anova(self.df, feature_col="dur", cat_col="attack_cat")
        self.assertTrue(res["valid"])
        self.assertTrue(res["rejected"])
        self.assertLess(res["p_value"], 0.01)
        self.assertEqual(len(res["categories_evaluated"]), 3)

    def test_variance_ftest(self):
        """Test Two-Sample F-test flags unequal variance (scale 2.0 vs 8.0)."""
        res = run_variance_ftest(self.df, feature_col="dur", target_col="label")
        self.assertTrue(res["valid"])
        self.assertTrue(res["rejected"])
        self.assertNotEqual(res["variance_normal"], res["variance_attack"])


if __name__ == "__main__":
    unittest.main()
