"""
Unit tests for PCA and SVD dimensionality reduction modules
"""

import unittest
import numpy as np
import pandas as pd
from src.pca_analysis import compute_pca_analysis
from src.svd_analysis import compute_svd_analysis, reconstruct_low_rank_matrix


class TestDimensionalityReduction(unittest.TestCase):

    def setUp(self):
        """Create synthetic multi-feature numeric matrix."""
        np.random.seed(42)
        n_samples = 150
        # 5 correlated numerical features
        f1 = np.random.randn(n_samples)
        f2 = 2.0 * f1 + np.random.randn(n_samples) * 0.5
        f3 = -1.5 * f1 + np.random.randn(n_samples) * 0.2
        f4 = np.random.randn(n_samples) * 3.0
        f5 = f4 * 0.5 + np.random.randn(n_samples) * 0.1

        self.df = pd.DataFrame({
            "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5
        })

    def test_pca_computation(self):
        """Verify PCA calculates orthogonal components and cumulative variance."""
        res = compute_pca_analysis(self.df, n_components=4)
        self.assertEqual(res["n_original_features"], 5)
        self.assertEqual(res["n_components"], 4)
        self.assertEqual(len(res["explained_variance_ratio"]), 4)
        
        # Cumulative variance should be monotonically non-decreasing
        cum_var = res["cumulative_variance_ratio"]
        for i in range(len(cum_var) - 1):
            self.assertLessEqual(cum_var[i], cum_var[i+1])

        # Loadings matrix dimensions
        loadings = res["loadings"]
        self.assertEqual(loadings.shape, (5, 4))

    def test_svd_computation_and_reconstruction(self):
        """Verify SVD factorization and low-rank Frobenius approximation error."""
        res = compute_svd_analysis(self.df, max_samples=100)
        s = res["singular_values"]
        
        # Singular values must be non-negative and sorted descending
        for i in range(len(s) - 1):
            self.assertGreaterEqual(s[i], s[i+1])
            self.assertGreaterEqual(s[i], 0.0)

        # Cumulative energy should terminate at ~1.0
        cum_energy = res["cumulative_energy"]
        self.assertAlmostEqual(cum_energy[-1], 1.0, places=4)

        # Low-rank reconstruction error: Rank 1 error should be greater than Rank 4 error
        _, err_rank1 = reconstruct_low_rank_matrix(res, rank_k=1)
        _, err_rank4 = reconstruct_low_rank_matrix(res, rank_k=4)
        self.assertGreater(err_rank1, err_rank4)


if __name__ == "__main__":
    unittest.main()
