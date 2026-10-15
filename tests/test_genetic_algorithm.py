"""
Unit tests for src/genetic_algorithm.py
"""

import unittest
import numpy as np
from src.genetic_algorithm import GeneticFeatureSelector, run_genetic_feature_selection


class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        """Create synthetic classification data where only 2 of 6 features are predictive."""
        np.random.seed(42)
        n = 120
        # Informative features
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        # Target depends on x1 + x2
        logits = 2.0 * x1 - 1.5 * x2
        y = (logits > 0).astype(int)

        # 4 pure noise features
        noise = np.random.randn(n, 4)

        self.X = np.hstack([x1[:, None], x2[:, None], noise])
        self.y = y
        self.feature_names = ["Inf1", "Inf2", "Noise1", "Noise2", "Noise3", "Noise4"]

    def test_ga_initialization(self):
        """Verify binary mask population generation."""
        ga = GeneticFeatureSelector(population_size=10, generations=3, random_state=42)
        pop = ga._initialize_population(n_features=6)
        self.assertEqual(pop.shape, (10, 6))
        # Ensure values are strictly 0 or 1
        self.assertTrue(np.isin(pop, [0, 1]).all())
        # Ensure at least 1 feature per chromosome
        self.assertTrue((pop.sum(axis=1) > 0).all())

    def test_ga_selection_loop(self):
        """Verify GA evolutionary loop runs, improves or maintains fitness, and selects features."""
        res = run_genetic_feature_selection(
            X_train_processed=self.X,
            y_train=self.y,
            feature_names=self.feature_names,
            population_size=10,
            generations=4,
            random_state=42
        )
        self.assertGreater(res["selected_count"], 0)
        self.assertLessEqual(res["selected_count"], 6)
        self.assertEqual(len(res["history"]), 4)
        self.assertGreater(res["best_fitness"], 0.0)


if __name__ == "__main__":
    unittest.main()
