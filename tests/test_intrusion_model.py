"""
Unit tests for src/intrusion_model.py and src/evaluation.py
"""

import unittest
import numpy as np
import pandas as pd
from src.intrusion_model import IntrusionDetectionPipeline
from src.evaluation import evaluate_classifier_performance


class TestIntrusionModel(unittest.TestCase):

    def setUp(self):
        """Create synthetic tabular data with numerical and categorical features."""
        np.random.seed(42)
        n = 100
        self.X_tr = pd.DataFrame({
            "dur": np.random.uniform(0.01, 2.0, size=n),
            "spkts": np.random.randint(1, 100, size=n),
            "proto": np.random.choice(["tcp", "udp", "icmp"], size=n)
        })
        # Target roughly correlates with dur
        self.y_tr = (self.X_tr["dur"] > 1.0).astype(int)

        self.X_val = pd.DataFrame({
            "dur": np.random.uniform(0.01, 2.0, size=30),
            "spkts": np.random.randint(1, 100, size=30),
            "proto": np.random.choice(["tcp", "udp", "icmp"], size=30)
        })
        self.y_val = (self.X_val["dur"] > 1.0).astype(int)

        self.num_cols = ["dur", "spkts"]
        self.cat_cols = ["proto"]

    def test_baseline_pipeline_fit_and_eval(self):
        """Verify baseline classifier training, prediction, and metric computation."""
        pipe = IntrusionDetectionPipeline(classifier_name="Random Forest", feature_strategy="baseline", random_state=42)
        pipe.fit(self.X_tr, self.y_tr, self.num_cols, self.cat_cols)

        preds = pipe.predict(self.X_val)
        self.assertEqual(len(preds), len(self.y_val))

        eval_res = evaluate_classifier_performance(pipe, self.X_val, self.y_val, model_label="Baseline RF")
        self.assertIn("accuracy", eval_res)
        self.assertIn("f1_score", eval_res)
        self.assertIn("confusion_matrix", eval_res)
        self.assertGreaterEqual(eval_res["accuracy"], 0.0)
        self.assertLessEqual(eval_res["accuracy"], 1.0)

    def test_pca_pipeline(self):
        """Verify PCA-transformed classification pipeline."""
        pipe = IntrusionDetectionPipeline(classifier_name="Logistic Regression", feature_strategy="pca", pca_components=2, random_state=42)
        pipe.fit(self.X_tr, self.y_tr, self.num_cols, self.cat_cols)
        preds = pipe.predict(self.X_val)
        self.assertEqual(len(preds), len(self.y_val))
        self.assertEqual(pipe.transformed_feature_dim, 2)

    def test_ga_pipeline(self):
        """Verify GA-subset classification pipeline."""
        pipe = IntrusionDetectionPipeline(classifier_name="Decision Tree", feature_strategy="ga", ga_selected_indices=[0, 1], random_state=42)
        pipe.fit(self.X_tr, self.y_tr, self.num_cols, self.cat_cols)
        preds = pipe.predict(self.X_val)
        self.assertEqual(len(preds), len(self.y_val))
        self.assertEqual(pipe.transformed_feature_dim, 2)


if __name__ == "__main__":
    unittest.main()
