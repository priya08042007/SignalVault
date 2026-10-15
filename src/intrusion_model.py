"""
Signal Vault — Network Intelligence & Packet Analytics Platform
Module: intrusion_model.py
Description: Intrusion Detection System (IDS) model pipeline implementation
supporting Baseline, PCA-compressed, and Genetic Algorithm-selected feature spaces.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import joblib

from src.preprocessing import build_preprocessing_pipeline, categorize_columns


def get_classifier(model_name: str, random_state: int = 42):
    """Factory function returning configured scikit-learn classifier."""
    model_name_clean = model_name.lower().strip()
    if "forest" in model_name_clean:
        return RandomForestClassifier(
            n_estimators=60,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
    elif "decision" in model_name_clean or "tree" in model_name_clean:
        return DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced",
            random_state=random_state
        )
    else: # Logistic Regression default
        return LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state
        )


class IntrusionDetectionPipeline:
    """
    Leakage-free Intrusion Detection Pipeline managing preprocessing,
    dimensionality reduction/feature filtering, and classification.
    """

    def __init__(
        self,
        classifier_name: str = "Random Forest",
        feature_strategy: str = "baseline", # "baseline", "pca", "ga"
        pca_components: Optional[int] = None,
        ga_selected_indices: Optional[List[int]] = None,
        random_state: int = 42
    ):
        self.classifier_name = classifier_name
        self.feature_strategy = feature_strategy
        self.pca_components = pca_components
        self.ga_selected_indices = ga_selected_indices
        self.random_state = random_state

        self.preprocessor = None
        self.pca_transformer = None
        self.classifier = None
        self.training_time_sec = 0.0
        self.input_feature_dim = 0
        self.transformed_feature_dim = 0

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        numerical_cols: List[str],
        categorical_cols: List[str]
    ) -> "IntrusionDetectionPipeline":
        """
        Fits preprocessing, optional dimensionality reduction, and classifier
        strictly on the training partition.
        """
        start_time = time.time()
        self.input_feature_dim = len(numerical_cols) + len(categorical_cols)

        # 1. Fit ColumnTransformer on training partition
        self.preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)
        X_tr_proc = self.preprocessor.fit_transform(X_train)

        # 2. Apply feature reduction strategy
        if self.feature_strategy == "pca":
            n_comp = self.pca_components if self.pca_components else min(10, X_tr_proc.shape[1])
            self.pca_transformer = PCA(n_components=n_comp, random_state=self.random_state)
            X_tr_final = self.pca_transformer.fit_transform(X_tr_proc)
            self.transformed_feature_dim = n_comp

        elif self.feature_strategy == "ga":
            if not self.ga_selected_indices:
                raise ValueError("ga_selected_indices must be provided when feature_strategy is 'ga'.")
            X_tr_final = X_tr_proc[:, self.ga_selected_indices]
            self.transformed_feature_dim = len(self.ga_selected_indices)

        else: # Baseline: All eligible features
            X_tr_final = X_tr_proc
            self.transformed_feature_dim = X_tr_proc.shape[1]

        # 3. Fit Classifier
        self.classifier = get_classifier(self.classifier_name, random_state=self.random_state)
        self.classifier.fit(X_tr_final, y_train)

        self.training_time_sec = round(time.time() - start_time, 3)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms input features and outputs predicted class labels."""
        X_final = self._transform_inputs(X)
        return self.classifier.predict(X_final)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Returns class probabilities if supported by classifier."""
        if not hasattr(self.classifier, "predict_proba"):
            return None
        X_final = self._transform_inputs(X)
        return self.classifier.predict_proba(X_final)

    def _transform_inputs(self, X: pd.DataFrame) -> np.ndarray:
        """Helper to safely apply fitted preprocessing pipeline to new inputs."""
        X_proc = self.preprocessor.transform(X)
        if self.feature_strategy == "pca":
            return self.pca_transformer.transform(X_proc)
        elif self.feature_strategy == "ga":
            return X_proc[:, self.ga_selected_indices]
        return X_proc

    def save_model(self, filepath: str) -> None:
        """Serializes the trained pipeline with joblib."""
        joblib.dump(self, filepath)

    @staticmethod
    def load_model(filepath: str) -> "IntrusionDetectionPipeline":
        """Deserializes a trained pipeline with joblib."""
        return joblib.load(filepath)
