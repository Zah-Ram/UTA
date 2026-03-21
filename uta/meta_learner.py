"""
Per-class logistic regression meta-learner for UTA.

Rather than applying a uniform uncertainty threshold, UTA trains independent
logistic regression models for each classification target. This accommodates
the distinct uncertainty signatures of individual diseases or categories.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from .descriptors import compute_trajectory_descriptors, descriptors_to_matrix


class PerClassMetaLearner:
    """Per-class logistic regression meta-learner for misclassification detection.

    For each class, a separate logistic regression model is trained on the
    23 trajectory descriptors to predict whether the base classifier's
    prediction is correct or incorrect.

    Args:
        max_iter: Maximum iterations for logistic regression.
        class_weight: Class weighting strategy ('balanced' recommended).
        random_state: Random seed for reproducibility.
    """

    def __init__(self, max_iter=500, class_weight='balanced', random_state=42):
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.models = {}
        self.descriptor_names = None

    def fit(self, class_idx, main_probs, aux_probs, main_logits, aux_logits, errors):
        """Train meta-learner for a single class.

        Args:
            class_idx: Integer class index (used as dictionary key).
            main_probs: np.ndarray (N,), main classifier probabilities.
            aux_probs: List of 4 np.ndarrays (N,), auxiliary probabilities.
            main_logits: np.ndarray (N,), main classifier logits.
            aux_logits: List of 4 np.ndarrays (N,), auxiliary logits.
            errors: np.ndarray (N,), binary error labels (1=error, 0=correct).
        """
        descriptors = compute_trajectory_descriptors(
            main_probs, aux_probs, main_logits, aux_logits
        )
        X, self.descriptor_names = descriptors_to_matrix(descriptors)

        lr = LogisticRegression(
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        lr.fit(X, errors)
        self.models[class_idx] = lr

    def predict_error_probability(self, class_idx, main_probs, aux_probs,
                                   main_logits, aux_logits):
        """Predict error probability for a single class.

        Args:
            class_idx: Integer class index.
            main_probs, aux_probs, main_logits, aux_logits: Same as fit().

        Returns:
            np.ndarray (N,), predicted probability of misclassification.
        """
        if class_idx not in self.models:
            raise ValueError(f"No model trained for class {class_idx}")

        descriptors = compute_trajectory_descriptors(
            main_probs, aux_probs, main_logits, aux_logits
        )
        X, _ = descriptors_to_matrix(descriptors)

        return self.models[class_idx].predict_proba(X)[:, 1]

    def get_feature_importance(self, class_idx):
        """Get feature importance (logistic regression coefficients) for a class.

        Args:
            class_idx: Integer class index.

        Returns:
            Dict mapping descriptor names to coefficient values.
        """
        if class_idx not in self.models:
            raise ValueError(f"No model trained for class {class_idx}")

        coefficients = self.models[class_idx].coef_[0]
        return dict(zip(self.descriptor_names, coefficients))
