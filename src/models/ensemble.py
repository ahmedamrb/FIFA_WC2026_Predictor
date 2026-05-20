"""Ensemble model combining outcome and goals models."""

import numpy as np


class WC2026Ensemble:
    """Soft-voting ensemble over LR, RF, and XGBoost outcome classifiers.

    Averages the predicted probability arrays from each base model (shape
    (n_samples, 3) each) and returns the element-wise mean.
    """

    def __init__(self, lr_model, rf_model, xgb_model):
        """Store the three fitted base classifiers.

        Args:
            lr_model: Fitted sklearn LR pipeline.
            rf_model: Fitted RandomForestClassifier.
            xgb_model: Fitted XGBClassifier.
        """
        self._models = [lr_model, rf_model, xgb_model]

    def predict_proba(self, X):
        """Average predicted probabilities across all three base models.

        Args:
            X: Feature DataFrame matching training columns.

        Returns:
            np.ndarray of shape (n_samples, 3) with rows summing to 1.0.
        """
        proba_arrays = [m.predict_proba(X) for m in self._models]
        return np.mean(proba_arrays, axis=0)

    def predict(self, X):
        """Return the argmax class from averaged probabilities.

        Args:
            X: Feature DataFrame matching training columns.

        Returns:
            np.ndarray of shape (n_samples,) with integer class labels
            in {0, 1, 2} (0 = away win, 1 = draw, 2 = home win).
        """
        return np.argmax(self.predict_proba(X), axis=1)
