"""Ensemble model combining outcome and goals models."""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss


class WC2026Ensemble:
    """Soft-voting ensemble over LR, RF, and XGBoost outcome classifiers.

    Supports learned per-model weights that minimise OOF log-loss on the
    training set, or falls back to uniform (equal) weights.
    """

    def __init__(self, lr_model, rf_model, xgb_model, weights=None):
        """Store the three fitted base classifiers and optional per-model weights.

        Args:
            lr_model: Fitted sklearn LR pipeline.
            rf_model: Fitted RandomForestClassifier.
            xgb_model: Fitted XGBClassifier.
            weights: Optional array-like of length 3 with non-negative weights
                that are normalised to sum to 1.  Defaults to uniform (1/3 each).
        """
        self._models = [lr_model, rf_model, xgb_model]
        if weights is None:
            self._weights = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        else:
            arr = np.asarray(weights, dtype=float)
            self._weights = arr / arr.sum()

    @property
    def weights(self):
        """Return a copy of the current per-model weights."""
        return self._weights.copy()

    def predict_proba(self, X):
        """Return probability array as a weighted average of base model probabilities.

        Args:
            X: Feature DataFrame matching training columns.

        Returns:
            np.ndarray of shape (n_samples, 3) with rows summing to 1.0.
        """
        proba_arrays = np.array([m.predict_proba(X) for m in self._models])
        return np.einsum("i,ijk->jk", self._weights, proba_arrays)

    def predict(self, X):
        """Return the argmax class from weighted-average probabilities.

        Args:
            X: Feature DataFrame matching training columns.

        Returns:
            np.ndarray of shape (n_samples,) with integer class labels
            in {0, 1, 2} (0 = away win, 1 = draw, 2 = home win).
        """
        return np.argmax(self.predict_proba(X), axis=1)


def optimize_ensemble_weights(oof_preds, y_train):
    """Find non-negative weights summing to 1 that minimise OOF log-loss.

    Uses a softmax reparametrisation so the optimisation is unconstrained.
    The learned weights are then applied to the final (full-data) base models.

    Args:
        oof_preds: List of 3 numpy arrays, each of shape (n_train, 3),
            containing out-of-fold predicted probabilities from LR, RF, XGB.
        y_train: Training outcome labels (array-like, values in {0, 1, 2}).

    Returns:
        np.ndarray of shape (3,) with optimised weights summing to 1.
    """
    y_arr = np.asarray(y_train)
    stacked = np.array(oof_preds)  # (3, n_train, 3)

    def objective(w_raw):
        w = np.exp(w_raw) / np.exp(w_raw).sum()
        proba = np.einsum("i,ijk->jk", w, stacked)
        return log_loss(y_arr, proba)

    result = minimize(
        objective,
        x0=np.zeros(3),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    raw = result.x
    weights = np.exp(raw) / np.exp(raw).sum()
    return weights
