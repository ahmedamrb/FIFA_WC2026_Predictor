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


class TemperatureScaling:
    """Post-hoc probability calibration via a single temperature parameter T.

    Applies the transformation p_cal_i = p_i^(1/T) / sum_j(p_j^(1/T)).
    Fit by minimising NLL on a held-out validation set.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def calibrate(self, proba: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to a probability array.

        Args:
            proba: Array of shape (n_samples, 3) with rows summing to 1.

        Returns:
            Calibrated array of same shape.
        """
        clipped = np.clip(proba, 1e-9, 1.0)
        powered = clipped ** (1.0 / self.temperature)
        return powered / powered.sum(axis=1, keepdims=True)

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "TemperatureScaling":
        """Find T that minimises NLL on the provided proba/y pair.

        Args:
            proba: Array of shape (n_val, 3).
            y: Integer ground-truth labels (0/1/2), shape (n_val,).

        Returns:
            self (fitted).
        """
        from scipy.optimize import minimize_scalar
        from sklearn.metrics import log_loss

        def nll(t: float) -> float:
            if t <= 0:
                return 1e9
            clipped = np.clip(proba, 1e-9, 1.0)
            powered = clipped ** (1.0 / t)
            cal_t = powered / powered.sum(axis=1, keepdims=True)
            return log_loss(y, cal_t)

        result = minimize_scalar(nll, bounds=(0.1, 5.0), method="bounded")
        self.temperature = float(result.x)
        return self


class TemperatureScaledEnsemble:
    """WC2026Ensemble wrapped with temperature scaling calibration."""

    def __init__(self, ensemble: "WC2026Ensemble", temperature: float):
        self._ensemble = ensemble
        self._ts = TemperatureScaling(temperature)

    @property
    def temperature(self) -> float:
        return self._ts.temperature

    def predict_proba(self, X) -> np.ndarray:
        raw = self._ensemble.predict_proba(X)
        return self._ts.calibrate(raw)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
