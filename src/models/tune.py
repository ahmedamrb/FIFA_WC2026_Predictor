"""Hyperparameter tuning via Optuna for outcome and goals models."""

import optuna
from sklearn.model_selection import StratifiedKFold, KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS_XGB = 100
N_TRIALS_RF = 50
N_TRIALS_GOALS = 50


def get_cv_splits(X, y=None, n_splits=5):
    """Return a cross-validation splitter appropriate for the task type.

    Args:
        X: Feature matrix (unused, kept for a consistent call signature).
        y: Target array. If provided, returns StratifiedKFold (classification).
            If None, returns KFold (regression).
        n_splits: Number of cross-validation folds.

    Returns:
        A StratifiedKFold or KFold splitter instance.
    """
    if y is not None:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)


if __name__ == "__main__":
    pass
