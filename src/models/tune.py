"""Hyperparameter tuning via Optuna for outcome and goals models."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier

matplotlib.use("Agg")  # headless backend — must be set before any other plt calls
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS_XGB = 100
N_TRIALS_RF = 50
N_TRIALS_GOALS = 50

_PLOTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "plots"


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


def tune_xgboost_outcome(X_train, y_train):
    """Run Optuna hyperparameter search for the XGBoost outcome classifier.

    Performs 5-fold stratified cross-validation inside each trial. Saves
    optimisation history and parameter importance plots to outputs/plots/.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training outcome Series (0/1/2).

    Returns:
        dict: Best hyperparameters found by the study.
    """
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    X_arr = X_train.values
    y_arr = y_train.values

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_losses = []
        for train_idx, val_idx in cv.split(X_arr, y_arr):
            X_fold_train, X_fold_val = X_arr[train_idx], X_arr[val_idx]
            y_fold_train, y_fold_val = y_arr[train_idx], y_arr[val_idx]
            model = XGBClassifier(**params)
            model.fit(X_fold_train, y_fold_train, verbose=False)
            proba = model.predict_proba(X_fold_val)
            fold_losses.append(log_loss(y_fold_val, proba))
        return float(np.mean(fold_losses))

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=N_TRIALS_XGB, show_progress_bar=True)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best CV log-loss: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")

    # --- Optimisation history plot ---
    hist_path = _PLOTS_DIR / "optuna_xgb_history.png"
    try:
        from optuna.visualization.matplotlib import plot_optimization_history
        ax = plot_optimization_history(study)
        ax.get_figure().savefig(hist_path, bbox_inches="tight", dpi=100)
        plt.close("all")
    except Exception as exc:  # noqa: BLE001
        print(f"Optuna history plot failed ({exc}); saving manual fallback.")
        trials = study.trials
        best_so_far = []
        running_best = float("inf")
        for t in trials:
            if t.value is not None and t.value < running_best:
                running_best = t.value
            best_so_far.append(running_best)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([t.number for t in trials if t.value is not None],
                [t.value for t in trials if t.value is not None],
                alpha=0.4, label="Trial value")
        ax.plot(range(len(best_so_far)), best_so_far, label="Best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Log-loss")
        ax.set_title("Optuna XGB Optimisation History")
        ax.legend()
        fig.savefig(hist_path, bbox_inches="tight", dpi=100)
        plt.close("all")

    # --- Parameter importance plot ---
    imp_path = _PLOTS_DIR / "optuna_xgb_param_importance.png"
    try:
        from optuna.visualization.matplotlib import plot_param_importances
        ax = plot_param_importances(study)
        ax.get_figure().savefig(imp_path, bbox_inches="tight", dpi=100)
        plt.close("all")
    except Exception as exc:  # noqa: BLE001
        print(f"Optuna param importance plot failed ({exc}); saving placeholder.")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"Param importance unavailable:\n{exc}",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.axis("off")
        fig.savefig(imp_path, bbox_inches="tight", dpi=100)
        plt.close("all")

    return study.best_params


def tune_random_forest_outcome(X_train, y_train):
    """Run Optuna hyperparameter search for the Random Forest outcome classifier.

    Performs 5-fold stratified cross-validation inside each trial. Saves
    optimisation history and parameter importance plots to outputs/plots/.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training outcome Series (0/1/2).

    Returns:
        dict: Best hyperparameters found by the study.
    """
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    X_arr = X_train.values
    y_arr = y_train.values

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_depth": trial.suggest_categorical("max_depth", [5, 10, 20, None]),
            "random_state": 42,
            "n_jobs": -1,
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_losses = []
        for train_idx, val_idx in cv.split(X_arr, y_arr):
            X_fold_train, X_fold_val = X_arr[train_idx], X_arr[val_idx]
            y_fold_train, y_fold_val = y_arr[train_idx], y_arr[val_idx]
            model = RandomForestClassifier(**params)
            model.fit(X_fold_train, y_fold_train)
            proba = model.predict_proba(X_fold_val)
            fold_losses.append(log_loss(y_fold_val, proba, labels=[0, 1, 2]))
        return float(np.mean(fold_losses))

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=N_TRIALS_RF, show_progress_bar=True)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best CV log-loss: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")

    # --- Optimisation history plot ---
    hist_path = _PLOTS_DIR / "optuna_rf_history.png"
    try:
        from optuna.visualization.matplotlib import plot_optimization_history
        ax = plot_optimization_history(study)
        ax.get_figure().savefig(hist_path, bbox_inches="tight", dpi=100)
        plt.close("all")
    except Exception as exc:  # noqa: BLE001
        print(f"Optuna history plot failed ({exc}); saving manual fallback.")
        trials = study.trials
        best_so_far = []
        running_best = float("inf")
        for t in trials:
            if t.value is not None and t.value < running_best:
                running_best = t.value
            best_so_far.append(running_best)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([t.number for t in trials if t.value is not None],
                [t.value for t in trials if t.value is not None],
                alpha=0.4, label="Trial value")
        ax.plot(range(len(best_so_far)), best_so_far, label="Best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Log-loss")
        ax.set_title("Optuna RF Optimisation History")
        ax.legend()
        fig.savefig(hist_path, bbox_inches="tight", dpi=100)
        plt.close("all")

    # --- Parameter importance plot ---
    imp_path = _PLOTS_DIR / "optuna_rf_param_importance.png"
    try:
        from optuna.visualization.matplotlib import plot_param_importances
        ax = plot_param_importances(study)
        ax.get_figure().savefig(imp_path, bbox_inches="tight", dpi=100)
        plt.close("all")
    except Exception as exc:  # noqa: BLE001
        print(f"Optuna param importance plot failed ({exc}); saving placeholder.")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"Param importance unavailable:\n{exc}",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.axis("off")
        fig.savefig(imp_path, bbox_inches="tight", dpi=100)
        plt.close("all")

    return study.best_params


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

    from src.models.outcome_model import evaluate_model, load_splits

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    best_params = tune_xgboost_outcome(X_train, y_train)

    # Retrain on full training set with best params and evaluate on val
    tuned_model = XGBClassifier(
        **best_params,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    tuned_model.fit(X_train, y_train, verbose=False)
    tuned_val_metrics = evaluate_model(tuned_model, X_val, y_val, "Tuned XGB (val)")
    print(
        f"Baseline XGB val log-loss: 1.1591, "
        f"Tuned val log-loss: {tuned_val_metrics['log_loss']:.4f}"
    )
