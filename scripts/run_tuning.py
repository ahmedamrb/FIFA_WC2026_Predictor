"""Hyperparameter tuning script.

Runs Optuna hyperparameter search for all four models (XGBoost outcome,
Random Forest outcome, XGBoost home goals, XGBoost away goals) and saves
the best parameters to data/processed/best_hyperparams.json.

Usage:
    python scripts/run_tuning.py
"""

import json
import sys
from pathlib import Path

# Allow imports from the repo root regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.outcome_model import load_splits
import pandas as pd

from src.models.tune import (
    N_TRIALS_GOALS,
    N_TRIALS_RF,
    tune_random_forest_outcome,
    tune_xgboost_goals,
    tune_xgboost_outcome,
)

_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main():
    """Load training data, run XGBoost tuning, and persist best hyperparameters."""
    print("=== Subphase 5.2 — Optuna Tuning: XGBoost Outcome Model ===\n")
    X_train, y_train, w_train, X_val, y_val, X_test, y_test = load_splits()

    print(f"\nRunning Optuna search ({100} trials)...\n")
    best_params = tune_xgboost_outcome(X_train, y_train, sample_weight=w_train)

    # Load existing params file (if any) so we don't clobber other keys
    hyperparams_path = _PROCESSED_DIR / "best_hyperparams.json"
    if hyperparams_path.exists():
        with hyperparams_path.open("r", encoding="utf-8") as fh:
            all_params = json.load(fh)
    else:
        all_params = {}

    all_params["xgb_outcome"] = best_params

    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with hyperparams_path.open("w", encoding="utf-8") as fh:
        json.dump(all_params, fh, indent=2)

    print(f"\nBest hyperparameters saved to: {hyperparams_path}")

    # -----------------------------------------------------------------------
    print("\n=== Subphase 5.3 — Optuna Tuning: Random Forest Outcome Model ===\n")
    print(f"Running Optuna search ({N_TRIALS_RF} trials)...\n")
    best_rf_params = tune_random_forest_outcome(X_train, y_train, sample_weight=w_train)

    # Reload params file to merge RF results without clobbering XGB results
    with hyperparams_path.open("r", encoding="utf-8") as fh:
        all_params = json.load(fh)

    all_params["rf_outcome"] = best_rf_params

    with hyperparams_path.open("w", encoding="utf-8") as fh:
        json.dump(all_params, fh, indent=2)

    print(f"\nBest RF hyperparameters saved to: {hyperparams_path}")

    # -----------------------------------------------------------------------
    print("\n=== Subphase 5.4 — Optuna Tuning: Goals Models ===\n")

    features_train_path = (
        Path(__file__).resolve().parents[1] / "data" / "processed" / "features_train.parquet"
    )
    goals_df = pd.read_parquet(features_train_path)
    y_home = goals_df.loc[X_train.index, "home_score"]
    y_away = goals_df.loc[X_train.index, "away_score"]

    print(f"Running Optuna search for home goals ({N_TRIALS_GOALS} trials)...\n")
    xgb_home_goals = tune_xgboost_goals(X_train, y_home, label="home")

    with hyperparams_path.open("r", encoding="utf-8") as fh:
        all_params = json.load(fh)
    all_params["xgb_home_goals"] = xgb_home_goals
    with hyperparams_path.open("w", encoding="utf-8") as fh:
        json.dump(all_params, fh, indent=2)
    print(f"\nBest home-goals hyperparameters saved to: {hyperparams_path}")

    print(f"\nRunning Optuna search for away goals ({N_TRIALS_GOALS} trials)...\n")
    xgb_away_goals = tune_xgboost_goals(X_train, y_away, label="away")

    with hyperparams_path.open("r", encoding="utf-8") as fh:
        all_params = json.load(fh)
    all_params["xgb_away_goals"] = xgb_away_goals
    with hyperparams_path.open("w", encoding="utf-8") as fh:
        json.dump(all_params, fh, indent=2)
    print(f"\nBest away-goals hyperparameters saved to: {hyperparams_path}")


if __name__ == "__main__":
    main()
