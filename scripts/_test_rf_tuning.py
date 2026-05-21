"""Temporary script to run ONLY the RF Optuna tuning for Subphase 5.3 verification."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.outcome_model import load_splits
from src.models.tune import tune_random_forest_outcome


def main():
    X_train, y_train, w_train, X_val, y_val, X_test, y_test = load_splits()
    print(f"X_train shape: {X_train.shape}")

    print("\n=== Subphase 5.3 — Optuna Tuning: Random Forest Outcome Model ===\n")
    best_rf_params = tune_random_forest_outcome(X_train, y_train)

    hyperparams_path = Path("data/processed/best_hyperparams.json")
    if hyperparams_path.exists():
        with open(hyperparams_path, "r") as f:
            all_params = json.load(f)
    else:
        all_params = {}

    all_params["rf_outcome"] = best_rf_params

    with open(hyperparams_path, "w") as f:
        json.dump(all_params, f, indent=2)

    print(f"\nRF best hyperparameters saved to: {hyperparams_path}")
    print(f"All keys in best_hyperparams.json: {list(all_params.keys())}")


if __name__ == "__main__":
    main()
