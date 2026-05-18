"""Baseline model training and evaluation script for FIFA WC 2026 Predictor.

Trains Logistic Regression, Random Forest, and XGBoost outcome classifiers plus
Poisson goals regressors. Evaluates each on the WC 2022 validation set and the
WC 2018 held-out test set. Prints a consolidated comparison table and writes
all metrics to data/processed/baseline_results.json.

Usage (from repo root):
    python scripts/run_baseline_models.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Allow imports from the repo root regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.goals_model import evaluate_goals_model, train_poisson_goals
from src.models.outcome_model import (
    evaluate_model,
    load_splits,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)

_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _cast_floats(metrics: dict) -> dict:
    """Recursively cast numpy floats to Python floats in a metrics dict.

    Args:
        metrics: Dict potentially containing numpy scalar values.

    Returns:
        New dict with all numeric values converted to Python-native floats.
    """
    return {k: float(v) for k, v in metrics.items()}


def main() -> None:
    """Run full baseline training and evaluation pipeline.

    Steps:
        1. Load train/val/test feature splits.
        2. Extract home/away goal targets from the parquet file.
        3. Train LR, RF, and XGB outcome classifiers.
        4. Train Poisson home/away goals regressors.
        5. Evaluate all outcome models on val (WC 2022) and test (WC 2018).
        6. Evaluate goals models on val (WC 2022).
        7. Print a consolidated comparison table.
        8. Save all metrics to data/processed/baseline_results.json.
    """
    # ------------------------------------------------------------------
    # 1. Load splits
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1 — Loading feature splits")
    print("=" * 70)
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    # ------------------------------------------------------------------
    # 2. Extract goal targets aligned to split indices
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2 — Loading goal targets")
    print("=" * 70)
    features_path = _PROCESSED_DIR / "features_train.parquet"
    df_full = pd.read_parquet(features_path)

    y_home_train = df_full.loc[X_train.index, "home_score"]
    y_away_train = df_full.loc[X_train.index, "away_score"]
    y_home_val = df_full.loc[X_val.index, "home_score"]
    y_away_val = df_full.loc[X_val.index, "away_score"]
    print(f"  Goal targets loaded — train: {len(y_home_train)}, val: {len(y_home_val)}")

    # ------------------------------------------------------------------
    # 3. Train outcome models
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3 — Training outcome classifiers")
    print("=" * 70)

    print("\n--- Logistic Regression ---")
    lr_model = train_logistic_regression(X_train, y_train)

    print("\n--- Random Forest ---")
    rf_model = train_random_forest(X_train, y_train)

    print("\n--- XGBoost ---")
    xgb_model = train_xgboost(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Train Poisson goals models
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4 — Training Poisson goals regressors")
    print("=" * 70)
    home_goals_model, away_goals_model = train_poisson_goals(
        X_train, y_home_train, y_away_train
    )
    print("  Poisson models trained.")

    # ------------------------------------------------------------------
    # 5. Evaluate outcome models on val and test
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5 — Evaluating outcome classifiers")
    print("=" * 70)

    lr_val = evaluate_model(lr_model, X_val, y_val, "LR — Val (WC 2022)")
    lr_test = evaluate_model(lr_model, X_test, y_test, "LR — Test (WC 2018)")

    rf_val = evaluate_model(rf_model, X_val, y_val, "RF — Val (WC 2022)")
    rf_test = evaluate_model(rf_model, X_test, y_test, "RF — Test (WC 2018)")

    xgb_val = evaluate_model(xgb_model, X_val, y_val, "XGB — Val (WC 2022)")
    xgb_test = evaluate_model(xgb_model, X_test, y_test, "XGB — Test (WC 2018)")

    # ------------------------------------------------------------------
    # 6. Evaluate goals model on val
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6 — Evaluating goals regressors on val (WC 2022)")
    print("=" * 70)
    goals_val = evaluate_goals_model(
        home_goals_model, away_goals_model, X_val, y_home_val, y_away_val
    )

    # ------------------------------------------------------------------
    # 7. Consolidated comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7 — Consolidated outcome model comparison")
    print("=" * 70)

    comparison_data = {
        "LR": {
            "val_log_loss": lr_val["log_loss"],
            "val_accuracy": lr_val["accuracy"],
            "val_brier": lr_val["brier_score"],
            "test_log_loss": lr_test["log_loss"],
            "test_accuracy": lr_test["accuracy"],
        },
        "RF": {
            "val_log_loss": rf_val["log_loss"],
            "val_accuracy": rf_val["accuracy"],
            "val_brier": rf_val["brier_score"],
            "test_log_loss": rf_test["log_loss"],
            "test_accuracy": rf_test["accuracy"],
        },
        "XGB": {
            "val_log_loss": xgb_val["log_loss"],
            "val_accuracy": xgb_val["accuracy"],
            "val_brier": xgb_val["brier_score"],
            "test_log_loss": xgb_test["log_loss"],
            "test_accuracy": xgb_test["accuracy"],
        },
    }

    comparison_df = pd.DataFrame(comparison_data).T
    comparison_df.index.name = "Model"
    print(f"\n{comparison_df.to_string(float_format='{:.4f}'.format)}")

    # ------------------------------------------------------------------
    # 8. Save metrics to JSON
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 8 — Saving results to data/processed/baseline_results.json")
    print("=" * 70)

    results = {
        "LR_val": _cast_floats(lr_val),
        "LR_test": _cast_floats(lr_test),
        "RF_val": _cast_floats(rf_val),
        "RF_test": _cast_floats(rf_test),
        "XGB_val": _cast_floats(xgb_val),
        "XGB_test": _cast_floats(xgb_test),
        "goals_val": _cast_floats(goals_val),
    }

    output_path = _PROCESSED_DIR / "baseline_results.json"
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"  Saved → {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()

