"""Full training pipeline for FIFA WC 2026 Predictor.

Subphases 5.6–5.7: trains all tuned outcome and goals models, assembles
and evaluates the soft-voting ensemble, and saves results to
baseline_results.json.

Later subphases (5.8–5.9) will extend this script with calibration and
model serialisation.

Usage:
    python scripts/train.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.models.goals_model import evaluate_goals_model, train_tuned_goals_models
from src.models.outcome_model import evaluate_model, load_splits, train_tuned_models
from src.models.ensemble import WC2026Ensemble

_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main():
    """Train all tuned models, evaluate, and persist updated metrics."""
    print("=== Subphase 5.6 — Train Tuned Final Models ===\n")

    # ------------------------------------------------------------------
    # Load data splits
    # ------------------------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    # ------------------------------------------------------------------
    # Load tuned hyperparameters
    # ------------------------------------------------------------------
    hyperparams_path = _PROCESSED_DIR / "best_hyperparams.json"
    with hyperparams_path.open("r", encoding="utf-8") as fh:
        best_params = json.load(fh)
    print(f"\nLoaded hyperparameters from: {hyperparams_path}")
    print(f"  Keys present: {list(best_params.keys())}")

    # ------------------------------------------------------------------
    # Load goals targets (aligned to train/val/test index)
    # ------------------------------------------------------------------
    features_path = _PROCESSED_DIR / "features_train.parquet"
    goals_df = pd.read_parquet(features_path)
    y_home_train = goals_df.loc[X_train.index, "home_score"]
    y_away_train = goals_df.loc[X_train.index, "away_score"]
    y_home_val = goals_df.loc[X_val.index, "home_score"]
    y_away_val = goals_df.loc[X_val.index, "away_score"]
    y_home_test = goals_df.loc[X_test.index, "home_score"]
    y_away_test = goals_df.loc[X_test.index, "away_score"]

    # ------------------------------------------------------------------
    # Train tuned outcome models
    # ------------------------------------------------------------------
    lr_model, rf_model, xgb_model = train_tuned_models(X_train, y_train, best_params)

    # ------------------------------------------------------------------
    # Evaluate tuned outcome models on val and test
    # ------------------------------------------------------------------
    print("\n--- Tuned Outcome Model Evaluation ---")
    lr_tuned_val = evaluate_model(lr_model, X_val, y_val, "LR (tuned) — Val (WC 2022)")
    lr_tuned_test = evaluate_model(lr_model, X_test, y_test, "LR (tuned) — Test (WC 2018)")
    rf_tuned_val = evaluate_model(rf_model, X_val, y_val, "RF (tuned) — Val (WC 2022)")
    rf_tuned_test = evaluate_model(rf_model, X_test, y_test, "RF (tuned) — Test (WC 2018)")
    xgb_tuned_val = evaluate_model(xgb_model, X_val, y_val, "XGB (tuned) — Val (WC 2022)")
    xgb_tuned_test = evaluate_model(xgb_model, X_test, y_test, "XGB (tuned) — Test (WC 2018)")

    # ------------------------------------------------------------------
    # Subphase 5.7 — Assemble and evaluate soft-voting ensemble
    # ------------------------------------------------------------------
    print("\n--- Ensemble Evaluation ---")
    ensemble = WC2026Ensemble(lr_model, rf_model, xgb_model)
    ensemble_val = evaluate_model(ensemble, X_val, y_val, "Ensemble — Val (WC 2022)")
    ensemble_test = evaluate_model(ensemble, X_test, y_test, "Ensemble — Test (WC 2018)")

    # ------------------------------------------------------------------
    # Print baseline vs tuned comparison
    # ------------------------------------------------------------------
    baseline_path = _PROCESSED_DIR / "baseline_results.json"
    with baseline_path.open("r", encoding="utf-8") as fh:
        results = json.load(fh)

    print("\n=== Baseline vs Tuned — Validation Log-Loss ===")
    comparisons = [
        ("LR",  "LR_val",  lr_tuned_val),
        ("RF",  "RF_val",  rf_tuned_val),
        ("XGB", "XGB_val", xgb_tuned_val),
    ]
    for name, baseline_key, tuned_metrics in comparisons:
        baseline_ll = results[baseline_key]["log_loss"]
        tuned_ll = tuned_metrics["log_loss"]
        tag = "↓ IMPROVED" if tuned_ll < baseline_ll else "↑ WORSE"
        print(f"  {name:5s}: baseline={baseline_ll:.4f}  tuned={tuned_ll:.4f}  {tag}")

    # ------------------------------------------------------------------
    # Full comparison table: LR / RF / XGB / Ensemble
    # ------------------------------------------------------------------
    print("\n=== Subphase 5.7 — Final Model Comparison ===")
    header = (
        f"{'Model':<12}{'Val LL':>10}{'Val Acc':>10}{'Val Brier':>11}"
        f"{'Test LL':>10}{'Test Acc':>10}"
    )
    print(header)
    print("-" * len(header))
    table_rows = [
        ("LR",       lr_tuned_val,   lr_tuned_test),
        ("RF",       rf_tuned_val,   rf_tuned_test),
        ("XGB",      xgb_tuned_val,  xgb_tuned_test),
        ("Ensemble", ensemble_val,   ensemble_test),
    ]
    for name, v, t in table_rows:
        print(
            f"{name:<12}{v['log_loss']:>10.4f}{v['accuracy']:>10.4f}"
            f"{v['brier_score']:>11.4f}{t['log_loss']:>10.4f}{t['accuracy']:>10.4f}"
        )

    # ------------------------------------------------------------------
    # Train tuned goals models
    # ------------------------------------------------------------------
    xgb_home, xgb_away, poisson_home, poisson_away = train_tuned_goals_models(
        X_train, y_home_train, y_away_train, best_params
    )

    # ------------------------------------------------------------------
    # Evaluate tuned goals models
    # ------------------------------------------------------------------
    print("\n--- Tuned XGBoost Goals Model Evaluation (Val) ---")
    goals_xgb_val = evaluate_goals_model(
        xgb_home, xgb_away, X_val, y_home_val, y_away_val
    )

    print("\n--- Tuned XGBoost Goals Model Evaluation (Test) ---")
    goals_xgb_test = evaluate_goals_model(
        xgb_home, xgb_away, X_test, y_home_test, y_away_test
    )

    print("\n--- Poisson Fallback Goals Model Evaluation (Val) ---")
    goals_poisson_val = evaluate_goals_model(
        poisson_home, poisson_away, X_val, y_home_val, y_away_val
    )

    # ------------------------------------------------------------------
    # Append tuned metrics to baseline_results.json
    # ------------------------------------------------------------------
    results["LR_tuned_val"] = lr_tuned_val
    results["LR_tuned_test"] = lr_tuned_test
    results["RF_tuned_val"] = rf_tuned_val
    results["RF_tuned_test"] = rf_tuned_test
    results["XGB_tuned_val"] = xgb_tuned_val
    results["XGB_tuned_test"] = xgb_tuned_test
    results["ensemble_val"] = ensemble_val
    results["ensemble_test"] = ensemble_test
    results["goals_xgb_tuned_val"] = goals_xgb_val
    results["goals_xgb_tuned_test"] = goals_xgb_test
    results["goals_poisson_tuned_val"] = goals_poisson_val

    with baseline_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nUpdated metrics saved to: {baseline_path}")
    print("\n=== Subphase 5.6–5.7 complete ===")


if __name__ == "__main__":
    main()
