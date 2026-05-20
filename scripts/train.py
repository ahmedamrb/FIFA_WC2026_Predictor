"""Full training pipeline for FIFA WC 2026 Predictor.

Subphases 5.6–5.8: trains all tuned outcome and goals models, assembles
the soft-voting ensemble, checks calibration, applies CalibratedClassifierCV
if it reduces val log-loss, and saves results to baseline_results.json.

Usage:
    python scripts/train.py
"""

import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.models.goals_model import evaluate_goals_model, train_tuned_goals_models
from src.models.outcome_model import evaluate_model, load_splits, train_tuned_models
from src.models.ensemble import WC2026Ensemble
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from src.evaluation.metrics import plot_calibration_curves

_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


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
    # Append tuned/ensemble metrics to results dict
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

    # ------------------------------------------------------------------
    # Subphase 5.8 — Calibration Check
    # ------------------------------------------------------------------
    print("\n=== Subphase 5.8 — Calibration Check ===\n")

    # Plot calibration curves for best individual model (RF) and ensemble
    plot_calibration_curves(rf_model, X_val, y_val, "rf_tuned")
    plot_calibration_curves(ensemble, X_val, y_val, "ensemble")

    # Calibrate XGBoost (most likely overconfident); use 5-fold CV on training
    # data — cv='prefit' was removed in sklearn 1.6+.
    cal_xgb = CalibratedClassifierCV(xgb_model, cv=5, method="isotonic")
    cal_xgb.fit(X_train, y_train)

    # Reassemble ensemble with calibrated XGBoost
    cal_ensemble = WC2026Ensemble(lr_model, rf_model, cal_xgb)

    # Evaluate calibrated ensemble
    cal_ensemble_val = evaluate_model(
        cal_ensemble, X_val, y_val, "Ensemble (calibrated) — Val (WC 2022)"
    )
    cal_ensemble_test = evaluate_model(
        cal_ensemble, X_test, y_test, "Ensemble (calibrated) — Test (WC 2018)"
    )

    # Plot calibration curve for the calibrated ensemble
    plot_calibration_curves(cal_ensemble, X_val, y_val, "ensemble_calibrated")

    # Apply only if val log-loss does not worsen
    pre_ll = ensemble_val["log_loss"]
    post_ll = cal_ensemble_val["log_loss"]

    if post_ll <= pre_ll:
        print(
            f"\n  Calibration APPLIED"
            f"  (val log-loss {pre_ll:.4f} → {post_ll:.4f},  Δ={post_ll - pre_ll:+.4f})"
        )
        results["ensemble_calibrated_val"] = cal_ensemble_val
        results["ensemble_calibrated_test"] = cal_ensemble_test
        calibration_applied = True
    else:
        print(
            f"\n  Calibration NOT applied — no improvement"
            f"  (val log-loss {pre_ll:.4f} → {post_ll:.4f},  Δ={post_ll - pre_ll:+.4f})"
        )
        calibration_applied = False

    results["calibration_applied"] = calibration_applied

    # ------------------------------------------------------------------
    # Persist all metrics to baseline_results.json
    # ------------------------------------------------------------------
    with baseline_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nUpdated metrics saved to: {baseline_path}")

    # ------------------------------------------------------------------
    # Subphase 5.9 — Model Serialisation
    # ------------------------------------------------------------------
    print("\n=== Subphase 5.9 — Model Serialisation ===")

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models_to_save = [
        ("outcome_lr.pkl",        lr_model,     "classifier"),
        ("outcome_rf.pkl",        rf_model,     "classifier"),
        ("outcome_xgb.pkl",       xgb_model,    "classifier"),
        ("home_goals_xgb.pkl",    xgb_home,     "regressor"),
        ("away_goals_xgb.pkl",    xgb_away,     "regressor"),
        ("home_goals_poisson.pkl", poisson_home, "regressor"),
        ("away_goals_poisson.pkl", poisson_away, "regressor"),
    ]

    for filename, model, kind in models_to_save:
        dest = _MODELS_DIR / filename
        joblib.dump(model, dest)
        loaded = joblib.load(dest)
        sample = X_val.iloc[:1]
        if kind == "classifier":
            loaded.predict_proba(sample)
        else:
            loaded.predict(sample)
        print(f"  Saved & verified: models/{filename}")

    print("\n=== Subphases 5.6–5.9 complete ===")


if __name__ == "__main__":
    main()
