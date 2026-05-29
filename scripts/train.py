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
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.models.goals_model import evaluate_goals_model, train_tuned_goals_models
from src.models.outcome_model import evaluate_model, load_splits, train_tuned_models
from src.models.ensemble import WC2026Ensemble, optimize_ensemble_weights
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from src.evaluation.metrics import plot_calibration_curves

_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _generate_oof_preds(X_train, y_train, w_train, best_params, n_splits=5):
    """Generate out-of-fold predictions for LR, RF, and XGBoost.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training outcome Series.
        w_train: numpy array of per-sample training weights.
        best_params: Dict with 'rf_outcome' and 'xgb_outcome' keys.
        n_splits: Number of stratified CV folds.

    Returns:
        List of 3 numpy arrays, each (n_train, 3), for [LR, RF, XGB].
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n = len(X_train)
    oof = [np.zeros((n, 3)) for _ in range(3)]  # [lr, rf, xgb]
    X_arr = X_train.values
    y_arr = y_train.to_numpy()
    rp = best_params["rf_outcome"]
    xp = best_params["xgb_outcome"]

    for fold_num, (tr_idx, va_idx) in enumerate(cv.split(X_arr, y_arr), 1):
        X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
        y_tr = y_arr[tr_idx]
        w_tr = w_train[tr_idx]

        # LR
        lr_fold = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
        ])
        lr_fold.fit(X_tr, y_tr, lr__sample_weight=w_tr)
        oof[0][va_idx] = lr_fold.predict_proba(X_va)

        # RF
        rf_fold = RandomForestClassifier(
            n_estimators=rp["n_estimators"],
            max_features=rp["max_features"],
            min_samples_split=rp["min_samples_split"],
            min_samples_leaf=rp["min_samples_leaf"],
            max_depth=rp["max_depth"],
            random_state=42,
        )
        rf_fold.fit(X_tr, y_tr, sample_weight=w_tr)
        oof[1][va_idx] = rf_fold.predict_proba(X_va)

        # XGB
        xgb_fold = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=xp["n_estimators"],
            max_depth=xp["max_depth"],
            learning_rate=xp["learning_rate"],
            subsample=xp["subsample"],
            colsample_bytree=xp["colsample_bytree"],
            reg_alpha=xp["reg_alpha"],
            reg_lambda=xp["reg_lambda"],
            random_state=42,
        )
        xgb_fold.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
        oof[2][va_idx] = xgb_fold.predict_proba(X_va)

        print(f"  OOF fold {fold_num}/{n_splits} done")

    return oof


def main():
    """Train all tuned models, evaluate, and persist updated metrics."""
    print("=== Subphase 5.6 — Train Tuned Final Models ===\n")

    # ------------------------------------------------------------------
    # Load data splits
    # ------------------------------------------------------------------
    X_train, y_train, w_train, X_val, y_val, X_test, y_test, _val_df, _test_df = load_splits()

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
    # Train tuned outcome models (with sample weights)
    # ------------------------------------------------------------------
    lr_model, rf_model, xgb_model = train_tuned_models(
        X_train, y_train, best_params, sample_weight=w_train
    )

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
    # Subphase 5.7 — Learn ensemble weights from OOF predictions
    # ------------------------------------------------------------------
    print("\n=== Subphase 5.7 — Learning Ensemble Weights from OOF Predictions ===")
    print("  Generating 5-fold OOF predictions (takes ~1–2 min) ...")
    oof_preds = _generate_oof_preds(X_train, y_train, w_train, best_params)
    learned_weights = optimize_ensemble_weights(oof_preds, y_train)
    print(
        f"  Learned weights: LR={learned_weights[0]:.4f}, "
        f"RF={learned_weights[1]:.4f}, XGB={learned_weights[2]:.4f}"
    )

    # ------------------------------------------------------------------
    # Assemble and evaluate soft-voting ensemble with learned weights
    # ------------------------------------------------------------------
    print("\n--- Ensemble Evaluation ---")
    ensemble = WC2026Ensemble(lr_model, rf_model, xgb_model, weights=learned_weights)
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

    # Reassemble ensemble with calibrated XGBoost and same learned weights
    cal_ensemble = WC2026Ensemble(lr_model, rf_model, cal_xgb, weights=learned_weights)

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
    # Temperature Scaling (single-parameter calibration, robust on small val)
    # ------------------------------------------------------------------
    from src.models.ensemble import TemperatureScaling, TemperatureScaledEnsemble

    _ts = TemperatureScaling()
    _raw_val_proba = ensemble.predict_proba(X_val)
    _ts.fit(_raw_val_proba, y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.array(y_val))
    print(f"\n  Temperature scaling fit: T = {_ts.temperature:.4f}")

    _ts_ensemble = TemperatureScaledEnsemble(ensemble, _ts.temperature)
    from sklearn.metrics import log_loss as _log_loss
    _ts_pre_ll = _log_loss(y_val, _raw_val_proba)
    _ts_post_proba = _ts_ensemble.predict_proba(X_val)
    _ts_post_ll = _log_loss(y_val, _ts_post_proba)

    if _ts_post_ll <= _ts_pre_ll:
        print(
            f"  Temperature scaling APPLIED  "
            f"(val log-loss {_ts_pre_ll:.4f} \u2192 {_ts_post_ll:.4f},  \u0394={_ts_post_ll - _ts_pre_ll:+.4f})"
        )
        results["temperature"] = _ts.temperature
        results["temperature_applied"] = True
    else:
        print(
            f"  Temperature scaling NOT applied \u2014 no improvement  "
            f"(val log-loss {_ts_pre_ll:.4f} \u2192 {_ts_post_ll:.4f},  \u0394={_ts_post_ll - _ts_pre_ll:+.4f})"
        )
        results["temperature"] = 1.0
        results["temperature_applied"] = False

    # ------------------------------------------------------------------
    # Persist all metrics to baseline_results.json
    # ------------------------------------------------------------------
    with baseline_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nUpdated metrics saved to: {baseline_path}")

    # Save temperature value for dashboard consumption
    import json as _json_temp
    _temp_path = _PROCESSED_DIR / "temperature.json"
    _json_temp.dump(
        {"temperature": results.get("temperature", 1.0), "applied": results.get("temperature_applied", False)},
        _temp_path.open("w")
    )
    print(f"  Saved: {_temp_path}  (T={results.get('temperature', 1.0):.4f}, applied={results.get('temperature_applied', False)})")

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
