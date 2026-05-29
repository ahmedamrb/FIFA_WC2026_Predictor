"""Fit temperature scaling on the validation split (WC 2022) and persist the result.

Loads the three trained base classifiers, assembles the soft-voting ensemble
(using ensemble weights from baseline_results.json if available, else uniform),
fits a single-parameter temperature on the val-set raw probabilities, and writes
data/processed/temperature.json.

Usage:
    python scripts/fit_temperature.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import log_loss

# ---------------------------------------------------------------------------
# Make src/ importable when running from any working directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.models.outcome_model import load_splits
from src.models.ensemble import WC2026Ensemble, TemperatureScaling

_MODELS_DIR = _REPO_ROOT / "models"
_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"


def main():
    # ------------------------------------------------------------------
    # 1. Load trained base classifiers
    # ------------------------------------------------------------------
    print("Loading trained models from models/ ...")
    lr_model  = joblib.load(_MODELS_DIR / "outcome_lr.pkl")
    rf_model  = joblib.load(_MODELS_DIR / "outcome_rf.pkl")
    xgb_model = joblib.load(_MODELS_DIR / "outcome_xgb.pkl")
    print("  outcome_lr.pkl, outcome_rf.pkl, outcome_xgb.pkl loaded.")

    # ------------------------------------------------------------------
    # 2. Load validation split (WC 2022 rows)
    # ------------------------------------------------------------------
    print("\nLoading validation split via load_splits() ...")
    _, _, _, X_val, y_val, _, _, _, _ = load_splits()
    print(f"  X_val shape: {X_val.shape}, y_val size: {len(y_val)}")

    # ------------------------------------------------------------------
    # 3. Read ensemble weights from baseline_results.json if present
    # ------------------------------------------------------------------
    baseline_path = _PROCESSED_DIR / "baseline_results.json"
    with baseline_path.open("r", encoding="utf-8") as fh:
        baseline = json.load(fh)

    raw_weights = baseline.get("ensemble_weights")
    if raw_weights is not None:
        weights = np.array(raw_weights, dtype=float)
        print(
            f"\nLoaded ensemble weights from baseline_results.json: "
            f"LR={weights[0]:.4f}, RF={weights[1]:.4f}, XGB={weights[2]:.4f}"
        )
    else:
        weights = None
        print("\nNo 'ensemble_weights' key in baseline_results.json — using uniform weights (1/3 each).")

    # ------------------------------------------------------------------
    # 4. Assemble the ensemble
    # ------------------------------------------------------------------
    ensemble = WC2026Ensemble(lr_model, rf_model, xgb_model, weights=weights)
    print(
        f"Ensemble weights: LR={ensemble.weights[0]:.4f}, "
        f"RF={ensemble.weights[1]:.4f}, XGB={ensemble.weights[2]:.4f}"
    )

    # ------------------------------------------------------------------
    # 5. Get raw val probabilities and pre-scaling log-loss
    # ------------------------------------------------------------------
    y_val_arr = y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.array(y_val)
    raw_proba = ensemble.predict_proba(X_val)
    pre_ll = log_loss(y_val_arr, raw_proba)
    print(f"\nPre-scaling val log-loss:  {pre_ll:.6f}")

    # ------------------------------------------------------------------
    # 6. Fit temperature scaling
    # ------------------------------------------------------------------
    ts = TemperatureScaling()
    ts.fit(raw_proba, y_val_arr)
    T = ts.temperature
    print(f"Fitted temperature T:      {T:.6f}")

    cal_proba = ts.calibrate(raw_proba)
    post_ll = log_loss(y_val_arr, cal_proba)
    print(f"Post-scaling val log-loss: {post_ll:.6f}  (Δ={post_ll - pre_ll:+.6f})")

    applied = post_ll <= pre_ll

    # ------------------------------------------------------------------
    # 7. Persist temperature.json
    # ------------------------------------------------------------------
    temp_path = _PROCESSED_DIR / "temperature.json"
    payload = {"temperature": T, "applied": applied}
    with temp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nSaved: {temp_path}")
    print(f"  temperature = {T:.6f}")
    print(f"  applied     = {applied}")

    if applied:
        print(
            f"\n  Temperature scaling APPLIED  "
            f"(val log-loss {pre_ll:.4f} -> {post_ll:.4f},  Delta={post_ll - pre_ll:+.4f})"
        )
    else:
        print(
            f"\n  Temperature scaling NOT applied — no improvement  "
            f"(val log-loss {pre_ll:.4f} -> {post_ll:.4f},  Delta={post_ll - pre_ll:+.4f})"
        )


if __name__ == "__main__":
    main()
