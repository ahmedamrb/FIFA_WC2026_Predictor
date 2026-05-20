"""Evaluation metrics for the FIFA WC 2026 Predictor.

Functions:
    plot_calibration_curves: Reliability diagram for multi-class outcome models.
"""

import matplotlib
matplotlib.use("Agg")  # headless backend — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.calibration import calibration_curve

_PLOTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "plots"


def plot_calibration_curves(model, X_val, y_val, label: str) -> float:
    """Plot reliability diagrams for all 3 outcome classes and save to disk.

    For each class (Away Win, Draw, Home Win), computes the calibration curve
    (fraction of positives vs mean predicted probability) using one-vs-rest
    encoding, then overlays all 3 curves on a single axes with the perfect-
    calibration diagonal.

    Uses n_bins=5 with strategy='quantile' to handle the small 64-row
    validation set safely (uniform binning at higher counts produces empty bins
    that raise errors).

    Args:
        model: Any object with a `predict_proba(X)` method returning an array
               of shape (n_samples, 3). Works with both sklearn estimators and
               WC2026Ensemble.
        X_val: Feature DataFrame used as input to predict_proba.
        y_val: True outcome labels (0=Away Win, 1=Draw, 2=Home Win).
        label: String appended to the output filename, e.g. "rf_tuned" or
               "ensemble".

    Returns:
        max_ece: Maximum mean-squared ECE across the three classes (float).
                 Caller can use this to decide whether calibration is needed.
    """
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    proba = model.predict_proba(X_val)
    y_arr = np.asarray(y_val)

    class_config = [
        (0, "Away Win", "tab:red"),
        (1, "Draw", "tab:blue"),
        (2, "Home Win", "tab:green"),
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")

    max_ece = 0.0
    for class_idx, class_name, color in class_config:
        y_binary = (y_arr == class_idx).astype(int)
        fraction_pos, mean_pred = calibration_curve(
            y_binary,
            proba[:, class_idx],
            n_bins=5,
            strategy="quantile",
        )
        ax.plot(mean_pred, fraction_pos, marker="o", color=color, label=class_name)
        class_ece = float(np.mean((fraction_pos - mean_pred) ** 2))
        if class_ece > max_ece:
            max_ece = class_ece

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curves — {label}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    out_path = _PLOTS_DIR / f"calibration_{label}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(
        f"  Calibration plot saved: outputs/plots/calibration_{label}.png"
        f"  (max ECE={max_ece:.4f})"
    )
    return max_ece
