"""Evaluation metrics for the FIFA WC 2026 Predictor.

Functions:
    plot_calibration_curves: Reliability diagram for multi-class outcome models.
"""

import json
import math
import matplotlib
matplotlib.use("Agg")  # headless backend — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss as sk_log_loss

_PLOTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "plots"
_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


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


def compile_final_metrics(
    wc2018_df,
    wc2022_df,
    wc2018_stub=None,
    wc2022_stub=None,
):
    """Assemble final backtest metrics for both tournaments and save to JSON.

    Computes log-loss, accuracy, Brier score, flat-stake ROI, and value-bet
    ROI for each tournament from enriched backtest DataFrames.  When stub
    DataFrames are supplied the output includes a nested ``"comparison"``
    section showing the delta between the ``"real"`` and ``"stub_2.0"`` modes.

    Args:
        wc2018_df: Enriched backtest DataFrame for WC 2018 (test set, real odds).
        wc2022_df: Enriched backtest DataFrame for WC 2022 (val set, real odds).
        wc2018_stub: Optional stub-odds DataFrame for WC 2018 (all odds = 2.0).
        wc2022_stub: Optional stub-odds DataFrame for WC 2022 (all odds = 2.0).

    Returns:
        dict with keys ``'wc2018'`` and ``'wc2022'``, each mapping to a
        sub-dict with keys: ``log_loss``, ``accuracy``, ``brier_score``,
        ``flat_stake_roi``, ``value_bet_roi``.  When stubs are provided,
        a ``'comparison'`` key is added at the top level.  ROI values are
        percentages (float).  ``value_bet_roi`` is ``None`` when no Value
        bets exist.
    """
    def _brier(df):
        proba = df[
            ["predicted_away_win_prob", "predicted_draw_prob", "predicted_home_win_prob"]
        ].values
        actual = df["actual_outcome"].values
        n = len(actual)
        y_onehot = np.zeros_like(proba)
        y_onehot[np.arange(n), actual] = 1
        return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1) / 3))

    def _per_tournament_metrics(df) -> dict:
        proba_cols = [
            "predicted_away_win_prob",
            "predicted_draw_prob",
            "predicted_home_win_prob",
        ]
        proba = df[proba_cols].values
        actual = df["actual_outcome"].values

        ll = float(sk_log_loss(actual, proba, labels=[0, 1, 2]))
        acc = float(accuracy_score(actual, df["predicted_outcome"].values))
        brier = _brier(df)
        flat_roi = float(df["profit"].sum() / len(df) * 100.0)

        value_df = df[df["bet_recommendation"] == "Value"] if "bet_recommendation" in df.columns else df.iloc[0:0]
        value_roi: float | None = None
        if len(value_df) > 0:
            value_roi = float(value_df["profit"].sum() / len(value_df) * 100.0)

        odds_matched: int | None = None
        if "odds_matched" in df.columns:
            odds_matched = int(df["odds_matched"].sum())

        result = {
            "log_loss": round(ll, 6),
            "accuracy": round(acc, 6),
            "brier_score": round(brier, 6),
            "flat_stake_roi": round(flat_roi, 4),
            "value_bet_roi": round(value_roi, 4) if value_roi is not None else None,
        }
        if odds_matched is not None:
            result["odds_matched"] = odds_matched
            result["odds_defaulted"] = len(df) - odds_matched
        return result

    summary = {}
    for key, df in [("wc2018", wc2018_df), ("wc2022", wc2022_df)]:
        summary[key] = _per_tournament_metrics(df)

    # ------------------------------------------------------------------
    # Comparison section (real vs stub_2.0)
    # ------------------------------------------------------------------
    comparison: dict = {}
    if wc2018_stub is not None and wc2022_stub is not None:
        stub_summary: dict = {}
        for key, df in [("wc2018", wc2018_stub), ("wc2022", wc2022_stub)]:
            stub_summary[key] = _per_tournament_metrics(df)

        for key in ("wc2018", "wc2022"):
            real = summary[key]
            stub = stub_summary[key]
            delta_roi = (
                (real["flat_stake_roi"] - stub["flat_stake_roi"])
                if real["flat_stake_roi"] is not None and stub["flat_stake_roi"] is not None
                else None
            )
            delta_value_roi = (
                (real["value_bet_roi"] - stub["value_bet_roi"])
                if real["value_bet_roi"] is not None and stub["value_bet_roi"] is not None
                else None
            )
            comparison[key] = {
                "real": real,
                "stub_2.0": stub,
                "delta": {
                    "flat_stake_roi": round(delta_roi, 4) if delta_roi is not None else None,
                    "value_bet_roi": round(delta_value_roi, 4) if delta_value_roi is not None else None,
                },
            }
        summary["comparison"] = comparison

    # ------------------------------------------------------------------
    # Save to JSON
    # ------------------------------------------------------------------
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _PROCESSED_DIR / "final_backtest_metrics.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Saved -> data/processed/final_backtest_metrics.json")

    # ------------------------------------------------------------------
    # Print formatted summary table (real-odds rows)
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  FINAL BACKTEST METRICS SUMMARY  (real odds)")
    print("=" * 65)
    print(f"  {'Metric':<24} {'WC 2018 (test)':>17} {'WC 2022 (val)':>17}")
    print(f"  {'-' * 60}")

    _ROI_KEYS = {"flat_stake_roi", "value_bet_roi"}
    for metric_key, label in [
        ("log_loss", "Log-loss"),
        ("accuracy", "Accuracy"),
        ("brier_score", "Brier Score"),
        ("flat_stake_roi", "Flat-stake ROI (%)"),
        ("value_bet_roi", "Value-bet ROI (%)"),
        ("odds_matched", "Odds matched"),
        ("odds_defaulted", "Odds defaulted (2.0)"),
    ]:
        v18 = summary["wc2018"].get(metric_key)
        v22 = summary["wc2022"].get(metric_key)
        if v18 is None and v22 is None:
            continue

        def _fmt(v, is_roi):
            if v is None:
                return "N/A"
            if isinstance(v, int):
                return str(v)
            return f"{v:+.2f}" if is_roi else f"{v:.4f}"

        is_roi = metric_key in _ROI_KEYS
        print(f"  {label:<24} {_fmt(v18, is_roi):>17} {_fmt(v22, is_roi):>17}")

    # If comparison data exists, print the delta section
    if comparison:
        print("\n" + "=" * 65)
        print("  ROI DELTA  (real − stub_2.0)")
        print("=" * 65)
        print(f"  {'Metric':<30} {'WC 2018':>14} {'WC 2022':>14}")
        print(f"  {'-' * 60}")
        for metric_key, label in [
            ("flat_stake_roi", "Flat-stake ROI Δ (pp)"),
            ("value_bet_roi", "Value-bet ROI Δ (pp)"),
        ]:
            d18 = comparison.get("wc2018", {}).get("delta", {}).get(metric_key)
            d22 = comparison.get("wc2022", {}).get("delta", {}).get(metric_key)

            def _dfmt(v):
                return "N/A" if v is None else f"{v:+.2f}"

            print(f"  {label:<30} {_dfmt(d18):>14} {_dfmt(d22):>14}")

    print("=" * 65)

    return summary
