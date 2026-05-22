"""Backtest runner for FIFA WC 2026 Predictor.

Loads trained models, retrieves val/test splits, and runs backtests for
WC 2022 (val) and WC 2018 (test).  Saves CSVs to data/processed/.

Usage:
    python scripts/run_backtest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.models.outcome_model import load_splits
from src.models.ensemble import WC2026Ensemble
from src.evaluation.backtest import run_backtest

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _compute_brier(df: pd.DataFrame) -> float:
    """Compute mean Brier score from a backtest results DataFrame."""
    proba = df[["predicted_away_win_prob", "predicted_draw_prob", "predicted_home_win_prob"]].values
    actual = df["actual_outcome"].values
    n = len(actual)
    y_onehot = np.zeros_like(proba)
    y_onehot[np.arange(n), actual] = 1
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1) / 3))


def main() -> None:
    """Load models and run backtest on WC 2022 (val) and WC 2018 (test)."""
    print("=== Subphase 6.2 — Run Backtests for WC 2022 and WC 2018 ===\n")

    # ------------------------------------------------------------------
    # Load splits (9-tuple)
    # ------------------------------------------------------------------
    X_train, y_train, w_train, X_val, y_val, X_test, y_test, val_df, test_df = load_splits()

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("\nLoading models...")
    lr = joblib.load(_MODELS_DIR / "outcome_lr.pkl")
    rf = joblib.load(_MODELS_DIR / "outcome_rf.pkl")
    xgb = joblib.load(_MODELS_DIR / "outcome_xgb.pkl")
    home_goals = joblib.load(_MODELS_DIR / "home_goals_xgb.pkl")
    away_goals = joblib.load(_MODELS_DIR / "away_goals_xgb.pkl")
    print("  All 5 models loaded.")

    ensemble = WC2026Ensemble(lr, rf, xgb)

    # ------------------------------------------------------------------
    # Pull goals targets and metadata from the full parquet (aligned by index)
    # ------------------------------------------------------------------
    features_df = pd.read_parquet(_PROCESSED_DIR / "features_train.parquet")

    # Val — WC 2022
    print("\n--- Val: WC 2022 ---")
    wc2022_df = run_backtest(
        ensemble=ensemble,
        goals_home_model=home_goals,
        goals_away_model=away_goals,
        X=X_val,
        y_outcome=y_val,
        y_home=features_df.loc[X_val.index, "home_score"],
        y_away=features_df.loc[X_val.index, "away_score"],
        match_dates=val_df["date"],
        home_teams=val_df["home_team"],
        away_teams=val_df["away_team"],
        label="wc2022",
    )

    # Test — WC 2018
    print("\n--- Test: WC 2018 ---")
    wc2018_df = run_backtest(
        ensemble=ensemble,
        goals_home_model=home_goals,
        goals_away_model=away_goals,
        X=X_test,
        y_outcome=y_test,
        y_home=features_df.loc[X_test.index, "home_score"],
        y_away=features_df.loc[X_test.index, "away_score"],
        match_dates=test_df["date"],
        home_teams=test_df["home_team"],
        away_teams=test_df["away_team"],
        label="wc2018",
    )

    # ------------------------------------------------------------------
    # Side-by-side summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SIDE-BY-SIDE SUMMARY")
    print("=" * 60)

    metrics = {}
    for label, df in [("WC 2022 (val)", wc2022_df), ("WC 2018 (test)", wc2018_df)]:
        proba_cols = ["predicted_away_win_prob", "predicted_draw_prob", "predicted_home_win_prob"]
        proba = df[proba_cols].values
        actual = df["actual_outcome"].values
        ll = log_loss(actual, proba, labels=[0, 1, 2])
        acc = accuracy_score(actual, df["predicted_outcome"].values)
        brier = _compute_brier(df)
        metrics[label] = {"log_loss": ll, "accuracy": acc, "brier": brier, "rows": len(df)}

    header = f"  {'Metric':<20} {'WC 2022 (val)':>16} {'WC 2018 (test)':>16}"
    print(header)
    print(f"  {'-' * 54}")
    print(f"  {'Matches':<20} {metrics['WC 2022 (val)']['rows']:>16} {metrics['WC 2018 (test)']['rows']:>16}")
    print(f"  {'Log-loss':<20} {metrics['WC 2022 (val)']['log_loss']:>16.4f} {metrics['WC 2018 (test)']['log_loss']:>16.4f}")
    print(f"  {'Accuracy':<20} {metrics['WC 2022 (val)']['accuracy']:>16.4f} {metrics['WC 2018 (test)']['accuracy']:>16.4f}")
    print(f"  {'Brier Score':<20} {metrics['WC 2022 (val)']['brier']:>16.4f} {metrics['WC 2018 (test)']['brier']:>16.4f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Post-run validation
    # ------------------------------------------------------------------
    print("\nPost-run validation:")
    for label, df, path_label in [
        ("WC 2022", wc2022_df, "wc2022"),
        ("WC 2018", wc2018_df, "wc2018"),
    ]:
        csv_path = _PROCESSED_DIR / f"backtest_{path_label}.csv"
        loaded = pd.read_csv(csv_path)
        null_count = loaded.isnull().sum().sum()
        prob_sums = (
            loaded["predicted_home_win_prob"]
            + loaded["predicted_draw_prob"]
            + loaded["predicted_away_win_prob"]
        ).round(6)
        all_sum_to_one = (prob_sums == 1.0).all()
        print(
            f"  [{label}] rows={len(loaded)} | nulls={null_count} | "
            f"prob_sums_to_1={all_sum_to_one}"
        )

    print("\n=== Subphase 6.2 complete ===")


if __name__ == "__main__":
    main()

