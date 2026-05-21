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
import pandas as pd

from src.models.outcome_model import load_splits
from src.models.ensemble import WC2026Ensemble
from src.evaluation.backtest import run_backtest

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main() -> None:
    """Load models and run backtest on WC 2022 (val) and WC 2018 (test)."""
    print("=== Subphase 6.1 — Backtest ===\n")

    # ------------------------------------------------------------------
    # Load splits (9-tuple)
    # ------------------------------------------------------------------
    X_train, y_train, w_train, X_val, y_val, X_test, y_test, val_df, test_df = load_splits()

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    lr = joblib.load(_MODELS_DIR / "outcome_lr.pkl")
    rf = joblib.load(_MODELS_DIR / "outcome_rf.pkl")
    xgb = joblib.load(_MODELS_DIR / "outcome_xgb.pkl")
    home_goals = joblib.load(_MODELS_DIR / "home_goals_xgb.pkl")
    away_goals = joblib.load(_MODELS_DIR / "away_goals_xgb.pkl")

    ensemble = WC2026Ensemble(lr, rf, xgb)

    # ------------------------------------------------------------------
    # Pull goals targets and metadata from the full parquet (aligned by index)
    # ------------------------------------------------------------------
    features_df = pd.read_parquet(_PROCESSED_DIR / "features_train.parquet")

    # Val — WC 2022
    print("\n--- Val: WC 2022 ---")
    run_backtest(
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
    run_backtest(
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

    print("\n=== Subphase 6.1 complete ===")


if __name__ == "__main__":
    main()

