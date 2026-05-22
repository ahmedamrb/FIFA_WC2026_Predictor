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
from src.evaluation.backtest import run_backtest, simulate_betting
from src.betting.edge import compute_edge
from src.evaluation.metrics import compile_final_metrics

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
_ODDS_PATH = Path(__file__).resolve().parents[1] / "data" / "bookmaker_odds.csv"


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
    print("=== Subphase 6.2 / 6.3 / 6.4 / 6.5 — Backtests, Simulation, Edge & Metrics ===\n")

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

    # ------------------------------------------------------------------
    # Simulated betting
    # ------------------------------------------------------------------
    print("\n--- Simulated Betting ---")
    wc2022_bet_df = simulate_betting(wc2022_df, label="wc2022")
    wc2018_bet_df = simulate_betting(wc2018_df, label="wc2018")

    # Combined ROI across both tournaments
    combined_profit = wc2022_bet_df["profit"].sum() + wc2018_bet_df["profit"].sum()
    combined_stake = len(wc2022_bet_df) + len(wc2018_bet_df)
    combined_roi = combined_profit / combined_stake * 100.0
    print(f"\n  Combined ROI (WC 2022 + WC 2018): {combined_roi:+.2f}%")
    print(
        f"  (profit={combined_profit:+.2f} over {combined_stake} matches "
        f"@ 1 unit/match)"
    )

    # ------------------------------------------------------------------
    # Betting edge
    # ------------------------------------------------------------------
    print("\n--- Betting Edge (Subphase 6.4) ---")
    odds_df = pd.read_csv(_ODDS_PATH)

    wc2022_edge_df = compute_edge(wc2022_bet_df, odds_df)
    wc2018_edge_df = compute_edge(wc2018_bet_df, odds_df)

    # Re-save enriched CSVs
    wc2022_edge_df.to_csv(_PROCESSED_DIR / "backtest_wc2022.csv", index=False)
    wc2018_edge_df.to_csv(_PROCESSED_DIR / "backtest_wc2018.csv", index=False)
    print("  Enriched backtest CSVs re-saved.")

    # Print recommendation counts
    for label, df in [("WC 2022", wc2022_edge_df), ("WC 2018", wc2018_edge_df)]:
        counts = df["bet_recommendation"].value_counts().to_dict()
        value_n = counts.get("Value", 0)
        neutral_n = counts.get("Neutral", 0)
        avoid_n = counts.get("Avoid", 0)
        print(
            f"  [{label}] Value={value_n} | Neutral={neutral_n} | Avoid={avoid_n}"
        )

    # Combined recommendation counts
    combined_edge_df = pd.concat([wc2022_edge_df, wc2018_edge_df], ignore_index=True)
    total_value = (combined_edge_df["bet_recommendation"] == "Value").sum()
    print(f"\n  Total 'Value' flags (both tournaments): {total_value}")

    # Value-bet ROI
    value_df = combined_edge_df[combined_edge_df["bet_recommendation"] == "Value"]
    if len(value_df) > 0:
        value_profit = value_df["profit"].sum()
        value_roi = value_profit / len(value_df) * 100.0
        print(
            f"  Value-bet ROI: {value_roi:+.2f}%  "
            f"(profit={value_profit:+.2f} over {len(value_df)} matches)"
        )
    else:
        print("  Value-bet ROI: N/A (no Value-flagged matches)")

    # Edge column validation
    edge_cols = ["home_win_edge", "draw_edge", "away_win_edge", "best_edge"]
    for label, df in [("WC 2022", wc2022_edge_df), ("WC 2018", wc2018_edge_df)]:
        all_finite = all(df[c].apply(lambda x: pd.notna(x) and abs(x) < 2.0).all() for c in edge_cols)
        rec_valid = df["bet_recommendation"].isin(["Value", "Neutral", "Avoid"]).all()
        rec_no_null = df["bet_recommendation"].notna().all()
        print(
            f"  [{label}] edge_cols_finite={all_finite} | "
            f"rec_valid={rec_valid} | rec_no_null={rec_no_null}"
        )

    # ------------------------------------------------------------------
    # Final metrics summary
    # ------------------------------------------------------------------
    print("\n--- Final Metrics Summary (Subphase 6.5) ---")
    compile_final_metrics(wc2018_edge_df, wc2022_edge_df)

    print("\n=== Subphase 6.5 complete ===")

if __name__ == "__main__":
    main()

