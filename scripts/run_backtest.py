"""Backtest runner for FIFA WC 2026 Predictor.

Loads trained models, retrieves val/test splits, and runs backtests for
WC 2022 (val) and WC 2018 (test).  Saves CSVs to data/processed/.

Two odds modes are executed back-to-back so the impact of real odds versus
the synthetic 2.0 baseline can be compared directly:

    stub_2.0    — all unmatched fixtures use decimal odds of 2.0 (baseline)
    real        — uses data/bookmaker_odds.csv plus historical archive CSVs
                  for WC 2018 and WC 2022 when they are populated

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
from src.data.odds import load_odds_for_backtest

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
_RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

_HISTORICAL_ODDS_PATHS = [
    _RAW_DIR / "historical_odds_wc2018.csv",
    _RAW_DIR / "historical_odds_wc2022.csv",
]


def _compute_brier(df: pd.DataFrame) -> float:
    """Compute mean Brier score from a backtest results DataFrame."""
    proba = df[["predicted_away_win_prob", "predicted_draw_prob", "predicted_home_win_prob"]].values
    actual = df["actual_outcome"].values
    n = len(actual)
    y_onehot = np.zeros_like(proba)
    y_onehot[np.arange(n), actual] = 1
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1) / 3))


def _print_model_summary(wc2022_df: pd.DataFrame, wc2018_df: pd.DataFrame) -> None:
    """Print side-by-side model quality metrics (log-loss, accuracy, Brier)."""
    print("\n" + "=" * 62)
    print("  MODEL QUALITY — independent of odds source")
    print("=" * 62)
    metrics = {}
    for label, df in [("WC 2022 (val)", wc2022_df), ("WC 2018 (test)", wc2018_df)]:
        proba_cols = ["predicted_away_win_prob", "predicted_draw_prob", "predicted_home_win_prob"]
        proba = df[proba_cols].values
        actual = df["actual_outcome"].values
        metrics[label] = {
            "log_loss": log_loss(actual, proba, labels=[0, 1, 2]),
            "accuracy": accuracy_score(actual, df["predicted_outcome"].values),
            "brier": _compute_brier(df),
            "rows": len(df),
        }
    print(f"  {'Metric':<20} {'WC 2022 (val)':>16} {'WC 2018 (test)':>16}")
    print(f"  {'-' * 54}")
    for k, lbl in [("rows", "Matches"), ("log_loss", "Log-loss"),
                   ("accuracy", "Accuracy"), ("brier", "Brier Score")]:
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
        print(
            f"  {lbl:<20} "
            f"{fmt(metrics['WC 2022 (val)'][k]):>16} "
            f"{fmt(metrics['WC 2018 (test)'][k]):>16}"
        )
    print("=" * 62)


def _print_odds_comparison(results_by_mode: dict) -> None:
    """Print a side-by-side odds-mode comparison table."""
    print("\n" + "=" * 82)
    print("  ODDS COMPARISON: stub_2.0  vs  real")
    print("=" * 82)

    tournaments = ["WC 2022 (val)", "WC 2018 (test)", "Combined"]
    header = f"  {'Metric':<26}"
    for t in tournaments:
        header += f"  {'stub_2.0':>10}  {'real':>10}"
    print(f"  {'Metric':<26}" +
          "".join(f"  {t:<22}" for t in tournaments))
    print(
        f"  {'':26}" +
        "  stub_2.0      real  " * len(tournaments)
    )
    print(f"  {'-' * 78}")

    def _get(mode: str, tournament: str, key: str):
        return results_by_mode.get(mode, {}).get(tournament, {}).get(key, "n/a")

    for key, label in [
        ("flat_roi", "Flat-stake ROI (%)"),
        ("value_roi", "Value-bet ROI (%)"),
        ("value_count", "Value bets"),
        ("matched", "Odds matched"),
        ("defaulted", "Odds defaulted (2.0)"),
        ("total_profit", "Total profit (units)"),
    ]:
        row = f"  {label:<26}"
        for t in tournaments:
            stub_val = _get("stub_2.0", t, key)
            real_val = _get("real", t, key)

            def _fmt(v):
                if v == "n/a":
                    return "n/a"
                if isinstance(v, float):
                    return f"{v:+.2f}" if "roi" in key or "profit" in key else f"{v:.2f}"
                return str(v)

            row += f"  {_fmt(stub_val):>10}  {_fmt(real_val):>10}"
        print(row)
    print("=" * 82)


def _run_mode(
    label_22: str,
    label_18: str,
    wc2022_df: pd.DataFrame,
    wc2018_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    odds_mode: str,
    odds_col: str = "bookmaker_odds_used",
) -> tuple:
    """Run simulate_betting + compute_edge for both tournaments under one odds mode."""
    wc2022_bet = simulate_betting(wc2022_df, label=label_22, odds_df=odds_df, odds_mode=odds_mode)
    wc2018_bet = simulate_betting(wc2018_df, label=label_18, odds_df=odds_df, odds_mode=odds_mode)

    # Pass odds_df directly to compute_edge; empty DataFrame with ODDS_COLUMNS schema
    # is fine — all merges produce NaN which fillna(2.0) covers.
    odds_for_edge = odds_df if odds_df is not None else pd.DataFrame(columns=["match_date", "home_team", "away_team", "home_win_odds", "draw_odds", "away_win_odds"])
    wc2022_edge = compute_edge(wc2022_bet, odds_for_edge)
    wc2018_edge = compute_edge(wc2018_bet, odds_for_edge)
    if odds_mode == "real":
        wc2022_edge.to_csv(_PROCESSED_DIR / "backtest_wc2022.csv", index=False)
        wc2018_edge.to_csv(_PROCESSED_DIR / "backtest_wc2018.csv", index=False)
        print(f"  [{odds_mode}] Enriched backtest CSVs saved.")

    return wc2022_edge, wc2018_edge


def _collect_stats(df: pd.DataFrame, odds_mode: str) -> dict:
    """Collect betting + coverage stats from an enriched backtest DataFrame."""
    total_profit = float(df["profit"].sum())
    n = len(df)
    # ROI is profit per unit staked (stake-aware: skipped/Kelly stakes differ from 1.0).
    total_stake = float(df["stake"].sum()) if "stake" in df.columns else float(n)
    flat_roi = total_profit / total_stake * 100.0 if total_stake > 0 else float("nan")

    value_df = df[df["bet_recommendation"] == "Value"] if "bet_recommendation" in df.columns else df.iloc[0:0]
    value_count = len(value_df)
    value_stake = float(value_df["stake"].sum()) if "stake" in value_df.columns else float(value_count)
    value_roi = float(value_df["profit"].sum() / value_stake * 100.0) if value_stake > 0 else float("nan")

    matched = int(df["odds_matched"].sum()) if "odds_matched" in df.columns else (0 if odds_mode == "stub_2.0" else n)
    defaulted = n - matched

    return {
        "flat_roi": flat_roi,
        "value_roi": value_roi,
        "value_count": value_count,
        "matched": matched,
        "defaulted": defaulted,
        "total_profit": total_profit,
    }


def main() -> None:
    """Load models, run backtests in two odds modes, and print a comparison."""
    print("=== Backtest Runner — Two-Mode Odds Comparison ===\n")

    # ------------------------------------------------------------------
    # Load splits (9-tuple)
    # ------------------------------------------------------------------
    X_train, y_train, w_train, X_val, y_val, X_test, y_test, val_df, test_df = load_splits()

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("Loading models...")
    lr = joblib.load(_MODELS_DIR / "outcome_lr.pkl")
    rf = joblib.load(_MODELS_DIR / "outcome_rf.pkl")
    xgb = joblib.load(_MODELS_DIR / "outcome_xgb.pkl")
    home_goals = joblib.load(_MODELS_DIR / "home_goals_xgb.pkl")
    away_goals = joblib.load(_MODELS_DIR / "away_goals_xgb.pkl")
    print("  All 5 models loaded.")

    ensemble = WC2026Ensemble(lr, rf, xgb)

    features_df = pd.read_parquet(_PROCESSED_DIR / "features_train.parquet")

    # ------------------------------------------------------------------
    # Generate predictions (once — same for both odds modes)
    # ------------------------------------------------------------------
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

    _print_model_summary(wc2022_df, wc2018_df)

    # ------------------------------------------------------------------
    # Load odds for each mode
    # ------------------------------------------------------------------
    stub_odds = load_odds_for_backtest(mode="stub_2.0")
    real_odds = load_odds_for_backtest(
        mode="real",
        historical_paths=_HISTORICAL_ODDS_PATHS,
    )

    real_n = len(real_odds)
    print(f"\n  Odds table: {real_n} rows loaded for 'real' mode.")
    if real_n > 0:
        hist_rows = real_odds[real_odds["source"].str.startswith("archive:", na=False)].shape[0]
        live_rows = real_n - hist_rows
        print(f"    Live rows  : {live_rows}")
        print(f"    Archive rows: {hist_rows}")

    # ------------------------------------------------------------------
    # Mode 1: stub_2.0 baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  MODE: stub_2.0 (all odds = 2.0 baseline)")
    print("=" * 50)
    wc2022_stub, wc2018_stub = _run_mode(
        "wc2022", "wc2018", wc2022_df, wc2018_df,
        stub_odds, odds_mode="stub_2.0",
    )

    # ------------------------------------------------------------------
    # Mode 2: real odds
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  MODE: real (canonical odds table + historical archives)")
    print("=" * 50)
    wc2022_real, wc2018_real = _run_mode(
        "wc2022", "wc2018", wc2022_df, wc2018_df,
        real_odds, odds_mode="real",
    )

    # ------------------------------------------------------------------
    # Build comparison stats
    # ------------------------------------------------------------------
    results_by_mode: dict = {"stub_2.0": {}, "real": {}}
    for mode, df22, df18 in [
        ("stub_2.0", wc2022_stub, wc2018_stub),
        ("real", wc2022_real, wc2018_real),
    ]:
        s22 = _collect_stats(df22, mode)
        s18 = _collect_stats(df18, mode)
        combined_profit = s22["total_profit"] + s18["total_profit"]
        combined_n = len(df22) + len(df18)
        combined_value_n = s22["value_count"] + s18["value_count"]
        combined_value_profit = (
            df22[df22.get("bet_recommendation", pd.Series(dtype=str)) == "Value"]["profit"].sum()
            + df18[df18.get("bet_recommendation", pd.Series(dtype=str)) == "Value"]["profit"].sum()
        ) if "bet_recommendation" in df22.columns else 0.0
        results_by_mode[mode]["WC 2022 (val)"] = s22
        results_by_mode[mode]["WC 2018 (test)"] = s18
        results_by_mode[mode]["Combined"] = {
            "flat_roi": combined_profit / combined_n * 100.0,
            "value_roi": (
                combined_value_profit / combined_value_n * 100.0
                if combined_value_n > 0 else float("nan")
            ),
            "value_count": combined_value_n,
            "matched": s22["matched"] + s18["matched"],
            "defaulted": s22["defaulted"] + s18["defaulted"],
            "total_profit": combined_profit,
        }

    _print_odds_comparison(results_by_mode)

    # ------------------------------------------------------------------
    # Post-run validation
    # ------------------------------------------------------------------
    print("\nPost-run validation:")
    for label_str, df, path_label in [
        ("WC 2022", wc2022_df, "wc2022"),
        ("WC 2018", wc2018_df, "wc2018"),
    ]:
        csv_path = _PROCESSED_DIR / f"backtest_{path_label}.csv"
        if not csv_path.exists():
            print(f"  [{label_str}] CSV not saved yet (real mode run needed).")
            continue
        loaded = pd.read_csv(csv_path)
        null_count = loaded.isnull().sum().sum()
        prob_sums = (
            loaded["predicted_home_win_prob"]
            + loaded["predicted_draw_prob"]
            + loaded["predicted_away_win_prob"]
        ).round(6)
        all_sum_to_one = (prob_sums == 1.0).all()
        print(
            f"  [{label_str}] rows={len(loaded)} | nulls={null_count} | "
            f"prob_sums_to_1={all_sum_to_one}"
        )

    # ------------------------------------------------------------------
    # Print recommendation counts (real mode)
    # ------------------------------------------------------------------
    print("\n--- Bet Recommendations (real mode) ---")
    for label_str, df in [("WC 2022", wc2022_real), ("WC 2018", wc2018_real)]:
        if "bet_recommendation" in df.columns:
            counts = df["bet_recommendation"].value_counts().to_dict()
            print(
                f"  [{label_str}] "
                f"Value={counts.get('Value', 0)} | "
                f"Neutral={counts.get('Neutral', 0)} | "
                f"Avoid={counts.get('Avoid', 0)}"
            )

    # ------------------------------------------------------------------
    # Final metrics summary (real mode — used by dashboard)
    # ------------------------------------------------------------------
    print("\n--- Final Metrics Summary ---")
    compile_final_metrics(
        wc2018_real, wc2022_real,
        wc2018_stub=wc2018_stub, wc2022_stub=wc2022_stub,
    )

    print("\n=== Backtest complete ===")


if __name__ == "__main__":
    main()

