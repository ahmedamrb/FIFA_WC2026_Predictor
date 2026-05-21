"""Walk-forward backtesting utilities for FIFA WC 2026 Predictor.

Functions:
    run_backtest: Generate predictions for a split, assemble results, compute metrics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

STAGE_LABELS = {
    0: "Non-WC",
    1: "Group",
    2: "R32",
    3: "R16",
    4: "QF",
    5: "SF/3rd",
    6: "Final",
}


def run_backtest(
    ensemble,
    goals_home_model,
    goals_away_model,
    X: pd.DataFrame,
    y_outcome: pd.Series,
    y_home: pd.Series,
    y_away: pd.Series,
    match_dates: pd.Series,
    home_teams: pd.Series,
    away_teams: pd.Series,
    label: str,
) -> pd.DataFrame:
    """Generate predictions for a data split and compute backtesting metrics.

    Produces outcome probabilities and scoreline predictions for each match,
    computes overall log-loss / accuracy / Brier score, prints a per-stage
    accuracy breakdown, saves a CSV to ``data/processed/``, and returns the
    results DataFrame.

    Args:
        ensemble: Fitted ensemble with ``predict_proba`` and ``predict`` methods.
            ``predict_proba`` must return an array of shape ``(n, 3)`` with
            column order ``[away_win=0, draw=1, home_win=2]``.
        goals_home_model: Fitted regressor for predicted home goals.
        goals_away_model: Fitted regressor for predicted away goals.
        X: Feature DataFrame aligned to the split (index matches y_outcome).
        y_outcome: Actual outcome Series (0=away win, 1=draw, 2=home win).
        y_home: Actual home goals Series.
        y_away: Actual away goals Series.
        match_dates: Series of match dates aligned to X.
        home_teams: Series of home team names aligned to X.
        away_teams: Series of away team names aligned to X.
        label: Short identifier used in printed output and the output filename
            (e.g. ``"val_wc2022"`` or ``"test_wc2018"``).

    Returns:
        DataFrame with 12 columns:
        ``match_date``, ``home_team``, ``away_team``,
        ``predicted_home_win_prob``, ``predicted_draw_prob``, ``predicted_away_win_prob``,
        ``predicted_outcome``, ``predicted_home_goals``, ``predicted_away_goals``,
        ``actual_outcome``, ``actual_home_goals``, ``actual_away_goals``.
    """
    n = len(X)

    # ------------------------------------------------------------------
    # Generate predictions
    # ------------------------------------------------------------------
    proba = ensemble.predict_proba(X)  # shape (n, 3): [away=0, draw=1, home=2]
    pred_outcome = ensemble.predict(X)  # shape (n,)

    pred_home_goals = np.clip(
        np.round(goals_home_model.predict(X)).astype(int), a_min=0, a_max=None
    )
    pred_away_goals = np.clip(
        np.round(goals_away_model.predict(X)).astype(int), a_min=0, a_max=None
    )

    # ------------------------------------------------------------------
    # Assemble results DataFrame
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(
        {
            "match_date": match_dates.values,
            "home_team": home_teams.values,
            "away_team": away_teams.values,
            "predicted_home_win_prob": proba[:, 2],
            "predicted_draw_prob": proba[:, 1],
            "predicted_away_win_prob": proba[:, 0],
            "predicted_outcome": pred_outcome,
            "predicted_home_goals": pred_home_goals,
            "predicted_away_goals": pred_away_goals,
            "actual_outcome": y_outcome.values,
            "actual_home_goals": y_home.values,
            "actual_away_goals": y_away.values,
        }
    )

    # ------------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------------
    ll = log_loss(y_outcome.values, proba, labels=[0, 1, 2])
    acc = accuracy_score(y_outcome.values, pred_outcome)

    y_onehot = np.zeros_like(proba)
    y_onehot[np.arange(n), y_outcome.values] = 1
    brier = float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1) / 3))

    print(f"[{label}] log-loss={ll:.4f} | accuracy={acc:.4f} | brier={brier:.4f}")

    # ------------------------------------------------------------------
    # Per-stage accuracy breakdown
    # ------------------------------------------------------------------
    stage_col = X["tournament_stage"].fillna(0).astype(int)
    stage_df = pd.DataFrame(
        {
            "stage": stage_col.values,
            "correct": (pred_outcome == y_outcome.values).astype(int),
        }
    )

    print(f"\n  {'Stage':<12} {'Matches':>8} {'Accuracy':>10}")
    print(f"  {'-' * 32}")
    for stage_val, group in stage_df.groupby("stage"):
        stage_name = STAGE_LABELS.get(int(stage_val), f"Stage {stage_val}")
        n_matches = len(group)
        stage_acc = group["correct"].mean()
        print(f"  {stage_name:<12} {n_matches:>8} {stage_acc:>10.4f}")
    print()

    # ------------------------------------------------------------------
    # Save CSV (12 main columns only)
    # ------------------------------------------------------------------
    output_cols = [
        "match_date", "home_team", "away_team",
        "predicted_home_win_prob", "predicted_draw_prob", "predicted_away_win_prob",
        "predicted_outcome", "predicted_home_goals", "predicted_away_goals",
        "actual_outcome", "actual_home_goals", "actual_away_goals",
    ]
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _PROCESSED_DIR / f"backtest_{label}.csv"
    results_df[output_cols].to_csv(csv_path, index=False)
    print(f"  Saved → data/processed/backtest_{label}.csv")

    return results_df

