"""Walk-forward backtesting utilities for FIFA WC 2026 Predictor.

Functions:
    run_backtest: Generate predictions for a split, assemble results, compute metrics.
    simulate_betting: Flat-stake betting simulation with bookmaker odds lookup.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
_PLOTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "plots"
_ODDS_PATH = Path(__file__).resolve().parents[2] / "data" / "bookmaker_odds.csv"

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

    # Use floor() rather than round() — for a Poisson(λ) model the mode is
    # floor(λ) (for non-integer λ), so floor gives the most-likely integer
    # outcome and correctly predicts 0 when λ < 1.
    pred_home_goals = np.clip(
        np.floor(goals_home_model.predict(X)).astype(int), a_min=0, a_max=None
    )
    pred_away_goals = np.clip(
        np.floor(goals_away_model.predict(X)).astype(int), a_min=0, a_max=None
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
    print(f"  Saved -> data/processed/backtest_{label}.csv")

    return results_df


def simulate_betting(
    backtest_df: pd.DataFrame,
    label: str,
    odds_col: str = "bookmaker_odds_used",
) -> pd.DataFrame:
    """Simulate flat-stake (1 unit) betting on each match in a backtest DataFrame.

    Looks up decimal odds from ``data/bookmaker_odds.csv`` matched by
    ``match_date``, ``home_team``, and ``away_team``.  Unmatched rows default
    to 2.0.  Bets are placed on the outcome with the highest predicted
    probability (``predicted_outcome``).

    Args:
        backtest_df: DataFrame returned by :func:`run_backtest`, containing at
            least the columns ``match_date``, ``home_team``, ``away_team``,
            ``predicted_outcome``, and ``actual_outcome``.
        label: Short identifier used in the plot filename
            (e.g. ``"wc2022"`` or ``"wc2018"``).
        odds_col: Name to assign to the bookmaker-odds output column.

    Returns:
        A copy of *backtest_df* with four additional columns:
        ``bookmaker_odds_used``, ``payout``, ``profit``, ``cumulative_profit``.
    """
    df = backtest_df.copy()

    # ------------------------------------------------------------------
    # Load odds and merge on (match_date, home_team, away_team)
    # ------------------------------------------------------------------
    if _ODDS_PATH.exists():
        odds_df = pd.read_csv(_ODDS_PATH, parse_dates=["match_date"])
        odds_df["match_date"] = pd.to_datetime(odds_df["match_date"]).dt.normalize()
    else:
        odds_df = pd.DataFrame(
            columns=["match_date", "home_team", "away_team",
                     "home_win_odds", "draw_odds", "away_win_odds"]
        )

    df["match_date"] = pd.to_datetime(df["match_date"]).dt.normalize()

    df = df.merge(
        odds_df[["match_date", "home_team", "away_team",
                 "home_win_odds", "draw_odds", "away_win_odds"]],
        on=["match_date", "home_team", "away_team"],
        how="left",
    )

    # ------------------------------------------------------------------
    # Pick odds for the predicted outcome; default to 2.0 when not found
    # ------------------------------------------------------------------
    # predicted_outcome: 0 = away win, 1 = draw, 2 = home win
    outcome_to_col = {0: "away_win_odds", 1: "draw_odds", 2: "home_win_odds"}

    def _pick_odds(row: pd.Series) -> float:
        col = outcome_to_col.get(int(row["predicted_outcome"]), "home_win_odds")
        val = row.get(col, float("nan"))
        if pd.isna(val) or val < 1.0:
            return 2.0
        return float(val)

    df[odds_col] = df.apply(_pick_odds, axis=1)

    # Ensure no nulls remain (belt-and-suspenders)
    df[odds_col] = df[odds_col].fillna(2.0)

    # ------------------------------------------------------------------
    # Compute payout, profit, cumulative_profit
    # ------------------------------------------------------------------
    correct = (df["predicted_outcome"] == df["actual_outcome"]).astype(int)
    df["payout"] = df[odds_col] * correct
    df["profit"] = df["payout"] - 1.0
    df["cumulative_profit"] = df["profit"].cumsum()

    # ------------------------------------------------------------------
    # ROI
    # ------------------------------------------------------------------
    total_stake = len(df) * 1.0  # 1 unit per match
    total_profit = df["profit"].sum()
    roi = total_profit / total_stake * 100.0

    n_matched = (~df[["home_win_odds", "draw_odds", "away_win_odds"]].isna().all(axis=1)).sum()
    n_defaulted = len(df) - n_matched
    print(
        f"  [{label}] Betting: total_stake={total_stake:.0f} | "
        f"total_profit={total_profit:+.2f} | ROI={roi:+.2f}% | "
        f"odds_matched={n_matched} defaulted={n_defaulted}"
    )

    # ------------------------------------------------------------------
    # Cumulative profit chart
    # ------------------------------------------------------------------
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(df) + 1), df["cumulative_profit"], marker="o", markersize=3,
            linewidth=1.5, color="#1f77b4")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(f"Cumulative Profit — {label.upper()} (Flat Stake 1 Unit)")
    ax.set_xlabel("Match Number")
    ax.set_ylabel("Cumulative Profit (units)")
    ax.grid(True, alpha=0.3)
    plot_path = _PLOTS_DIR / f"cumulative_profit_{label}.png"
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [{label}] Chart saved -> outputs/plots/cumulative_profit_{label}.png")

    # Drop the raw odds columns (keep only the derived ones)
    df.drop(
        columns=["home_win_odds", "draw_odds", "away_win_odds"],
        errors="ignore",
        inplace=True,
    )

    return df

