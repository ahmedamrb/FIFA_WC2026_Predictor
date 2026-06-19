"""Walk-forward backtesting utilities for FIFA WC 2026 Predictor.

Functions:
    run_backtest: Generate predictions for a split, assemble results, compute metrics.
    simulate_betting: Flat-stake betting simulation with bookmaker odds lookup.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.betting.staking import kelly_fraction

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
    odds_df: Optional[pd.DataFrame] = None,
    odds_mode: str = "real",
    odds_col: str = "bookmaker_odds_used",
    staking: str = "flat",
    kelly_frac: float = 0.25,
    kelly_cap: float = 0.05,
) -> pd.DataFrame:
    """Simulate betting on each match in a backtest DataFrame.

    Odds are supplied via the *odds_df* parameter (injected explicitly by the
    caller) rather than loaded internally.

    Two staking schemes:

    * ``staking="flat"`` — 1 unit per bet (the original behaviour).
    * ``staking="kelly"`` — fractional Kelly: stake = ``kelly_fraction`` of
      bankroll on the predicted outcome, sized by the model probability and the
      odds (see :func:`src.betting.staking.kelly_fraction`).

    In ``odds_mode="real"`` fixtures **without** a matched real odds row are
    skipped (stake 0) so ROI reflects only genuine prices.  In
    ``odds_mode="stub_2.0"`` every match is staked at the 2.0 baseline, giving a
    "what if everything were even money" reference.

    Args:
        backtest_df: DataFrame returned by :func:`run_backtest`, containing at
            least the columns ``match_date``, ``home_team``, ``away_team``,
            ``predicted_outcome``, ``actual_outcome``, and (for Kelly) the
            ``predicted_*_prob`` columns.
        label: Short identifier used in printed output and the plot filename
            (e.g. ``"wc2022"`` or ``"wc2018"``).
        odds_df: External odds DataFrame with columns ``match_date``,
            ``home_team``, ``away_team``, ``home_win_odds``, ``draw_odds``,
            ``away_win_odds``.  Pass ``None`` or an empty DataFrame to use only
            the 2.0 baseline (same as ``odds_mode='stub_2.0'``).
        odds_mode: ``"real"`` or ``"stub_2.0"``.  Controls whether unmatched
            fixtures are skipped (real) or staked at 2.0 (stub).
        odds_col: Name to assign to the bookmaker-odds output column.
        staking: ``"flat"`` or ``"kelly"``.
        kelly_frac: Kelly multiplier when ``staking="kelly"`` (0.25 = quarter).
        kelly_cap: Maximum stake fraction per bet when ``staking="kelly"``.

    Returns:
        A copy of *backtest_df* with these additional columns:
        ``bookmaker_odds_used``, ``odds_source``, ``odds_matched``, ``stake``,
        ``payout``, ``profit``, ``cumulative_profit``.
    """
    df = backtest_df.copy()

    # ------------------------------------------------------------------
    # Merge supplied odds onto backtest rows
    # ------------------------------------------------------------------
    if odds_df is not None and not odds_df.empty:
        odds_work = odds_df.copy()
        odds_work["match_date"] = pd.to_datetime(
            odds_work["match_date"], errors="coerce"
        ).dt.normalize()
        merge_cols = ["match_date", "home_team", "away_team",
                      "home_win_odds", "draw_odds", "away_win_odds"]
        # Keep source for provenance if present
        if "source" in odds_work.columns:
            merge_cols.append("source")
        df["match_date"] = pd.to_datetime(df["match_date"]).dt.normalize()
        df = df.merge(
            odds_work[merge_cols].drop_duplicates(
                subset=["match_date", "home_team", "away_team"]
            ),
            on=["match_date", "home_team", "away_team"],
            how="left",
        )
    else:
        # No external odds supplied — ensure columns exist with NaN
        df["home_win_odds"] = float("nan")
        df["draw_odds"] = float("nan")
        df["away_win_odds"] = float("nan")

    # ------------------------------------------------------------------
    # Track coverage: was a real odds row matched for this fixture?
    # ------------------------------------------------------------------
    odds_available = df["home_win_odds"].notna()
    df["odds_matched"] = odds_available
    df["odds_source"] = df.get("source", pd.Series([None] * len(df)))
    df["odds_source"] = df["odds_source"].where(odds_available, other="stub_2.0")

    # ------------------------------------------------------------------
    # Pick odds for the predicted outcome; fall back to 2.0 for gaps
    # ------------------------------------------------------------------
    # predicted_outcome: 0 = away win, 1 = draw, 2 = home win
    outcome_to_col = {0: "away_win_odds", 1: "draw_odds", 2: "home_win_odds"}

    def _pick_odds(row: pd.Series) -> float:
        col = outcome_to_col.get(int(row["predicted_outcome"]), "home_win_odds")
        val = row.get(col, float("nan"))
        if pd.isna(val) or float(val) < 1.0:
            return 2.0
        return float(val)

    df[odds_col] = df.apply(_pick_odds, axis=1)

    # ------------------------------------------------------------------
    # Determine the stake for each match
    # ------------------------------------------------------------------
    # In real mode, skip fixtures with no matched odds (stake 0) so ROI reflects
    # only genuine prices.  In stub mode every match is staked at the baseline.
    skip_unmatched = odds_mode == "real"
    prob_cols = {
        0: "predicted_away_win_prob",
        1: "predicted_draw_prob",
        2: "predicted_home_win_prob",
    }

    def _stake(row: pd.Series) -> float:
        if skip_unmatched and not bool(row["odds_matched"]):
            return 0.0
        if staking == "kelly":
            col = prob_cols.get(int(row["predicted_outcome"]))
            prob = float(row.get(col, 0.0)) if col else 0.0
            return kelly_fraction(prob, float(row[odds_col]), fraction=kelly_frac, cap=kelly_cap)
        return 1.0

    df["stake"] = df.apply(_stake, axis=1)

    # ------------------------------------------------------------------
    # Compute payout, profit, cumulative_profit (stake-aware)
    # ------------------------------------------------------------------
    correct = (df["predicted_outcome"] == df["actual_outcome"]).astype(int)
    df["payout"] = df["stake"] * df[odds_col] * correct
    df["profit"] = df["payout"] - df["stake"]
    df["cumulative_profit"] = df["profit"].cumsum()

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    total_stake = float(df["stake"].sum())
    total_profit = df["profit"].sum()
    roi = total_profit / total_stake * 100.0 if total_stake > 0 else float("nan")
    n_matched = int(odds_available.sum())
    n_defaulted = len(df) - n_matched
    n_bets = int((df["stake"] > 0).sum())

    print(
        f"  [{label}|{odds_mode}|{staking}] Betting: bets={n_bets} "
        f"stake={total_stake:.2f} | profit={total_profit:+.2f} | ROI={roi:+.2f}% | "
        f"matched={n_matched} defaulted={n_defaulted}"
    )

    # ------------------------------------------------------------------
    # Cumulative profit chart
    # ------------------------------------------------------------------
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    mode_suffix = f"_{odds_mode}" if odds_mode != "real" else ""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        range(1, len(df) + 1), df["cumulative_profit"],
        marker="o", markersize=3, linewidth=1.5, color="#1f77b4",
    )
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"Cumulative Profit — {label.upper()} ({odds_mode}, Flat Stake 1 Unit)"
    )
    ax.set_xlabel("Match Number")
    ax.set_ylabel("Cumulative Profit (units)")
    ax.grid(True, alpha=0.3)
    plot_path = _PLOTS_DIR / f"cumulative_profit_{label}{mode_suffix}.png"
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [{label}] Chart saved -> outputs/plots/cumulative_profit_{label}{mode_suffix}.png")

    # Drop merged raw odds columns; keep derived provenance columns
    df.drop(
        columns=["home_win_odds", "draw_odds", "away_win_odds", "source"],
        errors="ignore",
        inplace=True,
    )

    return df

