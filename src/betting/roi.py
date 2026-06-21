"""Actual ROI of the dashboard's best-value bets, settled against real results.

This module replays exactly the value-bet logic the prediction card surfaces —
the best edge vs *vig-free* implied probability, flagged only when it clears
``VALUE_THRESHOLD`` and is priced off real bookmaker odds, staked with the same
fractional-Kelly amount the card displays — and settles each bet against the
finished scoreline.

Each settled bet returns ``stake * (odds - 1)`` on a win or ``-stake`` on a loss.
Stakes are converted to dollars off a starting bankroll under one of two modes:

- ``"flat"`` — every bet is sized off the *fixed* starting bankroll (path
  independent; the cleanest read on the model's edge).
- ``"compound"`` — winnings are reinvested: each bet is sized off the *current*
  bankroll, settling bets in chronological order (maximises growth, adds
  variance).

Reported figures include the realised yield ``ROI = net profit / total staked``,
the bankroll ``growth = net profit / starting bankroll`` and the ending bankroll.
Only matches that have finished with a known score are settled; value bets on
fixtures still to be played are reported separately as ``pending``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.betting.edge import VALUE_THRESHOLD, remove_vig
from src.betting.staking import kelly_fraction

# Outcome order matches the (home, draw, away) ordering used everywhere else.
_OUTCOMES = ["Home Win", "Draw", "Away Win"]
_PROB_COLS = ["prob_home_win", "prob_draw", "prob_away_win"]
_ODDS_COLS = ["home_win_odds", "draw_odds", "away_win_odds"]

# One row per value bet produced by settle_value_bets().
LEDGER_COLUMNS = [
    "fixture_id",
    "match_date",
    "home_team",
    "away_team",
    "value_outcome",
    "edge",
    "odds",
    "stake",
    "settled",
    "won",
    "profit",
    "actual_outcome",
]

# Extra dollar columns bankroll_curve() appends to the settled ledger.
CURVE_COLUMNS = LEDGER_COLUMNS + ["stake_dollars", "pl_dollars", "bankroll"]


def _empty_summary(initial_bankroll: float, pending: int = 0) -> dict:
    """Summary dict for the case where no value bets have settled."""
    return {
        "n_bets": 0,
        "won": 0,
        "win_pct": 0.0,
        "staked": 0.0,
        "profit": 0.0,
        "roi": 0.0,
        "growth": 0.0,
        "initial": float(initial_bankroll),
        "final": float(initial_bankroll),
        "pending": int(pending),
    }


def settle_value_bets(
    comparison_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None,
    odds_df: pd.DataFrame | None,
    value_threshold: float = VALUE_THRESHOLD,
) -> pd.DataFrame:
    """Build the value-bet ledger for the WC 2026 fixtures.

    For every comparable fixture that carries real bookmaker odds, this picks
    the best-value outcome (largest model − vig-free implied edge), keeps it only
    when the edge clears *value_threshold* and the displayed fractional-Kelly
    stake is positive, then — for fixtures that have finished — settles it
    against the actual result.

    Args:
        comparison_df: Output of
            :func:`src.evaluation.live_tracking.build_comparison`.  Its index
            must align with *predictions_df* (both row-aligned to the fixtures
            CSV), and it must carry ``home_team``, ``away_team``, ``match_date``,
            ``comparable``, ``played``, ``has_score`` and ``actual_outcome``.
        predictions_df: Predictions table with ``prob_home_win`` / ``prob_draw``
            / ``prob_away_win`` columns, index-aligned to *comparison_df*.
        odds_df: Canonical odds table (``ODDS_COLUMNS`` schema), already
            deduplicated to the latest snapshot per fixture.
        value_threshold: Minimum vig-free edge for a leg to count as value.

    Returns:
        A DataFrame with the columns in :data:`LEDGER_COLUMNS` — one row per
        value bet.  ``settled`` is True for finished matches (``won``/``profit``
        populated); False for value bets still pending (``won``/``profit`` NA).
        Empty when no inputs or no value bets exist.
    """
    if (
        comparison_df is None
        or comparison_df.empty
        or predictions_df is None
        or odds_df is None
        or odds_df.empty
        or not all(c in predictions_df.columns for c in _PROB_COLS)
    ):
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    # Attach model probabilities by index (predictions are row-aligned to fixtures).
    comp = comparison_df.join(predictions_df[_PROB_COLS])

    # Latest real odds per fixture, joined on a normalised date + team pair so
    # the same fixture matches regardless of Timestamp vs "YYYY-MM-DD" formatting.
    od = odds_df.copy()
    od = od[od[_ODDS_COLS].notna().all(axis=1)]
    od["_d"] = pd.to_datetime(od["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    od = od.drop_duplicates(subset=["_d", "home_team", "away_team"], keep="first")

    comp = comp.copy()
    comp["_d"] = pd.to_datetime(comp["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged = comp.merge(
        od[["_d", "home_team", "away_team", *_ODDS_COLS]],
        on=["_d", "home_team", "away_team"],
        how="left",
    )

    rows = []
    for _, r in merged.iterrows():
        # No real odds -> never fabricate a value bet (mirrors the card).
        if not bool(r.get("comparable", False)) or pd.isna(r.get("home_win_odds")):
            continue

        probs = [r["prob_home_win"], r["prob_draw"], r["prob_away_win"]]
        odds = [r["home_win_odds"], r["draw_odds"], r["away_win_odds"]]
        if any(pd.isna(p) for p in probs):
            continue

        fair = remove_vig(odds[0], odds[1], odds[2])
        if any(pd.isna(f) for f in fair):
            continue

        edges = [float(probs[i] - fair[i]) for i in range(3)]
        best_idx = int(np.argmax(edges))
        best_edge = edges[best_idx]
        if best_edge <= value_threshold:
            continue  # not a value bet

        best_odds = float(odds[best_idx])
        stake = kelly_fraction(probs[best_idx], best_odds)
        if stake <= 0.0:
            continue  # displayed stake is 0% -> no money placed

        value_outcome = _OUTCOMES[best_idx]
        actual = r.get("actual_outcome")
        settled = (
            bool(r.get("played", False))
            and bool(r.get("has_score", False))
            and not pd.isna(actual)
        )

        won = profit = pd.NA
        if settled:
            won = bool(actual == value_outcome)
            profit = stake * (best_odds - 1.0) if won else -stake

        rows.append(
            {
                "fixture_id": r.get("fixture_id"),
                "match_date": r.get("match_date"),
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "value_outcome": value_outcome,
                "edge": best_edge,
                "odds": best_odds,
                "stake": float(stake),
                "settled": settled,
                "won": won,
                "profit": profit,
                "actual_outcome": actual if not pd.isna(actual) else pd.NA,
            }
        )

    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def bankroll_curve(
    ledger: pd.DataFrame,
    mode: str = "flat",
    initial_bankroll: float = 1000.0,
) -> pd.DataFrame:
    """Translate the settled value-bet ledger into a dollar bankroll trajectory.

    Settles bets in chronological order and converts each fractional-Kelly stake
    into dollars:

    - ``"flat"`` — every stake is sized off the *fixed* ``initial_bankroll``.
    - ``"compound"`` — each stake is sized off the *current* bankroll, so
      winnings are reinvested and losses shrink later stakes.

    Args:
        ledger: Output of :func:`settle_value_bets`.
        mode: ``"flat"`` or ``"compound"``.
        initial_bankroll: Starting bankroll in dollars.

    Returns:
        The settled bets (chronological) with the columns in
        :data:`CURVE_COLUMNS` — adding ``stake_dollars``, ``pl_dollars`` and the
        running ``bankroll`` after each bet. Empty when nothing has settled.
    """
    if ledger is None or ledger.empty:
        return pd.DataFrame(columns=CURVE_COLUMNS)

    settled = ledger[ledger["settled"].astype(bool)].copy()
    if settled.empty:
        return pd.DataFrame(columns=CURVE_COLUMNS)

    settled["_d"] = pd.to_datetime(settled["match_date"], errors="coerce")
    settled = settled.sort_values(["_d", "fixture_id"]).drop(columns="_d").reset_index(drop=True)

    bankroll = float(initial_bankroll)
    stakes, pls, balances = [], [], []
    for _, r in settled.iterrows():
        reference = float(initial_bankroll) if mode == "flat" else bankroll
        stake_dollars = float(r["stake"]) * reference
        pl = stake_dollars * (float(r["odds"]) - 1.0) if bool(r["won"]) else -stake_dollars
        bankroll += pl
        stakes.append(stake_dollars)
        pls.append(pl)
        balances.append(bankroll)

    settled["stake_dollars"] = stakes
    settled["pl_dollars"] = pls
    settled["bankroll"] = balances
    return settled[CURVE_COLUMNS]


def summarize_roi(
    curve: pd.DataFrame,
    initial_bankroll: float = 1000.0,
    pending: int = 0,
) -> dict:
    """Aggregate a dollar bankroll trajectory into headline ROI figures.

    Args:
        curve: Output of :func:`bankroll_curve`.
        initial_bankroll: Starting bankroll in dollars.
        pending: Count of value bets not yet settled (for display only).

    Returns:
        A dict with keys ``n_bets``, ``won``, ``win_pct``, ``staked`` (dollars),
        ``profit`` (dollars), ``roi`` (profit / staked), ``growth`` (profit /
        starting bankroll), ``initial``, ``final`` (ending bankroll) and
        ``pending``.
    """
    if curve is None or curve.empty:
        return _empty_summary(initial_bankroll, pending)

    n = len(curve)
    won = int(curve["won"].astype(bool).sum())
    staked = float(curve["stake_dollars"].sum())
    final = float(curve["bankroll"].iloc[-1])
    profit = final - float(initial_bankroll)

    return {
        "n_bets": n,
        "won": won,
        "win_pct": (won / n) if n else 0.0,
        "staked": staked,
        "profit": profit,
        "roi": (profit / staked) if staked > 0 else 0.0,
        "growth": (profit / initial_bankroll) if initial_bankroll else 0.0,
        "initial": float(initial_bankroll),
        "final": final,
        "pending": int(pending),
    }


def value_bet_roi(
    comparison_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None,
    odds_df: pd.DataFrame | None,
    mode: str = "flat",
    initial_bankroll: float = 1000.0,
    value_threshold: float = VALUE_THRESHOLD,
) -> dict:
    """Settle every best-value bet and summarise the realised dollar ROI.

    Convenience wrapper over :func:`settle_value_bets` →
    :func:`bankroll_curve` → :func:`summarize_roi`.

    Args:
        comparison_df, predictions_df, odds_df: See :func:`settle_value_bets`.
        mode: ``"flat"`` (size off fixed bankroll) or ``"compound"`` (reinvest).
        initial_bankroll: Starting bankroll in dollars.
        value_threshold: Minimum vig-free edge for a leg to count as value.

    Returns:
        The summary dict described in :func:`summarize_roi`.
    """
    ledger = settle_value_bets(comparison_df, predictions_df, odds_df, value_threshold)
    if ledger.empty:
        return _empty_summary(initial_bankroll, 0)

    pending = int((~ledger["settled"].astype(bool)).sum())
    curve = bankroll_curve(ledger, mode, initial_bankroll)
    return summarize_roi(curve, initial_bankroll, pending)
