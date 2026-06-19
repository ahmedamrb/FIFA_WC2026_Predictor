"""Betting edge computation utilities for FIFA WC 2026 Predictor.

Edge is the difference between the model's probability for an outcome and the
*vig-free* implied probability derived from bookmaker odds.  Removing the
bookmaker margin (overround) before comparing is essential: raw ``1/odds``
implied probabilities sum to ~1.05–1.10 on a 3-way market, which would
otherwise depress every edge and bias the favourite–longshot comparison.
"""

import numpy as np
import pandas as pd

# Value-bet thresholds on ``best_edge`` (vig-free).  ``VALUE_THRESHOLD`` is the
# single source of truth for the dashboard and backtest; re-tuned by
# scripts/tune_value_threshold.py against the real-odds backtest.
VALUE_THRESHOLD = 0.05
AVOID_THRESHOLD = -0.05


def remove_vig(home_odds, draw_odds, away_odds):
    """Convert decimal 1X2 odds to vig-free ("fair") implied probabilities.

    Uses the proportional (multiplicative) method: take the raw implied
    probabilities ``1/odds`` and divide each by their sum (the overround) so the
    three fair probabilities sum to 1.0, stripping the bookmaker margin.

    Accepts scalars or array-likes (e.g. pandas Series).  Any fixture missing a
    leg, or carrying a non-finite / sub-1.0 decimal odd, yields ``NaN`` for all
    three fair probabilities for that fixture — a 3-way market cannot be
    normalised from an incomplete book, and a missing price must never
    fabricate an edge.

    Args:
        home_odds, draw_odds, away_odds: Decimal odds (scalar or array-like).

    Returns:
        Tuple ``(fair_home, fair_draw, fair_away)``.  Floats for scalar input,
        ``numpy`` arrays for array-like input.
    """
    h = np.asarray(home_odds, dtype="float64")
    d = np.asarray(draw_odds, dtype="float64")
    a = np.asarray(away_odds, dtype="float64")

    scalar_input = h.ndim == 0
    h, d, a = np.atleast_1d(h), np.atleast_1d(d), np.atleast_1d(a)

    valid = (
        (h >= 1.0) & (d >= 1.0) & (a >= 1.0)
        & np.isfinite(h) & np.isfinite(d) & np.isfinite(a)
    )
    # Substitute a safe placeholder for invalid odds so the division never warns;
    # the result is masked back to NaN immediately afterwards.
    safe_h = np.where(valid, h, 1.0)
    safe_d = np.where(valid, d, 1.0)
    safe_a = np.where(valid, a, 1.0)

    imp_h = np.where(valid, 1.0 / safe_h, np.nan)
    imp_d = np.where(valid, 1.0 / safe_d, np.nan)
    imp_a = np.where(valid, 1.0 / safe_a, np.nan)
    booksum = imp_h + imp_d + imp_a

    fair_h = imp_h / booksum
    fair_d = imp_d / booksum
    fair_a = imp_a / booksum

    if scalar_input:
        return float(fair_h[0]), float(fair_d[0]), float(fair_a[0])
    return fair_h, fair_d, fair_a


def compute_edge(backtest_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """Merge bookmaker odds onto backtest results and compute betting edge per match.

    Args:
        backtest_df: DataFrame with columns ``match_date``, ``home_team``, ``away_team``,
            ``predicted_home_win_prob``, ``predicted_draw_prob``, ``predicted_away_win_prob``
            plus any additional columns produced by the backtest pipeline.
        odds_df: DataFrame with columns ``match_date``, ``home_team``, ``away_team``,
            ``home_win_odds``, ``draw_odds``, ``away_win_odds`` (decimal odds).

    Returns:
        Enriched DataFrame with the following new columns appended:
        - ``home_win_implied_prob``, ``draw_implied_prob``, ``away_win_implied_prob``:
          *vig-free* implied probabilities (see :func:`remove_vig`).  ``NaN`` when no
          real odds matched the fixture.
        - ``home_win_edge``, ``draw_edge``, ``away_win_edge``:
          model probability minus vig-free implied probability (``NaN`` when no odds).
        - ``best_edge``: maximum edge across the three outcomes (``NaN`` when no odds).
        - ``value_outcome``: ``"Home Win"`` / ``"Draw"`` / ``"Away Win"`` — the leg
          carrying ``best_edge`` (``pd.NA`` when no odds).
        - ``bet_recommendation``: ``"Value"`` when ``best_edge > VALUE_THRESHOLD``,
          ``"Neutral"`` when ``AVOID_THRESHOLD <= best_edge <= VALUE_THRESHOLD``,
          ``"Avoid"`` when ``best_edge < AVOID_THRESHOLD``, and ``"No Odds"`` when the
          fixture has no real bookmaker odds (never flagged as a bet).

        The raw odds columns (``home_win_odds``, ``draw_odds``, ``away_win_odds``) are
        dropped from the result; implied probability columns serve as the final
        representation of bookmaker pricing.

    Notes:
        Missing odds are deliberately **not** defaulted to 2.0.  Fabricating a price
        for a fixture without real odds produces spurious "Value" flags, which is a
        primary cause of poorly-performing value bets.
    """
    odds_copy = odds_df.copy()
    odds_copy["match_date"] = odds_copy["match_date"].astype(str)

    result = backtest_df.copy()
    result["match_date"] = result["match_date"].astype(str)

    result = result.merge(
        odds_copy[["match_date", "home_team", "away_team", "home_win_odds", "draw_odds", "away_win_odds"]],
        on=["match_date", "home_team", "away_team"],
        how="left",
    )

    # Vig-free implied probabilities. Unmatched fixtures -> NaN (no 2.0 default),
    # so an absent price can never manufacture an edge.
    fair_home, fair_draw, fair_away = remove_vig(
        result["home_win_odds"], result["draw_odds"], result["away_win_odds"]
    )
    result["home_win_implied_prob"] = fair_home
    result["draw_implied_prob"] = fair_draw
    result["away_win_implied_prob"] = fair_away

    result["home_win_edge"] = result["predicted_home_win_prob"] - result["home_win_implied_prob"]
    result["draw_edge"] = result["predicted_draw_prob"] - result["draw_implied_prob"]
    result["away_win_edge"] = result["predicted_away_win_prob"] - result["away_win_implied_prob"]

    edge_cols = ["home_win_edge", "draw_edge", "away_win_edge"]
    result["best_edge"] = result[edge_cols].max(axis=1)

    # Name the outcome carrying the best edge (only for rows that actually have odds).
    outcome_by_col = {
        "home_win_edge": "Home Win",
        "draw_edge": "Draw",
        "away_win_edge": "Away Win",
    }
    has_edge = result["best_edge"].notna()
    result["value_outcome"] = pd.NA
    if has_edge.any():
        result.loc[has_edge, "value_outcome"] = (
            result.loc[has_edge, edge_cols].idxmax(axis=1).map(outcome_by_col)
        )

    # Recommendation. Fixtures without real odds are "No Odds" and are never flagged.
    result["bet_recommendation"] = "No Odds"
    decided = result["best_edge"].notna()
    result.loc[decided, "bet_recommendation"] = "Avoid"
    result.loc[decided & (result["best_edge"] >= AVOID_THRESHOLD), "bet_recommendation"] = "Neutral"
    result.loc[decided & (result["best_edge"] > VALUE_THRESHOLD), "bet_recommendation"] = "Value"

    result = result.drop(columns=["home_win_odds", "draw_odds", "away_win_odds"])

    return result
