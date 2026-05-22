"""Betting edge computation utilities for FIFA WC 2026 Predictor."""

import pandas as pd


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
          implied probabilities derived from decimal odds (``1 / odds``).
        - ``home_win_edge``, ``draw_edge``, ``away_win_edge``:
          difference between model probability and implied probability.
        - ``best_edge``: maximum edge value across all three outcomes for each row.
        - ``bet_recommendation``: ``"Value"`` when ``best_edge > 0.05``,
          ``"Neutral"`` when ``-0.05 <= best_edge <= 0.05``, ``"Avoid"`` otherwise.

        The raw odds columns (``home_win_odds``, ``draw_odds``, ``away_win_odds``) are
        dropped from the result; implied probability columns serve as the final
        representation of bookmaker pricing.
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

    result["home_win_odds"] = result["home_win_odds"].fillna(2.0)
    result["draw_odds"] = result["draw_odds"].fillna(2.0)
    result["away_win_odds"] = result["away_win_odds"].fillna(2.0)

    result["home_win_implied_prob"] = 1.0 / result["home_win_odds"]
    result["draw_implied_prob"] = 1.0 / result["draw_odds"]
    result["away_win_implied_prob"] = 1.0 / result["away_win_odds"]

    result["home_win_edge"] = result["predicted_home_win_prob"] - result["home_win_implied_prob"]
    result["draw_edge"] = result["predicted_draw_prob"] - result["draw_implied_prob"]
    result["away_win_edge"] = result["predicted_away_win_prob"] - result["away_win_implied_prob"]

    result["best_edge"] = result[["home_win_edge", "draw_edge", "away_win_edge"]].max(axis=1)

    result["bet_recommendation"] = "Avoid"
    result.loc[result["best_edge"] >= -0.05, "bet_recommendation"] = "Neutral"
    result.loc[result["best_edge"] > 0.05, "bet_recommendation"] = "Value"

    result = result.drop(columns=["home_win_odds", "draw_odds", "away_win_odds"])

    return result
