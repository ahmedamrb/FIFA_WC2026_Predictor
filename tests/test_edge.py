"""Unit tests for betting edge computation — Subphase 6.6."""

import pathlib

import pandas as pd


_BACKTEST_WC2022 = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed" / "backtest_wc2022.csv"

_VALID_RECOMMENDATIONS = {"Value", "Neutral", "Avoid"}
_IMPLIED_PROB_COLUMNS = [
    "home_win_implied_prob",
    "draw_implied_prob",
    "away_win_implied_prob",
]


def test_edge_recommendation_valid_values():
    df = pd.read_csv(_BACKTEST_WC2022)
    assert "bet_recommendation" in df.columns, "Column 'bet_recommendation' not found in backtest_wc2022.csv"
    assert df["bet_recommendation"].notna().all(), "Column 'bet_recommendation' contains null values"
    invalid = set(df["bet_recommendation"].unique()) - _VALID_RECOMMENDATIONS
    assert not invalid, f"Unexpected bet_recommendation values: {invalid}"


def test_implied_prob_in_range():
    df = pd.read_csv(_BACKTEST_WC2022)
    for col in _IMPLIED_PROB_COLUMNS:
        assert col in df.columns, f"Column '{col}' not found in backtest_wc2022.csv"
        assert (df[col] > 0).all(), f"Column '{col}' contains values <= 0"
        assert (df[col] <= 1.0).all(), f"Column '{col}' contains values > 1.0"
