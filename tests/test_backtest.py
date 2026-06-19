"""Unit tests for backtesting output — Subphase 6.6 + odds-mode comparison extensions."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.evaluation.backtest import simulate_betting
from src.data.odds import load_odds_for_backtest, ODDS_COLUMNS

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BACKTEST_CSV = _REPO_ROOT / "data" / "processed" / "backtest_wc2022.csv"


# ---------------------------------------------------------------------------
# Original tests (unchanged)
# ---------------------------------------------------------------------------

def test_backtest_df_no_nulls():
    # Odds-derived columns (implied probs, edges, value_outcome) are legitimately
    # NaN for fixtures without real odds; only the core prediction/outcome/betting
    # columns must always be populated.
    df = pd.read_csv(_BACKTEST_CSV)
    core_cols = [
        "predicted_home_win_prob", "predicted_draw_prob", "predicted_away_win_prob",
        "predicted_outcome", "actual_outcome", "profit", "cumulative_profit",
    ]
    present = [c for c in core_cols if c in df.columns]
    null_count = df[present].isnull().sum().sum()
    assert null_count == 0, f"Expected zero nulls in core columns, found {null_count}"


def test_probabilities_sum_to_one():
    df = pd.read_csv(_BACKTEST_CSV)
    prob_sums = (
        df["predicted_home_win_prob"]
        + df["predicted_draw_prob"]
        + df["predicted_away_win_prob"]
    )
    assert np.allclose(prob_sums, 1.0, atol=1e-6), (
        "Predicted probabilities do not sum to 1.0 within tolerance 1e-6"
    )


# ---------------------------------------------------------------------------
# New: simulate_betting odds-injection tests
# ---------------------------------------------------------------------------

def _make_minimal_backtest(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "match_date": ["2022-11-20", "2022-11-21", "2022-11-22", "2022-11-23"][:n],
        "home_team": ["Qatar", "England", "Argentina", "France"][:n],
        "away_team": ["Ecuador", "Iran", "Saudi Arabia", "Australia"][:n],
        "predicted_outcome": [2, 2, 2, 2][:n],
        "actual_outcome": [2, 0, 2, 2][:n],
        "predicted_home_win_prob": [0.55, 0.70, 0.65, 0.60][:n],
        "predicted_draw_prob": [0.25, 0.18, 0.20, 0.25][:n],
        "predicted_away_win_prob": [0.20, 0.12, 0.15, 0.15][:n],
    })


def _make_real_odds(match_dates=None, home_teams=None, away_teams=None) -> pd.DataFrame:
    dates = match_dates or ["2022-11-20", "2022-11-21"]
    homes = home_teams or ["Qatar", "England"]
    aways = away_teams or ["Ecuador", "Iran"]
    return pd.DataFrame({
        "match_date": dates,
        "home_team": homes,
        "away_team": aways,
        "home_win_odds": [3.50, 1.40],
        "draw_odds": [3.50, 4.50],
        "away_win_odds": [2.10, 9.00],
        "source": ["OddsPortal-Archive"] * len(dates),
        "fetched_at": ["2022-11-19"] * len(dates),
    })


def test_simulate_betting_stub_mode_all_defaulted():
    """stub_2.0 mode must produce odds_matched=False for every row."""
    bt = _make_minimal_backtest()
    stub_odds = pd.DataFrame(columns=ODDS_COLUMNS)
    result = simulate_betting(bt, label="test", odds_df=stub_odds, odds_mode="stub_2.0")
    assert "odds_matched" in result.columns
    assert result["odds_matched"].sum() == 0, "stub_2.0 should match zero rows"


def test_simulate_betting_stub_mode_uses_2_0():
    """stub_2.0 mode must use 2.0 for all odds."""
    bt = _make_minimal_backtest()
    result = simulate_betting(bt, label="test", odds_df=None, odds_mode="stub_2.0")
    assert (result["bookmaker_odds_used"] == 2.0).all()


def test_simulate_betting_real_mode_matches_supplied_rows():
    """real mode must match the two supplied odds rows and mark the rest unmatched."""
    bt = _make_minimal_backtest()
    odds = _make_real_odds()
    result = simulate_betting(bt, label="test", odds_df=odds, odds_mode="real")
    assert "odds_matched" in result.columns
    matched = int(result["odds_matched"].sum())
    assert matched == 2, f"Expected 2 matched rows, got {matched}"
    unmatched = len(result) - matched
    assert unmatched == 2


def test_simulate_betting_real_mode_odds_source_populated():
    """odds_source must be set to the bookmaker name for matched rows."""
    bt = _make_minimal_backtest()
    odds = _make_real_odds()
    result = simulate_betting(bt, label="test", odds_df=odds, odds_mode="real")
    matched = result[result["odds_matched"] == True]
    assert (matched["odds_source"] == "OddsPortal-Archive").all()


def test_simulate_betting_returns_profit_column():
    """simulate_betting must return profit and cumulative_profit columns."""
    bt = _make_minimal_backtest()
    result = simulate_betting(bt, label="test", odds_df=None, odds_mode="stub_2.0")
    assert "profit" in result.columns
    assert "cumulative_profit" in result.columns


def test_simulate_betting_comparison_roi_differs_with_real_odds():
    """ROI from real odds must differ from stub ROI when odds differ from 2.0."""
    bt = _make_minimal_backtest(n=2)
    odds = _make_real_odds(
        match_dates=["2022-11-20", "2022-11-21"],
        home_teams=["Qatar", "England"],
        away_teams=["Ecuador", "Iran"],
    )
    stub_result = simulate_betting(bt, label="test_stub", odds_df=None, odds_mode="stub_2.0")
    real_result = simulate_betting(bt, label="test_real", odds_df=odds, odds_mode="real")
    stub_roi = stub_result["profit"].sum() / len(stub_result) * 100.0
    real_roi = real_result["profit"].sum() / len(real_result) * 100.0
    # With WC 2022 odds (Qatar 3.50, England 1.40) both different from 2.0
    assert stub_roi != pytest.approx(real_roi, abs=0.01), (
        "ROI should differ between stub and real modes when real odds differ from 2.0"
    )


# ---------------------------------------------------------------------------
# New: Kelly staking + skip-unmatched
# ---------------------------------------------------------------------------

def test_simulate_betting_kelly_stake_within_cap():
    """Kelly stakes are positive only on edges and never exceed the cap."""
    bt = _make_minimal_backtest()
    odds = _make_real_odds(
        match_dates=["2022-11-20", "2022-11-21"],
        home_teams=["Qatar", "England"],
        away_teams=["Ecuador", "Iran"],
    )
    result = simulate_betting(
        bt, label="test_kelly", odds_df=odds, odds_mode="real",
        staking="kelly", kelly_frac=0.25, kelly_cap=0.05,
    )
    assert "stake" in result.columns
    assert (result["stake"] >= 0).all()
    assert (result["stake"] <= 0.05 + 1e-9).all()
    # Stakes differ from flat 1-unit (edge-proportional sizing in effect).
    assert not (result["stake"] == 1.0).all()


def test_simulate_betting_real_mode_skips_unmatched():
    """In real mode, fixtures without matched odds get stake 0 (skipped)."""
    bt = _make_minimal_backtest()
    odds = _make_real_odds()  # matches only the first two fixtures
    result = simulate_betting(bt, label="test_skip", odds_df=odds, odds_mode="real")
    unmatched = result[~result["odds_matched"]]
    assert (unmatched["stake"] == 0.0).all()
    assert (unmatched["profit"] == 0.0).all()


def test_simulate_betting_flat_stub_stakes_all():
    """Stub mode keeps flat 1-unit stakes on every match."""
    bt = _make_minimal_backtest()
    result = simulate_betting(bt, label="test_flat", odds_df=None, odds_mode="stub_2.0")
    assert (result["stake"] == 1.0).all()


# ---------------------------------------------------------------------------
# New: load_odds_for_backtest mode tests
# ---------------------------------------------------------------------------

def test_load_odds_stub_mode_returns_empty():
    """stub_2.0 mode must always return an empty DataFrame."""
    result = load_odds_for_backtest(mode="stub_2.0")
    assert result.empty
    for col in ODDS_COLUMNS:
        assert col in result.columns


def test_load_odds_real_mode_returns_dataframe():
    """real mode must return a DataFrame with ODDS_COLUMNS schema."""
    result = load_odds_for_backtest(mode="real")
    for col in ODDS_COLUMNS:
        assert col in result.columns

