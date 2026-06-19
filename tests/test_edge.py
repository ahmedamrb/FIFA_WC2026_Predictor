"""Unit tests for betting edge computation — Subphase 6.6 + real-odds extensions."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.betting.edge import compute_edge, remove_vig, VALUE_THRESHOLD
from src.data.odds import validate_odds_df, normalize_team_name, get_latest_odds

_BACKTEST_WC2022 = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed" / "backtest_wc2022.csv"

# "No Odds" is a valid recommendation for fixtures without real bookmaker prices.
_VALID_RECOMMENDATIONS = {"Value", "Neutral", "Avoid", "No Odds"}
_IMPLIED_PROB_COLUMNS = [
    "home_win_implied_prob",
    "draw_implied_prob",
    "away_win_implied_prob",
]


# ---------------------------------------------------------------------------
# Original tests (unchanged)
# ---------------------------------------------------------------------------

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
        # Implied probs are NaN for fixtures without real odds; only the priced
        # rows must lie in (0, 1].
        priced = df[col].dropna()
        assert (priced > 0).all(), f"Column '{col}' contains values <= 0"
        assert (priced <= 1.0).all(), f"Column '{col}' contains values > 1.0"


# ---------------------------------------------------------------------------
# New: odds module unit tests
# ---------------------------------------------------------------------------

def _make_backtest_df(n: int = 4) -> pd.DataFrame:
    """Create a minimal synthetic backtest DataFrame."""
    return pd.DataFrame({
        "match_date": ["2022-11-20", "2022-11-21", "2022-11-22", "2022-11-23"][:n],
        "home_team": ["TeamA", "TeamB", "TeamC", "TeamD"][:n],
        "away_team": ["TeamE", "TeamF", "TeamG", "TeamH"][:n],
        "predicted_home_win_prob": [0.6, 0.4, 0.5, 0.3],
        "predicted_draw_prob": [0.2, 0.3, 0.25, 0.35],
        "predicted_away_win_prob": [0.2, 0.3, 0.25, 0.35],
        "predicted_outcome": [2, 0, 2, 1],
        "actual_outcome": [2, 0, 1, 1],
        "profit": [1.0, 1.2, -1.0, 0.8],
        "bookmaker_odds_used": [2.0, 2.2, 2.0, 1.8],
        "odds_matched": [True, True, False, False],
        "odds_source": ["OddsPortal", "OddsPortal", "stub_2.0", "stub_2.0"],
    })


def _make_odds_df() -> pd.DataFrame:
    """Create a minimal synthetic odds DataFrame."""
    return pd.DataFrame({
        "match_date": ["2022-11-20", "2022-11-21"],
        "home_team": ["TeamA", "TeamB"],
        "away_team": ["TeamE", "TeamF"],
        "home_win_odds": [1.80, 2.50],
        "draw_odds": [3.40, 3.20],
        "away_win_odds": [4.50, 2.80],
        "source": ["OddsPortal", "OddsPortal"],
        "fetched_at": ["2022-11-19", "2022-11-19"],
    })


def test_compute_edge_adds_implied_probs():
    """compute_edge must produce vig-free implied prob columns for priced rows."""
    bt = _make_backtest_df()
    odds = _make_odds_df()
    result = compute_edge(bt, odds)
    for col in _IMPLIED_PROB_COLUMNS:
        assert col in result.columns, f"Missing {col}"
        priced = result[col].dropna()
        assert (priced > 0).all()
        assert (priced <= 1.0).all()
    # Two of four synthetic fixtures have odds → two priced rows.
    assert result["home_win_implied_prob"].notna().sum() == 2


def test_compute_edge_implied_probs_are_vig_free():
    """The three implied probs must sum to 1.0 on priced rows (margin removed)."""
    bt = _make_backtest_df()
    odds = _make_odds_df()
    result = compute_edge(bt, odds)
    priced = result.dropna(subset=_IMPLIED_PROB_COLUMNS)
    sums = priced[_IMPLIED_PROB_COLUMNS].sum(axis=1)
    np.testing.assert_allclose(sums.to_numpy(), 1.0, atol=1e-9)


def test_compute_edge_adds_recommendation():
    """compute_edge must produce a bet_recommendation column with valid values."""
    bt = _make_backtest_df()
    odds = _make_odds_df()
    result = compute_edge(bt, odds)
    assert "bet_recommendation" in result.columns
    assert result["bet_recommendation"].isin(_VALID_RECOMMENDATIONS).all()


def test_compute_edge_unmatched_rows_get_no_odds():
    """Rows without matching odds must NOT fabricate an edge: NaN + 'No Odds'."""
    bt = _make_backtest_df()
    odds = pd.DataFrame(columns=["match_date", "home_team", "away_team",
                                  "home_win_odds", "draw_odds", "away_win_odds"])
    result = compute_edge(bt, odds)
    assert result["home_win_implied_prob"].isna().all()
    assert result["best_edge"].isna().all()
    assert (result["bet_recommendation"] == "No Odds").all()


def test_compute_edge_value_outcome_named():
    """value_outcome must name the leg carrying the best edge on priced rows."""
    bt = _make_backtest_df()
    odds = _make_odds_df()
    result = compute_edge(bt, odds)
    priced = result.dropna(subset=["best_edge"])
    assert priced["value_outcome"].isin({"Home Win", "Draw", "Away Win"}).all()
    # Unpriced rows carry no value outcome.
    assert result[result["best_edge"].isna()]["value_outcome"].isna().all()


def test_validate_odds_df_drops_invalid_rows():
    """validate_odds_df must drop rows with odds < 1.0."""
    df = pd.DataFrame({
        "match_date": ["2022-11-20", "2022-11-21"],
        "home_team": ["A", "B"],
        "away_team": ["C", "D"],
        "home_win_odds": [0.5, 2.0],  # first row invalid
        "draw_odds": [3.0, 3.0],
        "away_win_odds": [4.0, 4.0],
    })
    result = validate_odds_df(df)
    assert len(result) == 1
    assert float(result.iloc[0]["home_win_odds"]) == 2.0


def test_validate_odds_df_raises_on_missing_column():
    """validate_odds_df must raise ValueError if a required column is absent."""
    df = pd.DataFrame({
        "match_date": ["2022-11-20"],
        "home_team": ["A"],
        "away_team": ["C"],
        # home_win_odds missing intentionally
        "draw_odds": [3.0],
        "away_win_odds": [4.0],
    })
    with pytest.raises(ValueError, match="missing required columns"):
        validate_odds_df(df)


def test_normalize_team_name_applies_map():
    """normalize_team_name must resolve provider names via name_map."""
    name_map = {"USA": "United States", "Korea Republic": "South Korea"}
    assert normalize_team_name("USA", name_map) == "United States"
    assert normalize_team_name("Korea Republic", name_map) == "South Korea"


def test_normalize_team_name_passthrough():
    """normalize_team_name returns the original name when not in map."""
    assert normalize_team_name("Brazil", {}) == "Brazil"


def test_get_latest_odds_deduplicates():
    """get_latest_odds must keep only the most recent row per fixture."""
    df = pd.DataFrame({
        "match_date": ["2022-11-20", "2022-11-20"],
        "home_team": ["A", "A"],
        "away_team": ["B", "B"],
        "home_win_odds": [1.8, 1.9],
        "draw_odds": [3.2, 3.1],
        "away_win_odds": [4.0, 3.9],
        "source": ["S1", "S1"],
        "fetched_at": ["2022-11-18", "2022-11-19"],  # second row is newer
    })
    result = get_latest_odds(df)
    assert len(result) == 1
    assert float(result.iloc[0]["home_win_odds"]) == pytest.approx(1.9)


# ---------------------------------------------------------------------------
# New: vig removal
# ---------------------------------------------------------------------------

def test_remove_vig_sums_to_one():
    """Vig-free probabilities must sum to exactly 1.0."""
    fh, fd, fa = remove_vig(2.0, 3.0, 4.0)
    assert fh + fd + fa == pytest.approx(1.0)


def test_remove_vig_preserves_favourite_order():
    """The shortest odds (favourite) must keep the highest fair probability."""
    fh, fd, fa = remove_vig(1.5, 4.0, 6.0)
    assert fh > fd > fa


def test_remove_vig_strips_margin():
    """Fair home prob must be below the raw 1/odds implied prob (margin removed)."""
    fh, _, _ = remove_vig(2.0, 3.0, 4.0)
    assert fh < 0.5  # raw implied was 1/2.0 = 0.5


def test_remove_vig_missing_leg_is_nan():
    """A missing or sub-1.0 leg yields NaN for all three fair probabilities."""
    for bad in (float("nan"), 0.0, 0.9):
        fh, fd, fa = remove_vig(2.0, bad, 4.0)
        assert np.isnan(fh) and np.isnan(fd) and np.isnan(fa)


def test_remove_vig_vectorized():
    """remove_vig must accept array-like input and mask invalid rows independently."""
    h = np.array([2.0, 1.5, float("nan")])
    d = np.array([3.0, 4.0, 3.0])
    a = np.array([4.0, 6.0, 2.0])
    fh, fd, fa = remove_vig(h, d, a)
    sums = fh + fd + fa
    np.testing.assert_allclose(sums[:2], 1.0, atol=1e-9)
    assert np.isnan(sums[2])

