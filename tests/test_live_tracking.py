"""Unit tests for live result tracking + prediction comparison."""

import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.evaluation.live_tracking import (
    actual_outcome_from_row,
    build_comparison,
    summarize,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures / predictions / results
# ---------------------------------------------------------------------------

def _make_fixtures() -> pd.DataFrame:
    """5 fixtures (RangeIndex 0-4); row 4 is a TBD knockout placeholder."""
    return pd.DataFrame({
        "fixture_id": [101, 102, 103, 104, 105],
        "match_date": ["2026-06-11", "2026-06-12", "2026-06-12", "2026-06-13", "2026-06-28"],
        "home_team": ["Mexico", "Canada", "Brazil", "Spain", "TBD"],
        "away_team": ["South Africa", "Bosnia-Herzegovina", "Morocco", "Cape Verde Islands", "TBD"],
        "stage": ["GROUP_STAGE", "GROUP_STAGE", "GROUP_STAGE", "GROUP_STAGE", "LAST_32"],
    })


def _make_predictions() -> pd.DataFrame:
    """Predictions row-aligned with fixtures (index 0-4)."""
    return pd.DataFrame({
        "predicted_outcome": ["Home Win", "Home Win", "Home Win", "Home Win", "Home Win"],
        "predicted_home_goals": [2, 2, 2, 3, 2],
        "predicted_away_goals": [1, 1, 1, 1, 1],
    })


def _make_results() -> pd.DataFrame:
    """Results keyed by fixture_id: 2 finished, 1 live, 1 upcoming, 1 finished-but-TBD."""
    return pd.DataFrame({
        "fixture_id": [101, 102, 103, 104, 105],
        "status": ["FINISHED", "FINISHED", "IN_PLAY", "TIMED", "FINISHED"],
        "home_score": [2, 0, 1, None, 3],
        "away_score": [1, 2, 0, None, 1],
        "winner": ["HOME_TEAM", "AWAY_TEAM", None, None, "HOME_TEAM"],
        "minute": [None, None, 55, None, None],
    })


def _comp_by_fixture(comp: pd.DataFrame) -> dict:
    return comp.set_index("fixture_id").to_dict("index")


# ---------------------------------------------------------------------------
# actual_outcome_from_row
# ---------------------------------------------------------------------------

def test_actual_outcome_from_winner_field():
    assert actual_outcome_from_row({"winner": "HOME_TEAM", "home_score": 2, "away_score": 1}) == "Home Win"
    assert actual_outcome_from_row({"winner": "AWAY_TEAM", "home_score": 0, "away_score": 2}) == "Away Win"
    assert actual_outcome_from_row({"winner": "DRAW", "home_score": 1, "away_score": 1}) == "Draw"


def test_actual_outcome_falls_back_to_scores():
    assert actual_outcome_from_row({"winner": None, "home_score": 3, "away_score": 1}) == "Home Win"
    assert actual_outcome_from_row({"winner": None, "home_score": 1, "away_score": 1}) == "Draw"


def test_actual_outcome_none_without_score():
    assert actual_outcome_from_row({"winner": None, "home_score": None, "away_score": None}) is None


def test_actual_outcome_accepts_series():
    row = pd.Series({"winner": "AWAY_TEAM", "home_score": 0, "away_score": 1})
    assert actual_outcome_from_row(row) == "Away Win"


# ---------------------------------------------------------------------------
# build_comparison
# ---------------------------------------------------------------------------

def test_build_comparison_correct_outcome_and_exact_score():
    comp = build_comparison(_make_fixtures(), _make_predictions(), _make_results())
    by_id = _comp_by_fixture(comp)

    # 101: predicted Home Win 2-1, actual 2-1 -> correct outcome AND exact score
    row = by_id[101]
    assert row["actual_outcome"] == "Home Win"
    assert bool(row["outcome_correct"]) is True
    assert bool(row["exact_score_correct"]) is True
    assert bool(row["played"]) is True


def test_build_comparison_outcome_miss():
    comp = build_comparison(_make_fixtures(), _make_predictions(), _make_results())
    row = _comp_by_fixture(comp)[102]
    # predicted Home Win, actual 0-2 Away Win
    assert row["actual_outcome"] == "Away Win"
    assert bool(row["outcome_correct"]) is False
    assert bool(row["exact_score_correct"]) is False


def test_build_comparison_live_match_is_provisional():
    comp = build_comparison(_make_fixtures(), _make_predictions(), _make_results())
    row = _comp_by_fixture(comp)[103]
    assert bool(row["is_live"]) is True
    assert bool(row["played"]) is False
    assert bool(row["has_score"]) is True
    assert row["actual_outcome"] == "Home Win"  # provisional from 1-0
    assert bool(row["outcome_correct"]) is True


def test_build_comparison_upcoming_has_no_verdict():
    comp = build_comparison(_make_fixtures(), _make_predictions(), _make_results())
    row = _comp_by_fixture(comp)[104]
    assert bool(row["has_score"]) is False
    assert pd.isna(row["actual_outcome"])  # no score yet -> NA/None
    assert pd.isna(row["outcome_correct"])


def test_build_comparison_excludes_tbd_rows():
    comp = build_comparison(_make_fixtures(), _make_predictions(), _make_results())
    row = _comp_by_fixture(comp)[105]
    # Finished with a score, but teams are TBD -> not comparable, no verdict
    assert bool(row["comparable"]) is False
    assert pd.isna(row["outcome_correct"])
    assert pd.isna(row["exact_score_correct"])


def test_build_comparison_handles_missing_results():
    comp = build_comparison(_make_fixtures(), _make_predictions(), None)
    assert len(comp) == 5
    assert comp["has_score"].sum() == 0
    assert comp["actual_outcome"].isna().all()


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_counts_only_finished_comparable():
    comp = build_comparison(_make_fixtures(), _make_predictions(), _make_results())
    s = summarize(comp)
    # Finished + comparable -> fixtures 101 and 102 only
    assert s["played"] == 2
    assert s["outcome_correct"] == 1          # 101 correct, 102 miss
    assert s["exact"] == 1                     # 101 exact
    assert s["outcome_pct"] == pytest.approx(0.5)
    assert s["exact_pct"] == pytest.approx(0.5)
    assert s["live"] == 1                      # 103 in play


def test_summarize_empty():
    s = summarize(pd.DataFrame())
    assert s == {"played": 0, "outcome_correct": 0, "outcome_pct": 0.0,
                 "exact": 0, "exact_pct": 0.0,
                 "home_goals_correct": 0, "home_goals_pct": 0.0, "home_goals_mae": 0.0,
                 "away_goals_correct": 0, "away_goals_pct": 0.0, "away_goals_mae": 0.0,
                 "live": 0}


# ---------------------------------------------------------------------------
# Per-side goals models (home / away are independent regressors)
# ---------------------------------------------------------------------------

def test_build_comparison_per_side_goals_flags():
    comp = build_comparison(_make_fixtures(), _make_predictions(), _make_results())
    by_id = _comp_by_fixture(comp)

    # 101: pred 2-1, actual 2-1 -> both sides exact, zero error
    assert bool(by_id[101]["home_goals_correct"]) is True
    assert bool(by_id[101]["away_goals_correct"]) is True
    assert by_id[101]["home_goals_error"] == 0
    assert by_id[101]["away_goals_error"] == 0

    # 102: pred 2-1, actual 0-2 -> both miss; errors +2 (home) and -1 (away)
    assert bool(by_id[102]["home_goals_correct"]) is False
    assert bool(by_id[102]["away_goals_correct"]) is False
    assert by_id[102]["home_goals_error"] == 2
    assert by_id[102]["away_goals_error"] == -1


def test_build_comparison_per_side_goals_na_for_tbd_and_upcoming():
    comp = build_comparison(_make_fixtures(), _make_predictions(), _make_results())
    by_id = _comp_by_fixture(comp)
    assert pd.isna(by_id[105]["home_goals_correct"])   # TBD -> not comparable
    assert pd.isna(by_id[105]["away_goals_error"])
    assert pd.isna(by_id[104]["home_goals_correct"])   # upcoming -> no score


def test_summarize_per_side_goals_and_mae():
    comp = build_comparison(_make_fixtures(), _make_predictions(), _make_results())
    s = summarize(comp)
    # Finished + comparable -> fixtures 101 and 102
    assert s["home_goals_correct"] == 1          # 101 hit, 102 miss
    assert s["away_goals_correct"] == 1          # 101 hit, 102 miss
    assert s["home_goals_pct"] == pytest.approx(0.5)
    assert s["away_goals_pct"] == pytest.approx(0.5)
    # MAE: home |0|,|+2| -> 1.0 ; away |0|,|-1| -> 0.5
    assert s["home_goals_mae"] == pytest.approx(1.0)
    assert s["away_goals_mae"] == pytest.approx(0.5)
