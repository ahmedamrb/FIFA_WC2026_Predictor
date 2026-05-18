"""Unit tests for src/data/preprocess — Subphase 3.8."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    clean_results,
    compute_form_features,
    compute_h2h_features,
    merge_rankings,
)


def _make_results_row(date, home, away, hs, as_, tournament="Friendly"):
    return {
        "date": pd.Timestamp(date),
        "home_team": home,
        "away_team": away,
        "home_score": hs,
        "away_score": as_,
        "tournament": tournament,
        "neutral": False,
        "city": "Testcity",
        "country": "Testland",
    }


def test_clean_results_no_early_matches():
    rows = [
        _make_results_row("1985-06-01", "A", "B", 1, 0),
        _make_results_row("1988-07-15", "C", "D", 0, 0),
        _make_results_row("1989-12-31", "E", "F", 2, 1),
        _make_results_row("1994-07-17", "G", "H", 3, 2, "FIFA World Cup"),
        _make_results_row("2002-06-30", "I", "J", 0, 2, "FIFA World Cup"),
        _make_results_row("2010-07-11", "K", "L", 1, 0, "FIFA World Cup"),
    ]
    df = pd.DataFrame(rows)
    result = clean_results(df)
    assert (result["date"].dt.year >= 1990).all()
    assert len(result) == 3


def test_clean_results_outcome_values():
    rows = [
        _make_results_row("1994-07-17", "A", "B", 2, 0, "FIFA World Cup"),
        _make_results_row("2002-06-30", "C", "D", 1, 1, "FIFA World Cup"),
        _make_results_row("2010-07-11", "E", "F", 0, 3, "FIFA World Cup"),
    ]
    df = pd.DataFrame(rows)
    result = clean_results(df)
    assert result["outcome"].notna().all()
    assert set(result["outcome"].unique()).issubset({0, 1, 2})
    assert sorted(result["outcome"].unique()) == [0, 1, 2]


def _make_rankings_df(rows):
    """rows: list of (country, rank_date_str, rank, points)"""
    return pd.DataFrame([
        {
            "country_full": r[0],
            "rank_date": pd.Timestamp(r[1]),
            "rank": r[2],
            "total_points": float(r[3]),
        }
        for r in rows
    ])


def _make_matches_df(rows):
    """rows: list of (date_str, home, away)"""
    return pd.DataFrame([
        {
            "date": pd.Timestamp(r[0]),
            "home_team": r[1],
            "away_team": r[2],
            "home_score": 1,
            "away_score": 0,
            "tournament": "Friendly",
            "neutral": False,
        }
        for r in rows
    ])


def test_rankings_merge_no_nulls():
    matches = _make_matches_df([
        ("2020-06-01", "France", "Brazil"),
        ("2021-03-10", "Brazil", "Germany"),
        ("2022-09-05", "Germany", "France"),
    ])
    rankings = _make_rankings_df([
        ("France",  "2019-01-01", 2,  1800.0),
        ("Brazil",  "2019-01-01", 5,  1750.0),
        ("Germany", "2019-01-01", 10, 1700.0),
    ])
    result = merge_rankings(matches, rankings)
    for col in ["home_rank", "away_rank", "home_rank_points", "away_rank_points"]:
        assert result[col].notna().all(), f"{col} has nulls"


def test_rank_points_ratio_capped():
    matches = _make_matches_df([
        ("2022-01-01", "TeamHigh", "TeamLow"),
        ("2022-06-01", "TeamLow",  "TeamHigh"),
    ])
    rankings = _make_rankings_df([
        ("TeamHigh", "2021-01-01", 1,   2000.0),
        ("TeamLow",  "2021-01-01", 200, 50.0),
    ])
    result = merge_rankings(matches, rankings)
    assert result["rank_points_ratio"].between(0.1, 10.0).all()
    assert (result["rank_points_ratio"] == 10.0).any(), "upper clip not hit"
    assert (result["rank_points_ratio"] == 0.1).any(),  "lower clip not hit"


def test_form_wins_in_range():
    dates = pd.date_range("2010-01-01", periods=12, freq="ME")
    rows = []
    for i, d in enumerate(dates):
        if i % 2 == 0:
            rows.append({"date": d, "home_team": "Germany", "away_team": "France",
                         "home_score": i % 3, "away_score": (i + 1) % 3,
                         "tournament": "Friendly", "neutral": False})
        else:
            rows.append({"date": d, "home_team": "France", "away_team": "Germany",
                         "home_score": (i + 1) % 3, "away_score": i % 3,
                         "tournament": "Friendly", "neutral": False})
    df = pd.DataFrame(rows)
    result = compute_form_features(df)
    assert result["home_form_wins_5"].between(0, 5).all()


def test_h2h_win_rate_in_range():
    dates = pd.date_range("2010-01-01", periods=8, freq="ME")
    rows = []
    for i, d in enumerate(dates):
        if i % 2 == 0:
            rows.append({"date": d, "home_team": "Spain", "away_team": "Argentina",
                         "home_score": i % 3, "away_score": (i + 1) % 3,
                         "tournament": "Friendly", "neutral": False})
        else:
            rows.append({"date": d, "home_team": "Argentina", "away_team": "Spain",
                         "home_score": (i + 1) % 3, "away_score": i % 3,
                         "tournament": "Friendly", "neutral": False})
    df = pd.DataFrame(rows)
    result = compute_h2h_features(df)
    assert result["h2h_home_win_rate"].between(0.0, 1.0).all()
