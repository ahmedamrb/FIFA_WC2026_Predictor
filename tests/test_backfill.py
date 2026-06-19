"""Unit tests for backfilling finished WC 2026 scores into the raw history.

Covers the team-name normalization join (feed name vs history name), the
unplayed-rows-stay-empty guarantee, appending knockout matches that have no
existing row, skipping matches with no fixture metadata, and idempotency.
"""

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.backfill import backfill_raw_results

# Feed name (fixtures) <-> history name (results.csv) both resolve here.
_NAME_MAP = {
    "czech republic": "Czechia",
    "czechia": "Czechia",
    "dr congo": "Congo DR",
    "congo dr": "Congo DR",
}
_REVERSE_NAME_MAP = {"Czechia": "Czech Republic", "Congo DR": "DR Congo"}


def _write_results(tmp_path) -> pathlib.Path:
    """Raw history: 4 score-less WC2026 rows + 1 older WC row that must not change."""
    df = pd.DataFrame({
        "date": ["2018-06-14", "2026-06-11", "2026-06-12", "2026-06-12", "2026-06-20"],
        "home_team": ["Russia", "Mexico", "South Korea", "Canada", "Brazil"],
        # history spellings differ from the feed for Czechia / Congo DR:
        "away_team": ["Saudi Arabia", "South Africa", "Czech Republic", "DR Congo", "Morocco"],
        "home_score": [5, None, None, None, None],
        "away_score": [0, None, None, None, None],
        "tournament": ["FIFA World Cup"] * 5,
        "city": ["Moscow", "Mexico City", "Guadalajara", "Toronto", "Miami"],
        "country": ["Russia", "Mexico", "Mexico", "Canada", "United States"],
        "neutral": [False, False, True, False, True],
    })
    path = tmp_path / "results.csv"
    df.to_csv(path, index=False)
    return path


def _write_fixtures(tmp_path) -> pathlib.Path:
    """Fixtures use feed spellings; 205 is a knockout absent from results.csv."""
    df = pd.DataFrame({
        "fixture_id": [201, 202, 203, 204, 205],
        "match_date": ["2026-06-11", "2026-06-12", "2026-06-12", "2026-06-20", "2026-07-05"],
        "home_team": ["Mexico", "South Korea", "Canada", "Brazil", "Argentina"],
        "away_team": ["South Africa", "Czechia", "Congo DR", "Morocco", "Congo DR"],
        "stage": ["GROUP_STAGE"] * 4 + ["LAST_16"],
    })
    path = tmp_path / "wc2026_fixtures_flat.csv"
    df.to_csv(path, index=False)
    return path


def _live_results() -> pd.DataFrame:
    """201/202/203 finished group games, 204 not started, 205 finished knockout,
    206 finished but has no fixture metadata (should be skipped)."""
    return pd.DataFrame({
        "fixture_id": [201, 202, 203, 204, 205, 206],
        "status": ["FINISHED", "FINISHED", "FINISHED", "TIMED", "FINISHED", "FINISHED"],
        "home_score": [2, 2, 1, None, 3, 1],
        "away_score": [0, 1, 1, None, 2, 0],
    })


def _run(tmp_path):
    return backfill_raw_results(
        _live_results(),
        results_path=_write_results(tmp_path),
        fixtures_path=_write_fixtures(tmp_path),
        name_map=_NAME_MAP,
        reverse_name_map=_REVERSE_NAME_MAP,
        verbose=False,
    )


def test_summary_counts(tmp_path):
    summary = _run(tmp_path)
    assert summary["finished"] == 4          # 201,202,203,205 (204 unplayed, 206 dropped pre-count)
    assert summary["updated"] == 3           # 201,202,203 matched existing rows
    assert summary["appended"] == 1          # 205 knockout had no row
    assert summary["skipped"] == 1
    assert summary["skipped_fixture_ids"] == [206]


def test_existing_rows_filled_via_name_normalization(tmp_path):
    results_path = _write_results(tmp_path)
    backfill_raw_results(
        _live_results(), results_path=results_path, fixtures_path=_write_fixtures(tmp_path),
        name_map=_NAME_MAP, reverse_name_map=_REVERSE_NAME_MAP, verbose=False,
    )
    out = pd.read_csv(results_path)

    def score(home, away):
        row = out[(out["home_team"] == home) & (out["away_team"] == away)].iloc[0]
        return row["home_score"], row["away_score"]

    assert score("Mexico", "South Africa") == (2, 0)
    assert score("South Korea", "Czech Republic") == (2, 1)   # feed "Czechia" -> history
    assert score("Canada", "DR Congo") == (1, 1)              # feed "Congo DR" -> history


def test_unplayed_and_unrelated_rows_untouched(tmp_path):
    results_path = _write_results(tmp_path)
    backfill_raw_results(
        _live_results(), results_path=results_path, fixtures_path=_write_fixtures(tmp_path),
        name_map=_NAME_MAP, reverse_name_map=_REVERSE_NAME_MAP, verbose=False,
    )
    out = pd.read_csv(results_path)

    brazil = out[out["home_team"] == "Brazil"].iloc[0]
    assert pd.isna(brazil["home_score"]) and pd.isna(brazil["away_score"])

    russia = out[out["home_team"] == "Russia"].iloc[0]      # older WC, not 2026
    assert russia["home_score"] == 5 and russia["away_score"] == 0


def test_knockout_appended_with_history_spelling(tmp_path):
    results_path = _write_results(tmp_path)
    backfill_raw_results(
        _live_results(), results_path=results_path, fixtures_path=_write_fixtures(tmp_path),
        name_map=_NAME_MAP, reverse_name_map=_REVERSE_NAME_MAP, verbose=False,
    )
    out = pd.read_csv(results_path)

    appended = out[(out["home_team"] == "Argentina") & (out["away_team"] == "DR Congo")]
    assert len(appended) == 1
    row = appended.iloc[0]
    assert (row["home_score"], row["away_score"]) == (3, 2)
    assert row["tournament"] == "FIFA World Cup"


def test_idempotent(tmp_path):
    results_path = _write_results(tmp_path)
    fixtures_path = _write_fixtures(tmp_path)
    kwargs = dict(
        results_path=results_path, fixtures_path=fixtures_path,
        name_map=_NAME_MAP, reverse_name_map=_REVERSE_NAME_MAP, verbose=False,
    )

    backfill_raw_results(_live_results(), **kwargs)
    rows_after_first = len(pd.read_csv(results_path))

    second = backfill_raw_results(_live_results(), **kwargs)
    out = pd.read_csv(results_path)

    # No duplicate append on the second pass: the knockout now matches an existing row.
    assert second["appended"] == 0
    assert second["updated"] == 4
    assert len(out) == rows_after_first
    assert len(out[(out["home_team"] == "Argentina") & (out["away_team"] == "DR Congo")]) == 1
