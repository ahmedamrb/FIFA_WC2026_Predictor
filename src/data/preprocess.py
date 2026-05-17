"""Data preprocessing utilities for the FIFA WC 2026 Predictor.

Loads and normalises raw CSVs (results, rankings, fixtures, name map),
aligns team names to a single canonical form (fixture_name), and parses
date columns to datetime.  All downstream feature-engineering steps read
from the DataFrames produced here.
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_START_YEAR: int = 1990

FEATURE_COLUMNS: list[str] = []

_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and normalise the four core raw data files.

    Reads ``results.csv``, ``rankings.csv``, ``wc2026_fixtures_flat.csv``, and
    ``team_name_map.csv`` from ``data/raw/``.  Team names in ``results`` and
    ``rankings`` are mapped to the canonical ``fixture_name`` values defined in
    the name-map so that all three datasets share a consistent namespace.

    Returns:
        A 4-tuple ``(results_df, rankings_df, fixtures_df, name_map_df)`` where:

        * ``results_df``: International match results with normalised team names
          and a parsed ``date`` column.
        * ``rankings_df``: FIFA world rankings with a normalised ``country_full``
          column and a parsed ``rank_date`` column.
        * ``fixtures_df``: WC 2026 fixture schedule with a parsed ``match_date``
          column.
        * ``name_map_df``: The raw name-map DataFrame (unchanged).
    """
    # ------------------------------------------------------------------
    # Load raw CSVs
    # ------------------------------------------------------------------
    results_df = pd.read_csv(_RAW_DIR / "results.csv")
    rankings_df = pd.read_csv(_RAW_DIR / "rankings.csv", index_col=0)
    fixtures_df = pd.read_csv(_RAW_DIR / "wc2026_fixtures_flat.csv")
    name_map_df = pd.read_csv(_RAW_DIR / "team_name_map.csv")

    # ------------------------------------------------------------------
    # Build normalisation maps: source name → canonical fixture_name
    # ------------------------------------------------------------------
    results_to_fixture: dict[str, str] = dict(
        zip(name_map_df["results_name"], name_map_df["fixture_name"])
    )
    rankings_to_fixture: dict[str, str] = dict(
        zip(name_map_df["rankings_name"], name_map_df["fixture_name"])
    )

    # ------------------------------------------------------------------
    # Apply normalisation
    # ------------------------------------------------------------------
    results_df["home_team"] = results_df["home_team"].replace(results_to_fixture)
    results_df["away_team"] = results_df["away_team"].replace(results_to_fixture)
    rankings_df["country_full"] = rankings_df["country_full"].replace(rankings_to_fixture)

    # ------------------------------------------------------------------
    # Parse date columns
    # ------------------------------------------------------------------
    results_df["date"] = pd.to_datetime(results_df["date"])
    rankings_df["rank_date"] = pd.to_datetime(rankings_df["rank_date"])
    fixtures_df["match_date"] = pd.to_datetime(fixtures_df["match_date"])

    return results_df, rankings_df, fixtures_df, name_map_df


# ---------------------------------------------------------------------------
# Script entry point — quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results, rankings, fixtures, name_map = load_raw_data()

    for label, df in [
        ("results_df", results),
        ("rankings_df", rankings),
        ("fixtures_df", fixtures),
        ("name_map_df", name_map),
    ]:
        print(f"{label:20s} shape: {df.shape}")

    print()
    print(f"results_df['date']         dtype: {results['date'].dtype}")
    print(f"rankings_df['rank_date']   dtype: {rankings['rank_date'].dtype}")
    print(f"fixtures_df['match_date']  dtype: {fixtures['match_date'].dtype}")

