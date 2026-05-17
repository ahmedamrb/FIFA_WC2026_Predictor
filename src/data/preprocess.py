"""Data preprocessing utilities for the FIFA WC 2026 Predictor.

Loads and normalises raw CSVs (results, rankings, fixtures, name map),
aligns team names to a single canonical form (fixture_name), and parses
date columns to datetime.  All downstream feature-engineering steps read
from the DataFrames produced here.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_START_YEAR: int = 1990

FEATURE_COLUMNS: list[str] = []

_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
_REFERENCE_DATE: pd.Timestamp = pd.Timestamp("2026-06-01")


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


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Filter, label, and weight the results DataFrame for model training.

    Performs three transformations on a copy of ``df``:

    1. **Date filter** — keeps only matches from ``TRAIN_START_YEAR`` (1990) onward.
    2. **Score filter** — drops rows where ``home_score`` or ``away_score`` is null
       (these are WC 2026 fixtures scheduled but not yet played).
    3. **Derived columns** — adds ``match_importance``, ``outcome``, and
       ``recency_weight``.

    Args:
        df: Results DataFrame as returned by ``load_raw_data()``, with a
            datetime ``date`` column and numeric ``home_score``/``away_score``.

    Returns:
        A cleaned copy of ``df`` with a reset integer index and three new columns.
    """
    df = df.copy()

    # 1. Filter: keep only matches on or after TRAIN_START_YEAR
    df = df[df["date"].dt.year >= TRAIN_START_YEAR]

    # 2. Drop rows without completed scores (unplayed/future fixtures)
    df = df.dropna(subset=["home_score", "away_score"])

    # 3. match_importance — checked in priority order by np.select
    #    Uses "qualif" substring to capture both "qualifier" and "qualification"
    conditions_importance = [
        df["tournament"] == "FIFA World Cup",
        df["tournament"].str.contains("qualif", case=False, na=False),
        df["tournament"] == "Friendly",
    ]
    df["match_importance"] = np.select(
        conditions_importance, [3.0, 1.5, 0.5], default=1.0
    )

    # 4. outcome: 2 = home win, 1 = draw, 0 = away win
    df["outcome"] = np.select(
        [
            df["home_score"] > df["away_score"],
            df["home_score"] == df["away_score"],
        ],
        [2, 1],
        default=0,
    ).astype(int)

    # 5. recency_weight: 0.85 ^ years_since_match (relative to _REFERENCE_DATE)
    years_since = (_REFERENCE_DATE - df["date"]).dt.days / 365.25
    df["recency_weight"] = 0.85 ** years_since

    return df.reset_index(drop=True)


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

    # ------------------------------------------------------------------
    # Subphase 3.2 verification
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("clean_results() — Subphase 3.2 verification")
    print("=" * 55)

    cleaned = clean_results(results)
    print(f"Cleaned shape:  {cleaned.shape}")

    print()
    print("outcome value counts (0=away win, 1=draw, 2=home win):")
    print(cleaned["outcome"].value_counts().sort_index().to_string())

    print()
    print(f"match_importance unique values: {sorted(cleaned['match_importance'].unique())}")
    print(f"match_importance null count: {cleaned['match_importance'].isna().sum()}")
    print(f"outcome null count: {cleaned['outcome'].isna().sum()}")
    print(f"outcome unique values: {sorted(cleaned['outcome'].unique())}")

    print()
    print(f"recency_weight  min : {cleaned['recency_weight'].min():.6f}")
    print(f"recency_weight  max : {cleaned['recency_weight'].max():.6f}")

    print()
    print("Spot check — older matches must have lower recency_weight:")
    row_1990 = cleaned[cleaned["date"].dt.year == 1990].iloc[0]
    row_2022 = cleaned[
        (cleaned["date"].dt.year == 2022) & (cleaned["tournament"] == "FIFA World Cup")
    ].iloc[0]
    for label, row in [("1990 match   ", row_1990), ("2022 WC match", row_2022)]:
        print(
            f"  {label}: {row['date'].date()}  "
            f"{row['home_team']} vs {row['away_team']}  "
            f"recency_weight = {row['recency_weight']:.6f}"
        )

