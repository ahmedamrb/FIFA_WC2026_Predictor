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

FEATURE_COLUMNS: list[str] = [
    # --- Rankings (Subphase 3.3) ---
    "home_rank",
    "away_rank",
    "home_rank_points",
    "away_rank_points",
    "rank_diff",
    "rank_points_ratio",
]

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


def merge_rankings(
    matches_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Attach FIFA rankings to each match row using the most recent available data.

    For every row in ``matches_df`` the function performs an as-of lookup
    against ``rankings_df`` to find the latest published ranking for both the
    home team and the away team that falls on or before the match date.  Rows
    with no prior ranking record (e.g. very early matches) are filled with the
    global median rank and median points.

    Args:
        matches_df: Match DataFrame produced by ``clean_results()`` or a
            fixtures DataFrame.  Must contain ``date_col``, ``home_team``, and
            ``away_team`` columns.
        rankings_df: Rankings DataFrame as returned by ``load_raw_data()``.
            Must contain ``country_full``, ``rank_date``, ``rank``, and
            ``total_points`` columns with a parsed datetime ``rank_date``.
        date_col: Name of the date column in ``matches_df``.  Defaults to
            ``"date"`` (clean results); pass ``"match_date"`` for fixtures.

    Returns:
        A copy of ``matches_df`` with six additional columns:
        ``home_rank``, ``away_rank``, ``home_rank_points``,
        ``away_rank_points``, ``rank_diff``, and ``rank_points_ratio``.
    """
    df = matches_df.copy()

    # Ensure rank_date is datetime (defensive in case caller skips load_raw_data)
    rank_df = rankings_df.copy()
    rank_df["rank_date"] = pd.to_datetime(rank_df["rank_date"])

    # Compute global medians used for null filling
    median_rank: float = float(rank_df["rank"].median())
    median_points: float = float(rank_df["total_points"].median())

    # Prepare a clean lookup table sorted by rank_date
    rank_lookup = (
        rank_df[["country_full", "rank_date", "rank", "total_points"]]
        .sort_values("rank_date")
        .reset_index(drop=True)
    )

    # Sort matches by date — required by merge_asof
    df = df.sort_values(date_col).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Home-team ranking lookup
    # ------------------------------------------------------------------
    home_lookup = rank_lookup.rename(
        columns={
            "country_full": "home_team",
            "rank_date": date_col,
            "rank": "home_rank",
            "total_points": "home_rank_points",
        }
    )
    df = pd.merge_asof(
        df,
        home_lookup[[date_col, "home_team", "home_rank", "home_rank_points"]],
        on=date_col,
        by="home_team",
        direction="backward",
    )

    # ------------------------------------------------------------------
    # Away-team ranking lookup
    # ------------------------------------------------------------------
    away_lookup = rank_lookup.rename(
        columns={
            "country_full": "away_team",
            "rank_date": date_col,
            "rank": "away_rank",
            "total_points": "away_rank_points",
        }
    )
    df = pd.merge_asof(
        df,
        away_lookup[[date_col, "away_team", "away_rank", "away_rank_points"]],
        on=date_col,
        by="away_team",
        direction="backward",
    )

    # ------------------------------------------------------------------
    # Fill nulls with global medians
    # ------------------------------------------------------------------
    df["home_rank"] = df["home_rank"].fillna(median_rank)
    df["away_rank"] = df["away_rank"].fillna(median_rank)
    df["home_rank_points"] = df["home_rank_points"].fillna(median_points)
    df["away_rank_points"] = df["away_rank_points"].fillna(median_points)

    # ------------------------------------------------------------------
    # Derived features
    # ------------------------------------------------------------------
    df["rank_diff"] = df["home_rank"] - df["away_rank"]
    away_pts_safe = df["away_rank_points"].clip(lower=1e-6)
    df["rank_points_ratio"] = (df["home_rank_points"] / away_pts_safe).clip(0.1, 10.0)

    return df


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

    # ------------------------------------------------------------------
    # Subphase 3.3 verification
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("merge_rankings() — Subphase 3.3 verification")
    print("=" * 55)

    ranked = merge_rankings(cleaned, rankings)

    ranking_cols = ["home_rank", "away_rank", "home_rank_points", "away_rank_points"]
    print("\nNull counts for ranking columns (must all be 0):")
    for col in ranking_cols:
        print(f"  {col}: {ranked[col].isna().sum()}")

    print(f"\nrank_diff       min: {ranked['rank_diff'].min():.2f}   max: {ranked['rank_diff'].max():.2f}")
    print(f"rank_points_ratio min: {ranked['rank_points_ratio'].min():.4f}   max: {ranked['rank_points_ratio'].max():.4f}")

    print("\nSpot check — France vs Croatia, WC 2018 Final (2018-07-15):")
    wc2018_final = ranked[
        (ranked["date"].dt.date == pd.Timestamp("2018-07-15").date())
        & (ranked["home_team"] == "France")
        & (ranked["away_team"] == "Croatia")
    ]
    if not wc2018_final.empty:
        row = wc2018_final.iloc[0]
        print(f"  home_rank (France): {row['home_rank']:.0f}   away_rank (Croatia): {row['away_rank']:.0f}")
        print(f"  home_rank_points:   {row['home_rank_points']:.1f}   away_rank_points: {row['away_rank_points']:.1f}")
        print(f"  rank_diff:          {row['rank_diff']:.0f}")
        print(f"  rank_points_ratio:  {row['rank_points_ratio']:.4f}")
    else:
        print("  Match not found — check team names.")

