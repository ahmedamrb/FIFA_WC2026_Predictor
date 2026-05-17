"""Cross-dataset alignment check for WC 2026 team names.

Verifies that all 48 WC 2026 team names from wc2026_fixtures_flat.csv can be
mapped to corresponding names in results.csv and rankings.csv, either directly
or via data/raw/team_name_map.csv.

Exit behaviour:
  Prints "ALIGNMENT CHECK PASSED" when all teams resolve correctly.
  Prints "ALIGNMENT CHECK FAILED" when any mismatches remain.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
FIXTURES_PATH = ROOT / "data" / "raw" / "wc2026_fixtures_flat.csv"
RESULTS_PATH = ROOT / "data" / "raw" / "results.csv"
RANKINGS_PATH = ROOT / "data" / "raw" / "rankings.csv"
NAME_MAP_PATH = ROOT / "data" / "raw" / "team_name_map.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixture_teams() -> set[str]:
    """Return unique WC 2026 team names, excluding placeholder 'TBD'."""
    df = pd.read_csv(FIXTURES_PATH)
    teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    return {t for t in teams if t != "TBD"}


def _load_results_teams() -> set[str]:
    """Return all unique team names that appear in results.csv."""
    df = pd.read_csv(RESULTS_PATH, usecols=["home_team", "away_team"])
    return set(pd.concat([df["home_team"], df["away_team"]]).unique())


def _load_rankings_teams() -> set[str]:
    """Return all unique team names from rankings.csv country_full column."""
    df = pd.read_csv(RANKINGS_PATH, usecols=["country_full"])
    return set(df["country_full"].unique())


def _load_name_map() -> dict[str, dict]:
    """Load team_name_map.csv and return a dict keyed by fixture_name.

    Each value is a dict with keys 'results_name' and 'rankings_name'.
    Returns an empty dict if the file does not exist.
    """
    if not NAME_MAP_PATH.exists():
        return {}

    df = pd.read_csv(NAME_MAP_PATH)
    required = {"fixture_name", "results_name", "rankings_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"team_name_map.csv is missing columns: {missing}")

    return {
        row["fixture_name"]: {
            "results_name": row["results_name"],
            "rankings_name": row["rankings_name"],
        }
        for _, row in df.iterrows()
    }


def _check_mismatches(
    fixture_teams: set[str],
    results_teams: set[str],
    rankings_teams: set[str],
    name_map: dict[str, dict],
    label: str,
) -> tuple[list[str], list[str]]:
    """Check which fixture teams cannot be resolved in each reference dataset.

    Args:
        fixture_teams: Set of WC 2026 team names from fixtures.
        results_teams: Set of all team names in results.csv.
        rankings_teams: Set of all team names in rankings.csv.
        name_map: Mapping from fixture_name to resolved names.
        label: Section header for printed output.

    Returns:
        (results_mismatches, rankings_mismatches) — lists of unresolved fixture names.
    """
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    results_mismatches: list[str] = []
    rankings_mismatches: list[str] = []

    for team in sorted(fixture_teams):
        mapping = name_map.get(team, {})
        resolved_results = mapping.get("results_name", team)
        resolved_rankings = mapping.get("rankings_name", team)

        if resolved_results not in results_teams:
            results_mismatches.append(team)

        if resolved_rankings not in rankings_teams:
            rankings_mismatches.append(team)

    if results_mismatches:
        print(f"\n  Teams NOT found in results.csv ({len(results_mismatches)}):")
        for t in results_mismatches:
            print(f"    - {t!r}")
    else:
        print("\n  All teams found in results.csv. ✓")

    if rankings_mismatches:
        print(f"\n  Teams NOT found in rankings.csv ({len(rankings_mismatches)}):")
        for t in rankings_mismatches:
            print(f"    - {t!r}")
    else:
        print("\n  All teams found in rankings.csv. ✓")

    return results_mismatches, rankings_mismatches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the cross-dataset alignment check and report results."""
    print("Loading datasets …")
    fixture_teams = _load_fixture_teams()
    results_teams = _load_results_teams()
    rankings_teams = _load_rankings_teams()

    print(f"  WC 2026 fixture teams : {len(fixture_teams)}")
    print(f"  results.csv teams     : {len(results_teams)}")
    print(f"  rankings.csv teams    : {len(rankings_teams)}")

    # --- Pass 1: no mapping ---------------------------------------------------
    res_miss_raw, rank_miss_raw = _check_mismatches(
        fixture_teams,
        results_teams,
        rankings_teams,
        name_map={},
        label="PASS 1 — Before applying team_name_map.csv",
    )
    total_raw = len(res_miss_raw) + len(rank_miss_raw)
    print(f"\n  Total mismatches (before mapping): {total_raw}")

    # --- Pass 2: with mapping -------------------------------------------------
    name_map = _load_name_map()
    if name_map:
        print(f"\nLoaded team_name_map.csv — {len(name_map)} entries.")
    else:
        print(f"\nteam_name_map.csv not found — skipping mapped pass.")
        print("\nALIGNMENT CHECK FAILED — create team_name_map.csv to resolve mismatches.")
        return

    res_miss_mapped, rank_miss_mapped = _check_mismatches(
        fixture_teams,
        results_teams,
        rankings_teams,
        name_map=name_map,
        label="PASS 2 — After applying team_name_map.csv",
    )

    total_mapped = len(res_miss_mapped) + len(rank_miss_mapped)
    resolved = total_raw - total_mapped

    print(f"\n  Resolved mismatches : {resolved}")
    print(f"  Remaining mismatches: {total_mapped}")

    if total_mapped == 0:
        print("\nALIGNMENT CHECK PASSED")
    else:
        print("\nALIGNMENT CHECK FAILED — remaining issues:")
        for t in res_miss_mapped:
            print(f"  results  : {t!r}")
        for t in rank_miss_mapped:
            print(f"  rankings : {t!r}")


if __name__ == "__main__":
    main()

