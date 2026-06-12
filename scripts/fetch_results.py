"""Fetch live / full-time WC 2026 actual scores and compare to predictions.

Usage
-----
# Fetch current statuses + scores from football-data.org and write
# data/processed/wc2026_results.csv, then print an accuracy summary
python scripts/fetch_results.py

# Print the comparison summary from the existing CSV without hitting the API
python scripts/fetch_results.py --summary

Prerequisites
-------------
Add FD_API_KEY=<your_key> to .env (https://football-data.org — free tier).
The same key is already used by src/data/ingest.fetch_wc2026_fixtures.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest import fetch_wc2026_results  # noqa: E402
from src.evaluation.live_tracking import build_comparison, summarize  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RAW = _REPO_ROOT / "data" / "raw"
_PROCESSED = _REPO_ROOT / "data" / "processed"
_RESULTS_CSV = _PROCESSED / "wc2026_results.csv"


def _load_fixtures_and_predictions() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    fixtures_path = _RAW / "wc2026_fixtures_flat.csv"
    fixtures = pd.read_csv(fixtures_path) if fixtures_path.exists() else None

    pred_path = _PROCESSED / "wc2026_predictions.csv"
    predictions = pd.read_csv(pred_path, index_col=0) if pred_path.exists() else None
    return fixtures, predictions


def _print_status_counts(results: pd.DataFrame) -> None:
    print("\n=== Match status counts ===")
    counts = results["status"].value_counts(dropna=False)
    for status, n in counts.items():
        print(f"  {str(status):<12} {n:>3}")


def _print_recent_finals(comparison: pd.DataFrame, limit: int = 10) -> None:
    played = comparison[comparison["comparable"] & comparison["played"]]
    if played.empty:
        print("\n  No finished comparable matches yet.")
        return
    played = played.sort_values("match_date").tail(limit)
    print(f"\n=== Last {len(played)} finished matches (predicted vs actual) ===")
    for _, r in played.iterrows():
        h, a = int(r["home_score"]), int(r["away_score"])
        verdict = "OK " if bool(r["outcome_correct"]) else "MISS"
        exact = " *exact*" if bool(r["exact_score_correct"]) else ""
        print(
            f"  {r['match_date']}  {r['home_team']} {h}-{a} {r['away_team']}"
            f"  | pred {r['predicted_outcome']} -> actual {r['actual_outcome']}"
            f"  [{verdict}]{exact}"
        )


def _print_summary(comparison: pd.DataFrame) -> None:
    s = summarize(comparison)
    print("\n=== Prediction accuracy (finished, known-team matches) ===")
    print(f"  Played             : {s['played']}")
    print(f"  Outcomes correct   : {s['outcome_correct']}/{s['played']} ({s['outcome_pct']:.0%})")
    print(f"  Exact scorelines   : {s['exact']}/{s['played']} ({s['exact_pct']:.0%})")
    print(f"  Home goals exact   : {s['home_goals_correct']}/{s['played']} ({s['home_goals_pct']:.0%})  MAE {s['home_goals_mae']:.2f}")
    print(f"  Away goals exact   : {s['away_goals_correct']}/{s['played']} ({s['away_goals_pct']:.0%})  MAE {s['away_goals_mae']:.2f}")
    print(f"  Live now           : {s['live']}")


def run_fetch() -> pd.DataFrame:
    print("\n=== Fetching WC 2026 results from football-data.org ===")
    results = fetch_wc2026_results(write_csv=True)
    _print_status_counts(results)
    return results


def run_from_csv() -> pd.DataFrame | None:
    if not _RESULTS_CSV.exists():
        print(f"  No results file found at {_RESULTS_CSV}. Run without --summary first.")
        return None
    return pd.read_csv(_RESULTS_CSV)


def main() -> None:
    parser = argparse.ArgumentParser(description="WC 2026 results fetcher + prediction comparison")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print the comparison summary from the existing CSV without calling the API",
    )
    args = parser.parse_args()

    if args.summary:
        results = run_from_csv()
        if results is None:
            sys.exit(0)
    else:
        try:
            results = run_fetch()
        except (EnvironmentError, RuntimeError) as exc:
            print(f"  ERROR: {exc}")
            print("  Make sure FD_API_KEY is set in .env.")
            sys.exit(1)

    fixtures, predictions = _load_fixtures_and_predictions()
    if fixtures is None or predictions is None:
        print("\n  Skipping comparison — fixtures or predictions file not found.")
        return

    comparison = build_comparison(fixtures, predictions, results)
    _print_recent_finals(comparison)
    _print_summary(comparison)


if __name__ == "__main__":
    main()
