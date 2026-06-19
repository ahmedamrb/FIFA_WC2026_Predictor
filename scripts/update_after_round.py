"""Refresh predictions after a round of WC 2026 matches — one command.

Run this after each round (or any time new results land). It chains the steps
that actually matter once the tournament is live:

    [1/5] fetch live/full-time scores      -> data/processed/wc2026_results.csv
    [2/5] backfill finished scores         -> data/raw/results.csv   (the missing link)
    [3/5] re-run feature engineering       -> data/processed/features_*.parquet
    [4/5] regenerate predictions           -> data/processed/wc2026_predictions.csv
    [5/5] print prediction-vs-actual accuracy

Step 2 is the reason a plain pipeline re-run is not enough: the WC 2026 fixtures
live in data/raw/results.csv as *score-less* rows, which feature engineering
treats as unplayed. Until the finished scores are written onto those rows, every
re-run produces identical predictions (see src/data/backfill.py).

Tuning and training are intentionally NOT part of this flow: a round of ~16
matches against a ~26k-row training set won't move the models, and tuning is a
4-6 hour job. Re-run scripts/run_tuning.py + scripts/train.py only deliberately.
(scripts/predict.py is an unused stub; predictions come from
scripts/precompute_predictions.py.)

This script only updates the working tree — review the diff and commit yourself.

Usage
-----
    python scripts/update_after_round.py                 # full run (needs FD_API_KEY)
    python scripts/update_after_round.py --no-fetch      # use existing wc2026_results.csv (offline)
    python scripts/update_after_round.py --skip-features --skip-predictions   # data sync only
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.ingest import fetch_wc2026_results  # noqa: E402
from src.data.backfill import backfill_raw_results  # noqa: E402
from src.data.preprocess import export_features  # noqa: E402
from src.evaluation.live_tracking import build_comparison, summarize  # noqa: E402
import scripts.precompute_predictions as precompute  # noqa: E402

_RAW = _REPO_ROOT / "data" / "raw"
_PROCESSED = _REPO_ROOT / "data" / "processed"
_RESULTS_CSV = _PROCESSED / "wc2026_results.csv"


def _print_accuracy() -> None:
    fixtures_path = _RAW / "wc2026_fixtures_flat.csv"
    pred_path = _PROCESSED / "wc2026_predictions.csv"
    if not (fixtures_path.exists() and pred_path.exists() and _RESULTS_CSV.exists()):
        print("  Skipping accuracy summary — fixtures, predictions or results missing.")
        return

    fixtures = pd.read_csv(fixtures_path)
    predictions = pd.read_csv(pred_path, index_col=0)
    results = pd.read_csv(_RESULTS_CSV)

    comparison = build_comparison(fixtures, predictions, results)
    s = summarize(comparison)
    print("\n=== Prediction accuracy (finished, known-team matches) ===")
    print(f"  Played             : {s['played']}")
    print(f"  Outcomes correct   : {s['outcome_correct']}/{s['played']} ({s['outcome_pct']:.0%})")
    print(f"  Exact scorelines   : {s['exact']}/{s['played']} ({s['exact_pct']:.0%})")
    print(f"  Home goals exact   : {s['home_goals_correct']}/{s['played']} ({s['home_goals_pct']:.0%})  MAE {s['home_goals_mae']:.2f}")
    print(f"  Away goals exact   : {s['away_goals_correct']}/{s['played']} ({s['away_goals_pct']:.0%})  MAE {s['away_goals_mae']:.2f}")
    print(f"  Live now           : {s['live']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="Skip the football-data.org call; use the existing wc2026_results.csv (offline).",
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Skip the feature-engineering rebuild (step 3).",
    )
    parser.add_argument(
        "--skip-predictions", action="store_true",
        help="Skip regenerating wc2026_predictions.csv (step 4).",
    )
    args = parser.parse_args()

    # [1/5] fetch ----------------------------------------------------------
    print("\n[1/5] Fetching live WC 2026 results ...")
    if args.no_fetch:
        if not _RESULTS_CSV.exists():
            print(f"  --no-fetch given but {_RESULTS_CSV} does not exist. Aborting.")
            sys.exit(1)
        print(f"  --no-fetch: using existing {_RESULTS_CSV}")
    else:
        try:
            fetch_wc2026_results(write_csv=True)
        except (EnvironmentError, RuntimeError) as exc:
            print(f"  ERROR: {exc}")
            print("  Set FD_API_KEY in .env, or re-run with --no-fetch to use the saved CSV.")
            sys.exit(1)

    # [2/5] backfill -------------------------------------------------------
    print("\n[2/5] Backfilling finished scores into data/raw/results.csv ...")
    summary = backfill_raw_results()
    if summary["updated"] == 0 and summary["appended"] == 0:
        print("  No finished matches written (none played yet, or already up to date).")

    # [3/5] feature engineering -------------------------------------------
    if args.skip_features:
        print("\n[3/5] Skipping feature engineering (--skip-features).")
    else:
        print("\n[3/5] Rebuilding feature matrices ...")
        export_features()

    # [4/5] predictions ----------------------------------------------------
    if args.skip_predictions:
        print("\n[4/5] Skipping prediction regeneration (--skip-predictions).")
    else:
        print("\n[4/5] Regenerating predictions ...")
        precompute.main()

    # [5/5] accuracy summary ----------------------------------------------
    print("\n[5/5] Accuracy summary")
    _print_accuracy()

    print("\nDone. Changes are in the working tree — review with `git diff --stat` and commit when ready.")


if __name__ == "__main__":
    main()
