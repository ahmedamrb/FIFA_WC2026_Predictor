"""Fetch live WC 2026 odds and optionally import historical archive odds.

Usage
-----
# Fetch live odds from The Odds API and append to data/bookmaker_odds.csv
python scripts/fetch_odds.py --live

# Import historical archive odds for WC 2018 and WC 2022
python scripts/fetch_odds.py --historical

# Do both
python scripts/fetch_odds.py --live --historical

# Specify a different bookmaker (by key, e.g. bet365, unibet)
python scripts/fetch_odds.py --live --bookmaker bet365

Prerequisites
-------------
1. Add ODDS_API_KEY=<your_key> to .env  (https://the-odds-api.com — free tier)
2. Populate data/raw/historical_odds_wc2018.csv and
   data/raw/historical_odds_wc2022.csv with archived odds before using
   --historical.  See those files for the required column schema.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.odds import (
    fetch_live_odds,
    import_historical_odds,
    append_to_canonical,
    load_odds_for_backtest,
    ODDS_COLUMNS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
_DEFAULT_BOOKMAKER = "Paddy Power"
_HISTORICAL_PATHS = {
    "wc2018": _RAW_DIR / "historical_odds_wc2018.csv",
    "wc2022": _RAW_DIR / "historical_odds_wc2022.csv",
}


def run_live(bookmaker: str | None = None) -> None:
    """Fetch live WC 2026 odds from The Odds API."""
    print("\n=== Fetching live WC 2026 odds from The Odds API ===")
    try:
        df = fetch_live_odds(bookmaker=bookmaker, append=True)
        if df.empty:
            print("  No live odds returned (tournament may not be live yet).")
        else:
            print(f"  Fetched and saved {len(df)} fixture odds rows.")
            print(df[ODDS_COLUMNS].head(5).to_string(index=False))
    except RuntimeError as exc:
        print(f"  ERROR: {exc}")
        print("  Make sure ODDS_API_KEY is set in .env and the tournament is live/upcoming.")


def run_historical() -> None:
    """Import historical odds for WC 2018 and WC 2022 from archive CSVs."""
    print("\n=== Importing historical odds ===")
    for label, path in _HISTORICAL_PATHS.items():
        if not path.exists():
            print(f"  [{label}] SKIPPED — file not found: {path}")
            print(
                f"         Create the file using the schema in data/raw/historical_odds_wc2018.csv"
                " (see README in that file)."
            )
            continue
        try:
            df = import_historical_odds(path)
            append_to_canonical(df)
            print(f"  [{label}] Imported and appended {len(df)} rows.")
        except Exception as exc:
            print(f"  [{label}] ERROR: {exc}")


def run_summary() -> None:
    """Print current state of the canonical odds table."""
    print("\n=== Canonical odds table summary ===")
    odds_df = load_odds_for_backtest(mode="real")
    if odds_df.empty:
        print("  No odds in canonical table yet.")
        return

    total = len(odds_df)
    by_source = odds_df.groupby("source").size().to_dict()
    earliest = odds_df["match_date"].min()
    latest = odds_df["match_date"].max()

    print(f"  Total rows : {total}")
    print(f"  Date range : {earliest} to {latest}")
    print(f"  By source  :")
    for src, count in sorted(by_source.items()):
        print(f"    {src:<30} {count:>4} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Odds fetcher for FIFA WC 2026 Predictor")
    parser.add_argument("--live", action="store_true", help="Fetch live WC 2026 odds from The Odds API")
    parser.add_argument("--historical", action="store_true", help="Import historical WC 2018/2022 archive odds")
    parser.add_argument(
        "--bookmaker",
        default=_DEFAULT_BOOKMAKER,
        help=(
            "Specific bookmaker key/title for live fetch "
            f"(default: {_DEFAULT_BOOKMAKER})"
        ),
    )
    parser.add_argument("--summary", action="store_true", help="Print canonical odds table summary")
    args = parser.parse_args()

    if not any([args.live, args.historical, args.summary]):
        parser.print_help()
        print("\nNo action specified. Use --live, --historical, or --summary.")
        sys.exit(0)

    if args.live:
        run_live(bookmaker=args.bookmaker)
    if args.historical:
        run_historical()

    # Always print summary at the end
    run_summary()


if __name__ == "__main__":
    main()
