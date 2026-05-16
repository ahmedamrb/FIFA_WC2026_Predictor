"""Download and verify historical World Cup data files from the openfootball project.

Usage (from project root):
    python scripts/fetch_openfootball_data.py
    python scripts/fetch_openfootball_data.py --skip-existing
    python scripts/fetch_openfootball_data.py --dry-run
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root (parent of scripts/) is on sys.path so `src` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest import fetch_openfootball_data  # noqa: E402

# Mirrors the constant defined in fetch_openfootball_data so --dry-run works
# without importing private internals.
_TOURNAMENTS = {
    "wc1998": ("1998--france",              ["cup.txt", "cup_finals.txt"]),
    "wc2002": ("2002--south-korea-n-japan", ["cup.txt", "cup_finals.txt"]),
    "wc2006": ("2006--germany",             ["cup.txt", "cup_finals.txt"]),
    "wc2010": ("2010--south-africa",        ["cup.txt", "cup_finals.txt"]),
    "wc2014": ("2014--brazil",              ["cup.txt", "cup_finals.txt"]),
    "wc2018": ("2018--russia",              ["cup.txt", "cup_finals.txt"]),
    "wc2022": ("2022--qatar",               ["cup.txt", "cup_finals.txt"]),
}
_BASE_URL = "https://raw.githubusercontent.com/openfootball/worldcup/master"

# WC group stage has 48 matches (2026 format applies to 2026 only).
# Historic WC had 64 matches total.  Flag if total differs from this.
_EXPECTED_MATCH_LINES = 64


def verify_downloads(downloaded: dict) -> None:
    """Print a verification summary for every downloaded tournament.

    Args:
        downloaded: Mapping returned by fetch_openfootball_data(), wc_key →
                    list of local Paths.
    """
    header = f"{'Tournament':<12} | {'cup.txt':>7} | {'cup_finals.txt':>14} | {'Total':>5} | Status"
    separator = "-" * len(header)
    print()
    print(header)
    print(separator)

    any_warning = False
    for wc_key, paths in downloaded.items():
        counts: dict[str, int] = {}
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                counts[path.name] = 0
                continue
            counts[path.name] = sum(1 for line in text.splitlines() if " @ " in line)

        cup_count = counts.get("cup.txt", 0)
        finals_count = counts.get("cup_finals.txt", 0)
        total = cup_count + finals_count
        status = "✓" if total == _EXPECTED_MATCH_LINES else "✗"

        print(
            f"{wc_key:<12} | {cup_count:>7} | {finals_count:>14} | {total:>5} | {status}"
        )

        if total != _EXPECTED_MATCH_LINES:
            any_warning = True

    print(separator)
    if any_warning:
        print(
            f"\nWARNING: One or more tournaments have a match-line count != {_EXPECTED_MATCH_LINES}. "
            "The files may be incomplete or the format may differ."
        )
    else:
        print(f"\nAll tournaments verified ({_EXPECTED_MATCH_LINES} match lines each).")


def _dry_run() -> None:
    """Print the list of URLs that would be fetched, then exit."""
    print("Dry-run mode — the following URLs would be fetched:\n")
    for wc_key, (upstream_dir, filenames) in _TOURNAMENTS.items():
        for filename in filenames:
            url = f"{_BASE_URL}/{upstream_dir}/{filename}"
            print(f"  [{wc_key}] {url}")
    print()


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch historical WC data files from openfootball."
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip files that already exist and are non-empty.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be fetched without downloading anything.",
    )
    args = parser.parse_args()

    if args.dry_run:
        _dry_run()
        return

    print("Fetching openfootball historical WC data...\n")
    downloaded = fetch_openfootball_data(skip_existing=args.skip_existing)

    verify_downloads(downloaded)


if __name__ == "__main__":
    main()
