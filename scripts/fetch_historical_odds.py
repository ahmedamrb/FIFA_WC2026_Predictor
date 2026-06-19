"""Acquire real historical 1X2 closing odds for WC 2018 / WC 2022.

This populates ``data/raw/historical_odds_wc{2018,2022}.csv`` (currently empty
templates) so the backtest can validate value bets against genuine prices
instead of the stubbed 2.0 baseline.  ``scripts/run_backtest.py`` reads those
files directly; ``scripts/fetch_odds.py --historical`` can then import them into
the canonical odds table.

Prioritised source chain (each tournament is sourced independently; the first
source that yields rows wins, the rest are skipped):

  1. The Odds API *historical* endpoint  - paid tier, 2020-06 onward -> WC 2022 ONLY.
  2. Free public compiled CSV (URL)       - e.g. the FIFA-World-Cup-Prediction repo
                                             (covers WC 2018) or a Kaggle export.
  3. Local drop-in CSV                     - data/raw/historical_odds_wc{YEAR}_raw.csv
                                             that you export manually (e.g. from
                                             OddsPortal, which is JS-rendered and
                                             ToS-restricted so it is NOT auto-scraped).

Any source can be added/swapped by editing the SOURCES registry below.  CSV
sources accept flexible column names (football-data.co.uk B365H/D/A & AvgH/D/A,
generic home_win_odds/draw_odds/away_win_odds, "1"/"X"/"2", etc.).

Usage
-----
python scripts/fetch_historical_odds.py                 # both tournaments, full chain
python scripts/fetch_historical_odds.py --only wc2018   # one tournament
python scripts/fetch_historical_odds.py --dry-run       # report coverage, write nothing

Prerequisites
-------------
ODDS_API_KEY in .env enables source #1 (needs a paid historical tier).  Without
it the script falls through to the CSV sources automatically.
"""

import argparse
import logging
import os
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.odds import (  # noqa: E402
    ODDS_COLUMNS,
    normalize_team_name,
    validate_odds_df,
    _load_team_name_map,
    _ODDS_API_SPORT_CANDIDATES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RAW_DIR = _REPO_ROOT / "data" / "raw"
_RESULTS_CSV = _RAW_DIR / "results.csv"

_HISTORICAL_API_BASE = "https://api.the-odds-api.com/v4/historical/sports"

# Per-tournament archive destination + the actual match-date window used to drive
# the API snapshot queries and to filter results.csv.
_ARCHIVES = {
    "wc2018": _RAW_DIR / "historical_odds_wc2018.csv",
    "wc2022": _RAW_DIR / "historical_odds_wc2022.csv",
}
_WINDOWS = {
    "wc2018": ("2018-06-14", "2018-07-15"),
    "wc2022": ("2022-11-20", "2022-12-18"),
}

# Source chain per tournament. Edit here to add/swap a source.
#   type: "odds_api" | "csv_url" | "local_csv"
SOURCES = {
    "wc2018": [
        # WC 2018 predates The Odds API historical window - rely on CSV sources.
        {
            "type": "csv_url",
            "location": "https://raw.githubusercontent.com/mrthlinh/"
            "FIFA-World-Cup-Prediction/master/data/results.csv",
        },
        {"type": "local_csv", "location": _RAW_DIR / "historical_odds_wc2018_raw.csv"},
    ],
    "wc2022": [
        {"type": "odds_api", "tournament": "wc2022"},
        {"type": "local_csv", "location": _RAW_DIR / "historical_odds_wc2022_raw.csv"},
    ],
}

# Flexible column-name candidates for CSV sources (first match wins, case-insensitive).
_COL_CANDIDATES = {
    "match_date": ["match_date", "date", "datetime", "kickoff"],
    "home_team": ["home_team", "home", "hometeam", "team_home"],
    "away_team": ["away_team", "away", "awayteam", "team_away"],
    "home_win_odds": ["home_win_odds", "avgh", "b365h", "psh", "home_odds", "odds_home", "1"],
    "draw_odds": ["draw_odds", "avgd", "b365d", "psd", "draw_odds", "odds_draw", "x"],
    "away_win_odds": ["away_win_odds", "avga", "b365a", "psa", "away_odds", "odds_away", "2"],
}


# ---------------------------------------------------------------------------
# Normalisation - every source funnels through here
# ---------------------------------------------------------------------------
def _pick_column(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _to_canonical(df: pd.DataFrame, source: str, window: tuple[str, str]) -> pd.DataFrame:
    """Map an arbitrary odds DataFrame to ODDS_COLUMNS, normalise, and window-filter."""
    if df is None or df.empty:
        return pd.DataFrame(columns=ODDS_COLUMNS)

    resolved = {canon: _pick_column(df, cands) for canon, cands in _COL_CANDIDATES.items()}
    missing = [k for k, v in resolved.items() if v is None]
    if missing:
        logger.warning("Source '%s' missing columns %s - skipping.", source, missing)
        return pd.DataFrame(columns=ODDS_COLUMNS)

    out = pd.DataFrame({
        "match_date": pd.to_datetime(df[resolved["match_date"]], errors="coerce").dt.strftime("%Y-%m-%d"),
        "home_team": df[resolved["home_team"]].astype(str),
        "away_team": df[resolved["away_team"]].astype(str),
        "home_win_odds": pd.to_numeric(df[resolved["home_win_odds"]], errors="coerce"),
        "draw_odds": pd.to_numeric(df[resolved["draw_odds"]], errors="coerce"),
        "away_win_odds": pd.to_numeric(df[resolved["away_win_odds"]], errors="coerce"),
    })

    name_map = _load_team_name_map()
    out["home_team"] = out["home_team"].apply(lambda n: normalize_team_name(n, name_map))
    out["away_team"] = out["away_team"].apply(lambda n: normalize_team_name(n, name_map))
    out["source"] = f"archive:{source}"
    out["fetched_at"] = out["match_date"]

    start, end = window
    out = out[(out["match_date"] >= start) & (out["match_date"] <= end)]
    out = out.dropna(subset=["home_win_odds", "draw_odds", "away_win_odds"])
    out = validate_odds_df(out)
    return out[ODDS_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------
def _load_csv_url(url: str, source: str, window: tuple[str, str]) -> pd.DataFrame:
    logger.info("[%s] Trying CSV URL: %s", source, url)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
    except Exception as exc:  # noqa: BLE001 - any failure should fall through
        logger.warning("[%s] CSV URL failed: %s", source, exc)
        return pd.DataFrame(columns=ODDS_COLUMNS)
    return _to_canonical(df, source=f"url:{url.rsplit('/', 1)[-1]}", window=window)


def _load_local_csv(path: Path, source: str, window: tuple[str, str]) -> pd.DataFrame:
    if not Path(path).exists():
        logger.info("[%s] No local drop-in at %s - skipping.", source, path)
        return pd.DataFrame(columns=ODDS_COLUMNS)
    logger.info("[%s] Reading local drop-in: %s", source, path)
    try:
        df = pd.read_csv(path, comment="#")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] Local CSV read failed: %s", source, exc)
        return pd.DataFrame(columns=ODDS_COLUMNS)
    return _to_canonical(df, source=f"local:{Path(path).name}", window=window)


def _wc_fixture_dates(window: tuple[str, str]) -> list[str]:
    """Distinct WC match dates within the window, derived from results.csv."""
    if not _RESULTS_CSV.exists():
        return []
    res = pd.read_csv(_RESULTS_CSV)
    res["date"] = pd.to_datetime(res["date"], errors="coerce")
    start, end = pd.to_datetime(window[0]), pd.to_datetime(window[1])
    mask = (res["date"] >= start) & (res["date"] <= end)
    if "tournament" in res.columns:
        mask &= res["tournament"].astype(str).str.contains("World Cup", case=False, na=False)
    dates = sorted(res.loc[mask, "date"].dt.strftime("%Y-%m-%d").dropna().unique())
    return list(dates)


def _load_odds_api_historical(tournament: str, window: tuple[str, str]) -> pd.DataFrame:
    """Best-effort fetch from The Odds API historical endpoint (paid tier)."""
    key = os.getenv("ODDS_API_KEY")
    if not key:
        logger.info("[%s] ODDS_API_KEY not set - skipping API source.", tournament)
        return pd.DataFrame(columns=ODDS_COLUMNS)

    dates = _wc_fixture_dates(window)
    if not dates:
        logger.info("[%s] No fixture dates from results.csv - skipping API source.", tournament)
        return pd.DataFrame(columns=ODDS_COLUMNS)

    name_map = _load_team_name_map()
    rows: list[dict] = []
    for sport in _ODDS_API_SPORT_CANDIDATES:
        rows.clear()
        ok = True
        for d in dates:
            # Snapshot ~24h before the day's matches -> closing-ish prices.
            snap = f"{d}T12:00:00Z"
            url = f"{_HISTORICAL_API_BASE}/{sport}/odds"
            try:
                resp = requests.get(
                    url,
                    params={
                        "regions": "uk,eu",
                        "markets": "h2h",
                        "oddsFormat": "decimal",
                        "date": snap,
                        "apiKey": key,
                    },
                    timeout=20,
                )
            except requests.RequestException as exc:
                logger.warning("[%s] API request error (%s): %s", tournament, sport, exc)
                ok = False
                break
            if resp.status_code != 200:
                logger.info(
                    "[%s] API sport '%s' returned %s (tier/coverage) - trying next.",
                    tournament, sport, resp.status_code,
                )
                ok = False
                break
            payload = resp.json()
            events = payload.get("data", payload) if isinstance(payload, dict) else payload
            for ev in events or []:
                parsed = _parse_h2h_event(ev, name_map)
                if parsed:
                    rows.append(parsed)
        if ok and rows:
            logger.info("[%s] API source '%s' yielded %d rows.", tournament, sport, len(rows))
            break

    if not rows:
        return pd.DataFrame(columns=ODDS_COLUMNS)
    df = pd.DataFrame(rows)
    return _to_canonical(df, source=f"oddsapi:{tournament}", window=window)


def _parse_h2h_event(ev: dict, name_map: dict) -> dict | None:
    """Extract median-ish 1X2 decimal odds from one historical API event."""
    home = normalize_team_name(str(ev.get("home_team", "")), name_map)
    away = normalize_team_name(str(ev.get("away_team", "")), name_map)
    date = str(ev.get("commence_time", ""))[:10]
    if not (home and away and date):
        return None
    # Use the first bookmaker's h2h market.
    for bk in ev.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt.get("key") != "h2h":
                continue
            prices = {o.get("name"): o.get("price") for o in mkt.get("outcomes", [])}
            h = prices.get(ev.get("home_team"))
            a = prices.get(ev.get("away_team"))
            d = prices.get("Draw")
            if h and a and d:
                return {
                    "match_date": date,
                    "home_team": home,
                    "away_team": away,
                    "home_win_odds": h,
                    "draw_odds": d,
                    "away_win_odds": a,
                }
    return None


_LOADERS = {
    "csv_url": lambda s, t, w: _load_csv_url(s["location"], source=t, window=w),
    "local_csv": lambda s, t, w: _load_local_csv(s["location"], source=t, window=w),
    "odds_api": lambda s, t, w: _load_odds_api_historical(tournament=t, window=w),
}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def source_tournament(tournament: str, dry_run: bool) -> int:
    """Walk the source chain for one tournament; write the archive on first hit."""
    window = _WINDOWS[tournament]
    print(f"\n=== {tournament.upper()} ===")
    for spec in SOURCES[tournament]:
        loader = _LOADERS.get(spec["type"])
        if loader is None:
            continue
        df = loader(spec, tournament, window)
        if df is not None and not df.empty:
            print(f"  Source '{spec['type']}' -> {len(df)} normalised rows.")
            if dry_run:
                print("  [dry-run] not written.")
                print(df.head(5).to_string(index=False))
            else:
                _write_archive(tournament, df)
            return len(df)
        print(f"  Source '{spec['type']}' -> 0 rows, falling through.")
    print(f"  No source produced odds for {tournament}. Archive left unchanged.")
    print("  Tip: export odds to "
          f"{_RAW_DIR / (tournament.replace('wc', 'historical_odds_wc') + '_raw.csv')} "
          "(OddsPortal etc.) and re-run.")
    return 0


def _write_archive(tournament: str, df: pd.DataFrame) -> None:
    """Write normalised rows into the archive CSV, preserving the header comments."""
    path = _ARCHIVES[tournament]
    header_comment = (
        f"# Historical 1X2 closing odds for {tournament.upper()}.\n"
        f"# Auto-populated by scripts/fetch_historical_odds.py.\n"
        f"# Schema: {','.join(ODDS_COLUMNS)}\n"
    )
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(header_comment)
        df[ODDS_COLUMNS].to_csv(fh, index=False)
    print(f"  Wrote {len(df)} rows -> {path.relative_to(_REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire historical WC 2018/2022 odds.")
    parser.add_argument("--only", choices=list(_ARCHIVES), help="Source a single tournament.")
    parser.add_argument("--dry-run", action="store_true", help="Report coverage without writing.")
    args = parser.parse_args()

    targets = [args.only] if args.only else list(_ARCHIVES)
    total = 0
    for t in targets:
        total += source_tournament(t, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print(f"  Total rows sourced: {total}")
    if total == 0:
        print("  No real odds acquired. The vig-removal / Kelly / no-fake-edge")
        print("  fixes still apply; only the real-odds backtest is unavailable.")
    else:
        print("  Next: python scripts/run_backtest.py   (expect 'Odds matched' > 0)")
    print("=" * 60)


if __name__ == "__main__":
    main()
