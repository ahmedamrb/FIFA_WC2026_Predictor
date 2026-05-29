"""Bookmaker odds ingestion, normalization, and loading utilities.

Public API
----------
fetch_live_odds(api_key, regions, bookmaker, append)
    Fetch WC 2026 1X2 odds from The Odds API and optionally append to the
    canonical odds table.

import_historical_odds(path)
    Import a pre-match odds archive CSV (WC 2018 / WC 2022) and normalise it
    to the canonical schema.

normalize_team_name(name, name_map)
    Resolve a provider team name to the canonical model name via team_name_map.

validate_odds_df(df)
    Assert schema and value constraints; drop invalid rows with a warning.

get_latest_odds(df)
    Deduplicate to the most-recent fetched_at snapshot per fixture.

load_odds_for_backtest(mode, historical_paths)
    Unified loader used by the backtest pipeline.  mode='real' merges the
    canonical table with any supplied historical archive files;
    mode='stub_2.0' returns an empty DataFrame so callers fall back to 2.0.

append_to_canonical(new_df)
    Append validated rows to data/bookmaker_odds.csv, deduplicating by
    (match_date, home_team, away_team, source, fetched_at).
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _REPO_ROOT / "data" / "raw"
_ODDS_PATH = _REPO_ROOT / "data" / "bookmaker_odds.csv"
_TEAM_NAME_MAP_PATH = _RAW_DIR / "team_name_map.csv"

# Canonical odds schema
ODDS_COLUMNS = [
    "match_date",
    "home_team",
    "away_team",
    "home_win_odds",
    "draw_odds",
    "away_win_odds",
    "source",
    "fetched_at",
]

# The Odds API — try multiple sport keys (key changes once the tournament starts)
_ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
_ODDS_API_SPORT_CANDIDATES = [
    "soccer_fifa_world_cup_2026",
    "soccer_worldcup",
    "soccer_world_cup",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_team_name_map() -> dict:
    """Return a dict mapping provider/archive team names to canonical model names."""
    if not _TEAM_NAME_MAP_PATH.exists():
        return {}
    df = pd.read_csv(_TEAM_NAME_MAP_PATH)
    mapping: dict = {}
    for _, row in df.iterrows():
        fixture = str(row.get("fixture_name", "")).strip()
        canonical = str(row.get("results_name", "")).strip()
        if fixture and canonical:
            mapping[fixture] = canonical
    return mapping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_team_name(name: str, name_map: dict) -> str:
    """Resolve a provider or archive team name to the canonical model name.

    Falls back to the original name (stripped) if no mapping exists.
    """
    return name_map.get(name.strip(), name.strip())


def validate_odds_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and value constraints; drop malformed rows with a warning.

    Raises:
        ValueError: If any required column is entirely absent.

    Returns:
        Cleaned copy of *df* with invalid rows removed.
    """
    required = {
        "match_date", "home_team", "away_team",
        "home_win_odds", "draw_odds", "away_win_odds",
    }
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Odds DataFrame missing required columns: {missing_cols}")

    df = df.copy()
    initial_len = len(df)

    # Coerce odds to numeric
    odds_cols = ["home_win_odds", "draw_odds", "away_win_odds"]
    for col in odds_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where any odds is null or < 1.0 (invalid decimal odds)
    valid_mask = (df[odds_cols] >= 1.0).all(axis=1) & df[odds_cols].notna().all(axis=1)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning("Dropping %d rows with invalid odds (< 1.0 or null).", invalid_count)
    df = df[valid_mask]

    # Drop rows missing key match fields
    df = df.dropna(subset=["match_date", "home_team", "away_team"])

    dropped = initial_len - len(df)
    if dropped > 0:
        logger.info("validate_odds_df: dropped %d rows, %d remain.", dropped, len(df))

    return df.reset_index(drop=True)


def get_latest_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate to the most-recent fetched_at snapshot per fixture.

    Groups by (match_date, home_team, away_team) and keeps the row with the
    latest fetched_at timestamp, regardless of source.

    Returns:
        Deduplicated DataFrame sorted by match_date ascending.
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce")
    df = df.sort_values("fetched_at", ascending=False)
    deduped = df.groupby(
        ["match_date", "home_team", "away_team"], as_index=False, sort=False
    ).first()
    return deduped.sort_values("match_date").reset_index(drop=True)


def fetch_live_odds(
    api_key: Optional[str] = None,
    regions: str = "uk,eu",
    bookmaker: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Fetch live 1X2 odds from The Odds API for WC 2026 fixtures.

    Tries the registered FIFA World Cup 2026 sport keys in sequence and uses
    the first that returns a 200 response.  Team names are normalised via
    team_name_map.csv before any rows are saved.

    Args:
        api_key: The Odds API key. Reads from ODDS_API_KEY env var if None.
        regions: Comma-separated bookmaker regions (default ``"uk,eu"``).
        bookmaker: Specific bookmaker key/title to select.  Uses the first
            available bookmaker when None.
        append: When True, append the fetched rows to data/bookmaker_odds.csv.

    Returns:
        DataFrame with ODDS_COLUMNS schema.

    Raises:
        RuntimeError: If ODDS_API_KEY is missing or all sport-key requests fail.
    """
    key = api_key or os.getenv("ODDS_API_KEY")
    if not key:
        raise RuntimeError(
            "ODDS_API_KEY not set. Add it to .env or pass api_key= explicitly."
        )

    name_map = _load_team_name_map()
    resp = None

    for sport in _ODDS_API_SPORT_CANDIDATES:
        url = f"{_ODDS_API_BASE}/{sport}/odds"
        try:
            resp = requests.get(
                url,
                params={
                    "regions": regions,
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                    "apiKey": key,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                logger.info("The Odds API: using sport key '%s'.", sport)
                break
            resp = None
        except requests.RequestException as exc:
            logger.warning("Request to %s failed: %s", url, exc)
            resp = None

    if resp is None or resp.status_code != 200:
        status = resp.status_code if resp is not None else "no response"
        raise RuntimeError(
            f"The Odds API returned status {status}. "
            "Check ODDS_API_KEY and that the tournament is live/upcoming."
        )

    events = resp.json()
    if not events:
        logger.warning("The Odds API returned 0 events.")
        return pd.DataFrame(columns=ODDS_COLUMNS)

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rows = []

    for ev in events:
        match_date = ev.get("commence_time", "")[:10]
        raw_home = ev.get("home_team", "")
        raw_away = ev.get("away_team", "")
        home_team = normalize_team_name(raw_home, name_map)
        away_team = normalize_team_name(raw_away, name_map)

        bookmakers_list = ev.get("bookmakers", [])
        if not bookmakers_list:
            continue

        # Pick the specified bookmaker or fall back to the first available
        bk = None
        if bookmaker:
            for b in bookmakers_list:
                if b.get("key") == bookmaker or b.get("title") == bookmaker:
                    bk = b
                    break
        if bk is None:
            bk = bookmakers_list[0]

        markets = bk.get("markets", [])
        h2h = next((m for m in markets if m.get("key") == "h2h"), None)
        if not h2h:
            continue

        outcomes = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}
        home_odds = outcomes.get(raw_home)
        away_odds = outcomes.get(raw_away)
        draw_odds_val = outcomes.get("Draw")

        if not all([home_odds, draw_odds_val, away_odds]):
            continue

        rows.append({
            "match_date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_win_odds": float(home_odds),
            "draw_odds": float(draw_odds_val),
            "away_win_odds": float(away_odds),
            "source": bk.get("title", "TheOddsAPI"),
            "fetched_at": fetched_at,
        })

    new_df = pd.DataFrame(rows, columns=ODDS_COLUMNS)
    new_df = validate_odds_df(new_df)
    logger.info("fetch_live_odds: fetched %d rows.", len(new_df))

    if append and not new_df.empty:
        append_to_canonical(new_df)

    return new_df


def import_historical_odds(path: Union[str, Path]) -> pd.DataFrame:
    """Import a pre-match odds archive CSV (e.g. WC 2018 / WC 2022) into canonical schema.

    The input CSV must contain at minimum:
        match_date, home_team, away_team, home_win_odds, draw_odds, away_win_odds

    Optional columns ``source`` and ``fetched_at`` are filled with defaults if
    absent.  Team names are normalised via team_name_map.csv.

    Args:
        path: Path to the historical archive CSV.

    Returns:
        Validated DataFrame with ODDS_COLUMNS columns.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Historical odds file not found: {path}")

    df = pd.read_csv(path, comment="#")
    if df.empty:
        return pd.DataFrame(columns=ODDS_COLUMNS)

    name_map = _load_team_name_map()

    df["home_team"] = df["home_team"].astype(str).apply(
        lambda n: normalize_team_name(n, name_map)
    )
    df["away_team"] = df["away_team"].astype(str).apply(
        lambda n: normalize_team_name(n, name_map)
    )
    df["match_date"] = (
        pd.to_datetime(df["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    )

    if "source" not in df.columns:
        df["source"] = f"archive:{path.stem}"
    if "fetched_at" not in df.columns:
        df["fetched_at"] = df["match_date"]

    df = validate_odds_df(df)
    return df[ODDS_COLUMNS].reset_index(drop=True)


def append_to_canonical(new_df: pd.DataFrame) -> None:
    """Append validated odds rows to data/bookmaker_odds.csv.

    Deduplicates on (match_date, home_team, away_team, source, fetched_at) so
    re-running a fetch does not produce duplicate rows.
    """
    new_df = validate_odds_df(new_df)
    if new_df.empty:
        return

    if _ODDS_PATH.exists():
        existing = pd.read_csv(_ODDS_PATH)
        combined = pd.concat([existing, new_df[ODDS_COLUMNS]], ignore_index=True)
    else:
        combined = new_df[ODDS_COLUMNS].copy()

    combined = combined.drop_duplicates(
        subset=["match_date", "home_team", "away_team", "source", "fetched_at"]
    )
    combined.to_csv(_ODDS_PATH, index=False)
    logger.info("Canonical odds table updated: %d total rows.", len(combined))


def load_odds_for_backtest(
    mode: str = "real",
    historical_paths: Optional[List[Union[str, Path]]] = None,
) -> pd.DataFrame:
    """Unified odds loader for the backtest pipeline.

    Args:
        mode: ``"real"`` — merge the canonical table with any historical
              archive files supplied via *historical_paths*, then deduplicate
              to the latest snapshot per fixture.
              ``"stub_2.0"`` — return an empty DataFrame; callers then fall
              back to the explicit 2.0 baseline (no silent masking).
        historical_paths: List of archive CSV paths to merge into the odds
            DataFrame for historical coverage (WC 2018 / WC 2022).

    Returns:
        DataFrame with ODDS_COLUMNS schema, or an empty DataFrame for stub mode.
    """
    if mode == "stub_2.0":
        return pd.DataFrame(columns=ODDS_COLUMNS)

    frames = []

    if _ODDS_PATH.exists():
        canonical = pd.read_csv(_ODDS_PATH)
        frames.append(canonical)

    if historical_paths:
        for p in historical_paths:
            try:
                hist = import_historical_odds(p)
                frames.append(hist)
                logger.info("Loaded historical odds from %s (%d rows).", p, len(hist))
            except FileNotFoundError:
                logger.warning(
                    "Historical odds file not found (skipping): %s", p
                )

    if not frames:
        return pd.DataFrame(columns=ODDS_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["match_date", "home_team", "away_team", "source", "fetched_at"],
        keep="last",
    )
    combined = get_latest_odds(combined)
    return combined[ODDS_COLUMNS]
