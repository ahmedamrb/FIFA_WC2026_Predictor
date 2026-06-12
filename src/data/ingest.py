"""Data ingestion utilities."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# football-data.org WC 2026 matches endpoint (shared by fixtures + results fetch)
WC_MATCHES_URL = "https://api.football-data.org/v4/competitions/WC/matches"
WC_SEASON = "2026"

# Column order for the flattened results table written to
# data/processed/wc2026_results.csv (one row per WC 2026 match).
RESULTS_COLUMNS = [
    "fixture_id",
    "status",
    "home_score",
    "away_score",
    "half_home",
    "half_away",
    "winner",
    "minute",
    "last_updated",
    "fetched_at",
]


def _resolve_fd_api_key(api_key: str | None = None) -> str:
    """Return a usable football-data.org API key.

    Resolution order: the explicit ``api_key`` argument, then ``FD_API_KEY``
    from the environment / project ``.env``. Raises EnvironmentError when no
    real key is found (the placeholder ``your_key_here`` is treated as unset).
    """
    if api_key and api_key != "your_key_here":
        return api_key

    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("FD_API_KEY")
    if not key or key == "your_key_here":
        raise EnvironmentError(
            "FD_API_KEY is not set. Add your football-data.org API key to the .env "
            "file (or pass api_key=...)."
        )
    return key


def _fetch_wc_matches_json(api_key: str) -> dict:
    """GET the full WC 2026 matches payload from football-data.org.

    Raises:
        RuntimeError: If the API returns a non-2xx status code.
    """
    headers = {"X-Auth-Token": api_key}
    params = {"season": WC_SEASON}
    response = requests.get(WC_MATCHES_URL, headers=headers, params=params, timeout=30)

    if not response.ok:
        raise RuntimeError(
            f"API request failed: HTTP {response.status_code}\n{response.text[:500]}"
        )
    return response.json()


def fetch_wc2026_fixtures() -> None:
    """Fetch all FIFA World Cup 2026 fixtures from football-data.org and save raw JSON.

    Reads FD_API_KEY from the .env file at project root. Saves the full API
    response to data/raw/wc2026_fixtures.json.

    Raises:
        EnvironmentError: If FD_API_KEY is missing or still set to the placeholder.
        RuntimeError: If the API returns a non-2xx status code.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("FD_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise EnvironmentError(
            "FD_API_KEY is not set. Add your football-data.org API key to the .env file."
        )

    url = "https://api.football-data.org/v4/competitions/WC/matches"
    params = {"season": "2026"}
    headers = {"X-Auth-Token": api_key}

    print(f"Fetching WC 2026 fixtures from {url} ...")
    response = requests.get(url, headers=headers, params=params, timeout=30)

    if not response.ok:
        raise RuntimeError(
            f"API request failed: HTTP {response.status_code}\n{response.text[:500]}"
        )

    data = response.json()
    out_path = PROJECT_ROOT / "data" / "raw" / "wc2026_fixtures.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    count = data.get("resultSet", {}).get("count", "?")
    print(f"Saved {count} fixtures to {out_path}")


def fetch_wc2026_results(api_key: str | None = None, write_csv: bool = True) -> pd.DataFrame:
    """Fetch current WC 2026 match statuses and scores from football-data.org.

    Pulls the live match feed and flattens each match into a results row keyed
    by ``fixture_id`` (the same id used in data/raw/wc2026_fixtures_flat.csv).
    Captures live (in-play) and full-time scores; columns are listed in
    ``RESULTS_COLUMNS``. Matches that have not started have null scores.

    Args:
        api_key: Optional explicit football-data.org key. When omitted, falls
            back to ``FD_API_KEY`` from the environment / ``.env``.
        write_csv: When True, also writes data/processed/wc2026_results.csv.

    Returns:
        A DataFrame with one row per match and columns ``RESULTS_COLUMNS``.

    Raises:
        EnvironmentError: If no API key can be resolved.
        RuntimeError: If the API returns a non-2xx status code.
    """
    key = _resolve_fd_api_key(api_key)
    data = _fetch_wc_matches_json(key)

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for match in data.get("matches", []):
        score = match.get("score") or {}
        full_time = score.get("fullTime") or {}
        half_time = score.get("halfTime") or {}
        rows.append({
            "fixture_id": match["id"],
            "status": match.get("status"),
            "home_score": full_time.get("home"),
            "away_score": full_time.get("away"),
            "half_home": half_time.get("home"),
            "half_away": half_time.get("away"),
            "winner": score.get("winner"),
            "minute": match.get("minute"),
            "last_updated": match.get("lastUpdated"),
            "fetched_at": fetched_at,
        })

    df = pd.DataFrame(rows, columns=RESULTS_COLUMNS)

    if write_csv:
        out_path = PROJECT_ROOT / "data" / "processed" / "wc2026_results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} match results to {out_path}")

    return df


def fetch_openfootball_data(skip_existing: bool = False) -> dict:
    """Download historical World Cup match text files from the openfootball GitHub repo.

    Files are saved under data/raw/openfootball/<wc_key>/.

    Args:
        skip_existing: If True, skip files that already exist and are non-empty.

    Returns:
        A dict mapping each wc_key to the list of local file Paths that were
        written (or already existed when skip_existing=True).
    """
    TOURNAMENTS = {
        "wc1998": ("1998--france",              ["cup.txt", "cup_finals.txt"]),
        "wc2002": ("2002--south-korea-n-japan", ["cup.txt", "cup_finals.txt"]),
        "wc2006": ("2006--germany",             ["cup.txt", "cup_finals.txt"]),
        "wc2010": ("2010--south-africa",        ["cup.txt", "cup_finals.txt"]),
        "wc2014": ("2014--brazil",              ["cup.txt", "cup_finals.txt"]),
        "wc2018": ("2018--russia",              ["cup.txt", "cup_finals.txt"]),
        "wc2022": ("2022--qatar",               ["cup.txt", "cup_finals.txt"]),
    }
    BASE_URL = "https://raw.githubusercontent.com/openfootball/worldcup/master"

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    result: dict[str, list[Path]] = {}

    for wc_key, (upstream_dir, filenames) in TOURNAMENTS.items():
        out_dir = _PROJECT_ROOT / "data" / "raw" / "openfootball" / wc_key
        out_dir.mkdir(parents=True, exist_ok=True)
        result[wc_key] = []

        for filename in filenames:
            out_path = out_dir / filename

            if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                print(f"[skip] {wc_key}/{filename} already exists")
                result[wc_key].append(out_path)
                continue

            url = f"{BASE_URL}/{upstream_dir}/{filename}"
            print(f"[fetch] {url}")

            try:
                response = requests.get(url, timeout=30)
            except requests.RequestException as exc:
                print(f"[warn]  Could not fetch {url}: {exc}")
                continue

            if response.status_code != 200:
                print(f"[warn]  HTTP {response.status_code} for {url} — skipping")
                continue

            out_path.write_text(response.text, encoding="utf-8")
            print(f"[done]  Saved {out_path}")
            result[wc_key].append(out_path)

            time.sleep(0.5)

    return result


def flatten_fixtures() -> None:
    """Normalise wc2026_fixtures.json into a flat 7-column CSV.

    Reads data/raw/wc2026_fixtures.json from the repo root, extracts each
    match into a row with columns fixture_id, match_date, kickoff_utc,
    home_team, away_team, stage, and status, and writes the result to
    data/raw/wc2026_fixtures_flat.csv.
    Rows where home_team or away_team is falsy are skipped.

    Raises:
        FileNotFoundError: If wc2026_fixtures.json does not exist.
    """
    src_path = PROJECT_ROOT / "data" / "raw" / "wc2026_fixtures.json"
    out_path = PROJECT_ROOT / "data" / "raw" / "wc2026_fixtures_flat.csv"

    with src_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for match in raw.get("matches", []):
        home_team = (match.get("homeTeam") or {}).get("name") or "TBD"
        away_team = (match.get("awayTeam") or {}).get("name") or "TBD"
        rows.append({
            "fixture_id": match["id"],
            "match_date": match["utcDate"][:10],
            "kickoff_utc": match["utcDate"],
            "home_team": home_team,
            "away_team": away_team,
            "stage": match.get("stage"),
            "status": match.get("status"),
        })

    df = pd.DataFrame(rows, columns=["fixture_id", "match_date", "kickoff_utc", "home_team", "away_team", "stage", "status"])
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} fixtures to {out_path}")
