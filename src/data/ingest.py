"""Data ingestion utilities."""

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
