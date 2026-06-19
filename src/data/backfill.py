"""Backfill finished WC 2026 scores from the live feed into the raw history.

The live results feed (``data/processed/wc2026_results.csv``, keyed by
``fixture_id``) records actual scores during the tournament, but on its own those
scores never reach the model: :func:`src.data.preprocess.export_features` warms
up form / head-to-head features from ``data/raw/results.csv``, where the WC 2026
fixtures sit as *score-less* rows that the pipeline treats as unplayed fixtures
(see ``preprocess.py`` ``fixture_mask = home_score.isna()``).

:func:`backfill_raw_results` copies the finished scores onto those rows so that a
subsequent feature-engineering run reflects completed matches. ``results.csv`` has
no ``fixture_id`` column, so matches are joined on the (canonical) team-name pair,
resolved through the project's existing alias map
(:func:`src.data.odds._load_team_name_map`) — both the feed name (``"Czechia"``)
and the history name (``"Czech Republic"``) normalize to the same canonical name.

The function is idempotent: re-running rewrites the same scores. Finished matches
that have no row yet in ``results.csv`` (e.g. knockout fixtures whose teams were
``TBD`` when the schedule was first imported) are appended as new rows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.odds import _load_team_name_map, normalize_team_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW = PROJECT_ROOT / "data" / "raw"
_PROCESSED = PROJECT_ROOT / "data" / "processed"

_RESULTS_CSV = _RAW / "results.csv"
_FIXTURES_CSV = _RAW / "wc2026_fixtures_flat.csv"
_LIVE_RESULTS_CSV = _PROCESSED / "wc2026_results.csv"
_TEAM_NAME_MAP_CSV = _RAW / "team_name_map.csv"

# football-data.org status groupings that count as a final, scored result.
_FINAL_STATUSES = {"FINISHED", "AWARDED"}

# Column order written back to data/raw/results.csv.
_RESULTS_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]


def _reverse_name_map(path: Path = _TEAM_NAME_MAP_CSV) -> dict:
    """Map canonical fixture names -> ``results.csv`` names (for appended rows)."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict = {}
    for _, row in df.iterrows():
        fixture = str(row.get("fixture_name", "")).strip()
        results = str(row.get("results_name", "")).strip()
        if fixture and results:
            out[fixture] = results
    return out


def _canon(name, name_map: dict) -> str:
    """Canonicalize a team name, tolerating NaN/None."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    return normalize_team_name(str(name), name_map)


def backfill_raw_results(
    live_results: pd.DataFrame | None = None,
    *,
    write: bool = True,
    results_path: Path = _RESULTS_CSV,
    fixtures_path: Path = _FIXTURES_CSV,
    live_results_path: Path = _LIVE_RESULTS_CSV,
    name_map: dict | None = None,
    reverse_name_map: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Write finished WC 2026 scores from the live feed into ``data/raw/results.csv``.

    Args:
        live_results: Live results frame (output shape of
            :func:`src.data.ingest.fetch_wc2026_results`, keyed by ``fixture_id``).
            When ``None``, read from ``live_results_path``.
        write: When True, write the updated frame back to ``results_path``.
        results_path / fixtures_path / live_results_path: Path overrides (tests).
        name_map: Alias map (history/feed name -> canonical). Defaults to
            :func:`src.data.odds._load_team_name_map`.
        reverse_name_map: Canonical -> ``results.csv`` name, used when appending
            new (knockout) rows. Defaults to the project ``team_name_map.csv``.
        verbose: Print a one-line summary plus any appended/skipped matches.

    Returns:
        Dict with keys ``finished``, ``updated``, ``appended``,
        ``appended_matches`` (list of "Home vs Away" strings), ``skipped``
        (finished matches whose teams couldn't be resolved from fixtures), and
        ``skipped_fixture_ids``.
    """
    if live_results is None:
        if not Path(live_results_path).exists():
            raise FileNotFoundError(
                f"Live results not found at {live_results_path}. "
                "Run scripts/fetch_results.py (or update_after_round.py without --no-fetch) first."
            )
        live_results = pd.read_csv(live_results_path)

    if name_map is None:
        name_map = _load_team_name_map()
    if reverse_name_map is None:
        reverse_name_map = _reverse_name_map()

    fixtures = pd.read_csv(fixtures_path)
    results = pd.read_csv(results_path)

    # --- finished matches with a real score ---
    finished = live_results[live_results["status"].isin(_FINAL_STATUSES)].copy()
    finished = finished[finished["home_score"].notna() & finished["away_score"].notna()]

    # --- attach team names + date via fixture_id ---
    fx_cols = ["fixture_id", "match_date", "home_team", "away_team"]
    finished = finished.merge(fixtures[fx_cols], on="fixture_id", how="left")

    skipped_ids = finished[finished["home_team"].isna() | finished["away_team"].isna()]
    skipped_fixture_ids = [int(x) for x in skipped_ids["fixture_id"].tolist()]
    finished = finished.drop(skipped_ids.index)

    # --- index existing WC 2026 rows by canonical team pair ---
    wc_mask = (
        (results["tournament"] == "FIFA World Cup")
        & results["date"].astype(str).str.startswith("2026")
    )
    pair_to_index: dict[tuple[str, str], int] = {}
    for idx in results.index[wc_mask]:
        key = (
            _canon(results.at[idx, "home_team"], name_map),
            _canon(results.at[idx, "away_team"], name_map),
        )
        pair_to_index[key] = idx

    updated = 0
    appended_rows: list[dict] = []
    appended_matches: list[str] = []

    for _, m in finished.iterrows():
        home_canon = _canon(m["home_team"], name_map)
        away_canon = _canon(m["away_team"], name_map)
        home_score = int(round(float(m["home_score"])))
        away_score = int(round(float(m["away_score"])))
        key = (home_canon, away_canon)

        if key in pair_to_index:
            idx = pair_to_index[key]
            results.at[idx, "home_score"] = home_score
            results.at[idx, "away_score"] = away_score
            updated += 1
        else:
            home_name = reverse_name_map.get(home_canon, str(m["home_team"]))
            away_name = reverse_name_map.get(away_canon, str(m["away_team"]))
            appended_rows.append({
                "date": str(m["match_date"]),
                "home_team": home_name,
                "away_team": away_name,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": "FIFA World Cup",
                "city": pd.NA,
                "country": pd.NA,
                "neutral": True,
            })
            appended_matches.append(f"{home_name} {home_score}-{away_score} {away_name}")

    if appended_rows:
        results = pd.concat(
            [results, pd.DataFrame(appended_rows, columns=_RESULTS_COLUMNS)],
            ignore_index=True,
        )

    if write:
        results.to_csv(results_path, index=False)

    summary = {
        "finished": int(len(finished)),
        "updated": updated,
        "appended": len(appended_rows),
        "appended_matches": appended_matches,
        "skipped": len(skipped_fixture_ids),
        "skipped_fixture_ids": skipped_fixture_ids,
    }

    if verbose:
        print(
            f"  Backfill: {summary['finished']} finished -> "
            f"{updated} updated, {summary['appended']} appended, "
            f"{summary['skipped']} skipped."
        )
        for line in appended_matches:
            print(f"    + appended (no existing row): {line}")
        if skipped_fixture_ids:
            print(
                "    ! skipped finished matches with no fixture metadata: "
                f"{skipped_fixture_ids}"
            )

    return summary
