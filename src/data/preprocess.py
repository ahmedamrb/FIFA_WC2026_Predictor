"""Data preprocessing utilities for the FIFA WC 2026 Predictor.

Loads and normalises raw CSVs (results, rankings, fixtures, name map),
aligns team names to a single canonical form (fixture_name), and parses
date columns to datetime.  All downstream feature-engineering steps read
from the DataFrames produced here.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_START_YEAR: int = 1990

FEATURE_COLUMNS: list[str] = [
    # --- Rankings (Subphase 3.3) ---
    "home_rank",
    "away_rank",
    "home_rank_points",
    "away_rank_points",
    "rank_diff",
    "rank_points_ratio",
    "rank_points_diff",
    # --- Form 5-match window (Subphase 3.4) ---
    "home_form_wins_5",
    "home_form_goals_scored_5",
    "home_form_goals_conceded_5",
    "home_form_wdl_points_5",
    "away_form_wins_5",
    "away_form_goals_scored_5",
    "away_form_goals_conceded_5",
    "away_form_wdl_points_5",
    # --- Form 10-match window (Subphase 3.4) ---
    "home_form_wins_10",
    "home_form_goals_scored_10",
    "home_form_goals_conceded_10",
    "home_form_wdl_points_10",
    "away_form_wins_10",
    "away_form_goals_scored_10",
    "away_form_goals_conceded_10",
    "away_form_wdl_points_10",
    "form_diff_wdl_5",
    "home_goal_efficiency_5",
    "away_goal_efficiency_5",
    # --- Head-to-Head (Subphase 3.5) ---
    "h2h_home_win_rate",
    "h2h_matches_count",
    "h2h_avg_goals_home",
    "h2h_avg_goals_away",
    # --- Context & Venue (Subphase 3.6) ---
    "tournament_stage",
    "is_wc_match",
    "is_neutral_venue",
    "host_nation_advantage",
    "home_days_rest",
    "away_days_rest",
    "rest_diff",
]

_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
_REFERENCE_DATE: pd.Timestamp = pd.Timestamp("2026-06-01")

_OPENFOOTBALL_DIR = _RAW_DIR / "openfootball"

# Stage header keywords found across all WC openfootball txt files (1998–2022).
# Checked via str.startswith so longer keys take priority over "final".
_WC_KNOCKOUT_STAGE_MAP: dict[str, int] = {
    "round of 16": 3,
    "quarter-finals": 4,
    "quarterfinals": 4,
    "semi-finals": 5,
    "semifinals": 5,
    "match for third place": 5,
    "third-place play-off": 5,
    "third place play-off": 5,
    "third place match": 5,
    "final": 6,
}

_WC2026_FIXTURE_STAGE_MAP: dict[str, int] = {
    "GROUP_STAGE": 1,
    "LAST_32": 2,
    "LAST_16": 3,
    "QUARTER_FINALS": 4,
    "SEMI_FINALS": 5,
    "THIRD_PLACE": 5,
    "FINAL": 6,
}

_WC2026_HOST_TEAMS: frozenset[str] = frozenset({"United States", "Canada", "Mexico"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and normalise the four core raw data files.

    Reads ``results.csv``, ``rankings.csv``, ``wc2026_fixtures_flat.csv``, and
    ``team_name_map.csv`` from ``data/raw/``.  Team names in ``results`` and
    ``rankings`` are mapped to the canonical ``fixture_name`` values defined in
    the name-map so that all three datasets share a consistent namespace.

    Returns:
        A 4-tuple ``(results_df, rankings_df, fixtures_df, name_map_df)`` where:

        * ``results_df``: International match results with normalised team names
          and a parsed ``date`` column.
        * ``rankings_df``: FIFA world rankings with a normalised ``country_full``
          column and a parsed ``rank_date`` column.
        * ``fixtures_df``: WC 2026 fixture schedule with a parsed ``match_date``
          column.
        * ``name_map_df``: The raw name-map DataFrame (unchanged).
    """
    # ------------------------------------------------------------------
    # Load raw CSVs
    # ------------------------------------------------------------------
    results_df = pd.read_csv(_RAW_DIR / "results.csv")
    rankings_df = pd.read_csv(_RAW_DIR / "rankings.csv", index_col=0)
    fixtures_df = pd.read_csv(_RAW_DIR / "wc2026_fixtures_flat.csv")
    name_map_df = pd.read_csv(_RAW_DIR / "team_name_map.csv")

    # ------------------------------------------------------------------
    # Build normalisation maps: source name → canonical fixture_name
    # ------------------------------------------------------------------
    results_to_fixture: dict[str, str] = dict(
        zip(name_map_df["results_name"], name_map_df["fixture_name"])
    )
    rankings_to_fixture: dict[str, str] = dict(
        zip(name_map_df["rankings_name"], name_map_df["fixture_name"])
    )

    # ------------------------------------------------------------------
    # Apply normalisation
    # ------------------------------------------------------------------
    results_df["home_team"] = results_df["home_team"].replace(results_to_fixture)
    results_df["away_team"] = results_df["away_team"].replace(results_to_fixture)
    rankings_df["country_full"] = rankings_df["country_full"].replace(rankings_to_fixture)

    # ------------------------------------------------------------------
    # Parse date columns
    # ------------------------------------------------------------------
    results_df["date"] = pd.to_datetime(results_df["date"])
    rankings_df["rank_date"] = pd.to_datetime(rankings_df["rank_date"])
    fixtures_df["match_date"] = pd.to_datetime(fixtures_df["match_date"])

    return results_df, rankings_df, fixtures_df, name_map_df


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Filter, label, and weight the results DataFrame for model training.

    Performs three transformations on a copy of ``df``:

    1. **Date filter** — keeps only matches from ``TRAIN_START_YEAR`` (1990) onward.
    2. **Score filter** — drops rows where ``home_score`` or ``away_score`` is null
       (these are WC 2026 fixtures scheduled but not yet played).
    3. **Derived columns** — adds ``match_importance``, ``outcome``, and
       ``recency_weight``.

    Args:
        df: Results DataFrame as returned by ``load_raw_data()``, with a
            datetime ``date`` column and numeric ``home_score``/``away_score``.

    Returns:
        A cleaned copy of ``df`` with a reset integer index and three new columns.
    """
    df = df.copy()

    # 1. Filter: keep only matches on or after TRAIN_START_YEAR
    df = df[df["date"].dt.year >= TRAIN_START_YEAR]

    # 2. Drop rows without completed scores (unplayed/future fixtures)
    df = df.dropna(subset=["home_score", "away_score"])

    # 3. match_importance — checked in priority order by np.select
    #    Uses "qualif" substring to capture both "qualifier" and "qualification"
    conditions_importance = [
        df["tournament"] == "FIFA World Cup",
        df["tournament"].str.contains("qualif", case=False, na=False),
        df["tournament"] == "Friendly",
    ]
    df["match_importance"] = np.select(
        conditions_importance, [3.0, 1.5, 0.5], default=1.0
    )

    # 4. outcome: 2 = home win, 1 = draw, 0 = away win
    df["outcome"] = np.select(
        [
            df["home_score"] > df["away_score"],
            df["home_score"] == df["away_score"],
        ],
        [2, 1],
        default=0,
    ).astype(int)

    # 5. recency_weight: 0.85 ^ years_since_match (relative to _REFERENCE_DATE)
    years_since = (_REFERENCE_DATE - df["date"]).dt.days / 365.25
    df["recency_weight"] = 0.85 ** years_since

    return df.reset_index(drop=True)


def merge_rankings(
    matches_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Attach FIFA rankings to each match row using the most recent available data.

    For every row in ``matches_df`` the function performs an as-of lookup
    against ``rankings_df`` to find the latest published ranking for both the
    home team and the away team that falls on or before the match date.  Rows
    with no prior ranking record (e.g. very early matches) are filled with the
    global median rank and median points.

    Args:
        matches_df: Match DataFrame produced by ``clean_results()`` or a
            fixtures DataFrame.  Must contain ``date_col``, ``home_team``, and
            ``away_team`` columns.
        rankings_df: Rankings DataFrame as returned by ``load_raw_data()``.
            Must contain ``country_full``, ``rank_date``, ``rank``, and
            ``total_points`` columns with a parsed datetime ``rank_date``.
        date_col: Name of the date column in ``matches_df``.  Defaults to
            ``"date"`` (clean results); pass ``"match_date"`` for fixtures.

    Returns:
        A copy of ``matches_df`` with six additional columns:
        ``home_rank``, ``away_rank``, ``home_rank_points``,
        ``away_rank_points``, ``rank_diff``, and ``rank_points_ratio``.
    """
    df = matches_df.copy()

    # Ensure rank_date is datetime (defensive in case caller skips load_raw_data)
    rank_df = rankings_df.copy()
    rank_df["rank_date"] = pd.to_datetime(rank_df["rank_date"])

    # Compute global medians used for null filling
    median_rank: float = float(rank_df["rank"].median())
    median_points: float = float(rank_df["total_points"].median())

    # Prepare a clean lookup table sorted by rank_date
    rank_lookup = (
        rank_df[["country_full", "rank_date", "rank", "total_points"]]
        .sort_values("rank_date")
        .reset_index(drop=True)
    )

    # Sort matches by date — required by merge_asof
    df = df.sort_values(date_col).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Home-team ranking lookup
    # ------------------------------------------------------------------
    home_lookup = rank_lookup.rename(
        columns={
            "country_full": "home_team",
            "rank_date": date_col,
            "rank": "home_rank",
            "total_points": "home_rank_points",
        }
    )
    df = pd.merge_asof(
        df,
        home_lookup[[date_col, "home_team", "home_rank", "home_rank_points"]],
        on=date_col,
        by="home_team",
        direction="backward",
    )

    # ------------------------------------------------------------------
    # Away-team ranking lookup
    # ------------------------------------------------------------------
    away_lookup = rank_lookup.rename(
        columns={
            "country_full": "away_team",
            "rank_date": date_col,
            "rank": "away_rank",
            "total_points": "away_rank_points",
        }
    )
    df = pd.merge_asof(
        df,
        away_lookup[[date_col, "away_team", "away_rank", "away_rank_points"]],
        on=date_col,
        by="away_team",
        direction="backward",
    )

    # ------------------------------------------------------------------
    # Fill nulls with global medians
    # ------------------------------------------------------------------
    df["home_rank"] = df["home_rank"].fillna(median_rank)
    df["away_rank"] = df["away_rank"].fillna(median_rank)
    df["home_rank_points"] = df["home_rank_points"].fillna(median_points)
    df["away_rank_points"] = df["away_rank_points"].fillna(median_points)

    # ------------------------------------------------------------------
    # Derived features
    # ------------------------------------------------------------------
    df["rank_diff"] = df["home_rank"] - df["away_rank"]
    df["rank_points_diff"] = df["home_rank_points"] - df["away_rank_points"]
    away_pts_safe = df["away_rank_points"].clip(lower=1e-6)
    df["rank_points_ratio"] = (df["home_rank_points"] / away_pts_safe).clip(0.1, 10.0)

    return df


def compute_form_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling form features (5- and 10-match windows) for both teams.

    Iterates through matches in chronological order.  For each match the
    form statistics are computed from the team's accumulated history
    **before** that match, so there is no data leakage.

    Features computed for each *side* (``home`` / ``away``) and each
    *window* (``5`` / ``10``):

    * ``{side}_form_wins_{w}`` — number of wins in the last *w* matches.
    * ``{side}_form_goals_scored_{w}`` — average goals scored per match.
    * ``{side}_form_goals_conceded_{w}`` — average goals conceded per match.
    * ``{side}_form_wdl_points_{w}`` — total WDL points (W=3, D=1, L=0).

    Also adds ``form_diff_wdl_5 = home_form_wdl_points_5 - away_form_wdl_points_5``.

    Rows with zero prior history for a team receive ``NaN`` for that team's
    form columns; these are filled with the global column median after
    iterating all rows.

    Args:
        matches_df: Match DataFrame with ``date``, ``home_team``,
            ``away_team``, ``home_score``, and ``away_score`` columns.

    Returns:
        A copy of ``matches_df`` sorted by date with 17 new form columns
        appended.  All columns are null-free after median fill.
    """
    from collections import defaultdict

    df = matches_df.copy().sort_values("date").reset_index(drop=True)

    WINDOWS = (5, 10)

    # team -> list of (goals_scored, goals_conceded, wdl_points) in match order
    team_history: dict[str, list[tuple[int, int, int]]] = defaultdict(list)

    # Pre-allocate output lists for efficiency
    feature_data: dict[str, list] = {}
    for side in ("home", "away"):
        for w in WINDOWS:
            for stat in ("wins", "goals_scored", "goals_conceded", "wdl_points"):
                feature_data[f"{side}_form_{stat}_{w}"] = []

    def _stats_from_history(
        history: list[tuple[int, int, int]], window: int
    ) -> tuple:
        """Return (wins, avg_scored, avg_conceded, wdl_points) or all-None."""
        last_n = history[-window:]
        if not last_n:
            return None, None, None, None
        wins = sum(1 for entry in last_n if entry[2] == 3)
        avg_scored = sum(entry[0] for entry in last_n) / len(last_n)
        avg_conceded = sum(entry[1] for entry in last_n) / len(last_n)
        wdl_pts = sum(entry[2] for entry in last_n)
        return wins, avg_scored, avg_conceded, wdl_pts

    for row in df.itertuples(index=False):
        home_team: str = row.home_team
        away_team: str = row.away_team

        for side, team in (("home", home_team), ("away", away_team)):
            hist = team_history[team]
            for w in WINDOWS:
                wins, avg_scored, avg_conceded, wdl_pts = _stats_from_history(hist, w)
                feature_data[f"{side}_form_wins_{w}"].append(wins)
                feature_data[f"{side}_form_goals_scored_{w}"].append(avg_scored)
                feature_data[f"{side}_form_goals_conceded_{w}"].append(avg_conceded)
                feature_data[f"{side}_form_wdl_points_{w}"].append(wdl_pts)

        # Update team histories AFTER recording features (no leakage)
        h_score = row.home_score
        a_score = row.away_score
        if pd.notna(h_score) and pd.notna(a_score):
            h, a = int(h_score), int(a_score)
            if h > a:
                hw, aw = 3, 0
            elif h == a:
                hw, aw = 1, 1
            else:
                hw, aw = 0, 3
            team_history[home_team].append((h, a, hw))
            team_history[away_team].append((a, h, aw))

    # Assign computed columns
    for col, values in feature_data.items():
        df[col] = values

    # Derived diff (computed before fill so NaN propagates correctly,
    # then filled below together with the component columns)
    df["form_diff_wdl_5"] = (
        df["home_form_wdl_points_5"] - df["away_form_wdl_points_5"]
    )
    df["home_goal_efficiency_5"] = (
        df["home_form_goals_scored_5"] / (df["home_form_goals_conceded_5"] + 0.1)
    )
    df["away_goal_efficiency_5"] = (
        df["away_form_goals_scored_5"] / (df["away_form_goals_conceded_5"] + 0.1)
    )

    all_form_cols = list(feature_data.keys()) + [
        "form_diff_wdl_5",
        "home_goal_efficiency_5",
        "away_goal_efficiency_5",
    ]

    # Report null statistics before fill
    null_rows = df[all_form_cols].isna().any(axis=1).sum()
    print(f"\nForm features: {null_rows} rows had >=1 null before median fill")
    null_by_col = {col: int(df[col].isna().sum()) for col in all_form_cols if df[col].isna().any()}
    if null_by_col:
        print("  Columns with nulls before fill:")
        for col, cnt in null_by_col.items():
            print(f"    {col}: {cnt}")

    # Fill nulls with global column median
    for col in all_form_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    print("Null counts after fill (all should be 0):")
    for col in all_form_cols:
        print(f"  {col}: {int(df[col].isna().sum())}")

    return df


def compute_h2h_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head features for each match using prior meetings only.

    Iterates through matches in chronological order.  For each match the
    function looks back at the last 5 **prior meetings** between the two
    teams (symmetric — both (A vs B) and (B vs A) count) and computes
    statistics from the perspective of the *current* home team.

    Features computed:

    * ``h2h_home_win_rate`` — fraction of prior meetings won by the current
      home team (regardless of which side they were on). Default fill: 0.33
    * ``h2h_matches_count`` — total prior meetings found (0 if none).
    * ``h2h_avg_goals_home`` — average goals scored by the current home team
      per prior meeting. Default fill: 1.3
    * ``h2h_avg_goals_away`` — average goals scored by the current away team
      per prior meeting. Default fill: 1.1

    Rows with no prior history (``h2h_matches_count == 0``) are filled with
    the neutral defaults listed above.  The h2h history is updated **after**
    each row is processed to prevent data leakage.

    Args:
        matches_df: Match DataFrame with ``date``, ``home_team``,
            ``away_team``, ``home_score``, and ``away_score`` columns.
            Must be sortable by ``date``.

    Returns:
        A copy of ``matches_df`` sorted by date with 4 new h2h columns
        appended.  All columns are null-free after neutral-default fill.
    """
    from collections import defaultdict

    df = matches_df.copy().sort_values("date").reset_index(drop=True)

    # h2h_history: frozenset({team_a, team_b}) -> list of
    # (home_team_name, home_goals, away_goals) for each recorded meeting
    h2h_history: dict = defaultdict(list)

    h2h_home_win_rate: list = []
    h2h_matches_count: list = []
    h2h_avg_goals_home: list = []
    h2h_avg_goals_away: list = []

    for row in df.itertuples(index=False):
        home_team: str = row.home_team
        away_team: str = row.away_team
        key = frozenset({home_team, away_team})

        prior = h2h_history[key][-5:]  # last 5 prior meetings
        n = len(prior)

        if n == 0:
            h2h_home_win_rate.append(float("nan"))
            h2h_matches_count.append(0)
            h2h_avg_goals_home.append(float("nan"))
            h2h_avg_goals_away.append(float("nan"))
        else:
            wins = 0
            goals_home_total = 0.0
            goals_away_total = 0.0
            for rec_home, rec_home_goals, rec_away_goals in prior:
                if rec_home == home_team:
                    g_home = rec_home_goals
                    g_away = rec_away_goals
                else:
                    # home_team was the away side in this prior meeting
                    g_home = rec_away_goals
                    g_away = rec_home_goals
                goals_home_total += g_home
                goals_away_total += g_away
                if g_home > g_away:
                    wins += 1
            h2h_home_win_rate.append(wins / n)
            h2h_matches_count.append(n)
            h2h_avg_goals_home.append(goals_home_total / n)
            h2h_avg_goals_away.append(goals_away_total / n)

        # Update h2h history AFTER recording features (no leakage)
        h_score = row.home_score
        a_score = row.away_score
        if pd.notna(h_score) and pd.notna(a_score):
            h2h_history[key].append((home_team, int(h_score), int(a_score)))

    df["h2h_home_win_rate"] = h2h_home_win_rate
    df["h2h_matches_count"] = h2h_matches_count
    df["h2h_avg_goals_home"] = h2h_avg_goals_home
    df["h2h_avg_goals_away"] = h2h_avg_goals_away

    # Fill rows with no prior meetings with neutral defaults
    no_history_mask = df["h2h_matches_count"] == 0
    neutral_fill_count = int(no_history_mask.sum())
    df.loc[no_history_mask, "h2h_home_win_rate"] = 0.33
    df.loc[no_history_mask, "h2h_avg_goals_home"] = 1.3
    df.loc[no_history_mask, "h2h_avg_goals_away"] = 1.1

    print(f"\nH2H features: {neutral_fill_count} rows filled with neutral defaults")

    H2H_COLS = ["h2h_home_win_rate", "h2h_matches_count", "h2h_avg_goals_home", "h2h_avg_goals_away"]
    print("Null counts after fill (all should be 0):")
    for col in H2H_COLS:
        print(f"  {col}: {int(df[col].isna().sum())}")

    return df


def _build_wc_stage_lookup() -> dict[pd.Timestamp, int]:
    """Parse openfootball WC txt files to map each knockout-round match date to its stage ordinal.

    Returns:
        A dict mapping normalised match dates (time set to midnight) to their
        tournament_stage ordinal (3=R16, 4=QF, 5=SF/3rd, 6=Final).
        Group-stage matches are intentionally omitted (they default to 1 elsewhere).
    """
    import re

    # WC 1998/2002: "27 June" or "15 June" (day month_name, no day-of-week)
    _DATE_LONGMONTH = re.compile(
        r"^\s*(\d{1,2})\s+"
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\b",
        re.IGNORECASE,
    )
    # WC 2006–2022: "Sat Jun 24", "Mon Jun 28 16:00" (day-of-week month_abbr day)
    _DATE_DAYOFWEEK = re.compile(
        r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\w{3,4})\s+(\d{1,2})\b",
        re.IGNORECASE,
    )

    lookup: dict[pd.Timestamp, int] = {}

    for year in [1998, 2002, 2006, 2010, 2014, 2018, 2022]:
        txt_path = _OPENFOOTBALL_DIR / f"wc{year}" / "cup_finals.txt"
        if not txt_path.exists():
            continue

        current_stage: int | None = None

        with open(txt_path, encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.rstrip()
                stripped = line.strip()

                # --- Stage header detection (bullet U+25AA) ---
                if stripped.startswith("\u25aa "):
                    # Strip bullet, drop date-range suffix after "|"
                    header_text = stripped[2:].split("|")[0].strip().lower()
                    matched_stage: int | None = None
                    for key, val in _WC_KNOCKOUT_STAGE_MAP.items():
                        if header_text.startswith(key):
                            matched_stage = val
                            break
                    current_stage = matched_stage
                    continue

                # Only extract dates when inside a known knockout section
                if current_stage is None:
                    continue

                # Skip TOC lines that span a date range ("Sat Jun 30 - Tue Jul 3")
                if " - " in stripped:
                    continue

                # --- Date extraction ---
                ts: pd.Timestamp | None = None

                # Try day-of-week format first: "Sat Jun 24", "Mon Jun 28 16:00"
                m_dow = _DATE_DAYOFWEEK.search(line)
                if m_dow:
                    month_abbr, day_str = m_dow.group(1), m_dow.group(2)
                    try:
                        ts = pd.to_datetime(
                            f"{day_str} {month_abbr} {year}",
                            format="%d %b %Y",
                            errors="coerce",
                        )
                    except Exception:
                        ts = pd.NaT

                # Fallback: try long-month format: "27 June", "15 June"
                if ts is None or pd.isna(ts):
                    m_long = _DATE_LONGMONTH.search(line)
                    if m_long:
                        day_str2, month_name = m_long.group(1), m_long.group(2)
                        try:
                            ts = pd.to_datetime(
                                f"{day_str2} {month_name} {year}",
                                format="%d %B %Y",
                                errors="coerce",
                            )
                        except Exception:
                            ts = pd.NaT

                if ts is not None and not pd.isna(ts):
                    normalised = ts.normalize()
                    if normalised not in lookup:
                        lookup[normalised] = current_stage

    return lookup


def compute_context_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Add context and venue features to a match DataFrame.

    Works on both historical results (``date`` column) and WC 2026 fixture
    data (``match_date`` column).  When a ``stage`` column is present the
    WC 2026 fixture-stage mapping is used for ``tournament_stage``; otherwise
    the openfootball stage lookup is used for historical WC matches.

    Features added:

    * ``tournament_stage`` — ordinal: 0=non-WC, 1=Group, 2=R32, 3=R16,
      4=QF, 5=SF/3rd, 6=Final.
    * ``is_wc_match`` — 1 if FIFA World Cup match, else 0.
    * ``is_neutral_venue`` — 1 if played at a neutral venue, else 0.
    * ``host_nation_advantage`` — 1 if WC 2026 and home team is USA/Canada/Mexico.
    * ``home_days_rest`` — days since home team's previous match (30 if first).
    * ``away_days_rest`` — days since away team's previous match (30 if first).

    Args:
        matches_df: Match DataFrame.  Must contain ``date`` or ``match_date``,
            ``home_team``, and ``away_team``.  Optional columns used when
            present: ``tournament``, ``neutral``, ``stage``.

    Returns:
        A copy of ``matches_df`` sorted by match date with 6 new columns.
    """
    df = matches_df.copy()

    # ------------------------------------------------------------------
    # Detect date column
    # ------------------------------------------------------------------
    if "date" in df.columns:
        date_col = "date"
    elif "match_date" in df.columns:
        date_col = "match_date"
    else:
        raise ValueError("matches_df must contain a 'date' or 'match_date' column")

    df = df.sort_values(date_col).reset_index(drop=True)
    _date: pd.Series = df[date_col]

    # ------------------------------------------------------------------
    # is_wc_match
    # ------------------------------------------------------------------
    if "tournament" in df.columns:
        df["is_wc_match"] = (df["tournament"] == "FIFA World Cup").astype(int)
    else:
        # Fixture-only DataFrame — all rows are WC matches
        df["is_wc_match"] = 1

    # ------------------------------------------------------------------
    # tournament_stage
    # ------------------------------------------------------------------
    if "stage" in df.columns:
        # WC 2026 fixtures path — use the fixture stage column
        df["tournament_stage"] = (
            df["stage"].map(_WC2026_FIXTURE_STAGE_MAP).fillna(0).astype(int)
        )
    else:
        # Historical data path — parse openfootball files
        stage_lookup = _build_wc_stage_lookup()
        date_stage = _date.dt.normalize().map(stage_lookup)
        df["tournament_stage"] = np.where(
            df["is_wc_match"] == 0,
            0,
            np.where(date_stage.notna(), date_stage, 1),
        ).astype(int)

    # ------------------------------------------------------------------
    # is_neutral_venue
    # ------------------------------------------------------------------
    if "neutral" in df.columns:
        df["is_neutral_venue"] = df["neutral"].fillna(0).astype(int)
    else:
        # WC 2026 fixtures — treat all as neutral
        df["is_neutral_venue"] = 1

    # ------------------------------------------------------------------
    # host_nation_advantage
    # ------------------------------------------------------------------
    df["host_nation_advantage"] = (
        (df["is_wc_match"] == 1)
        & (_date.dt.year == 2026)
        & (df["home_team"].isin(_WC2026_HOST_TEAMS))
    ).astype(int)

    # ------------------------------------------------------------------
    # home_days_rest and away_days_rest
    # ------------------------------------------------------------------
    last_match: dict[str, pd.Timestamp] = {}
    home_rest: list[int] = []
    away_rest: list[int] = []

    for row in df.itertuples(index=False):
        match_date = getattr(row, date_col)
        for team, rest_list in ((row.home_team, home_rest), (row.away_team, away_rest)):
            if team in last_match:
                delta = (match_date - last_match[team]).days
                rest_list.append(max(0, delta))
            else:
                rest_list.append(30)

        last_match[row.home_team] = match_date
        last_match[row.away_team] = match_date

    df["home_days_rest"] = home_rest
    df["away_days_rest"] = away_rest
    df["rest_diff"] = df["home_days_rest"] - df["away_days_rest"]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    print(
        f"\ntournament_stage value counts:\n"
        f"{df['tournament_stage'].value_counts().sort_index()}"
    )
    print(f"\nis_wc_match value counts:\n{df['is_wc_match'].value_counts()}")
    print(f"\nis_neutral_venue value counts:\n{df['is_neutral_venue'].value_counts()}")
    print(
        f"\nhome_days_rest  min: {df['home_days_rest'].min()}  "
        f"max: {df['home_days_rest'].max()}"
    )
    print(
        f"away_days_rest  min: {df['away_days_rest'].min()}  "
        f"max: {df['away_days_rest'].max()}"
    )

    return df


def compute_elo_ratings(
    matches_df: pd.DataFrame,
    starting_elo: float = 1500.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute Elo ratings for each team at the time of each match.

    Processes matches in chronological order.  For each match the function
    records the pre-match Elo for both teams and then updates Elo using the
    standard formula weighted by a K-factor derived from ``match_importance``.
    Rows without a completed outcome (NaN) are assigned the current Elo but
    do not trigger an update, so the function handles prediction fixtures
    naturally.

    K-factor mapping:
        - WC (importance >= 3.0): K=60
        - Qualifier (importance >= 1.5): K=50
        - Friendly (importance <= 0.5): K=20
        - Other competitive (default): K=40

    Args:
        matches_df: Match DataFrame with ``date``, ``home_team``, and
            ``away_team`` columns.  When present, ``outcome`` and
            ``match_importance`` are used to update Elo after each match.
        starting_elo: Initial Elo rating assigned to teams on first appearance.

    Returns:
        A 2-tuple ``(df_with_elo, final_elo)`` where ``df_with_elo`` is a
        copy of ``matches_df`` with ``home_elo``, ``away_elo``, and
        ``elo_diff`` columns appended, and ``final_elo`` is a dict mapping
        each team to their rating after the last completed match.
    """
    date_col = "date" if "date" in matches_df.columns else "match_date"
    df = matches_df.copy().sort_values(date_col).reset_index(drop=True)

    elo: dict[str, float] = {}
    home_elo_list: list[float] = []
    away_elo_list: list[float] = []

    has_outcome = "outcome" in df.columns
    has_importance = "match_importance" in df.columns

    for row in df.itertuples(index=False):
        home: str = row.home_team
        away: str = row.away_team

        r_home = elo.get(home, starting_elo)
        r_away = elo.get(away, starting_elo)

        home_elo_list.append(r_home)
        away_elo_list.append(r_away)

        # Only update Elo for rows with a completed outcome
        outcome_val = getattr(row, "outcome", None) if has_outcome else None
        if outcome_val is not None and not (isinstance(outcome_val, float) and np.isnan(outcome_val)):
            importance = (
                float(getattr(row, "match_importance", 1.0)) if has_importance else 1.0
            )
            if np.isnan(importance):
                importance = 1.0

            if importance >= 3.0:
                k = 60.0
            elif importance >= 1.5:
                k = 50.0
            elif importance <= 0.5:
                k = 20.0
            else:
                k = 40.0

            e_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home) / 400.0))
            e_away = 1.0 - e_home

            outcome = int(outcome_val)
            if outcome == 2:
                s_home = 1.0
            elif outcome == 1:
                s_home = 0.5
            else:
                s_home = 0.0
            s_away = 1.0 - s_home

            elo[home] = r_home + k * (s_home - e_home)
            elo[away] = r_away + k * (s_away - e_away)

    df["home_elo"] = home_elo_list
    df["away_elo"] = away_elo_list
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    if elo:
        elo_vals = list(elo.values())
        print(
            f"\nElo ratings: {len(elo)} teams, "
            f"range {min(elo_vals):.0f}\u2013{max(elo_vals):.0f}"
        )

    return df, dict(elo)


# ---------------------------------------------------------------------------
# Feature matrix builders (Subphase 3.7)
# ---------------------------------------------------------------------------


def build_feature_matrix(
    matches_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    is_predict: bool = False,
) -> pd.DataFrame:
    """Build the full feature matrix from a match DataFrame and rankings.

    Applies the complete feature engineering pipeline in order:
    rankings merge → form features → h2h features → context features.

    For training (``is_predict=False``) the returned DataFrame includes
    target columns (``outcome``, ``home_score``, ``away_score``) and
    sample-weight columns (``match_importance``, ``recency_weight``).
    Only matches from 1998 onward are included.

    For prediction (``is_predict=True``) only ``FEATURE_COLUMNS`` are
    returned, suitable for inference.

    Args:
        matches_df: DataFrame of matches.  Must contain ``date`` or
            ``match_date``, ``home_team``, and ``away_team``.  For the
            training path it must also have ``home_score``, ``away_score``,
            ``outcome``, ``match_importance``, and ``recency_weight``.
        rankings_df: FIFA rankings DataFrame as returned by
            ``load_raw_data()``.
        is_predict: If ``False`` (default), returns training matrix with
            target columns.  If ``True``, returns prediction matrix with
            ``FEATURE_COLUMNS`` only.

    Returns:
        A DataFrame with ``FEATURE_COLUMNS`` and (for training) additional
        target and weight columns.

    Raises:
        ValueError: If any ``FEATURE_COLUMNS`` column contains null values
            after the full pipeline.
    """
    df = matches_df.copy()

    # Normalise date column name so all pipeline steps use "date"
    if "match_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"match_date": "date"})

    ranked_df = merge_rankings(df, rankings_df, date_col="date")
    formed_df = compute_form_features(ranked_df)
    h2h_df = compute_h2h_features(formed_df)
    context_df = compute_context_features(h2h_df)

    if not is_predict:
        context_df = context_df[context_df["date"].dt.year >= 1998]
        output_cols = FEATURE_COLUMNS + [
            "date", "tournament",
            "home_team", "away_team",
            "home_score", "away_score", "outcome", "match_importance", "recency_weight",
        ]
        result = context_df[output_cols].copy()
    else:
        result = context_df[FEATURE_COLUMNS].copy()

    null_counts = result[FEATURE_COLUMNS].isna().sum()
    null_total = int(null_counts.sum())
    if null_total != 0:
        problem_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(
            f"FEATURE_COLUMNS contain {null_total} nulls after full pipeline: {problem_cols}"
        )

    return result


def export_features() -> None:
    """Run the full feature engineering pipeline and export parquet files.

    Produces two output files in ``data/processed/``:

    * ``features_train.parquet`` — training matrix with ``FEATURE_COLUMNS``
      plus target columns (``outcome``, ``home_score``, ``away_score``) and
      sample-weight columns (``match_importance``, ``recency_weight``).
    * ``features_predict.parquet`` — prediction matrix with ``FEATURE_COLUMNS``
      only, containing one row per WC 2026 fixture.

    The prediction matrix is built by concatenating historical results with
    WC 2026 fixtures so that form and head-to-head histories are correctly
    warmed up before the fixture rows are processed.

    Raises:
        ValueError: If any ``FEATURE_COLUMNS`` column contains null values.
    """
    _PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results_df, rankings_df, fixtures_df, _ = load_raw_data()
    cleaned = clean_results(results_df)

    # ------------------------------------------------------------------
    # Training matrix
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Building training feature matrix …")
    print("=" * 60)
    train_df = build_feature_matrix(cleaned, rankings_df, is_predict=False)
    train_path = _PROCESSED_DIR / "features_train.parquet"
    train_df.to_parquet(train_path, engine="pyarrow", index=False)
    print(f"\nSaved: {train_path}")

    # ------------------------------------------------------------------
    # Prediction matrix
    # Concatenate historical results with fixtures so form/h2h histories
    # are warmed up before the fixture rows are processed.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Building prediction feature matrix …")
    print("=" * 60)

    fixtures_prep = fixtures_df.copy()
    if "match_date" in fixtures_prep.columns and "date" not in fixtures_prep.columns:
        fixtures_prep = fixtures_prep.rename(columns={"match_date": "date"})
    # Ensure compute_context_features handles fixture rows correctly
    # when combined with historical data that has tournament/neutral columns
    fixtures_prep["tournament"] = "FIFA World Cup"
    fixtures_prep["neutral"] = 1

    combined = pd.concat([cleaned, fixtures_prep], ignore_index=True)

    ranked = merge_rankings(combined, rankings_df, date_col="date")
    formed = compute_form_features(ranked)
    h2h = compute_h2h_features(formed)
    context = compute_context_features(h2h)

    # Fixture rows are those without completed home_score / away_score
    predict_df = context[context["home_score"].isna()][FEATURE_COLUMNS].copy()

    null_sum = int(predict_df.isna().sum().sum())
    if null_sum != 0:
        problem = predict_df.isna().sum()[predict_df.isna().sum() > 0].to_dict()
        raise ValueError(
            f"Prediction features contain {null_sum} nulls after pipeline: {problem}"
        )

    predict_path = _PROCESSED_DIR / "features_predict.parquet"
    predict_df.to_parquet(predict_path, engine="pyarrow", index=False)
    print(f"\nSaved: {predict_path}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    train_nulls = train_df[FEATURE_COLUMNS].isna().sum()
    predict_nulls = predict_df.isna().sum()
    train_non_zero = train_nulls[train_nulls > 0].to_dict()
    predict_non_zero = predict_nulls[predict_nulls > 0].to_dict()

    feat_identical = (
        set(predict_df.columns) == set(FEATURE_COLUMNS)
        and set(FEATURE_COLUMNS).issubset(set(train_df.columns))
    )

    print("\n" + "=" * 60)
    print("--- features_train.parquet ---")
    print(f"Shape: {train_df.shape}")
    print(f"Columns: {train_df.columns.tolist()}")
    print(f"Null counts: {train_non_zero if train_non_zero else 'all zero'}")

    print("\n--- features_predict.parquet ---")
    print(f"Shape: {predict_df.shape}")
    print(f"Columns: {predict_df.columns.tolist()}")
    print(f"Null counts: {predict_non_zero if predict_non_zero else 'all zero'}")

    print(f"\nFeature columns identical: {feat_identical}")


# ---------------------------------------------------------------------------
# Script entry point — quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results, rankings, fixtures, name_map = load_raw_data()

    for label, df in [
        ("results_df", results),
        ("rankings_df", rankings),
        ("fixtures_df", fixtures),
        ("name_map_df", name_map),
    ]:
        print(f"{label:20s} shape: {df.shape}")

    print()
    print(f"results_df['date']         dtype: {results['date'].dtype}")
    print(f"rankings_df['rank_date']   dtype: {rankings['rank_date'].dtype}")
    print(f"fixtures_df['match_date']  dtype: {fixtures['match_date'].dtype}")

    # ------------------------------------------------------------------
    # Subphase 3.2 verification
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("clean_results() — Subphase 3.2 verification")
    print("=" * 55)

    cleaned = clean_results(results)
    print(f"Cleaned shape:  {cleaned.shape}")

    print()
    print("outcome value counts (0=away win, 1=draw, 2=home win):")
    print(cleaned["outcome"].value_counts().sort_index().to_string())

    print()
    print(f"match_importance unique values: {sorted(cleaned['match_importance'].unique())}")
    print(f"match_importance null count: {cleaned['match_importance'].isna().sum()}")
    print(f"outcome null count: {cleaned['outcome'].isna().sum()}")
    print(f"outcome unique values: {sorted(cleaned['outcome'].unique())}")

    print()
    print(f"recency_weight  min : {cleaned['recency_weight'].min():.6f}")
    print(f"recency_weight  max : {cleaned['recency_weight'].max():.6f}")

    print()
    print("Spot check — older matches must have lower recency_weight:")
    row_1990 = cleaned[cleaned["date"].dt.year == 1990].iloc[0]
    row_2022 = cleaned[
        (cleaned["date"].dt.year == 2022) & (cleaned["tournament"] == "FIFA World Cup")
    ].iloc[0]
    for label, row in [("1990 match   ", row_1990), ("2022 WC match", row_2022)]:
        print(
            f"  {label}: {row['date'].date()}  "
            f"{row['home_team']} vs {row['away_team']}  "
            f"recency_weight = {row['recency_weight']:.6f}"
        )

    # ------------------------------------------------------------------
    # Subphase 3.3 verification
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("merge_rankings() — Subphase 3.3 verification")
    print("=" * 55)

    ranked = merge_rankings(cleaned, rankings)

    ranking_cols = ["home_rank", "away_rank", "home_rank_points", "away_rank_points"]
    print("\nNull counts for ranking columns (must all be 0):")
    for col in ranking_cols:
        print(f"  {col}: {ranked[col].isna().sum()}")

    print(f"\nrank_diff       min: {ranked['rank_diff'].min():.2f}   max: {ranked['rank_diff'].max():.2f}")
    print(f"rank_points_ratio min: {ranked['rank_points_ratio'].min():.4f}   max: {ranked['rank_points_ratio'].max():.4f}")

    print("\nSpot check — France vs Croatia, WC 2018 Final (2018-07-15):")
    wc2018_final = ranked[
        (ranked["date"].dt.date == pd.Timestamp("2018-07-15").date())
        & (ranked["home_team"] == "France")
        & (ranked["away_team"] == "Croatia")
    ]
    if not wc2018_final.empty:
        row = wc2018_final.iloc[0]
        print(f"  home_rank (France): {row['home_rank']:.0f}   away_rank (Croatia): {row['away_rank']:.0f}")
        print(f"  home_rank_points:   {row['home_rank_points']:.1f}   away_rank_points: {row['away_rank_points']:.1f}")
        print(f"  rank_diff:          {row['rank_diff']:.0f}")
        print(f"  rank_points_ratio:  {row['rank_points_ratio']:.4f}")
    else:
        print("  Match not found — check team names.")

    # ------------------------------------------------------------------
    # Subphase 3.4 verification
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("compute_form_features() — Subphase 3.4 verification")
    print("=" * 60)

    formed = compute_form_features(ranked)

    FORM_COLS_5 = [
        "home_form_wins_5", "home_form_goals_scored_5",
        "home_form_goals_conceded_5", "home_form_wdl_points_5",
        "away_form_wins_5", "away_form_goals_scored_5",
        "away_form_goals_conceded_5", "away_form_wdl_points_5",
    ]
    FORM_COLS_10 = [c.replace("_5", "_10") for c in FORM_COLS_5]
    ALL_FORM_COLS = FORM_COLS_5 + FORM_COLS_10 + ["form_diff_wdl_5"]

    missing = [c for c in ALL_FORM_COLS if c not in formed.columns]
    print(f"\nAll 17 form columns present: {not missing}")
    if missing:
        print(f"  Missing: {missing}")

    print("\nNull counts (all must be 0):")
    all_null_ok = True
    for col in ALL_FORM_COLS:
        n = int(formed[col].isna().sum())
        flag = "OK" if n == 0 else "FAIL"
        if n != 0:
            all_null_ok = False
        print(f"  {flag}  {col}: {n}")
    print(f"All nulls zero: {all_null_ok}")

    wins5_min = formed["home_form_wins_5"].min()
    wins5_max = formed["home_form_wins_5"].max()
    wins10_min = formed["home_form_wins_10"].min()
    wins10_max = formed["home_form_wins_10"].max()
    print(f"\nhome_form_wins_5  range: [{wins5_min:.0f}, {wins5_max:.0f}]   expected [0, 5]")
    print(f"home_form_wins_10 range: [{wins10_min:.0f}, {wins10_max:.0f}]  expected [0, 10]")
    print(f"  wins_5  OK: {0 <= wins5_min and wins5_max <= 5}")
    print(f"  wins_10 OK: {0 <= wins10_min and wins10_max <= 10}")

    goals_cols = [c for c in ALL_FORM_COLS if "goals" in c]
    goals_ok = all(
        formed[c].min() >= 0 and formed[c].max() <= 15 for c in goals_cols
    )
    print(f"\nGoals cols all in [0, 15]: {goals_ok}")
    for c in goals_cols:
        print(f"  {c}: min={formed[c].min():.3f}  max={formed[c].max():.3f}")

    # Spot check: manually compute Brazil's form before 2022-12-09 QF vs Croatia
    cutoff = pd.Timestamp("2022-12-09")
    brazil_hist = ranked[
        ((ranked["home_team"] == "Brazil") | (ranked["away_team"] == "Brazil"))
        & (ranked["date"] < cutoff)
    ].sort_values("date").tail(5)
    print(f"\nSpot check — Brazil's last 5 matches before {cutoff.date()}:")
    print(brazil_hist[["date", "home_team", "away_team", "home_score", "away_score"]].to_string(index=False))

    # Manual computation from the printed rows above
    manual_wins = 0
    manual_scored = 0.0
    manual_wdl = 0
    for _, brow in brazil_hist.iterrows():
        if brow["home_team"] == "Brazil":
            gs, gc = int(brow["home_score"]), int(brow["away_score"])
        else:
            gs, gc = int(brow["away_score"]), int(brow["home_score"])
        if gs > gc:
            pts, win = 3, 1
        elif gs == gc:
            pts, win = 1, 0
        else:
            pts, win = 0, 0
        manual_wins += win
        manual_scored += gs
        manual_wdl += pts
    n_rows = len(brazil_hist)
    manual_avg_scored = manual_scored / n_rows if n_rows else float("nan")

    # Look up the computed value for Brazil in the 2022-12-09 match
    br_match = formed[
        ((formed["home_team"] == "Brazil") | (formed["away_team"] == "Brazil"))
        & (formed["date"] == cutoff)
    ]
    if not br_match.empty:
        bm = br_match.iloc[0]
        is_home = bm["home_team"] == "Brazil"
        side_pfx = "home" if is_home else "away"
        comp_wins = bm[f"{side_pfx}_form_wins_5"]
        comp_scored = bm[f"{side_pfx}_form_goals_scored_5"]
        comp_wdl = bm[f"{side_pfx}_form_wdl_points_5"]
        print(f"\nManual  wins={manual_wins}  avg_scored={manual_avg_scored:.4f}  wdl={manual_wdl}")
        print(f"Computed wins={comp_wins:.0f}  avg_scored={comp_scored:.4f}  wdl={comp_wdl:.0f}")
        print(f"Spot check PASS: {manual_wins == comp_wins and abs(manual_avg_scored - comp_scored) < 1e-6 and manual_wdl == comp_wdl}")
    else:
        print("  Brazil's 2022-12-09 match not found — check dataset.")

    # ------------------------------------------------------------------
    # Subphase 3.5 verification
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("compute_h2h_features() — Subphase 3.5 verification")
    print("=" * 60)

    h2h_df = compute_h2h_features(formed)

    H2H_COLS = ["h2h_home_win_rate", "h2h_matches_count", "h2h_avg_goals_home", "h2h_avg_goals_away"]
    missing_h2h = [c for c in H2H_COLS if c not in h2h_df.columns]
    print(f"\nAll 4 h2h columns present: {not missing_h2h}")

    print("\nNull counts (all must be 0):")
    for col in H2H_COLS:
        n = int(h2h_df[col].isna().sum())
        flag = "OK" if n == 0 else "FAIL"
        print(f"  {flag}  {col}: {n}")

    print(f"\nh2h_home_win_rate  range: [{h2h_df['h2h_home_win_rate'].min():.3f}, "
          f"{h2h_df['h2h_home_win_rate'].max():.3f}]")
    print(f"h2h_matches_count  range: [{int(h2h_df['h2h_matches_count'].min())}, "
          f"{int(h2h_df['h2h_matches_count'].max())}]")
    print(f"h2h_avg_goals_home range: [{h2h_df['h2h_avg_goals_home'].min():.3f}, "
          f"{h2h_df['h2h_avg_goals_home'].max():.3f}]")
    print(f"h2h_avg_goals_away range: [{h2h_df['h2h_avg_goals_away'].min():.3f}, "
          f"{h2h_df['h2h_avg_goals_away'].max():.3f}]")

    # Spot check: Germany vs Brazil, WC 2014 Semi (2014-07-08), Belo Horizonte
    # Brazil were home team in the data; Germany won 7-1
    # Prior meetings between Germany and Brazil should exist in dataset
    cutoff_h2h = pd.Timestamp("2014-07-08")
    h2h_match = h2h_df[
        (h2h_df["date"] == cutoff_h2h)
        & (h2h_df["home_team"] == "Brazil")
        & (h2h_df["away_team"] == "Germany")
    ]
    if not h2h_match.empty:
        hm = h2h_match.iloc[0]
        print(f"\nSpot check — Brazil vs Germany, 2014-07-08 WC Semi:")
        print(f"  h2h_matches_count:  {int(hm['h2h_matches_count'])}")
        print(f"  h2h_home_win_rate:  {hm['h2h_home_win_rate']:.3f}  (Brazil win rate in prior H2H)")
        print(f"  h2h_avg_goals_home: {hm['h2h_avg_goals_home']:.3f}  (Brazil avg goals in prior H2H)")
        print(f"  h2h_avg_goals_away: {hm['h2h_avg_goals_away']:.3f}  (Germany avg goals in prior H2H)")
    else:
        print("\nSpot check match not found — check team names.")

    # ------------------------------------------------------------------
    # Subphase 3.6 verification
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("compute_context_features() — Subphase 3.6 verification")
    print("=" * 60)

    context_df = compute_context_features(h2h_df)

    CONTEXT_COLS = [
        "tournament_stage", "is_wc_match", "is_neutral_venue",
        "host_nation_advantage", "home_days_rest", "away_days_rest",
    ]

    missing_ctx = [c for c in CONTEXT_COLS if c not in context_df.columns]
    print(f"\nAll 6 context columns present: {not missing_ctx}")
    if missing_ctx:
        print(f"  Missing: {missing_ctx}")

    print("\nNull counts (all must be 0):")
    all_null_ok_ctx = True
    for col in CONTEXT_COLS:
        n = int(context_df[col].isna().sum())
        flag = "OK" if n == 0 else "FAIL"
        if n != 0:
            all_null_ok_ctx = False
        print(f"  {flag}  {col}: {n}")
    print(f"All nulls zero: {all_null_ok_ctx}")

    stage_vals = set(context_df["tournament_stage"].unique())
    stage_ok = stage_vals.issubset(set(range(7)))
    print(f"\ntournament_stage unique values: {sorted(stage_vals)}  (expect subset of 0-6)")
    print(f"tournament_stage range OK: {stage_ok}")

    wc_vals = set(context_df["is_wc_match"].unique())
    wc_binary_ok = wc_vals.issubset({0, 1})
    print(f"\nis_wc_match unique values: {sorted(wc_vals)}  (expect subset of {{0, 1}})")
    print(f"is_wc_match binary OK: {wc_binary_ok}")

    iv_vals = set(context_df["is_neutral_venue"].unique())
    iv_binary_ok = iv_vals.issubset({0, 1})
    print(f"\nis_neutral_venue unique values: {sorted(iv_vals)}  (expect subset of {{0, 1}})")
    print(f"is_neutral_venue binary OK: {iv_binary_ok}")

    home_rest_min = int(context_df["home_days_rest"].min())
    home_rest_max = int(context_df["home_days_rest"].max())
    away_rest_min = int(context_df["away_days_rest"].min())
    away_rest_max = int(context_df["away_days_rest"].max())
    rest_ok = home_rest_min >= 0 and away_rest_min >= 0
    print(f"\nhome_days_rest  min: {home_rest_min}  max: {home_rest_max}  (min must be >= 0)")
    print(f"away_days_rest  min: {away_rest_min}  max: {away_rest_max}  (min must be >= 0)")
    print(f"Days rest range OK: {rest_ok}")

    wc2022_final = context_df[
        (context_df["date"].dt.date == pd.Timestamp("2022-12-18").date())
        & (context_df["tournament"] == "FIFA World Cup")
    ]
    if not wc2022_final.empty:
        stage_final = int(wc2022_final.iloc[0]["tournament_stage"])
        print(
            f"\nSpot check — WC 2022 Final (2022-12-18) tournament_stage: "
            f"{stage_final}  (expect 6)  {'PASS' if stage_final == 6 else 'FAIL'}"
        )
    else:
        print("\nWC 2022 Final match not found — check dataset.")

    wc2022_qf = context_df[
        (context_df["date"].dt.date == pd.Timestamp("2022-12-09").date())
        & (context_df["tournament"] == "FIFA World Cup")
    ]
    if not wc2022_qf.empty:
        stage_qf = int(wc2022_qf.iloc[0]["tournament_stage"])
        print(
            f"Spot check — WC 2022 QF (2022-12-09) tournament_stage: "
            f"{stage_qf}  (expect 4)  {'PASS' if stage_qf == 4 else 'FAIL'}"
        )

    sf_match = context_df[
        (context_df["date"].dt.date == pd.Timestamp("2014-07-08").date())
        & (context_df["home_team"] == "Brazil")
        & (context_df["away_team"] == "Germany")
    ]
    if not sf_match.empty:
        stage_sf = int(sf_match.iloc[0]["tournament_stage"])
        print(
            f"Spot check — 2014 SF Brazil vs Germany tournament_stage: "
            f"{stage_sf}  (expect 5)  {'PASS' if stage_sf == 5 else 'FAIL'}"
        )

    hna_hist = int(context_df["host_nation_advantage"].sum())
    print(f"\nhost_nation_advantage sum (historical data): {hna_hist}  (expect 0)")

    _, _, fixtures_df_ctx, _ = load_raw_data()
    fixtures_context = compute_context_features(fixtures_df_ctx)
    hna_fixtures = int(fixtures_context["host_nation_advantage"].sum())
    print(f"host_nation_advantage sum (WC 2026 fixtures): {hna_fixtures}  (expect > 0)")
    print(f"host_nation_advantage >= 1 on fixtures: {hna_fixtures > 0}")
    hna_rows = fixtures_context[fixtures_context["host_nation_advantage"] == 1][
        ["match_date", "home_team", "away_team", "tournament_stage", "host_nation_advantage"]
    ]
    if not hna_rows.empty:
        print("Sample host_nation_advantage=1 rows:")
        print(hna_rows.head(5).to_string(index=False))

    print("\nWC 2026 fixtures tournament_stage value counts:")
    print(fixtures_context["tournament_stage"].value_counts().sort_index())

