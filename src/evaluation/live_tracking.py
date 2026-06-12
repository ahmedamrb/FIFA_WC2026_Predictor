"""Compare WC 2026 predictions against live / full-time actual results.

Pure (Streamlit-free) helpers that join the prediction table to the live
results feed and flag, per match, whether the predicted outcome (W/D/L) and
the exact scoreline matched reality.

Joins
-----
- ``results_df`` is keyed by ``fixture_id`` (see src/data/ingest.fetch_wc2026_results).
- ``predictions_df`` is row-aligned with ``fixtures_df`` (same index = original CSV
  row position), mirroring how app/components/bracket.py looks predictions up.

Knockout fixtures whose teams are still ``TBD`` carry placeholder predictions and
are flagged ``comparable = False`` so they never count toward accuracy.

Outcome encoding follows the rest of the project as label strings:
``"Home Win"`` / ``"Draw"`` / ``"Away Win"`` (matching ``predicted_outcome``).
"""

from __future__ import annotations

import pandas as pd

# football-data.org status groupings
_LIVE_STATUSES = {"IN_PLAY", "PAUSED"}
_FINAL_STATUSES = {"FINISHED", "AWARDED"}

_WINNER_TO_OUTCOME = {
    "HOME_TEAM": "Home Win",
    "AWAY_TEAM": "Away Win",
    "DRAW": "Draw",
}

# Columns produced by build_comparison()
COMPARISON_COLUMNS = [
    "fixture_id",
    "home_team",
    "away_team",
    "match_date",
    "stage",
    "status",
    "home_score",
    "away_score",
    "minute",
    "predicted_outcome",
    "actual_outcome",
    "outcome_correct",
    "predicted_home_goals",
    "predicted_away_goals",
    "exact_score_correct",
    "home_goals_correct",
    "away_goals_correct",
    "home_goals_error",
    "away_goals_error",
    "comparable",
    "has_score",
    "is_live",
    "played",
]


def _to_int_or_none(value) -> int | None:
    """Return an int for a non-null numeric score, else None."""
    if value is None or pd.isna(value):
        return None
    return int(value)


def actual_outcome_from_row(row) -> str | None:
    """Derive the actual W/D/L label for a results row.

    Prefers football-data.org's ``winner`` field; falls back to comparing the
    full-time scores. Returns ``None`` when no score is available yet.
    """
    winner = row.get("winner") if hasattr(row, "get") else row["winner"]
    if winner in _WINNER_TO_OUTCOME:
        return _WINNER_TO_OUTCOME[winner]

    home = _to_int_or_none(row.get("home_score") if hasattr(row, "get") else row["home_score"])
    away = _to_int_or_none(row.get("away_score") if hasattr(row, "get") else row["away_score"])
    if home is None or away is None:
        return None
    if home > away:
        return "Home Win"
    if home < away:
        return "Away Win"
    return "Draw"


def build_comparison(
    fixtures_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None,
    results_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join fixtures, predictions and live results into one comparison table.

    Args:
        fixtures_df: data/raw/wc2026_fixtures_flat.csv (default RangeIndex);
            needs ``fixture_id, match_date, home_team, away_team, stage``.
        predictions_df: data/processed/wc2026_predictions.csv loaded with
            index_col=0 (index aligned to fixtures row position). May be None.
        results_df: output of fetch_wc2026_results (keyed by ``fixture_id``).
            May be None.

    Returns:
        A DataFrame with one row per fixture and the columns in
        ``COMPARISON_COLUMNS``. ``outcome_correct`` / ``exact_score_correct``
        are nullable booleans (``pd.NA`` when not yet decidable or not
        comparable).
    """
    base_cols = ["fixture_id", "match_date", "home_team", "away_team", "stage"]
    comp = fixtures_df[base_cols].copy()

    # --- attach predictions by row index (same alignment as the bracket) ---
    pred_cols = ["predicted_outcome", "predicted_home_goals", "predicted_away_goals"]
    if predictions_df is not None:
        present = [c for c in pred_cols if c in predictions_df.columns]
        comp = comp.join(predictions_df[present])
    for col in pred_cols:
        if col not in comp.columns:
            comp[col] = pd.NA

    # --- attach results by fixture_id ---
    res_cols = ["status", "home_score", "away_score", "winner", "minute"]
    if results_df is not None and not results_df.empty:
        res = results_df.set_index("fixture_id")
        present = [c for c in res_cols if c in res.columns]
        comp = comp.merge(res[present], left_on="fixture_id", right_index=True, how="left")
    for col in res_cols:
        if col not in comp.columns:
            comp[col] = pd.NA

    # --- derived flags ---
    comp["comparable"] = (comp["home_team"] != "TBD") & (comp["away_team"] != "TBD")
    comp["has_score"] = comp["home_score"].notna() & comp["away_score"].notna()
    comp["is_live"] = comp["status"].isin(_LIVE_STATUSES)
    comp["played"] = comp["status"].isin(_FINAL_STATUSES)

    comp["actual_outcome"] = comp.apply(
        lambda r: actual_outcome_from_row(r) if r["has_score"] else None, axis=1
    )

    def _outcome_correct(r):
        if not (r["comparable"] and r["has_score"]) or pd.isna(r["predicted_outcome"]):
            return pd.NA
        return bool(r["predicted_outcome"] == r["actual_outcome"])

    def _exact_correct(r):
        if not (r["comparable"] and r["has_score"]):
            return pd.NA
        if pd.isna(r["predicted_home_goals"]) or pd.isna(r["predicted_away_goals"]):
            return pd.NA
        return bool(
            _to_int_or_none(r["predicted_home_goals"]) == _to_int_or_none(r["home_score"])
            and _to_int_or_none(r["predicted_away_goals"]) == _to_int_or_none(r["away_score"])
        )

    comp["outcome_correct"] = comp.apply(_outcome_correct, axis=1).astype("boolean")
    comp["exact_score_correct"] = comp.apply(_exact_correct, axis=1).astype("boolean")

    # --- per-side goals: home and away goals come from two separate models ---
    def _side_goals_correct(pred_col, score_col):
        def fn(r):
            if not (r["comparable"] and r["has_score"]):
                return pd.NA
            if pd.isna(r[pred_col]) or pd.isna(r[score_col]):
                return pd.NA
            return bool(_to_int_or_none(r[pred_col]) == _to_int_or_none(r[score_col]))
        return fn

    def _side_goals_error(pred_col, score_col):
        def fn(r):
            if not (r["comparable"] and r["has_score"]):
                return pd.NA
            if pd.isna(r[pred_col]) or pd.isna(r[score_col]):
                return pd.NA
            return float(_to_int_or_none(r[pred_col]) - _to_int_or_none(r[score_col]))
        return fn

    comp["home_goals_correct"] = comp.apply(
        _side_goals_correct("predicted_home_goals", "home_score"), axis=1).astype("boolean")
    comp["away_goals_correct"] = comp.apply(
        _side_goals_correct("predicted_away_goals", "away_score"), axis=1).astype("boolean")
    comp["home_goals_error"] = comp.apply(
        _side_goals_error("predicted_home_goals", "home_score"), axis=1).astype("Float64")
    comp["away_goals_error"] = comp.apply(
        _side_goals_error("predicted_away_goals", "away_score"), axis=1).astype("Float64")

    return comp[COMPARISON_COLUMNS]


def summarize(comparison_df: pd.DataFrame) -> dict:
    """Summarise live tracking accuracy over decided, comparable matches.

    Only ``FINISHED`` matches with known teams count toward accuracy; in-play
    matches are reported separately via ``live``. The home and away goals come
    from two separate regressors, so per-side exact-hit rates and mean absolute
    errors (MAE) are reported independently.

    Returns a dict with keys: ``played``, ``outcome_correct``, ``outcome_pct``,
    ``exact``, ``exact_pct``, ``home_goals_correct``, ``home_goals_pct``,
    ``home_goals_mae``, ``away_goals_correct``, ``away_goals_pct``,
    ``away_goals_mae``, ``live``.
    """
    empty = {
        "played": 0, "outcome_correct": 0, "outcome_pct": 0.0,
        "exact": 0, "exact_pct": 0.0,
        "home_goals_correct": 0, "home_goals_pct": 0.0, "home_goals_mae": 0.0,
        "away_goals_correct": 0, "away_goals_pct": 0.0, "away_goals_mae": 0.0,
        "live": 0,
    }
    if comparison_df is None or comparison_df.empty:
        return empty

    done = comparison_df[comparison_df["comparable"] & comparison_df["played"]]
    n = len(done)
    correct = int(done["outcome_correct"].fillna(False).astype(bool).sum())
    exact = int(done["exact_score_correct"].fillna(False).astype(bool).sum())
    home_gc = int(done["home_goals_correct"].fillna(False).astype(bool).sum())
    away_gc = int(done["away_goals_correct"].fillna(False).astype(bool).sum())
    home_mae = float(done["home_goals_error"].astype("float").abs().mean()) if n else 0.0
    away_mae = float(done["away_goals_error"].astype("float").abs().mean()) if n else 0.0
    live = int((comparison_df["comparable"] & comparison_df["is_live"]).sum())

    return {
        "played": n,
        "outcome_correct": correct,
        "outcome_pct": (correct / n) if n else 0.0,
        "exact": exact,
        "exact_pct": (exact / n) if n else 0.0,
        "home_goals_correct": home_gc,
        "home_goals_pct": (home_gc / n) if n else 0.0,
        "home_goals_mae": home_mae,
        "away_goals_correct": away_gc,
        "away_goals_pct": (away_gc / n) if n else 0.0,
        "away_goals_mae": away_mae,
        "live": live,
    }
