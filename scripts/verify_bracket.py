"""Standalone verification script for Subphase 7.3 — Tournament Bracket.

Loads all required data and model files, exercises every bracket helper
function, and asserts correctness. Prints group standings, bracket tree,
and "ALL ASSERTIONS PASSED" on success.

Usage (from repo root, venv activated):
    python scripts/verify_bracket.py
"""

import sys
from pathlib import Path

# Make repo root importable so app/ and src/ packages resolve correctly
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import joblib
import pandas as pd

from src.models.ensemble import WC2026Ensemble
from app.components.bracket import (
    FEATURE_COLUMNS,
    _build_rank_lookup,
    _identify_groups,
    _simulate_group_stage,
    _select_qualifiers,
    _build_bracket_tree,
    _draw_bracket_figure,
    _predict_knockout_match,
    _prob_to_color,
)

_MODELS = _REPO_ROOT / "models"
_PROCESSED = _REPO_ROOT / "data" / "processed"
_RAW = _REPO_ROOT / "data" / "raw"


def load_resources():
    print("Loading models...")
    outcome_lr  = joblib.load(_MODELS / "outcome_lr.pkl")
    outcome_rf  = joblib.load(_MODELS / "outcome_rf.pkl")
    outcome_xgb = joblib.load(_MODELS / "outcome_xgb.pkl")
    ensemble = WC2026Ensemble(outcome_lr, outcome_rf, outcome_xgb)

    fp = _PROCESSED / "features_predict.parquet"
    assert fp.exists(), f"Missing {fp} — run scripts/run_feature_engineering.py first"
    features_predict = pd.read_parquet(fp)

    fixtures = pd.read_csv(_RAW / "wc2026_fixtures_flat.csv", parse_dates=["match_date"])

    print(f"  features_predict shape: {features_predict.shape}")
    print(f"  fixtures shape:         {fixtures.shape}")
    return ensemble, features_predict, fixtures


def verify_rank_lookup():
    print("\n--- _build_rank_lookup ---")
    rank_lookup = _build_rank_lookup()
    assert isinstance(rank_lookup, dict), "rank_lookup must be a dict"
    assert len(rank_lookup) > 0, "rank_lookup must not be empty"
    for team, rank in list(rank_lookup.items())[:5]:
        print(f"  {team}: {rank}")
    # All values must be positive integers
    for team, rank in rank_lookup.items():
        assert isinstance(rank, int) and rank > 0, (
            f"Rank for '{team}' is {rank!r}, expected positive int"
        )
    print(f"  Total teams in lookup: {len(rank_lookup)}")
    return rank_lookup


def verify_identify_groups(fixtures):
    print("\n--- _identify_groups ---")
    groups = _identify_groups(fixtures)
    assert len(groups) == 12, f"Expected 12 groups, got {len(groups)}"
    for grp in groups:
        assert len(grp) == 4, f"Group {grp} has {len(grp)} teams"
    all_teams = [t for g in groups for t in g]
    assert len(set(all_teams)) == 48, f"Expected 48 unique teams, got {len(set(all_teams))}"
    print(f"  12 groups confirmed; 48 unique GROUP_STAGE teams confirmed")
    for i, grp in enumerate(groups):
        label = chr(ord("A") + i)
        print(f"  Group {label}: {grp}")
    return groups


def verify_group_stage(fixtures, features, ensemble, groups, rank_lookup):
    print("\n--- _simulate_group_stage ---")
    standings = _simulate_group_stage(fixtures, features, ensemble, groups, rank_lookup)
    assert len(standings) == 12, f"Expected 12 group standings, got {len(standings)}"
    for label, df in sorted(standings.items()):
        assert len(df) == 4, f"Group {label} has {len(df)} rows, expected 4"
        assert list(df.columns) == ["team", "expected_pts", "rank"]
        assert df["expected_pts"].sum() > 0, f"Group {label} has zero expected pts"
    # Print all standings
    for label in sorted(standings.keys()):
        df = standings[label]
        print(f"\n  Group {label}:")
        for _, row in df.iterrows():
            print(f"    {row['team']:25s}  exp_pts={row['expected_pts']:.2f}  rank={row['rank']}")
    return standings


def verify_select_qualifiers(standings, rank_lookup):
    print("\n--- _select_qualifiers ---")
    qualifiers = _select_qualifiers(standings, rank_lookup)
    assert len(qualifiers) == 32, f"Expected 32 qualifiers, got {len(qualifiers)}"
    assert len(set(qualifiers)) == 32, "Duplicate teams in qualifiers list"
    for q in qualifiers:
        assert isinstance(q, str) and len(q) > 0, f"Invalid qualifier: {q!r}"
    print(f"  32 unique qualifiers confirmed")
    for i, team in enumerate(qualifiers, 1):
        print(f"  Seed {i:2d}: {team} (rank {rank_lookup.get(team, 200)})")
    return qualifiers


def verify_bracket_tree(qualifiers, rank_lookup):
    print("\n--- _build_bracket_tree ---")
    tree = _build_bracket_tree(qualifiers, rank_lookup)

    expected_counts = {
        "LAST_32": 16,
        "LAST_16": 8,
        "QUARTER_FINALS": 4,
        "SEMI_FINALS": 2,
        "FINAL": 1,
        "THIRD_PLACE": 1,
    }
    for rnd, expected_n in expected_counts.items():
        assert rnd in tree, f"Missing round '{rnd}' in bracket tree"
        actual_n = len(tree[rnd])
        assert actual_n == expected_n, (
            f"Round '{rnd}': expected {expected_n} matches, got {actual_n}"
        )

    for rnd, matches in tree.items():
        for match in matches:
            assert "winner" in match and isinstance(match["winner"], str), (
                f"Match in {rnd} has no valid winner"
            )
            assert "win_prob" in match, f"Match in {rnd} missing win_prob"
            assert 0.50 <= match["win_prob"] <= 1.0, (
                f"win_prob {match['win_prob']} out of [0.5, 1.0] in {rnd}"
            )
            assert "loser" in match and match["loser"] != match["winner"], (
                f"Match in {rnd}: winner == loser"
            )

    # Print full bracket
    print("\n  Full bracket tree:")
    for rnd in ["LAST_32", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL", "THIRD_PLACE"]:
        print(f"\n  {rnd}:")
        for i, m in enumerate(tree[rnd]):
            print(
                f"    [{i}] {m['team1']} vs {m['team2']} → "
                f"winner={m['winner']} ({m['win_prob']:.0%}), loser={m['loser']}"
            )

    return tree


def verify_figure(tree):
    print("\n--- _draw_bracket_figure ---")
    fig = _draw_bracket_figure(tree)
    assert fig is not None, "Figure is None"
    assert len(fig.layout.shapes) > 0, "Figure has no shapes (expected match boxes)"
    assert len(fig.data) > 0, "Figure has no traces (expected connector lines)"
    print(f"  Shapes (match boxes + third place): {len(fig.layout.shapes)}")
    print(f"  Traces (connector lines): {len(fig.data)}")
    print("  Figure verified OK")


def verify_helpers():
    print("\n--- Helper function spot-checks ---")

    # _predict_knockout_match: better-ranked team should win
    winner, prob = _predict_knockout_match("Brazil", "San Marino", {"Brazil": 5, "San Marino": 195})
    assert winner == "Brazil", f"Expected Brazil to beat San Marino, got {winner}"
    assert 0.50 <= prob <= 0.97, f"win_prob {prob} out of range"

    # Equal ranks → team1 wins with prob 0.5
    winner2, prob2 = _predict_knockout_match("A", "B", {"A": 10, "B": 10})
    assert winner2 == "A", f"Expected A (team1) to win equal-rank match, got {winner2}"
    assert abs(prob2 - 0.5) < 1e-9, f"Expected prob=0.5 for equal ranks, got {prob2}"

    # _prob_to_color boundaries
    light = _prob_to_color(0.50)
    dark  = _prob_to_color(0.97)
    assert light.startswith("rgb("), f"Unexpected color format: {light}"
    assert dark.startswith("rgb("), f"Unexpected color format: {dark}"
    print(f"  _prob_to_color(0.50) = {light}")
    print(f"  _prob_to_color(0.97) = {dark}")
    print("  Helper checks passed")


def main():
    ensemble, features_predict, fixtures = load_resources()

    verify_helpers()
    rank_lookup = verify_rank_lookup()
    groups = verify_identify_groups(fixtures)
    standings = verify_group_stage(fixtures, features_predict, ensemble, groups, rank_lookup)
    qualifiers = verify_select_qualifiers(standings, rank_lookup)
    tree = verify_bracket_tree(qualifiers, rank_lookup)
    verify_figure(tree)

    print("\n" + "=" * 50)
    print("ALL ASSERTIONS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
