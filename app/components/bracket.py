"""Tournament bracket UI component for FIFA WC 2026 Predictor."""

from __future__ import annotations

from collections import defaultdict
from math import exp
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RAW = _REPO_ROOT / "data" / "raw"

FEATURE_COLUMNS = [
    "home_rank", "away_rank", "home_rank_points", "away_rank_points",
    "rank_diff", "rank_points_ratio", "rank_points_diff",
    "home_form_wins_5", "home_form_goals_scored_5", "home_form_goals_conceded_5",
    "home_form_wdl_points_5",
    "away_form_wins_5", "away_form_goals_scored_5", "away_form_goals_conceded_5",
    "away_form_wdl_points_5",
    "home_form_wins_10", "home_form_goals_scored_10", "home_form_goals_conceded_10",
    "home_form_wdl_points_10",
    "away_form_wins_10", "away_form_goals_scored_10", "away_form_goals_conceded_10",
    "away_form_wdl_points_10",
    "form_diff_wdl_5", "home_goal_efficiency_5", "away_goal_efficiency_5",
    "h2h_home_win_rate", "h2h_matches_count", "h2h_avg_goals_home", "h2h_avg_goals_away",
    "tournament_stage", "is_wc_match", "is_neutral_venue", "host_nation_advantage",
    "home_days_rest", "away_days_rest", "rest_diff",
]

# Bracket figure constants
BOX_W = 0.175
BOX_H_HALF = 0.022
Y_MIN, Y_MAX = 0.02, 0.98

ROUND_ORDER = ["LAST_32", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"]
ROUND_X = {
    "LAST_32":        0.01,
    "LAST_16":        0.215,
    "QUARTER_FINALS": 0.42,
    "SEMI_FINALS":    0.625,
    "FINAL":          0.83,
}
ROUND_LABELS = {
    "LAST_32":        "Round of 32",
    "LAST_16":        "Round of 16",
    "QUARTER_FINALS": "Quarter Finals",
    "SEMI_FINALS":    "Semi Finals",
    "FINAL":          "Final",
}


def _build_rank_lookup() -> dict[str, int]:
    """Build fixture_name → FIFA rank mapping using latest available ranking data.

    Rankings CSV columns: (unnamed idx), rank, country_full, country_abrv,
    total_points, previous_points, rank_change, confederation, rank_date.
    Team name map columns: fixture_name, results_name, rankings_name.
    """
    rankings = pd.read_csv(_RAW / "rankings.csv")
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])
    rankings = rankings.dropna(subset=["rank", "country_full"])

    # Keep only the most recent ranking entry per team
    latest_idx = rankings.groupby("country_full")["rank_date"].idxmax()
    latest = rankings.loc[latest_idx].set_index("country_full")["rank"]
    latest_dict: dict[str, int] = {team: int(r) for team, r in latest.items()}

    name_map = pd.read_csv(_RAW / "team_name_map.csv")
    lookup: dict[str, int] = {}
    for _, row in name_map.iterrows():
        fixture_name = row["fixture_name"]
        rankings_name = row["rankings_name"]
        lookup[fixture_name] = latest_dict.get(rankings_name, 200)

    return lookup


def _identify_groups(fixtures_df: pd.DataFrame) -> list[list[str]]:
    """Identify 12 groups of 4 teams from GROUP_STAGE fixtures.

    Builds an adjacency set per team (opponents). Each team has exactly 3
    opponents in its group, so frozenset({team} ∪ adj[team]) has 4 elements.
    Deduplicates to produce 12 unique groups, sorted by first team name.
    """
    group_stage = fixtures_df[fixtures_df["stage"] == "GROUP_STAGE"]
    adj: dict[str, set[str]] = defaultdict(set)
    for _, row in group_stage.iterrows():
        adj[row["home_team"]].add(row["away_team"])
        adj[row["away_team"]].add(row["home_team"])

    seen: set[frozenset] = set()
    groups: list[list[str]] = []
    for team in sorted(adj.keys()):
        group_set = frozenset([team] + list(adj[team]))
        if group_set not in seen:
            seen.add(group_set)
            groups.append(sorted(group_set))

    groups.sort(key=lambda g: g[0])
    assert len(groups) == 12, f"Expected 12 groups, got {len(groups)}"
    for grp in groups:
        assert len(grp) == 4, f"Group starting with '{grp[0]}' has {len(grp)} teams, expected 4"
    return groups


def _simulate_group_stage(
    fixtures_df: pd.DataFrame,
    features_df: pd.DataFrame,
    ensemble,
    groups: list[list[str]],
    rank_lookup: dict[str, int],
) -> dict[str, pd.DataFrame]:
    """Compute expected points per team from ML model probabilities.

    For each GROUP_STAGE fixture the model returns [p_away_win, p_draw, p_home_win].
    Expected points: home += 3*p_hw + 1*p_draw; away += 3*p_aw + 1*p_draw.

    The row's .name is its original 0-based CSV RangeIndex, which maps directly
    to features_df.iloc[row.name] (features are in the same CSV row order).

    Returns a dict mapping group label (A–L) to a sorted DataFrame with columns
    ['team', 'expected_pts', 'rank'], sorted by expected_pts DESC then rank ASC.
    """
    labels = [chr(ord("A") + i) for i in range(12)]  # A through L
    expected_pts: dict[str, float] = defaultdict(float)

    group_stage = fixtures_df[fixtures_df["stage"] == "GROUP_STAGE"]
    for _, row in group_stage.iterrows():
        row_idx = row.name  # original 0-based CSV position
        X = features_df.iloc[[row_idx]][FEATURE_COLUMNS]
        proba = ensemble.predict_proba(X)[0]  # [p_away_win, p_draw, p_home_win]
        expected_pts[row["home_team"]] += 3 * proba[2] + 1 * proba[1]
        expected_pts[row["away_team"]] += 3 * proba[0] + 1 * proba[1]

    standings: dict[str, pd.DataFrame] = {}
    for label, grp in zip(labels, groups):
        rows = [
            {
                "team": team,
                "expected_pts": expected_pts.get(team, 0.0),
                "rank": rank_lookup.get(team, 200),
            }
            for team in grp
        ]
        df = pd.DataFrame(rows)
        df = df.sort_values(
            ["expected_pts", "rank"], ascending=[False, True]
        ).reset_index(drop=True)
        standings[label] = df

    return standings


def _select_qualifiers(
    group_standings: dict[str, pd.DataFrame],
    rank_lookup: dict[str, int],
) -> list[str]:
    """Select 32 qualifiers: top 2 per group (24) + best 8 third-placed teams.

    All 32 are then sorted by FIFA rank ascending (best rank = seed 1).
    """
    qualifiers: list[str] = []
    # (expected_pts DESC, rank ASC) for sorting third-place teams
    third_place: list[tuple[float, int, str]] = []

    for label in sorted(group_standings.keys()):
        df = group_standings[label]
        qualifiers.append(df.iloc[0]["team"])
        qualifiers.append(df.iloc[1]["team"])
        third_row = df.iloc[2]
        rank = rank_lookup.get(third_row["team"], 200)
        third_place.append((third_row["expected_pts"], rank, third_row["team"]))

    third_place.sort(key=lambda t: (-t[0], t[1]))
    qualifiers.extend(t[2] for t in third_place[:8])

    # Seed by FIFA rank: lowest rank number = strongest = first seed
    qualifiers.sort(key=lambda t: rank_lookup.get(t, 200))
    return qualifiers


def _predict_knockout_match(
    team1: str,
    team2: str,
    rank_lookup: dict[str, int],
) -> tuple[str, float]:
    """Predict knockout match winner via logistic function on FIFA rank difference.

    rank_diff > 0 means team1 has a better (lower) rank number → favoured.
    win_prob is capped to [0.50, 0.97].
    """
    rank1 = rank_lookup.get(team1, 200)
    rank2 = rank_lookup.get(team2, 200)
    rank_diff = rank2 - rank1  # positive → team1 has better (lower) rank
    raw_prob = 1 / (1 + exp(-rank_diff / 25))
    if raw_prob >= 0.5:
        return team1, min(0.97, raw_prob)
    else:
        return team2, min(0.97, 1 - raw_prob)


def _build_bracket_tree(
    qualifiers_32: list[str],
    rank_lookup: dict[str, int],
) -> dict[str, list[dict]]:
    """Simulate all knockout rounds and return a bracket tree dict.

    Each round maps to a list of match dicts:
        {'team1', 'team2', 'winner', 'win_prob', 'loser'}

    Seeding / bracket routing:
    - LAST_32[i]      = (seed_i vs seed_{31-i})
    - LAST_16[i]      = winner(L32[i]) vs winner(L32[15-i])   for i in 0..7
    - QF[i]           = winner(L16[i]) vs winner(L16[7-i])    for i in 0..3
    - SF[0]           = winner(QF[0]) vs winner(QF[3])
    - SF[1]           = winner(QF[1]) vs winner(QF[2])
    - FINAL           = winner(SF[0]) vs winner(SF[1])
    - THIRD_PLACE     = loser(SF[0]) vs loser(SF[1])
    """

    def make_match(t1: str, t2: str) -> dict:
        winner, win_prob = _predict_knockout_match(t1, t2, rank_lookup)
        loser = t2 if winner == t1 else t1
        return {"team1": t1, "team2": t2, "winner": winner, "win_prob": win_prob, "loser": loser}

    l32 = [make_match(qualifiers_32[i], qualifiers_32[31 - i]) for i in range(16)]
    l16 = [make_match(l32[i]["winner"], l32[15 - i]["winner"]) for i in range(8)]
    qf  = [make_match(l16[i]["winner"], l16[7 - i]["winner"]) for i in range(4)]
    sf0 = make_match(qf[0]["winner"], qf[3]["winner"])
    sf1 = make_match(qf[1]["winner"], qf[2]["winner"])
    final       = make_match(sf0["winner"], sf1["winner"])
    third_place = make_match(sf0["loser"], sf1["loser"])

    return {
        "LAST_32":        l32,
        "LAST_16":        l16,
        "QUARTER_FINALS": qf,
        "SEMI_FINALS":    [sf0, sf1],
        "FINAL":          [final],
        "THIRD_PLACE":    [third_place],
    }


def _prob_to_color(win_prob: float) -> str:
    """Map win probability to a green shade.

    rgb(144,238,144) (light green) at 50% → rgb(0,100,0) (dark green) at 97%.
    """
    t = (win_prob - 0.50) / 0.47
    r = int(144 * (1 - t))
    g = int(100 + 138 * (1 - t))
    b = int(144 * (1 - t))
    return f"rgb({r},{g},{b})"


def _y_positions(n: int) -> list[float]:
    """Return N evenly-spaced Y coordinates in [Y_MIN, Y_MAX]."""
    return [Y_MIN + (i + 0.5) * (Y_MAX - Y_MIN) / n for i in range(n)]


def _draw_bracket_figure(bracket_tree: dict) -> go.Figure:
    """Create a Plotly figure showing the full knockout bracket.

    Each round is a column of match boxes coloured by win probability
    (light-to-dark green). L-shaped connector lines link matches across rounds.
    The Third Place match is shown as a separate box below the Final column.
    """
    fig = go.Figure()
    fig.update_layout(
        height=700,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        showlegend=False,
        margin=dict(l=5, r=5, t=40, b=5),
        xaxis=dict(range=[0, 1.05], visible=False, fixedrange=True),
        yaxis=dict(range=[-0.05, 1.05], visible=False, fixedrange=True),
        title=dict(
            text="Predicted Tournament Bracket (FIFA Ranking Simulation)",
            font=dict(color="white", size=14),
            x=0.5,
        ),
    )

    # Pre-compute Y positions per round
    y_by_round: dict[str, list[float]] = {
        rnd: _y_positions(len(bracket_tree[rnd])) for rnd in ROUND_ORDER
    }

    # --- Connector lines (single aggregated Scatter trace) ---
    connector_x: list = []
    connector_y: list = []

    # (current_round, next_round, fn: match_index_in_curr → match_index_in_next)
    routing = [
        ("LAST_32",        "LAST_16",        lambda i: min(i, 15 - i)),
        ("LAST_16",        "QUARTER_FINALS", lambda i: min(i, 7 - i)),
        ("QUARTER_FINALS", "SEMI_FINALS",    lambda i: 0 if i in (0, 3) else 1),
        ("SEMI_FINALS",    "FINAL",          lambda i: 0),
    ]

    for curr_rnd, next_rnd, idx_fn in routing:
        x_from = ROUND_X[curr_rnd] + BOX_W
        x_to   = ROUND_X[next_rnd]
        x_mid  = (x_from + x_to) / 2
        for i, y_from in enumerate(y_by_round[curr_rnd]):
            j = idx_fn(i)
            y_to = y_by_round[next_rnd][j]
            connector_x += [x_from, x_mid, x_mid, x_to, None]
            connector_y += [y_from, y_from, y_to, y_to, None]

    fig.add_trace(
        go.Scatter(
            x=connector_x, y=connector_y,
            mode="lines",
            line=dict(color="#555555", width=1),
            hoverinfo="skip",
        )
    )

    # --- Match boxes and annotations for each main round ---
    for rnd in ROUND_ORDER:
        x = ROUND_X[rnd]
        matches = bracket_tree[rnd]
        y_positions = y_by_round[rnd]

        # Round label above the column (paper y-reference so it sits above plot area)
        fig.add_annotation(
            x=x + BOX_W / 2, y=1.01,
            text=ROUND_LABELS[rnd],
            font=dict(size=10, color="#cccccc"), showarrow=False,
            xref="x", yref="paper", align="center",
        )

        for match, y in zip(matches, y_positions):
            fig.add_shape(
                type="rect",
                x0=x, x1=x + BOX_W, y0=y - BOX_H_HALF, y1=y + BOX_H_HALF,
                xref="x", yref="y",
                fillcolor=_prob_to_color(match["win_prob"]),
                line=dict(color="#555555", width=1),
            )
            fig.add_annotation(
                x=x + BOX_W / 2, y=y + BOX_H_HALF * 0.45,
                text=f"<b>{match['winner']}</b> {match['win_prob']:.0%}",
                font=dict(size=8, color="white"), showarrow=False,
                xref="x", yref="y", align="center",
            )
            fig.add_annotation(
                x=x + BOX_W / 2, y=y - BOX_H_HALF * 0.45,
                text=match["loser"],
                font=dict(size=8, color="#aaaaaa"), showarrow=False,
                xref="x", yref="y", align="center",
            )

    # --- Third Place match box (bottom of FINAL column, inside yaxis range) ---
    tp = bracket_tree["THIRD_PLACE"][0]
    x_tp = ROUND_X["FINAL"]
    y_tp = 0.02  # Y_MIN — inside yaxis range

    fig.add_shape(
        type="rect",
        x0=x_tp, x1=x_tp + BOX_W, y0=y_tp - BOX_H_HALF, y1=y_tp + BOX_H_HALF,
        xref="x", yref="y",
        fillcolor=_prob_to_color(tp["win_prob"]),
        line=dict(color="#888888", width=1),
    )
    fig.add_annotation(
        x=x_tp + BOX_W / 2, y=y_tp + BOX_H_HALF * 0.45,
        text=f"<b>{tp['winner']}</b> {tp['win_prob']:.0%}",
        font=dict(size=8, color="white"), showarrow=False,
        xref="x", yref="y", align="center",
    )
    fig.add_annotation(
        x=x_tp + BOX_W / 2, y=y_tp - BOX_H_HALF * 0.45,
        text=tp["loser"],
        font=dict(size=8, color="#aaaaaa"), showarrow=False,
        xref="x", yref="y", align="center",
    )
    fig.add_annotation(
        x=x_tp + BOX_W / 2, y=y_tp + BOX_H_HALF + 0.02,
        text="3rd Place",
        font=dict(size=9, color="#cccccc"), showarrow=False,
        xref="x", yref="y", align="center",
    )

    return fig


def render_bracket(fixtures_df: pd.DataFrame, features_predict_df, ensemble) -> None:
    """Render group standings and predicted tournament bracket.

    Args:
        fixtures_df: DataFrame from wc2026_fixtures_flat.csv with default
            RangeIndex; rows 0–71 are GROUP_STAGE, aligned with features_predict_df.
        features_predict_df: DataFrame from features_predict.parquet (104 rows),
            positionally aligned with fixtures_df. May be None if not yet generated.
        ensemble: WC2026Ensemble instance with predict_proba().
    """
    if features_predict_df is None:
        st.warning(
            "features_predict.parquet not found. "
            "Run `python scripts/run_feature_engineering.py` first."
        )
        return

    rank_lookup = _build_rank_lookup()
    groups = _identify_groups(fixtures_df)
    group_standings = _simulate_group_stage(
        fixtures_df, features_predict_df, ensemble, groups, rank_lookup
    )
    qualifiers_32 = _select_qualifiers(group_standings, rank_lookup)
    bracket_tree = _build_bracket_tree(qualifiers_32, rank_lookup)

    # --- Group Standings ---
    st.subheader("Group Stage — Predicted Standings")
    st.caption(
        "Expected points computed from ML model probabilities. "
        "Top 2 per group + best 8 third-placed teams advance."
    )

    labels = sorted(group_standings.keys())
    for row_start in range(0, 12, 4):
        cols = st.columns(4)
        for col_i, label in enumerate(labels[row_start:row_start + 4]):
            df = group_standings[label]
            display = df[["team", "expected_pts", "rank"]].copy()
            display.columns = ["Team", "Exp. Pts", "FIFA Rank"]
            display["Exp. Pts"] = display["Exp. Pts"].round(1)
            with cols[col_i]:
                st.markdown(f"**Group {label}**")
                st.dataframe(display, hide_index=True, use_container_width=True)

    st.divider()

    # --- Knockout Bracket ---
    st.subheader("Knockout Stage — Predicted Bracket (Simulated)")
    fig = _draw_bracket_figure(bracket_tree)
    st.plotly_chart(fig, use_container_width=True)

    tp = bracket_tree["THIRD_PLACE"][0]
    st.caption(
        f"3rd Place Match: {tp['team1']} vs {tp['team2']} — "
        f"predicted winner: **{tp['winner']}** ({tp['win_prob']:.0%})"
    )
