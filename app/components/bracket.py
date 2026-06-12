"""Tournament bracket UI component for FIFA WC 2026 Predictor."""

from __future__ import annotations

from collections import defaultdict
from math import exp
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.flags import flag_url as _flag_url
from components.tooltips import TOOLTIPS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RAW = _REPO_ROOT / "data" / "raw"

# Bracket figure constants
BOX_W = 0.195
BOX_H_HALF = 0.028
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


_FLAG_SZ_X = 0.021   # flag image width in x-axis units
_FLAG_SZ_Y = 0.022   # flag image height in y-axis units
_FLAG_OFF  = 0.005   # gap from box left edge to flag left edge
_TEXT_OFF  = 0.030   # gap from box left edge to text start


def _add_team_row(
    fig: go.Figure,
    x_box: float,
    y_row: float,
    team: str,
    text: str,
    font_size: int,
    font_color: str,
    bold: bool = False,
) -> None:
    """Add a flag image + team name annotation as one row inside a match box."""
    url = _flag_url(team)
    if url:
        fig.add_layout_image(
            source=url,
            x=x_box + _FLAG_OFF,
            y=y_row,
            xref="x", yref="y",
            sizex=_FLAG_SZ_X,
            sizey=_FLAG_SZ_Y,
            xanchor="left",
            yanchor="middle",
            layer="above",
        )
    label = f"<b>{text}</b>" if bold else text
    fig.add_annotation(
        x=x_box + _TEXT_OFF,
        y=y_row,
        text=label,
        font=dict(size=font_size, color=font_color),
        showarrow=False,
        xref="x", yref="y",
        align="left",
        xanchor="left",
    )


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
    predictions_df: pd.DataFrame,
    groups: list[list[str]],
    rank_lookup: dict[str, int],
) -> dict[str, pd.DataFrame]:
    """Compute expected points per team from precomputed prediction probabilities.

    For each GROUP_STAGE fixture reads prob_home_win, prob_draw, prob_away_win from
    predictions_df (loaded with index_col=0, so labels match the original CSV row index).
    Expected points: home += 3*p_hw + 1*p_draw; away += 3*p_aw + 1*p_draw.

    Returns a dict mapping group label (A–L) to a sorted DataFrame with columns
    ['team', 'expected_pts', 'rank'], sorted by expected_pts DESC then rank ASC.
    """
    labels = [chr(ord("A") + i) for i in range(12)]  # A through L
    expected_pts: dict[str, float] = defaultdict(float)

    group_stage = fixtures_df[fixtures_df["stage"] == "GROUP_STAGE"]
    for _, row in group_stage.iterrows():
        row_idx = row.name  # original CSV positional index
        pred = predictions_df.loc[row_idx]
        expected_pts[row["home_team"]] += 3 * float(pred["prob_home_win"]) + 1 * float(pred["prob_draw"])
        expected_pts[row["away_team"]] += 3 * float(pred["prob_away_win"]) + 1 * float(pred["prob_draw"])

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
    """Map win probability to a dark green background for match boxes.

    Dark forest (#0d2b14) at 50% → rich green (#0a5c1e) at 97%.
    Dark backgrounds ensure strong contrast with white/light text.
    """
    t = (win_prob - 0.50) / 0.47
    r = int(13 * (1 - t))
    g = int(43 + 49 * t)
    b = int(20 * (1 - t))
    return f"rgb({r},{g},{b})"


def _prob_to_border_color(win_prob: float) -> str:
    """Bright accent border: muted green (#39a84a) at 50% → vivid green (#00e642) at 97%."""
    t = (win_prob - 0.50) / 0.47
    r = int(57 * (1 - t))
    g = int(168 + 62 * t)
    b = int(74 * (1 - t))
    return f"rgb({r},{g},{b})"


def _loser_fill_color(win_prob: float) -> str:
    """Darker muted fill for the loser band of a match box."""
    t = (win_prob - 0.50) / 0.47
    r = int(8 * (1 - t))
    g = int(20 + 20 * t)
    b = int(10 * (1 - t))
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
        height=900,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=5, r=5, t=50, b=5),
        xaxis=dict(range=[0, 1.05], visible=False, fixedrange=True),
        yaxis=dict(range=[-0.05, 1.05], visible=False, fixedrange=True),
        title=dict(
            text="🏆 Predicted Tournament Bracket (FIFA Ranking Simulation)",
            font=dict(color="#f0f0f0", size=17),
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
            line=dict(color="#4a5a6a", width=2),
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
            text=f"<b>{ROUND_LABELS[rnd]}</b>",
            font=dict(size=13, color="#e8e8e8"), showarrow=False,
            xref="x", yref="paper", align="center",
        )
        fig.add_shape(
            type="line",
            x0=x, x1=x + BOX_W, y0=0.985, y1=0.985,
            xref="x", yref="paper",
            line=dict(color="#4a4a6a", width=1),
        )

        for match, y in zip(matches, y_positions):
            y_top = y + BOX_H_HALF * 0.42
            y_bot = y - BOX_H_HALF * 0.42
            # Winner band (top half) — scatter trace so layout images render above it
            fig.add_trace(go.Scatter(
                x=[x, x + BOX_W, x + BOX_W, x, x],
                y=[y, y, y + BOX_H_HALF, y + BOX_H_HALF, y],
                fill="toself",
                fillcolor=_prob_to_color(match["win_prob"]),
                line=dict(color="rgba(0,0,0,0)", width=0),
                mode="lines",
                hoverinfo="text",
                text=f"{match['winner']} wins<br>Win prob: {match['win_prob']:.0%}",
                showlegend=False,
            ))
            # Loser band (bottom half)
            fig.add_trace(go.Scatter(
                x=[x, x + BOX_W, x + BOX_W, x, x],
                y=[y - BOX_H_HALF, y - BOX_H_HALF, y, y, y - BOX_H_HALF],
                fill="toself",
                fillcolor=_loser_fill_color(match["win_prob"]),
                line=dict(color="rgba(0,0,0,0)", width=0),
                mode="lines",
                hoverinfo="skip",
                showlegend=False,
            ))
            # Border overlay (full box, transparent fill)
            fig.add_trace(go.Scatter(
                x=[x, x + BOX_W, x + BOX_W, x, x],
                y=[y - BOX_H_HALF, y - BOX_H_HALF, y + BOX_H_HALF, y + BOX_H_HALF, y - BOX_H_HALF],
                fill="toself",
                fillcolor="rgba(0,0,0,0)",
                line=dict(color=_prob_to_border_color(match["win_prob"]), width=2),
                mode="lines",
                hoverinfo="skip",
                showlegend=False,
            ))
            _add_team_row(
                fig, x, y_top, match["winner"],
                match["winner"], font_size=11, font_color="#ffffff", bold=True,
            )
            fig.add_annotation(
                x=x + BOX_W - 0.004,
                y=y_top,
                text=f"<b>{match['win_prob']:.0%}</b>",
                font=dict(size=9, color="#00e642"),
                showarrow=False,
                xref="x", yref="y",
                align="right",
                xanchor="right",
                bgcolor="rgba(0,30,10,0.85)",
                bordercolor="#00e642",
                borderwidth=1,
                borderpad=2,
            )
            _add_team_row(
                fig, x, y_bot, match["loser"],
                match["loser"],
                font_size=10, font_color="#9a9aaa",
            )

    # --- Third Place match box (bottom of FINAL column, inside yaxis range) ---
    tp = bracket_tree["THIRD_PLACE"][0]
    x_tp = ROUND_X["FINAL"]
    y_tp = 0.02  # Y_MIN — inside yaxis range

    fig.add_trace(go.Scatter(
        x=[x_tp, x_tp + BOX_W, x_tp + BOX_W, x_tp, x_tp],
        y=[y_tp, y_tp, y_tp + BOX_H_HALF, y_tp + BOX_H_HALF, y_tp],
        fill="toself",
        fillcolor=_prob_to_color(tp["win_prob"]),
        line=dict(color="rgba(0,0,0,0)", width=0),
        mode="lines",
        hoverinfo="text",
        text=f"{tp['winner']} wins<br>Win prob: {tp['win_prob']:.0%}",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[x_tp, x_tp + BOX_W, x_tp + BOX_W, x_tp, x_tp],
        y=[y_tp - BOX_H_HALF, y_tp - BOX_H_HALF, y_tp, y_tp, y_tp - BOX_H_HALF],
        fill="toself",
        fillcolor=_loser_fill_color(tp["win_prob"]),
        line=dict(color="rgba(0,0,0,0)", width=0),
        mode="lines",
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[x_tp, x_tp + BOX_W, x_tp + BOX_W, x_tp, x_tp],
        y=[y_tp - BOX_H_HALF, y_tp - BOX_H_HALF, y_tp + BOX_H_HALF, y_tp + BOX_H_HALF, y_tp - BOX_H_HALF],
        fill="toself",
        fillcolor="rgba(0,0,0,0)",
        line=dict(color=_prob_to_border_color(tp["win_prob"]), width=2),
        mode="lines",
        hoverinfo="skip",
        showlegend=False,
    ))
    _add_team_row(
        fig, x_tp, y_tp + BOX_H_HALF * 0.42, tp["winner"],
        tp["winner"], font_size=11, font_color="#ffffff", bold=True,
    )
    fig.add_annotation(
        x=x_tp + BOX_W - 0.004,
        y=y_tp + BOX_H_HALF * 0.42,
        text=f"<b>{tp['win_prob']:.0%}</b>",
        font=dict(size=9, color="#00e642"),
        showarrow=False,
        xref="x", yref="y",
        align="right",
        xanchor="right",
        bgcolor="rgba(0,30,10,0.85)",
        bordercolor="#00e642",
        borderwidth=1,
        borderpad=2,
    )
    _add_team_row(
        fig, x_tp, y_tp - BOX_H_HALF * 0.42, tp["loser"],
        tp["loser"],
        font_size=10, font_color="#9a9aaa",
    )
    fig.add_annotation(
        x=x_tp + BOX_W / 2, y=y_tp + BOX_H_HALF + 0.025,
        text="<b>🥉 3rd Place</b>",
        font=dict(size=11, color="#e8c84a"), showarrow=False,
        xref="x", yref="y", align="center",
    )

    return fig


def render_bracket(fixtures_df: pd.DataFrame, predictions_df) -> None:
    """Render group standings and predicted tournament bracket.

    Args:
        fixtures_df: DataFrame from wc2026_fixtures_flat.csv with default
            RangeIndex; rows 0–71 are GROUP_STAGE, aligned with predictions_df.
        predictions_df: DataFrame from wc2026_predictions.csv (loaded with
            index_col=0), containing prob_home_win, prob_draw, prob_away_win
            columns. May be None if not yet generated.
    """
    if predictions_df is None:
        st.warning(
            "wc2026_predictions.csv not found. "
            "Run: python scripts/precompute_predictions.py"
        )
        return

    rank_lookup = _build_rank_lookup()
    groups = _identify_groups(fixtures_df)
    group_standings = _simulate_group_stage(
        fixtures_df, predictions_df, groups, rank_lookup
    )
    qualifiers_32 = _select_qualifiers(group_standings, rank_lookup)
    bracket_tree = _build_bracket_tree(qualifiers_32, rank_lookup)

    # --- Group Standings ---
    st.subheader("Group Stage — Predicted Standings")
    st.caption(
        "Points based on predicted match outcomes (3W / 1D / 0L). "
        "Top 2 per group + best 8 third-placed teams advance."
    )

    # Inject CSS once (outside the loop)
    st.markdown("""
<style>
.group-card {
    background: linear-gradient(180deg, #151E33 0%, #111829 100%);
    border: 1px solid #26324B;
    border-radius: 14px;
    padding: 10px 12px;
    margin-bottom: 10px;
}
.group-header {
    color: #FACC15;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 6px;
    padding-bottom: 5px;
    border-bottom: 1px solid #26324B;
}
.team-row {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 3px 4px;
    border-radius: 4px;
    margin-bottom: 2px;
    font-size: 0.82rem;
}
.team-row .pos {
    color: #6a7a9a;
    width: 14px;
    flex-shrink: 0;
    font-size: 0.75rem;
}
.team-row img {
    vertical-align: middle;
    flex-shrink: 0;
}
.team-row .flag-fallback {
    font-size: 0.9rem;
    flex-shrink: 0;
}
.team-row .name {
    flex: 1;
    color: #d8e0f0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.team-row .pts {
    color: #a0b0c0;
    font-size: 0.78rem;
    width: 36px;
    text-align: right;
    flex-shrink: 0;
}
.team-row .rank {
    color: #5a6a8a;
    font-size: 0.72rem;
    width: 34px;
    text-align: right;
    flex-shrink: 0;
}
.team-row.qualifier {
    background: rgba(0, 200, 100, 0.10);
}
.team-row.maybe {
    background: rgba(200, 200, 0, 0.06);
}
.team-row.eliminated {
    opacity: 0.45;
}
</style>
""", unsafe_allow_html=True)

    labels = sorted(group_standings.keys())
    _row_classes = ["qualifier", "qualifier", "maybe", "eliminated"]
    for row_start in range(0, 12, 4):
        cols = st.columns(4)
        for col_i, label in enumerate(labels[row_start:row_start + 4]):
            df = group_standings[label]
            rows_html = ""
            for i, row in df.iterrows():
                team = row["team"]
                pts = int(round(float(row["expected_pts"])))
                rank = int(row["rank"])
                url = _flag_url(team)
                flag_html = (
                    f'<img src="{url}" width="20" height="15" alt="">'
                    if url else
                    '<span class="flag-fallback">🏴</span>'
                )
                css_class = _row_classes[i] if i < 4 else "eliminated"
                rows_html += (
                    f'<div class="team-row {css_class}">'
                    f'<span class="pos">{i + 1}</span>'
                    f'{flag_html}'
                    f'<span class="name">{team}</span>'
                    f'<span class="pts">{pts}</span>'
                    f'<span class="rank">#{rank}</span>'
                    f'</div>'
                )
            col_header = (
                '<div style="display:flex;align-items:center;gap:6px;'
                'padding:0 4px 4px 4px;margin-bottom:2px;'
                'border-bottom:1px solid #2a2f3a;">'
                '<span style="width:14px;flex-shrink:0;"></span>'
                '<span style="width:20px;flex-shrink:0;"></span>'
                '<span style="flex:1;color:#6a7a9a;font-size:0.72rem;">Team</span>'
                '<span style="width:36px;flex-shrink:0;text-align:right;'
                'color:#6a7a9a;font-size:0.72rem;">xPts</span>'
                '<span style="width:34px;flex-shrink:0;text-align:right;'
                'color:#6a7a9a;font-size:0.72rem;">Rank</span>'
                '</div>'
            )
            card_html = (
                f'<div class="group-card">'
                f'<div class="group-header">Group {label}</div>'
                f'{col_header}'
                f'{rows_html}'
                f'</div>'
            )
            with cols[col_i]:
                st.markdown(card_html, unsafe_allow_html=True)

    st.caption(TOOLTIPS["expected_pts"])
    st.divider()

    # --- Knockout Bracket ---
    st.subheader("Knockout Stage — Predicted Bracket (Simulated)")
    fig = _draw_bracket_figure(bracket_tree)
    st.caption(
        "Group stage: teams are ranked by expected points (ML ensemble). "
        "Knockout rounds: winner predicted by FIFA ranking-based logistic model. "
        + TOOLTIPS["knockout_win_prob"]
    )
    st.plotly_chart(fig, width="stretch")

    tp = bracket_tree["THIRD_PLACE"][0]
    st.caption(
        f"3rd Place Match: {tp['team1']} vs {tp['team2']} — "
        f"predicted winner: **{tp['winner']}** ({tp['win_prob']:.0%})"
    )
