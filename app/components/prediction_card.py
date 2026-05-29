"""Prediction card UI component."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.tooltips import TOOLTIPS

FEATURE_COLUMNS = [
    # --- Rankings ---
    "home_rank",
    "away_rank",
    "home_rank_points",
    "away_rank_points",
    "rank_diff",
    "rank_points_ratio",
    "rank_points_diff",
    # --- Form 5-match window ---
    "home_form_wins_5",
    "home_form_goals_scored_5",
    "home_form_goals_conceded_5",
    "home_form_wdl_points_5",
    "away_form_wins_5",
    "away_form_goals_scored_5",
    "away_form_goals_conceded_5",
    "away_form_wdl_points_5",
    # --- Form 10-match window ---
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
    # --- Head-to-Head ---
    "h2h_home_win_rate",
    "h2h_matches_count",
    "h2h_avg_goals_home",
    "h2h_avg_goals_away",
    # --- Context & Venue ---
    "tournament_stage",
    "is_wc_match",
    "is_neutral_venue",
    "host_nation_advantage",
    "home_days_rest",
    "away_days_rest",
    "rest_diff",
]

_COLOR_HOME_WIN = "#00CC66"
_COLOR_DRAW = "#FFAA00"
_COLOR_AWAY_WIN = "#CC3333"

_BEST_VALUE_THRESHOLD = 0.05
_AVOID_THRESHOLD = -0.05
_CONF_HIGH = 0.55   # >=55% -> High confidence
_CONF_MED  = 0.45   # 45-54% -> Medium confidence; <45% -> Low confidence


def _edge_label_html(edge: float, is_best: bool, show_labels: bool, outcome_prob: float | None = None) -> str:
    edge_raw = f"Edge: {edge:+.1%}"
    if outcome_prob is not None:
        edge_raw += f"  ·  Model: {outcome_prob:.0%}"
    edge_tip = TOOLTIPS["edge"]
    edge_text = f'<span title="{edge_tip}">{edge_raw}</span>'
    if not show_labels:
        return edge_text
    badge_style = "padding:2px 8px;border-radius:4px;font-size:0.75em;font-weight:bold;"
    value_tip = TOOLTIPS["value_bet"]
    if is_best:
        badge = f"<span style='background:#1a7a3a;color:white;{badge_style}' title=\"{value_tip}\">\u2705 Best Value</span>"
    elif edge < _AVOID_THRESHOLD:
        badge = f"<span style='background:#7a1a1a;color:white;{badge_style}' title=\"{value_tip}\">\u274c Avoid</span>"
    else:
        badge = f"<span style='background:#555555;color:white;{badge_style}' title=\"{value_tip}\">&mdash; Neutral</span>"
    return f"{edge_text}&nbsp;&nbsp;{badge}"


def _confidence_tier_html(confidence: float) -> str:
    """Return a styled HTML badge for the confidence tier."""
    badge_style = "padding:3px 10px;border-radius:4px;font-size:0.8em;font-weight:bold;"
    pct = f"{confidence:.0%}"
    conf_tip = TOOLTIPS["confidence"]
    if confidence >= _CONF_HIGH:
        return f"<span style='background:#1a7a3a;color:white;{badge_style}' title=\"{conf_tip}\">High Confidence · {pct}</span>"
    elif confidence >= _CONF_MED:
        return f"<span style='background:#7a6a00;color:white;{badge_style}' title=\"{conf_tip}\">Medium Confidence · {pct}</span>"
    else:
        return f"<span style='background:#5a2a2a;color:white;{badge_style}' title=\"{conf_tip}\">Low Confidence · {pct}</span>"


def _lookup_odds(
    odds_df: pd.DataFrame | None,
    match_date,
    home_team: str,
    away_team: str,
) -> dict | None:
    """Return the latest real odds row for a fixture, or None if unavailable."""
    if odds_df is None or odds_df.empty:
        return None
    try:
        target_date = pd.to_datetime(match_date).normalize()
    except Exception:
        return None

    mask = (
        (pd.to_datetime(odds_df["match_date"], errors="coerce").dt.normalize() == target_date)
        & (odds_df["home_team"] == home_team)
        & (odds_df["away_team"] == away_team)
    )
    hits = odds_df[mask]
    if hits.empty:
        return None

    # Prefer the most recently fetched row
    if "fetched_at" in hits.columns:
        hits = hits.sort_values("fetched_at", ascending=False)
    row = hits.iloc[0]
    return {
        "home_win_odds": float(row.get("home_win_odds", 2.0)),
        "draw_odds": float(row.get("draw_odds", 3.0)),
        "away_win_odds": float(row.get("away_win_odds", 2.5)),
        "source": str(row.get("source", "unknown")),
        "fetched_at": str(row.get("fetched_at", "")),
    }


def render_prediction_card(
    fixture_row,
    features_row,
    ensemble,
    home_goals_model,
    away_goals_model,
    odds_df: pd.DataFrame | None = None,
):
    """Render a prediction card for a single WC 2026 fixture.

    Parameters
    ----------
    fixture_row : pd.Series
        Row from wc2026_fixtures_flat.csv with columns: match_date, home_team,
        away_team, stage, group, venue.
    features_row : pd.Series or None
        Row from features_predict.parquet containing ML feature columns.
        Pass None when no prediction data is available for the fixture.
    ensemble : WC2026Ensemble
        Ensemble model with .predict_proba(X) returning shape (n, 3) in order
        [Away Win=0, Draw=1, Home Win=2].
    home_goals_model : sklearn-compatible regressor
        Predicts expected home goals.
    away_goals_model : sklearn-compatible regressor
        Predicts expected away goals.
    odds_df : pd.DataFrame or None
        Canonical odds table.  When provided, the fixture's latest real odds
        are used as default values for the inputs; users can still override.
    """
    home_team = fixture_row["home_team"]
    away_team = fixture_row["away_team"]
    stage = fixture_row.get("stage", "")
    match_date = fixture_row.get("match_date", "")

    with st.container():
        st.subheader(f"{home_team}  vs  {away_team}")
        st.caption(f"{stage} — {match_date}")

        if features_row is None:
            st.info("No prediction data available for this fixture.")
            st.divider()
            return

        # Build 2D input array for models
        X = pd.DataFrame(
            features_row[FEATURE_COLUMNS].values.reshape(1, -1),
            columns=FEATURE_COLUMNS,
        )

        # --- Outcome probabilities ---
        # predict_proba returns shape (1, 3): [Away Win=0, Draw=1, Home Win=2]
        proba = ensemble.predict_proba(X)[0]
        prob_away_win = float(proba[0])
        prob_draw = float(proba[1])
        prob_home_win = float(proba[2])

        # --- W/D/L horizontal stacked bar chart ---
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[prob_home_win * 100],
                y=[""],
                orientation="h",
                name="Home Win",
                marker_color=_COLOR_HOME_WIN,
                text=[f"{prob_home_win:.0%}"],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate="Home Win: %{x:.1f}%<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=[prob_draw * 100],
                y=[""],
                orientation="h",
                name="Draw",
                marker_color=_COLOR_DRAW,
                text=[f"{prob_draw:.0%}"],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate="Draw: %{x:.1f}%<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=[prob_away_win * 100],
                y=[""],
                orientation="h",
                name="Away Win",
                marker_color=_COLOR_AWAY_WIN,
                text=[f"{prob_away_win:.0%}"],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate="Away Win: %{x:.1f}%<extra></extra>",
            )
        )
        fig.update_layout(
            barmode="stack",
            showlegend=False,
            width=500,
            height=100,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(range=[0, 100], showticklabels=False),
            yaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig, width='stretch', key=f"{fixture_row.name}_chart")

        # --- Legend labels below the chart ---
        leg_col1, leg_col2, leg_col3 = st.columns(3)
        leg_col1.markdown(
            f"<span style='color:{_COLOR_HOME_WIN}'>&#9632;</span> Home Win",
            unsafe_allow_html=True,
        )
        leg_col2.markdown(
            f"<span style='color:{_COLOR_DRAW}'>&#9632;</span> Draw",
            unsafe_allow_html=True,
        )
        leg_col3.markdown(
            f"<span style='color:{_COLOR_AWAY_WIN}'>&#9632;</span> Away Win",
            unsafe_allow_html=True,
        )

        # --- Predicted scoreline ---
        h_goals = max(0, int(float(home_goals_model.predict(X)[0])))
        a_goals = max(0, int(float(away_goals_model.predict(X)[0])))
        st.markdown(
            f"**Predicted:** {home_team} {h_goals} – {a_goals} {away_team}"
        )
        st.caption(TOOLTIPS["scoreline"])

        # --- Confidence score ---
        probs_array = np.array([prob_away_win, prob_draw, prob_home_win], dtype=float)
        confidence = float(np.max(probs_array))
        st.markdown(_confidence_tier_html(confidence), unsafe_allow_html=True)

        # --- Bookmaker odds inputs ---
        # Look up real odds from the canonical table; fall back to neutral defaults
        real_odds = _lookup_odds(odds_df, match_date, home_team, away_team)
        default_home_odds = real_odds["home_win_odds"] if real_odds else 2.0
        default_draw_odds = real_odds["draw_odds"] if real_odds else 3.0
        default_away_odds = real_odds["away_win_odds"] if real_odds else 2.5

        # Show source badge when real odds are available
        if real_odds:
            src_label = real_odds.get("source", "")
            fetched = real_odds.get("fetched_at", "")[:10]
            st.caption(f"Odds: {src_label} (fetched {fetched})")
        else:
            st.caption("Odds: no real odds found — using neutral defaults (edit below)")

        key_prefix = f"{fixture_row.name}_{home_team}_{away_team}"
        odds_col1, odds_col2, odds_col3 = st.columns(3)
        home_odds = odds_col1.number_input(
            "Home Win Odds",
            min_value=1.01,
            value=float(default_home_odds),
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_home_odds",
            help=TOOLTIPS["home_odds"],
        )
        draw_odds = odds_col2.number_input(
            "Draw Odds",
            min_value=1.01,
            value=float(default_draw_odds),
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_draw_odds",
            help=TOOLTIPS["draw_odds"],
        )
        away_odds = odds_col3.number_input(
            "Away Win Odds",
            min_value=1.01,
            value=float(default_away_odds),
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_away_odds",
            help=TOOLTIPS["away_odds"],
        )

        # --- Edge values ---
        home_edge = prob_home_win - (1.0 / home_odds)
        draw_edge = prob_draw - (1.0 / draw_odds)
        away_edge = prob_away_win - (1.0 / away_odds)

        edges = [home_edge, draw_edge, away_edge]
        best_val = max(edges)
        show_labels = best_val > _BEST_VALUE_THRESHOLD

        edge_col1, edge_col2, edge_col3 = st.columns(3)
        edge_col1.markdown(
            _edge_label_html(home_edge, home_edge == best_val and show_labels, show_labels, outcome_prob=prob_home_win),
            unsafe_allow_html=True,
        )
        edge_col2.markdown(
            _edge_label_html(draw_edge, draw_edge == best_val and show_labels, show_labels, outcome_prob=prob_draw),
            unsafe_allow_html=True,
        )
        edge_col3.markdown(
            _edge_label_html(away_edge, away_edge == best_val and show_labels, show_labels, outcome_prob=prob_away_win),
            unsafe_allow_html=True,
        )

        # Summary signal row — only shown when there is a positive-edge bet
        if show_labels:
            outcome_names = [f"{home_team} Win", "Draw", f"{away_team} Win"]
            outcome_probs = [prob_home_win, prob_draw, prob_away_win]
            all_edges = [home_edge, draw_edge, away_edge]
            best_idx = all_edges.index(max(all_edges))
            best_outcome_name = outcome_names[best_idx]
            best_outcome_prob = outcome_probs[best_idx]
            best_edge_val = all_edges[best_idx]

            is_high_conf_value = confidence >= _CONF_HIGH and best_outcome_prob == confidence

            if is_high_conf_value:
                summary_bg = "#0d4f2e"
                summary_label = f"⭐ High-Confidence Value Bet"
            else:
                summary_bg = "#1a2a4a"
                summary_label = f"✅ Best Value Bet"

            value_tip = TOOLTIPS["value_bet"]
            summary_html = (
                f"<div style='background:{summary_bg};color:white;padding:8px 14px;"
                f"border-radius:6px;margin-top:8px;font-size:0.85em;font-weight:bold;' title=\"{value_tip}\">"
                f"{summary_label}: {best_outcome_name} &nbsp;|&nbsp; "
                f"Edge: {best_edge_val:+.1%} &nbsp;|&nbsp; "
                f"Model: {best_outcome_prob:.0%}"
                f"</div>"
            )
            st.markdown(summary_html, unsafe_allow_html=True)

        st.divider()

