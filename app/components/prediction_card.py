"""Prediction card UI component."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import entropy

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

        # --- Confidence score ---
        probs_array = np.array([prob_away_win, prob_draw, prob_home_win], dtype=float)
        # Clamp tiny negatives from floating-point to avoid log(0)
        probs_array = np.clip(probs_array, 1e-9, 1.0)
        raw_entropy = entropy(probs_array)  # natural log (base=None default)
        confidence = float(np.clip(1.0 - raw_entropy / np.log(3), 0.0, 1.0))
        st.markdown(f"**Confidence:** {confidence:.0%}")

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
        )
        draw_odds = odds_col2.number_input(
            "Draw Odds",
            min_value=1.01,
            value=float(default_draw_odds),
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_draw_odds",
        )
        away_odds = odds_col3.number_input(
            "Away Win Odds",
            min_value=1.01,
            value=float(default_away_odds),
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_away_odds",
        )

        # --- Edge values ---
        home_edge = prob_home_win - (1.0 / home_odds)
        draw_edge = prob_draw - (1.0 / draw_odds)
        away_edge = prob_away_win - (1.0 / away_odds)

        edge_col1, edge_col2, edge_col3 = st.columns(3)
        edge_col1.markdown(f"Edge: {home_edge:+.1%}")
        edge_col2.markdown(f"Edge: {draw_edge:+.1%}")
        edge_col3.markdown(f"Edge: {away_edge:+.1%}")

        # --- Recommendation badge ---
        best_edge = max(home_edge, draw_edge, away_edge)
        if best_edge > 0.05:
            badge_html = (
                "<span style='background-color:#1a7a3a;color:white;"
                "padding:4px 10px;border-radius:4px;font-weight:bold;'>"
                "&#9989; Value</span>"
            )
        elif best_edge < -0.05:
            badge_html = (
                "<span style='background-color:#7a1a1a;color:white;"
                "padding:4px 10px;border-radius:4px;font-weight:bold;'>"
                "&#10060; Avoid</span>"
            )
        else:
            badge_html = (
                "<span style='background-color:#555555;color:white;"
                "padding:4px 10px;border-radius:4px;font-weight:bold;'>"
                "&#11036; Neutral</span>"
            )
        st.markdown(badge_html, unsafe_allow_html=True)

        st.divider()

