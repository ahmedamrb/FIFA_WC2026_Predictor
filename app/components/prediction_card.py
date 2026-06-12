"""Prediction card UI component."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from components.tooltips import TOOLTIPS


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


def _decided(value) -> bool:
    """True when a nullable-boolean flag has a concrete (non-NA) value."""
    return value is not None and not pd.isna(value)


def _status_badge_html(status, minute) -> str:
    """Return a styled HTML status badge (LIVE / HT / FT), or '' for upcoming."""
    badge_style = "padding:2px 8px;border-radius:4px;font-size:0.75em;font-weight:bold;"
    s = str(status or "").upper()
    tip = TOOLTIPS["match_status"]
    if s == "IN_PLAY":
        mins = f" {int(minute)}'" if minute is not None and not pd.isna(minute) else ""
        return f"<span style='background:#b3261e;color:white;{badge_style}' title=\"{tip}\">🔴 LIVE{mins}</span>"
    if s == "PAUSED":
        return f"<span style='background:#7a6a00;color:white;{badge_style}' title=\"{tip}\">HT</span>"
    if s in ("FINISHED", "AWARDED"):
        return f"<span style='background:#333333;color:#dddddd;{badge_style}' title=\"{tip}\">FT</span>"
    if s in ("", "SCHEDULED", "TIMED"):
        return ""  # upcoming — kickoff line already shows the time
    return f"<span style='background:#555555;color:white;{badge_style}' title=\"{tip}\">{s.title()}</span>"


def _result_badges_html(result_row: dict) -> str:
    """Return verdict badges (outcome correct/miss, exact score) for a played match."""
    badge_style = "padding:3px 10px;border-radius:4px;font-size:0.8em;font-weight:bold;"
    outcome_correct = result_row.get("outcome_correct")
    exact_correct = result_row.get("exact_score_correct")
    verdict_tip = TOOLTIPS["outcome_verdict"]
    exact_tip = TOOLTIPS["exact_score"]

    parts: list[str] = []
    if _decided(outcome_correct):
        if bool(outcome_correct):
            parts.append(
                f"<span style='background:#1a7a3a;color:white;{badge_style}' title=\"{verdict_tip}\">✅ Outcome correct</span>"
            )
        else:
            parts.append(
                f"<span style='background:#7a1a1a;color:white;{badge_style}' title=\"{verdict_tip}\">❌ Outcome miss</span>"
            )
    if _decided(exact_correct) and bool(exact_correct):
        parts.append(
            f"<span style='background:#b8860b;color:white;{badge_style}' title=\"{exact_tip}\">⭐ Exact score</span>"
        )
    return "&nbsp;&nbsp;".join(parts)


def _goals_compare_html(result_row: dict, home_team: str, away_team: str) -> str:
    """Per-side predicted-vs-actual goals (home and away models are independent)."""
    # Only meaningful for comparable matches (per-side flags are decided).
    if not (_decided(result_row.get("home_goals_correct")) or _decided(result_row.get("away_goals_correct"))):
        return ""
    h_pred, a_pred = result_row.get("predicted_home_goals"), result_row.get("predicted_away_goals")
    h_act, a_act = result_row.get("home_score"), result_row.get("away_score")
    if not (_decided(h_pred) and _decided(a_pred) and _decided(h_act) and _decided(a_act)):
        return ""

    def _side(pred, actual, correct) -> str:
        hit = _decided(correct) and bool(correct)
        glyph = "✓" if hit else "✗"
        color = _COLOR_HOME_WIN if hit else _COLOR_AWAY_WIN
        return (f"pred {int(pred)} &rarr; actual {int(actual)} "
                f"<span style='color:{color};font-weight:bold;'>{glyph}</span>")

    tip = TOOLTIPS["goals_compare"]
    return (
        f"<div style='font-size:0.85em;line-height:1.5;' title=\"{tip}\">"
        f"<b>Home goals</b> ({home_team}): {_side(h_pred, h_act, result_row.get('home_goals_correct'))}"
        f"<br><b>Away goals</b> ({away_team}): {_side(a_pred, a_act, result_row.get('away_goals_correct'))}"
        f"</div>"
    )


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
    prediction_row,
    odds_df: pd.DataFrame | None = None,
    result_row: dict | None = None,
):
    """Render a prediction card for a single WC 2026 fixture.

    Parameters
    ----------
    fixture_row : pd.Series
        Row from wc2026_fixtures_flat.csv with columns: match_date, home_team,
        away_team, stage, group, venue.
    prediction_row : pd.Series or None
        Row from predictions parquet/CSV containing pre-computed columns:
        prob_home_win, prob_draw, prob_away_win, predicted_home_goals,
        predicted_away_goals, confidence.  Pass None when unavailable.
    odds_df : pd.DataFrame or None
        Canonical odds table.  When provided, the fixture's latest real odds
        are used as default values for the inputs; users can still override.
    result_row : dict or None
        Live/full-time comparison row for this fixture (from
        src.evaluation.live_tracking.build_comparison): status, home_score,
        away_score, minute, actual_outcome, outcome_correct,
        exact_score_correct.  When None or score-less, no actual score is shown.
    """
    home_team = fixture_row["home_team"]
    away_team = fixture_row["away_team"]
    stage = fixture_row.get("stage", "")
    match_date = fixture_row.get("match_date", "")
    kickoff_utc = str(fixture_row.get("kickoff_utc", ""))

    with st.container():
        status_badge = ""
        if result_row is not None:
            status_badge = _status_badge_html(result_row.get("status"), result_row.get("minute"))
        title = f"{home_team}  vs  {away_team}"
        if status_badge:
            st.markdown(
                f"<h3 style='margin-bottom:0'>{title}&nbsp;&nbsp;{status_badge}</h3>",
                unsafe_allow_html=True,
            )
        else:
            st.subheader(title)
        if kickoff_utc and kickoff_utc not in ("", "nan", "NaT"):
            components.html(
                f"""<!DOCTYPE html><html><head><style>
                html,body{{margin:0;padding:0;overflow:hidden;font-family:"Source Sans Pro","Noto Sans",sans-serif;}}
                div{{font-size:14px;color:rgba(49,51,63,0.6);line-height:1;}}
                @media(prefers-color-scheme:dark){{div{{color:rgba(250,250,250,0.6);}}}}
                </style></head><body>
                <div>{stage} &mdash; <span id="kt"></span></div>
                <script>
                var d=new Date("{kickoff_utc}");
                document.getElementById("kt").textContent=d.toLocaleString(undefined,
                {{weekday:"short",month:"short",day:"numeric",year:"numeric",hour:"2-digit",minute:"2-digit",timeZoneName:"short"}});
                </script></body></html>""",
                height=22,
                scrolling=False,
            )
        else:
            date_str = pd.to_datetime(match_date).strftime("%a, %b %d %Y") if match_date else ""
            st.caption(f"{stage} — {date_str}")

        if prediction_row is None:
            st.info("No prediction data available for this fixture.")
            st.divider()
            return

        # --- Read pre-computed predictions ---
        prob_home_win = float(prediction_row["prob_home_win"])
        prob_draw     = float(prediction_row["prob_draw"])
        prob_away_win = float(prediction_row["prob_away_win"])

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
        h_goals    = int(prediction_row["predicted_home_goals"])
        a_goals    = int(prediction_row["predicted_away_goals"])
        confidence = float(prediction_row["confidence"])
        st.markdown(
            f"**Predicted:** {home_team} {h_goals} – {a_goals} {away_team}"
        )
        st.caption(TOOLTIPS["scoreline"])

        # --- Confidence score ---
        st.markdown(_confidence_tier_html(confidence), unsafe_allow_html=True)

        # --- Actual result (live or full-time) ---
        if result_row is not None and _decided(result_row.get("has_score")) and bool(result_row.get("has_score")):
            h_actual = result_row.get("home_score")
            a_actual = result_row.get("away_score")
            if _decided(h_actual) and _decided(a_actual):
                actual_tip = TOOLTIPS["actual_score"]
                st.markdown(
                    f"<span title=\"{actual_tip}\"><b>Actual:</b> "
                    f"{home_team} {int(h_actual)} &ndash; {int(a_actual)} {away_team}</span>",
                    unsafe_allow_html=True,
                )
                badges = _result_badges_html(result_row)
                if badges:
                    st.markdown(badges, unsafe_allow_html=True)

                # Per-side goals models (home model vs away model)
                goals_html = _goals_compare_html(result_row, home_team, away_team)
                if goals_html:
                    st.markdown(goals_html, unsafe_allow_html=True)

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

