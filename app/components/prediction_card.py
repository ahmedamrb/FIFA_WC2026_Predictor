"""Prediction card UI component.

Card anatomy (top to bottom):
  stage chip + live/FT status chip
  scoreboard: home flag/name · focal scoreline (actual when played, else predicted) · away flag/name
  kickoff time in the viewer's local timezone
  segmented W/D/L probability bar with legend
  chips: confidence tier, outcome/exact-score verdicts, per-side goals-model hits, value-bet signal
  expander: bookmaker odds inputs + per-outcome edge breakdown
"""

import pandas as pd
import streamlit as st

from components.flags import flag_url
from components.tooltips import TOOLTIPS

_BEST_VALUE_THRESHOLD = 0.05
_AVOID_THRESHOLD = -0.05
_CONF_HIGH = 0.55   # >=55% -> High confidence
_CONF_MED  = 0.45   # 45-54% -> Medium confidence; <45% -> Low confidence

_STAGE_LABELS = {
    "GROUP_STAGE": "Group Stage",
    "LAST_32": "Round of 32",
    "LAST_16": "Round of 16",
    "QUARTER_FINALS": "Quarter-finals",
    "SEMI_FINALS": "Semi-finals",
    "THIRD_PLACE": "Third Place",
    "FINAL": "Final",
}

# Probability-bar segments narrower than this hide their inline % label;
# the legend below always shows the exact numbers.
_MIN_SEG_LABEL = 0.14


def _decided(value) -> bool:
    """True when a nullable flag/number has a concrete (non-NA) value."""
    return value is not None and not pd.isna(value)


def _stage_label(stage) -> str:
    s = str(stage or "").strip()
    return _STAGE_LABELS.get(s, s.replace("_", " ").title())


def _chip(label: str, cls: str, tip: str = "") -> str:
    t = f' title="{tip}"' if tip else ""
    return f'<span class="pc-chip {cls}"{t}>{label}</span>'


def _status_chip(status, minute) -> str:
    """Live/HT/FT status chip for the card's top-right corner ('' for upcoming)."""
    s = str(status or "").upper()
    tip = TOOLTIPS["match_status"]
    if s == "IN_PLAY":
        mins = f" {int(minute)}&prime;" if _decided(minute) else ""
        return _chip(f"&#9679; LIVE{mins}", "chip-live", tip)
    if s == "PAUSED":
        return _chip("HT", "chip-amber", tip)
    if s in ("FINISHED", "AWARDED"):
        return _chip("FT", "chip-gray", tip)
    return ""


def _team_html(team: str) -> str:
    url = flag_url(team, size="w80")
    if url:
        flag = f'<img src="{url}" alt="" loading="lazy"/>'
    else:
        initials = "".join(w[0] for w in str(team).split()[:2]).upper() or "?"
        flag = f'<div class="flag-tbd">{initials}</div>'
    return f'<div class="pc-team">{flag}<div class="name">{team}</div></div>'


def _scoreboard_html(home_team, away_team, label, score, sub="", tip="") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    t = f' title="{tip}"' if tip else ""
    return (
        '<div class="pc-score-row">'
        + _team_html(home_team)
        + f'<div class="pc-center"{t}><div class="label">{label}</div>'
        + f'<div class="score">{score}</div>{sub_html}</div>'
        + _team_html(away_team)
        + "</div>"
    )


def _prob_block_html(p_home, p_draw, p_away, home_team, away_team) -> str:
    """Segmented W/D/L probability bar plus a three-item legend."""
    def seg(p: float, cls: str) -> str:
        label = f"{p:.0%}" if p >= _MIN_SEG_LABEL else ""
        return f'<div class="seg {cls}" style="width:{p * 100:.2f}%">{label}</div>'

    bar = (
        f'<div class="pc-bar" title="{TOOLTIPS["outcome_probs"]}">'
        f'{seg(p_home, "home")}{seg(p_draw, "draw")}{seg(p_away, "away")}</div>'
    )
    legend = (
        '<div class="pc-legend">'
        f'<span class="item"><span class="dot dot-home"></span>{home_team} <b>{p_home:.0%}</b></span>'
        f'<span class="item"><span class="dot dot-draw"></span>Draw <b>{p_draw:.0%}</b></span>'
        f'<span class="item"><span class="dot dot-away"></span>{away_team} <b>{p_away:.0%}</b></span>'
        "</div>"
    )
    return bar + legend


def _confidence_chip(confidence: float) -> str:
    if confidence >= _CONF_HIGH:
        cls, word = "chip-green", "High"
    elif confidence >= _CONF_MED:
        cls, word = "chip-amber", "Medium"
    else:
        cls, word = "chip-gray", "Low"
    return _chip(f"{word} confidence &middot; {confidence:.0%}", cls, TOOLTIPS["confidence"])


def _render_kickoff(kickoff_utc: str) -> None:
    """Kickoff datetime converted to the viewer's local timezone (needs JS, so iframe)."""
    st.iframe(
        f"""<!DOCTYPE html><html><head><style>
        html,body{{margin:0;padding:0;overflow:hidden;}}
        div{{font:500 12.5px -apple-system,'Segoe UI',Roboto,sans-serif;color:#8B96AD;
             text-align:center;line-height:24px;}}
        </style></head><body>
        <div>&#128467;&#65039; <span id="kt"></span></div>
        <script>
        var d=new Date("{kickoff_utc}");
        document.getElementById("kt").textContent=d.toLocaleString(undefined,
        {{weekday:"short",month:"short",day:"numeric",hour:"2-digit",minute:"2-digit",timeZoneName:"short"}});
        </script></body></html>""",
        height=24,
    )


def _edge_row_html(name: str, prob: float, odds: float, edge: float, is_best: bool, show_labels: bool) -> str:
    if edge > _BEST_VALUE_THRESHOLD:
        cls = "chip-green"
    elif edge < _AVOID_THRESHOLD:
        cls = "chip-red"
    else:
        cls = "chip-gray"
    badge = f"{edge:+.1%}"
    if is_best and show_labels:
        badge += " &middot; Best value"
    return (
        f'<div class="pc-edge-row" title="{TOOLTIPS["edge"]}">'
        f'<span class="nm">{name}</span>'
        f'<span class="nums">model {prob:.0%} &middot; implied {1.0 / odds:.0%}</span>'
        f"{_chip(badge, cls)}"
        "</div>"
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
        away_team, stage, kickoff_utc, fixture_id.
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

    status = result_row.get("status") if result_row is not None else None
    minute = result_row.get("minute") if result_row is not None else None

    h_act = a_act = None
    has_score = result_row is not None and _decided(result_row.get("has_score")) and bool(result_row.get("has_score"))
    if has_score:
        h_act = result_row.get("home_score")
        a_act = result_row.get("away_score")
        has_score = _decided(h_act) and _decided(a_act)

    with st.container(border=True):
        # --- Top row: stage chip + status chip ---
        st.markdown(
            f'<div class="pc-top"><span class="pc-stage">{_stage_label(stage)}</span>'
            f"{_status_chip(status, minute) or '<span></span>'}</div>",
            unsafe_allow_html=True,
        )

        # --- Scoreboard: actual score when played/live, else predicted ---
        h_pred_g = a_pred_g = None
        if prediction_row is not None:
            h_pred_g = int(prediction_row["predicted_home_goals"])
            a_pred_g = int(prediction_row["predicted_away_goals"])

        if has_score:
            label, score, tip = "Score", f"{int(h_act)} &ndash; {int(a_act)}", TOOLTIPS["actual_score"]
            sub = f"Model: {h_pred_g} &ndash; {a_pred_g}" if h_pred_g is not None else ""
        elif prediction_row is not None:
            label, score, tip = "AI Prediction", f"{h_pred_g} &ndash; {a_pred_g}", TOOLTIPS["scoreline"]
            sub = ""
        else:
            label, score, tip, sub = "Upcoming", "vs", "", ""
        st.markdown(
            _scoreboard_html(home_team, away_team, label, score, sub=sub, tip=tip),
            unsafe_allow_html=True,
        )

        # --- Kickoff time (browser-local via JS; static date as fallback) ---
        if kickoff_utc and kickoff_utc not in ("", "nan", "NaT"):
            _render_kickoff(kickoff_utc)
        elif match_date is not None and str(match_date) not in ("", "nan", "NaT"):
            date_str = pd.to_datetime(match_date).strftime("%a, %b %d %Y")
            st.markdown(f'<div class="pc-kickoff">&#128467;&#65039; {date_str}</div>', unsafe_allow_html=True)

        if prediction_row is None:
            st.markdown(
                _chip("Prediction available once both teams are known", "chip-gray"),
                unsafe_allow_html=True,
            )
            return

        # --- W/D/L probability bar ---
        prob_home_win = float(prediction_row["prob_home_win"])
        prob_draw     = float(prediction_row["prob_draw"])
        prob_away_win = float(prediction_row["prob_away_win"])
        st.markdown(
            _prob_block_html(prob_home_win, prob_draw, prob_away_win, home_team, away_team),
            unsafe_allow_html=True,
        )

        # --- Odds defaults (needed up-front so the value chip can be shown in the chips row) ---
        real_odds = _lookup_odds(odds_df, match_date, home_team, away_team)
        default_home_odds = real_odds["home_win_odds"] if real_odds else 2.0
        default_draw_odds = real_odds["draw_odds"] if real_odds else 3.0
        default_away_odds = real_odds["away_win_odds"] if real_odds else 2.5

        key_prefix = f"{fixture_row.name}_{home_team}_{away_team}"
        cur_home_odds = float(st.session_state.get(f"{key_prefix}_home_odds", default_home_odds))
        cur_draw_odds = float(st.session_state.get(f"{key_prefix}_draw_odds", default_draw_odds))
        cur_away_odds = float(st.session_state.get(f"{key_prefix}_away_odds", default_away_odds))

        home_edge = prob_home_win - (1.0 / cur_home_odds)
        draw_edge = prob_draw - (1.0 / cur_draw_odds)
        away_edge = prob_away_win - (1.0 / cur_away_odds)
        edges = [home_edge, draw_edge, away_edge]
        best_val = max(edges)
        show_labels = best_val > _BEST_VALUE_THRESHOLD

        outcome_names = [f"{home_team} win", "Draw", f"{away_team} win"]
        outcome_probs = [prob_home_win, prob_draw, prob_away_win]
        best_idx = edges.index(best_val)

        # --- Chips row: confidence, verdicts, goals-model hits, value signal ---
        confidence = float(prediction_row["confidence"])
        chips = [_confidence_chip(confidence)]

        if has_score:
            outcome_correct = result_row.get("outcome_correct")
            if _decided(outcome_correct):
                hit = bool(outcome_correct)
                chips.append(_chip(
                    "&#9989; Outcome correct" if hit else "&#10060; Outcome missed",
                    "chip-green" if hit else "chip-red",
                    TOOLTIPS["outcome_verdict"],
                ))
            exact_correct = result_row.get("exact_score_correct")
            if _decided(exact_correct) and bool(exact_correct):
                chips.append(_chip("&#11088; Exact score", "chip-gold", TOOLTIPS["exact_score"]))

            # Per-side goals models (home and away regressors are independent)
            for side, flag_key, pred_g, act_g in (
                ("Home", "home_goals_correct", h_pred_g, h_act),
                ("Away", "away_goals_correct", a_pred_g, a_act),
            ):
                side_flag = result_row.get(flag_key)
                if _decided(side_flag):
                    hit = bool(side_flag)
                    glyph = "&#10003;" if hit else "&#10007;"
                    chips.append(_chip(
                        f"{side} goals {pred_g}&rarr;{int(act_g)} {glyph}",
                        "chip-green" if hit else "chip-red",
                        TOOLTIPS["goals_compare"],
                    ))

        if show_labels:
            chips.append(_chip(
                f"&#128176; Value: {outcome_names[best_idx]} {best_val:+.1%}",
                "chip-gold",
                TOOLTIPS["value_bet"],
            ))

        st.markdown(f'<div class="pc-chips">{"".join(chips)}</div>', unsafe_allow_html=True)

        # --- Betting odds & edge analysis (collapsed to keep the card scannable) ---
        with st.expander("Betting odds & value analysis"):
            if real_odds:
                fetched = real_odds.get("fetched_at", "")[:10]
                st.caption(f"Odds: {real_odds.get('source', '')} (fetched {fetched})")
            else:
                st.caption("No real odds found — using neutral defaults (edit below).")

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

            # Recompute from the widget values (authoritative within the expander)
            home_edge = prob_home_win - (1.0 / home_odds)
            draw_edge = prob_draw - (1.0 / draw_odds)
            away_edge = prob_away_win - (1.0 / away_odds)
            edges = [home_edge, draw_edge, away_edge]
            best_val = max(edges)
            show_labels = best_val > _BEST_VALUE_THRESHOLD
            best_idx = edges.index(best_val)

            rows = "".join(
                _edge_row_html(name, prob, odds, edge, edge == best_val, show_labels)
                for name, prob, odds, edge in zip(
                    outcome_names, outcome_probs, [home_odds, draw_odds, away_odds], edges
                )
            )
            st.markdown(rows, unsafe_allow_html=True)

            # Summary signal — only shown when there is a positive-edge bet
            if show_labels:
                best_outcome_prob = outcome_probs[best_idx]
                is_high_conf_value = confidence >= _CONF_HIGH and best_outcome_prob == confidence
                banner_cls = "pc-value-banner hc" if is_high_conf_value else "pc-value-banner"
                banner_label = (
                    "&#11088; High-Confidence Value Bet" if is_high_conf_value else "&#9989; Best Value Bet"
                )
                st.markdown(
                    f'<div class="{banner_cls}" title="{TOOLTIPS["value_bet"]}">'
                    f"{banner_label}: {outcome_names[best_idx]} &nbsp;|&nbsp; "
                    f"Edge: {best_val:+.1%} &nbsp;|&nbsp; "
                    f"Model: {best_outcome_prob:.0%}"
                    "</div>",
                    unsafe_allow_html=True,
                )
