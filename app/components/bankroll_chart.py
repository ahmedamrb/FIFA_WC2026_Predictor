"""Bankroll balance chart for the value-bet performance section.

Plots the running bankroll after each settled value bet as a line, baselined at
the starting bankroll. The area is shaded green while in profit (above the
baseline) and red while in loss (below it). Baseline crossings are interpolated
so the green/red regions meet exactly at the break-even line instead of leaving
triangular artefacts at the crossover.
"""

import plotly.graph_objects as go
import streamlit as st

from components.theme import style_plotly

_GREEN_FILL = "rgba(34,197,94,0.20)"
_RED_FILL = "rgba(248,113,113,0.20)"
_LINE = "#E8EDF6"
_BASELINE = "rgba(139,150,173,0.55)"


def _split_at_base(xs, ys, base):
    """Return (xs, ys) with interpolated points inserted wherever the line
    crosses *base*, so a clamped series fills exactly up to the baseline."""
    out_x, out_y = [], []
    for i in range(len(xs)):
        out_x.append(xs[i])
        out_y.append(ys[i])
        if i < len(xs) - 1:
            y0, y1 = ys[i], ys[i + 1]
            if (y0 - base) * (y1 - base) < 0:  # strict sign change -> crosses base
                t = (base - y0) / (y1 - y0)
                out_x.append(xs[i] + t * (xs[i + 1] - xs[i]))
                out_y.append(base)
    return out_x, out_y


def render_bankroll_chart(curve, initial_bankroll, mode="flat"):
    """Render the running-bankroll line chart with green-profit / red-loss fills.

    Args:
        curve: Output of ``src.betting.roi.bankroll_curve`` — settled bets in
            chronological order with a ``bankroll`` column (plus ``match_date``,
            ``home_team``, ``away_team``, ``value_outcome``, ``won``,
            ``pl_dollars`` for hover labels).
        initial_bankroll: Starting bankroll in dollars (the baseline).
        mode: ``"flat"`` or ``"compound"`` — used in the caption only.
    """
    if curve is None or curve.empty:
        return

    base = float(initial_bankroll)
    balances = [base] + curve["bankroll"].astype(float).tolist()
    x = list(range(len(balances)))

    # Per-point hover labels (index 0 is the starting bankroll).
    dates = curve["match_date"].astype(str).str.slice(0, 10).tolist()
    labels = ["<b>Start</b>"]
    for i in range(len(curve)):
        r = curve.iloc[i]
        result = "✅ Won" if bool(r["won"]) else "❌ Lost"
        labels.append(
            f"{dates[i]} &middot; {r['home_team']} vs {r['away_team']}<br>"
            f"{r['value_outcome']} &middot; {result} &middot; P/L ${r['pl_dollars']:+,.2f}"
        )

    # Clamp to the baseline (with interpolated crossings) for the two area fills.
    sx, sy = _split_at_base(x, balances, base)
    green_y = [v if v >= base else base for v in sy]
    red_y = [v if v <= base else base for v in sy]

    fig = go.Figure()
    # Invisible baseline + green fill above it.
    fig.add_trace(go.Scatter(x=sx, y=[base] * len(sx), mode="lines",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=sx, y=green_y, mode="lines", fill="tonexty",
                             fillcolor=_GREEN_FILL, line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    # Invisible baseline (again) + red fill below it.
    fig.add_trace(go.Scatter(x=sx, y=[base] * len(sx), mode="lines",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=sx, y=red_y, mode="lines", fill="tonexty",
                             fillcolor=_RED_FILL, line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    # Bankroll line on top (actual bet points, with markers).
    fig.add_trace(go.Scatter(
        x=x, y=balances, mode="lines+markers", name="Bankroll",
        line=dict(color=_LINE, width=2.2), marker=dict(size=6, color=_LINE),
        text=labels, hovertemplate="%{text}<br><b>Bankroll:</b> $%{y:,.2f}<extra></extra>",
        showlegend=False,
    ))

    fig.add_hline(
        y=base, line_dash="dash", line_color=_BASELINE,
        annotation_text=f"Start ${base:,.0f}",
        annotation_position="bottom right",
        annotation_font_color="#8B96AD",
    )

    fig.update_layout(xaxis_title="Settled bet (date order)", yaxis_title="Bankroll ($)")
    fig.update_xaxes(tickmode="linear", dtick=1)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    style_plotly(fig, height=320)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Running bankroll after each settled value bet ({mode} staking). "
        "Green = in profit vs the starting bankroll, red = in loss."
    )
