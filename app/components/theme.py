"""Shared design system: colour tokens, global CSS, and chart styling helpers.

`inject_global_css()` must be called once per script run (right after
`st.set_page_config`) so that the `pc-*` / `sb-*` / `page-hero` classes used by
the components are available everywhere.
"""

import streamlit as st

# --- Colour tokens (keep in sync with .streamlit/config.toml) ---
BG = "#0A0E17"
SURFACE = "#141C2F"
SURFACE_LIGHT = "#1A2336"
BORDER = "#26324B"
TEXT = "#E8EDF6"
TEXT_MUTED = "#8B96AD"
TEXT_SOFT = "#AAB4C8"
GREEN = "#22C55E"
GOLD = "#FACC15"
RED = "#F87171"

# Outcome colours: blue (home) / slate (draw) / orange (away).
# Blue-orange is colour-blind safe and avoids implying good/bad semantics.
COLOR_HOME = "#4DA3FF"
COLOR_DRAW = "#8A94A6"
COLOR_AWAY = "#FF9F43"

FONT_STACK = "'Inter', 'Source Sans Pro', -apple-system, 'Segoe UI', Roboto, sans-serif"

_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ============ Typography ============ */
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp small, .stApp td, .stApp th {
    font-family: 'Inter', 'Source Sans Pro', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
}
h1, h2, h3 { letter-spacing: -0.015em; }

/* ============ Layout chrome ============ */
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 1.6rem; padding-bottom: 4rem; }

/* ============ Sidebar ============ */
[data-testid="stSidebar"] { background: #0D1322; border-right: 1px solid #1D2740; }
.sb-brand { display: flex; flex-direction: column; gap: 3px; padding: 0.3rem 0 1.0rem; }
.sb-brand .ttl { font-size: 1.18rem; font-weight: 800; color: #F2F6FF; line-height: 1.25; }
.sb-brand .sub { font-size: 0.76rem; color: #8B96AD; }
.sb-foot { margin-top: 1.4rem; padding-top: 0.9rem; border-top: 1px solid #1D2740;
           color: #66718A; font-size: 0.72rem; line-height: 1.7; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 4px; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    width: 100%; padding: 0.45rem 0.7rem; border-radius: 10px;
    border: 1px solid transparent; transition: background .15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(77,163,255,0.08); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: rgba(34,197,94,0.12); border-color: rgba(34,197,94,0.30);
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] label p { font-weight: 600; font-size: 0.92rem; }

/* ============ Page hero ============ */
.page-hero { margin: 0 0 1.1rem; }
.page-hero h1 { font-size: 1.85rem; font-weight: 800; margin: 0; color: #F2F6FF; }
.page-hero p { margin: 0.3rem 0 0; color: #8B96AD; font-size: 0.95rem; }

/* ============ Bordered containers as cards ============ */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg, #151E33 0%, #111829 100%);
    border: 1px solid #26324B !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 26px rgba(2, 6, 16, 0.35);
    padding: 1.0rem 1.1rem !important;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }

/* ============ Metrics ============ */
[data-testid="stMetric"] {
    background: #141C2F; border: 1px solid #26324B; border-radius: 14px;
    padding: 0.7rem 0.9rem;
}
[data-testid="stMetric"] label p { font-size: 0.76rem !important; color: #8B96AD !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ============ Expanders / buttons / widget labels ============ */
[data-testid="stExpander"] details {
    border: 1px solid #26324B; border-radius: 12px; background: rgba(10, 14, 23, 0.45);
}
[data-testid="stExpander"] summary { font-size: 0.86rem; font-weight: 600; color: #AAB4C8; }
[data-testid="stExpander"] summary:hover { color: #E8EDF6; }
.stButton button { border-radius: 10px; font-weight: 600; }
[data-testid="stWidgetLabel"] p { font-size: 0.8rem; color: #AAB4C8; font-weight: 600; }

/* ============ Prediction card internals ============ */
.pc-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.pc-stage {
    font-size: 0.66rem; letter-spacing: 0.12em; font-weight: 700; text-transform: uppercase;
    color: #8B96AD; background: rgba(139,150,173,0.12); padding: 3px 10px; border-radius: 999px;
}
.pc-kickoff { text-align: center; color: #8B96AD; font-size: 0.78rem; margin: 0; }

.pc-score-row { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center;
                gap: 10px; margin: 4px 0 2px; }
.pc-team { display: flex; flex-direction: column; align-items: center; gap: 6px; min-width: 0; }
.pc-team img { width: 44px; height: auto; border-radius: 5px; box-shadow: 0 2px 8px rgba(0,0,0,0.45); }
.pc-team .flag-tbd {
    width: 44px; height: 32px; border-radius: 5px; border: 1.5px dashed #3A4763;
    color: #66718A; font-size: 0.72rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}
.pc-team .name { font-weight: 600; font-size: 0.93rem; text-align: center; color: #E8EDF6; line-height: 1.2; }
.pc-center { text-align: center; min-width: 104px; }
.pc-center .label { font-size: 0.62rem; letter-spacing: 0.14em; text-transform: uppercase;
                    color: #8B96AD; font-weight: 700; margin-bottom: 3px; }
.pc-center .score { font-size: 2.05rem; font-weight: 800; color: #FFFFFF; line-height: 1;
                    font-variant-numeric: tabular-nums; white-space: nowrap; }
.pc-center .sub { font-size: 0.73rem; color: #8B96AD; margin-top: 4px; }

.pc-bar { display: flex; height: 22px; border-radius: 8px; overflow: hidden; gap: 2px; margin-top: 2px; }
.pc-bar .seg { display: flex; align-items: center; justify-content: center;
               font-size: 0.7rem; font-weight: 700; color: rgba(255,255,255,0.95); min-width: 3px; }
.pc-bar .seg.home { background: linear-gradient(180deg, #5BACFF, #3D8EE8); }
.pc-bar .seg.draw { background: #56627A; }
.pc-bar .seg.away { background: linear-gradient(180deg, #FFAC5C, #F08C24); }
.pc-legend { display: flex; justify-content: space-between; gap: 8px;
             font-size: 0.74rem; color: #AAB4C8; margin-top: 5px; }
.pc-legend .item { display: flex; align-items: center; gap: 5px; min-width: 0; }
.pc-legend .item b { color: #E8EDF6; }
.pc-legend .dot { width: 8px; height: 8px; border-radius: 2.5px; flex-shrink: 0; }
.pc-legend .dot-home { background: #4DA3FF; }
.pc-legend .dot-draw { background: #8A94A6; }
.pc-legend .dot-away { background: #FF9F43; }

.pc-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 2px; }
.pc-chip { display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 3px 10px;
           border-radius: 999px; border: 1px solid transparent; line-height: 1.5; }
.pc-chip.chip-green { background: rgba(34,197,94,0.12); color: #4ADE80; border-color: rgba(34,197,94,0.35); }
.pc-chip.chip-red   { background: rgba(239,68,68,0.12); color: #F87171; border-color: rgba(239,68,68,0.30); }
.pc-chip.chip-gold  { background: rgba(234,179,8,0.12); color: #FACC15; border-color: rgba(234,179,8,0.35); }
.pc-chip.chip-amber { background: rgba(234,179,8,0.10); color: #E8C84A; border-color: rgba(234,179,8,0.25); }
.pc-chip.chip-gray  { background: rgba(139,150,173,0.12); color: #AAB4C8; border-color: rgba(139,150,173,0.25); }
.pc-chip.chip-live  { background: rgba(239,68,68,0.14); color: #F87171; border-color: rgba(239,68,68,0.45);
                      animation: pc-pulse 1.5s ease-in-out infinite; }
@keyframes pc-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.55; } }

.pc-edge-row { display: flex; align-items: center; justify-content: space-between; gap: 8px;
               padding: 5px 2px; border-bottom: 1px dashed rgba(139,150,173,0.15); }
.pc-edge-row:last-of-type { border-bottom: none; }
.pc-edge-row .nm { font-weight: 600; font-size: 0.82rem; color: #E8EDF6; flex: 1; min-width: 0; }
.pc-edge-row .nums { color: #8B96AD; font-size: 0.74rem; white-space: nowrap; }

.pc-value-banner {
    background: linear-gradient(90deg, rgba(234,179,8,0.16), rgba(234,179,8,0.04));
    border: 1px solid rgba(234,179,8,0.35); color: #FACC15;
    padding: 8px 12px; border-radius: 10px; font-size: 0.8rem; font-weight: 700; margin-top: 4px;
}
.pc-value-banner.hc {
    background: linear-gradient(90deg, rgba(34,197,94,0.18), rgba(34,197,94,0.05));
    border-color: rgba(34,197,94,0.40); color: #4ADE80;
}

/* ============ Mobile ============ */
@media (max-width: 640px) {
    .block-container { padding-left: 0.9rem; padding-right: 0.9rem; padding-top: 1rem; }
    .page-hero h1 { font-size: 1.45rem; }
    [data-testid="stVerticalBlockBorderWrapper"] { padding: 0.85rem 0.9rem !important; }
    .pc-center .score { font-size: 1.65rem; }
    .pc-team img { width: 36px; }
    .pc-team .flag-tbd { width: 36px; height: 26px; }
    .pc-team .name { font-size: 0.85rem; }
}
</style>
"""


def inject_global_css() -> None:
    """Inject the shared stylesheet. Call once, right after st.set_page_config."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    """Render a page hero: bold title plus a muted one-line subtitle."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="page-hero"><h1>{title}</h1>{sub}</div>', unsafe_allow_html=True)


def style_plotly(fig, height: int | None = None):
    """Apply the app theme to a Plotly figure (transparent bg, Inter, soft grid)."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, color=TEXT_SOFT, size=13),
        colorway=[COLOR_HOME, COLOR_AWAY, GREEN, RED, "#A78BFA"],
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(139,150,173,0.12)", zerolinecolor="rgba(139,150,173,0.25)")
    fig.update_yaxes(gridcolor="rgba(139,150,173,0.12)", zerolinecolor="rgba(139,150,173,0.25)")
    if height is not None:
        fig.update_layout(height=height)
    return fig
