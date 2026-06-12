"""Streamlit dashboard entry point."""

import os
import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src/ importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.odds import load_odds_for_backtest  # noqa: E402
from src.data.ingest import fetch_wc2026_results  # noqa: E402
from src.evaluation.live_tracking import build_comparison, summarize  # noqa: E402
from components.tooltips import TOOLTIPS  # noqa: E402
from components.theme import inject_global_css, page_header  # noqa: E402

# Path constants
_PROCESSED = _REPO_ROOT / "data" / "processed"
_RAW = _REPO_ROOT / "data" / "raw"

# Must be the first Streamlit call
st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()


@st.cache_resource
def load_resources() -> dict:
    """Load all models and data files once and cache them for the session."""
    predictions_path = _PROCESSED / "wc2026_predictions.csv"
    predictions = pd.read_csv(predictions_path, index_col=0) if predictions_path.exists() else None

    ft = _PROCESSED / "features_train.parquet"
    features_train = pd.read_parquet(ft) if ft.exists() else None

    fixtures = pd.read_csv(_RAW / "wc2026_fixtures_flat.csv", parse_dates=["match_date"])
    odds = load_odds_for_backtest(mode="real")

    with open(_PROCESSED / "final_backtest_metrics.json", encoding="utf-8") as f:
        backtest_metrics = json.load(f)

    backtest_wc2018 = pd.read_csv(_PROCESSED / "backtest_wc2018.csv")
    backtest_wc2022 = pd.read_csv(_PROCESSED / "backtest_wc2022.csv")

    return {
        "predictions": predictions,
        "fixtures": fixtures,
        "odds": odds,
        "backtest_metrics": backtest_metrics,
        "backtest_wc2018": backtest_wc2018,
        "backtest_wc2022": backtest_wc2022,
        "features_train": features_train,
    }


def _get_fd_key() -> str | None:
    """Resolve the football-data.org key from Streamlit secrets or the env/.env."""
    try:
        key = st.secrets.get("FD_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass  # no secrets.toml configured
    return os.getenv("FD_API_KEY")


@st.cache_data(ttl=60)
def load_results_live():
    """Fetch live WC 2026 results (cached 60s); fall back to the committed CSV.

    Returns a results DataFrame (keyed by fixture_id) or None when neither a
    live fetch nor a cached CSV is available.
    """
    results_csv = _PROCESSED / "wc2026_results.csv"
    key = _get_fd_key()
    if key:
        try:
            return fetch_wc2026_results(api_key=key, write_csv=True)
        except Exception:
            pass  # fall back to the committed snapshot below
    if results_csv.exists():
        return pd.read_csv(results_csv)
    return None


# Load all resources (cached after first run)
resources = load_resources()

# Sidebar navigation
_NAV_PREDICTIONS = "⚽ Match Predictions"
_NAV_BRACKET = "🏆 Tournament Bracket"
_NAV_PERFORMANCE = "📊 Model Performance"
_NAV_INFO = "🧠 Data & Model Info"

with st.sidebar:
    st.markdown(
        '<div class="sb-brand"><span class="ttl">⚽ WC 2026 Predictor</span>'
        '<span class="sub">ML match predictions · live tracking</span></div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        [_NAV_PREDICTIONS, _NAV_BRACKET, _NAV_PERFORMANCE, _NAV_INFO],
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="sb-foot">Data: football-data.org · FIFA rankings<br>'
        "Models: XGBoost · Random Forest · LogReg</div>",
        unsafe_allow_html=True,
    )

# Page routing
if page == _NAV_PREDICTIONS:
    page_header(
        "Match Predictions",
        "Win probabilities, scorelines and live results for every fixture — "
        "open a card's odds section for value-bet analysis.",
    )

    from app.components.prediction_card import render_prediction_card

    fixtures = resources["fixtures"]
    predictions = resources["predictions"]

    # --- Filter & refresh toolbar ---
    with st.container(border=True):
        col_stage, col_conf, col_dates = st.columns(3)

        with col_stage:
            stages = ["All"] + sorted(fixtures["stage"].dropna().unique().tolist())
            selected_stage = st.selectbox("Filter by Stage", stages, help=TOOLTIPS["stage_filter"])

        with col_conf:
            min_conf_pct = st.slider(
                "Min. Model Confidence",
                min_value=0, max_value=70, value=0, step=5, format="%d%%",
                help=TOOLTIPS["min_confidence_filter"]
            )
            min_conf_threshold = min_conf_pct / 100.0

        with col_dates:
            min_date = fixtures["match_date"].min().date()
            max_date = fixtures["match_date"].max().date()
            date_range = st.date_input(
                "Filter by Date",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help=TOOLTIPS["date_filter"],
            )

        # --- Live score refresh controls (kept outside the fragment so toggling
        #     them rebuilds the auto-refresh interval) ---
        ctrl_refresh, ctrl_auto, _ctrl_sp = st.columns([1.2, 1.2, 3])
        with ctrl_refresh:
            if st.button("🔄 Refresh scores", help="Pull the latest live / full-time scores now."):
                load_results_live.clear()
                st.rerun()
        with ctrl_auto:
            auto_refresh = st.checkbox(
                "Auto-refresh (60s)",
                value=False,
                help="Automatically re-pull scores every 60 seconds while you watch.",
            )

    # Apply filters
    filtered = fixtures.copy()
    if selected_stage != "All":
        filtered = filtered[filtered["stage"] == selected_stage]

    # date_input returns a tuple of (start, end) or a single date
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["match_date"].dt.date >= start_date)
            & (filtered["match_date"].dt.date <= end_date)
        ]

    # Preserve the original CSV row index (0-based) for positional lookup in
    # features_predict.parquet, which has rows in the same order as the CSV.
    filtered = filtered.sort_values("match_date")

    def render_results_section():
        results = load_results_live()
        comparison_df = build_comparison(fixtures, predictions, results)
        results_lookup = comparison_df.set_index("fixture_id").to_dict("index")

        # Freshness / availability note
        if results is not None and not results.empty and "fetched_at" in results.columns:
            fetched = str(results["fetched_at"].dropna().max())
            st.caption(f"Scores updated: {fetched} UTC")
        elif results is None:
            st.caption("Live scores unavailable \u2014 showing predictions only.")

        # --- Running accuracy banner (all finished known-team matches) ---
        s = summarize(comparison_df)
        if s["played"] > 0 or s["live"] > 0:
            b1, b2, b3, b4, b5, b6 = st.columns(6)
            b1.metric("Matches played", s["played"], help=TOOLTIPS["live_accuracy"])
            b2.metric(
                "Outcomes correct",
                f"{s['outcome_correct']}/{s['played']} ({s['outcome_pct']:.0%})",
                help=TOOLTIPS["outcome_verdict"],
            )
            b3.metric(
                "Exact scorelines",
                f"{s['exact']}/{s['played']} ({s['exact_pct']:.0%})",
                help=TOOLTIPS["exact_score"],
            )
            b4.metric(
                "Home goals exact",
                f"{s['home_goals_correct']}/{s['played']} ({s['home_goals_pct']:.0%})",
                delta=f"MAE {s['home_goals_mae']:.2f}",
                delta_color="off",
                help=TOOLTIPS["home_goals_model"],
            )
            b5.metric(
                "Away goals exact",
                f"{s['away_goals_correct']}/{s['played']} ({s['away_goals_pct']:.0%})",
                delta=f"MAE {s['away_goals_mae']:.2f}",
                delta_color="off",
                help=TOOLTIPS["away_goals_model"],
            )
            b6.metric("Live now", s["live"], help=TOOLTIPS["match_status"])
            st.divider()

        if filtered.empty:
            st.info("No fixtures match the selected filters.")
            return

        cards = []
        for _, fixture_row in filtered.iterrows():
            prediction_row = predictions.loc[fixture_row.name] if predictions is not None else None

            # Pre-filter by confidence when threshold is active
            if prediction_row is None or (min_conf_threshold > 0 and float(prediction_row["confidence"]) < min_conf_threshold):
                continue
            cards.append((fixture_row, prediction_row))

        filter_note = f" \u00b7 confidence \u2265{min_conf_pct}% filter active" if min_conf_threshold > 0 else ""
        st.caption(f"Showing {len(cards)} fixture(s){filter_note}")

        # Two-column card grid on desktop; Streamlit stacks columns on mobile.
        for i in range(0, len(cards), 2):
            cols = st.columns(2, gap="medium")
            for col, (fixture_row, prediction_row) in zip(cols, cards[i:i + 2]):
                with col:
                    render_prediction_card(
                        fixture_row,
                        prediction_row,
                        odds_df=resources["odds"],
                        result_row=results_lookup.get(fixture_row["fixture_id"]),
                    )

    # Auto-refresh via a dynamically built fragment (built-in; no extra deps).
    if hasattr(st, "fragment"):
        _section = st.fragment(run_every=(60 if auto_refresh else None))(render_results_section)
        _section()
    else:
        render_results_section()
elif page == _NAV_BRACKET:
    page_header(
        "Tournament Bracket",
        "Predicted group standings and a simulated knockout path to the final.",
    )
    from app.components.bracket import render_bracket
    render_bracket(resources["fixtures"], resources["predictions"])
elif page == _NAV_PERFORMANCE:
    page_header(
        "Model Performance",
        "Walk-forward backtests on World Cup 2018 and 2022 — no look-ahead bias.",
    )

    from app.components.performance_charts import (
        render_calibration_chart,
        render_cumulative_profit_chart,
        render_metrics_bar_chart,
    )

    metrics = resources["backtest_metrics"]
    wc2018_df = resources["backtest_wc2018"]
    wc2022_df = resources["backtest_wc2022"]

    # --- Summary metric row ---
    combined_log_loss = (metrics["wc2018"]["log_loss"] + metrics["wc2022"]["log_loss"]) / 2
    combined_accuracy = (metrics["wc2018"]["accuracy"] + metrics["wc2022"]["accuracy"]) / 2
    combined_roi = (
        metrics["wc2018"]["flat_stake_roi"] + metrics["wc2022"]["flat_stake_roi"]
    ) / 2

    col1, col2, col3 = st.columns(3)
    col1.metric("Combined Log-loss", f"{combined_log_loss:.4f}", help=TOOLTIPS["log_loss"])
    col2.metric("Combined Accuracy", f"{combined_accuracy:.1%}", help=TOOLTIPS["accuracy"])
    col3.metric("Combined Flat-stake ROI", f"{combined_roi:.2f}%", help=TOOLTIPS["flat_stake_roi"])

    st.divider()

    # --- Charts ---
    st.subheader("Metrics by Tournament")
    render_metrics_bar_chart(metrics)

    st.subheader("Cumulative Profit")
    render_cumulative_profit_chart(wc2018_df, wc2022_df)

    st.subheader("Model Calibration")
    render_calibration_chart()
elif page == _NAV_INFO:
    page_header(
        "Data & Model Info",
        "What the models learn from, how they are trained, and version history.",
    )

    from app.components.model_info import (
        render_feature_importance,
        render_model_registry,
        render_training_summary,
    )

    # --- Last Retrained metric (read from MODEL_REGISTRY.md) ---
    _registry_path = _REPO_ROOT / "models" / "MODEL_REGISTRY.md"
    last_retrained = "Unknown"
    if _registry_path.exists():
        import re
        _registry_text = _registry_path.read_text(encoding="utf-8")
        _dates = re.findall(r"\|\s*(\d{4}-\d{2}-\d{2})\s*\|", _registry_text)
        if _dates:
            last_retrained = max(_dates)

    st.metric("Last Retrained", last_retrained, help=TOOLTIPS["last_retrained"])
    st.divider()

    # --- Feature Importance ---
    st.subheader("Feature Importance")
    render_feature_importance()

    st.divider()

    # --- Training Data Summary ---
    st.subheader("Training Data Summary")
    if resources.get("features_train") is not None:
        render_training_summary(resources["features_train"])
    else:
        st.warning("features_train.parquet not found.")

    st.divider()

    # --- Model Registry ---
    st.subheader("Model Registry")
    render_model_registry()
