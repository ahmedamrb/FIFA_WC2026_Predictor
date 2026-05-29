"""Streamlit dashboard entry point."""

import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src/ importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.odds import load_odds_for_backtest  # noqa: E402
from components.tooltips import TOOLTIPS  # noqa: E402

# Path constants
_PROCESSED = _REPO_ROOT / "data" / "processed"
_RAW = _REPO_ROOT / "data" / "raw"

# Must be the first Streamlit call
st.set_page_config(page_title="WC 2026 Predictor", layout="wide")


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


# Load all resources (cached after first run)
resources = load_resources()

# Sidebar navigation
with st.sidebar:
    st.title("WC 2026 Predictor")
    page = st.radio(
        "Navigate",
        ["Match Predictions", "Tournament Bracket", "Model Performance", "Data & Model Info"],
    )

# Page routing
if page == "Match Predictions":
    st.title("Match Predictions")

    from app.components.prediction_card import render_prediction_card, FEATURE_COLUMNS

    fixtures = resources["fixtures"]
    predictions = resources["predictions"]

    # --- Filters ---
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

    if filtered.empty:
        st.info("No fixtures match the selected filters.")
    else:
        caption_placeholder = st.empty()
        n_shown = 0
        for _, fixture_row in filtered.iterrows():
            prediction_row = predictions.loc[fixture_row.name] if predictions is not None else None

            # Pre-filter by confidence when threshold is active
            if prediction_row is None or (min_conf_threshold > 0 and float(prediction_row["confidence"]) < min_conf_threshold):
                continue

            n_shown += 1
            render_prediction_card(fixture_row, prediction_row, odds_df=resources["odds"])
        filter_note = f" \u00b7 confidence \u2265{min_conf_pct}% filter active" if min_conf_threshold > 0 else ""
        caption_placeholder.caption(f"Showing {n_shown} fixture(s){filter_note}")
elif page == "Tournament Bracket":
    st.title("Tournament Bracket")
    from app.components.bracket import render_bracket
    render_bracket(resources["fixtures"], resources["predictions"])
elif page == "Model Performance":
    st.title("Model Performance")

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
elif page == "Data & Model Info":
    st.title("Data & Model Info")

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
