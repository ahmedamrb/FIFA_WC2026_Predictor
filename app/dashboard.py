"""Streamlit dashboard entry point."""

import sys
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Make src/ importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.models.ensemble import WC2026Ensemble  # noqa: E402

# Path constants
_MODELS_DIR = _REPO_ROOT / "models"
_PROCESSED = _REPO_ROOT / "data" / "processed"
_RAW = _REPO_ROOT / "data" / "raw"

# Must be the first Streamlit call
st.set_page_config(page_title="WC 2026 Predictor", layout="wide")


@st.cache_resource
def load_resources() -> dict:
    """Load all models and data files once and cache them for the session."""
    outcome_lr = joblib.load(_MODELS_DIR / "outcome_lr.pkl")
    outcome_rf = joblib.load(_MODELS_DIR / "outcome_rf.pkl")
    outcome_xgb = joblib.load(_MODELS_DIR / "outcome_xgb.pkl")
    home_goals_xgb = joblib.load(_MODELS_DIR / "home_goals_xgb.pkl")
    away_goals_xgb = joblib.load(_MODELS_DIR / "away_goals_xgb.pkl")
    home_goals_poisson = joblib.load(_MODELS_DIR / "home_goals_poisson.pkl")
    away_goals_poisson = joblib.load(_MODELS_DIR / "away_goals_poisson.pkl")

    ensemble = WC2026Ensemble(outcome_lr, outcome_rf, outcome_xgb)

    fp = _PROCESSED / "features_predict.parquet"
    features_predict = pd.read_parquet(fp) if fp.exists() else None

    fixtures = pd.read_csv(_RAW / "wc2026_fixtures_flat.csv", parse_dates=["match_date"])
    odds = pd.read_csv(_REPO_ROOT / "data" / "bookmaker_odds.csv")

    with open(_PROCESSED / "final_backtest_metrics.json", encoding="utf-8") as f:
        backtest_metrics = json.load(f)

    backtest_wc2018 = pd.read_csv(_PROCESSED / "backtest_wc2018.csv")
    backtest_wc2022 = pd.read_csv(_PROCESSED / "backtest_wc2022.csv")

    return {
        "outcome_lr": outcome_lr,
        "outcome_rf": outcome_rf,
        "outcome_xgb": outcome_xgb,
        "home_goals_xgb": home_goals_xgb,
        "away_goals_xgb": away_goals_xgb,
        "home_goals_poisson": home_goals_poisson,
        "away_goals_poisson": away_goals_poisson,
        "ensemble": ensemble,
        "features_predict": features_predict,
        "fixtures": fixtures,
        "odds": odds,
        "backtest_metrics": backtest_metrics,
        "backtest_wc2018": backtest_wc2018,
        "backtest_wc2022": backtest_wc2022,
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

    from app.components.prediction_card import render_prediction_card

    fixtures = resources["fixtures"]
    features_predict = resources["features_predict"]
    ensemble = resources["ensemble"]
    home_goals_model = resources["home_goals_xgb"]
    away_goals_model = resources["away_goals_xgb"]

    # --- Filters ---
    col_stage, col_dates = st.columns([1, 2])

    with col_stage:
        stages = ["All"] + sorted(fixtures["stage"].dropna().unique().tolist())
        selected_stage = st.selectbox("Filter by Stage", stages)

    with col_dates:
        min_date = fixtures["match_date"].min().date()
        max_date = fixtures["match_date"].max().date()
        date_range = st.date_input(
            "Filter by Date",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
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
        st.caption(f"Showing {len(filtered)} fixture(s)")
        for _, fixture_row in filtered.iterrows():
            # features_predict.parquet has no team-name columns; rows align
            # positionally with wc2026_fixtures_flat.csv (same order, 0-indexed).
            features_row = None
            if features_predict is not None:
                fixture_idx = fixture_row.name  # original 0-based index from CSV
                if 0 <= fixture_idx < len(features_predict):
                    features_row = features_predict.iloc[fixture_idx]
            render_prediction_card(
                fixture_row, features_row, ensemble, home_goals_model, away_goals_model
            )
elif page == "Tournament Bracket":
    st.title("Tournament Bracket")
    from app.components.bracket import render_bracket
    render_bracket(resources["fixtures"], resources["features_predict"], resources["ensemble"])
elif page == "Model Performance":
    st.title("Model Performance")
    st.write("Coming soon…")
elif page == "Data & Model Info":
    st.title("Data & Model Info")
    st.write("Coming soon…")
