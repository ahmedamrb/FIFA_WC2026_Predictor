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
    st.write("Coming soon…")
elif page == "Tournament Bracket":
    st.title("Tournament Bracket")
    st.write("Coming soon…")
elif page == "Model Performance":
    st.title("Model Performance")
    st.write("Coming soon…")
elif page == "Data & Model Info":
    st.title("Data & Model Info")
    st.write("Coming soon…")
