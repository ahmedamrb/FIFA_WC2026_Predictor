"""Pre-compute WC 2026 predictions and feature importances for the Streamlit dashboard.

Run this script locally before deploying the dashboard so the app never needs
to load large .pkl model files at runtime.

Outputs
-------
data/processed/wc2026_predictions.csv   — one row per fixture
data/processed/feature_importances.json — XGBoost feature importance dict

Usage
-----
    python scripts/precompute_predictions.py
"""

import sys
import json
import datetime
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Make src/ importable from any working directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.models.ensemble import WC2026Ensemble, TemperatureScaledEnsemble  # noqa: E402
from src.data.preprocess import FEATURE_COLUMNS  # noqa: E402

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_MODELS_DIR = _REPO_ROOT / "models"
_PROCESSED = _REPO_ROOT / "data" / "processed"
_RAW = _REPO_ROOT / "data" / "raw"

_OUTPUT_PREDICTIONS = _PROCESSED / "wc2026_predictions.csv"
_OUTPUT_IMPORTANCES = _PROCESSED / "feature_importances.json"


def _load_models() -> dict:
    """Load the five required pkl model files from models/.

    Returns:
        Dict with keys: outcome_lr, outcome_rf, outcome_xgb,
        home_goals_xgb, away_goals_xgb.

    Raises:
        FileNotFoundError: If any required model file is missing.
    """
    required = {
        "outcome_lr": "outcome_lr.pkl",
        "outcome_rf": "outcome_rf.pkl",
        "outcome_xgb": "outcome_xgb.pkl",
        "home_goals_xgb": "home_goals_xgb.pkl",
        "away_goals_xgb": "away_goals_xgb.pkl",
    }
    models = {}
    for key, filename in required.items():
        path = _MODELS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Required model not found: {path}")
        print(f"  Loading {filename} ...")
        models[key] = joblib.load(path)
    return models


def _build_ensemble(models: dict):
    """Build WC2026Ensemble and optionally wrap with temperature scaling.

    Checks data/processed/temperature.json: if ``"applied": true``, wraps
    the ensemble in a TemperatureScaledEnsemble.

    Args:
        models: Dict returned by _load_models().

    Returns:
        WC2026Ensemble or TemperatureScaledEnsemble instance.
    """
    ensemble = WC2026Ensemble(
        models["outcome_lr"],
        models["outcome_rf"],
        models["outcome_xgb"],
    )

    temp_path = _PROCESSED / "temperature.json"
    if temp_path.exists():
        with open(temp_path, encoding="utf-8") as f:
            temp_data = json.load(f)
        if temp_data.get("applied", False):
            temperature = float(temp_data["temperature"])
            print(f"  Applying temperature scaling (T={temperature:.4f}) ...")
            ensemble = TemperatureScaledEnsemble(ensemble, temperature)
        else:
            print("  Temperature scaling found but not applied (applied=false).")
    else:
        print("  No temperature.json found — using uncalibrated ensemble.")

    return ensemble


def _load_features() -> pd.DataFrame:
    """Load the pre-computed prediction feature matrix.

    Returns:
        DataFrame with at least FEATURE_COLUMNS columns.

    Raises:
        FileNotFoundError: If features_predict.parquet is missing.
    """
    fp = _PROCESSED / "features_predict.parquet"
    if not fp.exists():
        raise FileNotFoundError(
            f"features_predict.parquet not found at {fp}. "
            "Run scripts/run_feature_engineering.py first."
        )
    return pd.read_parquet(fp)


def _load_fixtures() -> pd.DataFrame:
    """Load the raw WC 2026 fixture list.

    Returns:
        DataFrame with columns: home_team, away_team, match_date, stage.

    Raises:
        FileNotFoundError: If the fixture CSV is missing.
    """
    fp = _RAW / "wc2026_fixtures_flat.csv"
    if not fp.exists():
        raise FileNotFoundError(f"wc2026_fixtures_flat.csv not found at {fp}.")
    return pd.read_csv(fp)


def _outcome_label(prob_home: float, prob_draw: float, prob_away: float) -> str:
    """Return the string label for the most probable outcome.

    Args:
        prob_home: Probability of home win.
        prob_draw: Probability of draw.
        prob_away: Probability of away win.

    Returns:
        One of "Home Win", "Draw", or "Away Win".
    """
    idx = int(np.argmax([prob_home, prob_draw, prob_away]))
    return ["Home Win", "Draw", "Away Win"][idx]


def main() -> None:
    """Run the pre-computation pipeline."""
    print("=" * 60)
    print("WC 2026 Predictions Pre-computation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load models
    # ------------------------------------------------------------------
    print("\n[1/5] Loading models ...")
    models = _load_models()

    # ------------------------------------------------------------------
    # 2. Build ensemble
    # ------------------------------------------------------------------
    print("\n[2/5] Building ensemble ...")
    ensemble = _build_ensemble(models)

    # ------------------------------------------------------------------
    # 3. Load feature matrix and fixtures
    # ------------------------------------------------------------------
    print("\n[3/5] Loading feature matrix and fixture list ...")
    features_predict = _load_features()
    fixtures = _load_fixtures()

    n_features = len(features_predict)
    n_fixtures = len(fixtures)
    if n_features != n_fixtures:
        raise ValueError(
            f"Row count mismatch: features_predict has {n_features} rows "
            f"but wc2026_fixtures_flat.csv has {n_fixtures} rows. "
            "Re-run scripts/run_feature_engineering.py to regenerate "
            "features_predict.parquet from the current fixture list."
        )
    print(f"  {n_fixtures} fixtures / feature rows confirmed aligned.")

    # ------------------------------------------------------------------
    # 4. Run inference
    # ------------------------------------------------------------------
    print("\n[4/5] Running inference ...")

    X = features_predict[FEATURE_COLUMNS]

    # Outcome probabilities — shape (N, 3): [away_win, draw, home_win]
    proba = ensemble.predict_proba(X)

    # Goals predictions — clip to non-negative integers
    raw_home_goals = models["home_goals_xgb"].predict(X)
    raw_away_goals = models["away_goals_xgb"].predict(X)
    pred_home_goals = [max(0, round(float(v))) for v in raw_home_goals]
    pred_away_goals = [max(0, round(float(v))) for v in raw_away_goals]

    # ------------------------------------------------------------------
    # 5. Build and save output DataFrame
    # ------------------------------------------------------------------
    print("\n[5/5] Building output and saving ...")

    generated_at = datetime.date.today().isoformat()

    prob_home_win = np.round(proba[:, 2], 4)
    prob_draw = np.round(proba[:, 1], 4)
    prob_away_win = np.round(proba[:, 0], 4)
    confidence = np.round(np.max(proba, axis=1), 4)

    predicted_outcome = [
        _outcome_label(float(ph), float(pd_), float(pa))
        for ph, pd_, pa in zip(prob_home_win, prob_draw, prob_away_win)
    ]

    predictions_df = pd.DataFrame(
        {
            "home_team": fixtures["home_team"].values,
            "away_team": fixtures["away_team"].values,
            "match_date": fixtures["match_date"].astype(str).values,
            "stage": fixtures["stage"].values,
            "prob_home_win": prob_home_win,
            "prob_draw": prob_draw,
            "prob_away_win": prob_away_win,
            "confidence": confidence,
            "predicted_outcome": predicted_outcome,
            "predicted_home_goals": pred_home_goals,
            "predicted_away_goals": pred_away_goals,
            "generated_at": generated_at,
        }
    )

    predictions_df.to_csv(_OUTPUT_PREDICTIONS, index=True)
    print(f"  Saved predictions  → {_OUTPUT_PREDICTIONS}")

    # ------------------------------------------------------------------
    # Feature importances from XGBoost outcome model
    # ------------------------------------------------------------------
    outcome_xgb = models["outcome_xgb"]
    importances = outcome_xgb.feature_importances_.tolist()
    importance_dict = dict(zip(FEATURE_COLUMNS, importances))

    with open(_OUTPUT_IMPORTANCES, "w", encoding="utf-8") as f:
        json.dump(importance_dict, f, indent=2)
    print(f"  Saved importances  → {_OUTPUT_IMPORTANCES}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Rows written      : {len(predictions_df)}")
    print(f"  Predictions CSV   : {_OUTPUT_PREDICTIONS}")
    print(f"  Importances JSON  : {_OUTPUT_IMPORTANCES}")
    print(f"  Generated at      : {generated_at}")
    print("=" * 60)


if __name__ == "__main__":
    main()
