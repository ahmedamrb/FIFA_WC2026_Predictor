"""Goals scored regression model for FIFA WC 2026 Predictor.

Trains two PoissonRegressor models — one for home goals, one for away goals —
and two XGBRegressor models using tuned hyperparameters, evaluated on the
WC 2022 validation set.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data.preprocess import FEATURE_COLUMNS

_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def train_poisson_goals(X_train, y_home, y_away):
    """Train two PoissonRegressor models for home and away goals.

    Args:
        X_train: Feature matrix for training.
        y_home: Target home goals series.
        y_away: Target away goals series.

    Returns:
        Tuple of (home_model, away_model), both fitted PoissonRegressors.
    """
    home_model = Pipeline([
        ("scaler", StandardScaler()),
        ("poisson", PoissonRegressor(max_iter=500)),
    ])
    home_model.fit(X_train, y_home)

    away_model = Pipeline([
        ("scaler", StandardScaler()),
        ("poisson", PoissonRegressor(max_iter=500)),
    ])
    away_model.fit(X_train, y_away)

    return home_model, away_model


def evaluate_goals_model(home_model, away_model, X_val, y_home_val, y_away_val):
    """Evaluate goal prediction models on a validation set.

    Args:
        home_model: Fitted PoissonRegressor for home goals.
        away_model: Fitted PoissonRegressor for away goals.
        X_val: Validation feature matrix.
        y_home_val: True home goals for validation set.
        y_away_val: True away goals for validation set.

    Returns:
        Dict with keys: mae_home, rmse_home, mae_away, rmse_away, scoreline_accuracy.
    """
    home_pred = home_model.predict(X_val)
    away_pred = away_model.predict(X_val)

    print("\n=== Goals Model Evaluation ===")
    print(f"  All home predictions >= 0: {home_pred.min() >= 0}")
    print(f"  All away predictions >= 0: {away_pred.min() >= 0}")

    mae_home = mean_absolute_error(y_home_val, home_pred)
    rmse_home = np.sqrt(mean_squared_error(y_home_val, home_pred))
    mae_away = mean_absolute_error(y_away_val, away_pred)
    rmse_away = np.sqrt(mean_squared_error(y_away_val, away_pred))

    print(f"  MAE  (home goals): {mae_home:.4f}")
    print(f"  RMSE (home goals): {rmse_home:.4f}")
    print(f"  MAE  (away goals): {mae_away:.4f}")
    print(f"  RMSE (away goals): {rmse_away:.4f}")

    home_rounded = np.round(home_pred).astype(int)
    away_rounded = np.round(away_pred).astype(int)
    correct = (home_rounded == y_home_val.to_numpy()) & (away_rounded == y_away_val.to_numpy())
    scoreline_acc = correct.mean()
    print(f"  Exact scoreline accuracy: {scoreline_acc * 100:.1f}%")

    return {
        "mae_home": mae_home,
        "rmse_home": rmse_home,
        "mae_away": mae_away,
        "rmse_away": rmse_away,
        "scoreline_accuracy": scoreline_acc,
    }


def train_tuned_goals_models(X_train, y_home, y_away, best_params):
    """Retrain XGBoost and Poisson goals models with tuned hyperparameters.

    Trains two XGBRegressor models using tuned params from best_params, and
    two PoissonRegressor pipelines as a fallback.

    Args:
        X_train: Training feature DataFrame.
        y_home: Target home goals series.
        y_away: Target away goals series.
        best_params: Dict loaded from best_hyperparams.json. Must contain
            'xgb_home_goals' and 'xgb_away_goals' keys.

    Returns:
        Tuple of (xgb_home, xgb_away, poisson_home, poisson_away).
    """
    print("\n=== Training Tuned XGBoost Home Goals ===")
    hp = best_params["xgb_home_goals"]
    xgb_home = XGBRegressor(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        reg_alpha=hp["reg_alpha"],
        reg_lambda=hp["reg_lambda"],
        random_state=42,
    )
    xgb_home.fit(X_train, y_home)

    print("\n=== Training Tuned XGBoost Away Goals ===")
    ap = best_params["xgb_away_goals"]
    xgb_away = XGBRegressor(
        n_estimators=ap["n_estimators"],
        max_depth=ap["max_depth"],
        learning_rate=ap["learning_rate"],
        subsample=ap["subsample"],
        colsample_bytree=ap["colsample_bytree"],
        reg_alpha=ap["reg_alpha"],
        reg_lambda=ap["reg_lambda"],
        random_state=42,
    )
    xgb_away.fit(X_train, y_away)

    print("\n=== Training Poisson Fallback Goals Models ===")
    poisson_home, poisson_away = train_poisson_goals(X_train, y_home, y_away)

    return xgb_home, xgb_away, poisson_home, poisson_away


if __name__ == "__main__":
    from src.models.outcome_model import load_splits

    X_train, y_train, w_train, X_val, y_val, X_test, y_test = load_splits()

    df = pd.read_parquet(_PROCESSED_DIR / "features_train.parquet")
    y_home_train = df.loc[X_train.index, "home_score"]
    y_away_train = df.loc[X_train.index, "away_score"]
    y_home_val = df.loc[X_val.index, "home_score"]
    y_away_val = df.loc[X_val.index, "away_score"]

    print("\n=== Training Poisson Goals Models ===")
    home_model, away_model = train_poisson_goals(X_train, y_home_train, y_away_train)

    metrics = evaluate_goals_model(home_model, away_model, X_val, y_home_val, y_away_val)

    print("\n=== Metrics Dict ===")
    print(metrics)
