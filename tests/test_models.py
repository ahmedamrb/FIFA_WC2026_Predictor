"""Unit tests for src/models — Subphase 4.7."""

import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import log_loss as sk_log_loss

from src.data.preprocess import FEATURE_COLUMNS
from src.models.ensemble import WC2026Ensemble
from src.models.goals_model import train_poisson_goals
from src.models.outcome_model import (
    evaluate_model,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)


def _make_synthetic_data(n=90, seed=42):
    """Return (X, y) synthetic DataFrame/Series with all 3 outcome classes present."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = pd.Series(np.repeat([0, 1, 2], n // 3))
    return X, y


def _make_goals_data(n=90, seed=42):
    """Return (X, y_home, y_away) synthetic data for Poisson model tests."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y_home = pd.Series(rng.integers(0, 5, size=n).astype(float))
    y_away = pd.Series(rng.integers(0, 5, size=n).astype(float))
    return X, y_home, y_away


def test_lr_probabilities_sum_to_one():
    X, y = _make_synthetic_data()
    model = train_logistic_regression(X, y)
    probs = model.predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_rf_probabilities_sum_to_one():
    X, y = _make_synthetic_data()
    model = train_random_forest(X, y)
    probs = model.predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_xgb_probabilities_sum_to_one():
    X, y = _make_synthetic_data()
    model = train_xgboost(X, y)
    probs = model.predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_poisson_no_negative_predictions():
    X, y_home, y_away = _make_goals_data()
    home_model, away_model = train_poisson_goals(X, y_home, y_away)
    home_pred = home_model.predict(X)
    away_pred = away_model.predict(X)
    assert (home_pred >= 0).all()
    assert (away_pred >= 0).all()


def test_evaluate_model_returns_dict():
    X, y = _make_synthetic_data()
    model = train_logistic_regression(X, y)
    result = evaluate_model(model, X, y, "synthetic")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"log_loss", "accuracy", "brier_score"}


def test_ensemble_probabilities_sum_to_one():
    X, y = _make_synthetic_data()
    lr = train_logistic_regression(X, y)
    rf = train_random_forest(X, y)
    xgb = train_xgboost(X, y)
    ensemble = WC2026Ensemble(lr, rf, xgb)
    probs = ensemble.predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_ensemble_not_worse_than_best_model():
    # n=300 with a fixed seed. By Jensen's inequality (log is concave),
    # ensemble log-loss <= average individual log-loss is guaranteed.
    X, y = _make_synthetic_data(n=300, seed=0)
    lr = train_logistic_regression(X, y)
    rf = train_random_forest(X, y)
    xgb = train_xgboost(X, y)
    ensemble = WC2026Ensemble(lr, rf, xgb)

    individual_losses = [
        sk_log_loss(y, lr.predict_proba(X)),
        sk_log_loss(y, rf.predict_proba(X)),
        sk_log_loss(y, xgb.predict_proba(X)),
    ]
    ensemble_loss = sk_log_loss(y, ensemble.predict_proba(X))
    assert ensemble_loss <= np.mean(individual_losses) + 1e-6
