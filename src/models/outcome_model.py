"""Match outcome classification model for FIFA WC 2026 Predictor.

Provides data splitting, training, and evaluation for the three-class
outcome prediction: 0 = away win, 1 = draw, 2 = home win.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.preprocess import FEATURE_COLUMNS

_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_splits():
    """Load features_train.parquet and return train/val/test splits.

    Splits:
        - test  : WC 2018 matches (tournament == "FIFA World Cup", year == 2018)
        - val   : WC 2022 matches (tournament == "FIFA World Cup", year == 2022)
        - train : all remaining rows (1998+, no WC 2018 or WC 2022)

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test as pandas DataFrames/Series.
    """
    df = pd.read_parquet(_PROCESSED_DIR / "features_train.parquet")
    df["date"] = pd.to_datetime(df["date"])

    wc2018_mask = (df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2018)
    wc2022_mask = (df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2022)
    train_mask = ~wc2018_mask & ~wc2022_mask

    train_df = df[train_mask]
    val_df = df[wc2022_mask]
    test_df = df[wc2018_mask]

    for label, split in [("train", train_df), ("val (WC 2022)", val_df), ("test (WC 2018)", test_df)]:
        print(f"\n--- {label} split: {len(split)} rows ---")
        print(f"  Date range: {split['date'].min().date()} → {split['date'].max().date()}")
        print(f"  Outcome distribution:\n{split['outcome'].value_counts().sort_index().to_string()}")

    if len(val_df) != 64:
        print(f"WARNING: val set has {len(val_df)} rows, expected 64.")
    if len(test_df) != 64:
        print(f"WARNING: test set has {len(test_df)} rows, expected 64.")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["outcome"]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["outcome"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["outcome"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_logistic_regression(X_train, y_train):
    """Fit a StandardScaler + LogisticRegression pipeline on training data.

    The scaler is fitted exclusively on X_train inside the pipeline, ensuring
    no data leakage to the validation or test sets.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training outcome Series (0/1/2).

    Returns:
        Fitted sklearn Pipeline (scaler → logistic regression).
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(X_train, y_train):
    """Fit a RandomForestClassifier with default parameters on training data.

    No scaling is applied — tree-based models are scale-invariant.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training outcome Series (0/1/2).

    Returns:
        Fitted RandomForestClassifier.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Fit an XGBClassifier with default parameters on training data.

    Uses multi:softprob objective for 3-class probability output.
    No scaling applied — gradient boosting is scale-invariant.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training outcome Series (0/1/2).

    Returns:
        Fitted XGBClassifier.
    """
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, label):
    """Evaluate a fitted classifier and print key metrics.

    Args:
        model: Fitted sklearn-compatible classifier with predict_proba.
        X: Feature DataFrame.
        y: Ground-truth outcome Series (0/1/2).
        label: Human-readable label for the printed output.

    Returns:
        dict with keys "log_loss", "accuracy", "brier_score".
    """
    probs = model.predict_proba(X)

    row_sums = probs.sum(axis=1)
    probs_sum_to_one = np.allclose(row_sums, 1.0)
    print(f"\n=== {label} ===")
    print(f"  Prob sums to 1.0: {probs_sum_to_one}  (min={row_sums.min():.6f}, max={row_sums.max():.6f})")

    ll = log_loss(y, probs)
    acc = accuracy_score(y, model.predict(X))
    brier = float(np.mean(np.sum((probs - np.eye(3)[y.to_numpy()]) ** 2, axis=1)))

    print(f"  Log-loss : {ll:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Brier    : {brier:.4f}")
    print("  Confusion matrix:")
    print(confusion_matrix(y, model.predict(X)))

    return {"log_loss": ll, "accuracy": acc, "brier_score": brier}


def train_tuned_models(X_train, y_train, best_params):
    """Retrain LR, RF, and XGBoost on the full training set using tuned hyperparameters.

    LR is retrained with default parameters (no tuning was performed for LR).
    RF and XGBoost use their respective entries from best_params.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training outcome Series (0/1/2).
        best_params: Dict loaded from best_hyperparams.json. Must contain
            'xgb_outcome' and 'rf_outcome' keys.

    Returns:
        Tuple of (lr_model, rf_model, xgb_model).
    """
    print("\n=== Training Tuned Logistic Regression ===")
    lr_model = train_logistic_regression(X_train, y_train)

    print("\n=== Training Tuned Random Forest ===")
    rp = best_params["rf_outcome"]
    rf_model = RandomForestClassifier(
        n_estimators=rp["n_estimators"],
        max_features=rp["max_features"],
        min_samples_split=rp["min_samples_split"],
        min_samples_leaf=rp["min_samples_leaf"],
        max_depth=rp["max_depth"],
        random_state=42,
    )
    rf_model.fit(X_train, y_train)

    print("\n=== Training Tuned XGBoost ===")
    xp = best_params["xgb_outcome"]
    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=xp["n_estimators"],
        max_depth=xp["max_depth"],
        learning_rate=xp["learning_rate"],
        subsample=xp["subsample"],
        colsample_bytree=xp["colsample_bytree"],
        reg_alpha=xp["reg_alpha"],
        reg_lambda=xp["reg_lambda"],
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)

    return lr_model, rf_model, xgb_model


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    print("\n--- Shapes ---")
    print(f"X_train: {X_train.shape},  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape},  y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    print("\n=== Training Logistic Regression ===")
    lr_model = train_logistic_regression(X_train, y_train)

    val_metrics = evaluate_model(lr_model, X_val, y_val, "LR — Val (WC 2022)")
    test_metrics = evaluate_model(lr_model, X_test, y_test, "LR — Test (WC 2018)")

    results_dict = {"LR_val": val_metrics, "LR_test": test_metrics}

    # --- Random Forest ---
    print("\n=== Training Random Forest ===")
    rf_model = train_random_forest(X_train, y_train)

    rf_val_metrics = evaluate_model(rf_model, X_val, y_val, "RF — Val (WC 2022)")
    rf_test_metrics = evaluate_model(rf_model, X_test, y_test, "RF — Test (WC 2018)")

    # Feature importance table — top 15
    importance_df = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": rf_model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    importance_df.index += 1  # 1-based rank
    print("\n=== RF Feature Importances (Top 15) ===")
    print(importance_df.to_string())

    results_dict["RF_val"] = rf_val_metrics
    results_dict["RF_test"] = rf_test_metrics

    # --- XGBoost ---
    print("\n=== Training XGBoost ===")
    xgb_model = train_xgboost(X_train, y_train)

    xgb_val_metrics = evaluate_model(xgb_model, X_val, y_val, "XGB — Val (WC 2022)")
    xgb_test_metrics = evaluate_model(xgb_model, X_test, y_test, "XGB — Test (WC 2018)")

    # Feature importance table — top 15
    xgb_importance_df = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": xgb_model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    xgb_importance_df.index += 1  # 1-based rank
    print("\n=== XGB Feature Importances (Top 15) ===")
    print(xgb_importance_df.to_string())

    results_dict["XGB_val"] = xgb_val_metrics
    results_dict["XGB_test"] = xgb_test_metrics

    print("\n=== Results Dict ===")
    print(results_dict)
