"""Match outcome classification model for FIFA WC 2026 Predictor.

Provides data splitting, training, and evaluation for the three-class
outcome prediction: 0 = away win, 1 = draw, 2 = home win.
"""

from pathlib import Path

import pandas as pd

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


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    print("\n--- Shapes ---")
    print(f"X_train: {X_train.shape},  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape},  y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")
