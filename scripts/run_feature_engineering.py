"""Feature engineering pipeline script.

Runs the full feature engineering pipeline and exports
features_train.parquet and features_predict.parquet to data/processed/.

Usage:
    python scripts/run_feature_engineering.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import export_features


def main() -> None:
    export_features()


if __name__ == "__main__":
    main()
