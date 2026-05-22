"""Unit tests for backtesting output — Subphase 6.6."""

from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKTEST_CSV = _REPO_ROOT / "data" / "processed" / "backtest_wc2022.csv"


def test_backtest_df_no_nulls():
    df = pd.read_csv(_BACKTEST_CSV)
    null_count = df.isnull().sum().sum()
    assert null_count == 0, f"Expected zero nulls, found {null_count}"


def test_probabilities_sum_to_one():
    df = pd.read_csv(_BACKTEST_CSV)
    prob_sums = (
        df["predicted_home_win_prob"]
        + df["predicted_draw_prob"]
        + df["predicted_away_win_prob"]
    )
    assert np.allclose(prob_sums, 1.0, atol=1e-6), (
        "Predicted probabilities do not sum to 1.0 within tolerance 1e-6"
    )
