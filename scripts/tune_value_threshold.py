"""Sweep the value-bet edge threshold against the real-odds backtest.

Reads the enriched backtest CSVs (``data/processed/backtest_wc{2018,2022}.csv``
produced by ``scripts/run_backtest.py`` in real-odds mode) and reports, for a
grid of candidate thresholds, how many value bets are flagged, their ROI
(profit per unit staked), and the worst peak-to-trough drawdown.

It then recommends the threshold whose flagged-bet **count stays closest to the
current count** (keep-volume preference) while maximising ROI, and prints the
one-line edits to apply it.  This script is report-only — it never edits code.

Usage:
    python scripts/tune_value_threshold.py
    python scripts/tune_value_threshold.py --grid 0.02,0.04,0.06,0.08,0.10,0.12
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.betting.edge import VALUE_THRESHOLD  # noqa: E402

_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
_CSVS = [_PROCESSED / "backtest_wc2018.csv", _PROCESSED / "backtest_wc2022.csv"]
_DEFAULT_GRID = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]


def _load() -> pd.DataFrame:
    frames = []
    for path in _CSVS:
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            print(f"  WARN: {path.name} not found — run scripts/run_backtest.py first.")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Only fixtures priced off real odds carry a non-null best_edge.
    return df[df["best_edge"].notna()].copy()


def _max_drawdown(profit_series: pd.Series) -> float:
    """Worst peak-to-trough drop of cumulative profit (units)."""
    if profit_series.empty:
        return 0.0
    cum = profit_series.cumsum().to_numpy()
    running_max = np.maximum.accumulate(cum)
    return float((cum - running_max).min())


def _stats_at(df: pd.DataFrame, threshold: float) -> dict:
    flagged = df[df["best_edge"] > threshold].sort_values("match_date")
    count = len(flagged)
    stake = float(flagged["stake"].sum()) if "stake" in flagged.columns else float(count)
    profit = float(flagged["profit"].sum())
    roi = profit / stake * 100.0 if stake > 0 else float("nan")
    return {
        "threshold": threshold,
        "count": count,
        "roi": roi,
        "profit": profit,
        "max_drawdown": _max_drawdown(flagged["profit"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the value-bet edge threshold.")
    parser.add_argument(
        "--grid",
        type=lambda s: [float(x) for x in s.split(",")],
        default=_DEFAULT_GRID,
        help="Comma-separated thresholds to sweep.",
    )
    parser.add_argument(
        "--volume-band",
        type=float,
        default=0.25,
        help="Allowed fractional deviation from current bet count when picking (default 0.25).",
    )
    args = parser.parse_args()

    df = _load()
    if df.empty:
        print("\nNo real-odds backtest rows available. Source odds + run the backtest:")
        print("  python scripts/fetch_historical_odds.py")
        print("  python scripts/run_backtest.py")
        sys.exit(0)

    grid = sorted(set(args.grid + [VALUE_THRESHOLD]))
    rows = [_stats_at(df, t) for t in grid]

    print(f"\n  Real-odds backtest rows (with odds): {len(df)}")
    print(f"  Current VALUE_THRESHOLD = {VALUE_THRESHOLD:.3f}\n")
    print(f"  {'threshold':>9}  {'bets':>5}  {'ROI %':>8}  {'profit':>8}  {'max DD':>8}")
    print("  " + "-" * 48)
    for r in rows:
        marker = "  <- current" if abs(r["threshold"] - VALUE_THRESHOLD) < 1e-9 else ""
        print(
            f"  {r['threshold']:>9.3f}  {r['count']:>5d}  {r['roi']:>8.2f}  "
            f"{r['profit']:>+8.2f}  {r['max_drawdown']:>+8.2f}{marker}"
        )

    # Keep-volume pick: among thresholds whose count is within the band of the
    # current count, choose the one with the best ROI.
    current = next(r for r in rows if abs(r["threshold"] - VALUE_THRESHOLD) < 1e-9)
    target = current["count"]
    if target == 0:
        print("\n  Current threshold flags 0 bets — widen odds coverage before tuning.")
        return

    lo, hi = target * (1 - args.volume_band), target * (1 + args.volume_band)
    candidates = [r for r in rows if lo <= r["count"] <= hi and not np.isnan(r["roi"])]
    if not candidates:
        candidates = [r for r in rows if not np.isnan(r["roi"])]
    best = max(candidates, key=lambda r: r["roi"])

    print("\n  " + "=" * 48)
    print(f"  Recommended threshold: {best['threshold']:.3f}  "
          f"(bets={best['count']}, ROI={best['roi']:+.2f}%, maxDD={best['max_drawdown']:+.2f})")
    print(f"  Keep-volume target was ~{target} bets (+/-{args.volume_band:.0%}).")
    if abs(best["threshold"] - VALUE_THRESHOLD) < 1e-9:
        print("  → Current threshold is already the best in-band choice. No change needed.")
    else:
        print("  To apply, set in src/betting/edge.py:")
        print(f"      VALUE_THRESHOLD = {best['threshold']:.3f}")
        print("  (the dashboard reads this via prediction_card.py automatically).")
    print("  " + "=" * 48)


if __name__ == "__main__":
    main()
