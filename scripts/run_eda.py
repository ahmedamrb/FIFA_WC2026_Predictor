"""EDA entry point for the FIFA WC 2026 Predictor project.

Runs all exploratory data analysis sections in sequence and saves
all plots to the outputs/plots/ directory.

Usage:
    python scripts/run_eda.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd

# Repo root is one level above this script
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
OUTPUT_DIR = REPO_ROOT / "outputs" / "plots"


def eda_results():
    """Analyse the raw results dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_RAW / "results.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # --- Shape & types ---
    print("=== Results Dataset EDA ===")
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nNull counts per column (raw):")
    print(df.isnull().sum().to_string())

    # Identify null-score rows (future/unplayed fixtures)
    future_rows = df[df["home_score"].isnull()]
    if not future_rows.empty:
        print(
            f"\nNOTE: {len(future_rows)} rows have null scores — future/unplayed fixtures "
            f"(date range: {future_rows['date'].min().date()} "
            f"→ {future_rows['date'].max().date()}). "
            "Excluded from outcome and goals analysis."
        )

    # Working set: rows with valid (non-null) scores
    df_scored = df.dropna(subset=["home_score", "away_score"]).copy()
    df_scored["home_score"] = df_scored["home_score"].astype(int)
    df_scored["away_score"] = df_scored["away_score"].astype(int)
    scored_total = len(df_scored)
    print(f"\nNull counts for scored rows ({scored_total:,} rows used in analysis):")
    print(df_scored.isnull().sum().to_string())

    # --- Date range ---
    print(f"\nDate range (all rows): {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Date range (scored)  : {df_scored['date'].min().date()} → {df_scored['date'].max().date()}")

    # --- Tournament frequency ---
    print("\nTop 20 tournaments by match count:")
    print(df_scored["tournament"].value_counts().head(20).to_string())

    # --- WC matches per year ---
    wc = df_scored[df_scored["tournament"] == "FIFA World Cup"].copy()
    wc["year"] = wc["date"].dt.year
    print("\nFIFA World Cup match counts per year:")
    print(wc.groupby("year").size().to_string())

    # --- Outcome percentages (all scored matches) ---
    home_wins = (df_scored["home_score"] > df_scored["away_score"]).sum()
    draws = (df_scored["home_score"] == df_scored["away_score"]).sum()
    away_wins = (df_scored["home_score"] < df_scored["away_score"]).sum()
    print(f"\nAll scored matches ({scored_total:,}):")
    print(f"  Home wins : {home_wins / scored_total * 100:.1f}%")
    print(f"  Draws     : {draws / scored_total * 100:.1f}%")
    print(f"  Away wins : {away_wins / scored_total * 100:.1f}%")

    # --- Outcome percentages (WC only) ---
    wc_total = len(wc)
    wc_home = (wc["home_score"] > wc["away_score"]).sum()
    wc_draw = (wc["home_score"] == wc["away_score"]).sum()
    wc_away = (wc["home_score"] < wc["away_score"]).sum()
    print(f"\nWC scored matches ({wc_total:,}):")
    print(f"  Home wins : {wc_home / wc_total * 100:.1f}%")
    print(f"  Draws     : {wc_draw / wc_total * 100:.1f}%")
    print(f"  Away wins : {wc_away / wc_total * 100:.1f}%")

    # --- Average goals ---
    df_scored["total_goals"] = df_scored["home_score"] + df_scored["away_score"]
    wc["total_goals"] = wc["home_score"] + wc["away_score"]
    print(f"\nAverage goals per match (all scored) : {df_scored['total_goals'].mean():.3f}")
    print(f"Average goals per match (WC only)    : {wc['total_goals'].mean():.3f}")

    # --- Duplicate rows ---
    dup_count = df_scored.duplicated(subset=["date", "home_team", "away_team"]).sum()
    print(f"\nDuplicate rows in scored data (date + home + away): {dup_count}")
    if dup_count > 0:
        dup_rows = df_scored[
            df_scored.duplicated(subset=["date", "home_team", "away_team"], keep=False)
        ]
        print(dup_rows[["date", "home_team", "away_team", "home_score",
                        "away_score", "tournament"]].to_string())
        print("  NOTE: Likely data-entry duplicates in the source dataset.")

    # --- Score sanity check ---
    bad_scores = df_scored[
        (df_scored["home_score"] < 0) | (df_scored["away_score"] < 0)
        | (df_scored["home_score"] > 20) | (df_scored["away_score"] > 20)
    ]
    if bad_scores.empty:
        print("\nNo rows with negative or >20 scores found.")
    else:
        print(
            f"\nRows with score outside [0, 20] ({len(bad_scores)} found — "
            "may be legitimate historic results):"
        )
        print(bad_scores[["date", "home_team", "away_team", "home_score",
                           "away_score", "tournament"]].to_string())

    # --- Plot 1: match counts per decade ---
    df_scored["decade"] = (df_scored["date"].dt.year // 10) * 10
    decade_counts = df_scored.groupby("decade").size()

    fig, ax = plt.subplots(figsize=(10, 5))
    decade_counts.plot(kind="bar", ax=ax, color="#1f77b4", edgecolor="black")
    ax.set_title("International Match Counts by Decade")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Number of Matches")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "results_by_decade.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # --- Plot 2: score distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df_scored["home_score"].value_counts().sort_index().plot(
        kind="bar", ax=axes[0], color="#2ca02c", edgecolor="black"
    )
    axes[0].set_title("Home Score Distribution")
    axes[0].set_xlabel("Goals")
    axes[0].set_ylabel("Frequency")
    axes[0].tick_params(axis="x", rotation=0)

    df_scored["away_score"].value_counts().sort_index().plot(
        kind="bar", ax=axes[1], color="#d62728", edgecolor="black"
    )
    axes[1].set_title("Away Score Distribution")
    axes[1].set_xlabel("Goals")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    out_path2 = OUTPUT_DIR / "score_distribution.png"
    fig.savefig(out_path2, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path2}")


def eda_rankings():
    """Analyse the FIFA rankings dataset."""
    pass


def eda_fixtures():
    """Analyse the WC 2026 fixtures."""
    pass


def eda_correlations():
    """Explore cross-dataset correlations."""
    pass


def main():
    """Run all EDA sections in sequence."""
    eda_results()
    eda_rankings()
    eda_fixtures()
    eda_correlations()


if __name__ == "__main__":
    main()
