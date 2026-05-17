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
import numpy as np
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
            f"\nNOTE: {len(future_rows)} rows have null scores - future/unplayed fixtures "
            f"(date range: {future_rows['date'].min().date()} "
            f"to {future_rows['date'].max().date()}). "
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
    print(f"\nDate range (all rows): {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Date range (scored)  : {df_scored['date'].min().date()} to {df_scored['date'].max().date()}")

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
            f"\nRows with score outside [0, 20] ({len(bad_scores)} found - "
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_RAW / "rankings.csv"
    df = pd.read_csv(csv_path, index_col=0)

    print("=== Rankings Dataset EDA ===")

    # --- Shape & columns ---
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")

    # --- Data types & null counts ---
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nNull counts per column:")
    print(df.isnull().sum().to_string())

    # --- Parse rank_date to datetime ---
    df["rank_date"] = pd.to_datetime(df["rank_date"])
    print(f"\nrank_date dtype after parsing: {df['rank_date'].dtype}")
    print(f"Date range: {df['rank_date'].min().date()} to {df['rank_date'].max().date()}")

    # --- Unique countries ---
    n_countries = df["country_full"].nunique()
    print(f"\nUnique countries: {n_countries}")

    # --- Missing years 2000–2022 ---
    years_present = set(df["rank_date"].dt.year.unique())
    missing_years = [y for y in range(2000, 2023) if y not in years_present]
    if missing_years:
        print(f"\nMissing years between 2000 and 2022: {missing_years}")
    else:
        print("\nNo missing years between 2000 and 2022")

    # --- Publication frequency (median gap between consecutive dates for England) ---
    england_dates = (
        df[df["country_full"] == "England"]["rank_date"]
        .drop_duplicates()
        .sort_values()
    )
    gaps = england_dates.diff().dropna().dt.days
    median_gap = gaps.median()
    print(f"\nMedian gap between consecutive ranking dates (England): {median_gap:.0f} days")

    # --- Duplicate entries (same country_full + rank_date) ---
    dup_count = df.duplicated(subset=["country_full", "rank_date"]).sum()
    print(f"\nDuplicate entries (same country + rank_date): {dup_count}")

    # --- Plot 1: ranking history for 5 countries ---
    countries = ["Brazil", "France", "Germany", "Argentina", "England"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for country in countries:
        subset = df[df["country_full"] == country].sort_values("rank_date")
        ax.plot(subset["rank_date"], subset["rank"], label=country, linewidth=1.5)
    ax.invert_yaxis()  # lower rank number = better, so rank 1 should be at top
    ax.set_title("FIFA Ranking History - Top Nations")
    ax.set_xlabel("Date")
    ax.set_ylabel("FIFA Rank (lower = better)")
    ax.legend()
    plt.tight_layout()
    out_path1 = OUTPUT_DIR / "ranking_history.png"
    fig.savefig(out_path1, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path1}")

    # --- Plot 2: total_points histogram ---
    fig, ax = plt.subplots(figsize=(10, 5))
    df["total_points"].dropna().plot(kind="hist", bins=50, ax=ax, color="#1f77b4", edgecolor="black")
    ax.set_title("Distribution of FIFA Ranking Points")
    ax.set_xlabel("Total Points")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    out_path2 = OUTPUT_DIR / "ranking_points_distribution.png"
    fig.savefig(out_path2, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path2}")


def eda_fixtures():
    """Analyse the WC 2026 fixtures dataset."""
    csv_path = DATA_RAW / "wc2026_fixtures_flat.csv"
    df = pd.read_csv(csv_path, parse_dates=["match_date"])

    print("=== WC 2026 Fixtures EDA ===")

    # Total fixture count
    print(f"\nTotal fixture count: {len(df)}")

    # Unique stage names
    print(f"\nUnique stages ({df['stage'].nunique()}):")
    print(df["stage"].value_counts().to_string())

    # Date range
    print(f"\nEarliest fixture date : {df['match_date'].min().date()}")
    print(f"Latest fixture date   : {df['match_date'].max().date()}")

    # Fixture count per stage
    print("\nFixture count per stage:")
    print(df.groupby("stage").size().to_string())

    # All unique team names (from home + away combined)
    all_teams = pd.Series(
        pd.concat([df["home_team"], df["away_team"]]).unique()
    ).sort_values().reset_index(drop=True)
    print(f"\nUnique teams ({len(all_teams)}):")
    print(all_teams.to_string())

    # Missing values for key columns
    print("\nMissing values in key columns:")
    for col in ["match_date", "home_team", "away_team"]:
        print(f"  {col}: {df[col].isnull().sum()}")


def eda_correlations():
    """Explore feature correlations: ranking difference vs match outcome."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Feature Correlation Exploration ===")

    # --- Load data ---
    results = pd.read_csv(DATA_RAW / "results.csv", parse_dates=["date"])
    rankings = pd.read_csv(DATA_RAW / "rankings.csv", index_col=0)
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])
    name_map_df = pd.read_csv(DATA_RAW / "team_name_map.csv")

    # Build results_name -> rankings_name mapping
    results_to_rankings = dict(
        zip(name_map_df["results_name"], name_map_df["rankings_name"])
    )

    # --- Filter WC matches 1998–2022 with valid scores ---
    wc = results[
        (results["tournament"] == "FIFA World Cup")
        & (results["date"].dt.year >= 1998)
        & (results["date"].dt.year <= 2022)
        & results["home_score"].notna()
        & results["away_score"].notna()
    ].copy()
    wc["home_score"] = wc["home_score"].astype(int)
    wc["away_score"] = wc["away_score"].astype(int)
    wc = wc.sort_values("date").reset_index(drop=True)
    print(f"\nWC matches 1998\u20132022 with valid scores: {len(wc)}")

    # --- Apply name mapping ---
    wc["home_team_rk"] = wc["home_team"].map(lambda x: results_to_rankings.get(x, x))
    wc["away_team_rk"] = wc["away_team"].map(lambda x: results_to_rankings.get(x, x))

    # --- Prepare rankings for asof merge ---
    rk = rankings.dropna(subset=["rank", "rank_date"]).sort_values("rank_date").copy()
    median_rank = rk["rank"].median()

    # --- Merge home team rankings (most recent on or before match date) ---
    home_lookup = wc[["date", "home_team_rk"]].rename(columns={"home_team_rk": "country_full"})
    home_lookup = home_lookup.sort_values("date")
    home_merged = pd.merge_asof(
        home_lookup,
        rk[["rank_date", "country_full", "rank"]].rename(columns={"rank": "home_rank"}),
        left_on="date",
        right_on="rank_date",
        by="country_full",
    )
    wc["home_rank"] = home_merged["home_rank"].values

    # --- Merge away team rankings ---
    away_lookup = wc[["date", "away_team_rk"]].rename(columns={"away_team_rk": "country_full"})
    away_lookup = away_lookup.sort_values("date")
    away_merged = pd.merge_asof(
        away_lookup,
        rk[["rank_date", "country_full", "rank"]].rename(columns={"rank": "away_rank"}),
        left_on="date",
        right_on="rank_date",
        by="country_full",
    )
    wc["away_rank"] = away_merged["away_rank"].values

    # Fill missing rankings with global median
    missing_home = wc["home_rank"].isna().sum()
    missing_away = wc["away_rank"].isna().sum()
    if missing_home or missing_away:
        print(f"\nMissing home rankings filled with median: {missing_home}")
        print(f"Missing away rankings filled with median: {missing_away}")
    wc["home_rank"] = wc["home_rank"].fillna(median_rank)
    wc["away_rank"] = wc["away_rank"].fillna(median_rank)

    # --- Derived columns ---
    wc["rank_diff"] = wc["home_rank"] - wc["away_rank"]
    wc["outcome_binary"] = (wc["home_score"] > wc["away_score"]).astype(int)
    wc["result_label"] = wc.apply(
        lambda r: "Home Win" if r["home_score"] > r["away_score"]
        else ("Draw" if r["home_score"] == r["away_score"] else "Away Win"),
        axis=1,
    )

    # --- Pearson correlation ---
    corr = np.corrcoef(wc["rank_diff"].values, wc["outcome_binary"].values)[0, 1]
    print(f"\nPearson correlation (rank_diff vs home_win): {corr:.4f}")
    print("  (Negative: home team better ranked \u2192 higher win rate)")

    # --- Win-rate table by |rank_diff| bucket ---
    abs_diff = wc["rank_diff"].abs()

    def bucket(d):
        if d < 10:
            return "< 10 (close)"
        elif d <= 50:
            return "10\u201350 (moderate)"
        else:
            return "> 50 (large gap)"

    wc["rank_diff_bucket"] = abs_diff.apply(bucket)
    bucket_order = ["< 10 (close)", "10\u201350 (moderate)", "> 50 (large gap)"]
    win_rate_table = (
        wc.groupby("rank_diff_bucket")["outcome_binary"]
        .agg(home_win_rate="mean", match_count="count")
        .reindex(bucket_order)
    )
    win_rate_table["home_win_rate"] = win_rate_table["home_win_rate"].map("{:.1%}".format)
    print("\nHome win rate by ranking difference bucket (|rank_diff|):")
    print(win_rate_table.to_string())

    # --- Box plot: rank_diff by outcome ---
    groups = [
        wc.loc[wc["result_label"] == label, "rank_diff"].values
        for label in ["Home Win", "Draw", "Away Win"]
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(groups, tick_labels=["Home Win", "Draw", "Away Win"], patch_artist=True,
               boxprops=dict(facecolor="#aec6cf"), medianprops=dict(color="navy", linewidth=2))
    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="rank_diff = 0")
    ax.set_title("Ranking Difference by Match Outcome (WC 1998\u20132022)")
    ax.set_xlabel("Match Outcome")
    ax.set_ylabel("Ranking Difference (home_rank \u2212 away_rank)")
    ax.legend()
    plt.tight_layout()
    out1 = OUTPUT_DIR / "ranking_diff_by_outcome.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out1}")

    # --- Goals histogram ---
    max_goals = max(wc["home_score"].max(), wc["away_score"].max())
    bins = range(0, max_goals + 2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(wc["home_score"], bins=bins, align="left", rwidth=0.8,
                 color="#2ca02c", edgecolor="black")
    axes[0].set_title("Home Goals (WC 1998\u20132022)")
    axes[0].set_xlabel("Goals")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xticks(range(0, max_goals + 1))

    axes[1].hist(wc["away_score"], bins=bins, align="left", rwidth=0.8,
                 color="#d62728", edgecolor="black")
    axes[1].set_title("Away Goals (WC 1998\u20132022)")
    axes[1].set_xlabel("Goals")
    axes[1].set_xticks(range(0, max_goals + 1))

    plt.suptitle("Goals Distribution \u2014 FIFA World Cup 1998\u20132022", fontsize=13)
    plt.tight_layout()
    out2 = OUTPUT_DIR / "wc_goals_distribution.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved: {out2}")


def main():
    """Run all EDA sections in sequence."""
    eda_results()
    eda_rankings()
    eda_fixtures()
    eda_correlations()


if __name__ == "__main__":
    main()
