"""EDA entry point for the FIFA WC 2026 Predictor project.

Runs all exploratory data analysis sections in sequence and saves
all plots to the outputs/plots/ directory.

Usage:
    python scripts/run_eda.py
"""

OUTPUT_DIR = "outputs/plots/"


def eda_results():
    """Analyse the raw results dataset."""
    pass


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
