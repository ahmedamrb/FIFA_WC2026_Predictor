"""Performance charts UI component for the Model Performance page."""

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from components.tooltips import TOOLTIPS


def render_metrics_bar_chart(metrics_dict):
    """Render a grouped bar chart of log-loss, accuracy, and Brier score for WC 2018 and WC 2022.

    Args:
        metrics_dict: Dict with keys "wc2018" and "wc2022", each containing
            "log_loss", "accuracy", "brier_score", "flat_stake_roi", "value_bet_roi".
    """
    categories = ["Log-loss", "Accuracy", "Brier Score"]

    wc2018 = metrics_dict["wc2018"]
    wc2022 = metrics_dict["wc2022"]

    fig = go.Figure()
    direction_hints = ["lower is better", "higher is better", "lower is better"]

    fig.add_trace(go.Bar(
        name="WC 2018",
        x=categories,
        y=[wc2018["log_loss"], wc2018["accuracy"], wc2018["brier_score"]],
        customdata=direction_hints,
        hovertemplate="%{x} (WC 2018): %{y:.4f} — %{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="WC 2022",
        x=categories,
        y=[wc2022["log_loss"], wc2022["accuracy"], wc2022["brier_score"]],
        customdata=direction_hints,
        hovertemplate="%{x} (WC 2022): %{y:.4f} — %{customdata}<extra></extra>",
    ))

    fig.update_layout(
        barmode="group",
        title="Model Performance by Tournament",
        xaxis_title="Metric",
        yaxis_title="Value",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bars show backtest results on WC 2018 (held-out test) and WC 2022 (validation). Hover each bar for the exact value.")


def render_cumulative_profit_chart(wc2018_df, wc2022_df):
    """Render a dual-line chart of cumulative flat-stake profit over match sequence.

    Args:
        wc2018_df: DataFrame with a `cumulative_profit` column for WC 2018.
        wc2022_df: DataFrame with a `cumulative_profit` column for WC 2022.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(wc2022_df))),
        y=wc2022_df["cumulative_profit"].tolist(),
        mode="lines+markers",
        name="WC 2022",
        hovertemplate="Match #%{x}<br>Cumulative Profit: %{y:+.2f} units<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(wc2018_df))),
        y=wc2018_df["cumulative_profit"].tolist(),
        mode="lines+markers",
        name="WC 2018",
        hovertemplate="Match #%{x}<br>Cumulative Profit: %{y:+.2f} units<extra></extra>",
    ))

    # Horizontal reference line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="grey")

    fig.update_layout(
        title="Cumulative Flat-stake Profit",
        xaxis_title="Match #",
        yaxis_title="Cumulative Profit (units)",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Flat-stake simulation: one unit bet on the model's top-probability outcome for every match. Profit/loss accumulates over the tournament.")


def render_calibration_chart():
    """Display pre-saved calibration PNG images from outputs/plots/."""
    repo_root = Path(__file__).resolve().parents[2]
    plots_dir = repo_root / "outputs" / "plots"

    filenames = [
        "calibration_rf_tuned.png",
        "calibration_ensemble.png",
        "calibration_ensemble_calibrated.png",
    ]

    found_any = False
    for filename in filenames:
        path = plots_dir / filename
        if path.exists():
            found_any = True
            caption = path.stem  # filename without extension
            with st.expander(caption):
                st.image(str(path), caption=caption, use_container_width=True)
                st.caption("A well-calibrated model's curve follows the diagonal. Points above the diagonal mean the model underestimates probability; points below mean overconfidence.")

    if not found_any:
        st.info("No calibration charts found.")
