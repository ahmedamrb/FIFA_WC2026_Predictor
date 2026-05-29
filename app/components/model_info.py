"""Model information UI component."""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from components.tooltips import TOOLTIPS

# Make src/ importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from src.data.preprocess import FEATURE_COLUMNS


def render_feature_importance(xgb_model) -> None:
    """Render a horizontal bar chart of top-20 XGBoost feature importances.

    Args:
        xgb_model: A fitted XGBClassifier trained with FEATURE_COLUMNS.
    """
    importances = pd.Series(xgb_model.feature_importances_, index=FEATURE_COLUMNS)
    top20 = importances.sort_values(ascending=False).head(20).sort_values(ascending=True)

    fig = go.Figure(
        go.Bar(
            x=top20.values,
            y=top20.index,
            orientation="h",
            marker_color="#1f77b4",
        )
    )
    fig.update_layout(
        title="Top 20 Feature Importances (XGBoost)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=600,
        margin={"l": 200, "r": 40, "t": 60, "b": 60},
    )
    st.plotly_chart(fig, width='stretch')
    st.caption(TOOLTIPS["feature_importance"])


def render_training_summary(features_train_df: pd.DataFrame) -> None:
    """Render a summary table for the training dataset.

    Args:
        features_train_df: The loaded features_train.parquet DataFrame.
            Must contain columns: date, tournament, outcome, home_score, away_score,
            and all FEATURE_COLUMNS.
    """
    df = features_train_df
    min_date = pd.to_datetime(df["date"]).min().strftime("%Y-%m-%d")
    max_date = pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d")

    top_tournaments = df["tournament"].value_counts().head(10)

    rows = [
        ("Total training matches", len(df)),
        ("Earliest match", min_date),
        ("Latest match", max_date),
    ]
    for tournament, count in top_tournaments.items():
        rows.append((f"  {tournament}", count))

    summary_df = pd.DataFrame(rows, columns=["Metric", "Value"])
    summary_df["Value"] = summary_df["Value"].astype(str)
    st.caption("Training data covers international matches from 1998 onward, excluding WC 2018 (test) and WC 2022 (validation) windows. Match importance weights: WC ×3, qualifiers ×1.5, friendlies ×0.5.")
    st.dataframe(summary_df, width='stretch', hide_index=True)


def render_model_registry() -> None:
    """Parse and render the MODEL_REGISTRY.md table as a Streamlit dataframe."""
    registry_path = _REPO_ROOT / "models" / "MODEL_REGISTRY.md"
    if not registry_path.exists():
        st.warning(f"MODEL_REGISTRY.md not found at {registry_path}")
        return

    content = registry_path.read_text(encoding="utf-8")

    columns = ["Model File", "Date Trained", "Type", "Val Log-Loss", "Val Accuracy", "Val Brier", "Notes"]
    data_rows = []

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        # Skip separator rows (only dashes, pipes, and spaces)
        if not stripped.replace("|", "").replace("-", "").replace(" ", ""):
            continue
        cells = [c.strip() for c in stripped.split("|")[1:-1]]
        if len(cells) != 7:
            continue
        # Skip the header row
        if cells[0] == "Model File":
            continue
        data_rows.append(cells)

    if not data_rows:
        st.info("No model entries found in MODEL_REGISTRY.md.")
        return

    registry_df = pd.DataFrame(data_rows, columns=columns)
    registry_df = registry_df[registry_df["Date Trained"] != "N/A"].reset_index(drop=True)

    st.caption(
        "Val Log-Loss and Val Accuracy are measured on the WC 2022 validation set. "
        "Val Brier is mean squared probability error. Lower log-loss and Brier, "
        "higher accuracy = better."
    )
    st.dataframe(registry_df, width='stretch', hide_index=True)

