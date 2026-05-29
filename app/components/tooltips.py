"""Centralised tooltip text for all dashboard metrics and controls."""

TOOLTIPS: dict[str, str] = {
    "outcome_probs": (
        "Soft-voting ensemble of XGBoost, Random Forest, and Logistic Regression. "
        "Each segment shows the model's estimated probability (0–100%) for that match outcome."
    ),
    "scoreline": (
        "Predicted goals from two XGBoost regressors (one per team) trained on form, FIFA rankings, "
        "and head-to-head features. Values are rounded to the nearest integer."
    ),
    "confidence": (
        "Confidence = the highest of the three outcome probabilities. "
        "High ≥ 55%, Medium 45–54%, Low < 45%. Higher confidence means the model strongly favours one outcome."
    ),
    "home_odds": (
        "Bookmaker decimal odds for a Home Win. Implied probability = 1 ÷ odds. "
        "Edit to compare against any bookmaker."
    ),
    "draw_odds": "Bookmaker decimal odds for a Draw. Implied probability = 1 ÷ odds.",
    "away_odds": "Bookmaker decimal odds for an Away Win. Implied probability = 1 ÷ odds.",
    "edge": (
        "Edge = Model probability − Implied probability. Positive means the model thinks this outcome "
        "is underpriced by the bookmaker. 'Best Value' badge triggers when Edge > +5%."
    ),
    "value_bet": (
        "A Value Bet appears when the model's probability exceeds the bookmaker's implied probability "
        "by more than 5 percentage points — suggesting the market has underestimated that outcome."
    ),
    "min_confidence_filter": (
        "Hide fixtures where the model's highest-probability outcome falls below this threshold. "
        "Increase to see only the clearest predictions."
    ),
    "stage_filter": (
        "Filter fixtures by tournament stage: Group Stage, Round of 32, Round of 16, "
        "Quarter Finals, Semi Finals, or Final."
    ),
    "date_filter": "Show only fixtures scheduled within this date range.",
    "log_loss": (
        "Log-loss penalises confident wrong predictions heavily. Lower is better. "
        "Target: < 0.95. Averaged over WC 2018 and WC 2022 backtests."
    ),
    "accuracy": (
        "Percentage of matches where the highest-probability outcome matched the actual result. "
        "Higher is better. Target: > 52%. Averaged over WC 2018 and WC 2022."
    ),
    "brier_score": (
        "Mean squared error of the probability estimates across all three outcomes (0 = perfect). "
        "Lower is better. Target: < 0.22. Averaged over WC 2018 and WC 2022."
    ),
    "flat_stake_roi": (
        "Return on Investment (%) when betting one unit on the model's top-probability outcome "
        "for every match. Positive = simulated profit. Averaged over WC 2018 and WC 2022."
    ),
    "value_bet_roi": (
        "ROI (%) when betting only on outcomes where Edge > 5%. "
        "Filters out low-confidence bets to focus on underpriced markets."
    ),
    "expected_pts": (
        "Expected group-stage points = Σ(3 × P(win) + P(draw)) across all 3 group matches, "
        "using the ML ensemble probabilities. Teams are ranked by this value; FIFA ranking breaks ties."
    ),
    "knockout_win_prob": (
        "Knockout round winner probability is estimated by a logistic function on FIFA ranking "
        "difference: σ(Δrank / 25), capped to [50%, 97%]. The box colour indicates win confidence."
    ),
    "feature_importance": (
        "XGBoost 'gain' importance: average reduction in log-loss for tree splits using this feature, "
        "normalised to sum to 1. Higher = more predictive."
    ),
    "last_retrained": (
        "Most recent training date from MODEL_REGISTRY.md. "
        "Re-run scripts/train.py to update the models."
    ),
}
