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
    "actual_value_roi": (
        "Realised return on the model's best value bets (Edge > 5%, priced off real "
        "bookmaker odds) for fixtures that have already finished. ROI = net profit ÷ total "
        "staked, where each bet risks the displayed fractional-Kelly stake on the value "
        "outcome at its odds and is settled against the actual result. Pending bets on "
        "fixtures still to be played are excluded."
    ),
    "value_bets_settled": (
        "How many best-value bets (Edge > 5%) have been settled on finished fixtures, and "
        "how many of those won. Bets on fixtures that have not kicked off yet are pending."
    ),
    "bankroll_pl": (
        "Net profit/loss in dollars across all settled value bets, staking the displayed "
        "fractional-Kelly amount on each value outcome at its odds. Positive = simulated profit."
    ),
    "total_staked": (
        "Total amount risked across all settled value bets "
        "(the sum of each bet's displayed fractional-Kelly stake)."
    ),
    "staking_mode": (
        "Flat: every bet is sized off your fixed starting bankroll — path-independent and the "
        "cleanest read on the model's edge. Compound: winnings are reinvested, so each bet is "
        "sized off the current bankroll (bets settled in date order) — maximises growth but adds "
        "variance and is sensitive to the model's probability estimates."
    ),
    "staking_bankroll": (
        "Starting bankroll used to convert each fractional-Kelly stake into dollars. "
        "Defaults to $1,000."
    ),
    "final_bankroll": (
        "Your bankroll after settling every finished value bet at the displayed stake and odds, "
        "under the selected staking mode. The delta shows the percentage change from the "
        "starting bankroll."
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
    "actual_score": (
        "Live or full-time score from football-data.org. During a match the score updates "
        "with the in-play feed; after the final whistle it shows the full-time result."
    ),
    "match_status": (
        "Match state from football-data.org: Upcoming (not started), LIVE with the current "
        "minute, HT (half-time), or FT (finished)."
    ),
    "outcome_verdict": (
        "Whether the model's predicted W/D/L outcome matched the actual result. "
        "For in-play matches this is provisional and may change before full-time."
    ),
    "exact_score": (
        "The predicted scoreline exactly matched the actual goals for both teams — "
        "a much harder target than just calling the winner."
    ),
    "live_accuracy": (
        "Running tally over finished matches with known teams: how many predicted outcomes "
        "were correct and how many exact scorelines were hit. Knockout fixtures with TBD "
        "teams are excluded until the matchup is set."
    ),
    "goals_compare": (
        "Each scoreline comes from two separate XGBoost regressors — one for home goals, one "
        "for away goals. This shows each model's predicted goals vs the actual goals, with a "
        "✓ when that side's goal count was hit exactly."
    ),
    "home_goals_model": (
        "Home-goals XGBoost regressor: how often it predicted the exact home score, plus its "
        "mean absolute error (MAE) in goals over finished matches. Lower MAE is better."
    ),
    "away_goals_model": (
        "Away-goals XGBoost regressor: how often it predicted the exact away score, plus its "
        "mean absolute error (MAE) in goals over finished matches. Lower MAE is better."
    ),
}
