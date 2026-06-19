"""Stake-sizing utilities for FIFA WC 2026 Predictor betting simulation.

Flat staking treats a thin +5% edge identically to a +20% one.  Fractional
Kelly scales the stake with the edge so capital flows to the strongest bets
while damping the variance of full Kelly (which is too aggressive for a model
with thin probabilistic skill).
"""


def kelly_fraction(
    prob: float,
    decimal_odds: float,
    fraction: float = 0.25,
    cap: float = 0.05,
) -> float:
    """Fractional-Kelly stake as a fraction of bankroll for a single 1X2 bet.

    Full Kelly for a win/lose bet at decimal odds ``o`` with win probability
    ``p`` is ``f* = (b·p − (1 − p)) / b`` where ``b = o − 1`` is the net odds.
    We scale by *fraction* (quarter-Kelly by default) to damp variance, floor at
    0 (never stake on a non-positive edge), and cap at *cap* of bankroll.

    Args:
        prob: Model probability of the bet winning, in [0, 1].
        decimal_odds: Decimal odds offered (must be > 1.0 for a live bet).
        fraction: Kelly multiplier (0.25 = quarter-Kelly).
        cap: Maximum stake as a fraction of bankroll.

    Returns:
        Stake as a fraction of bankroll in ``[0, cap]``.  Returns ``0.0`` when
        the edge is non-positive or the odds/probability are invalid.
    """
    try:
        prob = float(prob)
        decimal_odds = float(decimal_odds)
    except (TypeError, ValueError):
        return 0.0

    if not (decimal_odds > 1.0) or not (0.0 <= prob <= 1.0):
        return 0.0

    b = decimal_odds - 1.0
    full_kelly = (b * prob - (1.0 - prob)) / b
    if full_kelly <= 0.0:
        return 0.0

    return min(full_kelly * fraction, cap)
