"""Unit tests for fractional-Kelly stake sizing."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.betting.staking import kelly_fraction


def test_kelly_zero_on_non_positive_edge():
    """No edge (fair coin at evens) or negative edge -> stake 0."""
    assert kelly_fraction(0.50, 2.0) == 0.0          # exactly fair
    assert kelly_fraction(0.30, 2.0) == 0.0          # negative edge
    assert kelly_fraction(0.0, 5.0) == 0.0


def test_kelly_positive_on_positive_edge():
    """A genuine edge produces a positive stake within the cap."""
    stake = kelly_fraction(0.60, 2.0, fraction=0.25, cap=0.05)
    assert 0.0 < stake <= 0.05


def test_kelly_scales_with_edge():
    """Bigger edge -> bigger stake (monotonic), all else equal."""
    small = kelly_fraction(0.55, 2.0, fraction=1.0, cap=1.0)
    big = kelly_fraction(0.70, 2.0, fraction=1.0, cap=1.0)
    assert big > small


def test_kelly_respects_cap():
    """Stake is never larger than the cap, even for a huge edge."""
    stake = kelly_fraction(0.95, 2.0, fraction=1.0, cap=0.05)
    assert stake == pytest.approx(0.05)


def test_kelly_invalid_odds():
    """Odds <= 1.0 or invalid inputs -> stake 0."""
    assert kelly_fraction(0.60, 1.0) == 0.0
    assert kelly_fraction(0.60, 0.5) == 0.0
    assert kelly_fraction(1.5, 2.0) == 0.0  # prob out of range


def test_kelly_full_formula():
    """Quarter-Kelly equals a quarter of the full-Kelly fraction."""
    p, o = 0.60, 2.5
    b = o - 1.0
    full = (b * p - (1 - p)) / b
    expected = full * 0.25
    assert kelly_fraction(p, o, fraction=0.25, cap=1.0) == pytest.approx(expected)
