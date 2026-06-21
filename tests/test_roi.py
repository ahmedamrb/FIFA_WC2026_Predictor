"""Unit tests for realised value-bet ROI settlement."""

import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.betting.roi import (
    bankroll_curve,
    settle_value_bets,
    summarize_roi,
    value_bet_roi,
)
from src.betting.staking import kelly_fraction
from src.evaluation.live_tracking import build_comparison


# ---------------------------------------------------------------------------
# Synthetic fixtures / predictions / results / odds
# ---------------------------------------------------------------------------

def _fixtures() -> pd.DataFrame:
    return pd.DataFrame({
        "fixture_id": [101, 102, 103, 104],
        "match_date": ["2026-06-11", "2026-06-12", "2026-06-13", "2026-06-28"],
        "home_team": ["Mexico", "Canada", "Brazil", "Spain"],
        "away_team": ["South Africa", "Bosnia-Herzegovina", "Morocco", "Portugal"],
        "stage": ["GROUP_STAGE", "GROUP_STAGE", "GROUP_STAGE", "GROUP_STAGE"],
    })


def _predictions() -> pd.DataFrame:
    # 101: strong home edge (value). 102: strong home edge (value, will lose).
    # 103: no edge vs its odds. 104: home edge but not yet played (pending).
    return pd.DataFrame({
        "prob_home_win": [0.70, 0.70, 0.34, 0.70],
        "prob_draw": [0.20, 0.20, 0.33, 0.20],
        "prob_away_win": [0.10, 0.10, 0.33, 0.10],
        "predicted_outcome": ["Home Win", "Home Win", "Draw", "Home Win"],
        "predicted_home_goals": [2, 2, 1, 2],
        "predicted_away_goals": [0, 0, 1, 0],
    })


def _results() -> pd.DataFrame:
    # 101 home win (bet wins). 102 away win (bet loses). 103 draw (no bet).
    # 104 not played.
    return pd.DataFrame({
        "fixture_id": [101, 102, 103],
        "status": ["FINISHED", "FINISHED", "FINISHED"],
        "home_score": [2, 0, 1],
        "away_score": [0, 2, 1],
        "winner": ["HOME_TEAM", "AWAY_TEAM", "DRAW"],
        "minute": [None, None, None],
    })


def _odds() -> pd.DataFrame:
    # Even-ish odds so a 0.70 model prob clears the +5% vig-free edge threshold.
    return pd.DataFrame({
        "match_date": ["2026-06-11", "2026-06-12", "2026-06-13", "2026-06-28"],
        "home_team": ["Mexico", "Canada", "Brazil", "Spain"],
        "away_team": ["South Africa", "Bosnia-Herzegovina", "Morocco", "Portugal"],
        "home_win_odds": [2.0, 2.0, 3.0, 2.0],
        "draw_odds": [3.5, 3.5, 3.0, 3.5],
        "away_win_odds": [4.0, 4.0, 3.0, 4.0],
        "source": ["test"] * 4,
        "fetched_at": ["2026-06-01"] * 4,
    })


def _comp():
    return build_comparison(_fixtures(), _predictions(), _results())


# ---------------------------------------------------------------------------
# settle_value_bets
# ---------------------------------------------------------------------------

def test_ledger_flags_value_bets_only():
    ledger = settle_value_bets(_comp(), _predictions(), _odds())
    # 101, 102, 104 carry a home value bet; 103 has no edge -> excluded.
    fixtures_with_bets = set(ledger["fixture_id"])
    assert fixtures_with_bets == {101, 102, 104}
    assert (ledger["value_outcome"] == "Home Win").all()


def test_ledger_settles_finished_only():
    ledger = settle_value_bets(_comp(), _predictions(), _odds())
    settled = ledger[ledger["settled"].astype(bool)]
    pending = ledger[~ledger["settled"].astype(bool)]
    assert set(settled["fixture_id"]) == {101, 102}   # finished
    assert set(pending["fixture_id"]) == {104}         # not played yet


def test_ledger_profit_matches_stake_and_odds():
    ledger = settle_value_bets(_comp(), _predictions(), _odds()).set_index("fixture_id")

    stake_101 = kelly_fraction(0.70, 2.0)
    # 101 won at odds 2.0 -> profit = stake * (odds - 1)
    assert bool(ledger.loc[101, "won"]) is True
    assert ledger.loc[101, "profit"] == pytest.approx(stake_101 * (2.0 - 1.0))
    # 102 lost -> profit = -stake
    assert bool(ledger.loc[102, "won"]) is False
    assert ledger.loc[102, "profit"] == pytest.approx(-stake_101)


# ---------------------------------------------------------------------------
# value_bet_roi
# ---------------------------------------------------------------------------

def test_roi_aggregates_settled_bets_flat():
    # initial_bankroll=1.0 keeps stakes in fraction-of-bankroll terms.
    summary = value_bet_roi(_comp(), _predictions(), _odds(), mode="flat", initial_bankroll=1.0)
    stake = kelly_fraction(0.70, 2.0)
    expected_profit = stake * (2.0 - 1.0) - stake  # one win, one loss at equal stakes
    assert summary["n_bets"] == 2
    assert summary["won"] == 1
    assert summary["pending"] == 1
    assert summary["staked"] == pytest.approx(2 * stake)
    assert summary["profit"] == pytest.approx(expected_profit)
    assert summary["roi"] == pytest.approx(expected_profit / (2 * stake))
    assert summary["final"] == pytest.approx(1.0 + expected_profit)
    assert summary["initial"] == 1.0


def test_roi_flat_vs_compound_differ():
    # 101 wins (@2.0) then 102 loses (@2.0); equal Kelly fractions.
    # Flat: stakes off the fixed $1000 -> the win and loss cancel, final == $1000.
    # Compound: the second stake is sized off the post-win bankroll, so the loss
    # is larger in dollars and the final bankroll dips below $1000.
    flat = value_bet_roi(_comp(), _predictions(), _odds(), mode="flat", initial_bankroll=1000.0)
    comp = value_bet_roi(_comp(), _predictions(), _odds(), mode="compound", initial_bankroll=1000.0)

    f = kelly_fraction(0.70, 2.0)  # 0.05 (quarter-Kelly hits the 5% cap)
    assert flat["final"] == pytest.approx(1000.0)
    assert comp["final"] == pytest.approx(1000.0 * (1 - f * f))
    assert comp["final"] < flat["final"]
    assert flat["initial"] == 1000.0 and comp["initial"] == 1000.0


def test_bankroll_curve_running_balance():
    curve = bankroll_curve(settle_value_bets(_comp(), _predictions(), _odds()),
                           mode="flat", initial_bankroll=1000.0)
    # Two settled bets, chronological (101 then 102).
    assert list(curve["fixture_id"]) == [101, 102]
    f = kelly_fraction(0.70, 2.0)
    assert curve["stake_dollars"].iloc[0] == pytest.approx(f * 1000.0)
    assert curve["bankroll"].iloc[0] == pytest.approx(1000.0 + f * 1000.0)   # win
    assert curve["bankroll"].iloc[1] == pytest.approx(1000.0)                # loss cancels


def test_roi_empty_without_results():
    summary = value_bet_roi(build_comparison(_fixtures(), _predictions(), None),
                            _predictions(), _odds())
    assert summary["n_bets"] == 0
    assert summary["roi"] == 0.0
    assert summary["final"] == 1000.0  # untouched starting bankroll
    # All three value bets are pending (none finished); 103 has no edge.
    assert summary["pending"] == 3


def test_roi_no_odds_means_no_bets():
    empty_odds = _odds().iloc[0:0]
    summary = value_bet_roi(_comp(), _predictions(), empty_odds, initial_bankroll=1000.0)
    assert summary == {
        "n_bets": 0, "won": 0, "win_pct": 0.0,
        "staked": 0.0, "profit": 0.0, "roi": 0.0, "growth": 0.0,
        "initial": 1000.0, "final": 1000.0, "pending": 0,
    }


def test_roi_skips_tbd_fixtures():
    fixtures = _fixtures()
    fixtures.loc[3, ["home_team", "away_team"]] = ["TBD", "TBD"]
    comp = build_comparison(fixtures, _predictions(), _results())
    ledger = settle_value_bets(comp, _predictions(), _odds())
    assert 104 not in set(ledger["fixture_id"])  # TBD -> not comparable
