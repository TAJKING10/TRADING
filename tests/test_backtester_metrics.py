"""Tests for _compute_metrics's Sharpe annualization — fixed this session.

It used to hardcode sqrt(365*24*4), correct only for 15-minute bars, and
applied that uniformly to 4h/1d trend-strategy equity curves too. Fixed to
derive the annualization factor from the equity curve's own timestamp
spacing instead.

Run with: pytest tests/test_backtester_metrics.py -v
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.backtest.backtester import _compute_metrics


def _equity_curve(bar_hours: float, n: int = 60, daily_return: float = 0.01) -> pd.DataFrame:
    """A deterministic equity curve compounding at the same per-DAY rate
    regardless of bar size, so differently-spaced curves are directly
    comparable — any Sharpe difference must come purely from annualization.
    A small deterministic alternating wobble gives it nonzero variance
    (a perfectly constant growth rate has zero std, so Sharpe is trivially 0)."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    per_bar_return = (1 + daily_return) ** (bar_hours / 24) - 1
    equity = 100.0
    rows = []
    for i in range(n):
        rows.append({"timestamp": start + timedelta(hours=bar_hours * i), "equity": equity})
        wobble = 1.001 if i % 2 == 0 else 0.999
        equity *= (1 + per_bar_return) * wobble
    return pd.DataFrame(rows)


def test_sharpe_annualization_factor_matches_bar_spacing_exactly():
    """The annualization factor itself (not just the final Sharpe) must equal
    365*24/bar_hours periods per year — e.g. 2190 for 4h bars, 365 for 1d bars."""
    curve_4h = _equity_curve(bar_hours=4)
    curve_1d = _equity_curve(bar_hours=24)

    for curve, expected_periods_per_year in [(curve_4h, 2190), (curve_1d, 365)]:
        returns = curve["equity"].pct_change().dropna()
        expected_sharpe = returns.mean() / returns.std() * np.sqrt(expected_periods_per_year)
        actual = _compute_metrics(curve, trades=[], starting_capital=100.0)["sharpe_annualized"]
        assert actual == pytest.approx(expected_sharpe, rel=0.01)


def test_sharpe_is_not_the_old_hardcoded_15min_factor():
    """Regression guard: for 4h bars, the OLD bug used sqrt(365*24*4) =
    sqrt(35040) instead of the correct sqrt(365*24/4) = sqrt(2190) -- a 4x
    overstatement. Confirm the new Sharpe is in the corrected ballpark."""
    curve_4h = _equity_curve(bar_hours=4, daily_return=0.002)
    m = _compute_metrics(curve_4h, trades=[], starting_capital=100.0)

    returns = curve_4h["equity"].pct_change().dropna()
    correct_sharpe = returns.mean() / returns.std() * np.sqrt(2190)
    old_buggy_sharpe = returns.mean() / returns.std() * np.sqrt(35040)

    assert m["sharpe_annualized"] == pytest.approx(correct_sharpe, rel=0.01)
    assert abs(m["sharpe_annualized"] - old_buggy_sharpe) > abs(correct_sharpe) * 1.0  # not even close to the old value


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
