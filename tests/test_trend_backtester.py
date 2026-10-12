"""Tests for the trade-level logging added to run_trend_backtest.

These intentionally use a hand-built, deterministic synthetic price series —
not live exchange data — so results are reproducible and don't depend on
network access or what the market happened to do today.

Run with: pytest tests/test_trend_backtester.py -v
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from ai_trader.backtest.trend_backtester import TrendConfig, run_trend_backtest
from ai_trader.features.trend_signals import compute_trend_signals


def _synthetic_ohlcv() -> pd.DataFrame:
    """Flat -> clean breakout ramp -> sharp pullback. Engineered to:
    - NOT trigger a breakout during the flat phase (no lookahead false positive)
    - trigger exactly one breakout entry during the ramp
    - give the position room to run (MFE > 0) before the pullback trips the
      chandelier stop, exiting at a net gain (so win/loss/R-multiple sign is known)
    """
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    closes = []

    # Flat phase: 25 bars at 100, tiny alternating noise so ATR isn't exactly zero.
    for i in range(25):
        closes.append(100.0 + (0.1 if i % 2 == 0 else -0.1))

    # Ramp phase: 20 bars, +2/bar, 100 -> ~140.
    for i in range(20):
        closes.append(closes[-1] + 2.0)

    # Pullback phase: 10 bars, -3/bar.
    for i in range(10):
        closes.append(closes[-1] - 3.0)

    # Flat tail so the backtest has bars after the exit too.
    for i in range(10):
        closes.append(closes[-1])

    rows = []
    for i, close in enumerate(closes):
        rows.append({
            "timestamp": start + timedelta(hours=i),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def backtest_result():
    df = _synthetic_ohlcv()
    signals = compute_trend_signals(
        df, entry_lookback=10, exit_lookback=5, trend_ema_fast=5, trend_ema_slow=15, atr_period=5,
    )
    config = TrendConfig(chandelier_mult=2.0, risk_per_trade_pct=0.01, max_position_pct=0.5,
                          starting_capital=1000.0, taker_fee=0.008, slippage=0.0005)
    return run_trend_backtest(signals, config)


def test_at_least_one_trade_opened_and_closed(backtest_result):
    assert len(backtest_result.trade_log) >= 1, "Synthetic breakout + pullback should produce exactly one round trip"


def test_fees_charged_on_both_entry_and_exit(backtest_result):
    for t in backtest_result.trade_log:
        assert t["entry_fee"] > 0, "Entry fee should be nonzero with taker_fee=0.008"
        assert t["exit_fee"] > 0, "Exit fee should be nonzero with taker_fee=0.008"


def test_mfe_and_mae_bound_the_trade_correctly(backtest_result):
    for t in backtest_result.trade_log:
        # MFE is the best price seen while in the position — must be >= 0 for a
        # trade that ran up before being stopped out (we engineered the ramp for this).
        assert t["mfe_pct"] >= 0
        # MAE must never be more favorable than MFE.
        assert t["mae_pct"] <= t["mfe_pct"]


def test_r_multiple_sign_matches_net_pnl(backtest_result):
    for t in backtest_result.trade_log:
        if t["r_multiple"] is None:
            continue
        # Same direction: a winning trade (positive net PnL) should have a
        # positive R-multiple (price moved favorably relative to initial stop
        # distance), and vice versa. Fees can flip a near-breakeven trade's
        # net_pnl sign without flipping the pre-fee R-multiple, so only assert
        # this for trades with a clearly nonzero result.
        if abs(t["net_pnl"]) > 0.01:
            assert (t["r_multiple"] > 0) == (t["net_pnl"] > 0)


def test_aggregate_metrics_reconcile_with_trade_log(backtest_result):
    """The whole point of per-trade logging: you should be able to reconstruct
    the headline metrics purely from trade_log, and they should match what
    _compute_metrics (used everywhere else, unchanged) reports."""
    trade_log = backtest_result.trade_log
    metrics = backtest_result.metrics

    assert len(trade_log) == metrics["num_trades"]

    wins = [t for t in trade_log if t["net_pnl"] > 0]
    reconstructed_win_rate = round(len(wins) / len(trade_log) * 100, 2) if trade_log else 0.0
    assert reconstructed_win_rate == pytest.approx(metrics["win_rate_pct"], abs=0.5)

    reconstructed_avg_holding = sum(t["holding_hours"] for t in trade_log) / len(trade_log)
    assert reconstructed_avg_holding == pytest.approx(metrics["avg_holding_hours"], rel=0.01)

    # Final equity = starting capital + sum of closed-trade net PnL (+ any
    # unrealized mark-to-market on a still-open position at the end, which
    # this synthetic series doesn't have — the pullback closes the trade).
    final_equity = backtest_result.equity_curve["equity"].iloc[-1]
    reconstructed_equity = 1000.0 + sum(t["net_pnl"] for t in trade_log)
    assert reconstructed_equity == pytest.approx(final_equity, rel=1e-6)


def test_no_lookahead_in_donchian_channel():
    """donchian_high at row i must depend only on rows strictly before i —
    changing a future bar's high must NOT change today's breakout level."""
    df = _synthetic_ohlcv()
    signals_a = compute_trend_signals(df, entry_lookback=10, exit_lookback=5)

    df_b = df.copy()
    df_b.loc[df_b.index[-1], "high"] = 999999.0  # blow up the very last bar's high
    signals_b = compute_trend_signals(df_b, entry_lookback=10, exit_lookback=5)

    # Every row except the last should be identical — the modified last bar's
    # high must not have leaked into any earlier row's donchian_high.
    pd.testing.assert_series_equal(
        signals_a["donchian_high"].iloc[:-1], signals_b["donchian_high"].iloc[:-1],
    )


def test_rejected_trade_logged_when_cash_insufficient():
    """Direct unit test with a hand-built signals_df (bypassing compute_trend_signals)
    so the breakout/rejection branch is hit deterministically, not dependent on
    incidental synthetic-price behavior."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    n = 5
    df = pd.DataFrame({
        "timestamp": [start + timedelta(hours=i) for i in range(n)],
        "close": [100.0] * n,
        "high": [100.5] * n,
        "low": [99.5] * n,
        "donchian_high": [99.0] * n,       # close (100) > donchian_high -> breakout condition met
        "donchian_low_exit": [1.0] * n,     # never hit
        "ema_slow": [50.0] * n,
        "trend_up": [True] * n,
        "atr": [5.0] * n,
        "breakout_long": [True, False, False, False, False],
    })
    # starting_capital far smaller than the cost of even the smallest sized trade.
    config = TrendConfig(chandelier_mult=2.0, risk_per_trade_pct=0.5, max_position_pct=1.0,
                          starting_capital=0.001, taker_fee=0.008, slippage=0.0005)
    result = run_trend_backtest(df, config)

    assert len(result.trade_log) == 0
    assert len(result.rejected) == 1
    assert result.rejected[0]["reason"] == "insufficient_cash"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
