"""Rule-based trend-following / breakout signals — classic Donchian breakout +
EMA trend filter + ATR chandelier exit. No ML, no training data leakage risk;
the only thing that can overfit here is the parameter choice, so we still
dev/holdout split when sweeping parameters.
"""
import pandas as pd

from ai_trader.features.indicators import atr, ema


def compute_trend_signals(
    df: pd.DataFrame,
    entry_lookback: int = 20,
    exit_lookback: int = 10,
    trend_ema_fast: int = 50,
    trend_ema_slow: int = 200,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Input df: timestamp, open, high, low, close, volume.

    donchian_high/donchian_low_exit are shifted by 1 bar — a breakout is
    "today's close exceeds the high of the PRIOR N bars," never including
    today's own bar (that would be lookahead).
    """
    out = df.copy()

    out["donchian_high"] = out["high"].rolling(entry_lookback).max().shift(1)
    out["donchian_low_exit"] = out["low"].rolling(exit_lookback).min().shift(1)

    out["ema_fast"] = ema(out["close"], trend_ema_fast)
    out["ema_slow"] = ema(out["close"], trend_ema_slow)
    out["trend_up"] = out["ema_fast"] > out["ema_slow"]

    out["atr"] = atr(out["high"], out["low"], out["close"], atr_period)

    out["breakout_long"] = (out["close"] > out["donchian_high"]) & out["trend_up"]

    return out
