"""Hand-rolled technical indicators (pandas/numpy only).

pandas-ta pulls in numba, which has no wheels for Python 3.14 yet — these are
the indicators actually used by the model, implemented directly.
"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_pct_b": pct_b})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Where does today's value sit vs the trailing window, as a 0-1 rank?
    Used to detect volatility *regime* (e.g. current ATR high/low vs recent history)
    rather than just the raw ATR value, which isn't comparable across price levels/time.
    """
    def _rank(x):
        return (x.rank(pct=True)).iloc[-1]
    return series.rolling(window).apply(_rank, raw=False)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw OHLCV into a model-ready feature frame.

    Input df must have columns: timestamp, open, high, low, close, volume.
    Output keeps timestamp + close (for labeling/backtest) plus feature_* columns.
    """
    out = df.copy()

    # --- momentum / returns at multiple horizons ---
    out["ret_1"] = out["close"].pct_change(1)
    out["ret_3"] = out["close"].pct_change(3)
    out["ret_12"] = out["close"].pct_change(12)
    out["ret_48"] = out["close"].pct_change(48)  # ~12h on 15m candles

    # --- trend context: short vs long EMA, and price vs a slower EMA ---
    out["ema_fast"] = ema(out["close"], 9)
    out["ema_slow"] = ema(out["close"], 26)
    out["ema_ratio"] = out["ema_fast"] / out["ema_slow"] - 1

    out["ema_100"] = ema(out["close"], 100)
    out["price_vs_ema100"] = out["close"] / out["ema_100"] - 1

    out["rsi_14"] = rsi(out["close"], 14)
    out["rsi_28"] = rsi(out["close"], 28)

    macd_df = macd(out["close"])
    out = pd.concat([out, macd_df], axis=1)

    bb_df = bollinger(out["close"])
    out = pd.concat([out, bb_df], axis=1)

    # --- volatility level + volatility regime (is vol unusually high/low right now?) ---
    out["atr_14"] = atr(out["high"], out["low"], out["close"], 14)
    out["atr_pct"] = out["atr_14"] / out["close"]
    out["atr_regime"] = rolling_percentile_rank(out["atr_pct"], 200)

    # --- volume ---
    vol_mean = out["volume"].rolling(20).mean()
    vol_std = out["volume"].rolling(20).std()
    out["volume_z"] = (out["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    out["volume_roc"] = out["volume"].pct_change(3)

    out["hl_range_pct"] = (out["high"] - out["low"]) / out["close"]

    # --- support/resistance proxy: where are we vs recent swing high/low? ---
    rolling_high_20 = out["high"].rolling(20).max()
    rolling_low_20 = out["low"].rolling(20).min()
    out["dist_from_high_20"] = out["close"] / rolling_high_20 - 1
    out["dist_from_low_20"] = out["close"] / rolling_low_20 - 1

    # --- seasonality: cyclical hour-of-day / day-of-week (crypto still has human-driven patterns) ---
    hour = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60
    dow = out["timestamp"].dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    feature_cols = [
        "ret_1", "ret_3", "ret_12", "ret_48",
        "ema_ratio", "price_vs_ema100", "rsi_14", "rsi_28",
        "macd", "macd_signal", "macd_hist",
        "bb_pct_b", "atr_pct", "atr_regime",
        "volume_z", "volume_roc", "hl_range_pct",
        "dist_from_high_20", "dist_from_low_20",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]
    out = out.rename(columns={c: f"feature_{c}" for c in feature_cols})

    keep = ["timestamp", "close"] + [f"feature_{c}" for c in feature_cols]
    return out[keep]


FEATURE_COLUMNS = [
    "feature_ret_1", "feature_ret_3", "feature_ret_12", "feature_ret_48",
    "feature_ema_ratio", "feature_price_vs_ema100", "feature_rsi_14", "feature_rsi_28",
    "feature_macd", "feature_macd_signal", "feature_macd_hist",
    "feature_bb_pct_b", "feature_atr_pct", "feature_atr_regime",
    "feature_volume_z", "feature_volume_roc", "feature_hl_range_pct",
    "feature_dist_from_high_20", "feature_dist_from_low_20",
    "feature_hour_sin", "feature_hour_cos", "feature_dow_sin", "feature_dow_cos",
]
