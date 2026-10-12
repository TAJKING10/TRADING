"""Event-driven backtest for the Donchian breakout + EMA trend filter strategy.

Entry: breakout_long signal from features/trend_signals.py.
Exit: whichever triggers first — chandelier ATR trailing stop, trend filter
flip, or a close back below the short Donchian low. Position sized so a
chandelier-stop hit loses exactly risk_per_trade_pct of equity (volatility-
adjusted sizing — standard for trend-following, unlike a fixed % stop).
"""
from dataclasses import dataclass, field

import pandas as pd

from ai_trader.backtest.backtester import DEFAULT_SLIPPAGE, DEFAULT_TAKER_FEE, _compute_metrics


@dataclass
class TrendConfig:
    chandelier_mult: float = 3.0
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 0.5
    starting_capital: float = 100.0
    taker_fee: float = DEFAULT_TAKER_FEE
    slippage: float = DEFAULT_SLIPPAGE


@dataclass
class TrendBacktestResult:
    equity_curve: pd.DataFrame
    trades: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def run_trend_backtest(signals_df: pd.DataFrame, config: TrendConfig) -> TrendBacktestResult:
    df = signals_df.dropna(subset=["donchian_high", "donchian_low_exit", "ema_slow", "atr"]).reset_index(drop=True)

    cash = config.starting_capital
    qty = 0.0
    entry_price = None
    highest_close_since_entry = None
    equity_rows = []
    trades = []

    for _, row in df.iterrows():
        price = row["close"]
        in_position = qty > 0

        if in_position:
            highest_close_since_entry = max(highest_close_since_entry, price)
            chandelier_stop = highest_close_since_entry - config.chandelier_mult * row["atr"]
            trend_flipped = not row["trend_up"]
            donchian_exit = price < row["donchian_low_exit"]
            hit_stop = price <= chandelier_stop

            if hit_stop or trend_flipped or donchian_exit:
                reason = "chandelier_stop" if hit_stop else "trend_flip" if trend_flipped else "donchian_exit"
                fill_price = price * (1 - config.slippage)
                proceeds = qty * fill_price * (1 - config.taker_fee)
                cash += proceeds
                trades.append(
                    {
                        "timestamp": row["timestamp"],
                        "side": "sell",
                        "price": fill_price,
                        "qty": qty,
                        "reason": reason,
                        "pnl_pct": fill_price / entry_price - 1,
                    }
                )
                qty = 0.0
                entry_price = None
                highest_close_since_entry = None

        else:
            if row["breakout_long"]:
                equity_now = cash  # flat, so equity == cash
                stop_distance = config.chandelier_mult * row["atr"]
                if stop_distance > 0:
                    risk_amount = equity_now * config.risk_per_trade_pct
                    qty_by_risk = risk_amount / stop_distance
                    max_notional = equity_now * config.max_position_pct
                    qty_by_cap = max_notional / price
                    buy_qty = min(qty_by_risk, qty_by_cap)

                    if buy_qty > 0:
                        fill_price = price * (1 + config.slippage)
                        cost = buy_qty * fill_price * (1 + config.taker_fee)
                        if cost <= cash:
                            cash -= cost
                            qty = buy_qty
                            entry_price = fill_price
                            highest_close_since_entry = fill_price
                            trades.append(
                                {"timestamp": row["timestamp"], "side": "buy", "price": fill_price, "qty": qty, "reason": "breakout"}
                            )

        mark_value = qty * price
        equity_rows.append({"timestamp": row["timestamp"], "equity": cash + mark_value})

    equity_curve = pd.DataFrame(equity_rows)
    metrics = _compute_metrics(equity_curve, trades, config.starting_capital)
    return TrendBacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)


def sweep_trend_params(
    df: pd.DataFrame,
    base_config: TrendConfig,
    entry_grid: list[int] = (10, 20, 40, 55),
    exit_grid: list[int] = (10, 20),
    chandelier_grid: list[float] = (2.0, 3.0, 4.0),
    trend_fast: int = 50,
    trend_slow: int = 200,
    min_trades: int = 10,
) -> pd.DataFrame:
    """Grid-search breakout/exit/stop parameters. Caller is responsible for
    only passing the DEV portion of history — this function has no concept
    of a holdout split, it just backtests whatever df it's given."""
    from ai_trader.features.trend_signals import compute_trend_signals

    rows = []
    for entry_lb in entry_grid:
        for exit_lb in exit_grid:
            signals = compute_trend_signals(
                df, entry_lookback=entry_lb, exit_lookback=exit_lb,
                trend_ema_fast=trend_fast, trend_ema_slow=trend_slow,
            )
            for mult in chandelier_grid:
                cfg = TrendConfig(
                    chandelier_mult=mult,
                    risk_per_trade_pct=base_config.risk_per_trade_pct,
                    max_position_pct=base_config.max_position_pct,
                    starting_capital=base_config.starting_capital,
                    taker_fee=base_config.taker_fee,
                    slippage=base_config.slippage,
                )
                result = run_trend_backtest(signals, cfg)
                m = result.metrics
                if m.get("num_trades", 0) >= min_trades:
                    rows.append({"entry_lookback": entry_lb, "exit_lookback": exit_lb, "chandelier_mult": mult, **m})

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sharpe_annualized", ascending=False).reset_index(drop=True)
