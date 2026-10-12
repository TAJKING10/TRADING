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
    trades: list = field(default_factory=list)  # buy/sell event pairs — feeds _compute_metrics, unchanged shape
    trade_log: list = field(default_factory=list)  # one consolidated record per round-trip trade, for audit/export
    rejected: list = field(default_factory=list)  # signals that fired but didn't result in a trade, with reason
    metrics: dict = field(default_factory=dict)


def run_trend_backtest(signals_df: pd.DataFrame, config: TrendConfig) -> TrendBacktestResult:
    """Fill-timing model (fixed this session — see "accurate fill timing" fix):

    A signal computed from bar i's CLOSE (breakout_long, trend flip, Donchian
    exit) cannot realistically be acted on until bar i has actually closed —
    so it fills at bar i+1's OPEN, not bar i's own close. The chandelier stop
    is different: it's a standing protective order at a level already known
    before bar i begins (set at entry, or from bar i-1's close+ATR), so it is
    checked INTRABAR against bar i's low and can fill inside bar i — gap-aware
    (if bar i opens through the stop, it fills at the open, the realistic
    worst case, not at the stale stop level).

    Three phases per bar:
      1. (bar's OPEN) execute whatever was decided at the PREVIOUS bar's close
      2. (bar's LOW)  check the standing chandelier stop intrabar
      3. (bar's CLOSE) decide what to do next bar (new pending entry/exit,
         and roll the chandelier stop level forward for bar i+1's check)
    """
    df = signals_df.dropna(subset=["donchian_high", "donchian_low_exit", "ema_slow", "atr"]).reset_index(drop=True)

    cash = config.starting_capital
    qty = 0.0
    entry_price = None
    chandelier_stop_level = None  # the ACTIVE standing stop, checked intrabar
    peak_close = None  # highest close since entry — same trailing-peak basis as before, unchanged
    equity_rows = []
    trades = []
    trade_log = []
    rejected = []
    open_trade = None  # working state for the currently-open position's trade_log entry
    pending_entry_stop_distance = None  # set when breakout_long fires, consumed at next bar's open
    pending_exit_reason = None  # "trend_flip" / "donchian_exit", set at close, consumed at next bar's open

    def _close_trade(fill_price: float, raw_price: float, reason: str, timestamp) -> None:
        nonlocal cash, qty, entry_price, chandelier_stop_level, peak_close, open_trade
        exit_fee = qty * fill_price * config.taker_fee
        proceeds = qty * fill_price - exit_fee
        cash += proceeds
        trades.append({
            "timestamp": timestamp, "side": "sell", "price": fill_price, "qty": qty,
            "reason": reason, "pnl_pct": fill_price / entry_price - 1,
        })
        if open_trade is not None:
            gross_pnl = (raw_price - open_trade["entry_raw_price"]) * qty
            net_pnl = proceeds - open_trade["cost_basis"]
            holding_hours = (timestamp - open_trade["entry_timestamp"]).total_seconds() / 3600
            r_multiple = (
                (fill_price - open_trade["entry_price"]) / open_trade["initial_stop_distance"]
                if open_trade["initial_stop_distance"] > 0 else None
            )
            trade_log.append({
                "entry_timestamp": open_trade["entry_timestamp"], "exit_timestamp": timestamp,
                "entry_price": open_trade["entry_price"], "exit_price": fill_price, "qty": qty,
                "initial_equity": open_trade["initial_equity"], "entry_reason": "breakout", "exit_reason": reason,
                "initial_stop_price": open_trade["initial_stop_price"],
                "initial_stop_distance": open_trade["initial_stop_distance"],
                "entry_fee": open_trade["entry_fee"], "entry_slippage_cost": open_trade["entry_slippage_cost"],
                "exit_fee": exit_fee, "exit_slippage_cost": qty * (raw_price - fill_price),
                "gross_pnl": gross_pnl, "net_pnl": net_pnl,
                "return_pct": (net_pnl / open_trade["cost_basis"] * 100) if open_trade["cost_basis"] else None,
                "r_multiple": r_multiple, "holding_hours": holding_hours,
                "mfe_pct": (open_trade["mfe_price"] / open_trade["entry_price"] - 1) * 100,
                "mae_pct": (open_trade["mae_price"] / open_trade["entry_price"] - 1) * 100,
            })
            open_trade = None
        qty = 0.0
        entry_price = None
        chandelier_stop_level = None
        peak_close = None

    for _, row in df.iterrows():
        bar_open = row["open"]
        bar_high = row["high"]
        bar_low = row["low"]
        bar_close = row["close"]

        # --- Phase 1: execute what was decided at the PREVIOUS bar's close, at THIS bar's open ---
        if qty > 0 and pending_exit_reason is not None:
            fill_price = bar_open * (1 - config.slippage)
            _close_trade(fill_price, bar_open, pending_exit_reason, row["timestamp"])
            pending_exit_reason = None

        if qty == 0 and pending_entry_stop_distance is not None:
            equity_now = cash
            stop_distance = pending_entry_stop_distance
            risk_amount = equity_now * config.risk_per_trade_pct
            qty_by_risk = risk_amount / stop_distance
            max_notional = equity_now * config.max_position_pct
            qty_by_cap = max_notional / bar_open
            buy_qty = min(qty_by_risk, qty_by_cap)

            if buy_qty > 0:
                fill_price = bar_open * (1 + config.slippage)
                entry_fee = buy_qty * fill_price * config.taker_fee
                cost = buy_qty * fill_price + entry_fee
                if cost <= cash:
                    cash -= cost
                    qty = buy_qty
                    entry_price = fill_price
                    peak_close = fill_price
                    chandelier_stop_level = fill_price - stop_distance
                    trades.append({
                        "timestamp": row["timestamp"], "side": "buy", "price": fill_price, "qty": qty, "reason": "breakout",
                    })
                    open_trade = {
                        "entry_timestamp": row["timestamp"], "entry_price": fill_price, "entry_raw_price": bar_open,
                        "initial_equity": equity_now, "initial_stop_price": chandelier_stop_level,
                        "initial_stop_distance": stop_distance, "entry_fee": entry_fee,
                        "entry_slippage_cost": buy_qty * (fill_price - bar_open), "cost_basis": cost,
                        "mfe_price": bar_high, "mae_price": bar_low,
                    }
                else:
                    rejected.append({
                        "timestamp": row["timestamp"], "reason": "insufficient_cash",
                        "required_cost": cost, "available_cash": cash,
                    })
            else:
                rejected.append({"timestamp": row["timestamp"], "reason": "zero_size", "stop_distance": stop_distance})
            pending_entry_stop_distance = None

        # --- Phase 2: standing chandelier stop, checked INTRABAR against this bar's low ---
        if qty > 0 and chandelier_stop_level is not None and bar_low <= chandelier_stop_level:
            # Gap-aware: if the bar opened through the stop already, you get the open
            # (the realistic worst case), not the stale stop level. Executed via Phase-1-style
            # open-based fill since the stop was already breached before any new close occurs.
            stop_fill_raw = min(bar_open, chandelier_stop_level)
            fill_price = stop_fill_raw * (1 - config.slippage)
            if open_trade is not None:
                open_trade["mfe_price"] = max(open_trade["mfe_price"], bar_high)
                open_trade["mae_price"] = min(open_trade["mae_price"], bar_low)
            _close_trade(fill_price, stop_fill_raw, "chandelier_stop", row["timestamp"])

        # --- Phase 3: decide what to do NEXT bar, using this bar's CLOSE (now fully known) ---
        if qty > 0:
            if open_trade is not None:
                open_trade["mfe_price"] = max(open_trade["mfe_price"], bar_high)
                open_trade["mae_price"] = min(open_trade["mae_price"], bar_low)
            trend_flipped = not row["trend_up"]
            donchian_exit = bar_close < row["donchian_low_exit"]
            if trend_flipped or donchian_exit:
                pending_exit_reason = "trend_flip" if trend_flipped else "donchian_exit"
            # Roll the standing stop forward for bar i+1's intrabar check — same
            # close-based trailing-peak basis as before, just now anchored to the
            # CLOSE of the bar that just finished rather than recomputed mid-bar.
            peak_close = max(peak_close, bar_close)
            chandelier_stop_level = peak_close - config.chandelier_mult * row["atr"]
        elif pending_entry_stop_distance is None:
            if row["breakout_long"]:
                pending_entry_stop_distance = config.chandelier_mult * row["atr"]
                if pending_entry_stop_distance <= 0:
                    rejected.append({"timestamp": row["timestamp"], "reason": "zero_stop_distance"})
                    pending_entry_stop_distance = None

        mark_value = qty * bar_close
        equity_rows.append({"timestamp": row["timestamp"], "equity": cash + mark_value})

    equity_curve = pd.DataFrame(equity_rows)
    metrics = _compute_metrics(equity_curve, trades, config.starting_capital)
    return TrendBacktestResult(
        equity_curve=equity_curve, trades=trades, trade_log=trade_log, rejected=rejected, metrics=metrics,
    )


def sweep_trend_params(
    df: pd.DataFrame,
    base_config: TrendConfig,
    entry_grid: list[int] = (10, 20, 40, 55),
    exit_grid: list[int] = (10, 20),
    chandelier_grid: list[float] = (2.0, 3.0, 4.0),
    trend_fast: int = 50,
    trend_slow: int = 200,
    min_trades: int = 10,
    return_results: bool = False,
):
    """Grid-search breakout/exit/stop parameters. Caller is responsible for
    only passing the DEV portion of history — this function has no concept
    of a holdout split, it just backtests whatever df it's given.

    With return_results=True, also returns {(entry_lb, exit_lb, mult): TrendBacktestResult}
    for every combo that cleared min_trades — lets the caller retrieve the winning
    combo's full trade_log without re-running the backtest a second time. Default
    False keeps the original (DataFrame-only) return shape for existing callers."""
    from ai_trader.features.trend_signals import compute_trend_signals

    rows = []
    results_by_combo = {}
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
                    if return_results:
                        results_by_combo[(entry_lb, exit_lb, mult)] = result

    sweep_df = (
        pd.DataFrame(rows).sort_values("sharpe_annualized", ascending=False).reset_index(drop=True)
        if rows else pd.DataFrame()
    )
    if return_results:
        return sweep_df, results_by_combo
    return sweep_df
