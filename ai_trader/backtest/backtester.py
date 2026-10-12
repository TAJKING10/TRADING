"""Event-driven long-only backtest over out-of-sample model predictions.

Uses real fees + a slippage assumption so the result isn't a fantasy backtest.
Binance spot taker fee is 0.1% by default; BNB fee discount not assumed.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TAKER_FEE = 0.001
SLIPPAGE = 0.0005  # assumed adverse price move on fill, each side


@dataclass
class BacktestConfig:
    buy_threshold: float = 0.55
    sell_threshold: float = 0.45
    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.03
    position_pct: float = 0.5  # fraction of capital deployed per trade
    starting_capital: float = 100.0


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def run_backtest(oos_predictions: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    df = oos_predictions.sort_values("timestamp").reset_index(drop=True)

    cash = config.starting_capital
    position_qty = 0.0
    entry_price = None
    equity_rows = []
    trades = []

    for _, row in df.iterrows():
        price = row["close"]
        in_position = position_qty > 0

        if in_position:
            change = price / entry_price - 1
            hit_stop = change <= -config.stop_loss_pct
            hit_target = change >= config.take_profit_pct
            sell_signal = row["pred_proba"] < config.sell_threshold

            if hit_stop or hit_target or sell_signal:
                fill_price = price * (1 - SLIPPAGE)
                proceeds = position_qty * fill_price * (1 - TAKER_FEE)
                cash += proceeds
                trades.append(
                    {
                        "timestamp": row["timestamp"],
                        "side": "sell",
                        "price": fill_price,
                        "qty": position_qty,
                        "reason": "stop" if hit_stop else "target" if hit_target else "signal",
                        "pnl_pct": change,
                    }
                )
                position_qty = 0.0
                entry_price = None

        else:
            buy_signal = row["pred_proba"] > config.buy_threshold
            if buy_signal:
                deploy = cash * config.position_pct
                fill_price = price * (1 + SLIPPAGE)
                qty = (deploy * (1 - TAKER_FEE)) / fill_price
                cash -= deploy
                position_qty = qty
                entry_price = fill_price
                trades.append(
                    {"timestamp": row["timestamp"], "side": "buy", "price": fill_price, "qty": qty, "reason": "signal"}
                )

        mark_value = position_qty * price
        equity_rows.append({"timestamp": row["timestamp"], "equity": cash + mark_value})

    equity_curve = pd.DataFrame(equity_rows)
    metrics = _compute_metrics(equity_curve, trades, config.starting_capital)
    return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)


def sweep_thresholds(
    oos_predictions: pd.DataFrame,
    base_config: BacktestConfig,
    buy_grid: list[float] = (0.52, 0.55, 0.58, 0.62, 0.66, 0.70),
    sell_grid: list[float] = (0.30, 0.35, 0.40, 0.45),
    min_trades: int = 20,
) -> pd.DataFrame:
    """Grid-search buy/sell thresholds on the SAME out-of-sample window.

    Caveat: picking thresholds by looking at this data is itself a (mild) form
    of overfitting — treat any "winning" config as a hypothesis to re-validate
    on fresh data going forward, not as a confirmed edge.
    """
    rows = []
    for buy_t in buy_grid:
        for sell_t in sell_grid:
            if sell_t >= buy_t:
                continue
            cfg = BacktestConfig(
                buy_threshold=buy_t,
                sell_threshold=sell_t,
                stop_loss_pct=base_config.stop_loss_pct,
                take_profit_pct=base_config.take_profit_pct,
                position_pct=base_config.position_pct,
                starting_capital=base_config.starting_capital,
            )
            result = run_backtest(oos_predictions, cfg)
            m = result.metrics
            if m.get("num_trades", 0) >= min_trades:
                rows.append({"buy_threshold": buy_t, "sell_threshold": sell_t, **m})

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("total_return_pct", ascending=False).reset_index(drop=True)


def _compute_metrics(equity_curve: pd.DataFrame, trades: list, starting_capital: float) -> dict:
    if equity_curve.empty:
        return {"error": "no data"}

    equity = equity_curve["equity"]
    returns = equity.pct_change().dropna()

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    total_return = equity.iloc[-1] / starting_capital - 1
    sharpe = (returns.mean() / returns.std() * np.sqrt(365 * 24 * 4)) if returns.std() > 0 else 0.0

    sell_trades = [t for t in trades if t["side"] == "sell"]
    wins = [t for t in sell_trades if t["pnl_pct"] > 0]
    win_rate = len(wins) / len(sell_trades) if sell_trades else 0.0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe_annualized": round(float(sharpe), 2),
        "num_trades": len(sell_trades),
        "win_rate_pct": round(win_rate * 100, 2),
        "final_equity": round(float(equity.iloc[-1]), 2),
    }
