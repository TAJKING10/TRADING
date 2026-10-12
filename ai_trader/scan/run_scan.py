"""Per-coin, per-horizon honest backtest scan. For each coin in the universe,
at a "near" (4h candles) and "far" (1d candles) horizon: sweep Donchian
breakout parameters on a dev slice, validate the winner on a never-touched
holdout slice, and record whether that holdout actually beat buy-and-hold.
Only coins/horizons that clear that bar get marked as having a validated edge.
"""
import logging
import time
from datetime import datetime, timezone

import pandas as pd

from ai_trader.backtest.trend_backtester import TrendConfig, run_trend_backtest, sweep_trend_params
from ai_trader.config.settings import Settings
from ai_trader.data.market_data import fetch_historical, fetch_latest_candles, make_exchange, make_training_data_exchange
from ai_trader.features.trend_signals import compute_trend_signals
from ai_trader.scan.universe import resolve_universe

logger = logging.getLogger("run_scan")

HORIZONS = [
    {"name": "near", "timeframe": "4h", "lookback_days": 365},
    {"name": "far", "timeframe": "1d", "lookback_days": 1095},
]

MIN_HOLDOUT_TRADES = 5
MIN_HOLDOUT_SHARPE = 0.5


def _buy_and_hold_return(df) -> float:
    return float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)


def _btc_correlation(df: pd.DataFrame, btc_returns: pd.Series) -> float | None:
    """Pearson correlation of this coin's returns against BTC's, over whatever
    period overlaps. Directly addresses the "are these 862 slots actually
    independent bets, or mostly one BTC-driven market move wearing 431 coin
    tickers" critique — computed from real fetched data, not asserted."""
    coin_returns = df.set_index("timestamp")["close"].pct_change()
    aligned_coin, aligned_btc = coin_returns.align(btc_returns, join="inner")
    aligned_coin = aligned_coin.dropna()
    aligned_btc = aligned_btc.loc[aligned_coin.index]
    if len(aligned_coin) < 30:  # too few overlapping points for a meaningful correlation
        return None
    corr = aligned_coin.corr(aligned_btc)
    return round(float(corr), 3) if pd.notna(corr) else None


def _serialize_trades(trade_log: list, dataset: str) -> list:
    """trade_log entries hold pandas Timestamps / numpy floats — make them
    plain JSON/Supabase-safe values, and tag which dataset slice they're from."""
    out = []
    for t in trade_log:
        row = dict(t)
        row["dataset"] = dataset
        row["entry_timestamp"] = row["entry_timestamp"].isoformat()
        row["exit_timestamp"] = row["exit_timestamp"].isoformat()
        for k, v in row.items():
            if isinstance(v, float):
                row[k] = round(v, 8)
        out.append(row)
    return out


def _scan_one(training_exchange, kraken, coin: dict, horizon: dict, base_config: TrendConfig, btc_returns: pd.Series) -> dict:
    df = fetch_historical(
        training_exchange, coin["training_symbol"], horizon["timeframe"], lookback_days=horizon["lookback_days"]
    )
    if len(df) < 300:
        return {"status": "insufficient_data", "n_candles": len(df)}

    btc_correlation = _btc_correlation(df, btc_returns)

    df = df.reset_index(drop=True)
    holdout_start = int(len(df) * 0.85)
    dev_df = df.iloc[:holdout_start].reset_index(drop=True)
    holdout_df = df.iloc[holdout_start:].reset_index(drop=True)

    sweep, dev_results_by_combo = sweep_trend_params(
        dev_df, base_config, min_trades=max(5, MIN_HOLDOUT_TRADES), return_results=True,
    )
    if sweep.empty:
        return {"status": "no_signal", "n_candles": len(df), "btc_correlation": btc_correlation}

    best = sweep.iloc[0]
    best_combo = (int(best["entry_lookback"]), int(best["exit_lookback"]), float(best["chandelier_mult"]))
    dev_trade_log = dev_results_by_combo[best_combo].trade_log
    holdout_signals = compute_trend_signals(
        holdout_df, entry_lookback=int(best["entry_lookback"]), exit_lookback=int(best["exit_lookback"])
    )
    honest_cfg = TrendConfig(
        chandelier_mult=float(best["chandelier_mult"]),
        risk_per_trade_pct=base_config.risk_per_trade_pct,
        max_position_pct=base_config.max_position_pct,
        starting_capital=base_config.starting_capital,
        taker_fee=base_config.taker_fee,
        slippage=base_config.slippage,
    )
    honest = run_trend_backtest(holdout_signals, honest_cfg)
    bh_return = _buy_and_hold_return(holdout_df)

    edge_validated = (
        honest.metrics.get("num_trades", 0) >= MIN_HOLDOUT_TRADES
        and honest.metrics.get("sharpe_annualized", -999) >= MIN_HOLDOUT_SHARPE
        and honest.metrics.get("total_return_pct", -999) > bh_return * 100
    )

    # Live status off the most recent candles from the EXECUTION exchange (Kraken) —
    # the price you'd actually trade at, not Binance's.
    live_candles = fetch_latest_candles(kraken, coin["kraken_symbol"], horizon["timeframe"], limit=300)
    live_signals = compute_trend_signals(
        live_candles, entry_lookback=int(best["entry_lookback"]), exit_lookback=int(best["exit_lookback"])
    )
    last = live_signals.iloc[-1]
    if bool(last["breakout_long"]):
        live_status = "BREAKOUT_ACTIVE"
    elif bool(last["trend_up"]):
        live_status = "UPTREND_WAITING"
    else:
        live_status = "NO_SIGNAL"

    return {
        "status": "ok",
        "edge_validated": bool(edge_validated),
        "params": {
            "entry_lookback": int(best["entry_lookback"]),
            "exit_lookback": int(best["exit_lookback"]),
            "chandelier_mult": float(best["chandelier_mult"]),
        },
        "holdout_metrics": honest.metrics,
        "buy_and_hold_pct": round(bh_return * 100, 2),
        "live_price": float(last["close"]),
        "live_status": live_status,
        "donchian_breakout_level": float(last["donchian_high"]) if pd.notna(last["donchian_high"]) else None,
        "atr": float(last["atr"]) if pd.notna(last["atr"]) else None,
        "n_candles": len(df),
        "btc_correlation": btc_correlation,
        # Per-trade audit trail for the winning combo — "dev" is what it chose the
        # params on, "holdout" is the one to trust (never touched during tuning).
        "trade_log": _serialize_trades(dev_trade_log, "dev") + _serialize_trades(honest.trade_log, "holdout"),
        "rejected_count": len(honest.rejected),
    }


def run_scan(settings: Settings) -> dict:
    kraken = make_exchange(settings)
    training_exchange = make_training_data_exchange(settings)
    kraken.load_markets()
    training_exchange.load_markets()

    universe = resolve_universe(kraken, training_exchange)
    logger.info("Scanning %d coins x %d horizons", len(universe), len(HORIZONS))

    base_config = TrendConfig(
        risk_per_trade_pct=settings.risk_per_trade_pct,
        max_position_pct=settings.max_position_pct,
        starting_capital=settings.capital_allocation_usdt,
        taker_fee=settings.taker_fee_pct,
        slippage=settings.slippage_pct,
    )

    # Fetched once per horizon (not per-coin) — every coin's btc_correlation is
    # computed against this same series.
    btc_returns_by_horizon = {}
    for horizon in HORIZONS:
        btc_df = fetch_historical(training_exchange, "BTC/USDT", horizon["timeframe"], lookback_days=horizon["lookback_days"])
        btc_returns_by_horizon[horizon["name"]] = btc_df.set_index("timestamp")["close"].pct_change()

    coins_result = {}
    for coin in universe:
        coin_entry = {}
        for horizon in HORIZONS:
            t0 = time.time()
            try:
                result = _scan_one(
                    training_exchange, kraken, coin, horizon, base_config, btc_returns_by_horizon[horizon["name"]],
                )
            except Exception as e:
                logger.exception("Scan failed for %s/%s", coin["base"], horizon["name"])
                result = {"status": "error", "error": str(e)}
            result["elapsed_s"] = round(time.time() - t0, 1)
            coin_entry[horizon["name"]] = result
            logger.info(
                "%s/%s: %s (%.1fs)", coin["base"], horizon["name"], result.get("status"), result["elapsed_s"]
            )
        coins_result[coin["base"]] = coin_entry

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizons": HORIZONS,
        "coins": coins_result,
    }
