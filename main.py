"""CLI entry point.

Usage:
    python main.py backtest        # fetch history, train walk-forward, report honest OOS backtest metrics
    python main.py train           # train + save a model version to artifacts/
    python main.py signal          # generate today's trade recommendation (reads real balances, places no orders)
    python main.py run             # start the live loop (dry-run or real money per .env DRY_RUN)
    python main.py smoke-test      # dry-run plumbing check before ever flipping DRY_RUN=false
    python main.py reset-risk      # clear a tripped kill switch after you've investigated why it tripped
"""
import argparse
import logging

from ai_trader.backtest.backtester import BacktestConfig, run_backtest, sweep_thresholds
from ai_trader.backtest.trend_backtester import TrendConfig, run_trend_backtest, sweep_trend_params
from ai_trader.features.trend_signals import compute_trend_signals
from ai_trader.config.settings import load_settings
from ai_trader.data.market_data import load_or_fetch_historical, make_training_data_exchange
from ai_trader.model.train import save_model, walk_forward_train
from ai_trader.risk.risk_manager import RiskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")


def cmd_backtest(args) -> None:
    settings = load_settings()
    exchange = make_training_data_exchange(settings)
    df = load_or_fetch_historical(settings, exchange, refresh=args.refresh)
    logger.info("Loaded %d training candles for %s %s (from %s)", len(df), settings.training_data_symbol, settings.timeframe, settings.training_data_exchange_name)

    result = walk_forward_train(df)
    logger.info(
        "Walk-forward metrics: dev avg_auc=%.3f avg_accuracy=%.3f | HOLDOUT auc=%.3f (n_dev=%d, n_holdout=%d)",
        result.metrics["avg_auc"], result.metrics["avg_accuracy"], result.metrics["holdout_auc"],
        result.metrics["n_dev"], result.metrics["n_holdout"],
    )

    config = BacktestConfig(
        stop_loss_pct=settings.stop_loss_pct,
        take_profit_pct=settings.take_profit_pct,
        position_pct=settings.max_position_pct,
        starting_capital=settings.capital_allocation_usdt,
    )
    bt = run_backtest(result.oos_predictions, config)
    logger.info("Dev-OOS backtest (fees+slippage, default thresholds): %s", bt.metrics)

    sweep = sweep_thresholds(result.oos_predictions, config)
    if sweep.empty:
        logger.warning("Threshold sweep found no configs with >=20 trades on dev data — too little signal.")
    else:
        logger.info("Top 5 threshold configs tuned on DEV data only (not yet validated on holdout):")
        for _, row in sweep.head(5).iterrows():
            logger.info("  %s", row.to_dict())

        best = sweep.iloc[0]
        honest_config = BacktestConfig(
            buy_threshold=float(best["buy_threshold"]),
            sell_threshold=float(best["sell_threshold"]),
            stop_loss_pct=settings.stop_loss_pct,
            take_profit_pct=settings.take_profit_pct,
            position_pct=settings.max_position_pct,
            starting_capital=settings.capital_allocation_usdt,
        )
        honest = run_backtest(result.holdout_predictions, honest_config)
        logger.info(
            "HONEST final check — best dev thresholds (buy=%.2f sell=%.2f) applied to the HOLDOUT "
            "the model never trained or tuned on: %s",
            honest_config.buy_threshold, honest_config.sell_threshold, honest.metrics,
        )
        logger.info("^ This holdout number is the one to trust. Everything above it was tuned on data it's now being checked against.")


def cmd_trend_backtest(args) -> None:
    settings = load_settings()
    exchange = make_training_data_exchange(settings)
    df = load_or_fetch_historical(settings, exchange, refresh=args.refresh)
    logger.info("Loaded %d training candles for %s %s (from %s)", len(df), settings.training_data_symbol, settings.timeframe, settings.training_data_exchange_name)

    df = df.reset_index(drop=True)
    holdout_start = int(len(df) * 0.85)
    dev_df = df.iloc[:holdout_start].reset_index(drop=True)
    holdout_df = df.iloc[holdout_start:].reset_index(drop=True)
    logger.info("Dev candles: %d | Holdout candles: %d (holdout never used for parameter search)", len(dev_df), len(holdout_df))

    base_config = TrendConfig(
        risk_per_trade_pct=settings.risk_per_trade_pct,
        max_position_pct=settings.max_position_pct,
        starting_capital=settings.capital_allocation_usdt,
    )

    sweep = sweep_trend_params(dev_df, base_config)
    if sweep.empty:
        logger.warning("Parameter sweep found no configs with enough trades on dev data.")
        return

    logger.info("Top 5 parameter configs on DEV data (ranked by Sharpe, not yet validated on holdout):")
    for _, row in sweep.head(5).iterrows():
        logger.info("  %s", row.to_dict())

    best = sweep.iloc[0]
    best_signals_holdout = compute_trend_signals(
        holdout_df, entry_lookback=int(best["entry_lookback"]), exit_lookback=int(best["exit_lookback"]),
    )
    honest_cfg = TrendConfig(
        chandelier_mult=float(best["chandelier_mult"]),
        risk_per_trade_pct=settings.risk_per_trade_pct,
        max_position_pct=settings.max_position_pct,
        starting_capital=settings.capital_allocation_usdt,
    )
    honest = run_trend_backtest(best_signals_holdout, honest_cfg)
    logger.info(
        "HONEST final check — best dev params (entry=%d exit=%d chandelier=%.1f) applied to the HOLDOUT "
        "never used for parameter search: %s",
        int(best["entry_lookback"]), int(best["exit_lookback"]), float(best["chandelier_mult"]), honest.metrics,
    )
    logger.info("^ This holdout number is the one to trust.")


def cmd_train(args) -> None:
    from datetime import datetime, timezone

    settings = load_settings()
    exchange = make_training_data_exchange(settings)
    df = load_or_fetch_historical(settings, exchange, refresh=args.refresh)
    result = walk_forward_train(df)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = save_model(result, settings.model_dir, tag)
    logger.info("Saved model to %s (avg_auc=%.3f)", path, result.metrics["avg_auc"])


def cmd_run(args) -> None:
    from ai_trader.core.trader import LiveTrader

    settings = load_settings()
    if not settings.dry_run:
        confirm = input(
            "DRY_RUN=false — this will place REAL orders with REAL money on "
            f"{settings.exchange_name}. Type 'yes' to continue: "
        )
        if confirm.strip().lower() != "yes":
            logger.info("Aborted.")
            return
    trader = LiveTrader(settings)
    trader.run_forever()


def cmd_scan(args) -> None:
    from ai_trader.scan.run_scan import run_scan
    from ai_trader.web.dashboard import save_dashboard

    settings = load_settings()
    scan_data = run_scan(settings)
    out_dir = settings.model_dir.parent / "site"
    path = save_dashboard(scan_data, out_dir)
    logger.info("Dashboard written to %s — open it directly in your browser.", path)


def cmd_signal(args) -> None:
    from ai_trader.core.daily_signal import run_and_save

    run_and_save()


def cmd_smoke_test(args) -> None:
    from ai_trader.core.smoke_test import run

    run()


def cmd_reset_risk(args) -> None:
    settings = load_settings()
    risk = RiskManager(settings)
    can_trade, reason = risk.can_trade()
    if can_trade:
        logger.info("Risk manager is not halted — nothing to reset.")
        return
    logger.warning("Current halt reason: %s", reason)
    confirm = input("Confirm you've investigated this and want to resume trading. Type 'yes': ")
    if confirm.strip().lower() == "yes":
        risk.manual_reset()
        logger.info("Risk manager reset. Trading will resume on next tick.")
    else:
        logger.info("Aborted.")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI crypto trading bot")
    sub = parser.add_subparsers(dest="command", required=True)

    p_backtest = sub.add_parser("backtest", help="Walk-forward train + honest out-of-sample backtest")
    p_backtest.add_argument("--refresh", action="store_true", help="Re-fetch full history instead of using cache")
    p_backtest.set_defaults(func=cmd_backtest)

    p_trend = sub.add_parser("trend-backtest", help="Backtest the Donchian breakout + EMA trend-filter strategy")
    p_trend.add_argument("--refresh", action="store_true")
    p_trend.set_defaults(func=cmd_trend_backtest)

    p_train = sub.add_parser("train", help="Train and save a model version")
    p_train.add_argument("--refresh", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_run = sub.add_parser("run", help="Start the live/paper trading loop")
    p_run.set_defaults(func=cmd_run)

    p_scan = sub.add_parser("scan", help="Scan Kraken coin universe, build honest dashboard (site/index.html)")
    p_scan.set_defaults(func=cmd_scan)

    p_signal = sub.add_parser("signal", help="Generate today's trade recommendation (no orders placed)")
    p_signal.set_defaults(func=cmd_signal)

    p_smoke = sub.add_parser("smoke-test", help="Dry-run plumbing check: orders, precision, kill switch")
    p_smoke.set_defaults(func=cmd_smoke_test)

    p_reset = sub.add_parser("reset-risk", help="Clear a tripped kill switch")
    p_reset.set_defaults(func=cmd_reset_risk)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
