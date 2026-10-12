"""Live/paper trading loop: fetch -> trend signal -> risk-check -> execute.

Uses the same Donchian breakout + EMA trend filter + chandelier exit strategy
as the dashboard scan — NOT the ML classifier (that one failed honest holdout
validation, see backtest results). These are DEFAULT rule parameters, not
necessarily the validated-edge ones for this specific symbol — check
`python main.py trend-backtest` or the dashboard for this symbol before
trusting it with real money.

Runs forever, ticking once per closed candle. Safe to Ctrl+C and restart —
risk state and position state are persisted to disk.
"""
import logging
import re
import time

from ai_trader.config.settings import Settings
from ai_trader.data.market_data import make_exchange, fetch_latest_candles
from ai_trader.execution import broker
from ai_trader.features.trend_signals import compute_trend_signals
from ai_trader.risk.risk_manager import RiskManager
from ai_trader.core.position_state import load_position, save_position

logger = logging.getLogger("trader")

ENTRY_LOOKBACK = 20
EXIT_LOOKBACK = 10
TREND_EMA_FAST = 50
TREND_EMA_SLOW = 200
CHANDELIER_MULT = 3.0


def timeframe_to_seconds(timeframe: str) -> int:
    match = re.match(r"(\d+)([mhd])", timeframe)
    if not match:
        raise ValueError(f"Unrecognized timeframe: {timeframe}")
    value, unit = int(match.group(1)), match.group(2)
    return value * {"m": 60, "h": 3600, "d": 86400}[unit]


class LiveTrader:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.exchange = make_exchange(settings)  # real keys live here; dry_run decides if orders actually send
        self.exchange.load_markets()
        self.risk = RiskManager(settings)
        self.position = load_position(settings.log_dir)
        self.base_asset, self.quote_asset = settings.symbol.split("/")
        self.tick_seconds = timeframe_to_seconds(settings.timeframe)

        mode = "DRY RUN (no real orders)" if settings.dry_run else "LIVE REAL MONEY"
        logger.warning("LiveTrader starting in %s mode on %s (%s)", mode, settings.symbol, settings.exchange_name)
        logger.warning(
            "Using default trend-following params (entry=%d exit=%d chandelier=%.1f) — "
            "run `trend-backtest` for %s specifically to see if these are actually validated for it.",
            ENTRY_LOOKBACK, EXIT_LOOKBACK, CHANDELIER_MULT, settings.symbol,
        )

    def current_equity(self, price: float) -> float:
        quote_bal = broker.fetch_quote_balance(self.exchange, self.quote_asset)
        base_bal = broker.fetch_base_balance(self.exchange, self.base_asset)
        return quote_bal + base_bal * price

    def tick(self) -> None:
        candles = fetch_latest_candles(self.exchange, self.settings.symbol, self.settings.timeframe, limit=max(300, TREND_EMA_SLOW + 20))
        candles = candles.iloc[:-1]  # drop the still-open candle
        signals = compute_trend_signals(
            candles, entry_lookback=ENTRY_LOOKBACK, exit_lookback=EXIT_LOOKBACK,
            trend_ema_fast=TREND_EMA_FAST, trend_ema_slow=TREND_EMA_SLOW,
        ).dropna(subset=["donchian_high", "donchian_low_exit", "ema_slow", "atr"])
        if signals.empty:
            logger.warning("Not enough data to compute trend signals — skipping tick.")
            return

        last = signals.iloc[-1]
        price = float(last["close"])

        equity = self.current_equity(price)
        self.risk.update_equity(equity)
        can_trade, halt_reason = self.risk.can_trade()

        logger.info(
            "tick price=%.2f trend_up=%s breakout=%s equity=%.2f in_position=%s can_trade=%s",
            price, bool(last["trend_up"]), bool(last["breakout_long"]), equity, self.position.in_position, can_trade,
        )

        if not can_trade:
            logger.error("Trading halted by risk manager: %s", halt_reason)
            return

        if self.position.in_position:
            self.position.highest_close_since_entry = max(self.position.highest_close_since_entry, price)
            chandelier_stop = self.position.highest_close_since_entry - CHANDELIER_MULT * float(last["atr"])
            trend_flipped = not bool(last["trend_up"])
            donchian_exit = price < float(last["donchian_low_exit"])
            hit_stop = price <= chandelier_stop

            if hit_stop or trend_flipped or donchian_exit:
                reason = "chandelier_stop" if hit_stop else "trend_flip" if trend_flipped else "donchian_exit"
                try:
                    broker.market_sell(self.exchange, self.settings.symbol, self.position.qty, price, dry_run=self.settings.dry_run)
                    pnl_pct = price / self.position.entry_price - 1
                    logger.warning("SOLD (%s): pnl_pct=%.2f%%", reason, pnl_pct * 100)
                    self.position.in_position = False
                    self.position.entry_price = None
                    self.position.qty = None
                    self.position.highest_close_since_entry = None
                    save_position(self.settings.log_dir, self.position)
                except broker.OrderRejected as e:
                    logger.error("Sell order rejected: %s", e)
            else:
                save_position(self.settings.log_dir, self.position)
        else:
            if bool(last["breakout_long"]):
                stop_distance = CHANDELIER_MULT * float(last["atr"])
                qty = self.risk.position_size(equity, price) if stop_distance <= 0 else min(
                    (equity * self.settings.risk_per_trade_pct) / stop_distance,
                    (equity * self.settings.max_position_pct) / price,
                )
                try:
                    broker.market_buy(self.exchange, self.settings.symbol, qty, price, dry_run=self.settings.dry_run)
                    logger.warning("BOUGHT qty=%.6f at ~%.2f (breakout)", qty, price)
                    self.position.in_position = True
                    self.position.entry_price = price
                    self.position.qty = qty
                    self.position.highest_close_since_entry = price
                    save_position(self.settings.log_dir, self.position)
                except broker.OrderRejected as e:
                    logger.error("Buy order rejected: %s", e)

    def run_forever(self) -> None:
        while True:
            try:
                self.tick()
            except Exception:
                logger.exception("Tick failed — will retry next interval.")
            time.sleep(self.tick_seconds)


def main() -> None:
    from ai_trader.config.settings import load_settings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    trader = LiveTrader(settings)
    trader.run_forever()


if __name__ == "__main__":
    main()
