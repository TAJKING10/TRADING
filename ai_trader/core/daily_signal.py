"""Daily advisory digest: generates a trade recommendation for a human to execute.

Does not place any orders. Reads real balances/trade history from the exchange
(read-only) to figure out whether you're currently holding a position and what
you likely paid for it, then applies the same Donchian breakout + EMA trend
filter strategy used by the dashboard scan — NOT the ML classifier, which
failed honest holdout validation. These are default rule parameters, not
necessarily a validated edge for this specific symbol; check the dashboard
or `trend-backtest` for that.
"""
import logging
from datetime import datetime, timezone

from ai_trader.config.settings import Settings, load_settings
from ai_trader.data.market_data import make_exchange, fetch_latest_candles
from ai_trader.execution import broker
from ai_trader.features.trend_signals import compute_trend_signals
from ai_trader.risk.risk_manager import RiskManager
from ai_trader.core.notify import send_whatsapp

logger = logging.getLogger("daily_signal")

ENTRY_LOOKBACK = 20
EXIT_LOOKBACK = 10
TREND_EMA_FAST = 50
TREND_EMA_SLOW = 200
CHANDELIER_MULT = 3.0
DUST_NOTIONAL = 5.0  # below this $ value, treat base balance as "not really a position"


def _approx_entry_price(exchange, symbol: str, base_balance: float) -> float | None:
    """Best-effort average entry price from recent trade history. Returns None if
    the exchange call fails or there's not enough history to tell — callers must
    handle that (don't silently assume a wrong entry price)."""
    try:
        trades = exchange.fetch_my_trades(symbol, limit=100)
    except Exception as e:
        logger.warning("Could not fetch trade history for entry-price estimate: %s", e)
        return None

    if not trades:
        return None

    remaining = base_balance
    cost = 0.0
    qty_accounted = 0.0
    for t in reversed(trades):  # most recent first
        if remaining <= 0:
            break
        side = t.get("side")
        amount = float(t.get("amount") or 0)
        price = float(t.get("price") or 0)
        if side == "buy" and amount > 0 and price > 0:
            take = min(amount, remaining)
            cost += take * price
            qty_accounted += take
            remaining -= take

    if qty_accounted <= 0:
        return None
    return cost / qty_accounted


def generate_digest(settings: Settings) -> str:
    if not settings.api_key or not settings.api_secret:
        return (
            f"No API_KEY/API_SECRET set for {settings.exchange_name}. Generate API keys "
            "(read + trade permissions) from your exchange account and put them in .env. "
            "DRY_RUN can stay true — reading balances/prices needs auth, but no orders are ever placed here."
        )

    exchange = make_exchange(settings)
    exchange.load_markets()
    base_asset, quote_asset = settings.symbol.split("/")

    candles = fetch_latest_candles(exchange, settings.symbol, settings.timeframe, limit=max(300, TREND_EMA_SLOW + 20))
    candles = candles.iloc[:-1]  # drop still-open candle
    signals = compute_trend_signals(
        candles, entry_lookback=ENTRY_LOOKBACK, exit_lookback=EXIT_LOOKBACK,
        trend_ema_fast=TREND_EMA_FAST, trend_ema_slow=TREND_EMA_SLOW,
    ).dropna(subset=["donchian_high", "donchian_low_exit", "ema_slow", "atr"])
    if signals.empty:
        return "Not enough market data to compute a signal right now."

    last = signals.iloc[-1]
    price = float(last["close"])
    atr = float(last["atr"])

    quote_bal = broker.fetch_quote_balance(exchange, quote_asset)
    base_bal = broker.fetch_base_balance(exchange, base_asset)
    base_notional = base_bal * price
    holding = base_notional >= DUST_NOTIONAL

    risk = RiskManager(settings)
    equity = quote_bal + base_notional
    risk.update_equity(equity)
    can_trade, halt_reason = risk.can_trade()

    lines = []
    lines.append(f"# Daily trade signal — {settings.symbol} ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})")
    lines.append("")
    lines.append(f"Price: {price:,.2f} {quote_asset} | Trend: {'UP' if bool(last['trend_up']) else 'DOWN'} | Equity: {equity:,.2f} {quote_asset}")
    lines.append("_Strategy: Donchian breakout + EMA trend filter (default params — check the dashboard for whether this symbol has shown a validated edge)._")
    lines.append("")

    if not can_trade:
        lines.append(f"**ACTION: DO NOT TRADE.** Kill switch is active: {halt_reason}")
        lines.append("Run `python main.py reset-risk` only after you understand why this tripped.")
        return "\n".join(lines)

    if holding:
        entry = _approx_entry_price(exchange, settings.symbol, base_bal)
        if entry:
            change = price / entry - 1
            lines.append(f"You appear to hold ~{base_bal:.6f} {base_asset} (est. entry {entry:,.2f}, {change:+.2%}).")
            trend_flipped = not bool(last["trend_up"])
            donchian_exit = price < float(last["donchian_low_exit"])
            if trend_flipped:
                lines.append("**ACTION: SELL.** Trend filter flipped bearish (EMA50 crossed below EMA200).")
            elif donchian_exit:
                lines.append(f"**ACTION: SELL.** Price broke below the {EXIT_LOOKBACK}-period low.")
            else:
                lines.append(f"**ACTION: HOLD.** Trend intact. Manually track your own chandelier stop: "
                              f"highest close since entry minus {CHANDELIER_MULT}x ATR (ATR currently {atr:,.2f}).")
        else:
            lines.append(f"You appear to hold ~{base_bal:.6f} {base_asset} but entry price couldn't be determined from trade history.")
            lines.append(f"**ACTION: use your own judgment on stop-loss; trend is currently "
                          f"{'UP' if bool(last['trend_up']) else 'DOWN'}.**")
    else:
        if bool(last["breakout_long"]):
            stop_distance = CHANDELIER_MULT * atr
            qty = min(
                (equity * settings.risk_per_trade_pct) / stop_distance if stop_distance > 0 else 0,
                (equity * settings.max_position_pct) / price,
            )
            notional = qty * price
            stop_price = price - stop_distance
            lines.append(f"**ACTION: BUY.** Breakout above the {ENTRY_LOOKBACK}-period high, trend confirmed up.")
            lines.append(f"  Suggested size: {qty:.6f} {base_asset} (~{notional:,.2f} {quote_asset})")
            lines.append(f"  Initial chandelier stop: {stop_price:,.2f} (trails up with price)")
        else:
            donchian_level = last["donchian_high"]
            lines.append(f"**ACTION: WAIT.** No breakout yet. Watching for a close above {donchian_level:,.2f}.")

    lines.append("")
    lines.append("_This is a rule-based signal, not a guarantee. Size and risk at your own discretion._")
    return "\n".join(lines)


def run_and_save() -> str:
    settings = load_settings()
    digest = generate_digest(settings)
    out_dir = settings.log_dir / "digests"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.md"
    path.write_text(digest, encoding="utf-8")
    logger.info("Digest written to %s", path)
    print(digest)

    if settings.whatsapp_phone and settings.whatsapp_apikey:
        send_whatsapp(settings.whatsapp_phone, settings.whatsapp_apikey, digest)
    else:
        logger.info("WHATSAPP_PHONE/WHATSAPP_APIKEY not set — skipping WhatsApp delivery (digest file still written).")

    return digest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_and_save()
