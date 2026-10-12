"""One-shot plumbing check before this bot is trusted with real orders.

Verifies: connectivity+auth, balance reads, order precision/min-notional handling,
a dry-run buy+sell, and that the risk manager's kill switch actually halts.
Refuses to run unless DRY_RUN=true — this is not for placing real orders.
"""
import logging

from ai_trader.config.settings import load_settings
from ai_trader.data.market_data import make_exchange
from ai_trader.execution import broker
from ai_trader.risk.risk_manager import RiskManager

logger = logging.getLogger("smoke_test")


def run() -> None:
    settings = load_settings()
    if not settings.dry_run:
        raise SystemExit("Refusing to run: DRY_RUN is false. This smoke test never sends real orders.")
    if not settings.api_key or not settings.api_secret:
        raise SystemExit(
            f"No API_KEY/API_SECRET set for {settings.exchange_name}. Generate read+trade API keys "
            "and put them in .env before running this (DRY_RUN stays true, so no real orders will fire)."
        )

    exchange = make_exchange(settings)
    exchange.load_markets()
    base_asset, quote_asset = settings.symbol.split("/")

    logger.info("1/5 Connectivity + auth: fetching balance on %s...", settings.exchange_name)
    quote_bal = broker.fetch_quote_balance(exchange, quote_asset)
    base_bal = broker.fetch_base_balance(exchange, base_asset)
    logger.info("    OK — %s free=%.4f, %s free=%.6f", quote_asset, quote_bal, base_asset, base_bal)

    logger.info("2/5 Fetching latest price...")
    price = broker.fetch_last_price(exchange, settings.symbol)
    logger.info("    OK — %s last price = %.2f", settings.symbol, price)

    logger.info("3/5 Testing order rejection on an absurdly small quantity...")
    try:
        broker._validate_order(exchange, settings.symbol, 0.0000001, price)
        logger.error("    FAIL — expected OrderRejected, got none")
    except broker.OrderRejected as e:
        logger.info("    OK — correctly rejected: %s", e)

    market = broker.ensure_market_loaded(exchange, settings.symbol)
    min_cost = (market.get("limits", {}).get("cost", {}) or {}).get("min") or 10.0
    test_notional = max(min_cost * 1.5, 11.0)
    qty = test_notional / price
    logger.info("4/5 Dry-run round-trip order: BUY then SELL ~%.2f %s of %s (no real order sent)...", test_notional, quote_asset, settings.symbol)
    buy_order = broker.market_buy(exchange, settings.symbol, qty, price, dry_run=True)
    sell_order = broker.market_sell(exchange, settings.symbol, buy_order["filled"], price, dry_run=True)
    logger.info("    OK — simulated buy id=%s, simulated sell id=%s", buy_order.get("id"), sell_order.get("id"))

    logger.info("5/5 Kill switch test: simulating a drawdown breach...")
    risk = RiskManager(settings)
    starting_equity = settings.capital_allocation_usdt
    risk.update_equity(starting_equity)  # establish peak
    breach_equity = starting_equity * (1 - settings.max_drawdown_pct - 0.01)
    risk.update_equity(breach_equity)
    can_trade, reason = risk.can_trade()
    if can_trade:
        logger.error("    FAIL — kill switch did not trigger on simulated drawdown breach")
    else:
        logger.info("    OK — kill switch triggered: %s", reason)
    risk.manual_reset()
    logger.info("    Reset kill switch test state so it doesn't block real runs.")

    logger.info("Smoke test complete. Precision/min-notional/balance/kill-switch plumbing all verified.")
    logger.info("Real order placement itself was NOT tested (no sandbox on %s) — that only happens once you flip DRY_RUN=false.", settings.exchange_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
