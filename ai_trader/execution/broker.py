"""Order execution wrapper. Every order passes through precision/min-notional
checks — this is the layer where a rounding bug would otherwise burn money.
"""
import logging

import ccxt

logger = logging.getLogger("broker")


class OrderRejected(Exception):
    pass


def ensure_market_loaded(exchange: ccxt.Exchange, symbol: str) -> dict:
    if not exchange.markets:
        exchange.load_markets()
    return exchange.markets[symbol]


def fetch_last_price(exchange: ccxt.Exchange, symbol: str) -> float:
    ticker = exchange.fetch_ticker(symbol)
    return float(ticker["last"])


def fetch_quote_balance(exchange: ccxt.Exchange, quote_asset: str) -> float:
    balance = exchange.fetch_balance()
    return float(balance.get(quote_asset, {}).get("free", 0.0))


def fetch_base_balance(exchange: ccxt.Exchange, base_asset: str) -> float:
    balance = exchange.fetch_balance()
    return float(balance.get(base_asset, {}).get("free", 0.0))


def _validate_order(exchange: ccxt.Exchange, symbol: str, qty: float, price: float) -> float:
    market = ensure_market_loaded(exchange, symbol)
    qty = float(exchange.amount_to_precision(symbol, qty))

    min_amount = (market.get("limits", {}).get("amount", {}) or {}).get("min")
    if min_amount and qty < min_amount:
        raise OrderRejected(f"Quantity {qty} below exchange minimum {min_amount} for {symbol}")

    notional = qty * price
    min_cost = (market.get("limits", {}).get("cost", {}) or {}).get("min")
    if min_cost and notional < min_cost:
        raise OrderRejected(f"Order notional {notional:.2f} below exchange minimum {min_cost} for {symbol}")

    return qty


def _simulated_order(symbol: str, side: str, qty: float, price: float) -> dict:
    return {
        "id": "dry-run",
        "symbol": symbol,
        "side": side,
        "type": "market",
        "amount": qty,
        "price": price,
        "filled": qty,
        "status": "closed",
        "dry_run": True,
    }


def market_buy(exchange: ccxt.Exchange, symbol: str, qty: float, reference_price: float, dry_run: bool = True) -> dict:
    qty = _validate_order(exchange, symbol, qty, reference_price)
    if dry_run:
        logger.warning("[DRY RUN] Would place MARKET BUY %s qty=%s at ~%.2f — no order sent", symbol, qty, reference_price)
        return _simulated_order(symbol, "buy", qty, reference_price)
    logger.info("Placing MARKET BUY %s qty=%s (symbol=%s)", symbol, qty, symbol)
    order = exchange.create_order(symbol, "market", "buy", qty)
    logger.info("Buy order result: %s", order)
    return order


def market_sell(exchange: ccxt.Exchange, symbol: str, qty: float, reference_price: float, dry_run: bool = True) -> dict:
    qty = _validate_order(exchange, symbol, qty, reference_price)
    if dry_run:
        logger.warning("[DRY RUN] Would place MARKET SELL %s qty=%s at ~%.2f — no order sent", symbol, qty, reference_price)
        return _simulated_order(symbol, "sell", qty, reference_price)
    logger.info("Placing MARKET SELL %s qty=%s (symbol=%s)", symbol, qty, symbol)
    order = exchange.create_order(symbol, "market", "sell", qty)
    logger.info("Sell order result: %s", order)
    return order
