"""Which coins to scan: every coin that is BOTH liquid-enough to trade on
Kraken (a real USD spot pair) AND has deep history on the training-data
exchange (Binance) for honest backtesting. Dynamically resolved from both
exchanges' live market lists every run — not a fixed list — so new Kraken
listings get picked up automatically. Coins without Binance history are
skipped rather than backtested dishonestly; stablecoins/fiat are excluded
since their "price prediction" is meaningless by construction.
"""
import logging

import ccxt

logger = logging.getLogger("universe")

EXCLUDED_BASES = {
    # Stablecoins / fiat-pegged — no meaningful price direction to predict.
    "USDT", "USDC", "USDE", "USDS", "USD1", "DAI", "TUSD", "PYUSD", "USDP",
    "EUR", "EURT", "GBP", "RLUSD",
}


def resolve_universe(kraken: ccxt.Exchange, binance: ccxt.Exchange) -> list[dict]:
    resolved = []
    skipped = 0
    for market in kraken.markets.values():
        if not market.get("active") or not market.get("spot") or market.get("quote") != "USD":
            continue
        base = market.get("base")
        if not base or base in EXCLUDED_BASES:
            continue

        binance_symbol = f"{base}/USDT"
        binance_market = binance.markets.get(binance_symbol)
        if not binance_market or not binance_market.get("active") or not binance_market.get("spot"):
            skipped += 1
            continue

        resolved.append({"base": base, "kraken_symbol": f"{base}/USD", "binance_symbol": binance_symbol})

    resolved.sort(key=lambda c: c["base"])
    logger.info(
        "Universe: %d coins tradeable on Kraken/USD with Binance/USDT history (%d Kraken/USD pairs skipped — no Binance counterpart)",
        len(resolved), skipped,
    )
    return resolved
