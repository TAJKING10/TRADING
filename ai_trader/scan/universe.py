"""Which coins to scan: every coin that is BOTH liquid-enough to trade on
Kraken (a real USD spot pair) AND has deep history on the training-data
exchange (TRAINING_DATA_EXCHANGE, KuCoin by default) for honest backtesting.
Dynamically resolved from both exchanges' live market lists every run — not
a fixed list — so new Kraken listings get picked up automatically. Coins
without training-exchange history are skipped rather than backtested
dishonestly; stablecoins/fiat are excluded since their "price prediction"
is meaningless by construction.
"""
import logging

import ccxt

logger = logging.getLogger("universe")

EXCLUDED_BASES = {
    # Stablecoins / fiat-pegged — no meaningful price direction to predict.
    "USDT", "USDC", "USDE", "USDS", "USD1", "DAI", "TUSD", "PYUSD", "USDP",
    "EUR", "EURT", "GBP", "RLUSD",
}


def resolve_universe(kraken: ccxt.Exchange, training_exchange: ccxt.Exchange) -> list[dict]:
    resolved = []
    skipped = 0
    for market in kraken.markets.values():
        if not market.get("active") or not market.get("spot") or market.get("quote") != "USD":
            continue
        base = market.get("base")
        if not base or base in EXCLUDED_BASES:
            continue

        training_symbol = f"{base}/USDT"
        training_market = training_exchange.markets.get(training_symbol)
        if not training_market or not training_market.get("active") or not training_market.get("spot"):
            skipped += 1
            continue

        resolved.append({"base": base, "kraken_symbol": f"{base}/USD", "training_symbol": training_symbol})

    resolved.sort(key=lambda c: c["base"])
    logger.info(
        "Universe: %d coins tradeable on Kraken/USD with training-exchange/USDT history (%d Kraken/USD pairs skipped — no counterpart)",
        len(resolved), skipped,
    )
    return resolved
