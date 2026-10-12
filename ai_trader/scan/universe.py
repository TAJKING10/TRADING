"""Which coins to scan: must be liquid on Kraken (where you'd actually trade)
AND have deep history on the training-data exchange (Binance) for honest
backtesting. A curated list, not "literally every pair Kraken lists" — most
of Kraken's long tail is too illiquid to backtest or trade meaningfully, and
scanning hundreds of pairs would take hours per run for no real benefit.
"""
import logging

import ccxt

logger = logging.getLogger("universe")

CANDIDATE_BASES = [
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "LINK",
    "DOT", "AVAX", "MATIC", "UNI", "ATOM", "XLM", "BCH", "ETC",
]


def resolve_universe(kraken: ccxt.Exchange, binance: ccxt.Exchange) -> list[dict]:
    resolved = []
    skipped = []
    for base in CANDIDATE_BASES:
        kraken_symbol = f"{base}/USD"
        binance_symbol = f"{base}/USDT"
        if kraken_symbol in kraken.markets and binance_symbol in binance.markets:
            resolved.append({"base": base, "kraken_symbol": kraken_symbol, "binance_symbol": binance_symbol})
        else:
            skipped.append(base)

    if skipped:
        logger.warning("Skipped (not listed on both exchanges as expected): %s", skipped)
    return resolved
