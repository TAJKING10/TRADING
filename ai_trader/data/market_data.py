"""Historical + live OHLCV access via ccxt. Exchange-agnostic (Kraken, Binance, ...)."""
import time
from pathlib import Path

import ccxt
import pandas as pd

from ai_trader.config.settings import Settings

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _build(exchange_name: str, api_key: str = "", api_secret: str = "") -> ccxt.Exchange:
    exchange_class = getattr(ccxt, exchange_name, None)
    if exchange_class is None:
        raise ValueError(f"Unknown exchange '{exchange_name}' (not a ccxt exchange id)")
    return exchange_class(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        }
    )


def make_exchange(settings: Settings) -> ccxt.Exchange:
    """Execution exchange — real keys live here. DRY_RUN (checked in broker.py) decides
    whether orders actually get sent. USE_TESTNET only applies on exchanges with a sandbox
    (e.g. Binance) — Kraken has none, so dry_run is the real safety net there."""
    exchange = _build(settings.exchange_name, settings.api_key, settings.api_secret)
    if settings.use_testnet:
        try:
            exchange.set_sandbox_mode(True)
        except ccxt.NotSupported:
            raise ValueError(
                f"{settings.exchange_name} has no sandbox/testnet support. "
                "Set USE_TESTNET=false and rely on DRY_RUN=true instead."
            )
    return exchange


def make_training_data_exchange(settings: Settings) -> ccxt.Exchange:
    """Deep-history exchange for training data only — never used for live prices/orders.

    Kraken's public OHLCV caps out around ~720 candles with no real historical paging,
    so training data comes from TRAINING_DATA_EXCHANGE (KuCoin by default — Binance
    blocks GitHub Actions' datacenter IPs with HTTP 451), which does support deep
    pagination. No keys needed (public endpoint).
    """
    return _build(settings.training_data_exchange_name)


def _cache_path(settings: Settings) -> Path:
    safe_symbol = settings.training_data_symbol.replace("/", "")
    return settings.model_dir.parent / "data_cache" / f"{settings.training_data_exchange_name}_{safe_symbol}_{settings.timeframe}.parquet"


def fetch_historical(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int | None = None,
    limit_per_call: int = 500,
    max_candles: int = 300_000,
    lookback_days: int = 1095,
) -> pd.DataFrame:
    """Page through ccxt's OHLCV endpoint to build as full a historical dataset as the
    exchange will give us.

    With since=None, ccxt returns only the most recent `limit` candles and pagination
    can't walk further back (cursor would run past "now" and return nothing) — so
    default to a real lookback window instead of None.
    """
    if since_ms is None:
        since_ms = exchange.milliseconds() - lookback_days * 86_400_000

    all_rows = []
    cursor = since_ms
    while len(all_rows) < max_candles:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit_per_call)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if cursor is not None and last_ts <= cursor:
            break
        cursor = last_ts + 1
        if len(batch) < limit_per_call:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def load_or_fetch_historical(settings: Settings, exchange: ccxt.Exchange, refresh: bool = False) -> pd.DataFrame:
    cache = _cache_path(settings)
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists() and not refresh:
        existing = pd.read_parquet(cache)
        since_ms = int(existing["timestamp"].iloc[-1].timestamp() * 1000) + 1
        fresh = fetch_historical(exchange, settings.training_data_symbol, settings.timeframe, since_ms=since_ms)
        if not fresh.empty:
            combined = (
                pd.concat([existing, fresh])
                .drop_duplicates(subset="timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
        else:
            combined = existing
    else:
        combined = fetch_historical(
            exchange, settings.training_data_symbol, settings.timeframe, lookback_days=settings.history_lookback_days
        )

    combined.to_parquet(cache, index=False)
    return combined


def fetch_latest_candles(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    """Pull just enough recent candles for live feature computation."""
    batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(batch, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df
