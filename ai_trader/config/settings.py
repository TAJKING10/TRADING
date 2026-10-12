import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


@dataclass(frozen=True)
class Settings:
    exchange_name: str
    training_data_exchange_name: str
    api_key: str
    api_secret: str
    use_testnet: bool
    dry_run: bool

    symbol: str
    training_data_symbol: str
    timeframe: str
    history_lookback_days: int

    capital_allocation_usdt: float
    risk_per_trade_pct: float
    max_position_pct: float
    daily_loss_limit_pct: float
    max_drawdown_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    taker_fee_pct: float
    slippage_pct: float

    retrain_interval_hours: float
    model_dir: Path
    log_dir: Path

    whatsapp_phone: str
    whatsapp_apikey: str

    supabase_url: str
    supabase_service_role_key: str

    def validate(self) -> None:
        if not self.dry_run and (not self.api_key or not self.api_secret):
            raise ValueError(
                "DRY_RUN is false but API_KEY/API_SECRET are missing. "
                "Refusing to start a real-order session without credentials."
            )
        if self.capital_allocation_usdt <= 0:
            raise ValueError("CAPITAL_ALLOCATION_USDT must be positive.")
        if not (0 < self.risk_per_trade_pct <= 0.1):
            raise ValueError("RISK_PER_TRADE_PCT must be in (0, 0.1] — refusing larger single-trade risk.")
        if not (0 < self.daily_loss_limit_pct <= 0.2):
            raise ValueError("DAILY_LOSS_LIMIT_PCT must be in (0, 0.2].")
        if not (0 < self.max_drawdown_pct <= 0.3):
            raise ValueError("MAX_DRAWDOWN_PCT must be in (0, 0.3].")


def load_settings() -> Settings:
    model_dir = ROOT_DIR / os.getenv("MODEL_DIR", "artifacts")
    log_dir = ROOT_DIR / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        exchange_name=os.getenv("EXCHANGE", "kraken").strip().lower(),
        training_data_exchange_name=os.getenv("TRAINING_DATA_EXCHANGE", "kucoin").strip().lower(),
        api_key=os.getenv("API_KEY", ""),
        api_secret=os.getenv("API_SECRET", ""),
        use_testnet=_bool("USE_TESTNET", False),
        dry_run=_bool("DRY_RUN", True),
        symbol=os.getenv("SYMBOL", "BTC/USD"),
        training_data_symbol=os.getenv("TRAINING_DATA_SYMBOL", "BTC/USDT"),
        timeframe=os.getenv("TIMEFRAME", "15m"),
        history_lookback_days=_int("HISTORY_LOOKBACK_DAYS", 1095),
        capital_allocation_usdt=_float("CAPITAL_ALLOCATION_USDT", 100.0),
        risk_per_trade_pct=_float("RISK_PER_TRADE_PCT", 0.01),
        max_position_pct=_float("MAX_POSITION_PCT", 0.5),
        daily_loss_limit_pct=_float("DAILY_LOSS_LIMIT_PCT", 0.03),
        max_drawdown_pct=_float("MAX_DRAWDOWN_PCT", 0.10),
        stop_loss_pct=_float("STOP_LOSS_PCT", 0.015),
        take_profit_pct=_float("TAKE_PROFIT_PCT", 0.03),
        # Kraken base-tier spot taker fee is 0.80% (Oct 2026) — not Binance's
        # 0.1%, which this used to (incorrectly) assume. Override to match
        # your actual account's volume tier once you know it.
        taker_fee_pct=_float("TAKER_FEE_PCT", 0.008),
        slippage_pct=_float("SLIPPAGE_PCT", 0.0005),
        retrain_interval_hours=_float("RETRAIN_INTERVAL_HOURS", 24.0),
        model_dir=model_dir,
        log_dir=log_dir,
        whatsapp_phone=os.getenv("WHATSAPP_PHONE", ""),
        whatsapp_apikey=os.getenv("WHATSAPP_APIKEY", ""),
        supabase_url=os.getenv("SUPABASE_URL", ""),
        supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
    )
    settings.validate()
    return settings
