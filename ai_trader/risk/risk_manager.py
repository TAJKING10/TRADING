"""Position sizing + kill switch. This is what stands between a bad model and a wiped account.

State persists to disk so a crash-and-restart cannot silently reset the kill switch.
"""
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ai_trader.config.settings import Settings

logger = logging.getLogger("risk_manager")


@dataclass
class RiskState:
    peak_equity: float
    day_start_equity: float
    day_start_date: str  # ISO date (UTC)
    hard_halted: bool = False
    hard_halt_reason: str = ""


class RiskManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.state_path = settings.log_dir / "risk_state.json"
        self.state = self._load_or_init_state()

    def _load_or_init_state(self) -> RiskState:
        if self.state_path.exists():
            with open(self.state_path) as f:
                data = json.load(f)
            return RiskState(**data)
        today = datetime.now(timezone.utc).date().isoformat()
        state = RiskState(
            peak_equity=self.settings.capital_allocation_usdt,
            day_start_equity=self.settings.capital_allocation_usdt,
            day_start_date=today,
        )
        self._save(state)
        return state

    def _save(self, state: RiskState) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(state.__dict__, f, indent=2)

    def _roll_day_if_needed(self, equity: float) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        if self.state.day_start_date != today:
            self.state.day_start_date = today
            self.state.day_start_equity = equity

    def update_equity(self, equity: float) -> None:
        """Call this every loop tick with current mark-to-market equity."""
        self._roll_day_if_needed(equity)
        self.state.peak_equity = max(self.state.peak_equity, equity)

        drawdown_pct = (self.state.peak_equity - equity) / self.state.peak_equity
        daily_loss_pct = (self.state.day_start_equity - equity) / self.state.day_start_equity

        if drawdown_pct >= self.settings.max_drawdown_pct and not self.state.hard_halted:
            self.state.hard_halted = True
            self.state.hard_halt_reason = (
                f"Max drawdown breached: {drawdown_pct:.2%} >= {self.settings.max_drawdown_pct:.2%}"
            )
            logger.error("KILL SWITCH TRIGGERED: %s", self.state.hard_halt_reason)

        if daily_loss_pct >= self.settings.daily_loss_limit_pct and not self.state.hard_halted:
            self.state.hard_halted = True
            self.state.hard_halt_reason = (
                f"Daily loss limit breached: {daily_loss_pct:.2%} >= {self.settings.daily_loss_limit_pct:.2%}"
            )
            logger.error("KILL SWITCH TRIGGERED: %s", self.state.hard_halt_reason)

        self._save(self.state)

    def can_trade(self) -> tuple[bool, str]:
        if self.state.hard_halted:
            return False, self.state.hard_halt_reason
        return True, ""

    def manual_reset(self) -> None:
        """Explicit human action required to resume after a kill-switch halt."""
        self.state.hard_halted = False
        self.state.hard_halt_reason = ""
        self.state.peak_equity = max(self.state.peak_equity, self.settings.capital_allocation_usdt)
        self._save(self.state)
        logger.warning("Risk manager manually reset — trading resumed.")

    def position_size(self, equity: float, entry_price: float) -> float:
        """Quantity to buy so a stop-loss hit loses exactly risk_per_trade_pct of equity,
        capped so the position never exceeds max_position_pct of equity."""
        risk_amount = equity * self.settings.risk_per_trade_pct
        stop_distance = entry_price * self.settings.stop_loss_pct
        qty_by_risk = risk_amount / stop_distance

        max_notional = equity * self.settings.max_position_pct
        qty_by_cap = max_notional / entry_price

        return min(qty_by_risk, qty_by_cap)
