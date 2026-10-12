import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class PositionState:
    in_position: bool = False
    entry_price: Optional[float] = None
    qty: Optional[float] = None
    highest_close_since_entry: Optional[float] = None


def load_position(log_dir: Path) -> PositionState:
    path = log_dir / "position_state.json"
    if path.exists():
        with open(path) as f:
            return PositionState(**json.load(f))
    return PositionState()


def save_position(log_dir: Path, state: PositionState) -> None:
    path = log_dir / "position_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(state), f, indent=2)
