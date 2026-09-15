"""Risk-first controller for the JP-equity auto-trader.

Priority order:
1) Minimize loss days / catastrophic losses.
2) Trade only approved high-quality signals; NO_TRADE is valid.
3) Protect open profit with break-even/trailing logic.
4) Stop opening new trades when daily target is reached.

No strategy can guarantee zero losses. Defaults are conservative research/paper
settings and remain configurable. Live transport stays disabled elsewhere.
"""
from dataclasses import dataclass
from enum import Enum, auto

class Action(Enum):
    HOLD = auto()
    EXIT = auto()
    BLOCK_NEW_BUY = auto()
    ALLOW_NEW_BUY = auto()

@dataclass
class RiskConfig:
    daily_target_pct: float = 0.06
    max_daily_loss_pct: float = 0.01
    max_trade_loss_pct: float = 0.005
    break_even_arm_pct: float = 0.004
    break_even_floor_pct: float = 0.0005
    trailing_arm_pct: float = 0.008
    trailing_gap_pct: float = 0.004
    max_holding_minutes: int = 60
    min_signal_score: float = 0.85
    max_trades_per_day: int = 20

@dataclass
class DayState:
    start_equity: float
    realized_pnl: float = 0.0
    trades: int = 0

@dataclass
class PositionState:
    entry_price: float
    highest_price: float
    minutes_held: int


def daily_return(s: DayState) -> float:
    return s.realized_pnl / s.start_equity if s.start_equity > 0 else 0.0


def can_open_new_trade(s: DayState, signal_score: float, cfg: RiskConfig) -> bool:
    if daily_return(s) >= cfg.daily_target_pct:
        return False
    if daily_return(s) <= -cfg.max_daily_loss_pct:
        return False
    if s.trades >= cfg.max_trades_per_day:
        return False
    return signal_score >= cfg.min_signal_score


def exit_reason(p: PositionState, last_price: float, signal_still_valid: bool, cfg: RiskConfig) -> str | None:
    """Return a reason to exit; None means hold.

    Stops are price triggers, not guarantees of fill price. Gaps/illiquidity can
    cause a larger realized loss than the configured threshold.
    """
    if p.entry_price <= 0:
        return 'invalid_entry'
    ret = last_price / p.entry_price - 1.0
    peak = max(p.highest_price, last_price) / p.entry_price - 1.0

    if ret <= -cfg.max_trade_loss_pct:
        return 'hard_loss_cap'
    if peak >= cfg.break_even_arm_pct and ret <= cfg.break_even_floor_pct:
        return 'break_even_protect'
    if peak >= cfg.trailing_arm_pct and ret <= peak - cfg.trailing_gap_pct:
        return 'trailing_profit_protect'
    if not signal_still_valid:
        return 'signal_deteriorated'
    if p.minutes_held >= cfg.max_holding_minutes:
        return 'time_exit'
    return None
