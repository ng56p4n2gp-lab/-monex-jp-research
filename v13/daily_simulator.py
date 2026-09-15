"""V13 daily target simulator — research/paper only.
No broker calls. Gross returns only; fees/tax excluded by research requirement.
The 6% daily value is an optimization target, never a guarantee.
"""
from dataclasses import dataclass
from typing import Iterable, Optional

LIVE_TRADING_ENABLED = False

@dataclass(frozen=True)
class Forecast:
    symbol: str
    price: float
    predicted_mfe_pct: float
    predicted_mae_pct: float
    upside_probability: float
    crash_probability: float
    fusion_score: float = 0.0

@dataclass
class DayState:
    start_equity: float
    realized_return: float = 0.0
    high_water_return: float = 0.0
    trades: int = 0

@dataclass(frozen=True)
class DailyConfig:
    daily_target_pct: float = 0.06
    max_lot_yen: float = 200_000
    shares: int = 100
    min_upside_probability: float = 0.70
    max_crash_probability: float = 0.10
    max_predicted_mae_pct: float = 0.015
    high_water_giveback_pct: float = 0.005
    max_trades: int = 20


def remaining_target(state: DayState, cfg: DailyConfig) -> float:
    return max(0.0, cfg.daily_target_pct - state.realized_return)


def adaptive_tp_pct(state: DayState, f: Forecast, cfg: DailyConfig) -> float:
    """Large opportunity first; progressively smaller TP as daily target nears."""
    rem = remaining_target(state, cfg)
    if rem <= 0:
        return 0.0
    r = state.realized_return
    if r < 0.02:
        cap = 0.03
    elif r < 0.04:
        cap = 0.02
    elif r < 0.055:
        cap = 0.01
    else:
        cap = rem
    # Do not demand the full predicted MFE; apply a conservative haircut.
    forecast_cap = max(0.0, f.predicted_mfe_pct * 0.70)
    return max(0.0, min(rem, cap, forecast_cap))


def risk_budget_pct(state: DayState) -> float:
    """Shrink acceptable predicted adverse excursion as profit is locked in."""
    r = max(0.0, state.realized_return)
    if r >= 0.055: return 0.0025
    if r >= 0.04: return 0.0040
    if r >= 0.02: return 0.0060
    return 0.0100


def eligible(state: DayState, f: Forecast, cfg: DailyConfig) -> bool:
    if state.realized_return >= cfg.daily_target_pct or state.trades >= cfg.max_trades:
        return False
    if f.price <= 0 or f.price * cfg.shares > cfg.max_lot_yen:
        return False
    if f.upside_probability < cfg.min_upside_probability:
        return False
    if f.crash_probability > cfg.max_crash_probability:
        return False
    if f.predicted_mae_pct > min(cfg.max_predicted_mae_pct, risk_budget_pct(state)):
        return False
    return adaptive_tp_pct(state, f, cfg) > 0


def opportunity_score(f: Forecast) -> float:
    """Rank predicted upside first, but penalize MAE/crash and reward official fusion."""
    reward = f.predicted_mfe_pct * f.upside_probability
    risk = 1.5 * f.predicted_mae_pct + 0.08 * f.crash_probability
    return reward - risk + 0.02 * f.fusion_score


def choose_candidate(state: DayState, forecasts: Iterable[Forecast], cfg: DailyConfig) -> Optional[Forecast]:
    xs = [f for f in forecasts if eligible(state, f, cfg)]
    return max(xs, key=opportunity_score) if xs else None


def should_stop_day(state: DayState, cfg: DailyConfig) -> bool:
    if state.realized_return >= cfg.daily_target_pct:
        return True
    # Giveback guard: once positive high-water is established, stop if too much is surrendered.
    if state.high_water_return > 0 and state.high_water_return - state.realized_return >= cfg.high_water_giveback_pct:
        return True
    return state.trades >= cfg.max_trades


def apply_realized_trade(state: DayState, trade_return_pct: float) -> None:
    state.realized_return += float(trade_return_pct)
    state.high_water_return = max(state.high_water_return, state.realized_return)
    state.trades += 1
