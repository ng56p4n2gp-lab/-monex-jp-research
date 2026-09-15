"""Bearish previous-candle 50% midpoint entry strategy.
Signal only; broker-specific order transport remains isolated.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Candle:
    open: float
    high: float
    low: float
    close: float


def bearish_midpoint(prev: Candle) -> float | None:
    # Bullish and doji candles are explicitly rejected.
    if prev.close >= prev.open or prev.high <= prev.low:
        return None
    return (prev.high + prev.low) / 2.0


def midpoint_touch(prev: Candle, current: Candle) -> float | None:
    """Return midpoint when current candle trades through/touches it."""
    mid = bearish_midpoint(prev)
    if mid is None:
        return None
    if current.low <= mid <= current.high:
        return mid
    return None


def entry_signal(prev: Candle, current: Candle) -> bool:
    return midpoint_touch(prev, current) is not None
