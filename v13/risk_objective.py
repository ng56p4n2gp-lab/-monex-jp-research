"""Loss-first objective for V13 strategy selection.
A 6% day is a target metric, not a promise. Research/paper only.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Metrics:
    losing_day_rate: float
    max_drawdown: float
    worst_day: float
    mean_giveback: float
    profitable_day_rate: float
    target6_rate: float
    mean_day_return: float


def hard_reject(m: Metrics, max_losing_days=.20, max_dd=.10, worst_day_floor=-.03) -> bool:
    return (m.losing_day_rate > max_losing_days or
            m.max_drawdown > max_dd or
            m.worst_day < worst_day_floor)


def loss_first_score(m: Metrics) -> float:
    """Higher is better; loss/DD/giveback dominate +6% attainment."""
    if hard_reject(m): return float('-inf')
    return (
        -5.0*m.losing_day_rate
        -4.0*m.max_drawdown
        -2.0*max(0.0,-m.worst_day)
        -1.5*m.mean_giveback
        +1.0*m.profitable_day_rate
        +0.8*m.target6_rate
        +0.5*m.mean_day_return
    )
