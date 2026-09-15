"""Predictive guard: act BEFORE adverse moves when observable precursors deteriorate.

This is a probability/risk layer, not clairvoyance. It never claims to know
unpublished news or trader identity. All features must be timestamped and known
BEFORE the decision to avoid look-ahead bias.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class PredictiveInputs:
    upside_probability: float       # validated strategy model 0..1
    crash_probability: float        # next-horizon downside model 0..1
    orderflow_risk: float            # board/trade imbalance precursor 0..1
    market_risk: float               # index/futures/sector precursor 0..1
    event_risk: float                # scheduled disclosure/event risk 0..1
    news_risk: float                 # only already-public timestamped news 0..1
    liquidity_risk: float            # spread/depth/turnover risk 0..1
    anomaly_risk: float              # unusual volume/price/quote behavior 0..1

@dataclass
class PredictiveConfig:
    min_upside_probability: float = 0.85
    max_crash_probability: float = 0.10
    max_component_risk: float = 0.35
    max_total_risk: float = 0.25
    preemptive_exit_risk: float = 0.45


def total_risk(x: PredictiveInputs) -> float:
    # Conservative: emphasize the worst precursor while retaining broad context.
    risks=[x.crash_probability,x.orderflow_risk,x.market_risk,x.event_risk,x.news_risk,x.liquidity_risk,x.anomaly_risk]
    return 0.60*max(risks)+0.40*(sum(risks)/len(risks))


def allow_buy(x: PredictiveInputs, cfg: PredictiveConfig) -> tuple[bool,str]:
    """BUY only when upside is strong and no precursor vetoes the trade."""
    if x.upside_probability < cfg.min_upside_probability:
        return False,'upside_probability_too_low'
    if x.crash_probability > cfg.max_crash_probability:
        return False,'predicted_downside_risk'
    components=[x.orderflow_risk,x.market_risk,x.event_risk,x.news_risk,x.liquidity_risk,x.anomaly_risk]
    if max(components) > cfg.max_component_risk:
        return False,'risk_component_veto'
    if total_risk(x) > cfg.max_total_risk:
        return False,'aggregate_risk_veto'
    return True,'predictive_filters_passed'


def preemptive_exit(x: PredictiveInputs, cfg: PredictiveConfig) -> tuple[bool,str]:
    """May exit before an actual drop if precursor risk crosses validated threshold."""
    if x.crash_probability > cfg.max_crash_probability:
        return True,'crash_probability_spike'
    if total_risk(x) >= cfg.preemptive_exit_risk:
        return True,'precursor_risk_spike'
    return False,'hold'
