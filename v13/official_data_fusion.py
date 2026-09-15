"""Official-data fusion layer for V13. Research/paper only.
Never places an order. All external features must carry published_at and are
usable only when published_at <= decision_at (anti-lookahead).
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

LIVE_TRADING_ENABLED = False
REQUIRE_EXPLICIT_USER_APPROVAL = True

@dataclass(frozen=True)
class OfficialFeatures:
    decision_at: datetime
    price_score: float = 0.0
    supply_demand_score: float = 0.0
    material_score: float = 0.0
    anomaly_score: float = 0.0
    risk_score: float = 0.0
    margin_heat: float = 0.0
    short_pressure: float = 0.0
    regulation_risk: float = 0.0
    large_holder_change: float = 0.0
    event_risk: float = 0.0

@dataclass(frozen=True)
class FusionDecision:
    score: float
    grade: str
    allow_candidate: bool
    reason: str


def clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def known_before(decision_at: datetime, published_at: Optional[datetime]) -> bool:
    return published_at is not None and published_at <= decision_at


def fusion(f: OfficialFeatures) -> FusionDecision:
    # Positive engines: price + supply/demand + disclosed material + anomaly confirmation.
    positive = (0.34*clamp(f.price_score) + 0.26*clamp(f.supply_demand_score)
                + 0.24*clamp(f.material_score) + 0.16*clamp(f.anomaly_score))
    # Risk engines can veto a superficially attractive price forecast.
    danger = max(clamp(f.risk_score), clamp(f.margin_heat)*0.8,
                 clamp(f.regulation_risk), clamp(f.event_risk))
    penalty = (0.30*clamp(f.risk_score) + 0.18*clamp(f.margin_heat)
               + 0.12*clamp(f.short_pressure) + 0.18*clamp(f.regulation_risk)
               + 0.12*clamp(f.event_risk))
    score = positive - penalty
    if danger >= 0.75:
        return FusionDecision(score, 'C', False, 'risk_veto')
    if score >= 0.65: grade='S'
    elif score >= 0.45: grade='A'
    elif score >= 0.25: grade='B'
    else: grade='C'
    return FusionDecision(score, grade, grade in ('S','A'), 'official_data_fusion')


def order_permission(user_approved: bool) -> bool:
    """Hard gate: research code can never trade; future live adapter also needs approval."""
    return bool(LIVE_TRADING_ENABLED and REQUIRE_EXPLICIT_USER_APPROVAL and user_approved)
