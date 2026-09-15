"""V13 adaptive daily-target optimizer.

Research/paper logic only. No broker transport.
Goal: rank affordable JP equity candidates by expected intraday opportunity while
penalizing downside, then reduce requested profit as the day's +6% target nears.

IMPORTANT: predictions are probabilities, not guarantees. Features/models used to
produce CandidateForecast must only use information available at decision time.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class CandidateForecast:
    symbol: str
    price: float
    upside_probability: float   # calibrated P(positive target move)
    predicted_mfe_pct: float    # predicted max favorable excursion for remaining session
    predicted_mae_pct: float    # positive magnitude of predicted adverse excursion
    crash_probability: float
    institutional_sell_risk: float
    market_risk: float
    event_news_risk: float
    liquidity_risk: float

@dataclass
class Config:
    max_lot_cost_yen: int = 200_000
    lot_size: int = 100
    daily_target_pct: float = 0.06
    min_upside_probability: float = 0.85
    max_crash_probability: float = 0.10
    max_single_risk: float = 0.35
    risk_penalty: float = 1.5
    safety_haircut: float = 0.70


def remaining_target_pct(day_start_equity: float, realized_equity: float, cfg: Config) -> float:
    if day_start_equity <= 0:
        return 0.0
    target = day_start_equity * (1.0 + cfg.daily_target_pct)
    return max(0.0, target / realized_equity - 1.0) if realized_equity > 0 else cfg.daily_target_pct


def requested_take_profit_pct(day_start_equity: float, realized_equity: float, forecast: CandidateForecast, cfg: Config) -> float:
    """Start with larger opportunity; automatically shrink as +6% target approaches."""
    remaining = remaining_target_pct(day_start_equity, realized_equity, cfg)
    safe_forecast = max(0.0, forecast.predicted_mfe_pct * cfg.safety_haircut)
    return min(remaining, safe_forecast)


def risk_score(c: CandidateForecast) -> float:
    risks = [c.crash_probability,c.institutional_sell_risk,c.market_risk,c.event_news_risk,c.liquidity_risk]
    return 0.60 * max(risks) + 0.40 * (sum(risks) / len(risks))


def eligible(c: CandidateForecast, cfg: Config) -> bool:
    if c.price <= 0 or c.price * cfg.lot_size > cfg.max_lot_cost_yen:
        return False
    if c.upside_probability < cfg.min_upside_probability:
        return False
    if c.crash_probability > cfg.max_crash_probability:
        return False
    if max(c.institutional_sell_risk,c.market_risk,c.event_news_risk,c.liquidity_risk) > cfg.max_single_risk:
        return False
    return True


def opportunity_score(c: CandidateForecast, cfg: Config) -> float:
    """Reward large probable upside; penalize predicted adverse excursion and precursor risk."""
    expected_up = c.upside_probability * max(0.0, c.predicted_mfe_pct)
    downside = (1.0-c.upside_probability) * max(0.0,c.predicted_mae_pct)
    return expected_up - downside - cfg.risk_penalty * risk_score(c)


def rank_candidates(candidates: list[CandidateForecast], cfg: Config) -> list[CandidateForecast]:
    valid = [c for c in candidates if eligible(c,cfg)]
    return sorted(valid,key=lambda c: opportunity_score(c,cfg),reverse=True)


def choose_trade(candidates: list[CandidateForecast], day_start_equity: float, realized_equity: float, cfg: Config):
    """Return best candidate + adaptive TP. None means NO_TRADE or daily target complete."""
    if remaining_target_pct(day_start_equity,realized_equity,cfg) <= 0:
        return None
    ranked = rank_candidates(candidates,cfg)
    if not ranked:
        return None
    best = ranked[0]
    tp = requested_take_profit_pct(day_start_equity,realized_equity,best,cfg)
    if tp <= 0:
        return None
    return best,tp
