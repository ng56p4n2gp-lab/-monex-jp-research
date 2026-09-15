"""Compare daily exit policies on chronological paper trades.
Research only. No broker calls. Gross returns; fees/tax excluded.
Input rows must contain only signals/features known at decision time plus future labels
used strictly for evaluation.
"""
from dataclasses import dataclass
from typing import Iterable, Dict, List
import math

DAILY_TARGET = 0.06

@dataclass(frozen=True)
class TradeEval:
    day: str
    symbol: str
    predicted_mfe: float
    predicted_mae: float
    upside_probability: float
    crash_probability: float
    future_mfe: float
    future_mae: float
    future_end: float


def fixed_tp(t: TradeEval, pct: float) -> float:
    if t.future_mfe >= pct: return pct
    return t.future_end


def staged_tp(realized: float, t: TradeEval, schedule=(.03,.02,.01,.005)) -> float:
    if realized < .02: target=schedule[0]
    elif realized < .04: target=schedule[1]
    elif realized < .055: target=schedule[2]
    else: target=schedule[3]
    target=min(target,max(0.0,DAILY_TARGET-realized))
    return fixed_tp(t,target) if target>0 else 0.0


def predictive_tp(realized: float, t: TradeEval) -> float:
    rem=max(0.0,DAILY_TARGET-realized)
    if rem<=0: return 0.0
    if realized<.02: stage=.03
    elif realized<.04: stage=.02
    elif realized<.055: stage=.01
    else: stage=rem
    target=min(rem,stage,max(0.0025,t.predicted_mfe*.70))
    return fixed_tp(t,target)


def rank_score(t: TradeEval) -> float:
    return t.predicted_mfe*t.upside_probability - 1.5*t.predicted_mae - .08*t.crash_probability


def simulate_day(rows: Iterable[TradeEval], policy: str, giveback=.005, max_trades=20) -> Dict:
    realized=0.0; high=0.0; n=0
    ordered=sorted(rows,key=rank_score,reverse=True)
    for t in ordered:
        if realized>=DAILY_TARGET or n>=max_trades: break
        if t.upside_probability<.70 or t.crash_probability>.10: continue
        if t.predicted_mae>max(.0025,.01*(1-min(1,realized/DAILY_TARGET))): continue
        if policy.startswith('fixed_'):
            ret=fixed_tp(t,float(policy.split('_')[1])/100)
        elif policy=='staged_3_2_1_05': ret=staged_tp(realized,t)
        elif policy=='predictive': ret=predictive_tp(realized,t)
        else: raise ValueError(policy)
        realized += ret; n += 1; high=max(high,realized)
        if high>0 and high-realized>=giveback: break
    return {'return':realized,'trades':n,'target_hit':realized>=DAILY_TARGET,'losing':realized<0,'giveback':high-realized}


def summarize(day_results: List[Dict]) -> Dict:
    if not day_results: return {}
    rs=[x['return'] for x in day_results]
    curve=[]; eq=1.0; peak=1.0; maxdd=0.0
    for r in rs:
        eq*=1+r; peak=max(peak,eq); maxdd=max(maxdd,(peak-eq)/peak)
        curve.append(eq)
    return {
      'days':len(rs), 'target6_rate':sum(x['target_hit'] for x in day_results)/len(rs),
      'losing_day_rate':sum(x['losing'] for x in day_results)/len(rs),
      'mean_day_return':sum(rs)/len(rs), 'worst_day':min(rs), 'max_drawdown':maxdd,
      'mean_trades':sum(x['trades'] for x in day_results)/len(rs),
      'mean_giveback':sum(x['giveback'] for x in day_results)/len(rs)
    }

POLICIES=('fixed_0.5','fixed_1','fixed_2','fixed_3','staged_3_2_1_05','predictive')
