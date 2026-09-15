"""Timestamp-safe official-source feature records for V13.
Adapters may populate these from JPX/TDnet/EDINET. A record is unusable until its
actual public publication time, preventing look-ahead in backtests.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

class Source(str,Enum):
    JPX_MARGIN_DAILY='JPX_MARGIN_DAILY'
    JPX_MARGIN_RATIO='JPX_MARGIN_RATIO'
    JPX_MARGIN_REGULATION='JPX_MARGIN_REGULATION'
    JPX_SHORT_VALUE='JPX_SHORT_VALUE'
    JPX_SHORT_POSITION='JPX_SHORT_POSITION'
    JPX_STOCK_LOAN_FEE='JPX_STOCK_LOAN_FEE'
    TDNET='TDNET'
    EDINET='EDINET'

@dataclass(frozen=True)
class OfficialRecord:
    source: Source
    symbol: Optional[str]
    effective_date: str
    published_at: datetime
    value: float
    label: str=''

    def known_at(self, decision_at: datetime) -> bool:
        return self.published_at <= decision_at


def latest_known(records, decision_at, source=None, symbol=None):
    xs=[r for r in records if r.known_at(decision_at)]
    if source is not None: xs=[r for r in xs if r.source==source]
    if symbol is not None: xs=[r for r in xs if r.symbol in (None,symbol)]
    return max(xs,key=lambda r:r.published_at) if xs else None


def assert_no_lookahead(records, decision_at):
    bad=[r for r in records if r.published_at>decision_at]
    if bad: raise ValueError(f'future official records supplied: {len(bad)}')
    return True
