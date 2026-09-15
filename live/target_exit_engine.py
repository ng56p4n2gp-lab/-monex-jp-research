"""Target exit engine for JP equities.

Gross P/L only: fees and taxes are intentionally excluded.
No broker credentials or live order API are included here.
"""
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo
from threading import Event
import time as _time

JST = ZoneInfo("Asia/Tokyo")

@dataclass
class Config:
    target_profit_yen: int = 500  # change any time
    poll_seconds: float = 1.0

@dataclass
class Position:
    symbol: str
    shares: int
    entry_price: float


def gross_profit(position: Position, last_price: float) -> float:
    return (last_price - position.entry_price) * position.shares


def target_price(position: Position, cfg: Config) -> float:
    return position.entry_price + cfg.target_profit_yen / position.shares


def is_tse_session(now: datetime | None = None) -> bool:
    now = now.astimezone(JST) if now else datetime.now(JST)
    if now.weekday() >= 5:
        return False
    t = now.time().replace(tzinfo=None)
    return time(9, 0) <= t <= time(11, 30) or time(12, 30) <= t <= time(15, 30)


class BrokerAdapter:
    """Implement these methods only after an officially supported order API is confirmed."""
    def scan_and_buy(self) -> Position | None:
        raise NotImplementedError

    def last_price(self, symbol: str) -> float:
        raise NotImplementedError

    def sell_all(self, position: Position) -> None:
        raise NotImplementedError


def run_forever(broker: BrokerAdapter, cfg: Config, stop: Event | None = None):
    """Repeat during TSE sessions. Never opens another position while one is held."""
    stop = stop or Event()
    position: Position | None = None
    while not stop.is_set():
        if not is_tse_session():
            _time.sleep(max(cfg.poll_seconds, 5.0))
            continue

        if position is None:
            position = broker.scan_and_buy()
            _time.sleep(cfg.poll_seconds)
            continue

        px = broker.last_price(position.symbol)
        if gross_profit(position, px) >= cfg.target_profit_yen:
            broker.sell_all(position)
            position = None  # immediately resume searching for next entry

        _time.sleep(cfg.poll_seconds)
