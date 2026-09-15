"""Signal monitor -> automatic buy engine for JP equities.

Design goals:
- Monitor approved strategy signals continuously during TSE sessions.
- Cash-equity budget <= 200,000 JPY per position (default).
- One position at a time; no averaging down / duplicate signal orders.
- Submit BUY only through an explicitly implemented BrokerAdapter.
- Default live_trading_enabled=False: safe/paper-first.
- Never store credentials in this repository.

Broker-specific order transport is intentionally NOT implemented until an officially
supported Monex order API is confirmed and configured outside this public repo.
"""
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo
from threading import Event
import time as _time

JST = ZoneInfo("Asia/Tokyo")

@dataclass
class Config:
    max_position_yen: int = 200_000
    lot_size: int = 100
    target_profit_yen: int = 500
    poll_seconds: float = 1.0
    live_trading_enabled: bool = False

@dataclass(frozen=True)
class Signal:
    symbol: str
    signal_id: str          # e.g. symbol + completed 5m bar timestamp
    reference_price: float
    strategy: str

@dataclass
class Position:
    symbol: str
    shares: int
    avg_fill_price: float
    buy_order_id: str

class MarketDataAdapter:
    def latest_signal(self) -> Signal | None:
        """Return only a newly completed/confirmed strategy signal."""
        raise NotImplementedError

class BrokerAdapter:
    def available_cash(self) -> float:
        raise NotImplementedError

    def buy_market(self, symbol: str, shares: int) -> Position:
        """Must return only after broker-confirmed fill (or raise on failure)."""
        raise NotImplementedError

    def has_open_position(self) -> bool:
        raise NotImplementedError


def is_tse_session(now: datetime | None = None) -> bool:
    now = now.astimezone(JST) if now else datetime.now(JST)
    if now.weekday() >= 5:
        return False
    t = now.time().replace(tzinfo=None)
    return time(9, 0) <= t < time(11, 30) or time(12, 30) <= t < time(15, 30)


def shares_for(signal: Signal, cfg: Config, cash: float) -> int:
    """Standard 100-share lots, never above budget or available cash."""
    cap = min(float(cfg.max_position_yen), float(cash))
    lot_cost = signal.reference_price * cfg.lot_size
    if signal.reference_price <= 0 or lot_cost > cap:
        return 0
    lots = int(cap // lot_cost)
    return lots * cfg.lot_size


def run_signal_monitor(market: MarketDataAdapter, broker: BrokerAdapter, cfg: Config, stop: Event | None = None):
    """Monitor signals and auto-buy when all guards pass.

    Production safety: no second buy while a position is open. A signal ID is
    processed at most once. Live orders require live_trading_enabled=True.
    """
    stop = stop or Event()
    processed: set[str] = set()

    while not stop.is_set():
        if not is_tse_session() or broker.has_open_position():
            _time.sleep(cfg.poll_seconds)
            continue

        sig = market.latest_signal()
        if sig is None or sig.signal_id in processed:
            _time.sleep(cfg.poll_seconds)
            continue

        # Mark before order submission to prevent duplicate buys on retries/errors.
        processed.add(sig.signal_id)
        cash = broker.available_cash()
        qty = shares_for(sig, cfg, cash)
        if qty <= 0:
            _time.sleep(cfg.poll_seconds)
            continue

        if not cfg.live_trading_enabled:
            print(f"PAPER BUY signal={sig.signal_id} symbol={sig.symbol} shares={qty} strategy={sig.strategy}", flush=True)
            _time.sleep(cfg.poll_seconds)
            continue

        position = broker.buy_market(sig.symbol, qty)
        print(f"BUY FILLED symbol={position.symbol} shares={position.shares} avg={position.avg_fill_price}", flush=True)
        _time.sleep(cfg.poll_seconds)
