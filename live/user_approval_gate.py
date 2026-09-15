"""Fail-closed explicit user approval gate for any future live broker adapter.
Research code does not need approvals because it cannot place orders.
No broker integration is enabled here.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import secrets
from typing import Dict

LIVE_TRADING_ENABLED = False
REQUIRE_EXPLICIT_USER_APPROVAL = True

@dataclass(frozen=True)
class ApprovalRequest:
    side: str
    symbol: str
    shares: int
    reference_price: float
    reason: str
    created_at: datetime
    expires_at: datetime

@dataclass
class Approval:
    request: ApprovalRequest
    token: str
    used: bool = False

class ApprovalGate:
    """Approval is exact, short-lived, one-time and side/symbol/size bound."""
    def __init__(self, ttl_seconds: int = 120):
        self.ttl_seconds = ttl_seconds
        self._pending: Dict[str, Approval] = {}

    def new_request(self, side: str, symbol: str, shares: int, reference_price: float, reason: str) -> ApprovalRequest:
        side = side.upper()
        if side not in {"BUY", "SELL"}: raise ValueError("side must be BUY or SELL")
        if shares <= 0 or reference_price <= 0: raise ValueError("invalid order proposal")
        now = datetime.now(timezone.utc)
        return ApprovalRequest(side, symbol, shares, reference_price, reason, now, now + timedelta(seconds=self.ttl_seconds))

    def approve(self, request: ApprovalRequest) -> str:
        """Call only after the user explicitly approves this exact proposal."""
        now = datetime.now(timezone.utc)
        if now > request.expires_at: raise PermissionError("approval request expired")
        token = secrets.token_urlsafe(32)
        self._pending[token] = Approval(request=request, token=token)
        return token

    def consume(self, token: str, side: str, symbol: str, shares: int) -> ApprovalRequest:
        """Broker adapter must call this immediately before every real BUY/SELL."""
        if not LIVE_TRADING_ENABLED: raise PermissionError("live trading disabled")
        if not REQUIRE_EXPLICIT_USER_APPROVAL: raise PermissionError("approval requirement must remain enabled")
        a = self._pending.get(token)
        if a is None or a.used: raise PermissionError("missing or already-used approval")
        now = datetime.now(timezone.utc)
        r = a.request
        if now > r.expires_at: raise PermissionError("approval expired")
        if (r.side, r.symbol, r.shares) != (side.upper(), symbol, shares):
            raise PermissionError("approval does not match exact order")
        a.used = True
        return r
