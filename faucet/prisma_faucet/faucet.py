"""Test-token faucet (B7-01) — for onboarding only.

Plain statement, which every response also carries:
**test tokens have no monetary value.**

Enforced limits (all explicit, all recorded):

- address validation (bech32, chain prefix `prsm1`, and the account type);
- per-account cooldown (a window in which the same address gets nothing);
- per-IP rate limit (token bucket);
- a per-response cap and an alert threshold on the faucet account balance;
- anti-loop: a bounded memory of recent funding pairs (address, IP) — a
  single IP funding many addresses in the window is refused.

The faucet signs and submits a real bank transfer through `prismad` from its
own funded account; this module owns policy, not chain mechanics (the tx
helper is shared with the da/worker packages).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from prisma_worker.identity import bech32_decode

TEST_TOKEN_NOTICE = "test tokens have no monetary value"


class FaucetError(Exception):
    """Base class for faucet policy failures."""


class InvalidAddress(FaucetError):
    pass


class CooldownActive(FaucetError):
    pass


class RateLimited(FaucetError):
    pass


class FarmingSuspected(FaucetError):
    pass


@dataclass
class FaucetPolicy:
    amount_uprsm: int = 10_000_000
    per_address_cooldown_s: float = 3600.0
    per_ip_requests: float = 1.0            # token bucket refill per second
    per_ip_burst: int = 5
    window_seconds: float = 3600.0
    max_addresses_per_ip_in_window: int = 3
    low_balance_threshold_uprsm: int = 1_000_000_000
    validate_prefix: str = "prsm"


@dataclass
class _Bucket:
    tokens: float
    last: float


@dataclass
class Faucet:
    policy: FaucetPolicy = field(default_factory=FaucetPolicy)
    send: Optional[Callable[[str, int], str]] = None      # (address, amount) -> txhash
    balance: Optional[Callable[[], int]] = None
    log: Callable[[str], None] = print
    last_funded: dict = field(default_factory=dict)        # address -> (ts, amount)
    ip_history: dict = field(default_factory=dict)         # ip -> list[(ts, address)]
    buckets: dict = field(default_factory=dict)            # ip -> _Bucket
    metrics: dict = field(default_factory=lambda: {"requests": 0, "funded": 0,
                                                   "cooldown_refused": 0,
                                                   "rate_limited": 0,
                                                   "farming_refused": 0,
                                                   "invalid_address": 0})

    # --- validation -------------------------------------------------------

    def validate_address(self, address: str) -> str:
        try:
            hrp, data = bech32_decode(str(address))
        except ValueError as exc:
            raise InvalidAddress(f"not a bech32 address: {exc}") from exc
        if hrp != self.policy.validate_prefix:
            raise InvalidAddress(f"address prefix {hrp!r} is not {self.policy.validate_prefix!r}")
        if len(data) != 20:
            raise InvalidAddress("not a 20-byte account address")
        return address

    # --- bucket -----------------------------------------------------------

    def _allow_ip(self, ip: str, now: float) -> bool:
        bucket = self.buckets.get(ip)
        if bucket is None:
            # a fresh bucket starts full minus this request's token
            self.buckets[ip] = _Bucket(tokens=float(self.policy.per_ip_burst) - 1, last=now)
            return True
        bucket.tokens = min(self.policy.per_ip_burst,
                            bucket.tokens + (now - bucket.last) * self.policy.per_ip_requests)
        bucket.last = now
        if bucket.tokens < 1:
            return False
        bucket.tokens -= 1
        return True

    # --- the one entrypoint -------------------------------------------------

    def fund(self, address: str, *, ip: str = "unknown", now: Optional[float] = None) -> dict:
        now = now if now is not None else time.time()
        self.metrics["requests"] += 1
        try:
            self.validate_address(address)
        except InvalidAddress as exc:
            self.metrics["invalid_address"] += 1
            raise
        if self.balance is not None and self.balance() < self.policy.low_balance_threshold_uprsm:
            self.log(f"ALERT: faucet balance below {self.policy.low_balance_threshold_uprsm} uprsm")
        if not self._allow_ip(ip, now):
            self.metrics["rate_limited"] += 1
            raise RateLimited("per-IP rate limit exceeded; wait and retry")
        last = self.last_funded.get(address)
        if last and now - last[0] < self.policy.per_address_cooldown_s:
            self.metrics["cooldown_refused"] += 1
            raise CooldownActive(
                f"address {address} was funded {int(now - last[0])}s ago; the cooldown is "
                f"{int(self.policy.per_address_cooldown_s)}s")
        window = self.ip_history.setdefault(ip, [])
        window[:] = [(ts, addr) for ts, addr in window if now - ts < self.policy.window_seconds]
        if sum(1 for _, addr in window if addr != address) + 1 > self.policy.max_addresses_per_ip_in_window:
            self.metrics["farming_refused"] += 1
            raise FarmingSuspected(
                f"IP {ip} funded too many distinct addresses in "
                f"{int(self.policy.window_seconds)}s; refusing (anti-loop)")
        if self.send is None:
            raise FaucetError("no funding backend configured")
        txhash = self.send(address, self.policy.amount_uprsm)
        self.last_funded[address] = (now, self.policy.amount_uprsm)
        window.append((now, address))
        self.metrics["funded"] += 1
        self.log(f"funded {address} with {self.policy.amount_uprsm} uprsm (tx {txhash})")
        return {"funded": True, "address": address, "amount_uprsm": self.policy.amount_uprsm,
                "txhash": txhash, "cooldown_seconds": int(self.policy.per_address_cooldown_s),
                "notice": TEST_TOKEN_NOTICE}
