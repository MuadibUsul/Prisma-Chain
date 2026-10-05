"""B7-01: faucet policy — validation, cooldown, per-IP limits, anti-loop,
honest balance alerting. Funding uses an injected sender (no chain)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for extra in ("faucet", "worker"):
    sys.path.insert(0, str(REPO / extra))

from prisma_faucet.faucet import (  # noqa: E402
    TEST_TOKEN_NOTICE, CooldownActive, Faucet, FaucetPolicy, FarmingSuspected,
    InvalidAddress, RateLimited,
)

ADDRESS = "prsm1l3e9pgs3mmwuwrh95fecme0s0qtn28806jn3dq"   # the prismad-derived vector
OTHER = "prsm18n8zw0sah6gdjmmfedslj4qpfjhk6rfn4lmmad"


def faucet(**policy) -> tuple[Faucet, list[str]]:
    sent: list[str] = []
    f = Faucet(policy=FaucetPolicy(**policy), send=lambda address, amount: f"TX{len(sent)}",
               log=lambda _m: None)
    f.send_recorded = sent
    return f, sent


def test_validates_the_chain_prefix_and_length():
    f, _ = faucet()
    with pytest.raises(InvalidAddress, match="not a bech32 address"):
        f.validate_address("cosmos1l3e9pgs3mmwuwrh95fecme0s0qtn28806jn3dq")
    with pytest.raises(InvalidAddress, match="bech32"):
        f.validate_address("not-an-address")
    with pytest.raises(InvalidAddress, match="not a bech32 address"):
        f.validate_address("prsm1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
    assert f.validate_address(ADDRESS) == ADDRESS


def test_cooldown_per_address():
    f, _ = faucet(per_address_cooldown_s=3600)
    f.fund(ADDRESS, ip="1.2.3.4", now=1000)
    with pytest.raises(CooldownActive, match="was funded 1s ago"):
        f.fund(ADDRESS, ip="1.2.3.4", now=1001)
    f.fund(ADDRESS, ip="1.2.3.4", now=1000 + 3600)     # after the window: fine
    assert f.metrics["funded"] == 2 and f.metrics["cooldown_refused"] == 1


def test_per_ip_rate_limit():
    f, _ = faucet(per_ip_requests=1.0, per_ip_burst=2)
    f.validate_address = lambda a: a          # isolate the bucket from address validity
    f.fund("a1", ip="1.1.1.1", now=100)
    f.fund("a2", ip="1.1.1.1", now=100)
    # burst exhausted (2 tokens for 2 requests); refill is 1/s so an immediate
    # third request is limited
    with pytest.raises(RateLimited):
        f.fund("a3", ip="1.1.1.1", now=100)
    assert f.metrics["rate_limited"] == 1
    # ...but after 2 seconds one token has refilled
    f.fund("a4", ip="1.1.1.1", now=102)
    assert f.metrics["funded"] == 3


def test_anti_loop_detects_farming():
    f, _ = faucet(max_addresses_per_ip_in_window=3, per_ip_requests=100, per_ip_burst=10)
    addresses = [f"prsm1l3e9pgs3mmwuwrh95fecme0s0qtn28806jn3d{i}" for i in range(5)]
    # NOTE: those addresses are not valid bech32; fund() validates first, so the
    # farming check must be reachable with distinct VALID addresses. Use the two
    # real vectors plus monkeypatched validation for the drill.
    seen: list[str] = []

    def fake_validate(address):
        seen.append(address)
        return address

    f.validate_address = fake_validate
    for address in (ADDRESS, OTHER, "a3"):
        f.fund(address, ip="9.9.9.9", now=1000)
    with pytest.raises(FarmingSuspected, match="too many distinct addresses"):
        f.fund("a4", ip="9.9.9.9", now=1000)
    assert f.metrics["farming_refused"] == 1


def test_old_window_entries_expire():
    f, _ = faucet(max_addresses_per_ip_in_window=2, per_ip_requests=100, per_ip_burst=10,
                  window_seconds=100)
    f.validate_address = lambda a: a
    f.fund("a1", ip="2.2.2.2", now=1000)
    f.fund("a2", ip="2.2.2.2", now=1100)
    f.fund("a3", ip="2.2.2.2", now=2000)   # a1/a2 fell out of the 100s window? a2 at 1100 is 900s old -> yes
    assert f.metrics["farming_refused"] == 0


def test_every_response_carries_the_no_value_notice():
    f, sent = faucet()
    result = f.fund(ADDRESS, ip="3.3.3.3", now=1)
    assert result["notice"] == TEST_TOKEN_NOTICE
    assert "no monetary value" in TEST_TOKEN_NOTICE


def test_low_balance_alerts():
    alerts: list[str] = []
    f = Faucet(policy=FaucetPolicy(low_balance_threshold_uprsm=10 ** 12),
               send=lambda address, amount: "TX", balance=lambda: 5, log=alerts.append)
    f.fund(ADDRESS, ip="4.4.4.4", now=1)
    assert any("ALERT" in line for line in alerts)


def test_metrics_track_outcomes():
    f, _ = faucet()
    f.validate_address = lambda a: a
    f.fund("x1", ip="5.5.5.5", now=1)
    try:
        f.fund("x1", ip="5.5.5.5", now=1)
    except CooldownActive:
        pass
    assert f.metrics == {"requests": 2, "funded": 1, "cooldown_refused": 1,
                         "rate_limited": 0, "farming_refused": 0, "invalid_address": 0}
