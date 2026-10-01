"""Supply and audit-budget stress test for *valueless testnet* PRSM.

Amounts are integer micro-PRSM. This does not model a market price or prove
that a real-money bond deters fraud.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


BPS = 10_000


@dataclass(frozen=True)
class Parameters:
    genesis_supply: int = 1_000_000_000_000_000
    locked_reward_reserve: int = 400_000_000_000_000
    initial_annual_emission_bps: int = 500
    tail_annual_emission_bps: int = 50
    halving_months: int = 24
    burn_bps: int = 2_000
    worker_bps: int = 7_000
    watcher_bps: int = 1_000

    def validate(self) -> None:
        if self.genesis_supply <= 0 or self.locked_reward_reserve < 0:
            raise ValueError("invalid genesis supply or reserve")
        if self.locked_reward_reserve > self.genesis_supply:
            raise ValueError("reserve exceeds genesis supply")
        if self.halving_months <= 0:
            raise ValueError("halving_months must be positive")
        rates = (
            self.initial_annual_emission_bps,
            self.tail_annual_emission_bps,
            self.burn_bps,
            self.worker_bps,
            self.watcher_bps,
        )
        if any(rate < 0 or rate > BPS for rate in rates):
            raise ValueError("rates must be between 0 and 10000 basis points")
        if self.burn_bps + self.worker_bps + self.watcher_bps != BPS:
            raise ValueError("external fee allocations must add to 100%")


@dataclass(frozen=True)
class Month:
    external_fees: int
    tasks: int
    audit_fraction_bps: int
    replay_cost: int
    reserve_release: int = 0


@dataclass(frozen=True)
class Result:
    month: int
    supply: int
    newly_minted: int
    burned: int
    reserve_left: int
    worker_paid: int
    watcher_paid: int
    required_replay_budget: int
    watcher_shortfall: int


def annual_emission(params: Parameters, month: int) -> int:
    initial = params.genesis_supply * params.initial_annual_emission_bps // BPS
    tail = params.genesis_supply * params.tail_annual_emission_bps // BPS
    return max(tail, initial // (2 ** (month // params.halving_months)))


def simulate(params: Parameters, schedule: list[Month]) -> list[Result]:
    params.validate()
    supply = params.genesis_supply
    reserve_left = params.locked_reward_reserve
    results = []
    for index, item in enumerate(schedule):
        if min(item.external_fees, item.tasks, item.replay_cost, item.reserve_release) < 0:
            raise ValueError("negative scenario input")
        if not 0 <= item.audit_fraction_bps <= BPS:
            raise ValueError("audit fraction must be between 0 and 10000")
        if item.reserve_release > reserve_left:
            raise ValueError("reward reserve exhausted")
        minted = annual_emission(params, index) // 12
        burned = item.external_fees * params.burn_bps // BPS
        worker_paid = item.external_fees * params.worker_bps // BPS
        watcher_paid = item.external_fees - burned - worker_paid
        audits = (item.tasks * item.audit_fraction_bps + BPS - 1) // BPS
        required = audits * item.replay_cost
        reserve_left -= item.reserve_release
        supply += minted - burned
        results.append(Result(
            month=index + 1,
            supply=supply,
            newly_minted=minted,
            burned=burned,
            reserve_left=reserve_left,
            worker_paid=worker_paid,
            watcher_paid=watcher_paid,
            required_replay_budget=required,
            watcher_shortfall=max(0, required - watcher_paid),
        ))
    return results


def fraud_break_even_probability(saved_compute_cost: int, worker_bond: int) -> float:
    """Minimum detection probability making a skipped-compute attack unprofitable.

    Assumes detection loses the bond and avoids paying the compute cost. It
    deliberately excludes collusion, token price and other external losses.
    """
    if saved_compute_cost < 0 or worker_bond < 0:
        raise ValueError("cost and bond cannot be negative")
    total = saved_compute_cost + worker_bond
    return saved_compute_cost / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, help="JSON with params and monthly schedule")
    args = parser.parse_args()
    data = json.loads(args.scenario.read_text(encoding="utf-8"))
    params = Parameters(**data.get("params", {}))
    schedule = [Month(**item) for item in data["months"]]
    print(json.dumps([asdict(row) for row in simulate(params, schedule)], indent=2))


if __name__ == "__main__":
    main()
