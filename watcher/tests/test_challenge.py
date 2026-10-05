"""B3-02: the challenge decision policy, bond management, window tracking,
dispute driving and outcome handling — all with an injected DisputeOps."""

from __future__ import annotations

import json
import pathlib

import pytest

import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
for extra in ("watcher", "worker", "network"):
    sys.path.insert(0, str(REPO / extra))

from prisma_watcher.challenge import (  # noqa: E402
    ChallengeError, DecisionPolicy, DisputeDriver, NoEvidence, UnfundedChallenge, WindowClosed,
)

PROVIDER = "prsm1challenger"


class FakeOps:
    def __init__(self, *, height=100, balance=1_000_000, states=None, outcomes=None):
        self.height_value = height
        self.balance_value = balance
        self.states = states or {}
        self.outcomes = outcomes or []
        self.opened: list[dict] = []

    def height(self):
        return self.height_value

    def balance(self, account):
        return self.balance_value

    def dispute_state(self, task_id):
        return self.states.get(task_id, {})

    def open_challenge(self, task_id, bond, evidence_digest):
        self.opened.append({"task_id": task_id, "bond": bond, "digest": evidence_digest})
        return f"CHAL_TX_{len(self.opened)}"


def fraud_verification():
    return {"verdict": "fraud",
            "report": {"root_phase": {"checks": "final_output_root mismatch"},
                       "math_phase": {"freivalds": {"round": 12, "residual": "nonzero"}}}}


def clean_verification():
    return {"verdict": "clean", "report": {"root_phase": {"checks": "all passed"}}}


def task(task_id=7, deadline=500):
    return {"id": task_id, "challenge_end": deadline}


def test_evidence_gated_decision():
    policy = DecisionPolicy()
    decision = policy.decide(task(), fraud_verification(), height=100)
    assert decision["challenge"] is True
    assert "root_phase" in decision["evidence"] and "freivalds" in decision["evidence"]
    clean = policy.decide(task(), clean_verification(), height=100)
    assert clean["challenge"] is False and "not 'fraud'" in clean["reason"]


def test_never_challenges_without_evidence():
    policy = DecisionPolicy()
    decision = policy.decide(task(), {"verdict": "fraud", "report": {}}, height=100)
    assert decision["challenge"] is False and "refusing" in decision["reason"]
    with pytest.raises(NoEvidence):
        if not decision["challenge"]:
            raise NoEvidence(decision["reason"])


def test_never_challenges_after_the_window():
    policy = DecisionPolicy(safety_margin_blocks=5)
    late = policy.decide(task(deadline=103), fraud_verification(), height=100)
    assert late["challenge"] is False and "safety margin" in late["reason"]
    in_time = policy.decide(task(deadline=104), fraud_verification(), height=100)
    assert in_time["challenge"] is False and "safety margin" in in_time["reason"]
    far = policy.decide(task(deadline=500), fraud_verification(), height=100)
    assert far["challenge"] is True


def test_unfunded_challenger_is_refused_with_the_amount():
    ops = FakeOps(balance=100)
    driver = DisputeDriver(ops, PROVIDER)
    with pytest.raises(UnfundedChallenge, match="has 100 uprsm but the bond needs 10000"):
        driver.consider(task(), fraud_verification())


def test_challenge_opens_with_a_transcript_digest(tmp_path):
    ops = FakeOps()
    driver = DisputeDriver(ops, PROVIDER)
    result = driver.challenge(task(), fraud_verification(), record_dir=tmp_path)
    assert result["challenge"] is True and ops.opened[0]["bond"] == 10_000
    assert len(result["transcript_digest"]) == 64
    record = json.loads((tmp_path / "challenge-7.json").read_text())
    assert record["task_id"] == 7 and record["txhash"] == result["txhash"]
    # a clean task is never challenged
    assert driver.challenge(task(), clean_verification(), record_dir=tmp_path) == {
        "challenge": False, "reason": "verdict 'clean' is not 'fraud'"}
    assert len(ops.opened) == 1


def test_dispute_driving_records_the_transcript(tmp_path):
    ops = FakeOps(states={7: {"phase": "bisection"}}, outcomes=[])
    driver = DisputeDriver(ops, PROVIDER)

    def resolve(task_id):
        ops.states[7] = {"phase": "resolved", "result": "ChallengerWins"}

    original = ops.dispute_state
    calls = {"n": 0}

    def alternating(task_id):
        calls["n"] += 1
        if calls["n"] > 2:
            resolve(task_id)
        return original(task_id)

    ops.dispute_state = alternating
    record = driver.drive(7, max_steps=8, record_dir=tmp_path)
    assert record["outcome"] == "ChallengerWins"
    assert len(record["transcript"]) == 3
    assert record["digest"]
    saved = json.loads((tmp_path / "dispute-7.json").read_text())
    assert saved["outcome"] == "ChallengerWins" and saved["digest"] == record["digest"]


@pytest.mark.parametrize("phase,result,expected", [
    ("challenger_wins", "", "ChallengerWins"),
    ("worker_wins", "", "WorkerWins"),
    ("both_invalid", "", "BothInvalid"),
    ("timeout", "", "timeout"),
    ("resolved", "WorkerWins", "WorkerWins"),
    ("resolved", "BothInvalid", "BothInvalid"),
])
def test_outcome_mapping(phase, result, expected):
    assert DisputeDriver._outcome(phase, {"result": result}) == expected


def test_unresolved_dispute_times_out_explicitly(tmp_path):
    ops = FakeOps(states={7: {"phase": "bisection"}})
    driver = DisputeDriver(ops, PROVIDER)
    with pytest.raises(ChallengeError, match="did not resolve within 2 polls"):
        driver.drive(7, max_steps=2, record_dir=tmp_path)
