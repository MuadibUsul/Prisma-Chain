"""B2-09: every crash point classifies deterministically, the chain wins, and
recovery never submits a transaction (so no double-accept / double-commit)."""

from __future__ import annotations

import pathlib

import pytest

from prisma_worker import recovery
from prisma_worker.jobs import (
    PHASE_ABANDONED, PHASE_ACCEPTED, PHASE_ACCEPTING, PHASE_ACCEPT_FAILED,
    PHASE_DA_PENDING, PHASE_DA_QUORUM, PHASE_DISCOVERED, PHASE_EXECUTED,
    PHASE_EXECUTING, PHASE_SUBMITTED, PHASE_TERMINAL, Journal,
)

US = "prsm1worker"
OTHER = "prsm1someoneelse"


class FakeChain:
    """Records every call; recovery must only ever read."""

    def __init__(self, task: dict | None):
        self.task_document = task or {}
        self.tx_calls = 0

    def task(self, task_id):
        return dict(self.task_document)

    def accept_task(self, *a, **k):  # pragma: no cover - must never be called
        self.tx_calls += 1
        raise AssertionError("recovery must not submit transactions")


def journal_with(tmp_path: pathlib.Path, phase: str, **fields) -> Journal:
    journal = Journal(tmp_path / "journal")
    journal.record(7, PHASE_DISCOVERED)
    if phase != PHASE_DISCOVERED:
        journal.record(7, phase, **fields)
    return journal


CASES = [
    # (crash point, journal phase, chain task, expected action)
    ("before accept", PHASE_DISCOVERED, {"status": "posted"}, recovery.ACTION_RETRY),
    ("mid accept, chain already ours", PHASE_ACCEPTING, {"status": "accepted", "worker": US},
     recovery.ACTION_RECONCILE),
    ("mid accept, chain has another worker", PHASE_ACCEPTING,
     {"status": "accepted", "worker": OTHER}, recovery.ACTION_ABANDON),
    ("mid accept, chain shows nothing", PHASE_ACCEPTING, {"status": "posted"},
     recovery.ACTION_RECONCILE),
    ("accept failed, still open", PHASE_ACCEPT_FAILED, {"status": "posted"}, recovery.ACTION_RETRY),
    ("accept failed, taken", PHASE_ACCEPT_FAILED, {"status": "accepted", "worker": OTHER},
     recovery.ACTION_ABANDON),
    ("after accept", PHASE_ACCEPTED, {"status": "accepted", "worker": US}, recovery.ACTION_RESUME),
    ("mid execution", PHASE_EXECUTING, {"status": "accepted", "worker": US}, recovery.ACTION_RESUME),
    ("after execution, before commit", PHASE_EXECUTED, {"status": "accepted", "worker": US},
     recovery.ACTION_RESUME),
    ("after commit, before DA", PHASE_SUBMITTED, {"status": "result_submitted", "worker": US},
     recovery.ACTION_RECONCILE),
    ("after commit, chain disagrees", PHASE_SUBMITTED, {"status": "accepted", "worker": OTHER},
     recovery.ACTION_ABANDON),
    ("before DA quorum", PHASE_DA_PENDING, {"status": "result_submitted", "worker": US},
     recovery.ACTION_RESUME),
    ("after DA quorum", PHASE_DA_QUORUM, {"status": "result_submitted", "worker": US},
     recovery.ACTION_TERMINAL),
]


@pytest.mark.parametrize("name,phase,chain,expected", CASES)
def test_crash_points_classify(name, tmp_path, phase, chain, expected):
    journal = journal_with(tmp_path, phase)
    decision = recovery.recover_task(FakeChain(chain), journal, 7, account_address=US,
                                     log=lambda _m: None)
    assert decision.action == expected, f"{name}: {decision.action} != {expected}"


def test_terminal_and_abandoned_are_sticky(tmp_path):
    for phase in (PHASE_TERMINAL, PHASE_ABANDONED):
        journal = journal_with(tmp_path / phase, phase)
        decision = recovery.recover_task(FakeChain({"status": "posted"}), journal, 7,
                                         account_address=US, log=lambda _m: None)
        assert decision.action == recovery.ACTION_TERMINAL


def test_finalized_task_terminates_any_phase(tmp_path):
    for phase in (PHASE_ACCEPTED, PHASE_EXECUTED, PHASE_SUBMITTED, PHASE_DA_PENDING):
        journal = journal_with(tmp_path / phase, phase)
        decision = recovery.recover_task(FakeChain({"status": "finalized", "worker": US}), journal, 7,
                                         account_address=US, log=lambda _m: None)
        assert decision.action == recovery.ACTION_TERMINAL


def test_mid_accept_reconciles_without_a_transaction(tmp_path):
    journal = journal_with(tmp_path, PHASE_ACCEPTING)
    client = FakeChain({"status": "accepted", "worker": US})
    decision = recovery.recover_task(client, journal, 7, account_address=US, log=lambda _m: None)
    assert decision.action == recovery.ACTION_RECONCILE
    assert client.tx_calls == 0
    entry = journal.read(7)
    assert entry["phase"] == PHASE_ACCEPTED and entry.get("reconciled") is True


def test_other_workers_assignment_is_abandoned_and_never_re_entered(tmp_path):
    journal = journal_with(tmp_path / "j", PHASE_EXECUTING)
    recovery.recover_task(FakeChain({"status": "accepted", "worker": OTHER}), journal, 7,
                          account_address=US, log=lambda _m: None)
    assert journal.read(7)["phase"] == PHASE_ABANDONED
    again = recovery.recover_task(FakeChain({"status": "accepted", "worker": US}), journal, 7,
                                  account_address=US, log=lambda _m: None)
    assert again.action == recovery.ACTION_TERMINAL, "abandoned tasks stay abandoned"


def test_recovery_is_idempotent(tmp_path):
    journal = journal_with(tmp_path, PHASE_DA_PENDING)
    client = FakeChain({"status": "result_submitted", "worker": US})
    first = recovery.recover_all(client, journal, account_address=US, log=lambda _m: None)
    second = recovery.recover_all(client, journal, account_address=US, log=lambda _m: None)
    assert [d.action for d in first] == [d.action for d in second]
    assert client.tx_calls == 0


def test_da_resume_only_uploads_missing_providers(tmp_path):
    journal = journal_with(tmp_path, PHASE_DA_PENDING,
                           da_verified=[{"provider": "a"}, {"provider": "b"}])
    entry = journal.read(7)
    assert recovery.providers_to_upload(entry, ["a", "b", "c"]) == ["c"]
    assert recovery.providers_to_upload(entry, ["a", "b"]) == []


def test_summary_counts_actions(tmp_path):
    journal = Journal(tmp_path / "journal")
    journal.record(1, PHASE_ACCEPTED, txhash="x")
    journal.record(2, PHASE_DA_QUORUM)
    decisions = recovery.recover_all(FakeChain({"status": "accepted", "worker": US}), journal,
                                     account_address=US, log=lambda _m: None)
    report = recovery.summary(decisions)
    assert report["tasks"] == 2
    assert report["by_action"] == {recovery.ACTION_RESUME: 1, recovery.ACTION_TERMINAL: 1}
