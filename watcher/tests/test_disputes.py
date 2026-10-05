"""B3-03: restart recovery — the watcher never loses a running dispute and
never double-opens one."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
for extra in ("watcher", "worker", "network"):
    sys.path.insert(0, str(REPO / extra))

from prisma_watcher.disputes import ACTIVE_PHASES, DisputeLedger, DisputeLedgerError  # noqa: E402


class FakeOps:
    def __init__(self, states):
        self.states = states

    def dispute_state(self, task_id):
        return self.states.get(task_id, {})


def test_record_and_read_roundtrip(tmp_path):
    ledger = DisputeLedger(tmp_path / "ledger")
    ledger.record(7, "open", txhash="TX1", bond_uprsm=10_000, locked_roots=["aa" * 32])
    ledger.record(7, "midpoint", midpoints=[1, 2])
    entry = ledger.read(7)
    assert entry["phase"] == "midpoint" and entry["txhash"] == "TX1"
    assert entry["midpoints"] == [1, 2] and entry["bond_uprsm"] == 10_000
    assert [h["phase"] for h in entry["history"]] == ["open", "midpoint"]


def test_active_listing_and_duplicate_guard(tmp_path):
    ledger = DisputeLedger(tmp_path / "ledger")
    ledger.record(7, "open", txhash="TX1")
    assert [e["task_id"] for e in ledger.active()] == [7]
    with pytest.raises(DisputeLedgerError, match="refusing to open a second challenge"):
        ledger.ensure_not_duplicate(7)
    ledger.ensure_not_duplicate(8)          # no entry: fine
    ledger.record(7, "resolved", outcome="ChallengerWins")
    ledger.ensure_not_duplicate(7)          # resolved: fine to re-challenge later


def test_recover_marks_chain_resolved_disputes(tmp_path):
    ledger = DisputeLedger(tmp_path / "ledger")
    ledger.record(7, "wide", txhash="TX1")
    ops = FakeOps({7: {"phase": "resolved", "result": "ChallengerWins"}})
    report = ledger.recover(ops)
    assert report["active"] == 1 and report["resolved_on_chain"] == 1
    entry = ledger.read(7)
    assert entry["phase"] == "resolved" and entry["outcome"] == "ChallengerWins"


def test_recover_resumes_still_running_disputes(tmp_path):
    ledger = DisputeLedger(tmp_path / "ledger")
    ledger.record(7, "midpoint", txhash="TX1")
    ops = FakeOps({7: {"phase": "bisection", "round": 3}})
    driven: list[int] = []
    report = ledger.recover(ops, driver=lambda task_id: driven.append(task_id))
    assert report["still_running"] == 1 and driven == [7]
    entry = ledger.read(7)
    assert entry["phase"] == "midpoint" and entry["last_chain_phase"] == "bisection"
    assert entry["last_chain_state"]["round"] == 3


def test_recover_is_idempotent(tmp_path):
    ledger = DisputeLedger(tmp_path / "ledger")
    ledger.record(7, "open", txhash="TX1")
    ops = FakeOps({7: {"phase": "trail"}})
    first = ledger.recover(ops)
    second = ledger.recover(ops)
    assert first["still_running"] == second["still_running"] == 1
    assert ledger.read(7)["phase"] == "open"


def test_restart_equivalence_across_all_dispute_phases(tmp_path):
    """Kill the watcher at each phase; restart finds it and resumes exactly."""
    for phase in ("open", "trail", "midpoint", "wide", "arb_ready"):
        directory = tmp_path / f"ledger-{phase}"
        ledger = DisputeLedger(directory)
        ledger.record(7, phase, txhash=f"TX_{phase}", bond_uprsm=10_000,
                      locked_roots=["aa" * 32], midpoint_history=[1, 2, 3])
        restarted = DisputeLedger(directory)                     # a fresh process
        active = restarted.active()
        assert len(active) == 1 and active[0]["phase"] == phase
        assert active[0]["locked_roots"] == ["aa" * 32]
        assert active[0]["midpoint_history"] == [1, 2, 3]
        ops = FakeOps({7: {"phase": phase}})
        report = restarted.recover(ops)
        assert report["still_running"] == 1, f"phase {phase} must survive the restart"
