"""B6-06: job/VWR presentation — verification status is passthrough only."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "api"))

from prisma_api.presentation import (  # noqa: E402
    CHAIN_STATUS_MAP, JOB_STATES, job_state_for, present_all, present_job,
)


class FakeChain:
    def __init__(self, task=None, receipt=None):
        self.task_document = task or {}
        self.receipt_document = receipt

    def task(self, task_id):
        return dict(self.task_document)

    def receipt(self, task_id):
        return self.receipt_document


def test_state_mapping_covers_the_machine_and_is_conservative():
    for state in JOB_STATES:
        assert state in set(CHAIN_STATUS_MAP.values()) | {"cancelled", "failed"}
    assert job_state_for("posted") == "queued"
    assert job_state_for("result_submitted") == "committed"
    assert job_state_for("challenged") == "verifying"
    assert job_state_for("challenger_wins") == "fraud"
    assert job_state_for("some-future-chain-state") == "unknown", "never guess"


def test_presentation_before_a_task_exists():
    view = present_job({"job_id": "j-1", "task_id": None, "status": "queued"}, FakeChain())
    assert view["status"] == "queued" and view["verification"] is None
    assert view["worker"] is None and view["vwr_id"] is None


def test_presentation_carries_chain_fields_only():
    chain = FakeChain(task={"id": 42, "status": "result_submitted", "worker": "prsm1w",
                            "graph_id_v2": "8fb86087…", "final_output_root": "ab" * 32})
    view = present_job({"job_id": "j-1", "task_id": 42, "status": "submitted"}, chain)
    assert view["status"] == "committed" and view["worker"] == "prsm1w"
    assert view["graph_id"] == "8fb86087…" and view["final_root"] == "ab" * 32
    assert view["verification"]["confirmed"] is False
    assert "not confirmed" in view["verification"]["note"]


def test_presentation_with_a_confirmed_receipt():
    chain = FakeChain(task={"id": 42, "status": "finalized", "worker": "prsm1w"},
                      receipt={"receipt_id": "vwr-1", "verdict": "pass",
                               "settlement_txhash": "SETTLE_TX"})
    view = present_job({"job_id": "j-1", "task_id": 42}, chain)
    assert view["verification"]["confirmed"] is True
    assert view["verification"]["verdict"] == "pass"
    assert view["vwr_id"] == "vwr-1" and view["settlement_tx"] == "SETTLE_TX"


def test_unknown_chain_status_is_surfaced_not_hidden():
    view = present_job({"job_id": "j-1", "task_id": 9},
                       FakeChain(task={"id": 9, "status": "brand-new-state"}))
    assert view["status"] == "unknown"


def test_present_all_maps_each_job():
    chain = FakeChain(task={"id": 1, "status": "finalized"})
    views = present_all([{"job_id": "a", "task_id": 1}, {"job_id": "b", "task_id": None}], chain)
    assert [v["job_id"] for v in views] == ["a", "b"]
    assert views[0]["status"] == "finalized" and views[1]["status"] == "queued"
