"""B6-03/04/05: registry, matching, failure/reassign semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "api"))

from prisma_api.scheduler import Scheduler, SchedulerError  # noqa: E402

PROFILE = "qwen3-0.6b-layer0-v2"


def job(job_id="j1", profile=PROFILE):
    return {"job_id": job_id, "profile": profile}


def test_registry_requires_an_id_and_capabilities():
    scheduler = Scheduler()
    with pytest.raises(SchedulerError):
        scheduler.register("", (PROFILE,))
    with pytest.raises(SchedulerError):
        scheduler.register("w1", ())
    assert scheduler.register("w1", (PROFILE,))["registered"] is True
    with pytest.raises(SchedulerError, match="unknown worker"):
        scheduler.set_health("ghost", False)


def test_matching_is_capability_filtered_and_fifo(tmp_path=None):
    scheduler = Scheduler()
    scheduler.register("a", (PROFILE,))
    scheduler.register("b", ("other-profile",))
    assert scheduler.match(job("j1")) == "a", "the incapable worker is never chosen"
    scheduler.register("c", (PROFILE,))
    # FIFO fairness: after a was just assigned, the fresh worker is next
    assert scheduler.match(job("j2")) == "c"
    assert scheduler.match(job("j3")) == "a"
    assert scheduler.worker_for("j1") == "a"


def test_unhealthy_workers_are_skipped():
    scheduler = Scheduler()
    scheduler.register("a", (PROFILE,))
    scheduler.register("b", (PROFILE,))
    scheduler.set_health("a", False, reason="probe failed")
    assert scheduler.match(job("j1")) == "b"
    with pytest.raises(SchedulerError, match="unknown worker"):
        scheduler.set_health("ghost", True)


def test_failure_stages_map_to_reassignment_rules():
    for stage in ("pending", "assigned", "accepted", "running"):
        assert Scheduler._reassignable(stage) is True
    for stage in ("committed", "da_pending", "da_quorum", "finalized"):
        assert Scheduler._reassignable(stage) is False


def test_reassignment_never_touches_committed_jobs():
    scheduler = Scheduler()
    scheduler.register("a", (PROFILE,))
    with pytest.raises(SchedulerError, match="never reassigned"):
        scheduler.reassign(job("j1"), from_stage="committed")
    with pytest.raises(SchedulerError, match="never reassigned"):
        scheduler.reassign(job("j1"), from_stage="finalized")


def test_reassignment_on_failure_marks_the_worker_and_finds_another():
    scheduler = Scheduler()
    scheduler.register("a", (PROFILE,))
    scheduler.register("b", (PROFILE,))
    scheduler.match(job("j1"))
    assert scheduler.worker_for("j1") == "a"
    outcome = scheduler.report_failure("j1", "a", "running", reason="device error")
    assert outcome["reassignable"] is True
    replacement = scheduler.reassign(job("j1"), from_stage="running")
    assert replacement == "b"
    assert scheduler.workers["a"].healthy is False, "a failed worker leaves the rotation"
    assert scheduler.workers["a"].last_failure["stage"] == "running"
    # re-registration restores health (B6-03)
    scheduler.register("a", (PROFILE,))
    assert scheduler.workers["a"].healthy is True
    assert scheduler.match(job("j2")) == "a"


def test_no_compatible_worker_returns_none():
    scheduler = Scheduler()
    assert scheduler.match(job("j1")) is None
    scheduler.register("a", ("other",))
    assert scheduler.match(job("j1")) is None


def test_history_is_auditable():
    scheduler = Scheduler()
    scheduler.register("a", (PROFILE,))
    scheduler.match(job("j1"))
    scheduler.report_failure("j1", "a", "assigned", reason="worker vanished")
    events = [entry["event"] for entry in scheduler.history]
    assert events == ["registered", "assigned", "failure"]
