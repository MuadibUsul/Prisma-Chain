"""The scheduler (B6-03/04/05): worker registry, matching, failure/reassign.

Deliberately simple for the first release (the roadmap freezes this scope):
capability filter + FIFO ordering + health, with explicit reassignment rules
per lifecycle stage. No market auction, no real-time bidding.

Matching contract:

- a worker registers a capability (profile + optional GPU class);
- a job matches when its profile is in the worker's capability set and the
  worker is healthy;
- assignment is FIFO: the oldest compatible pending job goes to the least
  recently assigned healthy worker that does not already hold it.

Reassignment rules per stage (B6-05):

    pending          reassign freely (nothing accepted on chain)
    assigned         reassign after a worker failure (health change)
    accepted/running reassign only on failure evidence; a task bond exists
    committed+       never reassigned (the chain owns the dispute path)

A worker failure is recorded with the stage it failed at, so the reassignment
decision is auditable, and a worker that failed mid-execution is marked
unhealthy until it re-registers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


class SchedulerError(Exception):
    """Base class for scheduler failures."""


@dataclass
class WorkerRecord:
    worker_id: str
    capabilities: tuple[str, ...]           # profile names
    healthy: bool = True
    registered_at: float = field(default_factory=time.time)
    last_assigned_at: float = 0.0
    last_failure: Optional[dict] = None     # {"stage", "job_id", "at", "reason"}

    def supports(self, profile: str) -> bool:
        return profile in self.capabilities


@dataclass
class Scheduler:
    workers: dict = field(default_factory=dict)          # worker_id -> WorkerRecord
    assignments: dict = field(default_factory=dict)      # job_id -> worker_id
    history: list = field(default_factory=list)

    # --- registry (B6-03) ---------------------------------------------------

    def register(self, worker_id: str, capabilities: tuple[str, ...]) -> dict:
        if not worker_id:
            raise SchedulerError("worker_id is required")
        if not capabilities:
            raise SchedulerError("a worker must declare at least one capability")
        existing = self.workers.get(worker_id)
        record = WorkerRecord(worker_id=worker_id, capabilities=tuple(capabilities))
        if existing and not existing.healthy:
            record.last_failure = existing.last_failure  # re-registration clears health
        self.workers[worker_id] = record
        self.history.append({"at": time.time(), "event": "registered",
                             "worker_id": worker_id, "capabilities": list(capabilities)})
        return {"registered": True, "worker_id": worker_id,
                "capabilities": list(capabilities), "healthy": record.healthy}

    def unregister(self, worker_id: str) -> dict:
        self.workers.pop(worker_id, None)
        self.history.append({"at": time.time(), "event": "unregistered", "worker_id": worker_id})
        return {"unregistered": True, "worker_id": worker_id}

    def set_health(self, worker_id: str, healthy: bool, *, reason: str = "") -> None:
        record = self.workers.get(worker_id)
        if record is None:
            raise SchedulerError(f"unknown worker {worker_id}")
        record.healthy = healthy
        self.history.append({"at": time.time(), "event": "health", "worker_id": worker_id,
                             "healthy": healthy, "reason": reason})

    # --- matching (B6-04) -----------------------------------------------------

    def match(self, job: dict) -> Optional[str]:
        """The least-recently-assigned healthy worker supporting the profile."""
        candidates = [record for record in self.workers.values()
                      if record.healthy and record.supports(str(job.get("profile", "")))]
        if not candidates:
            return None
        candidates.sort(key=lambda record: (record.last_assigned_at, record.registered_at))
        chosen = candidates[0]
        chosen.last_assigned_at = time.time()
        self.assignments[str(job["job_id"])] = chosen.worker_id
        self.history.append({"at": time.time(), "event": "assigned",
                             "job_id": job["job_id"], "worker_id": chosen.worker_id})
        return chosen.worker_id

    def worker_for(self, job_id: str) -> Optional[str]:
        return self.assignments.get(str(job_id))

    # --- failure / reassignment (B6-05) ---------------------------------------

    def report_failure(self, job_id: str, worker_id: str, stage: str, *, reason: str = "") -> dict:
        if worker_id not in self.workers:
            raise SchedulerError(f"unknown worker {worker_id}")
        record = self.workers[worker_id]
        record.last_failure = {"stage": stage, "job_id": str(job_id), "at": time.time(),
                               "reason": reason}
        self.history.append({"at": time.time(), "event": "failure", "job_id": str(job_id),
                             "worker_id": worker_id, "stage": stage, "reason": reason})
        return {"stage": stage, "reassignable": self._reassignable(stage)}

    @staticmethod
    def _reassignable(stage: str) -> bool:
        return stage in ("pending", "assigned", "accepted", "running")

    def reassign(self, job: dict, *, from_stage: str) -> Optional[str]:
        """Reassign a failed job according to its stage.

        Committed-and-later stages are refused: the chain owns the dispute
        path from the commit on, and reassigning there would double-pay.
        """
        if not self._reassignable(from_stage):
            raise SchedulerError(
                f"a job in stage {from_stage!r} is never reassigned; the chain's dispute "
                "path owns it from the commit on")
        previous = self.assignments.pop(str(job["job_id"]), None)
        if previous is not None:
            record = self.workers.get(previous)
            if record is not None:
                record.healthy = False      # failed mid-job: out of rotation until re-registered
        self.history.append({"at": time.time(), "event": "reassign", "job_id": job["job_id"],
                             "from_worker": previous, "from_stage": from_stage})
        return self.match(job)
