"""Worker job discovery, compatibility filtering and atomic accept (B2-04).

Design decisions, all driven by the frozen interfaces:

- **Discovery** uses the chain (``query compute task --task-id N`` scanned
  upwards from the last seen id, bounded) because the B6 job API does not
  exist yet; the scan bound is explicit and configurable, never unbounded.
- **Compatibility** is a hard filter: a task whose model/spec/mode is not in
  the worker's configured set is never accepted. The configured set defaults
  to the frozen canonical profile of this release.
- **Journal-before-side-effects**: the intent (``accepting``) is written to
  a local write-ahead journal — atomically, fsynced — *before* the accept
  transaction is submitted, and the outcome (``accepted`` + tx hash or
  ``accept_failed``) is written after. A crash mid-accept leaves an
  ``accepting`` entry that the next run reconciles against the chain, so a
  duplicate or late accept can never double-execute.
- **Deadline**: a task whose deadline is within the safety margin (or already
  passed) is refused before any transaction.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import chain as chain_mod

# Journal phases (the write-ahead state machine of one task on this worker).
PHASE_DISCOVERED = "discovered"
PHASE_ACCEPTING = "accepting"
PHASE_ACCEPTED = "accepted"
PHASE_ACCEPT_FAILED = "accept_failed"
PHASE_EXECUTING = "executing"
PHASE_SUBMITTED = "submitted"
PHASE_TERMINAL = "terminal"

# The frozen canonical profile this release executes (canonical graph v2 with
# the A13W10 policy). Overridable, but only explicitly.
FROZEN_PROFILE_MODEL_IDS = ("qwen3-0.6b-layer0-v2",)
FROZEN_MODES = ("verifiable",)
OPEN_STATUSES = ("posted",)


class JobError(Exception):
    """Job handling failed in a way the operator must see."""


class IncompatibleJob(JobError):
    """The task does not match this worker's configured capability/profile."""


class Journal:
    """Write-ahead journal: one atomically written JSON file per task."""

    def __init__(self, directory: pathlib.Path | str):
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: int) -> pathlib.Path:
        return self.directory / f"task-{task_id:08d}.json"

    def read(self, task_id: int) -> Optional[dict]:
        try:
            return json.loads(self._path(task_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            raise JobError(f"journal entry for task {task_id} is corrupt: {exc}") from exc

    def record(self, task_id: int, phase: str, **fields) -> dict:
        previous = self.read(task_id) or {}
        entry = {**previous, "task_id": task_id, "phase": phase,
                 "updated_at_ms": int(time.time() * 1000), **fields}
        history = list(entry.get("history", []))
        history.append({"phase": phase, "at_ms": entry["updated_at_ms"]})
        entry["history"] = history[-32:]
        path = self._path(task_id)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, indent=1) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:  # directory fsync is not available on Windows
            dir_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):
            pass
        return entry

    def entries(self) -> list[dict]:
        out = []
        for path in sorted(self.directory.glob("task-*.json")):
            out.append(json.loads(path.read_text(encoding="utf-8")))
        return out


@dataclass
class TaskSummary:
    task_id: int
    status: str
    mode: str
    model_id: str
    spec_version: str
    requester: str = ""
    worker: str = ""
    deadline: int = 0
    reserved_bond: int = 0
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_chain(cls, payload: dict) -> "TaskSummary":
        return cls(
            task_id=int(payload.get("id", 0)),
            status=str(payload.get("status", "")),
            mode=str(payload.get("mode", "")),
            model_id=str(payload.get("model_id", "")),
            spec_version=str(payload.get("spec_version", "")),
            requester=str(payload.get("requester", "")),
            worker=str(payload.get("worker", "") or ""),
            deadline=int(payload.get("deadline", 0) or 0),
            reserved_bond=int(payload.get("reserved_bond", 0) or 0),
            raw=payload,
        )


def discover(client: chain_mod.ChainClient, *, start_id: int = 1, max_scan: int = 512,
             max_consecutive_missing: int = 64,
             log: Callable[[str], None] = lambda _message: None) -> list[TaskSummary]:
    """Scan task ids upwards until the scan bound or a run of misses.

    The frozen chain has per-id task queries only; the scan bound is an
    explicit configuration, never an unbounded loop.
    """
    tasks: list[TaskSummary] = []
    task_id = max(start_id, 1)
    scanned = 0
    missing_run = 0
    while scanned < max_scan:
        scanned += 1
        payload = client.task(task_id)
        if not payload:
            missing_run += 1
            if missing_run >= max_consecutive_missing:
                log(f"discovery: {missing_run} consecutive missing ids after {task_id}; stopping")
                break
        else:
            missing_run = 0
            summary = TaskSummary.from_chain(payload)
            if summary.status in OPEN_STATUSES:
                tasks.append(summary)
        task_id += 1
    return tasks


def compatibility(task: TaskSummary, *, model_ids: tuple[str, ...] = FROZEN_PROFILE_MODEL_IDS,
                  modes: tuple[str, ...] = FROZEN_MODES,
                  spec_versions: tuple[str, ...] = ()) -> tuple[bool, str]:
    """(ok, reason). Forbidden to accept anything that does not match."""
    if task.status not in OPEN_STATUSES:
        return False, f"status {task.status!r} is not open"
    if task.mode not in modes:
        return False, f"mode {task.mode!r} not in {list(modes)}"
    if task.model_id not in model_ids:
        return False, f"model {task.model_id!r} not in {list(model_ids)}"
    if spec_versions and task.spec_version not in spec_versions:
        return False, f"spec {task.spec_version!r} not in {list(spec_versions)}"
    return True, "compatible"


def check_deadline(task: TaskSummary, current_height: int, *, safety_margin: int = 5) -> tuple[bool, str]:
    if task.deadline and current_height + safety_margin >= task.deadline:
        return False, (f"deadline {task.deadline} is within the safety margin "
                       f"(height {current_height} + {safety_margin})")
    return True, "deadline ok"


def accept_task(client: chain_mod.ChainClient, journal: Journal, task: TaskSummary, *,
                account_scalar_hex: str, current_height: int, safety_margin: int = 5,
                log: Callable[[str], None] = print) -> dict:
    """Accept one compatible task exactly once, journaling before side effects."""
    existing = journal.read(task.task_id)
    if existing and existing.get("phase") in (PHASE_ACCEPTED, PHASE_EXECUTING,
                                              PHASE_SUBMITTED, PHASE_TERMINAL):
        log(f"task {task.task_id}: already {existing['phase']}; refusing to accept again")
        return existing
    if existing and existing.get("phase") == PHASE_ACCEPTING:
        # A previous run crashed between the journal write and the outcome.
        # Reconcile against the chain instead of guessing.
        on_chain = client.task(task.task_id) or {}
        worker = str(on_chain.get("worker", "") or "")
        if worker:
            entry = journal.record(task.task_id, PHASE_ACCEPTED,
                                   reconciled=True, note="crash during accept; chain shows a worker")
            log(f"task {task.task_id}: reconciled after crash (chain worker={worker})")
            return entry
        log(f"task {task.task_id}: previous accept attempt left no chain effect; retrying")

    ok, reason = check_deadline(task, current_height, safety_margin=safety_margin)
    if not ok:
        journal.record(task.task_id, PHASE_ACCEPT_FAILED, reason=reason)
        raise JobError(f"task {task.task_id}: {reason}")

    journal.record(task.task_id, PHASE_ACCEPTING,
                   model_id=task.model_id, mode=task.mode, spec_version=task.spec_version,
                   requester=task.requester, deadline=task.deadline)
    try:
        txhash = client.accept_task(task.task_id, account_scalar_hex)
    except chain_mod.ChainError as exc:
        journal.record(task.task_id, PHASE_ACCEPT_FAILED, reason=str(exc))
        raise
    entry = journal.record(task.task_id, PHASE_ACCEPTED, txhash=txhash)
    log(f"task {task.task_id}: accepted (tx {txhash})")
    return entry


def run_discovery_cycle(client: chain_mod.ChainClient, journal: Journal, *,
                        account_scalar_hex: str, model_ids: tuple[str, ...] = FROZEN_PROFILE_MODEL_IDS,
                        modes: tuple[str, ...] = FROZEN_MODES,
                        start_id: int = 1, max_scan: int = 512, accept_limit: int = 1,
                        safety_margin: int = 5,
                        log: Callable[[str], None] = print) -> list[dict]:
    """One discovery+accept pass: find compatible open tasks and accept up to
    ``accept_limit`` of them (default one, so a worker never races itself)."""
    height = int(client.status().get("sync_info", {}).get("latest_block_height", "0") or 0)
    accepted: list[dict] = []
    for task in discover(client, start_id=start_id, max_scan=max_scan, log=log):
        ok, reason = compatibility(task, model_ids=model_ids, modes=modes)
        if not ok:
            log(f"task {task.task_id}: skipped ({reason})")
            continue
        if len(accepted) >= accept_limit:
            log(f"task {task.task_id}: compatible but the accept limit is reached")
            break
        try:
            accepted.append(accept_task(client, journal, task, account_scalar_hex=account_scalar_hex,
                                        current_height=height, safety_margin=safety_margin, log=log))
        except chain_mod.ChainError as exc:
            log(f"task {task.task_id}: accept failed: {exc}")
    return accepted
