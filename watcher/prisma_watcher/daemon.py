"""Watcher daemon (B3-01): discovery, bundle retrieval, verification, journal.

Pipeline per completed task:

```text
discover      completed graph results from the chain (bounded task scan; the
              B6 API is a future second source)
retrieve      the bundle from DA providers (first provider that answers with a
              bundle whose task id matches), resumably and with a size bound
verify        the frozen WatcherV2: root phase (artifact/version/GraphIDV2/
              roots/manifest/final checks), Freivalds (detection only) and the
              cheap exact recomputations
report        a verdict per task + optional challenge intent (B3-02 drives the
              challenge transaction; this module never triggers slashing —
              probabilistic detection is detection-only, the chain adjudicates)
journal       per-task entries (phase, verdict, cursor) so a restart resumes
              instead of re-verifying everything
```

The daemon is deliberately layered: chain access, bundle retrieval and the
verifier are injected, so the lifecycle is testable without a network, and the
frozen WatcherV2 is the only component that decides about fraud.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol


class WatcherError(Exception):
    """Base class for watcher failures."""


class ChainOps(Protocol):
    def completed_tasks(self, *, start_id: int, max_scan: int) -> list[dict]:
        ...

    def height(self) -> int:
        ...


class BundleSource(Protocol):
    def fetch(self, task: dict, destination: pathlib.Path) -> pathlib.Path:
        ...


class Verifier(Protocol):
    def verify(self, bundle_path: pathlib.Path, task: dict) -> dict:
        ...


@dataclass
class WatcherMetrics:
    discovered: int = 0
    verified: int = 0
    fraud: int = 0
    clean: int = 0
    failed: int = 0
    resumed: int = 0

    def to_dict(self) -> dict:
        return {"discovered": self.discovered, "verified": self.verified, "fraud": self.fraud,
                "clean": self.clean, "failed": self.failed, "resumed": self.resumed}


@dataclass
class WatcherDaemon:
    chain: ChainOps
    bundles: BundleSource
    verifier: Verifier
    journal_dir: pathlib.Path
    max_scan: int = 512
    metrics: WatcherMetrics = field(default_factory=WatcherMetrics)
    log: Callable[[str], None] = print

    def __post_init__(self):
        self.journal_dir = pathlib.Path(self.journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)

    # --- journal ----------------------------------------------------------

    def journal_path(self, task_id: int) -> pathlib.Path:
        return self.journal_dir / f"task-{task_id:08d}.json"

    def journal_read(self, task_id: int) -> Optional[dict]:
        try:
            return json.loads(self.journal_path(task_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def journal_write(self, task_id: int, **fields) -> dict:
        entry = {**(self.journal_read(task_id) or {}), "task_id": task_id,
                 "updated_at_ms": int(time.time() * 1000), **fields}
        path = self.journal_path(task_id)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(entry, indent=1) + "\n", encoding="utf-8")
        tmp.replace(path)
        return entry

    def pending_tasks(self) -> list[int]:
        """Tasks seen before but not verified yet (restart recovery)."""
        pending: list[int] = []
        for path in sorted(self.journal_dir.glob("task-*.json")):
            entry = json.loads(path.read_text(encoding="utf-8"))
            if entry.get("phase") in ("discovered", "retrieved"):
                pending.append(int(entry["task_id"]))
        return pending

    # --- one pass ---------------------------------------------------------

    def poll_once(self, *, work_dir: Optional[pathlib.Path] = None) -> list[dict]:
        work_dir = per_dir = pathlib.Path(work_dir or (self.journal_dir / "work"))
        per_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict] = []
        try:
            tasks = self.chain.completed_tasks(start_id=1, max_scan=self.max_scan)
        except Exception as exc:  # noqa: BLE001 - a chain hiccup must not kill the loop
            self.metrics.failed += 1
            self.log(f"task discovery failed: {exc}")
            return results
        seen = {int(task.get("id", 0)) for task in tasks}
        for task_id in self.pending_tasks():                     # restart recovery first
            if task_id not in seen:
                continue
            self.metrics.resumed += 1
            self.log(f"resuming task {task_id} from the journal")
            task = next(task for task in tasks if int(task.get("id", 0)) == task_id)
            results.append(self._process(task, work_dir))
            seen.discard(task_id)
        for task in tasks:
            if int(task.get("id", 0)) in seen:
                results.append(self._process(task, work_dir))
        return results

    def _process(self, task: dict, work_dir: pathlib.Path) -> dict:
        task_id = int(task.get("id", 0))
        existing = self.journal_read(task_id)
        if existing and existing.get("phase") == "verified":
            self.log(f"task {task_id}: already verified ({existing.get('verdict')}); skipping")
            return {"task_id": task_id, "action": "skipped", "verdict": existing.get("verdict")}
        self.metrics.discovered += 1
        self.journal_write(task_id, phase="discovered", status=str(task.get("status", "")))
        destination = work_dir / f"task-{task_id}"
        try:
            bundle = self.bundles.fetch(task, destination)
        except Exception as exc:  # noqa: BLE001 - report and continue with other tasks
            self.metrics.failed += 1
            self.journal_write(task_id, phase="retrieval_failed", reason=str(exc))
            self.log(f"task {task_id}: bundle retrieval failed: {exc}")
            return {"task_id": task_id, "action": "retrieval_failed", "reason": str(exc)}
        self.journal_write(task_id, phase="retrieved", bundle=str(bundle))
        try:
            verdict = self.verifier.verify(bundle, task)
        except Exception as exc:  # noqa: BLE001
            self.metrics.failed += 1
            self.journal_write(task_id, phase="verification_failed", reason=str(exc))
            self.log(f"task {task_id}: verification failed: {exc}")
            return {"task_id": task_id, "action": "verification_failed", "reason": str(exc)}
        self.metrics.verified += 1
        outcome = str(verdict.get("verdict", "unknown"))
        if outcome in ("fraud", "FRAUD"):
            self.metrics.fraud += 1
            self.log(f"task {task_id}: FRAUD detected — the challenge path (B3-02) drives the "
                     "on-chain dispute; detection alone never slashes")
        else:
            self.metrics.clean += 1
        self.journal_write(task_id, phase="verified", verdict=outcome, detail=verdict.get("detail", ""))
        return {"task_id": task_id, "action": "verified", "verdict": outcome}

    # --- scheduler --------------------------------------------------------

    def run(self, *, interval_seconds: float = 15.0, max_iterations: Optional[int] = None,
            sleep: Callable[[float], None] = time.sleep,
            work_dir: Optional[pathlib.Path] = None) -> WatcherMetrics:
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            self.poll_once(work_dir=work_dir)
            if max_iterations is not None and iterations >= max_iterations:
                break
            sleep(interval_seconds)
        return self.metrics
