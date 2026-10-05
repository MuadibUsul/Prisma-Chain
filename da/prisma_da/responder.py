"""DA challenge responder (B4-03): answer on-chain typed chunk challenges
inside their deadlines, and report after-loss honestly.

Shape of the job:

```text
discover    open challenges naming this provider (the chain exposes per-task
            queries, so the adapter scans tasks with an explicit bound)
locate      the artifact + chunk in the local store
prove       the frozen typed chunk proof (storage.tile, verified by
            merkle_proofs.verify_inclusion) — never a fabricated proof
answer      the response transaction, before the deadline minus a safety margin
after-loss  the artifact is gone: record an objective report (task id, challenge,
            witness information) and stop; the chain's penalty path applies —
            nothing is fabricated to avoid it
metrics     discovered / answered / too-late / lost, response latency
```

The deadline scheduler uses **chain height**, not wall time (the chain's
deadline is a height), and polls with a bounded interval; no human is in the
loop. Chain interaction is behind ``ChainOps`` so the responder logic is fully
testable without a chain, and the ``prismad`` CLI adapter is the only place
that knows the command shapes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from .storage import ArtifactIndex, IntegrityError


class ResponderError(Exception):
    """Base class for responder failures."""


class ChainOps(Protocol):
    """What the responder needs from the chain (implemented by the CLI adapter)."""

    def height(self) -> int:
        ...

    def open_challenges(self, provider: str) -> list[dict]:
        ...

    def respond(self, challenge: dict, tile: dict) -> str:
        ...


@dataclass
class ResponderMetrics:
    discovered: int = 0
    answered: int = 0
    too_late: int = 0
    lost: int = 0
    failed: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def record_latency(self, seconds: float) -> None:
        self.latencies_ms.append(round(seconds * 1000, 3))

    def to_dict(self) -> dict:
        latencies = self.latencies_ms or [0.0]
        return {"discovered": self.discovered, "answered": self.answered, "too_late": self.too_late,
                "lost": self.lost, "failed": self.failed,
                "response_latency_ms": {"last": latencies[-1],
                                        "max": max(latencies) if self.latencies_ms else 0.0,
                                        "avg": round(sum(latencies) / len(latencies), 3)}}


@dataclass
class ChallengeResponder:
    chain: ChainOps
    storage: ArtifactIndex
    provider: str
    safety_margin_blocks: int = 5
    metrics: ResponderMetrics = field(default_factory=ResponderMetrics)
    losses: list[dict] = field(default_factory=list)
    log: Callable[[str], None] = print

    # --- one pass ---------------------------------------------------------

    def poll_once(self) -> list[dict]:
        """Discover and answer every open challenge that is still in time."""
        results: list[dict] = []
        try:
            challenges = self.chain.open_challenges(self.provider)
        except Exception as exc:  # noqa: BLE001 - a chain hiccup must not kill the loop
            self.metrics.failed += 1
            self.log(f"challenge discovery failed: {exc}")
            return results
        height = self.chain.height()
        for challenge in challenges:
            self.metrics.discovered += 1
            results.append(self._handle(challenge, height))
        return results

    def _handle(self, challenge: dict, height: int) -> dict:
        task_id = int(challenge.get("task_id", 0))
        deadline = int(challenge.get("deadline_height", 0) or 0)
        tile_i = int(challenge.get("tile_i", 0) or 0)
        tile_j = int(challenge.get("tile_j", 0) or 0)
        if deadline and height + self.safety_margin_blocks >= deadline:
            self.metrics.too_late += 1
            self.log(f"challenge for task {task_id}: deadline {deadline} is inside the safety "
                     f"margin (height {height} + {self.safety_margin_blocks}); not answering")
            return {"task_id": task_id, "action": "too_late", "deadline": deadline}
        try:
            tile = self.storage.tile(task_id, tile_i, tile_j)
        except IntegrityError as exc:
            self.metrics.lost += 1
            entry = {"task_id": task_id, "action": "lost", "reason": str(exc),
                     "challenge": challenge}
            self.losses.append(entry)
            self.log(f"challenge for task {task_id}: the artifact failed its hash and was "
                     f"quarantined; reporting after-loss objectively")
            return entry
        if tile is None:
            self.metrics.lost += 1
            entry = {"task_id": task_id, "action": "lost",
                     "reason": "artifact is not stored (or the chunk is out of range)",
                     "challenge": challenge}
            self.losses.append(entry)
            self.log(f"challenge for task {task_id}: no artifact to prove; reporting after-loss "
                     f"objectively (the chain's penalty path applies)")
            return entry
        started = time.monotonic()
        try:
            txhash = self.chain.respond(challenge, tile)
        except Exception as exc:  # noqa: BLE001 - record and keep serving
            self.metrics.failed += 1
            self.log(f"challenge for task {task_id}: response transaction failed: {exc}")
            return {"task_id": task_id, "action": "failed", "reason": str(exc)}
        self.metrics.answered += 1
        self.metrics.record_latency(time.monotonic() - started)
        self.log(f"challenge for task {task_id}: answered (tx {txhash})")
        return {"task_id": task_id, "action": "answered", "txhash": txhash,
                "output_root": tile.get("output_root"), "leaf": tile.get("leaf")}

    # --- scheduler --------------------------------------------------------

    def run(self, *, interval_seconds: float = 10.0, max_iterations: Optional[int] = None,
            sleep: Callable[[float], None] = time.sleep) -> ResponderMetrics:
        """The deadline scheduler loop: poll, answer, repeat. Bounded for tests."""
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            self.poll_once()
            if max_iterations is not None and iterations >= max_iterations:
                break
            sleep(interval_seconds)
        return self.metrics

    def live_challenges(self) -> set[int]:
        """Task ids with an unanswered challenge — GC must never delete these."""
        live: set[int] = set()
        try:
            for challenge in self.chain.open_challenges(self.provider):
                live.add(int(challenge.get("task_id", 0)))
        except Exception as exc:  # noqa: BLE001
            self.log(f"live-challenge lookup failed: {exc}")
        return live
