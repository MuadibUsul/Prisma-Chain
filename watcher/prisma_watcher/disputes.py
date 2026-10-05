"""Durable dispute tracking (B3-03): the watcher never loses a dispute it is
running, across restarts.

Every dispute the watcher participates in gets a journal file with:

```text
task_id, phase (open/trail/midpoint/wide/resolved), bond_uprsm, txhash,
transcript digest, deadlines (heights), locked roots, midpoint history,
last observed chain state
```

Restart procedure (`DisputeLedger.recover`): re-read every non-resolved
dispute, re-derive the chain's current dispute state through DisputeOps, and
resume exactly where the chain says things stand — the chain is the source of
truth, the journal only prevents *forgetting* that a dispute is running. A
dispute the chain has already resolved is marked resolved with the chain's
outcome; nothing is double-submitted (opening a second challenge is refused
because the ledger knows the first one exists).

The F.5C A6-09 requirement (byte-identical persisted dispute, same verdict
after a full-stack restart) is mirrored here at the watcher level.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .challenge import DisputeOps

ACTIVE_PHASES = ("open", "trail", "midpoint", "wide", "arb_ready")
RESOLVED_PHASES = ("resolved", "challenger_wins", "worker_wins", "both_invalid", "timeout")


class DisputeLedgerError(Exception):
    """Base class for ledger failures."""


@dataclass
class DisputeLedger:
    directory: pathlib.Path
    log: Callable[[str], None] = print

    def __post_init__(self):
        self.directory = pathlib.Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: int) -> pathlib.Path:
        return self.directory / f"dispute-{task_id:08d}.json"

    def read(self, task_id: int) -> Optional[dict]:
        try:
            return json.loads(self._path(task_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def record(self, task_id: int, phase: str, **fields) -> dict:
        previous = self.read(task_id) or {}
        entry = {**previous, "task_id": task_id, "phase": phase,
                 "updated_at_ms": int(time.time() * 1000), **fields}
        history = list(entry.get("history", []))
        history.append({"phase": phase, "at_ms": entry["updated_at_ms"]})
        entry["history"] = history[-64:]
        path = self._path(task_id)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(entry, indent=1) + "\n", encoding="utf-8")
        tmp.replace(path)
        return entry

    def active(self) -> list[dict]:
        entries = []
        for path in sorted(self.directory.glob("dispute-*.json")):
            entry = json.loads(path.read_text(encoding="utf-8"))
            if entry.get("phase") in ACTIVE_PHASES:
                entries.append(entry)
        return entries

    def ensure_not_duplicate(self, task_id: int) -> None:
        entry = self.read(task_id)
        if entry and entry.get("phase") in ACTIVE_PHASES:
            raise DisputeLedgerError(
                f"task {task_id} already has an active dispute (tx {entry.get('txhash')}); "
                "refusing to open a second challenge")

    def recover(self, ops: DisputeOps, *, driver: Optional[Callable[[int], dict]] = None) -> dict:
        """Re-derive every active dispute from the chain and update the ledger.

        - the chain resolved it -> record the outcome (nothing to drive);
        - the chain still runs it -> keep it active with the chain's state
          (and optionally hand it to ``driver`` for the next step).
        """
        report = {"active": 0, "resolved_on_chain": 0, "still_running": 0, "records": []}
        for entry in self.active():
            task_id = int(entry["task_id"])
            report["active"] += 1
            state = ops.dispute_state(task_id)
            phase = str(state.get("phase", state.get("status", "unknown")))
            normalized = phase.lower()
            if any(word in normalized for word in RESOLVED_PHASES) or "resolved" in normalized:
                outcome = str(state.get("result", state.get("verdict", phase)))
                self.record(task_id, "resolved", outcome=outcome,
                            chain_state={k: v for k, v in state.items()
                                         if k not in ("phase", "status")})
                report["resolved_on_chain"] += 1
                report["records"].append({"task_id": task_id, "action": "resolved",
                                          "outcome": outcome})
                self.log(f"dispute {task_id}: resolved on chain ({outcome})")
            else:
                self.record(task_id, entry.get("phase", "open"),
                            last_chain_phase=phase,
                            last_chain_state={k: v for k, v in state.items()
                                              if k not in ("phase", "status")})
                report["still_running"] += 1
                report["records"].append({"task_id": task_id, "action": "resume",
                                          "chain_phase": phase})
                self.log(f"dispute {task_id}: still running on chain ({phase}); resuming")
                if driver is not None:
                    driver(task_id)
        return report

