"""Crash recovery (B2-09): classify every journal state, reconcile against the
chain, and never resume into an inconsistent view.

Every crash point of the job pipeline maps to exactly one action:

```text
phase            action            why
discovered       retry             nothing was submitted; re-discovery is free
accepting        reconcile_chain   the chain decides: ours -> accepted, else retry accept
accept_failed    retry | abandon   retry while the task is still open; abandon once it is not
accepted         resume            execution is deterministic; re-executing is safe
executing        resume            partial output is discarded and recomputed
executed         resume (commit)   build+sign; the journal guard prevents a double commit
submitted        reconcile_chain   the chain must show our commit before DA
da_pending       resume (upload)   re-upload only to providers that did not verify yet
da_quorum        terminal          nothing left for the worker to do
abandoned        terminal          a previous run proved the chain disagrees
```

Rules that make this safe to run at any time:

- **No transactions here.** Recovery only reads the chain and moves journal
  phases; any transaction (accept, commit) is left to the normal pipeline,
  whose guards (journal-first idempotency, duplicate-submit refusal) already
  prevent double submission. Recovery can therefore run twice harmlessly.
- **The chain wins.** If the chain shows another worker on the task, or the
  task finalized/refunded, the task is abandoned or terminated — never
  resumed into an inconsistent view.
- **Minimal rework.** ``providers_to_upload`` returns only the providers whose
  attestations were not verified before the crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from . import chain as chain_mod
from .jobs import (
    PHASE_ABANDONED,
    PHASE_ACCEPTED,
    PHASE_ACCEPTING,
    PHASE_ACCEPT_FAILED,
    PHASE_DA_PENDING,
    PHASE_DA_QUORUM,
    PHASE_DISCOVERED,
    PHASE_EXECUTED,
    PHASE_EXECUTING,
    PHASE_SUBMITTED,
    PHASE_TERMINAL,
    Journal,
)

ACTION_RETRY = "retry"
ACTION_RESUME = "resume"
ACTION_RECONCILE = "reconcile_chain"
ACTION_TERMINAL = "terminal"
ACTION_ABANDON = "abandon"

_CHAIN_FINAL_STATES = ("finalized", "fraud", "refunded", "availability_failed")


@dataclass
class Decision:
    task_id: int
    phase: str
    action: str
    reason: str
    steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "phase": self.phase, "action": self.action,
                "reason": self.reason, "steps": self.steps}


def classify(entry: Optional[dict], *, chain_task: Optional[dict], account_address: str) -> Decision:
    """Pure classification of one journal entry against the chain view."""
    if not entry:
        return Decision(0, "none", ACTION_TERMINAL, "no journal entry; nothing to recover")
    task_id = int(entry.get("task_id", 0))
    phase = str(entry.get("phase", ""))
    chain_task = chain_task or {}
    status = str(chain_task.get("status", ""))
    chain_worker = str(chain_task.get("worker", "") or "")

    if phase == PHASE_ABANDONED:
        return Decision(task_id, phase, ACTION_TERMINAL, "already abandoned by a previous run")
    if phase == PHASE_TERMINAL:
        return Decision(task_id, phase, ACTION_TERMINAL, "already terminal")

    if status in _CHAIN_FINAL_STATES:
        return Decision(task_id, phase, ACTION_TERMINAL,
                        f"the chain says the task is {status}; the worker has nothing to add")

    if phase == PHASE_DISCOVERED:
        return Decision(task_id, phase, ACTION_RETRY, "nothing was submitted; safe to redo",
                        steps=["re-run discovery/accept"])
    if phase == PHASE_ACCEPTING:
        if chain_worker == account_address:
            return Decision(task_id, phase, ACTION_RECONCILE,
                            "crash during accept; the chain already shows this worker",
                            steps=["record the accepted phase (no transaction)"])
        if chain_worker:
            return Decision(task_id, phase, ACTION_ABANDON,
                            f"the chain assigned the task to {chain_worker[:12]}…",
                            steps=["record the abandoned phase"])
        return Decision(task_id, phase, ACTION_RECONCILE,
                        "crash during accept; the chain shows no assignment yet",
                        steps=["retry the accept through the normal pipeline (journal-guarded)"])
    if phase == PHASE_ACCEPT_FAILED:
        if status == "posted":
            return Decision(task_id, phase, ACTION_RETRY,
                            f"the task is still open on chain ({entry.get('reason', 'accept failed')})",
                            steps=["retry the accept"])
        return Decision(task_id, phase, ACTION_ABANDON,
                        f"the task is no longer open ({status or 'unknown'})",
                        steps=["record the abandoned phase"])
    if phase in (PHASE_ACCEPTED, PHASE_EXECUTING):
        if chain_worker and chain_worker != account_address:
            return Decision(task_id, phase, ACTION_ABANDON,
                            "the chain shows a different worker; never execute someone else's assignment")
        return Decision(task_id, phase, ACTION_RESUME,
                        "execution is deterministic; recompute from scratch",
                        steps=["re-execute the accepted task"])
    if phase == PHASE_EXECUTED:
        return Decision(task_id, phase, ACTION_RESUME, "resume at the commit step",
                        steps=["build and sign CommitV3 (duplicate guard active)"])
    if phase == PHASE_SUBMITTED:
        if chain_worker != account_address:
            return Decision(task_id, phase, ACTION_ABANDON,
                            "a commit was recorded locally but the chain does not credit this worker")
        return Decision(task_id, phase, ACTION_RECONCILE,
                        "the chain must show our commit before DA work resumes",
                        steps=["verify the tx inclusion for the journaled commit_id",
                               "then continue to the DA upload"])
    if phase == PHASE_DA_PENDING:
        return Decision(task_id, phase, ACTION_RESUME, "quorum not reached yet; resume the upload",
                        steps=["re-upload only to providers without a verified attestation"])
    if phase == PHASE_DA_QUORUM:
        return Decision(task_id, phase, ACTION_TERMINAL,
                        "verified quorum recorded; the chain finalizes")
    return Decision(task_id, phase, ACTION_TERMINAL, f"unknown phase {phase!r}; nothing safe to do")


def providers_to_upload(entry: dict, provider_names: list[str]) -> list[str]:
    """Providers whose attestation was NOT verified before the crash."""
    verified = {v.get("provider") for v in entry.get("da_verified", [])}
    return [name for name in provider_names if name not in verified]


def recover_task(client: chain_mod.ChainClient, journal: Journal, task_id: int, *,
                 account_address: str,
                 log: Callable[[str], None] = print) -> Decision:
    """Classify one task and apply the journal-only part of the decision."""
    entry = journal.read(task_id)
    chain_task: dict = {}
    if entry:
        chain_task = client.task(task_id)
    decision = classify(entry, chain_task=chain_task, account_address=account_address)
    if decision.action == ACTION_RECONCILE:
        if decision.phase == PHASE_ACCEPTING and str(chain_task.get("worker", "")) == account_address:
            journal.record(task_id, PHASE_ACCEPTED, reconciled=True,
                           note="recovery: the chain already shows this worker")
        else:
            journal.record(task_id, str(entry.get("phase", "")), recovered=True,
                           recovery_action=decision.action, recovery_reason=decision.reason)
        decision.steps = ["reconcile recorded in the journal"] + decision.steps
    elif decision.action == ACTION_ABANDON:
        journal.record(task_id, PHASE_ABANDONED, reason=decision.reason)
    elif decision.action == ACTION_TERMINAL and decision.phase not in (
            PHASE_TERMINAL, PHASE_ABANDONED, "none"):
        journal.record(task_id, PHASE_TERMINAL, reason=decision.reason)
    elif decision.action in (ACTION_RETRY, ACTION_RESUME):
        journal.record(task_id, decision.phase, recovered=True,
                       recovery_action=decision.action, recovery_reason=decision.reason)
    log(f"task {task_id}: {decision.phase} -> {decision.action} ({decision.reason})")
    return decision


def recover_all(client: chain_mod.ChainClient, journal: Journal, *,
                account_address: str,
                log: Callable[[str], None] = print) -> list[Decision]:
    """Recover every journalled task. Read-only on chain; journal writes only."""
    decisions = []
    for entry in journal.entries():
        decisions.append(recover_task(client, journal, int(entry.get("task_id", 0)),
                                      account_address=account_address, log=log))
    return decisions


def summary(decisions: list[Decision]) -> dict:
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.action] = counts.get(decision.action, 0) + 1
    return {"tasks": len(decisions), "by_action": counts,
            "decisions": [d.to_dict() for d in decisions]}
