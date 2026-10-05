"""Watcher automatic challenge (B3-02): decision policy, bond management,
window tracking, dispute driving, outcome handling.

The frozen rules this module respects:

- **Evidence, not probability.** A challenge is opened only on a detection
  result the frozen WatcherV2 recorded as fraud (root-phase failure, a
  Freivalds residual, or a cheap-op exact mismatch) — never on a suspicion,
  and never with a probability attached. The decision record quotes the
  detection evidence verbatim.
- **Never challenge after the window.** The chain's deadline is a height; a
  challenge inside the safety margin is not started.
- **Never spend into an unfunded state.** The challenger bond is checked
  against the local balance floor before any transaction; a shortfall is an
  explicit refusal that names the amount.
- **The chain adjudicates.** Outcomes (ChallengerWins / WorkerWins /
  BothInvalid / timeout) are recorded from the chain's dispute state, not
  invented locally.

Dispute driving is layered behind ``DisputeOps`` so the full state machine
(open -> trail -> midpoint -> wide -> outcome) is testable without a chain;
``PrismadDisputeOps`` is the only component that knows command shapes.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol


class ChallengeError(Exception):
    """Base class for challenge-flow failures."""


class NoEvidence(ChallengeError):
    """Refusing to challenge without recorded detection evidence."""


class UnfundedChallenge(ChallengeError):
    """The challenger cannot pay the bond: an actionable refusal."""


class WindowClosed(ChallengeError):
    """The challenge window is (about to be) over."""


class DisputeOps(Protocol):
    def height(self) -> int:
        ...

    def balance(self, account: str) -> int:
        ...

    def dispute_state(self, task_id: int) -> dict:
        ...

    def open_challenge(self, task_id: int, bond: int, evidence_digest: str) -> str:
        ...


@dataclass
class DecisionPolicy:
    """What triggers a challenge (detection-evidence based, never probabilistic)."""

    bond_uprsm: int = 10_000
    safety_margin_blocks: int = 5
    require_verdict: str = "fraud"

    def decide(self, task: dict, verification: dict, *, height: int) -> dict:
        verdict = str(verification.get("verdict", ""))
        if verdict != self.require_verdict:
            return {"challenge": False,
                    "reason": f"verdict {verdict!r} is not {self.require_verdict!r}"}
        report = verification.get("report") or {}
        evidence = self._evidence(report)
        if not evidence:
            return {"challenge": False,
                    "reason": "fraud verdict without recorded detection evidence; refusing "
                              "to challenge on a bare verdict"}
        deadline = int(task.get("challenge_end", task.get("deadline", 0)) or 0)
        if deadline and height + self.safety_margin_blocks >= deadline:
            return {"challenge": False,
                    "reason": f"challenge window closes at {deadline}, inside the safety "
                              f"margin at height {height}"}
        return {"challenge": True, "evidence": evidence, "deadline_height": deadline}

    @staticmethod
    def _evidence(report: dict) -> str:
        """A digest-ready quote of the concrete detection facts."""
        parts: list[str] = []
        root_phase = report.get("root_phase") or {}
        if root_phase:
            parts.append(f"root_phase: {json.dumps(root_phase, sort_keys=True)[:400]}")
        math_phase = report.get("math_phase") or {}
        for name in ("freivalds", "cheap_ops", "fraud", "mismatches"):
            block = math_phase.get(name)
            if block:
                parts.append(f"{name}: {json.dumps(block, sort_keys=True)[:400]}")
        return " | ".join(parts) if parts else ""


@dataclass
class DisputeDriver:
    """Drives one dispute to its outcome and records the transcript."""

    ops: DisputeOps
    challenger: str
    policy: DecisionPolicy = field(default_factory=DecisionPolicy)
    log: Callable[[str], None] = print

    def consider(self, task: dict, verification: dict) -> dict:
        """The full decision: evidence, bond, window. No side effects."""
        height = self.ops.height()
        decision = self.policy.decide(task, verification, height=height)
        if not decision.get("challenge"):
            return decision
        balance = self.ops.balance(self.challenger)
        if balance < self.policy.bond_uprsm:
            raise UnfundedChallenge(
                f"challenger {self.challenger} has {balance} uprsm but the bond needs "
                f"{self.policy.bond_uprsm}; fund the account before challenging")
        return decision

    def challenge(self, task: dict, verification: dict, *,
                  record_dir: Optional[pathlib.Path] = None) -> dict:
        """Open the challenge (evidence-gated) and record the transcript digest."""
        decision = self.consider(task, verification)
        if not decision.get("challenge"):
            return decision
        task_id = int(task.get("id", 0))
        evidence = decision["evidence"]
        transcript = {"task_id": task_id, "opened_height": self.ops.height(),
                      "evidence": evidence, "bond_uprsm": self.policy.bond_uprsm}
        digest = self._digest(transcript)
        txhash = self.ops.open_challenge(task_id, self.policy.bond_uprsm, digest)
        transcript["txhash"] = txhash
        transcript["digest"] = digest
        if record_dir:
            record_dir.mkdir(parents=True, exist_ok=True)
            (record_dir / f"challenge-{task_id}.json").write_text(
                json.dumps(transcript, indent=1) + "\n", encoding="utf-8")
        self.log(f"task {task_id}: challenge opened (tx {txhash}, transcript {digest[:16]}…)")
        return {"challenge": True, "task_id": task_id, "txhash": txhash,
                "transcript_digest": digest}

    @staticmethod
    def _digest(transcript: dict) -> str:
        import hashlib

        blob = json.dumps(transcript, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    # --- dispute driving --------------------------------------------------

    def drive(self, task_id: int, *, max_steps: int = 64,
              record_dir: Optional[pathlib.Path] = None) -> dict:
        """Follow the dispute until an outcome; record every observed state.

        The chain's dispute state machine (claiming -> bisection ->
        arb_ready -> resolved) is polled; the driver only takes steps the
        frozen state machine asks for and never invents moves.
        """
        transcript: list[dict] = []
        for _step in range(max_steps):
            state = self.ops.dispute_state(task_id)
            phase = str(state.get("phase", state.get("status", "unknown")))
            transcript.append({"height": self.ops.height(), "phase": phase,
                               "state": {k: v for k, v in state.items()
                                         if k not in ("phase", "status")}})
            outcome = self._outcome(phase, state)
            if outcome:
                record = {"task_id": task_id, "outcome": outcome,
                          "transcript": transcript,
                          "digest": self._digest({"task_id": task_id, "transcript": transcript})}
                if record_dir:
                    record_dir.mkdir(parents=True, exist_ok=True)
                    (record_dir / f"dispute-{task_id}.json").write_text(
                        json.dumps(record, indent=1) + "\n", encoding="utf-8")
                self.log(f"task {task_id}: dispute outcome {outcome} "
                         f"(transcript {record['digest'][:16]}…)")
                return record
            time.sleep(0)   # the real scheduler sleeps between polls; tests inject time
        raise ChallengeError(f"task {task_id}: dispute did not resolve within {max_steps} polls")

    @staticmethod
    def _outcome(phase: str, state: dict) -> Optional[str]:
        normalized = phase.lower()
        if "challenger_wins" in normalized or normalized == "challengerwins":
            return "ChallengerWins"
        if "worker_wins" in normalized or normalized == "workerwins":
            return "WorkerWins"
        if "both_invalid" in normalized:
            return "BothInvalid"
        if "timeout" in normalized or "timed_out" in normalized:
            return "timeout"
        if normalized in ("resolved", "finalized", "refund", "fraud"):
            result = str(state.get("result", state.get("verdict", "")))
            return result or normalized
        return None
