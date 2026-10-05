"""Job / VWR presentation (B6-06): one view of a job across the chain's
states, with verification status ONLY as the chain reports it.

The job state machine is fixed by the roadmap; the mapping from the chain's
task statuses is documented and conservative — an unknown chain state maps to
``unknown`` and is surfaced as such, never guessed into a friendlier name.
Fabricating a verification status the chain did not confirm is forbidden, so
``verification`` is a passthrough of the chain's fields (with None when the
chain has not confirmed anything yet).
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol

# The roadmap's job state machine (fixed).
JOB_STATES = ("queued", "assigned", "accepted", "executing", "committed",
              "availability_ready", "verifying", "finalized", "refunded", "fraud",
              "cancelled", "failed")

# chain task status -> job state (documented mapping; anything else -> unknown)
CHAIN_STATUS_MAP = {
    "posted": "queued",
    "pending": "queued",
    "assigned": "assigned",
    "accepted": "accepted",
    "executing": "executing",
    "result_submitted": "committed",
    "committed": "committed",
    "availability_pending": "availability_ready",
    "challenge_window_ready": "availability_ready",
    "challenged": "verifying",
    "challenger_wins": "fraud",
    "worker_wins": "finalized",
    "finalized": "finalized",
    "refunded": "refunded",
    "availability_failed": "refunded",
    "fraud": "fraud",
}


class ChainView(Protocol):
    """What presentation needs from the chain (all reads)."""

    def task(self, task_id: int) -> dict:
        ...

    def receipt(self, task_id: int) -> Optional[dict]:
        ...


def job_state_for(chain_status: str) -> str:
    return CHAIN_STATUS_MAP.get(str(chain_status), "unknown")


def present_job(job: dict, chain: ChainView) -> dict:
    """The roadmap's presentation fields for one job, from the chain only."""
    task_id = job.get("task_id")
    view = {
        "job_id": job.get("job_id"),
        "task_id": task_id,
        "status": job.get("status", "queued"),
        "worker": None,
        "graph_id": None,
        "final_root": None,
        "verification": None,
        "settlement_tx": None,
        "vwr_id": None,
    }
    if task_id is None:
        return view
    document = chain.task(int(task_id))
    if not document:
        view["status"] = "unknown"
        return view
    view["status"] = job_state_for(str(document.get("status", "")))
    view["worker"] = document.get("worker") or None
    view["graph_id"] = document.get("graph_id") or document.get("graph_id_v2") or None
    view["final_root"] = document.get("final_output_root") or document.get("output_root") or None
    receipt = chain.receipt(int(task_id))
    if receipt:
        # passthrough of what the chain confirmed — never inferred ahead
        view["verification"] = {
            "confirmed": True,
            "receipt_id": receipt.get("receipt_id") or receipt.get("id"),
            "verdict": receipt.get("verdict") or receipt.get("result"),
        }
        view["vwr_id"] = view["verification"]["receipt_id"]
        view["settlement_tx"] = receipt.get("settlement_txhash") or receipt.get("txhash")
    else:
        view["verification"] = {"confirmed": False,
                                "note": "the chain has not confirmed a receipt yet"}
    return view


def present_all(jobs: list[dict], chain: ChainView) -> list[dict]:
    return [present_job(job, chain) for job in jobs]
