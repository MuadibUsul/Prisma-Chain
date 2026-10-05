"""Job API v1 (B6-01): the developer-facing submission surface.

Design (all of it product code; the chain settles):

```text
POST /v1/jobs       submit — frozen supported profiles ONLY (explicit error
                    otherwise); Idempotency-Key honored for client retries
GET  /v1/jobs/{id}  status + result references
GET  /v1/jobs       cursor-paginated list
DELETE /v1/jobs/{id}  cancel — explicit lifecycle rule: only while a job is
                    still pending (nothing has been accepted on chain); a
                    later-stage cancel is refused with job_not_cancellable
GET  /v1/profiles    enumerate the frozen supported profiles; nothing else
```

Errors are machine-readable and stable: every failure is
``{"error": {"code", "message", "details"?}}`` with codes from
``ERROR_CODES``. The store is pluggable (``JobStore``) — the in-memory
implementation is for tests/dev; production uses the chain as the record and
the store only tracks the API's own view.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

# The frozen supported profiles of this release (B2-05 is the enforcement
# point for the artifacts; the API only ever claims these).
SUPPORTED_PROFILES = {
    "qwen3-0.6b-layer0-v2": {
        "spec_version": "v1",
        "mode": "lightweight",
        "description": "Qwen3-0.6B layer 0, CANONICAL_GRAPH_V2 wide-integer (frozen)",
        "graph_id_v2": "8fb86087697e277602bb4b2088ba48eb4172c21d9c4f83914582f2cbf00d4def",
        "policy_id": "eb9a9fef403c95cd0ab876893acf14e02928245306f1d1cc6e12f4d99c5eb7a0",
    },
}

ERROR_CODES = {
    "invalid_request": 400,
    "unsupported_profile": 400,
    "unauthorized": 401,
    "job_not_found": 404,
    "job_not_cancellable": 409,
    "forbidden": 403,
    "duplicate_job": 409,
    "rate_limited": 429,
    "internal": 500,
}


class JobStore(Protocol):
    def put(self, job: dict) -> None:
        ...

    def get(self, job_id: str) -> Optional[dict]:
        ...

    def by_idempotency_key(self, key: str) -> Optional[dict]:
        ...

    def list(self, cursor: str = "", limit: int = 50) -> tuple[list[dict], str]:
        ...

    def delete(self, job_id: str) -> bool:
        ...


@dataclass
class InMemoryJobStore:
    jobs: dict = field(default_factory=dict)
    order: list = field(default_factory=list)
    idempotent: dict = field(default_factory=dict)

    def put(self, job: dict) -> None:
        self.jobs[job["job_id"]] = job
        self.order.append(job["job_id"])
        if job.get("idempotency_key"):
            self.idempotent[job["idempotency_key"]] = job["job_id"]

    def get(self, job_id: str) -> Optional[dict]:
        return self.jobs.get(job_id)

    def by_idempotency_key(self, key: str) -> Optional[dict]:
        job_id = self.idempotent.get(key)
        return self.jobs.get(job_id) if job_id else None

    def list(self, cursor: str = "", limit: int = 50) -> tuple[list[dict], str]:
        start = 0
        if cursor:
            try:
                start = self.order.index(cursor)
            except ValueError:
                start = 0
        page = [self.jobs[job_id] for job_id in self.order[start:start + limit]]
        next_cursor = self.order[start + limit] if start + limit < len(self.order) else ""
        return page, next_cursor

    def delete(self, job_id: str) -> bool:
        job = self.jobs.pop(job_id, None)
        if job is None:
            return False
        self.order.remove(job_id)
        if job.get("idempotency_key"):
            self.idempotent.pop(job["idempotency_key"], None)
        return True


def error_response(code: str, message: str, **details) -> tuple[int, dict]:
    status = ERROR_CODES.get(code, 400)
    error: dict = {"code": code, "message": message}
    if details:
        error["details"] = details
    return status, {"error": error}


@dataclass
class JobAPI:
    store: JobStore
    submitter: Optional[Callable[[dict], str]] = None   # job -> chain task id (wired in B6-05)
    clock: Callable[[], float] = time.time

    # --- POST /v1/jobs -----------------------------------------------------

    def submit(self, body: Optional[dict], idempotency_key: str = "") -> tuple[int, dict]:
        body = body or {}
        profile = str(body.get("profile", ""))
        if not profile:
            return error_response("invalid_request", "profile is required")
        if profile not in SUPPORTED_PROFILES:
            return error_response(
                "unsupported_profile", f"profile {profile!r} is not supported",
                supported=sorted(SUPPORTED_PROFILES))
        if idempotency_key:
            existing = self.store.by_idempotency_key(idempotency_key)
            if existing:
                return 200, {"job_id": existing["job_id"], "status": existing["status"],
                             "idempotent_replay": True}
        job_id = "job-" + uuid.uuid4().hex[:20]
        job = {
            "job_id": job_id,
            "profile": profile,
            "spec_version": SUPPORTED_PROFILES[profile]["spec_version"],
            "status": "pending",
            "input": body.get("input"),
            "created_at": self.clock(),
            "idempotency_key": idempotency_key,
            "task_id": None,
            "result": None,
            "receipt": None,
        }
        if self.submitter is not None:
            try:
                job["task_id"] = self.submitter(job)
            except Exception as exc:  # noqa: BLE001 - surfaced as a stable error
                return error_response("internal", f"submission failed: {exc}")
        self.store.put(job)
        return 202, self._public(job)

    # --- GET ---------------------------------------------------------------

    def get(self, job_id: str) -> tuple[int, dict]:
        job = self.store.get(job_id)
        if job is None:
            return error_response("job_not_found", f"job {job_id} does not exist")
        return 200, self._public(job)

    def list(self, cursor: str = "", limit: int = 50) -> tuple[int, dict]:
        if limit < 1 or limit > 200:
            return error_response("invalid_request", "limit must be between 1 and 200")
        page, next_cursor = self.store.list(cursor, limit)
        return 200, {"jobs": [self._public(job) for job in page],
                     "next_cursor": next_cursor}

    def cancel(self, job_id: str) -> tuple[int, dict]:
        job = self.store.get(job_id)
        if job is None:
            return error_response("job_not_found", f"job {job_id} does not exist")
        if job["status"] != "pending" or job.get("task_id"):
            return error_response(
                "job_not_cancellable",
                f"job {job_id} is {job['status']}; only pending jobs (pre-accept) can be cancelled")
        self.store.delete(job_id)
        return 200, {"job_id": job_id, "cancelled": True}

    def profiles(self) -> tuple[int, dict]:
        return 200, {"profiles": [self._profile_public(name, meta)
                                  for name, meta in sorted(SUPPORTED_PROFILES.items())]}

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _profile_public(name: str, meta: dict) -> dict:
        return {"name": name, "spec_version": meta["spec_version"], "mode": meta["mode"],
                "description": meta["description"], "graph_id_v2": meta["graph_id_v2"],
                "policy_id": meta["policy_id"]}

    @staticmethod
    def _public(job: dict) -> dict:
        return {key: value for key, value in job.items()
                if key not in ("idempotency_key",)}


def handle_request(api: JobAPI, method: str, path: str, *, body: Optional[dict] = None,
                   idempotency_key: str = "") -> tuple[int, dict]:
    """Route an HTTP request to the API (framework-free; the FastAPI/uvicorn
    wrapper is thin and lives in server.py)."""
    parts = [piece for piece in path.split("/") if piece]
    if method == "POST" and parts == ["v1", "jobs"]:
        return api.submit(body, idempotency_key)
    if method == "GET" and parts == ["v1", "jobs"]:
        return api.list()
    if method == "GET" and len(parts) == 3 and parts[:2] == ["v1", "jobs"]:
        return api.get(parts[2])
    if method == "DELETE" and len(parts) == 3 and parts[:2] == ["v1", "jobs"]:
        return api.cancel(parts[2])
    if method == "GET" and parts == ["v1", "profiles"]:
        return api.profiles()
    return error_response("invalid_request", f"no route: {method} {path}")
