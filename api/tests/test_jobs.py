"""B6-01: the job API schema — profile gate, idempotency, pagination, cancel
semantics, and stable machine-readable errors."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for extra in ("api",):
    sys.path.insert(0, str(REPO / extra))

from prisma_api.jobs import (  # noqa: E402
    ERROR_CODES, SUPPORTED_PROFILES, InMemoryJobStore, JobAPI, error_response, handle_request,
)


def api() -> JobAPI:
    return JobAPI(InMemoryJobStore())


def test_submit_enforces_the_frozen_profile_gate():
    code, payload = handle_request(api(), "POST", "/v1/jobs",
                                   body={"profile": "some-huggingface-model"})
    assert code == 400 and payload["error"]["code"] == "unsupported_profile"
    assert payload["error"]["details"]["supported"] == ["qwen3-0.6b-layer0-v2"]
    code, payload = handle_request(api(), "POST", "/v1/jobs", body={})
    assert code == 400 and payload["error"]["code"] == "invalid_request"
    code, payload = handle_request(api(), "POST", "/v1/jobs",
                                   body={"profile": "qwen3-0.6b-layer0-v2"})
    assert code == 202 and payload["status"] == "pending"


def test_idempotency_key_replays_the_same_job():
    api = JobAPI(InMemoryJobStore())
    first = handle_request(api, "POST", "/v1/jobs",
                           body={"profile": "qwen3-0.6b-layer0-v2"}, idempotency_key="retry-1")
    assert first[0] == 202
    replay = handle_request(api, "POST", "/v1/jobs",
                            body={"profile": "qwen3-0.6b-layer0-v2"}, idempotency_key="retry-1")
    assert replay[0] == 200 and replay[1]["idempotent_replay"] is True
    assert replay[1]["job_id"] == first[1]["job_id"]
    different = handle_request(api, "POST", "/v1/jobs",
                               body={"profile": "qwen3-0.6b-layer0-v2"}, idempotency_key="retry-2")
    assert different[1]["job_id"] != first[1]["job_id"]


def test_get_and_list_with_cursor_pagination():
    api = JobAPI(InMemoryJobStore())
    ids = []
    for _ in range(5):
        _, payload = handle_request(api, "POST", "/v1/jobs",
                                    body={"profile": "qwen3-0.6b-layer0-v2"})
        ids.append(payload["job_id"])
    code, page = api.list(limit=2)
    assert code == 200 and len(page["jobs"]) == 2 and page["next_cursor"]
    code, page2 = api.list(cursor=page["next_cursor"], limit=2)
    assert len(page2["jobs"]) == 2
    assert {job["job_id"] for job in page["jobs"]}.isdisjoint(
        {job["job_id"] for job in page2["jobs"]})
    code, listing = handle_request(api, "GET", "/v1/jobs")
    assert code == 200 and len(listing["jobs"]) == 5
    code, payload = handle_request(api, "GET", f"/v1/jobs/{ids[0]}")
    assert code == 200
    code, payload = handle_request(api, "GET", "/v1/jobs/absent")
    assert code == 404 and payload["error"]["code"] == "job_not_found"


def test_cancel_semantics_are_explicit():
    api = JobAPI(InMemoryJobStore())
    _, payload = handle_request(api, "POST", "/v1/jobs",
                                body={"profile": "qwen3-0.6b-layer0-v2"})
    job_id = payload["job_id"]
    code, payload = handle_request(api, "DELETE", f"/v1/jobs/{job_id}")
    assert code == 200 and payload["cancelled"] is True
    code, _ = handle_request(api, "GET", f"/v1/jobs/{job_id}")
    assert code == 404
    _, payload = handle_request(api, "POST", "/v1/jobs",
                                body={"profile": "qwen3-0.6b-layer0-v2"})
    job_id = payload["job_id"]
    api.store.get(job_id)["task_id"] = 42
    api.store.get(job_id)["status"] = "accepted"
    code, payload = handle_request(api, "DELETE", f"/v1/jobs/{job_id}")
    assert code == 409 and payload["error"]["code"] == "job_not_cancellable"


def test_profiles_enumerate_only_the_frozen_supported():
    code, payload = handle_request(api(), "GET", "/v1/profiles")
    assert code == 200
    names = [p["name"] for p in payload["profiles"]]
    assert names == sorted(SUPPORTED_PROFILES)
    profile = payload["profiles"][0]
    assert profile["graph_id_v2"].startswith("8fb86087")
    assert profile["policy_id"].startswith("eb9a9fef")


def test_error_schema_is_stable_and_routed():
    code, payload = handle_request(api(), "GET", "/v1/unknown")
    assert code == 400 and payload["error"]["code"] == "invalid_request"
    assert set(ERROR_CODES) >= {"invalid_request", "unsupported_profile", "unauthorized",
                                "job_not_found", "job_not_cancellable", "duplicate_job",
                                "rate_limited", "internal"}
    for code_value in ERROR_CODES.values():
        assert 400 <= code_value <= 500


def test_submitter_binding_is_optional():
    calls: list[dict] = []

    def submitter(job: dict) -> str:
        calls.append(job)
        return "31337"

    api = JobAPI(InMemoryJobStore(), submitter=submitter)
    code, payload = handle_request(api, "POST", "/v1/jobs",
                                   body={"profile": "qwen3-0.6b-layer0-v2"})
    assert code == 202 and payload["task_id"] == "31337" and len(calls) == 1
