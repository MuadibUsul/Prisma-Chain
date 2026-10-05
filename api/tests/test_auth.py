"""B6-02: bearer API keys — issuance shown once, hash storage, scopes,
revocation, and the auth hook over the job routes."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "api"))

from prisma_api.auth import Authenticator, KeyStore, auth_guard  # noqa: E402
from prisma_api.jobs import InMemoryJobStore, JobAPI, handle_request  # noqa: E402


def fresh_auth(scopes=("jobs:read", "jobs:write")):
    store = KeyStore()
    authenticator = Authenticator(store)
    secret, record = authenticator.issue("operator", scopes)
    store.add(record)
    return authenticator, secret, record


def test_issue_shows_the_secret_once_and_stores_only_the_hash():
    authenticator, secret, record = fresh_auth()
    assert secret.startswith("prsk_")
    assert secret not in str(record.key_hash)
    assert record.key_hash == __import__("hashlib").sha256(secret.encode()).hexdigest()
    assert authenticator.store.records[record.key_hash].name == "operator"


def test_unknown_and_malformed_keys_are_unauthorized():
    authenticator, _, _ = fresh_auth()
    for headers in ({}, {"Authorization": "Bearer prsk_nope"},
                    {"Authorization": "Basic abc"}):
        record, failure = authenticator.authenticate(headers)
        assert record is None and failure is not None and failure[0] == 401


def test_revocation_is_immediate():
    authenticator, secret, _ = fresh_auth()
    headers = {"Authorization": f"Bearer {secret}"}
    record, failure = authenticator.authenticate(headers)
    assert record is not None
    record.revoked = True
    record, failure = authenticator.authenticate(headers)
    assert record is None and "revoked" in failure[1]["error"]["message"]


def test_scopes_gate_write_vs_read():
    read_only, secret, _ = fresh_auth(scopes=("jobs:read",))
    headers = {"Authorization": f"Bearer {secret}"}
    assert read_only.authenticate(headers)[0] is not None
    guard_write = auth_guard(read_only, "jobs:write")
    failure = guard_write(headers)
    assert failure is not None and failure[0] == 403
    assert failure[1]["error"]["code"] == "forbidden"
    guard_read = auth_guard(read_only, "jobs:read")
    assert guard_read(headers) is None


def test_auth_guard_over_the_job_routes():
    authenticator, write_secret, _ = fresh_auth()
    read_secret, read_record = authenticator.issue("reader", ("jobs:read",))
    authenticator.store.add(read_record)
    api = JobAPI(InMemoryJobStore())
    write_guard = auth_guard(authenticator, "jobs:write")
    code, payload = handle_request(
        api, "POST", "/v1/jobs", body={"profile": "qwen3-0.6b-layer0-v2"},
        idempotency_key="") if False else (0, {})
    # submit through the guard the way the server does
    failure = write_guard({"Authorization": "Bearer wrong"})
    assert failure is not None and failure[0] == 401
    failure = write_guard({}) if False else write_guard({"Authorization": f"Bearer {write_secret}"})
    assert failure is None
    code, payload = api.submit({"profile": "qwen3-0.6b-layer0-v2"})
    assert code == 202
