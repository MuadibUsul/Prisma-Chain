import base64
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prisma_network.core import (Capability, Conflict, ControlPlane, Identity, ModelPin,
                                 SignedCapability, TaskEnvelope, Unavailable, digest, verify)
from prisma_network.server import (ExecutionRequest, InferenceRequest, SignedExecution,
                                   WorkerAttestation, WorkerResponse, create_gateway_app,
                                   create_worker_app, _verified_pin_from_env)
from prisma_network.mock_model import app as mock_model_app


MODEL = "Qwen/Qwen3-14B"
PIN = ModelPin(model_id=MODEL, weights_digest="a" * 64,
               tokenizer_digest="c" * 64, runtime_digest="b" * 64,
               spec_version="light-v1")
MODEL_HASH = PIN.model_digest


def capability(identity, stage, count, now, group="g1", sequence=1):
    cap = Capability(node_id=identity.node_id, public_key=identity.public_key_b64,
                     sequence=sequence, issued_at_ms=now, expires_at_ms=now + 45_000,
                     group_id=group, chain_worker=f"dev-worker-{stage}",
                     stage_index=stage, stage_count=count,
                     model_id=MODEL, model_digest=MODEL_HASH,
                     weights_digest=PIN.weights_digest, tokenizer_digest=PIN.tokenizer_digest,
                     runtime_digest=PIN.runtime_digest, spec_version=PIN.spec_version,
                     probe_url=f"http://worker{stage}.local:9000/v1/probe",
                     api_url="http://worker0.local:9000/v1/execute" if stage == 0 else None,
                     gpu_count=1, vram_mb=48_000)
    return SignedCapability(capability=cap,
                            signature=identity.sign("prisma:capability:v1", cap.model_dump()))


def plane_for(tmp_path, identities, now):
    keys = {identity.node_id: identity.public_key_b64 for identity in identities}
    return ControlPlane(str(tmp_path / "control.db"), keys, {"worker0.local", "worker1.local"},
                        allow_http=True, clock_ms=lambda: now[0])


def task_body(task_id="1"):
    messages = [{"role": "user", "content": "hello"}]
    commitment = digest({"messages": messages, "max_tokens": 8, "temperature": 0.0})
    task = TaskEnvelope(task_id=task_id, mode="lightweight", model_id=MODEL,
                        model_digest=MODEL_HASH, spec_version=PIN.spec_version,
                        input_commitment=commitment, data_ref="encrypted:example",
                        max_fee="1000", deadline=100, privacy_tier="tier0_relative")
    return InferenceRequest(task=task, messages=messages, max_tokens=8, temperature=0.0)


def test_signed_discovery_requires_complete_measured_group(tmp_path):
    now = [1_700_000_000_000]
    gateway, first, second = (Identity.generate() for _ in range(3))
    plane = plane_for(tmp_path, (gateway, first, second), now)
    head = capability(first, 0, 2, now[0])
    plane.announce(head)
    plane.probe(first.node_id, ok=True, latency_ms=20, bandwidth_mbps=50)
    with pytest.raises(Unavailable):
        plane.route(MODEL, MODEL_HASH, PIN.spec_version)
    tail = capability(second, 1, 2, now[0])
    plane.announce(tail)
    plane.probe(second.node_id, ok=True, latency_ms=30, bandwidth_mbps=10)
    route = plane.route(MODEL, MODEL_HASH, PIN.spec_version)
    assert route.stage_node_ids == (first.node_id, second.node_id)
    assert route.api_url.endswith("/v1/execute")
    tampered = head.model_copy(deep=True)
    tampered.capability.gpu_count = 128
    with pytest.raises(ValueError, match="signature"):
        plane.announce(tampered)
    with pytest.raises(Conflict):
        plane.announce(capability(first, 0, 2, now[0], sequence=0))
    false_bundle = capability(first, 0, 2, now[0], sequence=2)
    false_bundle.capability.spec_version = "different-rules"
    false_bundle.signature = first.sign("prisma:capability:v1", false_bundle.capability.model_dump())
    with pytest.raises(ValueError, match="bundle"):
        plane.announce(false_bundle)
    now[0] += 16_000
    with pytest.raises(Unavailable):
        plane.route(MODEL, MODEL_HASH, PIN.spec_version)


def test_announcement_updates_are_atomic_and_material_change_reprobes(tmp_path):
    now = [1_700_000_000_000]
    gateway, worker = Identity.generate(), Identity.generate()
    plane = plane_for(tmp_path, (gateway, worker), now)
    other = plane_for(tmp_path, (gateway, worker), now)
    first = capability(worker, 0, 1, now[0], sequence=1)
    plane.announce(first)
    plane.probe(worker.node_id, ok=True, latency_ms=10, bandwidth_mbps=10)
    assert plane.route(MODEL, MODEL_HASH, PIN.spec_version).group_id == "g1"
    # A lease renewal may keep the recent probe; a changed endpoint/group may not.
    plane.announce(capability(worker, 0, 1, now[0], sequence=2))
    assert plane.route(MODEL, MODEL_HASH, PIN.spec_version).group_id == "g1"
    changed = capability(worker, 0, 1, now[0], group="g2", sequence=3)
    plane.announce(changed)
    assert plane.probe(worker.node_id, ok=True, latency_ms=1, bandwidth_mbps=100,
                       expected_sequence=2) is False
    with pytest.raises(Unavailable):
        plane.route(MODEL, MODEL_HASH, PIN.spec_version)
    plane.probe(worker.node_id, ok=True, latency_ms=10, bandwidth_mbps=10)
    assert plane.route(MODEL, MODEL_HASH, PIN.spec_version).group_id == "g2"
    # Two processes sharing the durable store must never regress the sequence.
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda args: _announce_result(*args),
                                 ((plane, capability(worker, 0, 1, now[0], group="g2", sequence=4)),
                                  (other, capability(worker, 0, 1, now[0], group="g2", sequence=5)))))
    assert outcomes.count("accepted") >= 1
    assert plane.announcements()[0].capability.sequence == 5


def _announce_result(plane, signed):
    try:
        plane.announce(signed)
        return "accepted"
    except Conflict:
        return "stale"


def test_expiry_takeover_fences_old_attempt_and_receipt_is_once(tmp_path):
    now = [1_700_000_000_000]
    old, standby = Identity.generate(), Identity.generate()
    plane = plane_for(tmp_path, (old, standby), now)
    first = plane.acquire("42", "g1", old.node_id, ttl_ms=1_000)
    plane.start_attempt("42", "g1", old.node_id, first["epoch"], "attempt-old")
    with pytest.raises(Conflict):
        plane.acquire("42", "g2", standby.node_id)
    now[0] += 1_001
    second = plane.acquire("42", "g2", standby.node_id)
    assert second["epoch"] == first["epoch"] + 1
    assert not plane.check_fence("42", "g1", old.node_id, first["epoch"])
    plane.start_attempt("42", "g2", standby.node_id, second["epoch"], "attempt-new")
    with pytest.raises(Conflict):
        plane.commit_receipt("42", "attempt-old", "g1", old.node_id, first["epoch"], {"x": 1})
    receipt = {"task_id": "42", "attempt_id": "attempt-new", "output_commitment": "c" * 64}
    assert plane.commit_receipt("42", "attempt-new", "g2", standby.node_id, second["epoch"], receipt) == receipt
    assert plane.commit_receipt("42", "attempt-new", "g2", standby.node_id, second["epoch"], receipt) == receipt
    with pytest.raises(Conflict):
        plane.commit_receipt("42", "attempt-new", "g2", standby.node_id, second["epoch"], {"x": 2})
    with pytest.raises(Conflict):
        plane.start_attempt("42", "g2", standby.node_id, second["epoch"], "another")


def test_gateway_requires_chain_authorization_by_default(tmp_path):
    now = [int(time.time() * 1000)]
    gateway = Identity.generate()
    plane = plane_for(tmp_path, (gateway,), now)
    client = TestClient(create_gateway_app(plane, gateway, "secret"))
    body = task_body().model_dump()
    assert client.post("/v1/inference", json=body).status_code == 401
    response = client.post("/v1/inference", json=body, headers={"Authorization": "Bearer secret"})
    assert response.status_code == 503
    assert "chain task authorizer" in response.json()["detail"]
    assert client.get("/v1/tasks/1", headers={"Authorization": "Bearer secret"}).status_code == 503


def test_gateway_worker_signatures_and_provisional_once(tmp_path):
    now = [int(time.time() * 1000)]
    gateway, worker = Identity.generate(), Identity.generate()
    plane = plane_for(tmp_path, (gateway, worker), now)
    plane.announce(capability(worker, 0, 1, now[0]))
    plane.probe(worker.node_id, ok=True, latency_ms=15, bandwidth_mbps=60)
    calls = []

    async def worker_call(_url, signed):
        calls.append(signed)
        assert verify(gateway.public_key_b64, "prisma:execution:v1",
                      signed.request.model_dump(), signed.signature)
        request = signed.request
        count = 9 if request.task.task_id == "2" else 3
        att = WorkerAttestation(task_id=request.task.task_id, attempt_id=request.attempt_id,
                                worker_node_id=worker.node_id, gateway_node_id=gateway.node_id,
                                group_id=request.group_id, lease_epoch=request.lease_epoch,
                                model_id=MODEL, model_digest=MODEL_HASH,
                                spec_version=request.task.spec_version,
                                input_commitment=request.task.input_commitment,
                                output_commitment=digest("answer"), output_tokens=count,
                                completed_at_ms=now[0])
        return WorkerResponse(output="answer", attestation=att,
                              signature=worker.sign("prisma:worker-receipt:v1", att.model_dump()))

    app = create_gateway_app(plane, gateway, "secret", allow_unfunded_dev_tasks=True,
                             worker_call=worker_call)
    client = TestClient(app)
    body = task_body().model_dump()
    headers = {"Authorization": "Bearer secret"}
    response = client.post("/v1/inference", json=body, headers=headers)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "provisional_delivery"
    assert data["output"] == "answer"
    receipt = data["receipt"]
    assert receipt["output_tokens"] == 3
    gateway_sig = receipt.pop("gateway_signature")
    assert verify(gateway.public_key_b64, "prisma:gateway-receipt:v1", receipt, gateway_sig)
    duplicate = client.post("/v1/inference", json=body, headers=headers)
    assert duplicate.status_code == 200
    assert duplicate.json()["already_completed"] is True
    assert duplicate.json()["output"] is None
    assert len(calls) == 1
    changed = task_body().model_dump()
    changed["task"]["spec_version"] = "new-spec"
    assert client.post("/v1/inference", json=changed, headers=headers).status_code == 409
    overcount = client.post("/v1/inference", json=task_body("2").model_dump(), headers=headers)
    assert overcount.status_code == 400
    assert plane.receipt("2") is None
    status = client.get("/v1/tasks/1", headers=headers)
    assert status.status_code == 200
    assert status.json()["chain_status"] == "unfunded_dev"
    assert status.json()["delivery_status"] == "provisional_delivery"
    assert status.json()["settlement_final"] is False


def test_task_status_separates_chain_settlement_from_delivery(tmp_path):
    now = [int(time.time() * 1000)]
    gateway = Identity.generate()
    plane = plane_for(tmp_path, (gateway,), now)
    plane.bound_worker_accounts[gateway.node_id] = "prsmworker"
    body = task_body("7").task
    lease = plane.acquire("7", "group", gateway.node_id)
    plane.start_attempt("7", "group", gateway.node_id, lease["epoch"], "attempt")
    plane.commit_receipt("7", "attempt", "group", gateway.node_id, lease["epoch"],
                         {"task_id": "7", "worker_node_id": gateway.node_id,
                          "mode": body.mode, "model_id": body.model_id,
                          "spec_version": body.spec_version,
                          "input_commitment": body.input_commitment})
    chain_task = {"id": 7, "status": "pending", "mode": body.mode,
                  "worker": "prsmworker",
                  "model_id": body.model_id, "spec_version": body.spec_version,
                  "input_commitment": base64.b64encode(bytes.fromhex(body.input_commitment)).decode(),
                  "challenge_end": 30}

    async def query(task_id):
        assert task_id == 7
        return chain_task

    async def height():
        return 25

    client = TestClient(create_gateway_app(plane, gateway, "secret", chain_task_query=query,
                                           chain_height_query=height))
    headers = {"Authorization": "Bearer secret"}
    assert client.get("/v1/tasks/7").status_code == 401
    assert client.get("/v1/tasks/07", headers=headers).status_code == 400
    status = client.get("/v1/tasks/7", headers=headers)
    assert status.status_code == 200
    assert status.json() == {"task_id": "7", "delivery_status": "provisional_delivery",
                             "chain_status": "pending", "settlement_final": False,
                             "observed_height": 25, "challenge_end": 30}
    chain_task["status"] = "settled"
    assert client.get("/v1/tasks/7", headers=headers).json()["settlement_final"] is True
    chain_task["status"] = "refunded"
    assert client.get("/v1/tasks/7", headers=headers).json()["chain_status"] == "refunded"
    chain_task["worker"] = "other-worker"
    assert client.get("/v1/tasks/7", headers=headers).status_code == 409
    chain_task["worker"] = "prsmworker"
    chain_task["model_id"] = "different-model"
    assert client.get("/v1/tasks/7", headers=headers).status_code == 409
    chain_task["input_commitment"] = "invalid"
    assert client.get("/v1/tasks/7", headers=headers).status_code == 503
    dev_client = TestClient(create_gateway_app(plane, gateway, "secret", chain_task_query=query,
                                               chain_height_query=height, allow_unfunded_dev_tasks=True))
    assert dev_client.get("/v1/tasks/7", headers=headers).json()["chain_status"] == "unfunded_dev"


def test_worker_rejects_stale_fence_before_model_call():
    gateway, worker = Identity.generate(), Identity.generate()
    body = task_body()
    calls = []

    async def model_call(_request):
        calls.append(1)
        return "answer", 3

    async def stale_lease(_task_id):
        return {"group_id": "g1", "owner_id": gateway.node_id, "epoch": 2,
                "expires_at_ms": int(time.time() * 1000) + 30_000}

    app = create_worker_app(worker, "g1", PIN,
                            {gateway.node_id: gateway.public_key_b64},
                            model_call=model_call, lease_lookup=stale_lease)
    request = ExecutionRequest(**body.model_dump(), attempt_id="attempt-1", lease_epoch=1,
                               gateway_node_id=gateway.node_id, group_id="g1")
    signed = SignedExecution(request=request,
                             signature=gateway.sign("prisma:execution:v1", request.model_dump()))
    response = TestClient(app).post("/v1/execute", json=signed.model_dump())
    assert response.status_code == 409
    assert calls == []


def test_worker_rejects_backend_token_overcount():
    gateway, worker = Identity.generate(), Identity.generate()
    body = task_body()

    async def model_call(_request):
        return "answer", 9

    async def valid_lease(_task_id):
        return {"group_id": "g1", "owner_id": gateway.node_id, "epoch": 1,
                "expires_at_ms": int(time.time() * 1000) + 30_000}

    app = create_worker_app(worker, "g1", PIN,
                            {gateway.node_id: gateway.public_key_b64},
                            model_call=model_call, lease_lookup=valid_lease)
    request = ExecutionRequest(**body.model_dump(), attempt_id="attempt-1", lease_epoch=1,
                               gateway_node_id=gateway.node_id, group_id="g1")
    signed = SignedExecution(request=request,
                             signature=gateway.sign("prisma:execution:v1", request.model_dump()))
    response = TestClient(app).post("/v1/execute", json=signed.model_dump())
    assert response.status_code == 400


def test_worker_rejects_wrong_execution_spec():
    gateway, worker = Identity.generate(), Identity.generate()
    body = task_body()
    task = body.task.model_copy(update={"spec_version": "different-rules"})
    body = body.model_copy(update={"task": task})

    async def model_call(_request):
        pytest.fail("mismatched spec reached model")

    async def valid_lease(_task_id):
        return {"group_id": "g1", "owner_id": gateway.node_id, "epoch": 1,
                "expires_at_ms": int(time.time() * 1000) + 30_000}

    app = create_worker_app(worker, "g1", PIN,
                            {gateway.node_id: gateway.public_key_b64},
                            model_call=model_call, lease_lookup=valid_lease)
    request = ExecutionRequest(**body.model_dump(), attempt_id="attempt-1", lease_epoch=1,
                               gateway_node_id=gateway.node_id, group_id="g1")
    signed = SignedExecution(request=request,
                             signature=gateway.sign("prisma:execution:v1", request.model_dump()))
    assert TestClient(app).post("/v1/execute", json=signed.model_dump()).status_code == 400


def test_startup_hashes_local_model_files(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weight.bin").write_bytes(b"weights")
    (model_dir / "tokenizer.json").write_bytes(b"tokenizer")
    weight_map = {"weight.bin": hashlib.sha256(b"weights").hexdigest()}
    tokenizer_map = {"tokenizer.json": hashlib.sha256(b"tokenizer").hexdigest()}
    manifest = tmp_path / "files.json"
    manifest.write_text(json.dumps({"weights": weight_map, "tokenizer": tokenizer_map}))
    pin = ModelPin(model_id=MODEL, weights_digest=digest(weight_map),
                   tokenizer_digest=digest(tokenizer_map), runtime_digest="b" * 64,
                   spec_version="light-v1")
    for key, value in {"PRISMA_MODEL_ID": pin.model_id, "PRISMA_WEIGHTS_DIGEST": pin.weights_digest,
                       "PRISMA_TOKENIZER_DIGEST": pin.tokenizer_digest,
                       "PRISMA_RUNTIME_DIGEST": pin.runtime_digest,
                       "PRISMA_SPEC_VERSION": pin.spec_version,
                       "PRISMA_MODEL_DIGEST": pin.model_digest,
                       "PRISMA_MODEL_DIR": str(model_dir),
                       "PRISMA_MODEL_FILES_FILE": str(manifest)}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("PRISMA_DEV_SKIP_FILE_HASH", raising=False)
    assert _verified_pin_from_env() == pin
    (model_dir / "weight.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        _verified_pin_from_env()


def test_mock_backend_is_explicitly_non_funded():
    client = TestClient(mock_model_app)
    health = client.get("/health").json()
    assert health["funded_tasks"] is False
    assert health["distributed_inference"] is False
    response = client.post("/v1/chat/completions", json={"model": "demo-model", "messages": [], "max_tokens": 8})
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "MOCK"


def test_checked_in_wire_examples_have_valid_independent_signatures():
    sample = json.loads((Path(__file__).parents[1] / "examples" / "contract.json").read_text(encoding="utf-8-sig"))
    announcement = SignedCapability.model_validate(sample["announcement"])
    cap = announcement.capability
    assert verify(cap.public_key, "prisma:capability:v1", cap.model_dump(), announcement.signature)
    receipt = sample["delivery_receipt"]
    att = WorkerAttestation.model_validate(receipt["worker_attestation"])
    assert verify(cap.public_key, "prisma:worker-receipt:v1", att.model_dump(), receipt["worker_signature"])
    signature = receipt.pop("gateway_signature")
    assert verify(sample["gateway_public_key"], "prisma:gateway-receipt:v1", receipt, signature)
