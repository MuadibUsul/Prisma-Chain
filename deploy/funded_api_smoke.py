"""Live-chain funded API lifecycle with a synthetic in-process model."""

import asyncio
import base64
import hashlib
import tempfile
import time
import uuid
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers

from chain_smoke import height, wait_for
from compute_smoke import account, find_task, submit, task_state
from light_smoke import put_bytes

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "network"))
from prisma_network.chain_auth import ChainTaskAuthorizer, GrpcChainQueries
from prisma_network.core import Capability, ControlPlane, Identity, ModelPin, SignedCapability, TaskEnvelope, canonical, digest
from prisma_network.server import (InferenceRequest, WorkerResponse, create_gateway_app,
                                   create_worker_app)


def main() -> None:
    wait_for(lambda: height() >= 1, "first block")
    suffix = uuid.uuid4().hex[:12]
    gateway_key, worker_key = Identity.generate(), Identity.generate()
    names = {role: f"api-{role}-{suffix}" for role in ("worker", "gateway", "monitor-a", "monitor-b")}
    addresses = {}
    for role, name in names.items():
        address = account(name)
        addresses[role] = address
        submit("bank", "send", "validator", address, "2000000uprsm", signer="validator")
        args = ["compute", "bond-worker", "--amount", "1000000"]
        if role in ("worker", "gateway"):
            args += ["--network-public-key", (worker_key if role == "worker" else gateway_key).public_key.hex()]
        submit(*args, signer=name)

    temp = tempfile.TemporaryDirectory(prefix="prisma-funded-api-", ignore_cleanup_errors=True)
    tokenizer_file = Path(temp.name) / "tokenizer.json"
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tokenizer_file))
    tokenizer_hash = hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
    pin = ModelPin(model_id="smoke-api-" + suffix, weights_digest="a" * 64,
                   tokenizer_digest=digest({"tokenizer.json": tokenizer_hash}),
                   runtime_digest="b" * 64, spec_version="light-v1")
    submit("compute", "register-model", "--model-id", pin.model_id,
           "--spec-version", pin.spec_version, "--mode", "lightweight",
           "--image-digest", pin.runtime_digest, "--tokenizer-digest", pin.tokenizer_digest,
           "--weights-digest", pin.weights_digest, signer="validator")

    messages = [{"role": "user", "content": "synthetic funded API smoke"}]
    commitment = digest({"messages": messages, "max_tokens": 8, "temperature": 0.0})
    deadline = height() + 1000
    submit("compute", "post-task", "--mode", "lightweight", "--model-id", pin.model_id,
           "--spec-version", pin.spec_version, "--input-commitment", commitment,
           "--data-ref", "encrypted:synthetic-api-smoke", "--max-fee", "10000",
           "--deadline", str(deadline), "--privacy-tier", "tier0_relative", signer="validator")
    task_id, posted = find_task(pin.model_id)
    if posted["status"] != "posted":
        raise RuntimeError("funded API task was not posted")
    submit("compute", "accept-task", "--task-id", str(task_id), signer=names["worker"])

    trusted = {node.node_id: node.public_key_b64 for node in (gateway_key, worker_key)}
    plane = ControlPlane(str(Path(temp.name) / "control.db"), trusted, {"worker.local"},
                         bound_worker_accounts={worker_key.node_id: addresses["worker"]},
                         allow_http=True)
    now = int(time.time() * 1000)
    capability = Capability(node_id=worker_key.node_id, public_key=worker_key.public_key_b64,
                            sequence=now, issued_at_ms=now, expires_at_ms=now + 45_000,
                            group_id="synthetic-api", chain_worker=addresses["worker"],
                            stage_index=0, stage_count=1, model_id=pin.model_id,
                            model_digest=pin.model_digest, weights_digest=pin.weights_digest,
                            tokenizer_digest=pin.tokenizer_digest, runtime_digest=pin.runtime_digest,
                            spec_version=pin.spec_version, probe_url="http://worker.local/v1/probe",
                            api_url="http://worker.local/v1/execute", gpu_count=0, vram_mb=0)
    plane.announce(SignedCapability(capability=capability,
                                     signature=worker_key.sign("prisma:capability:v1", capability.model_dump())))
    plane.probe(worker_key.node_id, ok=True, latency_ms=1, bandwidth_mbps=100)
    queries = GrpcChainQueries("127.0.0.1:9090", "http://127.0.0.1:26657", insecure_dev=True)
    worker_tokenizer = Tokenizer.from_file(str(tokenizer_file))
    gateway_tokenizer = Tokenizer.from_file(str(tokenizer_file))
    attempts = []

    async def model_call(request):
        attempts.append(request.attempt_id)
        if len(attempts) == 1:
            raise OSError("injected synthetic worker failure")
        output = "hello world"
        return output, len(worker_tokenizer.encode(output, add_special_tokens=False).ids)

    async def lease_lookup(task):
        return plane.lease(task) or {}

    async def count(output):
        return len(gateway_tokenizer.encode(output, add_special_tokens=False).ids)

    submissions = []

    def submit_to_chain(receipt):
        task = task_state(task_id)
        if task["status"] in ("pending", "settled"):
            if base64.b64decode(task["receipt_digest"]) != hashlib.sha256(canonical(receipt)).digest():
                raise RuntimeError("chain contains a different receipt")
            return
        if task["status"] != "accepted":
            raise RuntimeError("task is no longer available for result submission")
        receipt_bytes = canonical(receipt)
        receipt_file = put_bytes(receipt_bytes)
        submit("compute", "submit-result", "--task-id", str(task_id),
               "--output-digest", receipt["output_commitment"],
               "--output-tokens", str(receipt["output_tokens"]),
               "--receipt-digest", hashlib.sha256(receipt_bytes).hexdigest(),
               "--receipt-json", receipt_file, signer=names["worker"])

    async def receipt_submission(receipt):
        submissions.append(receipt["attempt_id"])
        if len(submissions) == 1:
            raise OSError("injected chain submission failure")
        await asyncio.to_thread(submit_to_chain, receipt)

    worker_app = create_worker_app(worker_key, "synthetic-api", pin, trusted,
                                   model_call=model_call, lease_lookup=lease_lookup,
                                   receipt_submission=receipt_submission)

    async def worker_call(_url, signed):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=worker_app),
                                     base_url="http://worker.local") as client:
            response = await client.post("/v1/execute", json=signed.model_dump())
            response.raise_for_status()
            return WorkerResponse.model_validate(response.json())

    async def receipt_submitter(_url, receipt):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=worker_app),
                                     base_url="http://worker.local") as client:
            response = await client.post("/v1/submit-receipt", json={"receipt": receipt})
            response.raise_for_status()

    gateway_app = create_gateway_app(plane, gateway_key, "synthetic-test-key",
                                     task_authorizer=ChainTaskAuthorizer(queries),
                                     chain_task_query=queries.task, chain_height_query=queries.height,
                                     token_counter=count, billing_pin=pin, worker_call=worker_call,
                                     receipt_submitter=receipt_submitter)
    client = TestClient(gateway_app)
    headers = {"Authorization": "Bearer synthetic-test-key"}
    envelope = TaskEnvelope(task_id=str(task_id), mode="lightweight", model_id=pin.model_id,
                            model_digest=pin.model_digest, spec_version=pin.spec_version,
                            input_commitment=commitment, data_ref="encrypted:synthetic-api-smoke",
                            max_fee="10000", deadline=deadline, privacy_tier="tier0_relative")
    body = InferenceRequest(task=envelope, messages=messages, max_tokens=8, temperature=0.0).model_dump()
    first = client.post("/v1/inference", json=body, headers=headers)
    if first.status_code != 502 or task_state(task_id)["status"] != "accepted":
        raise RuntimeError(f"synthetic worker failure did not leave escrow accepted: {first.text}")
    second = client.post("/v1/inference", json=body, headers=headers)
    if second.status_code != 200:
        raise RuntimeError(f"funded API inference failed: {second.text}")
    delivered = second.json()
    if (delivered["output"] != "hello world" or delivered["receipt"]["output_tokens"] != 2
            or len(set(attempts)) != 2 or delivered["chain_submission"] != "retry_required"):
        raise RuntimeError("funded retry did not produce an independently counted signed receipt")
    status = client.get(f"/v1/tasks/{task_id}", headers=headers)
    if (status.status_code != 200 or status.json()["chain_status"] != "accepted"
            or status.json()["billing"]["escrowed_uprsm"] != "10000"):
        raise RuntimeError(f"funded API accepted status mismatch: {status.text}")

    replay = client.post("/v1/inference", json=body, headers=headers)
    if (replay.status_code != 200 or not replay.json()["already_completed"]
            or replay.json()["chain_submission"] != "submitted"
            or len(attempts) != 2 or len(submissions) != 2):
        raise RuntimeError("saved receipt was not submitted without repeating inference")
    pending = client.get(f"/v1/tasks/{task_id}", headers=headers)
    if (pending.status_code != 200 or pending.json()["chain_status"] != "pending"
            or pending.json()["billing"]["proposed_charge_uprsm"] != "2000"):
        raise RuntimeError(f"funded API pending status mismatch: {pending.text}")
    for role in ("monitor-a", "monitor-b"):
        submit("compute", "attest-result", "--task-id", str(task_id), signer=names[role])
    wait_for(lambda: height() > int(task_state(task_id)["challenge_end"]),
             "funded API confirmation window", seconds=180)
    submit("compute", "finalize-task", "--task-id", str(task_id), signer="validator")
    settled = client.get(f"/v1/tasks/{task_id}", headers=headers)
    if (settled.status_code != 200 or settled.json()["chain_status"] != "settled"
            or not settled.json()["settlement_final"] or task_state(task_id)["charged_fee"] != 2000
            or settled.json()["billing"]["charged_uprsm"] != "2000"
            or settled.json()["billing"]["refunded_uprsm"] != "8000"):
        raise RuntimeError(f"funded API final status or charge mismatch: {settled.text}")
    replay = client.post("/v1/inference", json=body, headers=headers)
    if (replay.status_code != 200 or not replay.json()["already_completed"]
            or replay.json()["chain_submission"] != "confirmed"
            or len(attempts) != 2 or len(submissions) != 2):
        raise RuntimeError("funded API replay executed the model again")
    print(f"PASS: synthetic funded API task {task_id}; worker and submission failures retried "
          "under one task ID, signed delivery submitted by worker to real chain, "
          "2 text tokens charged, accepted/pending/settled observed")
    client.close()
    plane.db.close()
    temp.cleanup()


if __name__ == "__main__":
    main()
