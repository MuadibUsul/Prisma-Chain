"""Generate test-only signed wire examples with deterministic, public seeds."""

import base64
import json

from prisma_network.core import Capability, Identity, ModelPin, SignedCapability, digest
from prisma_network.server import WorkerAttestation


def example():
    gateway = Identity.from_seed_b64(base64.b64encode(b"\x01" * 32).decode())
    worker = Identity.from_seed_b64(base64.b64encode(b"\x02" * 32).decode())
    now = 1_700_000_000_000
    pin = ModelPin(model_id="Qwen/Qwen3-14B", weights_digest="a" * 64,
                   tokenizer_digest="c" * 64, runtime_digest="b" * 64,
                   spec_version="light-v1")
    model_hash = pin.model_digest
    cap = Capability(node_id=worker.node_id, public_key=worker.public_key_b64,
                     sequence=now, issued_at_ms=now, expires_at_ms=now + 45_000,
                     group_id="example-14b-2stage", chain_worker="dev-worker-0",
                     stage_index=0, stage_count=2,
                     model_id=pin.model_id, model_digest=model_hash,
                     weights_digest=pin.weights_digest, tokenizer_digest=pin.tokenizer_digest,
                     runtime_digest=pin.runtime_digest, spec_version=pin.spec_version,
                     probe_url="https://worker1.example/v1/probe",
                     api_url="https://worker1.example/v1/execute", gpu_count=1, vram_mb=48_000)
    announcement = SignedCapability(capability=cap,
                                    signature=worker.sign("prisma:capability:v1", cap.model_dump()))
    messages = [{"role": "user", "content": "hello"}]
    input_hash = digest({"messages": messages, "max_tokens": 8, "temperature": 0.0})
    att = WorkerAttestation(task_id="1", attempt_id="attempt-1", worker_node_id=worker.node_id,
                            gateway_node_id=gateway.node_id, group_id=cap.group_id, lease_epoch=1,
                            model_id=cap.model_id, model_digest=model_hash, spec_version="light-v1",
                            input_commitment=input_hash, output_commitment=digest("answer"),
                            output_tokens=3, completed_at_ms=now)
    worker_signature = worker.sign("prisma:worker-receipt:v1", att.model_dump())
    receipt_payload = {"task_id": "1", "attempt_id": "attempt-1", "mode": "lightweight",
                       "model_id": cap.model_id, "model_digest": model_hash, "spec_version": "light-v1",
                       "input_commitment": input_hash, "output_commitment": att.output_commitment,
                       "output_tokens": 3, "gateway_node_id": gateway.node_id,
                       "worker_node_id": worker.node_id, "group_id": cap.group_id, "lease_epoch": 1,
                       "completed_at_ms": now, "status": "delivered",
                       "worker_attestation": att.model_dump(), "worker_signature": worker_signature}
    return {"warning": "Test-only keys and example evidence; no chain settlement or real GPU run",
            "gateway_public_key": gateway.public_key_b64,
            "announcement": announcement.model_dump(),
            "delivery_receipt": {**receipt_payload,
                                 "gateway_signature": gateway.sign("prisma:gateway-receipt:v1", receipt_payload)}}


if __name__ == "__main__":
    print(json.dumps(example(), indent=2, sort_keys=True))
