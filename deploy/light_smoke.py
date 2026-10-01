"""Settle a synthetic signed lightweight task on the local PRSM devnet."""

import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "network"))
from prisma_network.core import Identity, ModelPin, canonical, digest
from prisma_network.server import WorkerAttestation

from chain_smoke import CHAIN_ID, COMPOSE, FLAGS, balance, cli, height, wait_for
from compute_smoke import account, check_gateway_status, find_task, query_worker, submit, supply, task_state

BOND = 1_000_000
FEE = 10_000


def put_bytes(data: bytes) -> str:
    path = "/tmp/prisma-light-smoke-" + uuid.uuid4().hex + ".json"
    command = ["docker", "compose", "-f", str(COMPOSE), "exec", "-T", "chain",
               "sh", "-c", "cat > " + path]
    result = subprocess.run(command, cwd=COMPOSE.parent.parent, input=data,
                            capture_output=True, timeout=20)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return path


def main() -> None:
    wait_for(lambda: height() >= 1, "first block")
    requester = cli("keys", "show", "validator", "-a", *FLAGS)
    suffix = uuid.uuid4().hex[:12]
    worker_name, gateway_name = "light-worker-" + suffix, "light-gateway-" + suffix
    monitor_names = ("light-monitor-a-" + suffix, "light-monitor-b-" + suffix)
    worker_identity, gateway_identity = Identity.generate(), Identity.generate()
    participants = [(worker_name, worker_identity.public_key),
                    (gateway_name, gateway_identity.public_key),
                    (monitor_names[0], None), (monitor_names[1], None)]
    addresses = {}
    for name, key in participants:
        address = account(name)
        addresses[name] = address
        submit("bank", "send", "validator", address, "2000000uprsm", signer="validator")
        args = ["compute", "bond-worker", "--amount", str(BOND)]
        if key is not None:
            args += ["--network-public-key", key.hex()]
        submit(*args, signer=name)
        if int(query_worker(address)["bonded_uprsm"]) != BOND:
            raise RuntimeError("network participant bond not recorded")

    pin = ModelPin(model_id="smoke-light-" + suffix, weights_digest="a" * 64,
                   tokenizer_digest="c" * 64, runtime_digest="b" * 64,
                   spec_version="light-v1")
    submit("compute", "register-model", "--model-id", pin.model_id,
           "--spec-version", pin.spec_version, "--mode", "lightweight",
           "--image-digest", pin.runtime_digest, "--tokenizer-digest", pin.tokenizer_digest,
           "--weights-digest", pin.weights_digest, signer="validator")
    input_commitment, output_commitment = digest({"synthetic": suffix}), digest("synthetic-output")
    submit("compute", "post-task", "--mode", "lightweight", "--model-id", pin.model_id,
           "--spec-version", pin.spec_version, "--input-commitment", input_commitment,
           "--data-ref", "encrypted:synthetic-smoke", "--max-fee", str(FEE),
           "--deadline", str(height() + 1000), "--privacy-tier", "tier0_relative",
           signer="validator")
    task_id, posted = find_task(pin.model_id)
    if posted["status"] != "posted" or posted["requester"] != requester:
        raise RuntimeError("lightweight escrow was not posted")
    submit("compute", "accept-task", "--task-id", str(task_id), signer=worker_name)
    accepted = task_state(task_id)
    if accepted["status"] != "accepted" or accepted["worker"] != addresses[worker_name]:
        raise RuntimeError("lightweight worker did not accept task")
    check_gateway_status(task_id, "accepted")

    attestation = WorkerAttestation(task_id=str(task_id), attempt_id="attempt-" + suffix,
                                    worker_node_id=worker_identity.node_id,
                                    gateway_node_id=gateway_identity.node_id,
                                    group_id="synthetic-local", lease_epoch=1,
                                    model_id=pin.model_id, model_digest=pin.model_digest,
                                    spec_version=pin.spec_version,
                                    input_commitment=input_commitment,
                                    output_commitment=output_commitment, output_tokens=3,
                                    completed_at_ms=1_700_000_000_000)
    payload = {**attestation.model_dump(), "mode": "lightweight",
               "worker_attestation": attestation.model_dump(),
               "worker_signature": worker_identity.sign("prisma:worker-receipt:v1",
                                                        attestation.model_dump())}
    receipt = {**payload, "gateway_signature": gateway_identity.sign("prisma:gateway-receipt:v1", payload)}
    receipt_bytes = canonical(receipt)
    receipt_path = put_bytes(receipt_bytes)
    receipt_digest = hashlib.sha256(receipt_bytes).hexdigest()
    base_args = ("compute", "submit-result", "--task-id", str(task_id),
                 "--output-digest", output_commitment, "--output-tokens", "3",
                 "--receipt-digest", receipt_digest)
    submit(*base_args, signer=worker_name, expected_code=1)
    if task_state(task_id)["status"] != "accepted":
        raise RuntimeError("hash-only lightweight result changed task status")
    submit(*base_args, "--receipt-json", receipt_path, signer=worker_name)
    pending = task_state(task_id)
    if (pending["status"] != "pending" or pending["charged_fee"] != 3_000
            or pending["gateway"] != addresses[gateway_name]):
        raise RuntimeError("signed lightweight receipt or token charge not recorded")
    check_gateway_status(task_id, "pending")

    for name in monitor_names:
        submit("compute", "attest-result", "--task-id", str(task_id), signer=name)
    wait_for(lambda: height() > int(task_state(task_id)["challenge_end"]),
             "lightweight confirmation window", seconds=180)
    before = {address: balance(address) for address in (requester, *addresses.values())}
    before_supply = supply()
    submit("compute", "finalize-task", "--task-id", str(task_id), signer="validator")
    after = {address: balance(address) for address in before}
    if (task_state(task_id)["status"] != "settled" or after[requester] - before[requester] != 7_000
            or after[addresses[worker_name]] - before[addresses[worker_name]] != 2_100
            or any(after[addresses[name]] - before[addresses[name]] != 150 for name in monitor_names)
            or after[addresses[gateway_name]] != before[addresses[gateway_name]]
            or before_supply - supply() != 600):
        raise RuntimeError("metered lightweight settlement did not match the escrow ledger")
    submit("compute", "finalize-task", "--task-id", str(task_id), signer="validator", expected_code=1)
    check_gateway_status(task_id, "settled")
    print(f"PASS: synthetic lightweight task {task_id} signed by distinct bonded worker/gateway; "
          "hash-only submission rejected, 3000uprsm charged, 7000 refunded, "
          "600 burned, 2100 worker and 150 each monitor, duplicate payout rejected")


if __name__ == "__main__":
    main()
